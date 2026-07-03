/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include <random>
#include <vector>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class ShapeRep { SAME_BOTH_DIMS = 0, VARYING_FIRST_DIM = 1 };
enum class ScalingDir { ROWWISE = 0, COLWISE = 1, BOTH = 2 };
enum class BlockDim { ONE_D = 1, TWO_D = 2 };

constexpr size_t kBlock = 128;

inline size_t align4(size_t x) { return ((x + 3) / 4) * 4; }

// Configure split-quantize reference: call non-grouped nvte_quantize_v2 on each tensor slice.
// Returns flat host buffers for per-tensor outputs and scales (in their per-tensor natural
// layout) so the test can index them and compare element-wise against the grouped layout.
struct PerTensorRef {
  std::vector<std::vector<uint8_t>> output;          // per tensor, FP8 raw bytes (R_t * K)
  std::vector<std::vector<uint8_t>> output_t;        // per tensor, FP8 raw bytes (K * R_t)
  std::vector<std::vector<float>> scale_inv;         // per tensor, layout per non-grouped impl
  std::vector<std::vector<float>> scale_inv_t;       // per tensor, layout per non-grouped impl
};

// Per-expert scale layout helpers mirroring the kernel + cuBLAS grouped GEMM
// expectation. Each expert's scales occupy a contiguous sub-block of the global
// scale buffer; these compute per-expert padded sizes (in floats) so the test
// can both size the buffer and compute per-expert base offsets.
//   1D rowwise   : blocks_X * roundup(M_t, 4)
//   1D colwise   : blocks_y_t * roundup(K, 4)
//   2D rowwise   : blocks_y_t * roundup(blocks_X, 4)
//   2D colwise   : blocks_X * roundup(blocks_y_t, 4)
inline size_t per_expert_scale_floats(BlockDim block_dim, bool columnwise, size_t M_t, size_t K) {
  constexpr size_t kBlk = 128;
  const size_t blocks_X = (K + kBlk - 1) / kBlk;
  const size_t blocks_y = (M_t + kBlk - 1) / kBlk;
  if (block_dim == BlockDim::ONE_D) {
    if (!columnwise) return blocks_X * align4(M_t);
    return blocks_y * align4(K);
  }
  // 2D
  if (!columnwise) return blocks_y * align4(blocks_X);
  return blocks_X * align4(blocks_y);
}

// Cumulative per-expert offset (in floats) for tensor `t`.
inline size_t per_expert_scale_offset(const std::vector<size_t>& first_dims, size_t t,
                                      BlockDim block_dim, bool columnwise, size_t K) {
  size_t offset = 0;
  for (size_t i = 0; i < t; ++i) {
    offset += per_expert_scale_floats(block_dim, columnwise, first_dims[i], K);
  }
  return offset;
}

template <typename InputType, typename OutputType>
void perform_test(ShapeRep shape_rep, BlockDim block_dim, ScalingDir dir,
                  const std::vector<size_t>& first_dims_h, size_t K,
                  bool force_pow_2_scales, float epsilon) {
  if (getDeviceComputeCapability() < hopperComputeCapability ||
      getDeviceComputeCapability() >= blackwellComputeCapability) {
    GTEST_SKIP();
  }

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  const size_t num_tensors = first_dims_h.size();
  size_t R_total = 0;
  for (size_t m : first_dims_h) {
    ASSERT_EQ(m % kBlock, 0u) << "Per-tensor first dim must be multiple of 128";
    R_total += m;
  }
  ASSERT_EQ(K % 16u, 0u);

  // Host data
  std::mt19937 gen(0xC0FFEEu);
  std::uniform_real_distribution<float> dist(-2.0f, 1.0f);
  std::vector<InputType> input_h(R_total * K);
  for (auto& v : input_h) v = static_cast<InputType>(dist(gen));

  // Tensor offsets (element offsets)
  std::vector<int64_t> offsets_h(num_tensors + 1, 0);
  for (size_t t = 0; t < num_tensors; ++t) {
    offsets_h[t + 1] = offsets_h[t] + static_cast<int64_t>(first_dims_h[t] * K);
  }
  std::vector<int64_t> first_dims_i64(num_tensors);
  for (size_t t = 0; t < num_tensors; ++t) first_dims_i64[t] = static_cast<int64_t>(first_dims_h[t]);

  const bool use_rowwise = (dir == ScalingDir::ROWWISE || dir == ScalingDir::BOTH);
  const bool use_colwise = (dir == ScalingDir::COLWISE || dir == ScalingDir::BOTH);

  const NVTEScalingMode mode =
      (block_dim == BlockDim::ONE_D) ? NVTE_BLOCK_SCALING_1D : NVTE_BLOCK_SCALING_2D;

  // Allocate grouped device buffers.
  InputType* input_d = nullptr;
  OutputType* output_d = nullptr;
  OutputType* output_t_d = nullptr;
  float* scale_inv_d = nullptr;
  float* scale_inv_t_d = nullptr;
  int64_t* offsets_d = nullptr;
  int64_t* first_dims_d = nullptr;

  const size_t blocks_X = (K + kBlock - 1) / kBlock;

  // Grouped scale buffers are sized as the sum of per-expert padded sub-blocks,
  // matching cuBLAS grouped FP8 block-scaling GEMM's per-expert layout.
  size_t scale_inv_elems = 0;
  size_t scale_inv_t_elems = 0;
  for (size_t t = 0; t < num_tensors; ++t) {
    scale_inv_elems += per_expert_scale_floats(block_dim, /*columnwise=*/false, first_dims_h[t], K);
    scale_inv_t_elems +=
        per_expert_scale_floats(block_dim, /*columnwise=*/true, first_dims_h[t], K);
  }
  std::vector<size_t> scale_inv_shape = {scale_inv_elems};
  std::vector<size_t> scale_inv_t_shape = {scale_inv_t_elems};

  const size_t input_bytes = R_total * K * sizeof(InputType);
  const size_t output_bytes = R_total * K * sizeof(OutputType);

  cudaMalloc(&input_d, input_bytes);
  cudaMemcpy(input_d, input_h.data(), input_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&offsets_d, (num_tensors + 1) * sizeof(int64_t));
  cudaMemcpy(offsets_d, offsets_h.data(), (num_tensors + 1) * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  if (shape_rep == ShapeRep::VARYING_FIRST_DIM) {
    cudaMalloc(&first_dims_d, num_tensors * sizeof(int64_t));
    cudaMemcpy(first_dims_d, first_dims_i64.data(), num_tensors * sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }
  if (use_rowwise) {
    cudaMalloc(&output_d, output_bytes);
    cudaMemset(output_d, 0, output_bytes);
    cudaMalloc(&scale_inv_d, scale_inv_elems * sizeof(float));
    cudaMemset(scale_inv_d, 0, scale_inv_elems * sizeof(float));
  }
  if (use_colwise) {
    cudaMalloc(&output_t_d, output_bytes);
    cudaMemset(output_t_d, 0, output_bytes);
    cudaMalloc(&scale_inv_t_d, scale_inv_t_elems * sizeof(float));
    cudaMemset(scale_inv_t_d, 0, scale_inv_t_elems * sizeof(float));
  }

  // Build grouped tensors.
  std::vector<size_t> logical_shape_vec = {R_total, K};
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());

  NVTEGroupedTensor in_gt = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors,
                                                       logical_shape);
  NVTEGroupedTensor out_gt = nvte_create_grouped_tensor(mode, num_tensors, logical_shape);

  NVTEBasicTensor in_data = {input_d, static_cast<NVTEDType>(itype), logical_shape};
  nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedRowwiseData, &in_data, sizeof(in_data));

  NVTEShape offsets_shape;
  offsets_shape.ndim = 1;
  offsets_shape.data[0] = num_tensors + 1;
  NVTEBasicTensor offsets_bt = {offsets_d, kNVTEInt64, offsets_shape};
  if (shape_rep == ShapeRep::VARYING_FIRST_DIM) {
    NVTEShape first_dims_shape;
    first_dims_shape.ndim = 1;
    first_dims_shape.data[0] = num_tensors;
    NVTEBasicTensor first_dims_bt = {first_dims_d, kNVTEInt64, first_dims_shape};
    nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedFirstDims, &first_dims_bt,
                                  sizeof(first_dims_bt));
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedFirstDims, &first_dims_bt,
                                  sizeof(first_dims_bt));
    nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedTensorOffsets, &offsets_bt,
                                  sizeof(offsets_bt));
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedTensorOffsets, &offsets_bt,
                                  sizeof(offsets_bt));
  }

  if (use_rowwise) {
    NVTEBasicTensor out_data = {output_d, static_cast<NVTEDType>(otype), logical_shape};
    NVTEShape scale_inv_shape_nv = nvte_make_shape(scale_inv_shape.data(), scale_inv_shape.size());
    NVTEBasicTensor scale_bt = {scale_inv_d, kNVTEFloat32, scale_inv_shape_nv};
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedRowwiseData, &out_data, sizeof(out_data));
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedRowwiseScaleInv, &scale_bt, sizeof(scale_bt));
  }
  if (use_colwise) {
    NVTEBasicTensor out_t_data = {output_t_d, static_cast<NVTEDType>(otype), logical_shape};
    NVTEShape scale_inv_t_shape_nv = nvte_make_shape(scale_inv_t_shape.data(),
                                                     scale_inv_t_shape.size());
    NVTEBasicTensor scale_t_bt = {scale_inv_t_d, kNVTEFloat32, scale_inv_t_shape_nv};
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedColumnwiseData, &out_t_data,
                                  sizeof(out_t_data));
    nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedColumnwiseScaleInv, &scale_t_bt,
                                  sizeof(scale_t_bt));
  }

  // Run grouped quantize.
  QuantizationConfigWrapper quant_config;
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(epsilon);
  nvte_group_quantize(in_gt, out_gt, quant_config, 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  // Pull grouped outputs back to host.
  std::vector<uint8_t> output_h(use_rowwise ? R_total * K : 0);
  std::vector<uint8_t> output_t_h(use_colwise ? R_total * K : 0);
  std::vector<float> scale_inv_h(use_rowwise ? scale_inv_elems : 0);
  std::vector<float> scale_inv_t_h(use_colwise ? scale_inv_t_elems : 0);
  if (use_rowwise) {
    cudaMemcpy(output_h.data(), output_d, R_total * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(scale_inv_h.data(), scale_inv_d, scale_inv_elems * sizeof(float),
               cudaMemcpyDeviceToHost);
  }
  if (use_colwise) {
    cudaMemcpy(output_t_h.data(), output_t_d, R_total * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(scale_inv_t_h.data(), scale_inv_t_d, scale_inv_t_elems * sizeof(float),
               cudaMemcpyDeviceToHost);
  }

  // Run split-quantize reference per tensor and compare element-wise.
  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t row_offset = static_cast<size_t>(offsets_h[t]) / K;

    std::vector<size_t> tshape = {M, K};
    Tensor ref_in("ref_in_" + std::to_string(t), tshape, itype);
    // The non-grouped 2D kernel requires rowwise output to be allocated even when only colwise
    // data is consumed. We always allocate both and compare only what the grouped kernel produced.
    const bool ref_rowwise = (block_dim == BlockDim::TWO_D) ? true : use_rowwise;
    const bool ref_colwise = use_colwise;
    Tensor ref_out("ref_out_" + std::to_string(t), tshape, otype, ref_rowwise, ref_colwise, mode);

    // Copy this tensor's input slice into ref_in.
    {
      auto* dst = ref_in.rowwise_dptr();
      const InputType* src = reinterpret_cast<const InputType*>(input_d) + row_offset * K;
      cudaMemcpy(dst, src, M * K * sizeof(InputType), cudaMemcpyDeviceToDevice);
    }

    QuantizationConfigWrapper qc;
    qc.set_force_pow_2_scales(force_pow_2_scales);
    qc.set_amax_epsilon(epsilon);
    nvte_quantize_v2(ref_in.data(), ref_out.data(), qc, 0);
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ref_out.to_cpu();  // sync output and scale_inv buffers from GPU to CPU

    // Compare data.
    if (use_rowwise) {
      const OutputType* ref_data = ref_out.rowwise_cpu_dptr<OutputType>();
      for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < K; ++c) {
          const uint8_t got = output_h[(row_offset + r) * K + c];
          const uint8_t exp = reinterpret_cast<const uint8_t*>(ref_data)[r * K + c];
          ASSERT_EQ(got, exp) << "rowwise data mismatch t=" << t << " r=" << r << " c=" << c;
        }
      }
    }
    if (use_colwise) {
      const OutputType* ref_data_t = ref_out.columnwise_cpu_dptr<OutputType>();
      // Per-expert columnwise data: contiguous (K, M_t) block at element offset
      // K * row_offset, matching cuBLAS grouped GEMM's per-expert data pointer.
      const size_t expert_data_off = static_cast<size_t>(row_offset) * K;
      for (size_t c = 0; c < K; ++c) {
        for (size_t r = 0; r < M; ++r) {
          const uint8_t got = output_t_h[expert_data_off + c * M + r];
          const uint8_t exp = reinterpret_cast<const uint8_t*>(ref_data_t)[c * M + r];
          ASSERT_EQ(got, exp) << "colwise data mismatch t=" << t << " c=" << c << " r=" << r;
        }
      }
    }

    // Compare scales. Per-expert layout: each expert's scales live in a
    // contiguous sub-block at per_expert_scale_offset(...).
    if (block_dim == BlockDim::ONE_D) {
      const size_t M_pad = align4(M);
      const size_t K_pad = align4(K);
      const size_t blocks_y_per_tensor = M / kBlock;
      if (use_rowwise) {
        const float* ref_sc = ref_out.rowwise_cpu_scale_inv_ptr<float>();
        // Per-expert RW: (blocks_X, roundup(M_t, 4)).
        const size_t expert_off =
            per_expert_scale_offset(first_dims_h, t, block_dim, false, K);
        for (size_t bx = 0; bx < blocks_X; ++bx) {
          for (size_t r = 0; r < M; ++r) {
            const float got = scale_inv_h[expert_off + bx * M_pad + r];
            const float exp = ref_sc[bx * M_pad + r];
            ASSERT_EQ(got, exp) << "1D rowwise scale mismatch t=" << t << " bx=" << bx
                                << " r=" << r;
          }
        }
      }
      if (use_colwise) {
        const float* ref_sct = ref_out.columnwise_cpu_scale_inv_ptr<float>();
        // Per-expert CW: (blocks_y_t, roundup(K, 4)) compact.
        const size_t expert_off = per_expert_scale_offset(first_dims_h, t, block_dim, true, K);
        for (size_t by = 0; by < blocks_y_per_tensor; ++by) {
          for (size_t c = 0; c < K; ++c) {
            const float got = scale_inv_t_h[expert_off + by * K_pad + c];
            const float exp = ref_sct[by * K_pad + c];
            ASSERT_EQ(got, exp) << "1D colwise scale mismatch t=" << t << " by=" << by
                                << " c=" << c;
          }
        }
      }
    } else {
      // 2D per-expert: rowwise (blocks_y_t, roundup(blocks_X, 4)); colwise
      // (blocks_X, roundup(blocks_y_t, 4)) compact.
      const size_t blocks_y_per_tensor = M / kBlock;
      const size_t bx_pad = align4(blocks_X);
      const size_t by_pad_t = align4(blocks_y_per_tensor);
      if (use_rowwise) {
        const float* ref_sc = ref_out.rowwise_cpu_scale_inv_ptr<float>();
        const size_t expert_off =
            per_expert_scale_offset(first_dims_h, t, block_dim, false, K);
        for (size_t by = 0; by < blocks_y_per_tensor; ++by) {
          for (size_t bx = 0; bx < blocks_X; ++bx) {
            const float got = scale_inv_h[expert_off + by * bx_pad + bx];
            const float exp = ref_sc[by * bx_pad + bx];
            ASSERT_EQ(got, exp) << "2D rowwise scale mismatch t=" << t << " by=" << by
                                << " bx=" << bx;
          }
        }
      }
      if (use_colwise) {
        const float* ref_sct = ref_out.columnwise_cpu_scale_inv_ptr<float>();
        const size_t expert_off = per_expert_scale_offset(first_dims_h, t, block_dim, true, K);
        for (size_t bx = 0; bx < blocks_X; ++bx) {
          for (size_t by = 0; by < blocks_y_per_tensor; ++by) {
            const float got = scale_inv_t_h[expert_off + bx * by_pad_t + by];
            const float exp = ref_sct[bx * by_pad_t + by];
            ASSERT_EQ(got, exp) << "2D colwise scale mismatch t=" << t << " bx=" << bx
                                << " by=" << by;
          }
        }
      }
    }
  }

  nvte_destroy_grouped_tensor(in_gt);
  nvte_destroy_grouped_tensor(out_gt);
  cudaFree(input_d);
  if (output_d) cudaFree(output_d);
  if (output_t_d) cudaFree(output_t_d);
  if (scale_inv_d) cudaFree(scale_inv_d);
  if (scale_inv_t_d) cudaFree(scale_inv_t_d);
  cudaFree(offsets_d);
  if (first_dims_d) cudaFree(first_dims_d);
}

struct TestConfig {
  ShapeRep shape_rep;
  BlockDim block_dim;
  ScalingDir dir;
  std::vector<size_t> first_dims;
  size_t K;
};

class GroupedFP8BlockwiseTestSuite : public ::testing::TestWithParam<TestConfig> {};

TEST_P(GroupedFP8BlockwiseTestSuite, Test) {
  const TestConfig& cfg = GetParam();
  perform_test<bf16, fp8e4m3>(cfg.shape_rep, cfg.block_dim, cfg.dir, cfg.first_dims, cfg.K,
                              /*force_pow_2_scales=*/false, /*epsilon=*/0.0f);
}

std::vector<TestConfig> make_configs() {
  std::vector<TestConfig> configs;
  std::vector<std::vector<size_t>> uniform = {{128, 128}, {256, 256, 256, 256}};
  std::vector<std::vector<size_t>> jagged = {
      {128, 256, 384, 512}, {256, 128, 512, 384, 1024}};
  std::vector<size_t> Ks = {128, 256, 512};
  for (auto bd : {BlockDim::ONE_D, BlockDim::TWO_D}) {
    for (auto dir : {ScalingDir::ROWWISE, ScalingDir::COLWISE, ScalingDir::BOTH}) {
      for (size_t K : Ks) {
        for (const auto& v : uniform) {
          configs.push_back({ShapeRep::SAME_BOTH_DIMS, bd, dir, v, K});
        }
        for (const auto& v : jagged) {
          configs.push_back({ShapeRep::VARYING_FIRST_DIM, bd, dir, v, K});
        }
      }
    }
  }
  return configs;
}

std::string make_name(const ::testing::TestParamInfo<TestConfig>& info) {
  const auto& c = info.param;
  std::string s = (c.shape_rep == ShapeRep::SAME_BOTH_DIMS ? "SAME" : "VARYFIRST");
  s += "_BD" + std::to_string(static_cast<int>(c.block_dim));
  s += (c.dir == ScalingDir::ROWWISE ? "_RW"
                                     : c.dir == ScalingDir::COLWISE ? "_CW" : "_BOTH");
  s += "_K" + std::to_string(c.K) + "_N" + std::to_string(c.first_dims.size());
  s += "_M";
  for (size_t m : c.first_dims) s += "_" + std::to_string(m);
  return s;
}

INSTANTIATE_TEST_SUITE_P(GroupedFP8Blockwise, GroupedFP8BlockwiseTestSuite,
                         ::testing::ValuesIn(make_configs()), make_name);

}  // namespace
