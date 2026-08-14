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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum class ShapeRep { SAME_BOTH_DIMS = 0, VARYING_FIRST_DIM = 1 };
enum class ScalingDir { ROWWISE = 0, COLWISE = 1 };
enum class BlockDim { ONE_D = 1, TWO_D = 2 };

constexpr size_t kBlock = 128;

inline size_t align4(size_t x) { return ((x + 3) / 4) * 4; }

// Per-expert padded scale size (in floats), matching the grouped FP8 block-scaling layout that
// the grouped quantize kernel writes and cuBLAS grouped GEMM consumes:
//   1D rowwise : blocks_X   * roundup(M_t, 4)     (scale shape {blocks_X,   roundup(M_t, 4)})
//   1D colwise : blocks_y_t * roundup(K, 4)       (scale shape {blocks_y_t, roundup(K, 4)})
//   2D rowwise : blocks_y_t * roundup(blocks_X,4) (scale shape {blocks_y_t, roundup(blocks_X, 4)})
//   2D colwise : blocks_X   * roundup(blocks_y_t,4)(scale shape {blocks_X,  roundup(blocks_y_t,4)})
inline void per_expert_scale_shape(BlockDim block_dim, bool columnwise, size_t M_t, size_t K,
                                   size_t& scale_y, size_t& scale_x) {
  const size_t blocks_X = (K + kBlock - 1) / kBlock;
  const size_t blocks_y = (M_t + kBlock - 1) / kBlock;
  if (block_dim == BlockDim::ONE_D) {
    if (!columnwise) {
      scale_y = blocks_X;
      scale_x = align4(M_t);
    } else {
      scale_y = blocks_y;
      scale_x = align4(K);
    }
  } else {
    if (!columnwise) {
      scale_y = blocks_y;
      scale_x = align4(blocks_X);
    } else {
      scale_y = blocks_X;
      scale_x = align4(blocks_y);
    }
  }
}

inline size_t per_expert_scale_floats(BlockDim block_dim, bool columnwise, size_t M_t, size_t K) {
  size_t y, x;
  per_expert_scale_shape(block_dim, columnwise, M_t, K, y, x);
  return y * x;
}

// Grouped FP8 block-scaling dequantize test.
//
// Methodology mirrors test_dequantize_mxfp8_grouped.cu: the grouped dequantize kernel is
// validated against single-tensor nvte_dequantize called in a loop for each tensor; results must
// be bitwise identical. We generate random FP8 data and random FP32 scales laid out in the
// grouped per-expert format, run nvte_group_dequantize, and for each expert slice out its data +
// scale sub-block, feed it to a per-tensor (direction-only) dequantize, and compare.
template <typename InputType, typename OutputType>
void performTest(ShapeRep shape_rep, BlockDim block_dim, bool rowwise,
                 const std::vector<size_t>& first_dims_h, size_t K) {
  // FP8 block-scaling grouped kernels are Hopper-only (SM90-SM99).
  if (getDeviceComputeCapability() < hopperComputeCapability ||
      getDeviceComputeCapability() >= blackwellComputeCapability) {
    GTEST_SKIP();
  }

  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;
  const bool columnwise = !rowwise;

  const size_t num_tensors = first_dims_h.size();
  size_t R_total = 0;
  for (size_t m : first_dims_h) {
    ASSERT_EQ(m % kBlock, 0u) << "Per-tensor first dim must be a multiple of 128";
    R_total += m;
  }
  ASSERT_EQ(K % 16u, 0u);

  const NVTEScalingMode mode =
      (block_dim == BlockDim::ONE_D) ? NVTE_BLOCK_SCALING_1D : NVTE_BLOCK_SCALING_2D;

  // Element offsets (both data and, for columnwise, the transposed (K, M_t) block are contiguous
  // per expert at element offset row_offset * K).
  std::vector<int64_t> offsets_h(num_tensors + 1, 0);
  for (size_t t = 0; t < num_tensors; ++t)
    offsets_h[t + 1] = offsets_h[t] + static_cast<int64_t>(first_dims_h[t] * K);
  std::vector<int64_t> first_dims_i64(num_tensors);
  for (size_t t = 0; t < num_tensors; ++t)
    first_dims_i64[t] = static_cast<int64_t>(first_dims_h[t]);

  // Per-expert scale sub-block offsets (in floats).
  std::vector<size_t> scale_off(num_tensors + 1, 0);
  for (size_t t = 0; t < num_tensors; ++t)
    scale_off[t + 1] =
        scale_off[t] + per_expert_scale_floats(block_dim, columnwise, first_dims_h[t], K);
  const size_t total_scales = scale_off[num_tensors];

  // ---- Random FP8 data (valid normals) + random FP32 scales ----
  std::mt19937 gen(0xD3C0DEu);
  const double minAbs = Numeric_Traits<InputType>::minNorm;
  const double maxAbs = Numeric_Traits<InputType>::maxNorm;
  std::uniform_real_distribution<> dis(minAbs, maxAbs);
  std::uniform_real_distribution<> dis_sign(-1.0, 1.0);
  std::uniform_real_distribution<float> scale_dis(0.25f, 4.0f);

  std::vector<InputType> data_h(R_total * K);
  for (auto& v : data_h) {
    double val = dis(gen);
    if (dis_sign(gen) < 0.0) val = -val;
    v = static_cast<InputType>(val);
  }
  std::vector<float> scales_h(total_scales);
  for (auto& s : scales_h) s = scale_dis(gen);

  // ---- Device buffers ----
  InputType* data_d = nullptr;
  float* scales_d = nullptr;
  OutputType* out_grouped_d = nullptr;
  int64_t* offsets_d = nullptr;
  int64_t* first_dims_d = nullptr;

  cudaMalloc(&data_d, R_total * K * sizeof(InputType));
  cudaMemcpy(data_d, data_h.data(), R_total * K * sizeof(InputType), cudaMemcpyHostToDevice);
  cudaMalloc(&scales_d, total_scales * sizeof(float));
  cudaMemcpy(scales_d, scales_h.data(), total_scales * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&out_grouped_d, R_total * K * sizeof(OutputType));
  cudaMemset(out_grouped_d, 0, R_total * K * sizeof(OutputType));
  cudaMalloc(&offsets_d, (num_tensors + 1) * sizeof(int64_t));
  cudaMemcpy(offsets_d, offsets_h.data(), (num_tensors + 1) * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  if (shape_rep == ShapeRep::VARYING_FIRST_DIM) {
    cudaMalloc(&first_dims_d, num_tensors * sizeof(int64_t));
    cudaMemcpy(first_dims_d, first_dims_i64.data(), num_tensors * sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }

  // ---- Build grouped input (quantized) + output (high precision) tensors ----
  std::vector<size_t> logical_shape_vec = {R_total, K};
  NVTEShape logical_shape = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());
  std::vector<size_t> data_1d = {R_total * K};
  NVTEShape data_shape = nvte_make_shape(data_1d.data(), data_1d.size());
  std::vector<size_t> scale_1d = {total_scales};
  NVTEShape scale_shape = nvte_make_shape(scale_1d.data(), scale_1d.size());

  NVTEShape offsets_shape;
  offsets_shape.ndim = 1;
  offsets_shape.data[0] = num_tensors + 1;
  NVTEShape first_dims_shape;
  first_dims_shape.ndim = 1;
  first_dims_shape.data[0] = num_tensors;
  NVTEBasicTensor offsets_bt = {offsets_d, kNVTEInt64, offsets_shape};
  NVTEBasicTensor first_dims_bt = {first_dims_d, kNVTEInt64, first_dims_shape};
  auto set_shape_meta = [&](NVTEGroupedTensor gt) {
    if (shape_rep == ShapeRep::VARYING_FIRST_DIM) {
      nvte_set_grouped_tensor_param(gt, kNVTEGroupedFirstDims, &first_dims_bt,
                                    sizeof(first_dims_bt));
      nvte_set_grouped_tensor_param(gt, kNVTEGroupedTensorOffsets, &offsets_bt, sizeof(offsets_bt));
    }
  };

  NVTEGroupedTensor in_gt = nvte_create_grouped_tensor(mode, num_tensors, logical_shape);
  NVTEBasicTensor in_data_bt = {data_d, static_cast<NVTEDType>(itype), data_shape};
  NVTEBasicTensor in_scale_bt = {scales_d, kNVTEFloat32, scale_shape};
  if (rowwise) {
    nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedRowwiseData, &in_data_bt, sizeof(in_data_bt));
    nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedRowwiseScaleInv, &in_scale_bt,
                                  sizeof(in_scale_bt));
  } else {
    nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedColumnwiseData, &in_data_bt,
                                  sizeof(in_data_bt));
    nvte_set_grouped_tensor_param(in_gt, kNVTEGroupedColumnwiseScaleInv, &in_scale_bt,
                                  sizeof(in_scale_bt));
  }
  set_shape_meta(in_gt);

  NVTEGroupedTensor out_gt =
      nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape);
  NVTEBasicTensor out_data_bt = {out_grouped_d, static_cast<NVTEDType>(otype), data_shape};
  nvte_set_grouped_tensor_param(out_gt, kNVTEGroupedRowwiseData, &out_data_bt, sizeof(out_data_bt));
  set_shape_meta(out_gt);

  // ---- Grouped dequantize ----
  nvte_group_dequantize(in_gt, out_gt, 0);
  cudaDeviceSynchronize();
  {
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  }
  std::vector<OutputType> out_grouped_h(R_total * K);
  cudaMemcpy(out_grouped_h.data(), out_grouped_d, R_total * K * sizeof(OutputType),
             cudaMemcpyDeviceToHost);

  // ---- Reference: host dequantize (out = float(fp8) * scale_inv) ----
  //
  // There is no non-grouped FP8 block-scaling dequantize to loop over (only the grouped path
  // implements it), so the reference is computed on the host. The per-expert scale sub-block
  // indexing mirrors what test_cast_float8blockwise_grouped.cu validates the grouped quantize
  // kernel writes; this makes the host reference an independent check of the grouped dequant.
  //   1D rowwise : scale[bx * roundup(M,4) + r]       (bx = c/128)
  //   1D colwise : scale[by * roundup(K,4) + c]       (by = r/128)
  //   2D rowwise : scale[by * roundup(blocks_X,4) + bx]
  //   2D colwise : scale[bx * roundup(blocks_y,4) + by]
  // Data is contiguous (M,K) for rowwise and transposed (K,M) for columnwise.
  auto scale_index = [&](size_t r, size_t c, size_t M) -> size_t {
    const size_t blocks_X = (K + kBlock - 1) / kBlock;
    const size_t blocks_y = (M + kBlock - 1) / kBlock;
    const size_t bx = c / kBlock;
    const size_t by = r / kBlock;
    if (block_dim == BlockDim::ONE_D) {
      return columnwise ? (by * align4(K) + c) : (bx * align4(M) + r);
    }
    return columnwise ? (bx * align4(blocks_y) + by) : (by * align4(blocks_X) + bx);
  };

  for (size_t t = 0; t < num_tensors; ++t) {
    const size_t M = first_dims_h[t];
    const size_t row_offset = static_cast<size_t>(offsets_h[t]) / K;
    const size_t data_off = row_offset * K;  // rowwise (M,K) or colwise transposed (K,M)
    const size_t s_off = scale_off[t];
    for (size_t r = 0; r < M; ++r) {
      for (size_t c = 0; c < K; ++c) {
        const size_t d_idx = columnwise ? (c * M + r) : (r * K + c);
        const float fp8v = static_cast<float>(data_h[data_off + d_idx]);
        const float sc = scales_h[s_off + scale_index(r, c, M)];
        const float ref = fp8v * sc;
        const float got = static_cast<float>(out_grouped_h[(row_offset + r) * K + c]);
        const float rel = std::fabs(got - ref) / std::max(std::fabs(ref), 1e-3f);
        ASSERT_LT(rel, 1e-2f) << "dequant mismatch t=" << t << " r=" << r << " c=" << c
                              << " got=" << got << " ref=" << ref << " (fp8=" << fp8v
                              << " scale=" << sc << ")";
      }
    }
  }

  nvte_destroy_grouped_tensor(in_gt);
  nvte_destroy_grouped_tensor(out_gt);
  cudaFree(data_d);
  cudaFree(scales_d);
  cudaFree(out_grouped_d);
  cudaFree(offsets_d);
  if (first_dims_d) cudaFree(first_dims_d);
}

struct TestConfig {
  ShapeRep shape_rep;
  BlockDim block_dim;
  bool rowwise;
  std::vector<size_t> first_dims;
  size_t K;
};

std::vector<TestConfig> make_configs() {
  std::vector<TestConfig> configs;
  std::vector<std::vector<size_t>> uniform = {{128, 128}, {256, 256, 256, 256}};
  std::vector<std::vector<size_t>> jagged = {{128, 256, 384, 512}, {256, 128, 512, 384, 1024}};
  std::vector<size_t> Ks = {128, 256, 512};
  for (auto bd : {BlockDim::ONE_D, BlockDim::TWO_D}) {
    for (bool rowwise : {true, false}) {
      for (size_t K : Ks) {
        for (const auto& v : uniform)
          configs.push_back({ShapeRep::SAME_BOTH_DIMS, bd, rowwise, v, K});
        for (const auto& v : jagged)
          configs.push_back({ShapeRep::VARYING_FIRST_DIM, bd, rowwise, v, K});
      }
    }
  }
  return configs;
}

}  // namespace

class GroupedDequantizeFP8BlockwiseTestSuite
    : public ::testing::TestWithParam<std::tuple<TestConfig, transformer_engine::DType>> {};

TEST_P(GroupedDequantizeFP8BlockwiseTestSuite, Test) {
  const TestConfig cfg = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  // FP8 block scaling is E4M3-centric (matches the grouped quantize test scope).
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      output_type, OutputType,
      performTest<fp8e4m3, OutputType>(cfg.shape_rep, cfg.block_dim, cfg.rowwise, cfg.first_dims,
                                       cfg.K););
}

INSTANTIATE_TEST_SUITE_P(
    GroupedFP8Blockwise, GroupedDequantizeFP8BlockwiseTestSuite,
    ::testing::Combine(::testing::ValuesIn(make_configs()),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<GroupedDequantizeFP8BlockwiseTestSuite::ParamType>& info) {
      const TestConfig& c = std::get<0>(info.param);
      std::string s = (c.shape_rep == ShapeRep::SAME_BOTH_DIMS ? "SAME" : "VARYFIRST");
      s += "_BD" + std::to_string(static_cast<int>(c.block_dim));
      s += (c.rowwise ? "_RW" : "_CW");
      s += "_K" + std::to_string(c.K) + "_N" + std::to_string(c.first_dims.size());
      s += "_" + test::typeName(std::get<1>(info.param));
      return s;
    });
