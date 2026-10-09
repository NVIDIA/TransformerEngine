/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <map>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"
#include "test_normalization.h"

using namespace transformer_engine;
using namespace test;

namespace {

using fp8e8m0 = byte;

template <typename InputType, typename ScaleType, typename OutputType>
void dequantize_1x_kernel(InputType* input_ptr, ScaleType* scale_ptr, OutputType* output_ptr,
  size_t rows, size_t cols, size_t scaling_mode_x, size_t scaling_mode_y){

  const size_t block_size_Y = scaling_mode_x;   // mind the mapping Y <-- x
  const size_t block_size_X = scaling_mode_y;   //              and X <-- y
  const size_t tile_size_Y = std::max(32lu, block_size_Y);
  const size_t tile_size_X = std::max(64lu, block_size_X);
  const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
  const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;
  const size_t blocks_per_tile_Y = tile_size_Y / block_size_Y;
  const size_t blocks_per_tile_X = tile_size_X / block_size_X;
  const size_t blocks_per_row = (cols + block_size_X - 1) / block_size_X;

  #pragma omp parallel for proc_bind(spread) schedule(static)
  for (size_t t = 0; t < tiles_num_Y * tiles_num_X; ++t) {
      const size_t tile_Y = t / tiles_num_X;
      const size_t tile_X = t % tiles_num_X;
      const size_t tile_offset_Y = tile_Y * tile_size_Y;
      const size_t tile_offset_X = tile_X * tile_size_X;

      for (size_t ii = 0; ii < blocks_per_tile_Y; ++ii) {
          const size_t block_idx_Y = tile_Y * blocks_per_tile_Y + ii;
          const size_t block_offset_Y = ii * block_size_Y;
          const size_t i_min = tile_offset_Y + block_offset_Y;
          const size_t i_max = std::min(i_min + block_size_Y, rows);

          for (size_t jj = 0; jj < blocks_per_tile_X; ++jj) {
              const size_t block_idx_X = tile_X * blocks_per_tile_X + jj;
              const size_t block_offset_X = jj * block_size_X;
              const size_t j_min = tile_offset_X + block_offset_X;
              const size_t j_max = std::min(j_min + block_size_X, cols);

              const size_t mx_scale_idx = block_idx_Y * blocks_per_row + block_idx_X;

              // TODO: padded SFs i.e. (4,128)
              const float scale_inv = exp2f(static_cast<float>(scale_ptr[mx_scale_idx]) - FP32_EXPONENT_BIAS);
              for (size_t i = i_min; i < i_max; ++i) {
                  for (size_t j = j_min; j < j_max; ++j) {
                    const size_t idx = i * cols + j;
                    const float elem = static_cast<float>(input_ptr[idx]);
                    output_ptr[idx] = static_cast<float>(elem * scale_inv);
                  }
              }
          }
      }
  }
}

template <typename InputType, typename ScaleType>
void dequantize_2x(Tensor& input, Tensor& output, bool is_training)
{
  input.to_cpu();
  auto scaling_mode = input.scaling_mode();
  assert(input.rowwise_shape().ndim == 2);

  if (is_training) {
    assert(input.columnwise_shape().ndim == 2);
  }

  dequantize_1x_kernel(input.rowwise_cpu_dptr<InputType>(),
                       input.rowwise_cpu_scale_inv_ptr<ScaleType>(),
                       output.rowwise_cpu_dptr<float>(),
                       input.rowwise_shape().data[0], input.rowwise_shape().data[1],
                       1, 32);
  if (is_training)
    dequantize_1x_kernel(input.columnwise_cpu_dptr<InputType>(),
                         input.columnwise_cpu_scale_inv_ptr<ScaleType>(),
                         output.columnwise_cpu_dptr<float>(),
                         input.columnwise_shape().data[0], input.columnwise_shape().data[1],
                         32, 1);
}

template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H, const bool zero_centered_gamma, NormType norm_type, bool is_training, const bool zero_centered_gamma_in_weight_dtype,
                 const bool use_cudnn) {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }

  using WeightType = InputType;
  DType itype = TypeInfo<InputType>::dtype;
  DType wtype = TypeInfo<WeightType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor z("z", std::vector<size_t>{ N, H }, otype, true, is_training, NVTE_MXFP8_1D_SCALING);
  Tensor gamma("gamma", std::vector<size_t>{ H }, wtype);
  Tensor beta("beta", std::vector<size_t>{ H }, wtype);
  Tensor mu("mu", std::vector<size_t>{ N }, DType::kFloat32);
  Tensor rsigma("rsigma", std::vector<size_t>{ N }, DType::kFloat32);
  Tensor workspace;


  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&beta);

  if (zero_centered_gamma_in_weight_dtype) {
    nvte_enable_zero_centered_gamma_in_weight_dtype(true);
  } else {
    nvte_enable_zero_centered_gamma_in_weight_dtype(false);
  }
  // RMSNorm shapes that are multiples of 128 use Transformer Engine's fused MXFP8 kernel
  // unless cuDNN is requested.
  nvte_enable_cudnn_norm_fwd(use_cudnn);

  // Forward kernel
  float epsilon = 1e-5;
  if (norm_type == NormType::LayerNorm){
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma,
                       0);
    workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma,
                       0);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, zero_centered_gamma,
                     0);

    workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, zero_centered_gamma,
                     0);
  }

  if (zero_centered_gamma_in_weight_dtype) {
    nvte_enable_zero_centered_gamma_in_weight_dtype(false);
  }
  nvte_enable_cudnn_norm_fwd(false);

  Tensor dequantized_output("dequantized_output", std::vector<size_t>{ N, H }, DType::kFloat32, true, true);

  dequantize_2x<OutputType, fp8e8m0>(z, dequantized_output, is_training);

  // Reference implementations
  std::unique_ptr<float[]> ref_mu = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> ref_rsigma = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> ref_output = std::make_unique<float[]>(N * H);


  compute_ref_stats(norm_type, input.rowwise_cpu_dptr<InputType>(), ref_mu.get(),
                    ref_rsigma.get(), N, H, epsilon);
  // use the GPU stats to tighten the tolerances
  float *ref_mu_ptr, *ref_rsigma_ptr;
  if (is_training){
    mu.to_cpu();
    rsigma.to_cpu();
    ref_mu_ptr = mu.rowwise_cpu_dptr<float>();
    ref_rsigma_ptr = rsigma.rowwise_cpu_dptr<float>();
  } else {
    ref_mu_ptr = ref_mu.get();
    ref_rsigma_ptr = ref_rsigma.get();
  }
  compute_ref_output(norm_type, input.rowwise_cpu_dptr<InputType>(),
                     gamma.rowwise_cpu_dptr<WeightType>(),
                     beta.rowwise_cpu_dptr<WeightType>(),
                     ref_output.get(),
                     ref_mu_ptr,
                     ref_rsigma_ptr,
                     N, H,
                     nullptr, // amax
                     1.f, // scale
                     zero_centered_gamma,
                     true, // The MXFP8 backends add 1 to a zero-centered gamma as cuDNN does
                     zero_centered_gamma_in_weight_dtype);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol_stats, rtol_stats] = getTolerances(DType::kFloat32);
  rtol_stats = 5e-5;
  if (is_training){
    compareResults("mu", mu, ref_mu.get(), true, atol_stats, rtol_stats);
    compareResults("rsigma", rsigma, ref_rsigma.get(), true, atol_stats, rtol_stats);
  }

  float atol, rtol;
  if (otype == DType::kFloat8E5M2){
    atol = 1.25e-1;
    rtol = 1.25e-1;
  } else if (otype == DType::kFloat8E4M3){
    if (itype == DType::kBFloat16){
      atol = 7e-2;
      rtol = 7e-2;
    } else {
      atol = 6.25e-2;
      rtol = 6.25e-2;
    }
  }
  compareResults("output_rowwise", dequantized_output, ref_output.get(), true, atol, rtol, false);
  if (is_training)
    compareResults("output_colwise", dequantized_output, ref_output.get(), false, atol, rtol, false);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
  {32, 32},
  {768, 2304},
  {2048, 12288},
};

// Index of compact scaling factor (i, j) in the GEMM-swizzled layout: 128x4 tiles of i x j,
// num_tiles_j tiles along j.
size_t swizzled_scale_idx(size_t i, size_t j, size_t num_tiles_j) {
  return ((i / 128) * num_tiles_j + j / 4) * 512 + (i % 32) * 16 + ((i % 128) / 32) * 4 + j % 4;
}

// RMSNorm with MXFP8 output written with GEMM-swizzled scaling factors must give the same
// data as with compact scaling factors, and the compact scaling factors in swizzled order.
// The swizzled output is computed on a single SM, as with an SM margin, which launches the
// kernels over chunks of rows.
template <typename InputType, typename OutputType>
void performSwizzledScalesTest(const size_t N, const size_t H, const bool is_training) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;
  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor gamma("gamma", std::vector<size_t>{ H }, itype);
  Tensor rsigma("rsigma", std::vector<size_t>{ N }, DType::kFloat32);
  Tensor rsigma_swizzled("rsigma_swizzled", std::vector<size_t>{ N }, DType::kFloat32);
  Tensor z("z", std::vector<size_t>{ N, H }, otype, true, is_training, NVTE_MXFP8_1D_SCALING);
  Tensor z_swizzled("z_swizzled", std::vector<size_t>{ N, H }, otype, true, is_training,
                    NVTE_MXFP8_1D_SCALING);
  z_swizzled.set_with_gemm_swizzled_scales(true);
  fillUniform(&input);
  fillUniform(&gamma);

  const auto run = [&](Tensor &out, Tensor &rs, const int sm_count) {
    Tensor workspace;
    nvte_rmsnorm_fwd(input.data(), gamma.data(), 1e-5f, out.data(), rs.data(),
                     workspace.data(), sm_count, false, 0);
    workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
    nvte_rmsnorm_fwd(input.data(), gamma.data(), 1e-5f, out.data(), rs.data(),
                     workspace.data(), sm_count, false, 0);
  };
  run(z, rsigma, prop.multiProcessorCount);
  run(z_swizzled, rsigma_swizzled, 1);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  z.to_cpu();
  z_swizzled.to_cpu();
  rsigma.to_cpu();
  compareResults("rsigma", rsigma_swizzled, rsigma.rowwise_cpu_dptr<float>(), true, 0, 0);

  const auto bytes = [](const OutputType *p) { return reinterpret_cast<const uint8_t *>(p); };
  compareResults("rowwise_data", bytes(z_swizzled.rowwise_cpu_dptr<OutputType>()),
                 bytes(z.rowwise_cpu_dptr<OutputType>()), N * H);
  const NVTEShape row_shape = z.rowwise_scale_inv_shape();
  const size_t row_dim0 = row_shape.data[0], row_dim1 = row_shape.data[1];
  std::vector<uint8_t> ref(row_dim0 * row_dim1);
  const uint8_t *compact = z.rowwise_cpu_scale_inv_ptr<uint8_t>();
  for (size_t i = 0; i < row_dim0; ++i) {
    for (size_t j = 0; j < row_dim1; ++j) {
      ref[swizzled_scale_idx(i, j, row_dim1 / 4)] = compact[i * row_dim1 + j];
    }
  }
  compareResults("rowwise_scales", z_swizzled.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref.data(),
                 ref.size());

  if (is_training) {
    compareResults("colwise_data", bytes(z_swizzled.columnwise_cpu_dptr<OutputType>()),
                   bytes(z.columnwise_cpu_dptr<OutputType>()), N * H);
    // Compact column-wise scaling factors are [rows / 32, cols]; the swizzled layout tiles
    // columns by 128 and scale rows by 4.
    const NVTEShape col_shape = z.columnwise_scale_inv_shape();
    const size_t col_dim0 = col_shape.data[0], col_dim1 = col_shape.data[1];
    ref.assign(col_dim0 * col_dim1, 0);
    compact = z.columnwise_cpu_scale_inv_ptr<uint8_t>();
    for (size_t i = 0; i < col_dim0; ++i) {
      for (size_t j = 0; j < col_dim1; ++j) {
        ref[swizzled_scale_idx(j, i, col_dim0 / 4)] = compact[i * col_dim1 + j];
      }
    }
    compareResults("colwise_scales", z_swizzled.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                   ref.data(), ref.size());
  }
}

std::vector<std::pair<size_t, size_t>> swizzled_scales_test_cases = {
  {128, 128},
  {768, 2304},
  {256, 7168},
};

std::vector<NormType> norms = {
  NormType::LayerNorm,
  NormType::RMSNorm
};

}  // namespace

class MxNormTestSuite : public ::testing::TestWithParam< std::tuple<NormType,
                                                                    transformer_engine::DType,
                                                                    transformer_engine::DType,
                                                                    std::pair<size_t, size_t>,
                                                                    bool, bool, bool, bool>> {};

TEST_P(MxNormTestSuite, TestMxNorm) {
  using namespace transformer_engine;
  using namespace test;

  const NormType norm_type = std::get<0>(GetParam());
  const DType input_type = std::get<1>(GetParam());
  const DType output_type = std::get<2>(GetParam());
  const auto size = std::get<3>(GetParam());
  const bool zero_centered_gamma = std::get<4>(GetParam());
  const bool is_training = std::get<5>(GetParam());
  const bool zero_centered_gamma_in_weight_dtype = std::get<6>(GetParam());
  const bool use_cudnn = std::get<7>(GetParam());
  if (norm_type == NormType::LayerNorm && use_cudnn) {
    GTEST_SKIP() << "LayerNorm with MXFP8 output always uses cuDNN";
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
      performTest<InputType, OutputType>(size.first, size.second, zero_centered_gamma, norm_type, is_training, zero_centered_gamma_in_weight_dtype,
                                         use_cudnn);
    );
  );
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MxNormTestSuite,
  ::testing::Combine(
    ::testing::Values(NormType::LayerNorm, NormType::RMSNorm),
    ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
    ::testing::Values(DType::kFloat8E5M2, DType::kFloat8E4M3),
    ::testing::ValuesIn(test_cases),
    ::testing::Values(true, false),
    ::testing::Values(true, false),
    ::testing::Values(true, false),
    ::testing::Values(false, true)),
  [](const testing::TestParamInfo<MxNormTestSuite::ParamType>& info) {
    std::string name = normToString.at(std::get<0>(info.param)) + "_" +
      test::typeName(std::get<1>(info.param)) + "X" +
      test::typeName(std::get<2>(info.param)) + "X" +
      std::to_string(std::get<3>(info.param).first) + "X" +
      std::to_string(std::get<3>(info.param).second) + "X" +
      std::to_string(std::get<4>(info.param)) + "out" +
      std::to_string(int(std::get<5>(info.param)) + 1) + "x" +
      std::to_string(std::get<6>(info.param)) +
      (std::get<7>(info.param) ? "Xcudnn" : "");
    return name;
  });

class MxNormSwizzledScalesTestSuite
    : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                 transformer_engine::DType,
                                                 std::pair<size_t, size_t>, bool>> {};

TEST_P(MxNormSwizzledScalesTestSuite, TestMxRMSNormSwizzledScales) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());
  const bool is_training = std::get<3>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
      performSwizzledScalesTest<InputType, OutputType>(size.first, size.second, is_training);
    );
  );
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MxNormSwizzledScalesTestSuite,
  ::testing::Combine(
    ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
    ::testing::Values(DType::kFloat8E5M2, DType::kFloat8E4M3),
    ::testing::ValuesIn(swizzled_scales_test_cases),
    ::testing::Values(true, false)),
  [](const testing::TestParamInfo<MxNormSwizzledScalesTestSuite::ParamType>& info) {
    return test::typeName(std::get<0>(info.param)) + "X" +
           test::typeName(std::get<1>(info.param)) + "X" +
           std::to_string(std::get<2>(info.param).first) + "X" +
           std::to_string(std::get<2>(info.param).second) + "X" +
           std::to_string(int(std::get<3>(info.param)) + 1) + "x";
  });
