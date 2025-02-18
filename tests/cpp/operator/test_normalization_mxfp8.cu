/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

using e8m0_t = byte;

enum NormType {
  LayerNorm,
  RMSNorm
};

std::map<NormType, std::string> normToString = {
  {NormType::LayerNorm, "LayerNorm"},
  {NormType::RMSNorm, "RMSNorm"}
};

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
void dequantize_2x(Tensor& input, Tensor& output_rowwise, Tensor& output_colwise)
{
  input.to_cpu();
  auto scaling_mode = input.scaling_mode();
  assert(input.shape().ndim == 2);
  assert(input.columnwise_shape().ndim == 2);

  dequantize_1x_kernel(input.cpu_dptr<InputType>(),
                       input.cpu_scale_inv_ptr<ScaleType>(),
                       output_rowwise.cpu_dptr<float>(),
                       input.shape().data[0], input.shape().data[1],
                       scaling_mode.x, scaling_mode.y);
  dequantize_1x_kernel(input.columnwise_cpu_dptr<InputType>(),
                       input.columnwise_cpu_scale_inv_ptr<ScaleType>(),
                       output_colwise.cpu_dptr<float>(),
                       input.columnwise_shape().data[0], input.columnwise_shape().data[1],
                       scaling_mode.y, scaling_mode.x);

}

template <typename InputType>
void compute_ref_stats(NormType norm_type,
                       const InputType *data, float *mu, float *rsigma,
                       const size_t N, const size_t H, const double epsilon){
  using compute_t = float;

  #pragma omp parallel for proc_bind(spread)
  for (size_t i = 0; i < N; ++i) {
    compute_t sum = 0;
    for (size_t j = 0; j < H; ++j) {
      sum += static_cast<compute_t>(data[i * H + j]);
    }
    compute_t m;
    if (norm_type == LayerNorm){
      mu[i] = sum / H;
      m = mu[i];
    } else { m = 0;}

    compute_t sum_sq = 0;
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      sum_sq += (current - m) * (current - m);
    }
    rsigma[i] = rsqrtf((sum_sq / H) + epsilon);
  }
}

template <typename InputType, typename OutputType>
void compute_ref_output(NormType norm_type,
                        const InputType *data, const InputType *gamma, const InputType *beta,
                        const float *mu, const float *rsigma,
                        const size_t N, const size_t H,
                        OutputType* output,
                        const bool zero_centered_gamma){
  using compute_t = float;

  #pragma omp parallel for proc_bind(spread)
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      compute_t g = static_cast<compute_t>(gamma[j]);
      if (zero_centered_gamma) {
        g += 1.0;
      }

      compute_t tmp;
      if (norm_type == LayerNorm) {
        tmp = (current - mu[i]) * rsigma[i] * g + static_cast<compute_t>(beta[j]);
      } else { // RMSNorm
        tmp = current * rsigma[i] * g;
      }

      output[i * H + j] = tmp;
    }
  }
}

template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H, const bool zero_centered_gamma, NormType norm_type) {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }

  using WeightType = InputType;
  DType itype = TypeInfo<InputType>::dtype;
  DType wtype = TypeInfo<WeightType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  const std::vector<int> row_mode = {1, 32, 0};

  Tensor input({ N, H }, itype);
  Tensor z({ N, H }, otype, row_mode, true /*is_tensor_2x*/);
  Tensor gamma({ H }, wtype);
  Tensor beta({ H }, wtype);
  Tensor mu({ N }, DType::kFloat32);
  Tensor rsigma({ N }, DType::kFloat32);
  Tensor workspace;


  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&beta);

  // Forward kernel
  float epsilon = 1e-5;
  if (norm_type == NormType::LayerNorm){
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma,
                       0);
    workspace = Tensor(workspace.shape(), workspace.dtype());
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace.data(),
                       prop.multiProcessorCount, zero_centered_gamma,
                       0);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, zero_centered_gamma,
                     0);

    workspace = Tensor(workspace.shape(), workspace.dtype());
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace.data(),
                     prop.multiProcessorCount, zero_centered_gamma,
                     0);
  }

  Tensor dequantized_rowwise_output({ N, H }, DType::kFloat32);
  Tensor dequantized_colwise_output({ N, H }, DType::kFloat32);

  dequantize_2x<OutputType, e8m0_t>(z, dequantized_rowwise_output, dequantized_colwise_output);

  // Reference implementations
  std::unique_ptr<float[]> ref_mu = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> ref_rsigma = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> ref_output = std::make_unique<float[]>(N * H);


  compute_ref_stats(norm_type, input.cpu_dptr<InputType>(), ref_mu.get(),
                    ref_rsigma.get(), N, H, epsilon);
  // use the GPU stats to tighten the tolerances
  mu.to_cpu();
  rsigma.to_cpu();
  compute_ref_output(norm_type, input.cpu_dptr<InputType>(),
                     gamma.cpu_dptr<WeightType>(),
                     beta.cpu_dptr<WeightType>(),
                     mu.cpu_dptr<float>(),
                     rsigma.cpu_dptr<float>(),
                     N, H,
                     ref_output.get(),
                     zero_centered_gamma);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol_stats, rtol_stats] = getTolerances(DType::kFloat32);
  rtol_stats = 5e-5;
  compareResults("mu", mu, ref_mu.get(), atol_stats, rtol_stats);
  compareResults("rsigma", rsigma, ref_rsigma.get(), atol_stats, rtol_stats);

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
  compareResults("output_rowwise", dequantized_rowwise_output, ref_output.get(), atol, rtol, false);
  compareResults("output_colwise", dequantized_colwise_output, ref_output.get(), atol, rtol, false);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
  {32, 32},
  {768, 2304},
  {2048, 12288},
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
                                                                    bool>> {};

TEST_P(MxNormTestSuite, TestMxNorm) {
  using namespace transformer_engine;
  using namespace test;

  const NormType norm_type = std::get<0>(GetParam());
  const DType input_type = std::get<1>(GetParam());
  const DType output_type = std::get<2>(GetParam());
  const auto size = std::get<3>(GetParam());
  const bool zero_centered_gamma = std::get<4>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
      performTest<InputType, OutputType>(size.first, size.second, zero_centered_gamma, norm_type);
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
    ::testing::Values(false, true)),
  [](const testing::TestParamInfo<MxNormTestSuite::ParamType>& info) {
    std::string name = normToString.at(std::get<0>(info.param)) + "_" +
      test::typeName(std::get<1>(info.param)) + "X" +
      test::typeName(std::get<2>(info.param)) + "X" +
      std::to_string(std::get<3>(info.param).first) + "X" +
      std::to_string(std::get<3>(info.param).second) + "X" +
      std::to_string(std::get<4>(info.param));
    return name;
  });
