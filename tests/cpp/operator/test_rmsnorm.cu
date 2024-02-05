/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/rmsnorm.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename InputType>
void compute_ref_stats(const InputType *data, float *rsigma, const size_t N, const size_t H,
                       const double epsilon) {
  using compute_t = float;
  for (size_t i = 0; i < N; ++i) {
    compute_t sum = 0;
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      sum += (current) * (current);
    }
    sum = sum / H;
    compute_t rs = rsqrtf(sum + epsilon);
    rsigma[i] = rs;
  }
}

template <typename InputType, typename OutputType>
void compute_ref_output(const InputType *data, const InputType *gamma, OutputType *output,
                        const float *rsigma, const size_t N, const size_t H, float *amax,
                        float scale, const bool zero_centered_gamma) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      compute_t g = static_cast<compute_t>(gamma[j]);
      if (zero_centered_gamma) {
        g += 1;
      }
      compute_t tmp = current * rsigma[i] * g;
      output[i * H + j] = static_cast<OutputType>(tmp * scale);
      current_max = fmaxf(current_max, fabsf(tmp));
    }
  }
  *amax = current_max;
}

template <typename InputType, typename OutputType>
void compute_ref_backward(const OutputType *output_grad, const InputType *data, const float *rsigma,
                          const InputType *gamma, InputType *data_grad, InputType *gamma_grad,
                          const size_t N, const size_t H, const bool zero_centered_gamma) {
  using compute_t = float;
  std::vector<compute_t> dgamma(H, 0.f);

  for (size_t i = 0; i < N; ++i) {
    // Reductions
    compute_t mdyy = 0;
    for (size_t j = 0; j < H; ++j) {
      const compute_t x = static_cast<compute_t>(data[i * H + j]);
      const compute_t y = x * rsigma[i];
      compute_t g = static_cast<compute_t>(gamma[j]);
      if (zero_centered_gamma) {
        g += 1;
      }
      const compute_t dz = static_cast<compute_t>(output_grad[i * H + j]);
      const compute_t dy = g * dz;
      dgamma[j] += y * dz;
      mdyy += dy * y;
    }
    mdyy /= H;

    // Input grads
    for (size_t j = 0; j < H; ++j) {
      const compute_t x = static_cast<compute_t>(data[i * H + j]);
      const compute_t y = x * rsigma[i];
      compute_t g = static_cast<compute_t>(gamma[j]);
      if (zero_centered_gamma) {
        g += 1;
      }
      const compute_t dz = static_cast<compute_t>(output_grad[i * H + j]);
      const compute_t dy = g * dz;
      const compute_t dx = rsigma[i] * (dy - mdyy * y);
      data_grad[i * H + j] = static_cast<InputType>(dx);
    }
  }

  // Weight grads
  for (size_t j = 0; j < H; ++j) {
    gamma_grad[j] = static_cast<InputType>(dgamma[j]);
  }
}

template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H, const bool zero_centered_gamma) {
  if (sizeof(InputType) < sizeof(OutputType)) {
    GTEST_SKIP() << "RMSNorm kernel does not support OutputType > InputType";
    return;
  }
  using WeightType = InputType;
  DType itype = TypeInfo<InputType>::dtype;
  DType wtype = TypeInfo<WeightType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  if ((itype == DType::kBFloat16 && otype == DType::kFloat16) ||
      (itype == DType::kFloat16 && otype == DType::kBFloat16)) {
    GTEST_SKIP() << "RMSNorm kernel does not support mixing Float16 and BFloat16";
    return;
  }

  Tensor input({N, H}, itype);
  Tensor z({N, H}, otype);
  Tensor gamma({H}, wtype);
  Tensor rsigma({N}, DType::kFloat32);
  Tensor dz({N, H}, wtype);
  Tensor dx({N, H}, itype);
  Tensor dgamma({H}, wtype);
  Tensor workspace, barrier, dgamma_part;

  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&dz);
  setRandomScale(&z);

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(N * H);
  std::unique_ptr<float[]> ref_rsigma = std::make_unique<float[]>(N);
  std::unique_ptr<InputType[]> ref_dx = std::make_unique<InputType[]>(N * H);
  std::unique_ptr<WeightType[]> ref_dgamma = std::make_unique<InputType[]>(H);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  // Forward kernel
  float epsilon = 1e-5;
  auto fwd_function = zero_centered_gamma ? nvte_rmsnorm1p_fwd : nvte_rmsnorm_fwd;
  fwd_function(input.data(), gamma.data(), epsilon, z.data(), rsigma.data(), 0,
               prop.multiProcessorCount, workspace.data(), barrier.data());
  workspace = Tensor(workspace.shape(), workspace.dtype());
  barrier = Tensor(barrier.shape(), barrier.dtype());
  fwd_function(input.data(), gamma.data(), epsilon, z.data(), rsigma.data(), 0,
               prop.multiProcessorCount, workspace.data(), barrier.data());

  // Backward kernel
  auto bwd_function = zero_centered_gamma ? nvte_rmsnorm1p_bwd : nvte_rmsnorm_bwd;
  bwd_function(dz.data(), input.data(), rsigma.data(), gamma.data(), dx.data(), dgamma.data(),
               dgamma_part.data(), 0, prop.multiProcessorCount, workspace.data(),
               barrier.data());
  workspace = Tensor(workspace.shape(), workspace.dtype());
  barrier = Tensor(barrier.shape(), barrier.dtype());
  dgamma_part = Tensor(dgamma_part.shape(), dgamma_part.dtype());
  bwd_function(dz.data(), input.data(), rsigma.data(), gamma.data(), dx.data(), dgamma.data(),
               dgamma_part.data(), 0, prop.multiProcessorCount, workspace.data(),
               barrier.data());

  // Reference implementations
  // use the GPU stats to tighten the tolerances
  rsigma.to_cpu();
  float ref_amax;
  compute_ref_stats(input.cpu_dptr<InputType>(), ref_rsigma.get(), N, H, epsilon);
  float ref_scale = isFp8Type(otype) ? z.scale() : 1.f;
  compute_ref_output(input.cpu_dptr<InputType>(), gamma.cpu_dptr<WeightType>(), ref_output.get(),
                     rsigma.cpu_dptr<float>(), N, H, &ref_amax, ref_scale,
                     zero_centered_gamma);
  compute_ref_backward(dz.cpu_dptr<WeightType>(), input.cpu_dptr<InputType>(),
                       rsigma.cpu_dptr<float>(), gamma.cpu_dptr<WeightType>(), ref_dx.get(),
                       ref_dgamma.get(), N, H, zero_centered_gamma);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  if (isFp8Type(otype)) {
    compareResults("amax", z.amax(), ref_amax, atol_amax, rtol_amax);
  }

  auto [atol_stats, rtol_stats] = getTolerances(DType::kFloat32);
  rtol_stats = 5e-5;
  compareResults("rsigma", rsigma, ref_rsigma.get(), atol_stats, rtol_stats);

  auto [atol, rtol] = getTolerances(otype);
  atol = 1e-8;
  compareResults("output", z, ref_output.get(), atol, rtol);

  double atol_bwd = 5e-6;
  double rtol_bwd = 1e-4;
  compareResults("dx", dx, ref_dx.get(), atol_bwd, rtol_bwd);
  compareResults("dgamma", dgamma, ref_dgamma.get(), atol_bwd, rtol_bwd);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
    {2048, 4096}, {768, 2048}, {256, 1024}, {128, 768}, {64, 512}, {173, 409},  // Primes 40, 80
    {71, 3571},                                                                 // Primes 20, 500
    {29, 17389}};                                                               // Primes 10, 2000

}  // namespace

class RMSNormTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                    transformer_engine::DType,
                                                                    std::pair<size_t, size_t>,
                                                                    bool>> {};

TEST_P(RMSNormTestSuite, TestRMSNorm) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());
  const bool zero_centered_gamma = std::get<3>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>(size.first, size.second, zero_centered_gamma);););
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, RMSNormTestSuite,
                         ::testing::Combine(::testing::Values(DType::kFloat32, DType::kBFloat16,
                                                              DType::kFloat16),
                                            ::testing::Values(DType::kFloat32, DType::kBFloat16,
                                                              DType::kFloat16, DType::kFloat8E4M3),
                                            ::testing::ValuesIn(test_cases),
                                            ::testing::Values(false, true)),
                         [](const testing::TestParamInfo<RMSNormTestSuite::ParamType> &info) {
                           std::string name =
                             test::typeName(std::get<0>(info.param)) + "X" +
                             test::typeName(std::get<1>(info.param)) + "X" +
                             std::to_string(std::get<2>(info.param).first) + "X" +
                             std::to_string(std::get<2>(info.param).second) + "X" +
                             std::to_string(std::get<3>(info.param));
                           return name;
                         });
