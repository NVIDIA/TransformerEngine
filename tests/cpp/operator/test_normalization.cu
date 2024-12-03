/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdlib.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum NormType {
  LayerNorm,
  RMSNorm
};

std::map<NormType, std::string> normToString = {
  {NormType::LayerNorm, "LayerNorm"},
  {NormType::RMSNorm, "RmsNorm"}
};

template <typename InputType>
void compute_ref_stats(NormType norm_type,
                       const InputType *data, float *mu, float *rsigma,
                       const size_t N, const size_t H, const double epsilon){
  using compute_t = float;
  compute_t current, m;
  for (size_t i = 0; i < N; ++i) {
    compute_t sum = 0;
    for (size_t j = 0; j < H; ++j) {
      sum += static_cast<compute_t>(data[i * H + j]);
    }
    if (norm_type == LayerNorm){
      mu[i] = sum / H;
      m = mu[i];
    } else { m = 0;}

    compute_t sum_sq = 0;
    for (size_t j = 0; j < H; ++j) {
      current = static_cast<compute_t>(data[i * H + j]);
      sum_sq += (current - m) * (current - m);
    }
    rsigma[i] = rsqrtf((sum_sq / H) + epsilon);
  }
}

// For now, cudnn does static_cast<compute_t>(gamma + static_cast<input_t>(1.0))
// This will be changed in the future release
const static bool use_cudnn_fwd = std::getenv("NVTE_FWD_NORM_USE_CUDNN");
const static bool use_cudnn_bwd = std::getenv("NVTE_BWD_NORM_USE_CUDNN");

template <typename InputType>
inline auto compute_gamma(InputType gamma, const bool zero_centered_gamma, const bool use_cudnn){

  using compute_t = float;
  if constexpr (std::is_same_v<InputType, fp8e5m2> || std::is_same_v<InputType, fp8e4m3>){
    compute_t g = static_cast<compute_t>(gamma);
    if (zero_centered_gamma) {
      g += static_cast<compute_t>(1.f);
    }
    return g;
  } else {
    if (use_cudnn){
      compute_t g = static_cast<compute_t>(0.f);
      InputType gi = gamma;
      if (zero_centered_gamma) {
        gi = gi + static_cast<InputType>(1.f);
      }
      g = static_cast<compute_t>(gi);
      return g;
    } else {
      compute_t g = static_cast<compute_t>(gamma);
      if (zero_centered_gamma) {
        g += static_cast<compute_t>(1.f);
      }
      return g;
    }
  }
}

template <typename InputType, typename OutputType>
void compute_ref_output(NormType norm_type,
                        const InputType *data, const InputType *gamma, const InputType *beta,
                        OutputType* output,
                        const float *mu, const float *rsigma,
                        const size_t N, const size_t H,
                        float *amax, float scale, const bool zero_centered_gamma) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      compute_t g = compute_gamma(gamma[j], zero_centered_gamma, use_cudnn_fwd);

      compute_t tmp;
      if (norm_type == LayerNorm) {
        tmp = (current - mu[i]) * rsigma[i] * g + static_cast<compute_t>(beta[j]);
      } else { // RMSNorm
        tmp = current * rsigma[i] * g;
      }

      output[i * H + j] = static_cast<OutputType>(tmp * scale);
      current_max = fmaxf(current_max, fabsf(tmp));
    }
  }
  *amax = current_max;
}


template <typename InputType, typename OutputType>
void compute_ref_backward(const NormType norm_type, const OutputType *output_grad, const InputType *data,
                          const float *mu, const float *rsigma,
                          const InputType *gamma,
                          InputType *data_grad,
                          InputType *gamma_grad, InputType *beta_grad,
                          const size_t N, const size_t H,
                          const bool zero_centered_gamma) {
  using compute_t = float;
  std::vector<compute_t> dgamma(H, 0.f);
  std::vector<compute_t> dbeta(H, 0.f);

  for (size_t i = 0 ; i < N; ++i) {
    // Reductions
    auto local_mu = (norm_type == LayerNorm) ? mu[i] : 0.;
    compute_t mdy = 0, mdyy = 0;
    for (size_t j = 0; j < H; ++j) {
      const compute_t x = static_cast<compute_t>(data[i * H + j]);
      const compute_t y = (x - local_mu) * rsigma[i];
      compute_t g = compute_gamma(gamma[j], zero_centered_gamma, use_cudnn_bwd);
      const compute_t dz = static_cast<compute_t>(output_grad[i * H + j]);
      const compute_t dy = g * dz;
      dgamma[j] += y * dz;
      if (norm_type == LayerNorm) {
        dbeta[j] += dz;
        mdy += dy;
      }
      mdyy += dy * y;
    }
    mdy /= H;
    mdyy /= H;

    // Input grads
    for (size_t j = 0; j < H; ++j) {
      const compute_t x = static_cast<compute_t>(data[i * H + j]);
      const compute_t y = (x - local_mu) * rsigma[i];
      compute_t g = compute_gamma(gamma[j], zero_centered_gamma, use_cudnn_bwd);
      const compute_t dz = static_cast<compute_t>(output_grad[i * H + j]);
      const compute_t dy = g * dz;
      const compute_t dx = rsigma[i] * (dy - mdyy * y - mdy);
      data_grad[i * H + j] = static_cast<InputType>(dx);
    }
  }

  // Weight grads
  for (size_t j = 0; j < H; ++j) gamma_grad[j] = static_cast<InputType>(dgamma[j]);
  if (norm_type == LayerNorm) for (size_t j = 0; j < H; ++j) beta_grad[j] = static_cast<InputType>(dbeta[j]);
}

template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H, const bool zero_centered_gamma,
                 NormType norm_type, bool use_cudnn) {
  if (sizeof(InputType) < sizeof(OutputType)) {
    GTEST_SKIP() << "LN kernel does not support OutputType > InputType";
    return;
  }
  using WeightType = InputType;
  DType itype = TypeInfo<InputType>::dtype;
  DType wtype = TypeInfo<WeightType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  if ((itype == DType::kBFloat16 && otype == DType::kFloat16) ||
      (itype == DType::kFloat16 && otype == DType::kBFloat16)) {
    GTEST_SKIP() << "LN kernel does not support mixing Float16 and BFloat16";
    return;
  }

  Tensor input({ N, H }, itype);
  Tensor z({ N, H }, otype);
  Tensor gamma({ H }, wtype);
  Tensor beta({ H }, wtype);
  Tensor mu({ N }, DType::kFloat32);
  Tensor rsigma({ N }, DType::kFloat32);
  Tensor dz({ N, H }, wtype);
  Tensor dx({ N, H }, itype);
  Tensor dgamma({ H }, wtype);
  Tensor dbeta({ H }, wtype);
  Tensor workspace_fwd, workspace_bwd;

  fillUniform(&input);
  fillUniform(&gamma);
  fillUniform(&beta);
  setRandomScale(&z);
  fillUniform(&dz);

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(N * H);
  std::unique_ptr<float[]> ref_mu = std::make_unique<float[]>(N);
  std::unique_ptr<float[]> ref_rsigma = std::make_unique<float[]>(N);
  std::unique_ptr<InputType[]> ref_dx = std::make_unique<InputType[]>(N * H);
  std::unique_ptr<WeightType[]> ref_dgamma = std::make_unique<InputType[]>(H);
  std::unique_ptr<WeightType[]> ref_dbeta = std::make_unique<InputType[]>(H);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  if (use_cudnn){
    setenv("NVTE_NORM_FWD_USE_CUDNN", "1", true /*overwrite*/);
    setenv("NVTE_NORM_BWD_USE_CUDNN", "1", true /*overwrite*/);
  }

  // Forward kernel
  float epsilon = 1e-5;
  if (norm_type == LayerNorm){
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace_fwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_fwd = Tensor(workspace_fwd.shape(), workspace_fwd.dtype());
    nvte_layernorm_fwd(input.data(), gamma.data(), beta.data(), epsilon,
                       z.data(), mu.data(), rsigma.data(), workspace_fwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);

    nvte_layernorm_bwd(dz.data(), input.data(),
                       mu.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), dbeta.data(),
                       workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_bwd = Tensor(workspace_bwd.shape(), workspace_bwd.dtype());
    nvte_layernorm_bwd(dz.data(), input.data(),
                       mu.data(), rsigma.data(), gamma.data(),
                       dx.data(), dgamma.data(), dbeta.data(),
                       workspace_bwd.data(),
                       prop.multiProcessorCount, zero_centered_gamma, 0);
  } else {
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_fwd = Tensor(workspace_fwd.shape(), workspace_fwd.dtype());
    nvte_rmsnorm_fwd(input.data(), gamma.data(), epsilon,
                     z.data(), rsigma.data(), workspace_fwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, 0);

    nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                     dx.data(), dgamma.data(),
                     workspace_bwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, 0);
    workspace_bwd = Tensor(workspace_bwd.shape(), workspace_bwd.dtype());
    nvte_rmsnorm_bwd(dz.data(), input.data(), rsigma.data(), gamma.data(),
                     dx.data(), dgamma.data(),
                     workspace_bwd.data(),
                     prop.multiProcessorCount, zero_centered_gamma, 0);
  }

  if (use_cudnn){
    unsetenv("NVTE_NORM_FWD_USE_CUDNN");
    unsetenv("NVTE_NORM_BWD_USE_CUDNN");
  }

  // Reference implementations
  // use the GPU stats to tighten the tolerances
  mu.to_cpu();
  rsigma.to_cpu();
  float ref_amax;
  compute_ref_stats(norm_type, input.cpu_dptr<InputType>(), ref_mu.get(),
                    ref_rsigma.get(), N, H, epsilon);
  float ref_scale = isFp8Type(otype) ? z.scale() : 1.f;
  compute_ref_output(norm_type, input.cpu_dptr<InputType>(),
                     gamma.cpu_dptr<WeightType>(),
                     beta.cpu_dptr<WeightType>(),
                     ref_output.get(),
                     mu.cpu_dptr<float>(),
                     rsigma.cpu_dptr<float>(),
                     N, H,
                     &ref_amax,
                     ref_scale,
                     zero_centered_gamma);
  compute_ref_backward(norm_type, dz.cpu_dptr<WeightType>(), input.cpu_dptr<InputType>(),
                       mu.cpu_dptr<float>(), rsigma.cpu_dptr<float>(),
                       gamma.cpu_dptr<WeightType>(),
                       ref_dx.get(), ref_dgamma.get(), ref_dbeta.get(),
                       N, H, zero_centered_gamma);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  if (isFp8Type(otype)) {
    compareResults("amax", z.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / z.scale();
    compareResults("scale_inv", z.scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }

  auto [atol_stats, rtol_stats] = getTolerances(DType::kFloat32);
  rtol_stats = 5e-5;
  compareResults("mu", mu, ref_mu.get(), atol_stats, rtol_stats);
  compareResults("rsigma", rsigma, ref_rsigma.get(), atol_stats, rtol_stats);

  auto [atol, rtol] = getTolerances(otype);
  if (otype == DType::kFloat32) {
    atol = 5e-7;
  }
  compareResults("output", z, ref_output.get(), atol, rtol);

  double atol_bwd = 5e-4;
  double rtol_bwd = 5e-4;
  compareResults("dx", dx, ref_dx.get(), atol_bwd, rtol_bwd);
  compareResults("dgamma", dgamma, ref_dgamma.get(), atol_bwd, rtol_bwd);
  compareResults("dbeta", dbeta, ref_dbeta.get(), atol_bwd, rtol_bwd);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
  {71, 229},
  {29, 541},
  {768, 6144},
  {2048, 12288},
};

}  // namespace

class NormTestSuite : public ::testing::TestWithParam<std::tuple<bool,
NormType,
transformer_engine::DType,
                                                               transformer_engine::DType,
                                                               std::pair<size_t, size_t>,
                                                               bool>> {};

TEST_P(NormTestSuite, TestNorm) {
    using namespace transformer_engine;
    using namespace test;

  const bool use_cudnn = std::get<0>(GetParam());
  const NormType norm_type = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const DType output_type = std::get<3>(GetParam());
    const auto size = std::get<4>(GetParam());
    const bool zero_centered_gamma = std::get<5>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<InputType, OutputType>(size.first, size.second, zero_centered_gamma, norm_type, use_cudnn);
      );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    NormTestSuite,
    ::testing::Combine(
        ::testing::Values(false), //TODO: enabling tests for cudnn backend
        ::testing::Values(NormType::LayerNorm, NormType::RMSNorm),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16, DType::kFloat8E4M3),
        ::testing::ValuesIn(test_cases),
        ::testing::Values(false, true)),
    [](const testing::TestParamInfo<NormTestSuite::ParamType>& info) {
    auto backend = std::get<0>(info.param) == false ? "Te" : "Cudnn";
std::string name =
  backend +
  normToString.at(std::get<1>(info.param)) + "_" +
  test::typeName(std::get<2>(info.param)) + "X" +
  test::typeName(std::get<3>(info.param)) + "X" +
  std::to_string(std::get<4>(info.param).first) + "X" +
  std::to_string(std::get<4>(info.param).second) + "X" +
  std::to_string(std::get<5>(info.param));
      return name;
    });
