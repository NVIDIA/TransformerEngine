/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

 #pragma once

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

namespace test {
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

template <typename InputType>
inline auto compute_gamma(InputType gamma, const bool zero_centered_gamma, const bool use_cudnn, const bool cudnn_zero_centered_gamma_in_weight_dtype) {

  using compute_t = float;

  // Zero-centered gamma in weight dtype is only supported in CuDNN backend currently
  // Remove the use_cudnn check here when it is supported by both backends.
  const bool zero_centered_gamma_in_weight_dtype = use_cudnn && cudnn_zero_centered_gamma_in_weight_dtype;

  if constexpr (std::is_same_v<InputType, fp8e5m2> || std::is_same_v<InputType, fp8e4m3> ||
                std::is_same_v<InputType, fp4e2m1>){
    compute_t g = static_cast<compute_t>(gamma);
    if (zero_centered_gamma) {
      g += static_cast<compute_t>(1.f);
    }
    return g;
  } else {
    if (zero_centered_gamma_in_weight_dtype){
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
                        float *amax, float scale, const bool zero_centered_gamma, const bool use_cudnn, const bool cudnn_zero_centered_gamma_in_weight_dtype) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      compute_t g = compute_gamma(gamma[j], zero_centered_gamma, use_cudnn, cudnn_zero_centered_gamma_in_weight_dtype);

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

  if (amax) {
    *amax = current_max;
  }
}


template <typename InputType, typename OutputType>
void compute_ref_backward(const NormType norm_type, const OutputType *output_grad,
                          const OutputType *add, const InputType *data,
                          const float *mu, const float *rsigma,
                          const InputType *gamma,
                          InputType *data_grad,
                          InputType *gamma_grad, InputType *beta_grad,
                          const size_t N, const size_t H,
                          const bool zero_centered_gamma, const bool use_cudnn,
                          const bool cudnn_zero_centered_gamma_in_weight_dtype) {
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
      compute_t g = compute_gamma(gamma[j], zero_centered_gamma, use_cudnn, cudnn_zero_centered_gamma_in_weight_dtype);
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
      compute_t g = compute_gamma(gamma[j], zero_centered_gamma, use_cudnn, cudnn_zero_centered_gamma_in_weight_dtype);
      const compute_t dz = static_cast<compute_t>(output_grad[i * H + j]);
      const compute_t dy = g * dz;
      const compute_t a = static_cast<compute_t>(add[i * H + j]);
      const compute_t dx = a + rsigma[i] * (dy - mdyy * y - mdy);
      data_grad[i * H + j] = static_cast<InputType>(dx);
    }
  }

  // Weight grads
  for (size_t j = 0; j < H; ++j) gamma_grad[j] = static_cast<InputType>(dgamma[j]);
  if (norm_type == LayerNorm) for (size_t j = 0; j < H; ++j) beta_grad[j] = static_cast<InputType>(dbeta[j]);
}

} // namespace
} // namespace test
