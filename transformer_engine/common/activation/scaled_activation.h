/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* Scaled activations apply an activation, multiply by a per-row scale
 * (act_scales[row]), do all math in fp32, and cast once at the store. The
 * backward path optionally also reduces the gradient of the per-row scale.
 *
 * Public launch APIs are templated on ParamOP / ActOP / DActOP (same shape as
 * gated_act_fn). Kernel definitions and explicit instantiations live in
 * scaled_activation.cu.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_ACTIVATION_SCALED_ACTIVATION_H_
#define TRANSFORMER_ENGINE_COMMON_ACTIVATION_SCALED_ACTIVATION_H_

#include <transformer_engine/activation.h>

#include "../common.h"
#include "../util/math.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"

namespace transformer_engine {
namespace detail {
namespace scaled_activation {

constexpr int kThreads = unary_kernel_threads;

// Launch bounds for the kThreads-wide kernels. A minimum of 4 resident 512-thread blocks needs
// 2048 threads per SM, which SM 7.5 (1024) and SM 8.6/8.7/8.9/12.x (1536) cannot hold, so ptxas
// ignores the minimum there with a warning. Leave it out on those targets; the generated code is
// the same.
#if defined(__CUDA_ARCH__) &&                                                \
    (__CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || \
     __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
#define NVTE_SCALED_ACTIVATION_LAUNCH_BOUNDS __launch_bounds__(kThreads)
#else
#define NVTE_SCALED_ACTIVATION_LAUNCH_BOUNDS __launch_bounds__(kThreads, 4)
#endif

constexpr int kReductionThreads = 256;
constexpr int kReductionWarps = kReductionThreads / THREADS_PER_WARP;

// Pick a CTA size for one-block-per-row scale-grad: enough threads to cover
// row_vectors, rounded up to a warp multiple, capped at kReductionThreads.
inline int choose_reduction_threads(const size_t row_vectors) {
  if (row_vectors >= static_cast<size_t>(kReductionThreads)) {
    return kReductionThreads;
  }
  const int needed = static_cast<int>(row_vectors);
  int rounded = (needed + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
  if (rounded < static_cast<int>(THREADS_PER_WARP)) {
    rounded = THREADS_PER_WARP;
  }
  return rounded;
}

template <typename... Ptrs>
Alignment row_vector_alignment(const size_t lead_dim, const int nvec, const Ptrs... ptrs) {
  if (nvec == 1) {
    return Alignment::SAME_ALIGNED;
  }
  if (lead_dim % static_cast<size_t>(nvec) != 0) {
    return Alignment::DIFFERENT;
  }
  const auto align = CheckAlignment(lead_dim, nvec, ptrs...);
  return align == Alignment::SAME_ALIGNED ? Alignment::SAME_ALIGNED : Alignment::DIFFERENT;
}

}  // namespace scaled_activation
}  // namespace detail

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void launch_scaled_gated_forward(const NVTETensor input, const NVTETensor act_scales,
                                 NVTETensor output, ParamOP param, int64_t glu_interleave_size,
                                 cudaStream_t stream, const char *api_name);

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void launch_scaled_gated_backward(const NVTETensor grad, const NVTETensor input,
                                  const NVTETensor act_scales, NVTETensor grad_input,
                                  NVTETensor grad_act_scales, ParamOP param,
                                  int64_t glu_interleave_size, cudaStream_t stream,
                                  const char *api_name);

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void launch_scaled_unary_forward(const NVTETensor input, const NVTETensor act_scales,
                                 NVTETensor output, ParamOP param, cudaStream_t stream,
                                 const char *api_name);

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void launch_scaled_unary_backward(const NVTETensor grad, const NVTETensor input,
                                  const NVTETensor act_scales, NVTETensor grad_input,
                                  NVTETensor grad_act_scales, ParamOP param, cudaStream_t stream,
                                  const char *api_name);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_ACTIVATION_SCALED_ACTIVATION_H_
