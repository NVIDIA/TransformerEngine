/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* Scaled activations apply an activation, multiply by a per-row scale
 * (act_scales[row]), do all math in fp32, and cast once at the store. The
 * backward path optionally also reduces the gradient of the per-row scale.
 *
 * The implementation consists of six __global__ kernels:
 *
 *   # | Kernel                                        | Activation             | Dir | grad_act_scales | Launch
 *  ---+-----------------------------------------------+------------------------+-----+-----------------+--------------------
 *   1 | scaled_gated_forward_kernel                   | SwiGLU / ClampedSwiGLU | fwd | --              | vectorized row segments
 *   2 | scaled_srelu_forward_kernel                   | SReLU (unary)          | fwd | --              | vectorized row grid
 *   3 | scaled_gated_backward_kernel                  | SwiGLU / ClampedSwiGLU | bwd | no              | vectorized row segments
 *   4 | scaled_srelu_backward_kernel                  | SReLU                  | bwd | no              | vectorized row grid
 *   5 | scaled_gated_backward_with_scale_grad_kernel  | SwiGLU / ClampedSwiGLU | bwd | yes             | vectorized, one block per row
 *   6 | scaled_srelu_backward_with_scale_grad_kernel  | SReLU                  | bwd | yes             | vectorized, one block per row
 *
 * The "with scale grad" variants compute
 * grad_act_scales[row] = sum_j dY * unscaled. This per-row reduction requires
 * the one-block-per-row launch; when grad_act_scales is null, the cheaper flat
 * elementwise grid is used instead.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_ACTIVATION_SCALED_ACTIVATION_H_
#define TRANSFORMER_ENGINE_COMMON_ACTIVATION_SCALED_ACTIVATION_H_

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"

namespace transformer_engine {
namespace detail {
namespace scaled_activation {

constexpr int kThreads = unary_kernel_threads;
constexpr int kReductionThreads = 256;
constexpr int kReductionWarps = kReductionThreads / THREADS_PER_WARP;
using WarpReducer = Reducer<float, 1, 1, 1>;

__device__ __forceinline__ float block_reduce_sum(float value, float *smem) {
  const int lane = threadIdx.x % THREADS_PER_WARP;
  const int warp = threadIdx.x / THREADS_PER_WARP;
  Empty params = {};
  WarpReducer reducer(params, /*bidm=*/0, /*bidn=*/0, /*warp_m=*/0, /*warp_n=*/0, lane,
                      /*smem=*/nullptr);
  Sum<float> sum;

  value = reducer.reduce(value, sum);
  if (lane == 0) {
    smem[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kReductionWarps ? smem[lane] : 0.0f;
  return warp == 0 ? reducer.reduce(value, sum) : value;
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
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_ACTIVATION_SCALED_ACTIVATION_H_
