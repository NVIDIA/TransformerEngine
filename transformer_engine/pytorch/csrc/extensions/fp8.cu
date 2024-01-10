/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

#include <cmath>
#include <string>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "common/util/logging.h"

namespace {
namespace fused_amax_and_scale_update {

// CUDA block size
constexpr size_t bsize = 256;

// amax value to use for updating scaling factor
enum class AmaxComputeAlgo { INVALID, MOST_RECENT, MAX };

/* CUDA kernel to update amax history and FP8 scaling factors
 *
 * Block dims: bsize x 1 x 1
 *
 * Grid dims: num_scales x 1 x 1
 */
__global__ void __launch_bounds__(bsize)
kernel(const float* amax_history_ptr,
       const float* scale_ptr,
       const float* scale_inv_ptr,
       const bool* non_weight_mask_ptr,
       float* updated_amax_history_ptr,
       float* updated_scale_ptr,
       float* updated_scale_inv_ptr,
       size_t amax_history_length,
       size_t amax_history_stride,
       AmaxComputeAlgo amax_compute_algo,
       float scaled_max,
       bool update_weight_scale_inv) {
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;

  // Update amax
  float amax = 0.f;
  {
    // Roll amax history
    const auto* amax_history = amax_history_ptr + bid;
    auto* updated_amax_history = updated_amax_history_ptr + bid;
    const auto& last_amax = amax_history[0];
    const auto& length = amax_history_length;
    const auto& stride = amax_history_stride;
    for (size_t i=tid; i<length; i+=bsize) {
      const auto& a = (i < length - 1) ? amax_history[(i+1)*stride] : last_amax;
      updated_amax_history[i*stride] = (i > 0) ? a : 0.f;
      amax = fmaxf(amax, a);
    }

    // Compute amax to use for scaling factor
    switch(amax_compute_algo) {
    case AmaxComputeAlgo::MOST_RECENT:
      amax = last_amax;
      break;
    case AmaxComputeAlgo::MAX:
      {
        __shared__ float shared_amax[bsize];
        shared_amax[tid] = amax;
        __syncthreads();
#pragma unroll
        for (size_t off = bsize / 2; off > 0; off /= 2) {
          if (tid < off) {
            shared_amax[tid] = fmaxf(shared_amax[tid], shared_amax[tid + off]);
          }
          __syncthreads();
        }
        amax = shared_amax[tid];
      }
      break;
    default:
      amax = 0.f;
    }
  }

  // Update scale and scale inverse
  if (tid == 0) {
    // Update scale
    float scale;
    if (isfinite(amax) && amax > 0) {
      scale = scaled_max / amax;
    } else {
      scale = scale_ptr[bid];
    }
    updated_scale_ptr[bid] = scale;

    // Update scale inverse
    float scale_inv;
    if (update_weight_scale_inv || non_weight_mask_ptr[bid]) {
      scale_inv = 1 / scale;
    } else {
      scale_inv = scale_inv_ptr[bid];
    }
    updated_scale_inv_ptr[bid] = scale_inv;
  }
}

}  // namespace fused_amax_and_scale_update_kernel
}  // namespace

void fused_amax_and_scale_update(const at::Tensor& amax_history,
                                 const at::Tensor& scale,
                                 const at::Tensor& scale_inv,
                                 const at::Tensor& non_weight_mask,
                                 at::Tensor updated_amax_history,
                                 at::Tensor updated_scale,
                                 at::Tensor updated_scale_inv,
                                 const std::string& amax_compute_algo,
                                 float fp8_max,
                                 float margin,
                                 bool update_weight_scale_inv) {
  // Check tensors
  NVTE_CHECK(amax_history.dim() == 2,
             "Expected amax history to have 2 dims, but found ",
             amax_history.dim(), ".");
  const size_t amax_history_length = amax_history.size(0);
  const size_t num_scales = amax_history.size(1);
  NVTE_CHECK(amax_history.scalar_type() == at::kFloat);
  NVTE_CHECK(amax_history.is_cuda());
  NVTE_CHECK(amax_history.is_contiguous());
  NVTE_CHECK(scale.numel() == num_scales,
             "Expected to update ", num_scales, " scales, ",
             "but found ", scale.numel(), ".");
  NVTE_CHECK(scale.scalar_type() == at::kFloat);
  NVTE_CHECK(scale.is_cuda());
  NVTE_CHECK(scale.is_contiguous());
  if (!update_weight_scale_inv) {
    NVTE_CHECK(scale_inv.numel() == num_scales,
               "Expected to update ", num_scales, " scale inverses, ",
               "but found ", scale_inv.numel(), ".");
    NVTE_CHECK(scale_inv.scalar_type() == at::kFloat);
    NVTE_CHECK(scale_inv.is_cuda());
    NVTE_CHECK(scale_inv.is_contiguous());
    NVTE_CHECK(non_weight_mask.numel() == num_scales,
               "Expected to update ", num_scales, " non-weight masks, ",
               "but found ", non_weight_mask.numel(), ".");
    NVTE_CHECK(non_weight_mask.scalar_type() == at::kBool);
    NVTE_CHECK(non_weight_mask.is_cuda());
    NVTE_CHECK(non_weight_mask.is_contiguous());
  }
  NVTE_CHECK(updated_amax_history.dim() == 2,
             "Expected updated amax history to have 2 dims, but found ",
             updated_amax_history.dim(), ".");
  NVTE_CHECK(updated_amax_history.size(0) == amax_history_length,
             "Expected updated amax history to have length ",
             amax_history_length, ", but found ",
             updated_amax_history.size(0), ".");
  NVTE_CHECK(updated_amax_history.size(1) == num_scales,
             "Expected to update ", num_scales, " updated amax histories, ",
             "but found ", updated_amax_history.size(1), ".");
  NVTE_CHECK(updated_amax_history.scalar_type() == at::kFloat);
  NVTE_CHECK(updated_amax_history.is_cuda());
  NVTE_CHECK(updated_amax_history.is_contiguous());
  NVTE_CHECK(updated_scale.numel() == num_scales,
             "Expected to update ", num_scales, " updated scales, ",
             "but found ", updated_scale.numel(), ".");
  NVTE_CHECK(updated_scale.scalar_type() == at::kFloat);
  NVTE_CHECK(updated_scale.is_cuda());
  NVTE_CHECK(updated_scale.is_contiguous());
  NVTE_CHECK(updated_scale_inv.numel() == num_scales,
             "Expected to update ", num_scales, " updated scale inverses, ",
             "but found ", updated_scale_inv.numel(), ".");
  NVTE_CHECK(updated_scale_inv.scalar_type() == at::kFloat);
  NVTE_CHECK(updated_scale_inv.is_cuda());
  NVTE_CHECK(updated_scale_inv.is_contiguous());

  // amax value to use for updating scaling factor
  using AmaxComputeAlgo = fused_amax_and_scale_update::AmaxComputeAlgo;
  AmaxComputeAlgo amax_compute_algo_ = AmaxComputeAlgo::INVALID;
  if (amax_compute_algo == "most_recent") {
    amax_compute_algo_ = AmaxComputeAlgo::MOST_RECENT;
  } else if (amax_compute_algo == "max") {
    amax_compute_algo_ = AmaxComputeAlgo::MAX;
  } else {
    NVTE_ERROR("Unsupported amax compute algorithm (", amax_compute_algo, ")");
  }

  // Expected maximum value after scale is applied
  const float scaled_max = fp8_max / std::pow(2.f, margin);

  // Launch CUDA kernel
  constexpr size_t block_size = fused_amax_and_scale_update::bsize;
  const size_t grid_size = num_scales;
  fused_amax_and_scale_update::kernel
    <<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
      static_cast<const float*>(amax_history.data_ptr()),
      static_cast<const float*>(scale.data_ptr()),
      static_cast<const float*>(scale_inv.data_ptr()),
      static_cast<const bool*>(non_weight_mask.data_ptr()),
      static_cast<float*>(updated_amax_history.data_ptr()),
      static_cast<float*>(updated_scale.data_ptr()),
      static_cast<float*>(updated_scale_inv.data_ptr()),
      amax_history_length,
      num_scales,
      amax_compute_algo_,
      scaled_max,
      update_weight_scale_inv);
}
