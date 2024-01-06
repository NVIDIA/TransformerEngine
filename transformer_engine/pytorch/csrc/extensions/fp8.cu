/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "common/util/logging.h"

namespace {

__global__ void
__launch_bounds__(32)
fused_scale_update_kernel(size_t num_scales,
                          const float * const amax_ptr,
                          const float * const old_scale_ptr,
                          const float * const old_scale_inv_ptr,
                          const bool * const non_weight_mask_ptr,
                          float * const scale_ptr,
                          float * const scale_inv_ptr,
                          const float scaled_max,
                          const bool update_weight_scale_inv) {
  const size_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  const size_t nthreads = blockDim.x * gridDim.x;
  for (size_t i = gid; i < num_scales; i += nthreads) {
    // Update scale
    float scale;
    const float amax = amax_ptr[i];
    if (isfinite(amax) && amax >= 0) {
      scale = scaled_max / amax;
    } else {
      scale = old_scale_ptr[i];
    }
    scale_ptr[i] = scale;

    // Update scale inverse
    float scale_inv;
    if (update_weight_scale_inv || non_weight_mask_ptr[i]) {
      scale_inv = 1 / scale;
    } else {
      scale_inv = old_scale_inv_ptr[i];
    }
    scale_inv_ptr[i] = scale_inv;
  }
}

}  // namespace

void fused_scale_update(const at::Tensor& amax,
                        const at::Tensor& old_scale,
                        const at::Tensor& old_scale_inv,
                        const at::Tensor& non_weight_mask,
                        at::Tensor scale,
                        at::Tensor scale_inv,
                        float fp8_max,
                        float margin,
                        bool update_weight_scale_inv) {
  // Check tensors
  const size_t num_scales = scale.numel();
  NVTE_CHECK(amax.numel() == num_scales,
             "Trying to update ", num_scales, " scales with ",
             amax.numel(), " amaxes");
  NVTE_CHECK(amax.scalar_type() == at::kFloat);
  NVTE_CHECK(amax.is_cuda());
  NVTE_CHECK(old_scale.numel() == num_scales,
             "Trying to update ", num_scales, " scales with ",
             old_scale.numel(), " old scales");
  NVTE_CHECK(old_scale.scalar_type() == at::kFloat);
  NVTE_CHECK(old_scale.is_cuda());
  if (!update_weight_scale_inv) {
    NVTE_CHECK(old_scale_inv.numel() == num_scales,
               "Trying to update ", num_scales, " scales with ",
               old_scale_inv.numel(), " old scale inverses");
    NVTE_CHECK(old_scale_inv.scalar_type() == at::kFloat);
    NVTE_CHECK(old_scale_inv.is_cuda());
    NVTE_CHECK(non_weight_mask.numel() == num_scales,
               "Trying to update ", num_scales, " scales with ",
               non_weight_mask.numel(), " non-weight masks");
    NVTE_CHECK(non_weight_mask.scalar_type() == at::kBool);
    NVTE_CHECK(non_weight_mask.is_cuda());
  }
  NVTE_CHECK(scale.scalar_type() == at::kFloat);
  NVTE_CHECK(scale.is_cuda());
  NVTE_CHECK(scale_inv.numel() == num_scales,
             "Trying to update ", num_scales, " scales with ",
             scale_inv.numel(), " scale inverses");
  NVTE_CHECK(scale_inv.scalar_type() == at::kFloat);
  NVTE_CHECK(scale_inv.is_cuda());

  // Expected maximum value after scale is applied
  const float scaled_max = fp8_max / std::pow(2.f, margin);

  // Launch CUDA kernel
  constexpr size_t block_size = 32;
  const size_t grid_size = (num_scales + block_size - 1) / block_size;
  fused_scale_update_kernel
    <<<grid_size, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
      num_scales,
      static_cast<const float*>(amax.data_ptr()),
      static_cast<const float*>(old_scale.data_ptr()),
      static_cast<const float*>(old_scale_inv.data_ptr()),
      static_cast<const bool*>(non_weight_mask.data_ptr()),
      static_cast<float*>(scale.data_ptr()),
      static_cast<float*>(scale_inv.data_ptr()),
      scaled_max,
      update_weight_scale_inv);
}
