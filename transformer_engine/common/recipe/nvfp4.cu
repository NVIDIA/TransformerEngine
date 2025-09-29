/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <cassert>

#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {
namespace nvfp4_recipe {

// constexpr float factor = 6.0 * 6.0 * 448.0 * 448.0;
constexpr float factor_inv = 1.0 / (6.0 * 6.0 * 448.0 * 448.0);

// Kernel to compute alpha *= amax_A * amax_B / factor
__global__ void compute_nvfp4_per_tensor_scale_kernel(float alpha_in, const float *amax_A,
                                                      const float *amax_B, float *alpha_out) {
  // factor is defined in the enclosing namespace
  *alpha_out = alpha_in * (*amax_A) * (*amax_B) * factor_inv;
}

}  // namespace nvfp4_recipe
}  // namespace transformer_engine

void nvte_nvfp4_compute_per_tensor_scale(const NVTETensor inpA, const bool use_rowwise_amax_A,
                                         const NVTETensor inpB, const bool use_rowwise_amax_B,
                                         float alpha_in, NVTETensor alpha_out,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_compute_per_tensor_scale);
  using namespace transformer_engine;

  auto *tA = convertNVTETensor(inpA);
  auto *tB = convertNVTETensor(inpB);
  auto *tOut = convertNVTETensor(alpha_out);

  void *amax_A_ptr = use_rowwise_amax_A ? tA->amax.dptr : tA->columnwise_amax.dptr;
  void *amax_B_ptr = use_rowwise_amax_B ? tB->amax.dptr : tB->columnwise_amax.dptr;
  void *alpha_ptr = tOut->data.dptr;

  // check for not null pointers
  NVTE_CHECK(amax_A_ptr != nullptr, "amax_A_ptr is null");
  NVTE_CHECK(amax_B_ptr != nullptr, "amax_B_ptr is null");
  NVTE_CHECK(alpha_ptr != nullptr, "alpha_ptr is null");

  nvfp4_recipe::compute_nvfp4_per_tensor_scale_kernel<<<1, 1, 0, stream>>>(
      alpha_in, reinterpret_cast<const float *>(amax_A_ptr),
      reinterpret_cast<const float *>(amax_B_ptr), reinterpret_cast<float *>(alpha_ptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}
