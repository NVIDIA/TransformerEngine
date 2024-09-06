/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../utils.cuh"
#include "norms.h"

namespace transformer_engine {

__global__ void reciprocalKernel(float* value_inv, const float* value) {
  reciprocal(value_inv, *value);
}
void ComputeScaleInv(Tensor* z) {
  NVTE_CHECK(z->amax.dptr != nullptr, "FP8 output must have amax tensor.");
  NVTE_CHECK(z->amax.dtype == DType::kFloat32);
  NVTE_CHECK(z->amax.shape == std::vector<size_t>{1});
  NVTE_CHECK(z->scale_inv.dptr == nullptr, "FP8 output scale_inv should be empty.");
  NVTE_CHECK(z->scale_inv.dtype == DType::kFloat32);
  NVTE_CHECK(z->scale_inv.shape == std::vector<size_t>{1});
  reciprocalKernel<<<1, 1>>>(reinterpret_cast<float*>(z->scale_inv.dptr),
                             reinterpret_cast<float*>(z->amax.dptr));
}
void ComputeScaleInv(void* scale_inv, void* scale) {
  NVTE_CHECK(scale != nullptr, "amax should be allocated.");
  NVTE_CHECK(scale_inv != nullptr, "scale_inv should be allocated.");
  reciprocalKernel<<<1, 1>>>(reinterpret_cast<float*>(scale_inv), reinterpret_cast<float*>(scale));
}

}  // namespace transformer_engine
