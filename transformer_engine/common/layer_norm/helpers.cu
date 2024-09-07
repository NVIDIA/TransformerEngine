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
void ComputeScaleInv(void* scale, void* scale_inv) {
  NVTE_CHECK(scale != nullptr, "amax should be allocated.");
  NVTE_CHECK(scale_inv != nullptr, "scale_inv should be allocated.");
  reciprocalKernel<<<1, 1>>>(reinterpret_cast<float*>(scale_inv), reinterpret_cast<float*>(scale));
}

}  // namespace transformer_engine
