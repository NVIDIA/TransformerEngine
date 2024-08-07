/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include "./common.h"
#include "./utils.cuh"

namespace transformer_engine {

namespace {

__global__ void __launch_bounds__(1)
    update_tensor_scale_inv_kernel(const float* __restrict__ scale_ptr,
                                   float* __restrict__ scale_inv_ptr) {
  const float scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  reciprocal<float>(scale_inv_ptr, scale);
}

}  // namespace

void update_tensor_scale_inv(Tensor* t, cudaStream_t stream) {
  if (t->scale_inv.dptr != nullptr) {
    update_tensor_scale_inv_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const float*>(t->scale.dptr), reinterpret_cast<float*>(t->scale_inv.dptr));
  }
}

}  // namespace transformer_engine
