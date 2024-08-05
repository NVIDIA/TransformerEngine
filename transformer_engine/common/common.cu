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
namespace update_tensor_scale_inv_impl {

struct Params {
  using Type = float;
  constexpr static size_t max_tensors = 16;
  const Type* scale_ptrs[max_tensors];
  Type* scale_inv_ptrs[max_tensors];
  size_t num_tensors = 0;
};

__global__ void __launch_bounds__(Params::max_tensors)
kernel(Params params) {
  using Type = Params::Type;
  const size_t tid = threadIdx.x;
  if (tid < params.num_tensors) {
    const Type* scale_ptr = params.scale_ptrs[tid];
    const Type scale = scale_ptr == nullptr ? 1 : *scale_ptr;
    reciprocal<float>(params.scale_inv_ptrs[tid], scale);
  }
}

}  // namespace update_tensor_scale_inv_impl
}  // namespace

void update_tensor_scale_inv(Tensor *t, cudaStream_t stream) {
  if (t->scale_inv.dptr == nullptr) {
    return;
  }
  using Params = update_tensor_scale_inv_impl::Params;
  Params params;
  params.scale_ptrs[0] = static_cast<const float*>(t->scale.dptr);
  params.scale_inv_ptrs[0] = static_cast<float*>(t->scale.dptr);
  params.num_tensors = 1;
  update_tensor_scale_inv_impl::kernel<<<1, Params::max_tensors, 0, stream>>>(params);
}

void update_multi_tensor_scale_inv(std::vector<Tensor*> *tensors, cudaStream_t stream) {
  // Add tensors to param struct
  using Params = update_tensor_scale_inv_impl::Params;
  Params params;
  for (auto* t : *tensors) {
    if (t->scale_inv.dptr == nullptr) {
      continue;
    }
    params.scale_ptrs[params.num_tensors] = static_cast<const float*>(t->scale.dptr);
    params.scale_inv_ptrs[params.num_tensors] = static_cast<float*>(t->scale.dptr);
    params.num_tensors++;
    if (params.num_tensors == Params::max_tensors) {
      // Launch kernel if param struct is full
      update_tensor_scale_inv_impl::kernel<<<1, Params::max_tensors, 0, stream>>>(params);
      params.num_tensors = 0;
    }
  }

  // Launch kernel if param struct is not empty
  if (params.num_tensors > 0) {
    update_tensor_scale_inv_impl::kernel<<<1, Params::max_tensors, 0, stream>>>(params);
  }
}

}  // namespace transformer_engine
