/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/padding.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

Tensor splits_to_offsets(Tensor first_dims, int64_t logical_last_dim) {
  STD_TORCH_CHECK(first_dims.is_cuda(), "first_dims must be on CUDA.");
  STD_TORCH_CHECK(first_dims.scalar_type() == ScalarType::Long,
                  "first_dims must have dtype int64.");
  STD_TORCH_CHECK(first_dims.dim() == 1, "first_dims must be a 1D tensor.");
  STD_TORCH_CHECK(logical_last_dim > 0, "logical_last_dim must be greater than 0.");

  auto first_dims_c = torch::stable::contiguous(first_dims);
  const auto num_tensors = static_cast<size_t>(first_dims_c.numel());
  auto output = allocateStableTensor({static_cast<int64_t>(num_tensors) + 1}, ScalarType::Long,
                                     first_dims_c.get_device_index());

  nvte_splits_to_offsets(static_cast<const int64_t*>(first_dims_c.data_ptr()),
                         static_cast<int64_t*>(output.data_ptr()), num_tensors, logical_last_dim,
                         getCurrentCUDAStreamRaw(first_dims_c.get_device_index()));

  return output;
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  m.impl("splits_to_offsets", TORCH_BOX(splits_to_offsets));
}

}  // namespace transformer_engine::pytorch::stable
