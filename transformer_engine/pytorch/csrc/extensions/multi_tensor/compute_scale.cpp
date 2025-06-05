/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

namespace transformer_engine::pytorch {

void multi_tensor_compute_scale_and_scale_inv_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    float max_fp8, bool force_pow_2_scales, float epsilon) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  int device_id = tensor_lists[0][0].device().index();

  nvte_multi_tensor_compute_scale_and_scale_inv_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors, max_fp8,
      force_pow_2_scales, epsilon, device_id, at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
