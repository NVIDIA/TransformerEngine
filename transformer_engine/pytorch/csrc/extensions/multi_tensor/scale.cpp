/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, float scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_scale_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                               num_tensors, scale, at::cuda::getCurrentCUDAStream());
}

void multi_tensor_scale_tensor_cuda(int chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor scale) {
  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto scale_cu = makeTransformerEngineTensor(scale);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  std::cout << "multi_tensor_scale_cuda TENSOR\n";
  nvte_multi_tensor_scale_tensor_cuda(chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists,
                                      num_tensors, scale_cu.data(), at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
