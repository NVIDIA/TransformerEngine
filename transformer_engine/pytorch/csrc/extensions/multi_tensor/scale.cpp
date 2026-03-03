/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

void multi_tensor_scale_cuda(int chunk_size, at::Tensor is_infinite,
                             std::vector<std::vector<at::Tensor>> tensor_lists, float scale) {
  auto is_infinite_cu = makeTransformerEngineTensor(is_infinite);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);

  nvte_multi_tensor_scale_cuda(chunk_size, is_infinite_cu.data(), tensor_lists_ptr.data(),
                               num_lists, num_tensors, scale, at::cuda::getCurrentCUDAStream());
}

void multi_tensor_scale_tensor_cuda(int chunk_size, at::Tensor is_infinite,
                                    std::vector<std::vector<at::Tensor>> tensor_lists,
                                    at::Tensor scale) {
  auto is_infinite_cu = makeTransformerEngineTensor(is_infinite);
  auto scale_cu = makeTransformerEngineTensor(scale);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  nvte_multi_tensor_scale_tensor_cuda(chunk_size, is_infinite_cu.data(), tensor_lists_ptr.data(),
                                      num_lists, num_tensors, scale_cu.data(),
                                      at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
