/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../../extensions.h"

namespace transformer_engine::pytorch {

at::Tensor multi_tensor_raw_moments_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists) {
  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = 0;
  for (int t = 0; t < ntensors; t++) {
    int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
    if (max_chunks_this_tensor > max_chunks_per_tensor) {
      max_chunks_per_tensor = max_chunks_this_tensor;
    }
  }

  auto ret = at::empty({ntensors, 5}, float_options);
  if (max_chunks_per_tensor == 0) {
    ret.zero_();
    return ret;
  }

  auto output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor * 5}, float_options);

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [_, __, tensor_lists_ptr, num_lists, num_tensors] =
      makeTransformerEngineTensorList(tensor_lists);
  auto output_per_tensor_cu = makeTransformerEngineTensor(output_per_tensor);
  auto ret_cu = makeTransformerEngineTensor(ret);

  nvte_multi_tensor_raw_moments_cuda(
      chunk_size, noop_flag_cu.data(), tensor_lists_ptr.data(), num_lists, num_tensors,
      output_per_tensor_cu.data(), ret_cu.data(), max_chunks_per_tensor,
      at::cuda::getCurrentCUDAStream());

  return ret;
}

}  // namespace transformer_engine::pytorch
