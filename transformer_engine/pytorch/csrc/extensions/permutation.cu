/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cub/cub.cuh>

#include "extensions.h"

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> moe_permute_fwd(
    at::Tensor input, const transformer_engine::DType dtype, at::Tensor indices,
    int64_t num_out_tokens, std::vector<at::Tensor> workspace, int64_t max_expanded_token_num) {
  using namespace transformer_engine::pytorch;
  const int num_tokens = input.size(0);
  int num_cols = input.size(1);
  const int topK = indices.size(1);

  // Initialize the workspace on the first run
  if (workspace.empty()) {
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

    at::Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
    at::Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
    at::Tensor sorted_row_id =
        torch::empty(max_expanded_token_num,
                     torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    size_t temp_storage_bytes = 0;
    int *temp_ptr = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_ptr, temp_ptr, temp_ptr,
                                    temp_ptr, max_expanded_token_num);
    at::Tensor temp_storage = torch::empty(
        temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    workspace.push_back(sorted_indices);
    workspace.push_back(row_id);
    workspace.push_back(sorted_row_id);
    workspace.push_back(temp_storage);
  }

  int *indices_ptr = reinterpret_cast<int *>(getDataPtr(indices, 0));
  int *sorted_indices_ptr = reinterpret_cast<int *>(getDataPtr(workspace[0], 0));
  int *row_id_ptr = reinterpret_cast<int *>(getDataPtr(workspace[1], 0));
  int *sorted_row_id_ptr = reinterpret_cast<int *>(getDataPtr(workspace[2], 0));

  void *d_temp_storage = getDataPtr(workspace[3], 0);
  size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, indices_ptr,
                                  sorted_indices_ptr, row_id_ptr, sorted_row_id_ptr,
                                  num_tokens * topK);

  // Activations type
  at::ScalarType _st;
  if (dtype == transformer_engine::DType::kFloat8E4M3 ||
      dtype == transformer_engine::DType::kFloat8E5M2)
    _st = at::ScalarType::Byte;
  else
    _st = input.scalar_type();

  // Output buffer alloc
  num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * topK;
  at::Tensor permuted_output = torch::empty(
      {num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
  at::Tensor row_id_map = torch::empty(
      {num_tokens * topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(), {static_cast<size_t>(input.size(0)), static_cast<size_t>(num_cols)}, dtype);
  auto permuted_output_cu = makeTransformerEngineTensor(
      permuted_output.data_ptr(),
      {static_cast<size_t>(permuted_output.size(0)), static_cast<size_t>(num_cols)}, dtype);
  auto sorted_row_id_cu =
      makeTransformerEngineTensor(sorted_row_id_ptr, {static_cast<size_t>(num_tokens * topK)},
                                  transformer_engine::DType::kInt32);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);

  nvte_permute(input_cu.data(), permuted_output_cu.data(), sorted_row_id_cu.data(),
               row_id_map_cu.data(), transformer_engine::TensorWrapper().data(),
               transformer_engine::TensorWrapper().data(),
               transformer_engine::TensorWrapper().data(), num_tokens, topK, num_cols,
               num_out_tokens, stream);

  return std::make_tuple(permuted_output, row_id_map, workspace);
}

at::Tensor moe_permute_bwd(at::Tensor input, const transformer_engine::DType dtype,
                           at::Tensor row_id_map, at::Tensor prob, int64_t num_tokens,
                           int64_t topK) {
  return moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK);
}

at::Tensor moe_unpermute_fwd(at::Tensor input, const transformer_engine::DType dtype,
                             at::Tensor row_id_map, at::Tensor prob, int64_t num_tokens,
                             int64_t topK) {
  using namespace transformer_engine::pytorch;
  int num_cols = input.size(1);

  // Activations type
  at::ScalarType _st;
  if (dtype == transformer_engine::DType::kFloat8E4M3 ||
      dtype == transformer_engine::DType::kFloat8E5M2)
    _st = at::ScalarType::Byte;
  else
    _st = input.scalar_type();

  // Output buffer alloc
  at::Tensor unpermuted_output = torch::empty(
      {num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(), {static_cast<size_t>(input.size(0)), static_cast<size_t>(num_cols)}, dtype);
  auto unpermuted_output_cu = makeTransformerEngineTensor(
      unpermuted_output.data_ptr(),
      {static_cast<size_t>(unpermuted_output.size(0)), static_cast<size_t>(num_cols)}, dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);

  nvte_unpermute(input_cu.data(), unpermuted_output_cu.data(), row_id_map_cu.data(), prob_cu.data(),
                 num_tokens, topK, num_cols, stream);

  return unpermuted_output;
}

std::tuple<at::Tensor, at::Tensor> moe_unpermute_bwd(at::Tensor input_bwd, at::Tensor input_fwd,
                                                     const transformer_engine::DType dtype,
                                                     at::Tensor row_id_map, at::Tensor prob) {
  using namespace transformer_engine::pytorch;
  const int topK = (prob.numel() > 0) ? prob.size(1) : 1;
  const int num_tokens = (prob.numel() > 0) ? prob.size(0) : row_id_map.size(0);
  int num_cols = input_bwd.size(1);

  // Activations type
  at::ScalarType _st;
  if (dtype == transformer_engine::DType::kFloat8E4M3 ||
      dtype == transformer_engine::DType::kFloat8E5M2)
    _st = at::ScalarType::Byte;
  else
    _st = input_bwd.scalar_type();

  // Output buffer alloc
  at::Tensor act_grad = torch::empty({input_fwd.size(0), num_cols},
                                     torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
  at::Tensor prob_grad = torch::empty(
      {num_tokens, topK}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto input_bwd_cu = makeTransformerEngineTensor(
      input_bwd.data_ptr(), {static_cast<size_t>(input_bwd.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto act_grad_cu = makeTransformerEngineTensor(
      act_grad.data_ptr(), {static_cast<size_t>(act_grad.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto input_fwd_cu = makeTransformerEngineTensor(
      input_fwd.data_ptr(), {static_cast<size_t>(input_fwd.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);
  auto prob_grad_cu = makeTransformerEngineTensor(prob_grad);

  nvte_permute(input_bwd_cu.data(), act_grad_cu.data(), transformer_engine::TensorWrapper().data(),
               row_id_map_cu.data(), prob_cu.data(), prob_grad_cu.data(), input_fwd_cu.data(),
               num_tokens, topK, num_cols, 0, stream);

  return std::make_tuple(act_grad, prob_grad);
}
