/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> moe_permute_fwd(
    at::Tensor input, const DType dtype, at::Tensor indices, int64_t num_out_tokens,
    std::vector<at::Tensor> workspace, int64_t max_expanded_token_num) {
  const int num_tokens = input.size(0);
  int num_cols = input.size(1);
  const int topK = indices.size(1);

  // Initialize the workspace on the first run
  if (workspace.empty()) {
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kMUSA).requires_grad(false);

    at::Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
    at::Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
    at::Tensor sorted_row_id =
        torch::empty(max_expanded_token_num,
                     torch::dtype(torch::kInt32).device(torch::kMUSA).requires_grad(false));

    size_t temp_storage_bytes = 0;
    nvte_device_radix_sort_pairs(nullptr, &temp_storage_bytes, nullptr, nullptr, nullptr, nullptr,
                                 max_expanded_token_num);
    at::Tensor temp_storage = torch::empty(
        temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kMUSA).requires_grad(false));

    workspace.push_back(sorted_indices);
    workspace.push_back(row_id);
    workspace.push_back(sorted_row_id);
    workspace.push_back(temp_storage);
  }

  void *indices_ptr = getDataPtr(indices, 0);
  void *sorted_indices_ptr = getDataPtr(workspace[0], 0);
  void *row_id_ptr = getDataPtr(workspace[1], 0);
  void *sorted_row_id_ptr = getDataPtr(workspace[2], 0);

  void *d_temp_storage = getDataPtr(workspace[3], 0);
  size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

  nvte_device_radix_sort_pairs(
      d_temp_storage, &temp_storage_bytes, reinterpret_cast<int *>(indices_ptr),
      reinterpret_cast<int *>(sorted_indices_ptr), reinterpret_cast<int *>(row_id_ptr),
      reinterpret_cast<int *>(sorted_row_id_ptr), num_tokens * topK);

  // Output buffer alloc
  num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * topK;
  at::Tensor permuted_output =
      torch::empty({num_out_tokens, num_cols},
                   torch::dtype(input.scalar_type()).device(torch::kMUSA).requires_grad(false));
  at::Tensor row_id_map = torch::empty(
      {num_tokens * topK}, torch::dtype(torch::kInt32).device(torch::kMUSA).requires_grad(false));

  auto stream = at::musa::getCurrentCUDAStream().stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto permuted_output_cu =
      makeTransformerEngineTensor(permuted_output.data_ptr(),
                                  std::vector<size_t>{static_cast<size_t>(permuted_output.size(0)),
                                                      static_cast<size_t>(num_cols)},
                                  dtype);
  auto sorted_row_id_cu = makeTransformerEngineTensor(
      sorted_row_id_ptr, std::vector<size_t>{static_cast<size_t>(num_tokens * topK)},
      DType::kInt32);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);

  nvte_permute(input_cu.data(), permuted_output_cu.data(), sorted_row_id_cu.data(),
               row_id_map_cu.data(), TensorWrapper().data(), TensorWrapper().data(),
               TensorWrapper().data(), num_tokens, topK, num_cols, num_out_tokens, stream);

  return std::make_tuple(permuted_output, row_id_map, workspace);
}

at::Tensor moe_permute_bwd(at::Tensor input, const DType dtype, at::Tensor row_id_map,
                           at::Tensor prob, int64_t num_tokens, int64_t topK) {
  return moe_unpermute_fwd(input, dtype, row_id_map, prob, num_tokens, topK);
}

at::Tensor moe_unpermute_fwd(at::Tensor input, const DType dtype, at::Tensor row_id_map,
                             at::Tensor prob, int64_t num_tokens, int64_t topK) {
  int num_cols = input.size(1);

  // Output buffer alloc
  at::Tensor unpermuted_output =
      torch::empty({num_tokens, num_cols},
                   torch::dtype(input.scalar_type()).device(torch::kMUSA).requires_grad(false));

  auto stream = at::musa::getCurrentCUDAStream().stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto unpermuted_output_cu = makeTransformerEngineTensor(
      unpermuted_output.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(unpermuted_output.size(0)),
                          static_cast<size_t>(num_cols)},
      dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);

  nvte_unpermute(input_cu.data(), unpermuted_output_cu.data(), row_id_map_cu.data(), prob_cu.data(),
                 num_tokens, topK, num_cols, stream);

  return unpermuted_output;
}

std::tuple<at::Tensor, at::Tensor> moe_unpermute_bwd(at::Tensor input_bwd, at::Tensor input_fwd,
                                                     const DType dtype, at::Tensor row_id_map,
                                                     at::Tensor prob) {
  const int topK = (prob.numel() > 0) ? prob.size(1) : 1;
  const int num_tokens = (prob.numel() > 0) ? prob.size(0) : row_id_map.size(0);
  int num_cols = input_bwd.size(1);

  // Output buffer alloc
  at::Tensor act_grad =
      torch::empty({input_fwd.size(0), num_cols},
                   torch::dtype(input_bwd.scalar_type()).device(torch::kMUSA).requires_grad(false));
  at::Tensor prob_grad = torch::empty(
      {num_tokens, topK}, torch::dtype(torch::kFloat32).device(torch::kMUSA).requires_grad(false));

  auto stream = at::musa::getCurrentCUDAStream().stream();

  auto input_bwd_cu = makeTransformerEngineTensor(
      input_bwd.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input_bwd.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto act_grad_cu = makeTransformerEngineTensor(
      act_grad.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(act_grad.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto input_fwd_cu = makeTransformerEngineTensor(
      input_fwd.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input_fwd.size(0)), static_cast<size_t>(num_cols)},
      dtype);
  auto row_id_map_cu = makeTransformerEngineTensor(row_id_map);
  auto prob_cu = makeTransformerEngineTensor(prob);
  auto prob_grad_cu = makeTransformerEngineTensor(prob_grad);

  nvte_permute(input_bwd_cu.data(), act_grad_cu.data(), TensorWrapper().data(),
               row_id_map_cu.data(), prob_cu.data(), prob_grad_cu.data(), input_fwd_cu.data(),
               num_tokens, topK, num_cols, 0, stream);

  return std::make_tuple(act_grad, prob_grad);
}

std::tuple<at::Tensor, at::Tensor> moe_permute_mask(const transformer_engine::DType dtype,
                                                    at::Tensor input, at::Tensor row_id_map,
                                                    at::Tensor probs, int num_tokens,
                                                    int num_experts, int num_out_tokens,
                                                    int hidden_size) {
  using namespace transformer_engine::pytorch;
  const transformer_engine::DType probs_dtype = GetTransformerEngineDType(probs.scalar_type());

  at::Tensor output =
      torch::empty({num_out_tokens, hidden_size},
                   torch::dtype(input.dtype()).device(torch::kPrivateUse1).requires_grad(false));

  at::Tensor permuted_probs =
      torch::empty({num_out_tokens},
                   torch::dtype(probs.dtype()).device(torch::kPrivateUse1).requires_grad(false));
  ;

  auto stream = at::musa::getCurrentMUSAStream().stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(hidden_size)},
      dtype);
  auto output_cu = makeTransformerEngineTensor(
      output.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_out_tokens), static_cast<size_t>(hidden_size)},
      dtype);
  auto row_id_map_cu =
      makeTransformerEngineTensor(row_id_map.data_ptr(),
                                  std::vector<size_t>{static_cast<size_t>(row_id_map.size(0)),
                                                      static_cast<size_t>(row_id_map.size(1))},
                                  transformer_engine::DType::kInt64);
  auto probs_cu = makeTransformerEngineTensor(
      probs.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
      probs_dtype);  // probs dtype
  auto permuted_probs_cu = makeTransformerEngineTensor(
      permuted_probs.data_ptr(), std::vector<size_t>{static_cast<size_t>(num_out_tokens)},
      probs_dtype);  // probs dtype

  if (dtype == probs_dtype) {
    nvte_permute_mask(input_cu.data(), output_cu.data(), row_id_map_cu.data(), probs_cu.data(),
                      permuted_probs_cu.data(), num_tokens, num_experts, num_out_tokens,
                      hidden_size, stream);
  } else {
    nvte_permute_mask_high_precision_probs(input_cu.data(), output_cu.data(), row_id_map_cu.data(),
                                           probs_cu.data(), permuted_probs_cu.data(), num_tokens,
                                           num_experts, num_out_tokens, hidden_size, stream);
  }
  return std::make_tuple(output, permuted_probs);
}

std::tuple<at::Tensor, at::Tensor> moe_unpermute_mask(const transformer_engine::DType dtype,
                                                      at::Tensor input, at::Tensor row_id_map,
                                                      at::Tensor merging_probs,
                                                      at::Tensor permuted_probs, int num_tokens,
                                                      int num_experts, int hidden_size,
                                                      at::Tensor preallocated_act) {
  using namespace transformer_engine::pytorch;
  const transformer_engine::DType probs_dtype =
      GetTransformerEngineDType(permuted_probs.scalar_type());

  at::Tensor output = preallocated_act;
  if (output.data_ptr() == nullptr) {
    output =
        torch::empty({num_tokens, hidden_size},
                     torch::dtype(input.dtype()).device(torch::kPrivateUse1).requires_grad(false));
  }

  at::Tensor unpermuted_probs = torch::empty(
      {num_tokens, num_experts},
      torch::dtype(permuted_probs.dtype()).device(torch::kPrivateUse1).requires_grad(false));
  ;

  auto stream = at::musa::getCurrentMUSAStream().stream();

  auto input_cu = makeTransformerEngineTensor(
      input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(input.size(0)), static_cast<size_t>(hidden_size)},
      dtype);
  auto output_cu = makeTransformerEngineTensor(
      output.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(hidden_size)},
      dtype);
  auto row_id_map_cu =
      makeTransformerEngineTensor(row_id_map.data_ptr(),
                                  std::vector<size_t>{static_cast<size_t>(row_id_map.size(0)),
                                                      static_cast<size_t>(row_id_map.size(1))},
                                  transformer_engine::DType::kInt64);

  auto merging_probs_cu = makeTransformerEngineTensor(
      merging_probs.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
      probs_dtype);
  //   auto permuted_probs_cu = makeTransformerEngineTensor(permuted_probs);
  auto permuted_probs_cu = makeTransformerEngineTensor(
      permuted_probs.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
      probs_dtype);
  auto unpermuted_probs_cu = makeTransformerEngineTensor(
      unpermuted_probs.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
      probs_dtype);

  if (dtype == probs_dtype) {
    nvte_unpermute_mask(input_cu.data(), output_cu.data(), row_id_map_cu.data(),
                        merging_probs_cu.data(), permuted_probs_cu.data(),
                        unpermuted_probs_cu.data(), num_tokens, num_experts, hidden_size, stream);
  } else {
    nvte_unpermute_mask_high_precision_probs(input_cu.data(), output_cu.data(),
                                             row_id_map_cu.data(), merging_probs_cu.data(),
                                             permuted_probs_cu.data(), unpermuted_probs_cu.data(),
                                             num_tokens, num_experts, hidden_size, stream);
  }
  return std::make_tuple(output, unpermuted_probs);
}

std::tuple<at::Tensor, at::Tensor> moe_unpermute_mask_bwd_with_merging_probs(
    const transformer_engine::DType dtype, at::Tensor fwd_output_grad, at::Tensor fwd_input,
    at::Tensor merging_probs, at::Tensor row_id_map, int num_tokens, int num_experts,
    int num_out_tokens, int hidden_size) {
  using namespace transformer_engine::pytorch;

  at::Tensor fwd_input_grad =
      torch::empty({num_out_tokens, hidden_size},
                   torch::dtype(torch::kBFloat16).device(torch::kPrivateUse1).requires_grad(false));
  at::Tensor merging_probs_grad =
      torch::empty({num_tokens, num_experts},
                   torch::dtype(torch::kBFloat16).device(torch::kPrivateUse1).requires_grad(false));

  auto stream = at::musa::getCurrentMUSAStream().stream();

  auto fwd_output_grad_cu = makeTransformerEngineTensor(
      fwd_output_grad.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(hidden_size)},
      dtype);
  auto fwd_input_grad_cu = makeTransformerEngineTensor(
      fwd_input_grad.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_out_tokens), static_cast<size_t>(hidden_size)},
      dtype);
  auto fwd_input_cu = makeTransformerEngineTensor(
      fwd_input.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_out_tokens), static_cast<size_t>(hidden_size)},
      dtype);
  auto merging_probs_cu = makeTransformerEngineTensor(
      merging_probs.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
      dtype);
  auto merging_probs_grad_cu = makeTransformerEngineTensor(
      merging_probs_grad.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)},
      dtype);
  auto row_id_map_cu =
      makeTransformerEngineTensor(row_id_map.data_ptr(),
                                  std::vector<size_t>{static_cast<size_t>(row_id_map.size(0)),
                                                      static_cast<size_t>(row_id_map.size(1))},
                                  transformer_engine::DType::kInt64);

  nvte_unpermute_mask_bwd_with_merging_probs(fwd_output_grad_cu.data(), fwd_input_grad_cu.data(),
                                             fwd_input_cu.data(), merging_probs_cu.data(),
                                             merging_probs_grad_cu.data(), row_id_map_cu.data(),
                                             num_tokens, num_experts, hidden_size, stream);

  return std::make_tuple(fwd_input_grad, merging_probs_grad);
}

}  // namespace transformer_engine::pytorch
