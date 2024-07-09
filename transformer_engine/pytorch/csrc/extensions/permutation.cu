/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cub/cub.cuh>

#include "extensions.h"

using torch::Tensor;

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute(Tensor input, Tensor indices,
                                                            int64_t num_out_tokens,
                                                            std::vector<Tensor> workspace,
                                                            int64_t max_expanded_token_num) {
  const int num_tokens = input.size(0);
  const int num_cols = input.size(1);
  const int num_topK = indices.size(1);

  // initialize the workspace on the first run
  if (workspace.empty()) {
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

    Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
    Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
    Tensor sorted_row_id =
        torch::empty(max_expanded_token_num,
                     torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    size_t temp_storage_bytes = 0;
    int *temp_ptr = nullptr;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, temp_ptr, temp_ptr, temp_ptr,
                                    temp_ptr, max_expanded_token_num);
    Tensor temp_storage = torch::empty(
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
                                  num_tokens * num_topK);

  // activations type
  const at::ScalarType _st = input.scalar_type();

  // Output buffer alloc
  num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * num_topK;
  Tensor permuted_output = torch::empty(
      {num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
  Tensor row_id_map =
      torch::empty({num_tokens * num_topK},
                   torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

  int *row_id_map_ptr = reinterpret_cast<int *>(getDataPtr(row_id_map, 0));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  void *input_ptr = getDataPtr(input, 0);
  void *permuted_output_ptr = getDataPtr(permuted_output, 0);

  switch (_st) {
    case at::ScalarType::Float: {
      nvte_permutation<float, true>(input_ptr, permuted_output_ptr, sorted_row_id_ptr,
                                    row_id_map_ptr, nullptr, num_tokens, num_topK, num_cols,
                                    num_out_tokens, nullptr, nullptr, stream);

      break;
    }
    case at::ScalarType::Half: {
      nvte_permutation<half, true>(input_ptr, permuted_output_ptr, sorted_row_id_ptr,
                                   row_id_map_ptr, nullptr, num_tokens, num_topK, num_cols,
                                   num_out_tokens, nullptr, nullptr, stream);

      break;
    }
    case at::ScalarType::BFloat16: {
      nvte_permutation<__nv_bfloat16, true>(input_ptr, permuted_output_ptr, sorted_row_id_ptr,
                                            row_id_map_ptr, nullptr, num_tokens, num_topK, num_cols,
                                            num_out_tokens, nullptr, nullptr, stream);

      break;
    }
    case at::ScalarType::Float8_e5m2: {
      nvte_permutation<__nv_fp8_e5m2, true>(input_ptr, permuted_output_ptr, sorted_row_id_ptr,
                                            row_id_map_ptr, nullptr, num_tokens, num_topK, num_cols,
                                            num_out_tokens, nullptr, nullptr, stream);

      break;
    }
    case at::ScalarType::Float8_e4m3fn: {
      nvte_permutation<__nv_fp8_e4m3, true>(input_ptr, permuted_output_ptr, sorted_row_id_ptr,
                                            row_id_map_ptr, nullptr, num_tokens, num_topK, num_cols,
                                            num_out_tokens, nullptr, nullptr, stream);

      break;
    }
    default:
      NVTE_ERROR("Wrong activation tensor type.");
  }

  return std::make_tuple(permuted_output, row_id_map, workspace);
}

Tensor moe_unpermute_fwd(Tensor input, Tensor row_id_map, Tensor prob, int64_t num_tokens,
                         int64_t num_topK) {
  const int num_cols = input.size(1);

  // activations type
  const at::ScalarType _st = input.scalar_type();

  // Output buffer alloc
  Tensor unpermuted_output = torch::empty(
      {num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

  int *row_id_map_ptr = reinterpret_cast<int *>(getDataPtr(row_id_map, 0));
  float *prob_ptr = (prob.numel() > 0) ? reinterpret_cast<float *>(getDataPtr(prob, 0)) : nullptr;
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  void *input_ptr = getDataPtr(input, 0);
  void *unpermuted_output_ptr = getDataPtr(unpermuted_output, 0);

  switch (_st) {
    case at::ScalarType::Float: {
      nvte_permutation<float, false>(input_ptr, unpermuted_output_ptr, nullptr, row_id_map_ptr,
                                     prob_ptr, num_tokens, num_topK, num_cols, 0, nullptr, nullptr,
                                     stream);

      break;
    }
    case at::ScalarType::Half: {
      nvte_permutation<half, false>(input_ptr, unpermuted_output_ptr, nullptr, row_id_map_ptr,
                                    prob_ptr, num_tokens, num_topK, num_cols, 0, nullptr, nullptr,
                                    stream);

      break;
    }
    case at::ScalarType::BFloat16: {
      nvte_permutation<__nv_bfloat16, false>(input_ptr, unpermuted_output_ptr, nullptr,
                                             row_id_map_ptr, prob_ptr, num_tokens, num_topK,
                                             num_cols, 0, nullptr, nullptr, stream);

      break;
    }
    case at::ScalarType::Float8_e5m2: {
      nvte_permutation<__nv_fp8_e5m2, false>(input_ptr, unpermuted_output_ptr, nullptr,
                                             row_id_map_ptr, prob_ptr, num_tokens, num_topK,
                                             num_cols, 0, nullptr, nullptr, stream);

      break;
    }
    case at::ScalarType::Float8_e4m3fn: {
      nvte_permutation<__nv_fp8_e4m3, false>(input_ptr, unpermuted_output_ptr, nullptr,
                                             row_id_map_ptr, prob_ptr, num_tokens, num_topK,
                                             num_cols, 0, nullptr, nullptr, stream);

      break;
    }
    default:
      NVTE_ERROR("Wrong activation tensor type.");
  }

  return unpermuted_output;
}

std::tuple<Tensor, Tensor> moe_unpermute_bwd(Tensor input_bwd, Tensor input_fwd, Tensor row_id_map,
                                             Tensor prob) {
  const int num_topK = (prob.numel() > 0) ? prob.size(1) : 1;
  const int num_tokens = (prob.numel() > 0) ? prob.size(0) : row_id_map.size(0);
  const int num_cols = input_bwd.size(1);

  int *row_id_map_ptr = reinterpret_cast<int *>(getDataPtr(row_id_map, 0));
  float *prob_ptr = (prob.numel() > 0) ? reinterpret_cast<float *>(getDataPtr(prob, 0)) : nullptr;

  // activations type
  const at::ScalarType _st = input_bwd.scalar_type();

  // Output buffer alloc
  Tensor act_grad = torch::empty({input_fwd.size(0), num_cols},
                                 torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
  Tensor prob_grad =
      torch::empty({num_tokens, num_topK},
                   torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
  float *prob_grad_ptr = reinterpret_cast<float *>(getDataPtr(prob_grad, 0));

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  void *input_bwd_ptr = getDataPtr(input_bwd, 0);
  void *input_fwd_ptr = getDataPtr(input_fwd, 0);
  void *act_grad_ptr = getDataPtr(act_grad, 0);

  switch (_st) {
    case at::ScalarType::Float: {
      nvte_permutation<float, true>(input_bwd_ptr, act_grad_ptr, nullptr, row_id_map_ptr, prob_ptr,
                                    num_tokens, num_topK, num_cols, 0, prob_grad_ptr, input_fwd_ptr,
                                    stream);

      break;
    }
    case at::ScalarType::Half: {
      nvte_permutation<half, true>(input_bwd_ptr, act_grad_ptr, nullptr, row_id_map_ptr, prob_ptr,
                                   num_tokens, num_topK, num_cols, 0, prob_grad_ptr, input_fwd_ptr,
                                   stream);

      break;
    }
    case at::ScalarType::BFloat16: {
      nvte_permutation<__nv_bfloat16, true>(input_bwd_ptr, act_grad_ptr, nullptr, row_id_map_ptr,
                                            prob_ptr, num_tokens, num_topK, num_cols, 0,
                                            prob_grad_ptr, input_fwd_ptr, stream);

      break;
    }
    case at::ScalarType::Float8_e5m2: {
      nvte_permutation<__nv_fp8_e5m2, true>(input_bwd_ptr, act_grad_ptr, nullptr, row_id_map_ptr,
                                            prob_ptr, num_tokens, num_topK, num_cols, 0,
                                            prob_grad_ptr, input_fwd_ptr, stream);

      break;
    }
    case at::ScalarType::Float8_e4m3fn: {
      nvte_permutation<__nv_fp8_e4m3, true>(input_bwd_ptr, act_grad_ptr, nullptr, row_id_map_ptr,
                                            prob_ptr, num_tokens, num_topK, num_cols, 0,
                                            prob_grad_ptr, input_fwd_ptr, stream);

      break;
    }
    default:
      NVTE_ERROR("Wrong activation tensor type.");
  }

  return std::make_tuple(act_grad, prob_grad);
}
