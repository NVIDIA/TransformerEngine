/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cub/cub.cuh>

#include "extensions.h"

using torch::Tensor;

std::tuple<Tensor, Tensor, std::vector<Tensor>> moe_permute_topK_op(
    Tensor              input,
    Tensor              indices,
    int64_t             num_out_tokens,
    std::vector<Tensor> workspace,
    int64_t             max_expanded_token_num)
{
    const int num_tokens = input.size(0);
    const int num_cols = input.size(1);
    const int num_topK = indices.size(1);

    // initialize the workspace on the first run
    if (workspace.empty()) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);

        Tensor sorted_indices = torch::empty(max_expanded_token_num, options);
        Tensor row_id = torch::range(0, max_expanded_token_num - 1, 1, options);
        Tensor sorted_row_id =
            torch::empty(max_expanded_token_num, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

        size_t temp_storage_bytes = 0;
        int *temp_ptr = nullptr;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes,
                                        temp_ptr, temp_ptr,
                                        temp_ptr, temp_ptr, max_expanded_token_num);
        Tensor temp_storage =
            torch::empty(temp_storage_bytes, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

        workspace.push_back(sorted_indices);
        workspace.push_back(row_id);
        workspace.push_back(sorted_row_id);
        workspace.push_back(temp_storage);
    }

    int *indices_ptr = get_ptr<int>(indices);
    int *sorted_indices_ptr = get_ptr<int>(workspace[0]);
    int *row_id_ptr = get_ptr<int>(workspace[1]);
    int *sorted_row_id_ptr = get_ptr<int>(workspace[2]);

    void *d_temp_storage = get_ptr<void>(workspace[3]);
    size_t temp_storage_bytes = std::numeric_limits<size_t>::max();

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                    indices_ptr, sorted_indices_ptr,
                                    row_id_ptr, sorted_row_id_ptr, num_tokens * num_topK);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    num_out_tokens = (num_out_tokens > 0) ? num_out_tokens : num_tokens * num_topK;
    Tensor permuted_output =
        torch::empty({num_out_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor row_id_map = 
        torch::empty({num_tokens * num_topK}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    void *input_ptr = getDataPtr(input, 0);
    void *permuted_output_ptr = getDataPtr(permuted_output, 0);

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        moe_permute_topK_kernel_launcher<float, true, 4>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        moe_permute_topK_kernel_launcher<half, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        moe_permute_topK_kernel_launcher<__nv_bfloat16, true, 8>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        moe_permute_topK_kernel_launcher<__nv_fp8_e5m2, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        moe_permute_topK_kernel_launcher<__nv_fp8_e4m3, true, 16>(
            input_ptr,
            permuted_output_ptr,
            sorted_row_id_ptr,
            row_id_map_ptr,
            nullptr,
            num_tokens,
            num_topK,
            num_cols,
            num_out_tokens,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(permuted_output, row_id_map, workspace);
}


Tensor moe_recover_topK_op(
    Tensor  input,
    Tensor  row_id_map,
    Tensor  prob,
    int64_t num_tokens,
    int64_t num_topK)
{
    const int num_cols = input.size(1);

    // activations type
    const at::ScalarType _st = input.scalar_type();

    // Output buffer alloc
    Tensor unpermuted_output =
        torch::empty({num_tokens, num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    void *input_ptr = getDataPtr(input, 0);
    void *unpermuted_output_ptr = getDataPtr(unpermuted_output, 0);

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        moe_permute_topK_kernel_launcher<float, false, 4>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Half:
    {
        moe_permute_topK_kernel_launcher<half, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        moe_permute_topK_kernel_launcher<__nv_bfloat16, false, 8>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        moe_permute_topK_kernel_launcher<__nv_fp8_e5m2, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        moe_permute_topK_kernel_launcher<__nv_fp8_e4m3, false, 16>(
            input_ptr,
            unpermuted_output_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return unpermuted_output;
}

std::tuple<Tensor, Tensor> moe_recover_topK_bwd_op(
    Tensor  input_bwd,
    Tensor  input_fwd,
    Tensor  row_id_map,
    Tensor  prob)
{
    const int num_topK = (prob.defined()) ? prob.size(1) : 1;
    const int num_tokens = (prob.defined()) ? prob.size(0) : row_id_map.size(0);
    const int num_cols = input_bwd.size(1);

    int *row_id_map_ptr = get_ptr<int>(row_id_map);
    float *prob_ptr = (prob.defined()) ? get_ptr<float>(prob) : nullptr;

    // activations type
    const at::ScalarType _st = input_bwd.scalar_type();

    // Output buffer alloc
    Tensor act_grad =
        torch::empty({input_fwd.size(0), num_cols}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    Tensor prob_grad =
        torch::empty({num_tokens, num_topK}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));
    float *prob_grad_ptr = get_ptr<float>(prob_grad);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    void *input_bwd_ptr = getDataPtr(input_bwd, 0);
    void *input_fwd_ptr = getDataPtr(input_fwd, 0);
    void *act_grad_ptr = getDataPtr(act_grad, 0);

    switch (_st)
    {
    case at::ScalarType::Float:
    {
        moe_permute_topK_kernel_launcher<float, true, 4>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Half:
    {
        moe_permute_topK_kernel_launcher<half, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#ifdef ENABLE_BF16
    case at::ScalarType::BFloat16:
    {
        moe_permute_topK_kernel_launcher<__nv_bfloat16, true, 8>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
#ifdef ENABLE_FP8
    case at::ScalarType::Float8_e5m2:
    {
        moe_permute_topK_kernel_launcher<__nv_fp8_e5m2, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
    case at::ScalarType::Float8_e4m3fn:
    {
        moe_permute_topK_kernel_launcher<__nv_fp8_e4m3, true, 16>(
            input_bwd_ptr,
            act_grad_ptr,
            nullptr,
            row_id_map_ptr,
            prob_ptr,
            num_tokens,
            num_topK,
            num_cols,
            0,
            stream,
            prob_grad_ptr,
            input_fwd_ptr);

        break;
    }
#endif
    default:
        throw std::runtime_error("Wrong activation tensor type.");
    }

    return std::make_tuple(act_grad, prob_grad);
}