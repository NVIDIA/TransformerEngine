/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cutlass/arch/memory.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include <transformer_engine/permutation.h>

static __global__ void moe_permute_topK_row_map(
    const int *sorted_row_id,
    int *row_id_map,
    const int num_rows,
    const int num_topK,
    const int num_out_tokens)
{
    // Each block corresponds to one source token
    // row_id_map[num_topK][num_rows]
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int idx = bid * blockDim.x + tid;

    if (idx >= num_rows * num_topK)
        return;

    int source_row = sorted_row_id[idx];
    int source_token_id = source_row / num_topK;
    int source_topK_id = source_row % num_topK;

    if (idx >= num_out_tokens)
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = -1;
    }
    else
    {
        row_id_map[source_topK_id * num_rows + source_token_id] = idx;
    }
}

template <typename T, typename TCompute, int kElementsPerAccess, bool hasProb>
__global__ void moe_recover_topK_kernel(const T *input,
                                        T *unpermuted_output,
                                        const int *row_id_map,
                                        const float *prob,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    // each block corresponds to one source token
    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x * blockDim.y)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        FragmentLoadStore frag_load_store;
        FragmentCompute frag_elem;
        FragmentCompute frag_sum;
        
        int source_row = row_id_map[source_token];

        if (source_row != -1)
        {
            const T *source_row_ptr = input + source_row * num_cols;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_sum = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_sum = frag_sum * s_prob[0];
            }
        }
        else
        {
            frag_sum.clear();
        }

        for (int k = 1; k < num_topK; k++)
        {
            source_row = row_id_map[k * num_rows + source_token];

            if (source_row == -1)
                continue;

            const T *source_row_ptr = input + source_row * num_cols;

            cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                frag_load_store, (source_row_ptr + i), true);
            frag_elem = src_converter(frag_load_store);

            if (hasProb)
            {
                frag_elem = frag_elem * s_prob[k];
            }
            
            for (int e = 0; e < kElementsPerAccess; e++)
            {
                frag_sum.at(e) = frag_sum.at(e) + frag_elem.at(e);
            }
        }

        T *dest_row_ptr = unpermuted_output + source_token * num_cols;
        frag_load_store = dst_converter(frag_sum);
        *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());
    }
}

template <typename T,
          typename TCompute,
          int kElementsPerAccess,
          int topKTile,
          bool hasProb>
__global__ void moe_permute_topK_kernel(const T *input_bwd,
                                        const T *input_fwd,
                                        T *act_grad,
                                        const float *prob,
                                        float *prob_grad,
                                        const int *row_id_map,
                                        const int num_rows,
                                        const int num_topK,
                                        const int num_cols)
{
    extern __shared__ int8_t s_mem[];
    TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

    using FragmentLoadStore = cutlass::Array<T, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<TCompute, kElementsPerAccess>;

    cutlass::NumericArrayConverter<TCompute, T, kElementsPerAccess> src_converter;
    cutlass::NumericArrayConverter<T, TCompute, kElementsPerAccess> dst_converter;

    const int source_token = blockIdx.x;
    const int tid = threadIdx.x;

    if (hasProb)
    {
        for (int i = tid; i < num_topK; i += blockDim.x)
        {
            s_prob[i] = TCompute(prob[source_token * num_topK + i]);
        }
        __syncthreads();
    }

    float accum[topKTile] = {0.0f};
    FragmentLoadStore frag_load_store;

    const T *source_row_ptr = input_bwd + source_token * num_cols;
    for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess)
    {
        cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
            frag_load_store, (source_row_ptr + i), true);
        FragmentCompute frag_src = src_converter(frag_load_store);

        int index = source_token;

        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            int dest_row = row_id_map[index];
            index += num_rows;

            if (dest_row != -1)
            {
                if (hasProb)
                {
                    frag_load_store = dst_converter(frag_src * s_prob[k]);
                }
                else
                {
                    frag_load_store = dst_converter(frag_src);
                }

                T *dest_row_ptr = act_grad + dest_row * num_cols;
                *(float4 *)(dest_row_ptr + i) = *(float4 *)(frag_load_store.data());

                if (hasProb)
                {
                    const T *input_fwd_ptr = input_fwd + dest_row * num_cols;
                    cutlass::arch::global_load<FragmentLoadStore, sizeof(FragmentLoadStore), cutlass::arch::CacheOperation::LastUse>(
                        frag_load_store, (input_fwd_ptr + i), true);
                    FragmentCompute frag_input_fwd = src_converter(frag_load_store);

                    for (int e = 0; e < kElementsPerAccess; e++)
                    {
                        accum[k] += float(frag_src.at(e) * frag_input_fwd.at(e));
                    }
                }
            }
        }
    }

    if (hasProb)
    {
        for (int k = 0; k < topKTile; k++)
        {
            if (k == num_topK) break;

            for (int mask = 16; mask > 0; mask /= 2)
            {
                accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
            }
        }

        if (tid == 0)
        {
            for (int k = 0; k < topKTile; k++)
            {
                if (k == num_topK) break;
                prob_grad[source_token * num_topK + k] = accum[k];
            }  
        }
    }
}

template <typename TInput, bool FWD, int kElementsPerAccess>
void moe_permute_topK_kernel_launcher(
    const void *input,
    void *output,
    const int *sorted_row_id,
    int *row_id_map,
    const float *prob,
    const int num_rows,
    const int num_topK,
    const int num_cols,
    const int num_out_tokens,
    cudaStream_t stream,
    float *prob_grad,
    const void *input_fwd)
{
    // Convert to cutlass type
    using T_fp16 = typename cutlass::platform::conditional<
        cutlass::platform::is_same<TInput, half>::value,
        cutlass::half_t, TInput>::type;
    using T_bf16 = typename cutlass::platform::conditional<
        cutlass::platform::is_same<T_fp16, __nv_bfloat16>::value,
        cutlass::bfloat16_t, T_fp16>::type;
    using T_fp8e5m2 = typename cutlass::platform::conditional<
        cutlass::platform::is_same<T_bf16, __nv_fp8_e5m2>::value,
        cutlass::float_e5m2_t, T_bf16>::type;
    using T_fp8e4m3 = typename cutlass::platform::conditional<
        cutlass::platform::is_same<T_fp8e5m2, __nv_fp8_e4m3>::value,
        cutlass::float_e4m3_t, T_fp8e5m2>::type;
    using T = T_fp8e4m3;

    using TCompute = typename cutlass::platform::conditional<
        (cutlass::platform::is_same<T, cutlass::float_e5m2_t>::value ||
        cutlass::platform::is_same<T, cutlass::float_e4m3_t>::value),
        cutlass::half_t, T>::type;

    if (FWD)
    {
        if (prob == nullptr)
        {
            if (input_fwd == nullptr)
            {
                // permute_topK fwd
                int threads = 64;
                int blocks = (num_rows * num_topK + threads - 1) / threads;
                moe_permute_topK_row_map<<<blocks, threads, 0, stream>>>(
                    sorted_row_id,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_out_tokens);

                blocks = num_rows;
                threads = std::min(num_cols / kElementsPerAccess, 1024);
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, false><<<blocks, threads, 0, stream>>>(
                    input,
                    nullptr,
                    output,
                    nullptr,
                    nullptr,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else
            {
                // unpermute_topK bwd without probs for topK == 1
                int blocks = num_rows;
                int threads = 32;

                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 1, false><<<blocks, threads, 0, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
        }
        else
        {
            // unpermute_topK bwd with probs
            int blocks = num_rows;
            int threads = 32;
            size_t smem_bytes = num_topK * sizeof(TCompute);

            if (num_topK <= 8)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 8, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 16)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 16, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 32)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 32, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 64)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 64, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else if (num_topK <= 128)
            {
                moe_permute_topK_kernel<T, TCompute, kElementsPerAccess, 128, true><<<blocks, threads, smem_bytes, stream>>>(
                    input,
                    input_fwd,
                    output,
                    prob,
                    prob_grad,
                    row_id_map,
                    num_rows,
                    num_topK,
                    num_cols);
            }
            else
            {
                throw std::runtime_error("num_topK cannot exceed 128.");
            }
        }
    }
    else
    {
        int blocks = num_rows;
        int threads = std::min(num_cols / kElementsPerAccess, 1024);
        size_t smem_bytes = num_topK * sizeof(TCompute);

        if (prob == nullptr)
        {
            // permute_topK bwd
            // unpermute_topK fwd without probs
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, false><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
        else
        {
            // unpermute_topK fwd with probs
            moe_recover_topK_kernel<T, TCompute, kElementsPerAccess, true><<<blocks, threads, smem_bytes, stream>>>(
                input,
                output,
                row_id_map,
                prob,
                num_rows,
                num_topK,
                num_cols);
        }
    }
}
