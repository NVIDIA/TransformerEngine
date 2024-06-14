/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include <cub/cub.cuh>

at::Tensor scaled_softmax_forward(at::Tensor input,
                                  float scale_factor
) {
    using namespace transformer_engine;
    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int batches = input.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);

    AT_ASSERTM(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
    AT_ASSERTM(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
    AT_ASSERTM(query_seq_len > 1, "Query sequence length must be greater than 1");

    // Output
  auto act_options = input.options().requires_grad(false);
  auto softmax_results =
      torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(), scale_factor,
                              at::cuda::getCurrentCUDAStream());

  return softmax_results;
}


at::Tensor scaled_softmax_backward(at::Tensor output_grad_,
                                   at::Tensor softmax_results_,
                                   float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 4D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 4D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(
          output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return output_grads;
}


at::Tensor scaled_masked_softmax_forward(at::Tensor input,
                                         at::Tensor mask,
                                         float scale_factor
) {
    using namespace transformer_engine;

    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");
    AT_ASSERTM(mask.dim() == 4, "expected 4D tensor");
    if (!input.is_contiguous())
        input = input.contiguous();
    if (!mask.is_contiguous())
        mask = mask.contiguous();

    const int batches = input.size(0);
    const int pad_batches = mask.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);

    AT_ASSERTM(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
    AT_ASSERTM(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
    AT_ASSERTM(query_seq_len > 1, "Query sequence length must be greater than 1");
    TORCH_CHECK(pad_batches == 1 || pad_batches == batches);
    TORCH_CHECK(mask.size(1) == 1);
    TORCH_CHECK(mask.size(2) == query_seq_len);
    TORCH_CHECK(mask.size(3) == key_seq_len);

    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);


    auto input_cu = makeTransformerEngineTensor(input);
    auto mask_cu = makeTransformerEngineTensor(mask);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_masked_softmax_forward(
          input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_masked_softmax_backward(at::Tensor output_grad_,
                                          at::Tensor softmax_results_,
                                          float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 3D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(
          output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return output_grads;
}


at::Tensor scaled_upper_triang_masked_softmax_forward(at::Tensor input,
                                                      float scale_factor
) {
    using namespace transformer_engine;

    AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int attn_batches = input.size(0);
    const int seq_len = input.size(1);
    AT_ASSERTM(seq_len <= 16384, "Sequence length must be 16384 or less");

    // Output
    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({attn_batches, seq_len, seq_len}, act_options);

    auto input_cu = makeTransformerEngineTensor(input);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_upper_triang_masked_softmax_forward(input_cu.data(),
                                                    softmax_results_cu.data(),
                                                    scale_factor,
                                                    at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_upper_triang_masked_softmax_backward(at::Tensor output_grads_,
                                                       at::Tensor softmax_results_,
                                                       float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grads_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    TORCH_CHECK(output_grads.size(1) == output_grads.size(2));

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_upper_triang_masked_softmax_backward(output_grads_cu.data(),
                                                     softmax_results_cu.data(),
                                                     output_grads_cu.data(),
                                                     scale_factor,
                                                     at::cuda::getCurrentCUDAStream());

  return output_grads;
}


at::Tensor scaled_aligned_causal_masked_softmax_forward(
    at::Tensor input,
    float scale_factor
) {
    using namespace transformer_engine;
    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int batches = input.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);

    AT_ASSERTM(key_seq_len <= 16384, "Key sequence length must be 16384 or less");
    AT_ASSERTM(key_seq_len % 8 == 0, "Key sequence length must be divisible by 8");
    AT_ASSERTM(query_seq_len >= 1, "Query sequence length must be greater or equal to 1");

    // Output
    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

    auto input_cu = makeTransformerEngineTensor(input);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_aligned_causal_masked_softmax_forward(
        input_cu.data(),
        softmax_results_cu.data(),
        scale_factor,
        at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_aligned_causal_masked_softmax_backward(
    at::Tensor output_grad_,
    at::Tensor softmax_results_,
    float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 4D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 4D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_aligned_causal_masked_softmax_backward(
        output_grads_cu.data(),
        softmax_results_cu.data(),
        output_grads_cu.data(),
        scale_factor,
        at::cuda::getCurrentCUDAStream());

    return output_grads;
}


/***************************************************************************************************
 * Support memory efficient cross entropy for Megatron-LM
 **************************************************************************************************/
 template<typename dtype, int BlockSize>
 void __global__  CrossEntropyFwdSumExpKernel(float* sum_exp_logits_ptr,
                                            dtype* vocab_parallel_logits_ptr,
                                            float* logits_max_ptr,
                                            size_t n_dim) {
    /***
    1024 | 1
    7 | 1016 | 2
    6 | 1016 | 3

    For example: 
    1024 | 1 -> [0,1023] [1024]
    7 | 1016 | 2 -> [0, 6], [7,1022], [1023,1024]
    ***/

    /***
    Thread model: size_t grid = rows;
    One block is responsible for one row.
    ***/
    size_t rowIdx = blockIdx.x;
    size_t tid = threadIdx.x;
    if(tid >= n_dim) return;

    size_t cur_vocab_parallel_logits_ptr_begin = rowIdx * n_dim; // 0, 1025
    size_t cur_vocab_parallel_logits_ptr_end = rowIdx * n_dim + n_dim; //cur_vocab_parallel_logits_ptr_end = 1025, 2050

    size_t end_mol_num = cur_vocab_parallel_logits_ptr_end % 8; //end_mol_num = 1, end_mol_num = 2
    size_t begin_mol_num = n_dim - end_mol_num; //begin_mol_num = 1024, begin_mol_num = 1023

    //valid range for evry row is [begin_offset, end_offset]
    size_t begin_offset = begin_mol_num % 8;//begin_offset = 0, begin_offset = 7
    size_t end_offset = n_dim - end_mol_num - 1;//end_offset = 1023, end_offset = 1022

    float cur_row_max = logits_max_ptr[rowIdx];
    float cur_thread_exp_sum = 0.0;
    float row_item = 0.0;

    typedef cub::BlockReduce<float, BlockSize> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    #pragma unroll
    for (size_t i = begin_offset + tid * 8; i <= end_offset - 7; i += 8 * BlockSize) {
    {
        int4 int4_arr = *reinterpret_cast<int4*>(&vocab_parallel_logits_ptr[cur_vocab_parallel_logits_ptr_begin + i]);
        dtype* bf_16_p = reinterpret_cast<dtype*>(&int4_arr);
        #pragma unroll
        for (int k = 0; k < 8; k ++) {
            dtype data_bf16 = bf_16_p[k];
            float data_fp32 = float(data_bf16); //convert to float
            row_item = __expf(data_fp32 - cur_row_max);
            cur_thread_exp_sum += row_item;
        }
    }
    }

    float row_sum = BlockReduceT(temp_storage).Sum(cur_thread_exp_sum);

    if (threadIdx.x == 0) {
    #pragma unroll
    for (size_t k = cur_vocab_parallel_logits_ptr_begin; k < (cur_vocab_parallel_logits_ptr_begin + begin_offset); k++) {
        float val = float(vocab_parallel_logits_ptr[k]);
        row_item = __expf(val - cur_row_max);
        row_sum += row_item;
    }
    #pragma unroll
        for (size_t k = cur_vocab_parallel_logits_ptr_begin + end_offset + 1; k < cur_vocab_parallel_logits_ptr_end; k ++) {
        float val = float(vocab_parallel_logits_ptr[k]);
        row_item = __expf(val - cur_row_max);
        row_sum += row_item;
    }
    sum_exp_logits_ptr[rowIdx] = row_sum;
    }
}


float __device__ __forceinline__ compute_mean_log(float data_fp32, float cur_row_max, float cur_row_exp_sum) {
    float row_item = expf(data_fp32 - cur_row_max);
    row_item = row_item / cur_row_exp_sum; //compute softmax
    row_item = __logf(row_item); //after softmax, compute log
    return row_item;
}


template<typename dtype, int BlockSize>
void __global__ CrossEntropyFwdMeanLogKernel(float* mean_log_probs_ptr,
                                            dtype* vocab_parallel_logits_ptr, 
                                            float* logits_max_ptr,
                                            float* sum_exp_logits_ptr,
                                            size_t n_dim) {
        size_t rowIdx = blockIdx.x;
        size_t tid = threadIdx.x;
        if (tid >= n_dim) return;

        size_t cur_vocab_parallel_logits_ptr_begin = rowIdx * n_dim; // 0, 1025
        size_t cur_vocab_parallel_logits_ptr_end = rowIdx * n_dim + n_dim; //cur_vocab_parallel_logits_ptr_end = 1025, 2050

        size_t end_mol_num = cur_vocab_parallel_logits_ptr_end % 8; //end_mol_num = 1, end_mol_num = 2
        size_t begin_mol_num = n_dim - end_mol_num; //begin_mol_num = 1024, begin_mol_num = 1023
        
        //valid range for evry row is [begin_offset, end_offset]
        size_t begin_offset = begin_mol_num % 8;//begin_offset = 0, begin_offset = 7
        size_t end_offset = n_dim - end_mol_num - 1;//end_offset = 1023, end_offset = 1022
                           
        
        float cur_row_exp_sum = sum_exp_logits_ptr[rowIdx];
        float cur_row_max = logits_max_ptr[rowIdx];
        float row_item = 0;
        float cur_thread_softmax_log_mean = 0.0;

        typedef cub::BlockReduce<float, BlockSize> BlockReduceT;
        __shared__ typename BlockReduceT::TempStorage temp_storage;
        
        for (size_t i = begin_offset + tid * 8; i <= end_offset - 7; i += 8 * BlockSize) {
            int4 int4_arr = *reinterpret_cast<int4*>(&vocab_parallel_logits_ptr[cur_vocab_parallel_logits_ptr_begin + i]);
            dtype* bf_16_p = reinterpret_cast<dtype*>(&int4_arr);
            #pragma unroll
            for (int k = 0; k < 8; k ++) {
                dtype data_bf16 = bf_16_p[k];
                float data_fp32 = float(data_bf16); //convert to float
                row_item = compute_mean_log(data_fp32, cur_row_max, cur_row_exp_sum);
                cur_thread_softmax_log_mean += row_item; //sum all "log value"
            }
        }

        float row_log_sum = BlockReduceT(temp_storage).Sum(cur_thread_softmax_log_mean);

        if(threadIdx.x == 0) {
            #pragma unroll
            for (size_t k = cur_vocab_parallel_logits_ptr_begin; k < (cur_vocab_parallel_logits_ptr_begin + begin_offset); k++) {
                float val = float(vocab_parallel_logits_ptr[k]);
                row_item = compute_mean_log(val, cur_row_max, cur_row_exp_sum);
                row_log_sum += row_item;
            }
            #pragma unroll
            for (size_t k = cur_vocab_parallel_logits_ptr_begin + end_offset + 1; k < cur_vocab_parallel_logits_ptr_end; k ++) {
                float val = float(vocab_parallel_logits_ptr[k]);
                row_item = compute_mean_log(val, cur_row_max, cur_row_exp_sum);
                row_log_sum += row_item;
            }
            mean_log_probs_ptr[rowIdx] = row_log_sum / n_dim;
        }
}


float __device__ __forceinline__ compute_exp_bwd_smooth(float row, float logits_max, float sum_exp_logits, 
    size_t i, int masked_target_1d, float softmax_update, float label_smoothing, 
    float smoothing, float average_grad, float grad_output) {
        row = __expf(row - logits_max);
        row /= sum_exp_logits;
        if (i == (size_t)masked_target_1d) { // i == masked_target_1d
            row = row - softmax_update;
        }
        if (label_smoothing > 0) {
            row -= smoothing * average_grad;
        }
        row = row * grad_output;
        return row;
}

template<typename dtype, int BlockSize>
void __global__  CrossEntropyBwdKernel(dtype* grad_input_ptr, // grad_input_ptr as output [4096, 256k]
                                       float * grad_output_ptr, //[4096]
                                       dtype* input_ptr,//[4096, 256k]
                                       float * target_mask_ptr,//[4096]
                                       int * masked_target_1d_ptr, //[4096]
                                       float* logits_max_ptr,//[4096]
                                       float* sum_exp_logits_ptr, //[4096]
                                       size_t n_dim,
                                       float label_smoothing,
                                       int vocab_size) {
        size_t rowIdx = blockIdx.x;
        size_t tid = threadIdx.x;
        if(tid >= n_dim) return;
        size_t cur_input_ptr_begin = rowIdx * n_dim;

        float grad_output = grad_output_ptr[rowIdx];
        float target_mask = target_mask_ptr[rowIdx];
        int masked_target_1d = masked_target_1d_ptr[rowIdx];
        float logits_max = logits_max_ptr[rowIdx];
        float sum_exp_logits = sum_exp_logits_ptr[rowIdx];        

        float softmax_update = 1.0 - (float)target_mask;
        float smoothing = 0.0;
        float average_grad = 0.0;

        if (label_smoothing > 0) {
            smoothing = label_smoothing * vocab_size / (vocab_size - 1);
            softmax_update *= (1.0 - smoothing);
            average_grad = 1.0 / vocab_size;
        }


        //size_t cur_vocab_parallel_logits_ptr_begin = rowIdx * n_dim; // 0, 1025
        size_t cur_input_ptr_end = rowIdx * n_dim + n_dim; //cur_vocab_parallel_logits_ptr_end = 1025, 2050

        size_t end_mol_num = cur_input_ptr_end % 8; //end_mol_num = 1, end_mol_num = 2
        size_t begin_mol_num = n_dim - end_mol_num; //begin_mol_num = 1024, begin_mol_num = 1023
        
        //valid range for evry row is [begin_offset, end_offset]
        size_t begin_offset = begin_mol_num % 8;//begin_offset = 0, begin_offset = 7
        size_t end_offset = n_dim - end_mol_num - 1;//end_offset = 1023, end_offset = 1022


        for (size_t i = begin_offset + tid * 8; i <= end_offset - 7; i += 8 * BlockSize) {
            int4 int4_arr = *reinterpret_cast<int4*>(&input_ptr[cur_input_ptr_begin + i]);
            dtype* bf_16_p = reinterpret_cast<dtype*>(&int4_arr);
            #pragma unroll
            for (int k = 0; k < 8; k ++) {
                dtype data_bf16 = bf_16_p[k];
                float data_fp32 = float(data_bf16); //convert to float

                data_fp32 = compute_exp_bwd_smooth(data_fp32, logits_max, sum_exp_logits, i + k, masked_target_1d, 
                     softmax_update, label_smoothing, smoothing, average_grad, grad_output);

                dtype row_bf16 = __float2bfloat16(data_fp32);
                grad_input_ptr[cur_input_ptr_begin + i + k] = row_bf16;
            }
        }

        if(threadIdx.x == 0) {
            #pragma unroll
            for (size_t k = cur_input_ptr_begin; k < (cur_input_ptr_begin + begin_offset); k++) {
                float val = float(input_ptr[k]);
                val = compute_exp_bwd_smooth(val, logits_max, sum_exp_logits, k - cur_input_ptr_begin, masked_target_1d, 
                    softmax_update, label_smoothing, smoothing, average_grad, grad_output);

                dtype row_bf16 = __float2bfloat16(val);
                grad_input_ptr[k] = row_bf16;
            }
            #pragma unroll
            for (size_t k = cur_input_ptr_begin + end_offset + 1; k < cur_input_ptr_end; k ++) {
                float val = float(input_ptr[k]);
                val = compute_exp_bwd_smooth(val, logits_max, sum_exp_logits, k - cur_input_ptr_begin, masked_target_1d, 
                    softmax_update, label_smoothing, smoothing, average_grad, grad_output);

                dtype row_bf16 = __float2bfloat16(val);
                grad_input_ptr[k] = row_bf16;
            }
        }
}


at::Tensor cross_entropy_forward_sum_exp(const at::Tensor &vocab_parallel_logits_ptr,
    const at::Tensor &logits_max_ptr
) {
    NVTE_CHECK(vocab_parallel_logits_ptr.scalar_type() == at::ScalarType::BFloat16);
    NVTE_CHECK(logits_max_ptr.scalar_type() == at::ScalarType::Float);
    NVTE_CHECK(vocab_parallel_logits_ptr.dim() == 3);
    NVTE_CHECK(logits_max_ptr.dim() == 2);

    size_t rows =  vocab_parallel_logits_ptr.size(0) * vocab_parallel_logits_ptr.size(1);
    size_t cols =  vocab_parallel_logits_ptr.size(2);

    size_t logits_max_rows = logits_max_ptr.size(0) * logits_max_ptr.size(1);
    NVTE_CHECK(rows == logits_max_rows);

    std::vector<int64_t> shape = {vocab_parallel_logits_ptr.size(0), vocab_parallel_logits_ptr.size(1)}; //shape same with logits_max_ptr
    at::Tensor sum_exp_logits_ptr = at::zeros(shape, at::CUDA(at::ScalarType::Float));

    size_t block = 128;
    size_t grid = rows; //one block is responsible one row

    CrossEntropyFwdSumExpKernel<at::BFloat16, 128><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(sum_exp_logits_ptr.data_ptr<float>(),
                                                                                                        vocab_parallel_logits_ptr.data_ptr<at::BFloat16>(), 
                                                                                                        logits_max_ptr.data_ptr<float>(), 
                                                                                                        cols);
    return sum_exp_logits_ptr;
}


at::Tensor cross_entropy_fwd_mean_log(const at::Tensor &vocab_parallel_logits_ptr,
    const at::Tensor &logits_max_ptr,
    const at::Tensor &sum_exp_logits_ptr
) {
    NVTE_CHECK(vocab_parallel_logits_ptr.scalar_type() == at::ScalarType::BFloat16);
    NVTE_CHECK(logits_max_ptr.scalar_type() == at::ScalarType::Float);
    NVTE_CHECK(sum_exp_logits_ptr.scalar_type() == at::ScalarType::Float);
    NVTE_CHECK(vocab_parallel_logits_ptr.dim() == 3);
    NVTE_CHECK(logits_max_ptr.dim() == 2);
    NVTE_CHECK(sum_exp_logits_ptr.dim() == 2);

    size_t rows =  vocab_parallel_logits_ptr.size(0) * vocab_parallel_logits_ptr.size(1);
    size_t cols =  vocab_parallel_logits_ptr.size(2);

    size_t logits_max_rows = logits_max_ptr.size(0) * logits_max_ptr.size(1);
    size_t sum_exp_logits_rows = sum_exp_logits_ptr.size(0) * sum_exp_logits_ptr.size(1);
    NVTE_CHECK(rows == logits_max_rows);
    NVTE_CHECK(rows == sum_exp_logits_rows);

    std::vector<int64_t> shape = {vocab_parallel_logits_ptr.size(0), vocab_parallel_logits_ptr.size(1)}; //shape same with logits_max_ptr
    at::Tensor mean_log_probs_ptr = at::zeros(shape, at::CUDA(at::ScalarType::Float));

    size_t block = 128;
    size_t grid = rows; //one block is responsible one row

    CrossEntropyFwdMeanLogKernel<at::BFloat16, 128><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(mean_log_probs_ptr.data_ptr<float>(),
                                                                                                        vocab_parallel_logits_ptr.data_ptr<at::BFloat16>(), 
                                                                                                        logits_max_ptr.data_ptr<float>(),
                                                                                                        sum_exp_logits_ptr.data_ptr<float>(),
                                                                                                        cols);
    return mean_log_probs_ptr;
}

at::Tensor cross_entropy_bwd(const at::Tensor &grad_output_ptr,
    const at::Tensor &input_ptr, //vocab_parallel_logits_ptr
    const at::Tensor &target_mask_ptr,
    const at::Tensor &masked_target_1d_ptr,
    const at::Tensor &logits_max_ptr,
    const at::Tensor &sum_exp_logits_ptr,
    float label_smoothing,
    size_t vocab_size
) {
    NVTE_CHECK(input_ptr.scalar_type() == at::ScalarType::BFloat16);
    NVTE_CHECK(logits_max_ptr.scalar_type() == at::ScalarType::Float);
    NVTE_CHECK(sum_exp_logits_ptr.scalar_type() == at::ScalarType::Float);
    NVTE_CHECK(masked_target_1d_ptr.scalar_type() == at::ScalarType::Int);

    NVTE_CHECK(grad_output_ptr.dim() == 2); //TODO need check if this is 2
    NVTE_CHECK(input_ptr.dim() == 3);
    NVTE_CHECK(target_mask_ptr.dim() == 2);//TODO need check if this is 2
    NVTE_CHECK(masked_target_1d_ptr.dim() == 1);//TODO need check if this is 2
    NVTE_CHECK(logits_max_ptr.dim() == 2);
    NVTE_CHECK(sum_exp_logits_ptr.dim() == 2);

    size_t rows =  input_ptr.size(0) * input_ptr.size(1);
    size_t cols =  input_ptr.size(2);

    std::vector<int64_t> shape = {input_ptr.size(0), input_ptr.size(1), input_ptr.size(2)}; //shape same with logits_max_ptr
    at::Tensor grad_input_ptr = at::zeros(shape, at::CUDA(at::ScalarType::BFloat16));

    size_t block = 128;
    size_t grid = rows; //one block is responsible one row
    //CrossEntropyBwdKernel
    CrossEntropyBwdKernel<at::BFloat16, 128><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input_ptr.data_ptr<at::BFloat16>(),
                                                                                                    grad_output_ptr.data_ptr<float>(), 
                                                                                                    input_ptr.data_ptr<at::BFloat16>(),
                                                                                                    target_mask_ptr.data_ptr<float>(), //TODO is float type ?
                                                                                                    masked_target_1d_ptr.data_ptr<int>(),//TODO is int type ?
                                                                                                    logits_max_ptr.data_ptr<float>(),
                                                                                                    sum_exp_logits_ptr.data_ptr<float>(),
                                                                                                    cols,
                                                                                                    label_smoothing,
                                                                                                    vocab_size);
    return grad_input_ptr;
}