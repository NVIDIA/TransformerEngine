/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_THD_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_THD_UTILS_H_

#include <cuda.h>
#include <cuda_bf16.h>

namespace transformer_engine {
namespace fused_attn {

/***************************************************************************************************
 * Support THD format for Context Parallel: Binary search an array for a target value
 **************************************************************************************************/

__forceinline__ __device__ int binary_search(int target, int *array, int len) {
  int left = 1, right = len - 1;
  while (left < right) {
    int mid = (left + right) / 2;
    if (array[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Generate partitioned indices for input tokens
 **************************************************************************************************/

__global__ void thd_partition_indices_kernel(int *output, int *cu_seqlens, int batch,
                                             int total_tokens, int world_size, int rank);

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/

__global__ void thd_read_half_tensor_kernel(void *half, void *tensor, int *cu_seqlens, int batch,
                                            int hidden_size_in_bytes, int half_idx,
                                            int dim_size_of_token);

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

struct LseCorrectionFunctor {
  __forceinline__ __device__ static void run(double *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    double val = lse[idx];
    float val_per_step = half_lse[half_idx];
    double max_scale = max(val, val_per_step);
    double min_scale = min(val, val_per_step);
    lse[idx] = max_scale + log(1.0 + exp(min_scale - max_scale));
  }
};

struct ReadLseFunctor {
  __forceinline__ __device__ static void run(float *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    half_lse[half_idx] = lse[idx];
  }
};

template <typename lse_dtype, bool lse_packed, typename Functor>
__global__ void thd_lse_kernel(lse_dtype *lse, float *half_lse, int *cu_seqlens, int batch,
                               int num_heads, int lse_seqlen, int second_half_lse_seqlen) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / 2;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  int num_total_tokens = cu_seqlens_s[batch];

  for (int token_id = tid; token_id < num_total_tokens; token_id += num_threads) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    for (int head_id = blockIdx.y; head_id < num_heads; head_id += gridDim.y) {
      size_t idx, half_idx;
      if constexpr (lse_packed) {
        idx = head_id * lse_seqlen + token_id + cu_seqlens_s[seq_id + 1];
        half_idx = head_id * second_half_lse_seqlen + token_id;
      } else {
        size_t row = static_cast<size_t>(seq_id) * num_heads + head_id;
        int col = token_id - cu_seqlens_s[seq_id];
        int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];

        idx = row * lse_seqlen + col + seq_len;
        half_idx = row * second_half_lse_seqlen + col;
      }

      Functor::run(lse, half_lse, idx, half_idx);
    }
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Out correction in forward
 **************************************************************************************************/

template <typename dtype, int only_second_half, int tile_size, bool lse_packed>
__global__ void thd_out_correction_kernel(dtype *out, dtype *out_per_step, float *lse,
                                          float *lse_per_step, int *cu_seqlens, int batch,
                                          int num_heads, int dim_per_head, int lse_seqlen,
                                          int lse_per_step_seqlen) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / (only_second_half + 1);
  }
  __syncthreads();

  int tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size;
  int lane_id = threadIdx.x % tile_size;
  int num_tiles = (blockDim.x * gridDim.x) / tile_size;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_loops_per_head = dim_per_head * sizeof(dtype) / sizeof(float4);

  for (int token_id = tile_id; token_id < num_total_tokens; token_id += num_tiles) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    for (int head_id = blockIdx.y; head_id < num_heads; head_id += gridDim.y) {
      size_t idx, idx_per_step;

      if constexpr (lse_packed) {
        idx = head_id * lse_seqlen + token_id + cu_seqlens_s[seq_id + 1] * only_second_half;
        idx_per_step = head_id * lse_per_step_seqlen + token_id;
      } else {
        size_t row = static_cast<size_t>(seq_id) * num_heads + head_id;
        int col = token_id - cu_seqlens_s[seq_id];
        int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
        idx = row * lse_seqlen + col + seq_len * only_second_half;
        idx_per_step = row * lse_per_step_seqlen + col;
      }
      float lse_corrected_exp = exp(lse_per_step[idx_per_step] - lse[idx]);

      idx = token_id + cu_seqlens_s[seq_id + 1] * only_second_half;
      idx = (idx * num_heads + head_id) * dim_per_head;
      idx_per_step = (static_cast<size_t>(token_id) * num_heads + head_id) * dim_per_head;
      dtype *cur_out = out + idx;
      dtype *cur_out_per_step = out_per_step + idx_per_step;

      for (int j = lane_id; j < num_loops_per_head; j += tile_size) {
        float4 data_per_step = reinterpret_cast<float4 *>(cur_out_per_step)[j];
        float4 data = reinterpret_cast<float4 *>(cur_out)[j];
        dtype *p_per_step = reinterpret_cast<dtype *>(&data_per_step);
        dtype *p = reinterpret_cast<dtype *>(&data);
        for (int k = 0; k < sizeof(float4) / sizeof(dtype); k++) {
          p[k] += (p_per_step[k] == 0 ? 0 : p_per_step[k] * lse_corrected_exp);
        }
        reinterpret_cast<float4 *>(cur_out)[j] = data;
      }
    }
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

struct EmptyFunctor {
  __forceinline__ __device__ static void run(void *token, void *token_per_step, int idx) {}
};

struct CopyFunctor {
  __forceinline__ __device__ static void run(void *token, void *token_per_step, int idx) {
    reinterpret_cast<float4 *>(token)[idx] = reinterpret_cast<float4 *>(token_per_step)[idx];
  }
};

template <typename dtype>
struct AddFunctor {
  __forceinline__ __device__ static void run(dtype *token, dtype *token_per_step, int idx) {
    float4 d_ = reinterpret_cast<float4 *>(token)[idx];
    dtype *p_ = reinterpret_cast<dtype *>(&d_);

    float4 d = reinterpret_cast<float4 *>(token_per_step)[idx];
    dtype *p = reinterpret_cast<dtype *>(&d);

#pragma unroll
    for (int i = 0; i < sizeof(float4) / sizeof(dtype); i++) {
      p_[i] += p[i];
    }

    reinterpret_cast<float4 *>(token)[idx] = d_;
  }
};

template <typename dtype, typename Functor_0, typename Functor_1, int functor_idx, int group_size>
__global__ void thd_grad_correction_kernel(dtype *grad, dtype *grad_per_step, int *cu_seqlens,
                                           int batch, int hidden_size, int dim_size_of_token) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    if constexpr (functor_idx < 2) {
      cu_seqlens_s[i] = cu_seqlens[i] / 2;
    } else {
      cu_seqlens_s[i] = cu_seqlens[i];
    }
  }
  __syncthreads();

  int group_id = (blockIdx.x * blockDim.x + threadIdx.x) / group_size;
  int lane_id = threadIdx.x % group_size;
  int num_groups = (blockDim.x * gridDim.x) / group_size;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_inner_loops = hidden_size * sizeof(dtype) / sizeof(float4);

  size_t offset = static_cast<size_t>(dim_size_of_token) * hidden_size;
  if constexpr (functor_idx < 2) {
    grad_per_step = grad_per_step + offset / 2 * blockIdx.y;
  } else {
    grad_per_step = grad_per_step + offset * blockIdx.y;
  }
  grad = grad + offset * blockIdx.y;

  for (int token_id = group_id; token_id < num_total_tokens; token_id += num_groups) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);

    int token_offset;
    bool is_first_half;
    if constexpr (functor_idx < 2) {
      token_offset = cu_seqlens_s[seq_id + functor_idx];
      is_first_half = (functor_idx == 0);
    } else {
      token_offset = 0;
      int len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
      is_first_half = (token_id - cu_seqlens_s[seq_id]) < (len / 2);
    }

    dtype *token = &grad[(token_id + token_offset) * static_cast<size_t>(hidden_size)];
    dtype *token_per_step = &grad_per_step[token_id * static_cast<size_t>(hidden_size)];
    for (int idx = lane_id; idx < num_inner_loops; idx += group_size) {
      if (is_first_half) {
        Functor_0::run(token, token_per_step, idx);
      } else {
        Functor_1::run(token, token_per_step, idx);
      }
    }
  }
}

}  // namespace fused_attn
}  // namespace transformer_engine

#endif
