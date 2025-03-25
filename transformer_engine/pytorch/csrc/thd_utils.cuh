/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_THD_UTILS_CUH_
#define TRANSFORMER_ENGINE_FUSED_ATTN_THD_UTILS_CUH_

#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>

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
                                             int total_tokens, int world_size, int rank) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    int seqlen = cu_seqlens[i];
    // Currently we assume that each sequence length is divisible by (world_size*2) since we have
    // to distribute each sequence evenly to different GPUs.
    assert(seqlen % (world_size * 2) == 0);
    cu_seqlens_s[i] = seqlen / world_size;
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (int token_id = tid; token_id < total_tokens / world_size; token_id += num_threads) {
    int seq_id = binary_search(token_id, cu_seqlens_s, batch + 1);
    int seq_len = cu_seqlens_s[seq_id + 1] - cu_seqlens_s[seq_id];
    int index = token_id - cu_seqlens_s[seq_id];
    int offset = index < seq_len / 2 ? rank : (world_size - 1) * 2 - rank;
    index += cu_seqlens_s[seq_id] * world_size + seq_len / 2 * offset;
    output[token_id] = index;
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/
__global__ void thd_read_half_tensor_kernel(void *half, void *tensor, int *cu_seqlens, int batch,
                                            int hidden_size_in_bytes, int half_idx,
                                            int dim_size_of_token) {
  extern __shared__ int cu_seqlens_s[];
  for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
    cu_seqlens_s[i] = cu_seqlens[i] / 2;
  }
  __syncthreads();

  int warpid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int laneid = threadIdx.x % 32;
  int num_warps = (blockDim.x * gridDim.x) / 32;
  int num_total_tokens = cu_seqlens_s[batch];
  int num_float4s_per_token = hidden_size_in_bytes / sizeof(float4);

  size_t offset = static_cast<size_t>(dim_size_of_token) * hidden_size_in_bytes;
  half = reinterpret_cast<void *>(reinterpret_cast<char *>(half) + offset / 2 * blockIdx.y);
  tensor = reinterpret_cast<void *>(reinterpret_cast<char *>(tensor) + offset * blockIdx.y);

  for (int token_id = warpid; token_id < num_total_tokens; token_id += num_warps) {
    int seqid = binary_search(token_id, cu_seqlens_s, batch + 1);

    size_t offset_in_bytes = static_cast<size_t>(token_id) * hidden_size_in_bytes;
    float4 *cur_half_token =
        reinterpret_cast<float4 *>(reinterpret_cast<char *>(half) + offset_in_bytes);

    offset_in_bytes =
        (static_cast<size_t>(token_id) + cu_seqlens_s[seqid + half_idx]) * hidden_size_in_bytes;
    float4 *cur_token =
        reinterpret_cast<float4 *>(reinterpret_cast<char *>(tensor) + offset_in_bytes);

    for (int idx = laneid; idx < num_float4s_per_token; idx += 32) {
      cur_half_token[idx] = cur_token[idx];
    }
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

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
 * Support BSHD, SBHD, and THD formats for Context Parallel: Out correction in forward
 **************************************************************************************************/

// Stores pointers to output and lse tensors for batch kernel launch.
template <int n>
struct TensorList {
  void *addresses_out[n];
  void *addresses_lse[n];
  int num_tensors_this_launch;
};

// describe QKV output tensor format for simplified computation.
struct QKVIndexCalculatorBase {
  int batch_size, seq_len, num_heads, dim_per_head;
  int *cu_seqlens;

  __forceinline__ __device__ QKVIndexCalculatorBase(int BatchSize, int SeqLen, int NumHeads,
                                                    int DimPerHead, int *CuSeqlens)
      : batch_size(BatchSize),
        seq_len(SeqLen),
        num_heads(NumHeads),
        dim_per_head(DimPerHead),
        cu_seqlens(CuSeqlens) {}

  // Computes full out tensor offset from token_id and head indices.
  __forceinline__ __device__ int compute_full_tensor_offset(int token_id, int head_id) {
    int offset = (token_id * num_heads + head_id) * dim_per_head;
    return offset;
  }
};

template <NVTE_QKV_Format format>
struct QKVIndexCalculator;

template <>
struct QKVIndexCalculator<NVTE_QKV_Format::NVTE_SBHD> : QKVIndexCalculatorBase {
  __forceinline__ __device__ QKVIndexCalculator(int BatchSize, int SeqLen, int NumHeads,
                                                int DimPerHead, int *CuSeqlens)
      : QKVIndexCalculatorBase(BatchSize, SeqLen, NumHeads, DimPerHead, CuSeqlens) {}

  __forceinline__ __device__ int compute_seq_id(int token_id) {
    return token_id % batch_size;
  }

  __forceinline__ __device__ int compute_local_token_id(int token_id, int seq_id) {
    return token_id / batch_size;
  }

  // Checks if the token is in the second half of the sequence.
  __forceinline__ __device__ bool is_second_half(int local_token_id, int seq_id) {
    return local_token_id >= seq_len / 2;
  }

  // Computes half out tensor offset from sequence, local token, and head indices.
  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int half_seq_len = seq_len / 2;
    int half_local_token_id = local_token_id - half_seq_len;
    int offset = half_local_token_id * batch_size * num_heads + seq_id * num_heads + head_id;
    offset *= dim_per_head;
    return offset;
  }
};

// describe lse tensor format for simplified computation.
template <>
struct QKVIndexCalculator<NVTE_QKV_Format::NVTE_BSHD> : QKVIndexCalculatorBase {
  __forceinline__ __device__ QKVIndexCalculator(int BatchSize, int SeqLen, int NumHeads,
                                                int DimPerHead, int *CuSeqlens)
      : QKVIndexCalculatorBase(BatchSize, SeqLen, NumHeads, DimPerHead, CuSeqlens) {}

  __forceinline__ __device__ int compute_seq_id(int token_id) {
    return token_id / seq_len;
  }

  __forceinline__ __device__ int compute_local_token_id(int token_id, int seq_id) {
    return token_id % seq_len;
  }

  __forceinline__ __device__ bool is_second_half(int local_token_id, int seq_id) {
    return local_token_id >= seq_len / 2;
  }

  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int half_seq_len = seq_len / 2;
    int half_local_token_id = local_token_id - half_seq_len;
    int offset = seq_id * half_seq_len * num_heads + half_local_token_id * num_heads + head_id;
    offset *= dim_per_head;
    return offset;
  }
};

template <>
struct QKVIndexCalculator<NVTE_QKV_Format::NVTE_THD> : QKVIndexCalculatorBase {
  __forceinline__ __device__ QKVIndexCalculator(int BatchSize, int SeqLen, int NumHeads,
                                                int DimPerHead, int *CuSeqlens)
      : QKVIndexCalculatorBase(BatchSize, SeqLen, NumHeads, DimPerHead, CuSeqlens) {}

  __forceinline__ __device__ int compute_seq_id(int token_id) {
    return binary_search(token_id, cu_seqlens, batch_size + 1);
  }

  __forceinline__ __device__ int compute_local_token_id(int token_id, int seq_id) {
    return token_id - cu_seqlens[seq_id];
  }

  __forceinline__ __device__ bool is_second_half(int local_token_id, int seq_id) {
    return local_token_id >= (cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]) / 2;
  }

  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int token_id = cu_seqlens[seq_id] + local_token_id;
    int half_token_id = token_id - cu_seqlens[seq_id + 1] / 2;
    int offset = half_token_id * num_heads + head_id;
    offset *= dim_per_head;
    return offset;
  }
};

struct LseIndexCalculatorBase {
  int batch_size, seq_len, num_heads, num_total_tokens;
  int *cu_seqlens;

  __forceinline__ __device__ LseIndexCalculatorBase(int BatchSize, int SeqLen, int NumHeads,
                                                    int NumTotalTokens, int *CuSeqlens)
      : batch_size(BatchSize),
        seq_len(SeqLen),
        num_heads(NumHeads),
        num_total_tokens(NumTotalTokens),
        cu_seqlens(CuSeqlens) {}
};

template <NVTE_QKV_Format out_format, bool softmax_lse_in_packed_format>
struct LseIndexCalculator : LseIndexCalculatorBase {
  /// When the pack format is not employed, the shape of lse is BHS
  __forceinline__ __device__ LseIndexCalculator(int BatchSize, int SeqLen, int NumHeads,
                                                int NumTotalTokens, int *CuSeqlens)
      : LseIndexCalculatorBase(BatchSize, SeqLen, NumHeads, NumTotalTokens, CuSeqlens) {}

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int offset = seq_id * num_heads * seq_len + head_id * seq_len + local_token_id;
    return offset;
  }
  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int half_seq_len = seq_len / 2;
    int half_local_token_id;
    if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
      half_local_token_id = local_token_id - (cu_seqlens[seq_id + 1] - cu_seqlens[seq_id]) / 2;
    } else {
      half_local_token_id = local_token_id - half_seq_len;
    }
    int offset = seq_id * num_heads * half_seq_len + head_id * half_seq_len + half_local_token_id;
    return offset;
  }
};

template <>
struct LseIndexCalculator<NVTE_QKV_Format::NVTE_THD, true> : LseIndexCalculatorBase {
  /// When the pack format is employed, the shape of lse is HT
  __forceinline__ __device__ LseIndexCalculator(int BatchSize, int SeqLen, int NumHeads,
                                                int NumTotalTokens, int *CuSeqlens)
      : LseIndexCalculatorBase(BatchSize, SeqLen, NumHeads, NumTotalTokens, CuSeqlens) {}

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int token_id = local_token_id + cu_seqlens[seq_id];
    int offset = head_id * num_total_tokens + token_id;
    return offset;
  }
  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int local_token_id,
                                                            int head_id) {
    int token_id = local_token_id + cu_seqlens[seq_id];
    int half_token_id = token_id - cu_seqlens[seq_id + 1] / 2;
    int num_half_total_tokens = num_total_tokens / 2;
    int offset = head_id * num_half_total_tokens + half_token_id;
    return offset;
  }
};

template <typename dtype, int tile_size, bool causal, NVTE_QKV_Format out_format,
          bool softmax_lse_in_packed_format, int max_tensors>
__global__ void fused_out_correction_kernel(dtype *out, TensorList<max_tensors> tensors, float *lse,
                                            int *cu_seqlens, int batch, int num_heads,
                                            int dim_per_head, int lse_seqlen, int num_total_tokens,
                                            int cp_size, int rank, int start) {
  extern __shared__ int cu_seqlens_s[];
  int full_compute_tensor_end;
  int num_total_valid_tokens;  // Number of total tokens actually involved in the computation

  if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
    for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
      cu_seqlens_s[i] = cu_seqlens[i];
    }
    __syncthreads();
    num_total_valid_tokens = cu_seqlens_s[batch];
  } else if constexpr (out_format == NVTE_QKV_Format::NVTE_SBHD ||
                       out_format == NVTE_QKV_Format::NVTE_BSHD) {
    num_total_valid_tokens = lse_seqlen * batch;
  }

  // Last tensor index for full tensor computation in this round.
  if constexpr (causal) {
    full_compute_tensor_end = min(start + tensors.num_tensors_this_launch, max(rank + 1, start));
  } else {
    full_compute_tensor_end = start + tensors.num_tensors_this_launch;
  }

  // It's necessary to handle out and lse differently because their formats maybe different.
  QKVIndexCalculator<out_format> out_calculator(batch, lse_seqlen, num_heads, dim_per_head,
                                                cu_seqlens_s);
  LseIndexCalculator<out_format, softmax_lse_in_packed_format> lse_calculator(
      batch, lse_seqlen, num_heads, num_total_tokens, cu_seqlens_s);

  int tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size;
  int lane_id = threadIdx.x % tile_size;
  int num_tiles = (blockDim.x * gridDim.x) / tile_size;
  int num_loops_per_head = dim_per_head * sizeof(dtype) / sizeof(float4);

  for (int token_id = tile_id; token_id < num_total_tokens; token_id += num_tiles) {
    int seq_id = out_calculator.compute_seq_id(token_id);
    int local_token_id = out_calculator.compute_local_token_id(token_id, seq_id);
    int head_id = blockIdx.y;

    size_t idx_out_full = out_calculator.compute_full_tensor_offset(token_id, head_id);
    size_t idx_out_half = out_calculator.compute_half_tensor_offset(seq_id, local_token_id, head_id);
    size_t idx_lse_full = lse_calculator.compute_full_tensor_offset(seq_id, local_token_id, head_id);
    size_t idx_lse_half = lse_calculator.compute_half_tensor_offset(seq_id, local_token_id, head_id);

    dtype *cur_out = out + idx_out_full;
    if (token_id >= num_total_valid_tokens) {
      // padding zeros
      for (int j = lane_id; j < num_loops_per_head; j += tile_size) {
        float4 data = {0.0f, 0.0f, 0.0f, 0.0f};
        reinterpret_cast<float4 *>(cur_out)[j] = data;
      }
      continue;
    }

    // start and end define the range of tensors to compute.
    int end = full_compute_tensor_end;
    bool is_second_half = out_calculator.is_second_half(local_token_id, seq_id);
    if (start + tensors.num_tensors_this_launch > full_compute_tensor_end && is_second_half) {
      // If the half part needs to be computed, end must be reassigned.
      end = start + tensors.num_tensors_this_launch;
    }

    float lse_temp = lse[idx_lse_full];

    for (int j = lane_id; j < num_loops_per_head; j += tile_size) {
      float4 data;
      if (start == 0) {
        data = {0.0f, 0.0f, 0.0f, 0.0f};
      } else {
        data = reinterpret_cast<float4 *>(cur_out)[j];
      }

      dtype *p = reinterpret_cast<dtype *>(&data);

      for (int i = start; i < end; i++) {
        size_t idx_out;
        size_t idx_lse;
        if (causal && is_second_half && i > rank) {
          idx_out = idx_out_half;
          idx_lse = idx_lse_half;
        } else {
          idx_out = idx_out_full;
          idx_lse = idx_lse_full;
        }
        dtype *cur_out_per_step = reinterpret_cast<dtype *>(tensors.addresses_out[i]) + idx_out;
        float4 data_per_step = reinterpret_cast<float4 *>(cur_out_per_step)[j];
        float lse_corrected_exp =
            exp(reinterpret_cast<float *>(tensors.addresses_lse[i])[idx_lse] - lse_temp);
        dtype *p_per_step = reinterpret_cast<dtype *>(&data_per_step);
        for (int k = 0; k < sizeof(float4) / sizeof(dtype); k++) {
          p[k] += (p_per_step[k] == 0 ? 0 : p_per_step[k] * lse_corrected_exp);
        }
      }
      reinterpret_cast<float4 *>(cur_out)[j] = data;
    }
  }
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

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
