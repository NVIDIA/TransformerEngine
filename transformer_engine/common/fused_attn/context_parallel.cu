/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace context_parallel {

struct LseCorrectionFunctor {
  __forceinline__ __device__ static void run(float *lse, float *half_lse, size_t idx,
                                             size_t half_idx) {
    float val = lse[idx];
    float val_per_step = half_lse[half_idx];
    float max_scale = max(val, val_per_step);
    float min_scale = min(val, val_per_step);
    lse[idx] = max_scale + log1pf(expf(min_scale - max_scale));
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
      p_[i] = p_[i] + p[i];
    }

    reinterpret_cast<float4 *>(token)[idx] = d_;
  }
};

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

template <bool lse_packed, typename Functor>
__global__ void thd_lse_kernel(float *lse, float *half_lse, int *cu_seqlens, int batch,
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
  int batch_size_, seq_len_, num_heads_, dim_per_head_, num_total_tokens_;
  int *half_cu_seqlens_;

  __forceinline__ __device__ QKVIndexCalculatorBase(int batch_size, int seq_len, int num_heads,
                                                    int dim_per_head, int num_total_tokens,
                                                    int *half_cu_seqlens)
      : batch_size_(batch_size),
        seq_len_(seq_len),
        num_heads_(num_heads),
        dim_per_head_(dim_per_head),
        num_total_tokens_(num_total_tokens),
        half_cu_seqlens_(half_cu_seqlens) {}
};

template <NVTE_QKV_Format format>
struct QKVIndexCalculator;

template <>
struct QKVIndexCalculator<NVTE_QKV_Format::NVTE_SBHD> : QKVIndexCalculatorBase {
  __forceinline__ __device__ QKVIndexCalculator(int batch_size, int seq_len, int num_heads,
                                                int dim_per_head, int num_total_tokens,
                                                int *half_cu_seqlens)
      : QKVIndexCalculatorBase(batch_size, seq_len, num_heads, dim_per_head, num_total_tokens,
                               half_cu_seqlens) {}

  // We design the thread's lowest-level traversal to follow the sequence dimension (since LSE is in BHS or HT format). This ensures coalesced memory access when reading the LSE tensor. Consequently, the SBHD layout's computation logic for seq_id and token_id remains identical to BSHD - both patterns make the underlying threads expand along the sequence dimension. This design preserves computational correctness while optimizing memory performance.
  __forceinline__ __device__ int compute_seq_id(int flat_token_id) {
    int half_seq_len = seq_len_ / 2;
    return flat_token_id / half_seq_len;
  }

  // When blockIdx.z == 1, the computation processes the second half of the sequence. Therefore, we need to apply an address offset equal to half of the seq_len_ length.
  __forceinline__ __device__ int compute_token_id(int flat_token_id, int seq_id) {
    int half_seq_len = seq_len_ / 2;
    return flat_token_id % half_seq_len + blockIdx.z * half_seq_len;
  }

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int token_id, int head_id) {
    int offset = token_id * batch_size_ * num_heads_ + seq_id * num_heads_ + head_id;
    offset *= dim_per_head_;
    return offset;
  }

  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int token_id, int head_id) {
    int half_seq_len = seq_len_ / 2;
    int half_token_id = token_id - half_seq_len;
    int offset = half_token_id * batch_size_ * num_heads_ + seq_id * num_heads_ + head_id;
    offset *= dim_per_head_;
    return offset;
  }
};

template <>
struct QKVIndexCalculator<NVTE_QKV_Format::NVTE_BSHD> : QKVIndexCalculatorBase {
  __forceinline__ __device__ QKVIndexCalculator(int batch_size, int seq_len, int num_heads,
                                                int dim_per_head, int num_total_tokens,
                                                int *half_cu_seqlens)
      : QKVIndexCalculatorBase(batch_size, seq_len, num_heads, dim_per_head, num_total_tokens,
                               half_cu_seqlens) {}

  __forceinline__ __device__ int compute_seq_id(int flat_token_id) {
    int half_seq_len = seq_len_ / 2;
    return flat_token_id / half_seq_len;
  }

  __forceinline__ __device__ int compute_token_id(int flat_token_id, int seq_id) {
    int half_seq_len = seq_len_ / 2;
    return flat_token_id % half_seq_len + blockIdx.z * half_seq_len;
  }

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int token_id, int head_id) {
    int offset = seq_id * seq_len_ * num_heads_ + token_id * num_heads_ + head_id;
    offset *= dim_per_head_;
    return offset;
  }

  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int token_id, int head_id) {
    int half_seq_len = seq_len_ / 2;
    int half_token_id = token_id - half_seq_len;
    int offset = seq_id * half_seq_len * num_heads_ + half_token_id * num_heads_ + head_id;
    offset *= dim_per_head_;
    return offset;
  }
};

template <>
struct QKVIndexCalculator<NVTE_QKV_Format::NVTE_THD> : QKVIndexCalculatorBase {
  __forceinline__ __device__ QKVIndexCalculator(int batch_size, int seq_len, int num_heads,
                                                int dim_per_head, int num_total_tokens,
                                                int *half_cu_seqlens)
      : QKVIndexCalculatorBase(batch_size, seq_len, num_heads, dim_per_head, num_total_tokens,
                               half_cu_seqlens) {}

  __forceinline__ __device__ int compute_seq_id(int flat_token_id) {
    return binary_search(flat_token_id, half_cu_seqlens_, batch_size_ + 1);
  }

  __forceinline__ __device__ int compute_token_id(int flat_token_id, int seq_id) {
    bool is_padding = (flat_token_id >= half_cu_seqlens_[batch_size_]);
    int half_seq_len = is_padding ? (num_total_tokens_ / 2 - half_cu_seqlens_[batch_size_])
                                  : (half_cu_seqlens_[seq_id + 1] - half_cu_seqlens_[seq_id]);

    return flat_token_id - half_cu_seqlens_[seq_id] + blockIdx.z * half_seq_len;
  }

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int token_id, int head_id) {
    int flat_token_id = half_cu_seqlens_[seq_id] * 2 + token_id;
    int offset = flat_token_id * num_heads_ + head_id;
    offset *= dim_per_head_;
    return offset;
  }

  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int token_id, int head_id) {
    int flat_token_id = half_cu_seqlens_[seq_id] * 2 + token_id;
    int half_flat_token_id = flat_token_id - half_cu_seqlens_[seq_id + 1];
    int offset = half_flat_token_id * num_heads_ + head_id;
    offset *= dim_per_head_;
    return offset;
  }
};

// describe lse tensor format for simplified computation.
struct LseIndexCalculatorBase {
  int batch_size_, seq_len_, num_heads_, num_total_tokens_;
  int *half_cu_seqlens_;

  __forceinline__ __device__ LseIndexCalculatorBase(int batch_size, int seq_len, int num_heads,
                                                    int num_total_tokens, int *half_cu_seqlens)
      : batch_size_(batch_size),
        seq_len_(seq_len),
        num_heads_(num_heads),
        num_total_tokens_(num_total_tokens),
        half_cu_seqlens_(half_cu_seqlens) {}
};

template <NVTE_QKV_Format out_format, bool softmax_lse_in_packed_format>
struct LseIndexCalculator : LseIndexCalculatorBase {
  /// When the pack format is not employed, the shape of lse is BHS
  __forceinline__ __device__ LseIndexCalculator(int batch_size, int seq_len, int num_heads,
                                                int num_total_tokens, int *half_cu_seqlens)
      : LseIndexCalculatorBase(batch_size, seq_len, num_heads, num_total_tokens, half_cu_seqlens) {}

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int token_id, int head_id) {
    int offset = seq_id * num_heads_ * seq_len_ + head_id * seq_len_ + token_id;
    return offset;
  }
  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int token_id, int head_id) {
    int half_seq_len = seq_len_ / 2;
    int half_token_id;
    if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
      half_token_id = token_id - (half_cu_seqlens_[seq_id + 1] - half_cu_seqlens_[seq_id]);
    } else {
      half_token_id = token_id - half_seq_len;
    }
    int offset = seq_id * num_heads_ * half_seq_len + head_id * half_seq_len + half_token_id;
    return offset;
  }
};

template <>
struct LseIndexCalculator<NVTE_QKV_Format::NVTE_THD, true> : LseIndexCalculatorBase {
  /// When the pack format is employed, the shape of lse is HT
  __forceinline__ __device__ LseIndexCalculator(int BatchSize, int SeqLen, int NumHeads,
                                                int NumTotalTokens, int *CuSeqlens)
      : LseIndexCalculatorBase(BatchSize, SeqLen, NumHeads, NumTotalTokens, CuSeqlens) {}

  __forceinline__ __device__ int compute_full_tensor_offset(int seq_id, int token_id, int head_id) {
    int flat_token_id = token_id + half_cu_seqlens_[seq_id] * 2;
    int offset = head_id * num_total_tokens_ + flat_token_id;
    return offset;
  }
  __forceinline__ __device__ int compute_half_tensor_offset(int seq_id, int token_id, int head_id) {
    int flat_token_id = token_id + half_cu_seqlens_[seq_id] * 2;
    int half_flat_token_id = flat_token_id - half_cu_seqlens_[seq_id + 1];
    int num_half_total_tokens = num_total_tokens_ / 2;
    int offset = head_id * num_half_total_tokens + half_flat_token_id;
    return offset;
  }
};

template <typename dtype, int tile_size, bool causal, NVTE_QKV_Format out_format,
          bool softmax_lse_in_packed_format, int max_tensors, int block>
__global__ void fused_out_correction_kernel(dtype *out, TensorList<max_tensors> tensors, float *lse,
                                            int *cu_seqlens, int batch, int num_heads,
                                            int dim_per_head, int lse_seqlen, int num_total_tokens,
                                            int cp_size, int rank, int start) {
  extern __shared__ int cu_seqlens_s[];

  constexpr int num_lse_per_block = block / tile_size;

  // Preload all lse elements needed by this block into shared memory, must first determine the memory addresses of LSE elements.
  __shared__ int lse_half_idx[num_lse_per_block];
  __shared__ int lse_full_idx[num_lse_per_block];

  // Cache lse values in shared memory
  __shared__ float lse_temp[num_lse_per_block];

  // Cache lse_per_step values in shared memory
  __shared__ float lse_temp_per_step[num_lse_per_block * max_tensors];

  int full_compute_tensor_end;
  int num_total_valid_tokens;  // Number of total tokens actually involved in the computation

  if constexpr (out_format == NVTE_QKV_Format::NVTE_THD) {
    for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
      cu_seqlens_s[i] = cu_seqlens[i] / 2;
    }
    __syncthreads();
    num_total_valid_tokens = cu_seqlens_s[batch] * 2;
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
                                                num_total_tokens, cu_seqlens_s);
  LseIndexCalculator<out_format, softmax_lse_in_packed_format> lse_calculator(
      batch, lse_seqlen, num_heads, num_total_tokens, cu_seqlens_s);

  int tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size;
  int lane_id = threadIdx.x % tile_size;
  int num_tiles = (blockDim.x * gridDim.x) / tile_size;
  int num_loops_per_head = dim_per_head * sizeof(dtype) / sizeof(float4);

  for (int token_id = tile_id; token_id < num_total_tokens / 2; token_id += num_tiles) {
    int seq_id = out_calculator.compute_seq_id(token_id);
    int local_token_id = out_calculator.compute_token_id(token_id, seq_id);
    int head_id = blockIdx.y;

    size_t idx_out_full =
        out_calculator.compute_full_tensor_offset(seq_id, local_token_id, head_id);
    size_t idx_lse_full =
        lse_calculator.compute_full_tensor_offset(seq_id, local_token_id, head_id);

    size_t idx_out_half, idx_lse_half;

    // start and end define the range of tensors to compute.
    int end = full_compute_tensor_end;
    bool is_second_half = (blockIdx.z == 1);
    if (start + tensors.num_tensors_this_launch > full_compute_tensor_end && is_second_half) {
      // If the half part needs to be computed, end must be reassigned.
      end = start + tensors.num_tensors_this_launch;
      idx_out_half = out_calculator.compute_half_tensor_offset(seq_id, local_token_id, head_id);
      idx_lse_half = lse_calculator.compute_half_tensor_offset(seq_id, local_token_id, head_id);
    }

    if (lane_id == 0) {
      lse_half_idx[threadIdx.x / tile_size] = idx_lse_half;
      lse_full_idx[threadIdx.x / tile_size] = idx_lse_full;
    }

    dtype *cur_out = out + idx_out_full;
    if (token_id >= num_total_valid_tokens / 2) {
      // padding zeros
      for (int j = lane_id; j < num_loops_per_head; j += tile_size) {
        float4 data = {0.0f, 0.0f, 0.0f, 0.0f};
        reinterpret_cast<float4 *>(cur_out)[j] = data;
      }
      continue;
    }

    __syncthreads();

    /// load lse and lse_per_step into shared memory
    for (int i = threadIdx.x; i < num_lse_per_block; i += blockDim.x) {
      lse_temp[i] = lse[lse_full_idx[i]];
    }

    for (int i = threadIdx.x; i < (end - start) * num_lse_per_block; i += blockDim.x) {
      int tensor_id = start + threadIdx.x / num_lse_per_block;
      if (causal && is_second_half && tensor_id > rank) {
        lse_temp_per_step[i] = reinterpret_cast<float *>(
            tensors.addresses_lse[tensor_id])[lse_half_idx[i % num_lse_per_block]];
      } else {
        lse_temp_per_step[i] = reinterpret_cast<float *>(
            tensors.addresses_lse[tensor_id])[lse_full_idx[i % num_lse_per_block]];
      }
    }

    __syncthreads();

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
            exp(lse_temp_per_step[(i - start) * num_lse_per_block + threadIdx.x / tile_size] -
                lse_temp[threadIdx.x / tile_size]);
        dtype *p_per_step = reinterpret_cast<dtype *>(&data_per_step);
        for (int k = 0; k < sizeof(float4) / sizeof(dtype); k++) {
          p[k] += (p_per_step[k] == static_cast<dtype>(0)
                       ? 0
                       : static_cast<float>(p_per_step[k]) * lse_corrected_exp);
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

/***************************************************************************************************
 * Support THD format for Context Parallel: Read the half of a THD tensor
 **************************************************************************************************/

void thd_read_half_tensor(const Tensor &tensor, const Tensor &cu_seqlens, Tensor &half,
                          int half_idx, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(tensor.dim() == 3 || tensor.dim() == 4);
  NVTE_CHECK(cu_seqlens.dtype() == DType::kInt32);

  auto cu_seqlens_shape = cu_seqlens.shape();
  auto tensor_shape = tensor.shape();

  NVTE_CHECK(cu_seqlens.dim() == 1);
  NVTE_CHECK(cu_seqlens_shape[0] >= 2);

  // Shapes of q and dq are [t, h, d], so the dimension of "t" is 0
  // Shapes of kv and dkv are [2, t, h, d], so the dimension of "t" is 1
  int seq_dim = tensor.dim() == 3 ? 0 : 1;

  int batch = cu_seqlens_shape[0] - 1;
  int num_heads = tensor_shape[seq_dim + 1];
  int dim_per_head = tensor_shape[seq_dim + 2];
  int hidden_size_in_bytes = (num_heads * dim_per_head * typeToNumBits(tensor.dtype())) / 8;

  // For 128-bits load/store
  NVTE_CHECK(hidden_size_in_bytes % 16 == 0);

  // Launch Kernel
  constexpr unsigned int block = 256;
  unsigned int grid_x = (tensor_shape[seq_dim] / 2 * 32 + block - 1) / block;
  unsigned int grid_y = 1;
  for (int i = 0; i < seq_dim; i++) {
    grid_y *= tensor_shape[i];
  }
  dim3 grid = {grid_x, grid_y};
  thd_read_half_tensor_kernel<<<grid, block, sizeof(int) * (batch + 1), stream>>>(
      half.data.dptr, tensor.data.dptr, reinterpret_cast<int *>(cu_seqlens.data.dptr), batch,
      hidden_size_in_bytes, half_idx, tensor_shape[seq_dim]);
}

/***************************************************************************************************
 * Support THD format for Context Parallel: softmax_lse related operations
 **************************************************************************************************/

void thd_second_half_lse_correction(Tensor lse, const Tensor &lse_per_step,
                                    const Tensor &cu_seqlens, bool lse_packed,
                                    cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(lse.dtype() == DType::kFloat32);
  NVTE_CHECK(lse_per_step.dtype() == DType::kFloat32);
  NVTE_CHECK(cu_seqlens.dtype() == DType::kInt32);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch, num_heads, lse_seqlen, second_half_lse_seqlen;
  auto cu_seqlens_shape = cu_seqlens.shape();
  auto lse_shape = lse.shape();
  auto lse_per_step_shape = lse_per_step.shape();

  if (lse_packed) {
    NVTE_CHECK(lse.dim() == 2);
    NVTE_CHECK(lse_per_step.dim() == 2);

    batch = cu_seqlens_shape[0] - 1;
    num_heads = lse_shape[0];
    lse_seqlen = lse_shape[1];
    second_half_lse_seqlen = lse_per_step_shape[1];

    NVTE_CHECK(lse_per_step_shape[0] == num_heads);
    NVTE_CHECK(second_half_lse_seqlen >= lse_seqlen / 2);
  } else {
    NVTE_CHECK(lse.dim() == 3);
    NVTE_CHECK(lse_per_step.dim() == 3);

    batch = lse_shape[0];
    num_heads = lse_shape[1];
    lse_seqlen = lse_shape[2];
    second_half_lse_seqlen = lse_per_step_shape[2];

    NVTE_CHECK(lse_per_step_shape[0] == batch);
    NVTE_CHECK(lse_per_step_shape[1] == num_heads);
    NVTE_CHECK(second_half_lse_seqlen == lse_seqlen / 2);
    NVTE_CHECK(cu_seqlens_shape[0] == batch + 1);
  }

  constexpr unsigned int block = 256;
  unsigned int grid_x = (lse_seqlen / 2 + block - 1) / block;
  unsigned int grid_y = num_heads;
  dim3 grid = {grid_x, grid_y};

  if (lse_packed) {
    thd_lse_kernel<true, LseCorrectionFunctor><<<grid, block, sizeof(int) * (batch + 1), stream>>>(
        reinterpret_cast<float *>(lse.data.dptr), reinterpret_cast<float *>(lse_per_step.data.dptr),
        reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, num_heads, lse_seqlen,
        second_half_lse_seqlen);
  } else {
    thd_lse_kernel<false, LseCorrectionFunctor><<<grid, block, sizeof(int) * (batch + 1), stream>>>(
        reinterpret_cast<float *>(lse.data.dptr), reinterpret_cast<float *>(lse_per_step.data.dptr),
        reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, num_heads, lse_seqlen,
        second_half_lse_seqlen);
  }
}

void thd_read_second_half_lse(const Tensor &lse, const Tensor &cu_seqlens, Tensor &half_lse,
                              bool lse_packed, int second_half_lse_seqlen, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(lse.dtype() == DType::kFloat32);
  NVTE_CHECK(cu_seqlens.dtype() == DType::kInt32);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  int batch, num_heads, lse_seqlen;

  auto cu_seqlens_shape = cu_seqlens.shape();
  auto lse_shape = lse.shape();

  if (lse_packed) {
    NVTE_CHECK(lse.dim() == 2);

    batch = cu_seqlens_shape[0] - 1;
    num_heads = lse_shape[0];
    lse_seqlen = lse_shape[1];

    NVTE_CHECK(second_half_lse_seqlen >= lse_seqlen / 2);
  } else {
    NVTE_CHECK(lse.dim() == 3);

    batch = lse_shape[0];
    num_heads = lse_shape[1];
    lse_seqlen = lse_shape[2];

    NVTE_CHECK(cu_seqlens_shape[0] == batch + 1);
    NVTE_CHECK(second_half_lse_seqlen == lse_seqlen / 2);
  }

  constexpr unsigned int block = 256;
  unsigned int grid_x = (lse_seqlen / 2 + block - 1) / block;
  unsigned int grid_y = num_heads;
  dim3 grid = {grid_x, grid_y};

  if (lse_packed) {
    thd_lse_kernel<true, ReadLseFunctor><<<grid, block, sizeof(int) * (batch + 1), stream>>>(
        reinterpret_cast<float *>(lse.data.dptr), reinterpret_cast<float *>(half_lse.data.dptr),
        reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, num_heads, lse_seqlen,
        second_half_lse_seqlen);
  } else {
    thd_lse_kernel<false, ReadLseFunctor><<<grid, block, sizeof(int) * (batch + 1), stream>>>(
        reinterpret_cast<float *>(lse.data.dptr), reinterpret_cast<float *>(half_lse.data.dptr),
        reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, num_heads, lse_seqlen,
        second_half_lse_seqlen);
  }
}

/***************************************************************************************************
  * Support BSHD, SBHD, and THD formats for Context Parallel: Fused out correction in forward
  **************************************************************************************************/

#define DISPATCH_SBHD_BSHD_AND_THD(TYPE, LEVEL, NAME, ...)                   \
  switch (TYPE) {                                                            \
    case NVTE_QKV_Format::NVTE_SBHD: {                                       \
      constexpr NVTE_QKV_Format LEVEL = NVTE_QKV_Format::NVTE_SBHD;          \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case NVTE_QKV_Format::NVTE_BSHD: {                                       \
      constexpr NVTE_QKV_Format LEVEL = NVTE_QKV_Format::NVTE_BSHD;          \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    case NVTE_QKV_Format::NVTE_THD: {                                        \
      constexpr NVTE_QKV_Format LEVEL = NVTE_QKV_Format::NVTE_THD;           \
      __VA_ARGS__;                                                           \
      break;                                                                 \
    }                                                                        \
    default:                                                                 \
      NVTE_ERROR("only implemented for NVTE_THD, NVTE_BSHD and NVTE_SBHD "); \
  }

// #define DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...)        \
//   switch (TYPE) {                                                     \
//     case DType::kFloat32: {                                           \
//       using scalar_t_##LEVEL = float;                                 \
//       __VA_ARGS__;                                                    \
//       break;                                                          \
//     }                                                                 \
//     case DType::kFloat16: {                                           \
//       using scalar_t_##LEVEL = half;                                  \
//       __VA_ARGS__;                                                    \
//       break;                                                          \
//     }                                                                 \
//     case DType::kBFloat16: {                                          \
//       \ using scalar_t_##LEVEL = DType::kBFloat16;                    \
//       __VA_ARGS__;                                                    \
//       break;                                                          \
//     }                                                                 \
//     default:                                                          \
//       AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
//   }

template <typename dtype, bool causal>
void fused_out_correction_helper(Tensor &out, const NVTETensorPack *out_per_step, const Tensor &lse,
                                 const NVTETensorPack *lse_per_step, const Tensor &cu_seqlens,
                                 NVTE_QKV_Format qkv_format, int cp_size, int rank,
                                 bool softmax_lse_in_packed_format, cudaStream_t stream) {
  int lse_seqlen;
  int batch;
  int num_heads;
  int dim_per_head;
  int total_tokens;
  int cu_seqlens_size = 0;
  int *cu_seqlens_ptr = nullptr;

  auto out_shape = out.shape();

  if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    lse_seqlen = out_shape[0];
    batch = out_shape[1];
    num_heads = out_shape[2];
    dim_per_head = out_shape[3];
    total_tokens = lse_seqlen * batch;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_BSHD) {
    lse_seqlen = out_shape[1];
    batch = out_shape[0];
    num_heads = out_shape[2];
    dim_per_head = out_shape[3];
    total_tokens = lse_seqlen * batch;
  } else if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    total_tokens = out_shape[0];
    num_heads = out_shape[1];
    dim_per_head = out_shape[2];
    auto cu_seqlens_shape = cu_seqlens.shape();
    batch = cu_seqlens_shape[0] - 1;
    cu_seqlens_size = sizeof(int) * (batch + 1);
    cu_seqlens_ptr = reinterpret_cast<int *>(cu_seqlens.data.dptr);
    if (softmax_lse_in_packed_format) {
      lse_seqlen = total_tokens;
    } else {
      auto lse_shape = lse.shape();
      lse_seqlen = lse_shape[2];
    }
  }
  constexpr int tile = 8;
  constexpr int block = 256;
  unsigned int grid_x;

  grid_x = (static_cast<size_t>(total_tokens) * tile / 2 + block - 1) / block;
  dim3 grid = {grid_x, (unsigned int)num_heads, 2};

  constexpr int max_tensors = 64;
  TensorList<max_tensors> tensors;

  for (int i = 0; i < cp_size; i += max_tensors) {
    int num_tensors = std::min(max_tensors, cp_size - i);
    tensors.num_tensors_this_launch = num_tensors;
    for (int j = 0; j < num_tensors; j++) {
      Tensor *out_temp = convertNVTETensorCheck(out_per_step->tensors[i + j]);
      tensors.addresses_out[j] = reinterpret_cast<dtype *>(out_temp->data.dptr);
      Tensor *lse_temp = convertNVTETensorCheck(lse_per_step->tensors[i + j]);
      tensors.addresses_lse[j] = reinterpret_cast<float *>(lse_temp->data.dptr);
    }

    NVTE_CHECK(!(softmax_lse_in_packed_format == true && qkv_format != NVTE_QKV_Format::NVTE_THD),
               "Packed lse only supports THD format.");

    DISPATCH_SBHD_BSHD_AND_THD(
        qkv_format, qkv_format_type, "fused_out_correction",
        TRANSFORMER_ENGINE_SWITCH_CONDITION(
            softmax_lse_in_packed_format, bool_softmax_lse_in_packed_format,
            fused_out_correction_kernel<dtype, tile, causal, qkv_format_type,
                                        bool_softmax_lse_in_packed_format, max_tensors, block>
            <<<grid, block, cu_seqlens_size, stream>>>(
                reinterpret_cast<dtype *>(out.data.dptr), tensors,
                reinterpret_cast<float *>(lse.data.dptr), cu_seqlens_ptr, batch, num_heads,
                dim_per_head, lse_seqlen, total_tokens, cp_size, rank, i););)
  }
}

// fused out correction after qkv calculation
void fused_out_correction(Tensor &out, const NVTETensorPack *out_per_step, const Tensor &lse,
                          const NVTETensorPack *lse_per_step, Tensor &cu_seqlens,
                          NVTE_QKV_Format qkv_format, int cp_size, int rank, bool causal,
                          bool softmax_lse_in_packed_format, cudaStream_t stream) {
  // in-place optimization: use out_per_step[0] as the final output to avoid extra allocation

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      causal, bool_causal,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          out.dtype(), dtype,
          fused_out_correction_helper<dtype, bool_causal>(out, out_per_step, lse, lse_per_step,
                                                          cu_seqlens, qkv_format, cp_size, rank,
                                                          softmax_lse_in_packed_format, stream););)
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Gradients correction in backward
 **************************************************************************************************/

template <typename dtype, typename Functor_0, typename Functor_1, int functor_idx>
static void thd_grad_correction_helper(Tensor grad, const Tensor &grad_per_step,
                                       const Tensor &cu_seqlens, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(grad.dim() == 3 || grad.dim() == 4);
  NVTE_CHECK(cu_seqlens.dtype() == DType::kInt32);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  auto grad_shape = grad.shape();
  auto cu_seqlens_shape = cu_seqlens.shape();
  auto grad_per_step_shape = grad_per_step.shape();

  // Shape of dq is [t, h, d], so the dimension of "t" is 0
  // Shape of dkv is [2, t, h, d], so the dimension of "t" is 1
  int seq_dim = grad.dim() == 3 ? 0 : 1;

  int total_tokens = grad_shape[seq_dim];
  int num_heads = grad_shape[seq_dim + 1];
  int dim_per_head = grad_shape[seq_dim + 2];
  int batch = cu_seqlens_shape[0] - 1;

  if constexpr (functor_idx < 2) {
    NVTE_CHECK(grad_per_step_shape[seq_dim] == total_tokens / 2);
  } else {
    NVTE_CHECK(grad_per_step_shape[seq_dim] == total_tokens);
  }
  NVTE_CHECK(grad_per_step_shape[seq_dim + 1] == num_heads);
  NVTE_CHECK(grad_per_step_shape[seq_dim + 2] == dim_per_head);

  size_t hidden_size = num_heads * dim_per_head;
  NVTE_CHECK(((hidden_size * typeToNumBits(grad.dtype())) / 8) % 16 == 0);

  constexpr unsigned int block = 256;
  unsigned int grid_x;
  if constexpr (functor_idx < 2) {
    grid_x = (total_tokens / 2 * 32 + block - 1) / block;
  } else {
    grid_x = (total_tokens * 32 + block - 1) / block;
  }
  unsigned int grid_y = 1;
  for (int i = 0; i < seq_dim; i++) {
    grid_y *= grad_shape[i];
  }
  dim3 grid = {grid_x, grid_y};

  thd_grad_correction_kernel<dtype, Functor_0, Functor_1, functor_idx, 32>
      <<<grid, block, sizeof(int) * (batch + 1), stream>>>(
          reinterpret_cast<dtype *>(grad.data.dptr),
          reinterpret_cast<dtype *>(grad_per_step.data.dptr),
          reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, hidden_size, total_tokens);
}

template <typename dtype>
static void thd_grad_dispatcher(Tensor grad, const Tensor &grad_per_step, const Tensor &cu_seqlens,
                                const std::string &first_half, const std::string &second_half,
                                cudaStream_t stream) {
  using namespace transformer_engine;
  if (first_half == "add" && second_half == "none") {
    thd_grad_correction_helper<dtype, AddFunctor<dtype>, EmptyFunctor, 0>(grad, grad_per_step,
                                                                          cu_seqlens, stream);
  } else if (first_half == "copy" && second_half == "none") {
    thd_grad_correction_helper<dtype, CopyFunctor, EmptyFunctor, 0>(grad, grad_per_step, cu_seqlens,
                                                                    stream);
  } else if (first_half == "none" && second_half == "add") {
    thd_grad_correction_helper<dtype, EmptyFunctor, AddFunctor<dtype>, 1>(grad, grad_per_step,
                                                                          cu_seqlens, stream);
  } else if (first_half == "none" && second_half == "copy") {
    thd_grad_correction_helper<dtype, EmptyFunctor, CopyFunctor, 1>(grad, grad_per_step, cu_seqlens,
                                                                    stream);
  } else if (first_half == "add" && second_half == "copy") {
    thd_grad_correction_helper<dtype, AddFunctor<dtype>, CopyFunctor, 2>(grad, grad_per_step,
                                                                         cu_seqlens, stream);
  } else if (first_half == "copy" && second_half == "add") {
    thd_grad_correction_helper<dtype, CopyFunctor, AddFunctor<dtype>, 2>(grad, grad_per_step,
                                                                         cu_seqlens, stream);
  } else {
    NVTE_ERROR("Unsupported Functor of first half and second_half\n");
  }
}

void thd_grad_correction(Tensor grad, const Tensor &grad_per_step, const Tensor &cu_seqlens,
                         const std::string &first_half, const std::string &second_half,
                         cudaStream_t stream) {
  using namespace transformer_engine;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      grad.dtype(), dtype,
      thd_grad_dispatcher<dtype>(grad, grad_per_step, cu_seqlens, first_half, second_half,
                                 stream););
}

/***************************************************************************************************
 * Support THD format for Context Parallel: Generate partitioned indices for input tokens
 **************************************************************************************************/

void thd_get_partitioned_indices(const Tensor &cu_seqlens, Tensor output, int total_tokens,
                                 int world_size, int rank, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(cu_seqlens.dtype() == DType::kInt32);
  NVTE_CHECK(cu_seqlens.dim() == 1);

  auto cu_seqlens_shape = cu_seqlens.shape();
  auto output_shape = output.shape();

  NVTE_CHECK(cu_seqlens_shape[0] >= 2);
  NVTE_CHECK(rank >= 0 && rank < world_size);
  NVTE_CHECK(world_size > 0);
  NVTE_CHECK(total_tokens > 0 && total_tokens % (world_size * 2) == 0);

  int batch = cu_seqlens_shape[0] - 1;

  constexpr unsigned int block = 256;
  unsigned int grid = (output_shape[0] + block - 1) / block;
  thd_partition_indices_kernel<<<grid, block, sizeof(int) * (batch + 1), stream>>>(
      reinterpret_cast<int *>(output.data.dptr), reinterpret_cast<int *>(cu_seqlens.data.dptr),
      batch, total_tokens, world_size, rank);
}

}  // namespace context_parallel
}  // namespace transformer_engine

void nvte_cp_thd_read_half_tensor(const NVTETensor &tensor, const NVTETensor &cu_seqlens,
                                  NVTETensor half, int half_idx, cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_read_half_tensor);
  using namespace transformer_engine;

  context_parallel::thd_read_half_tensor(*convertNVTETensorCheck(tensor),
                                         *convertNVTETensorCheck(cu_seqlens),
                                         *convertNVTETensorCheck(half), half_idx, stream);
}

void nvte_cp_thd_second_half_lse_correction(NVTETensor lse, const NVTETensor &lse_per_step,
                                            const NVTETensor &cu_seqlens, int lse_packed,
                                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_second_half_lse_correction);
  using namespace transformer_engine;

  context_parallel::thd_second_half_lse_correction(
      *convertNVTETensorCheck(lse), *convertNVTETensorCheck(lse_per_step),
      *convertNVTETensorCheck(cu_seqlens), lse_packed, stream);
}

void nvte_cp_thd_read_second_half_lse(const NVTETensor &lse, const NVTETensor &cu_seqlens,
                                      NVTETensor half_lse, int lse_packed,
                                      int second_half_lse_seqlen, cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_read_second_half_lse);
  using namespace transformer_engine;

  context_parallel::thd_read_second_half_lse(
      *convertNVTETensorCheck(lse), *convertNVTETensorCheck(cu_seqlens),
      *convertNVTETensorCheck(half_lse), lse_packed, second_half_lse_seqlen, stream);
}

void nvte_cp_fused_out_correction(NVTETensor out, const NVTETensorPack *out_per_step,
                                  const NVTETensor &lse, const NVTETensorPack *lse_per_step,
                                  const NVTETensor &cu_seqlens, NVTE_QKV_Format qkv_format,
                                  int cp_size, int rank, bool causal,
                                  bool softmax_lse_in_packed_format, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_out_correction);
  using namespace transformer_engine;

  context_parallel::fused_out_correction(*convertNVTETensorCheck(out), out_per_step,
                                         *convertNVTETensorCheck(lse), lse_per_step,
                                         *convertNVTETensorCheck(cu_seqlens), qkv_format, cp_size,
                                         rank, causal, softmax_lse_in_packed_format, stream);
}

void nvte_cp_thd_grad_correction(NVTETensor grad, const NVTETensor &grad_per_step,
                                 const NVTETensor &cu_seqlens, const char *first_half,
                                 const char *second_half, cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_grad_correction);
  using namespace transformer_engine;

  std::string first_half_str(first_half);
  std::string second_half_str(second_half);

  context_parallel::thd_grad_correction(
      *convertNVTETensorCheck(grad), *convertNVTETensorCheck(grad_per_step),
      *convertNVTETensorCheck(cu_seqlens), first_half_str, second_half_str, stream);
}

void nvte_cp_thd_get_partitioned_indices(const NVTETensor &cu_seqlens, NVTETensor output,
                                         int total_tokens, int world_size, int rank,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_get_partitioned_indices);
  using namespace transformer_engine;

  context_parallel::thd_get_partitioned_indices(*convertNVTETensorCheck(cu_seqlens),
                                                *convertNVTETensorCheck(output), total_tokens,
                                                world_size, rank, stream);
}
