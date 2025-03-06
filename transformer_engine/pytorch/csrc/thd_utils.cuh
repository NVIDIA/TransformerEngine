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

// format of out and lse, ignoring d as it’s always the last dimension.
enum QKVFormat { SBH, BSH, BHS, HBS, TH, HT };

template <int n>
struct TensorList {
  void *addresses_out[n];
  void *addresses_lse[n];
  int start_tensor_this_launch;
};

// describe tensor format for simplified computation.
template <QKVFormat format>
struct TensorFormat {
  // store the bsht order for simplified computation, where bsht corresponds to 0, 1, 2, 3, and store_format[3] marks whether bs is fused into t
  // For example, for the SBH format, the values of store_format are {1, 0, 2, 0}; for the TH format, the values of store_format are {3, 2, any-value, 1}
  int8_t store_format[4];
  int *cu_seqlens_s;
  // size of tensor, b s h t
  int size[4];
  __forceinline__ __device__ TensorFormat(int size_kernel[4], int *cu_seqlens = nullptr) {
    for (int i = 0; i < 4; i++) {
      size[i] = size_kernel[i];
    }
    // Initialize store_format based on the format.
    if constexpr (format == QKVFormat::TH) {
      cu_seqlens_s = cu_seqlens;
      store_format[0] = 3;
      store_format[1] = 2;
      store_format[3] = 1;
    } else if constexpr (format == QKVFormat::HT) {
      cu_seqlens_s = cu_seqlens;
      store_format[0] = 2;
      store_format[1] = 3;
      store_format[3] = 1;
    } else if constexpr (format == QKVFormat::SBH) {
      store_format[0] = 1;
      store_format[1] = 0;
      store_format[2] = 2;
      store_format[3] = 0;
    } else if constexpr (format == QKVFormat::HBS) {
      store_format[0] = 2;
      store_format[1] = 0;
      store_format[2] = 1;
      store_format[3] = 0;
    } else if constexpr (format == QKVFormat::BSH) {
      store_format[0] = 0;
      store_format[1] = 1;
      store_format[2] = 2;
      store_format[3] = 0;
    } else if constexpr (format == QKVFormat::BHS) {
      store_format[0] = 0;
      store_format[1] = 2;
      store_format[2] = 1;
      store_format[3] = 0;
    }
  }

  // calculate address according to index
  __forceinline__ __device__ int compute_address(int id[4]) {
    int address;
    if (store_format[3] == 1) {
      address = id[store_format[0]] * size[store_format[1]] + id[store_format[1]];
    } else {
      address = id[store_format[0]] * size[store_format[1]] + id[store_format[1]];
      address = address * size[store_format[2]] + id[store_format[2]];
    }
    return address;
  }

  // compute half right index
  __forceinline__ __device__ void compute_half_right(int id[4]) {
    if constexpr (format == QKVFormat::TH) {
      id[1] -= (cu_seqlens_s[id[0] + 1] - cu_seqlens_s[id[0]]) / 2;
      id[3] -= cu_seqlens_s[id[0] + 1] / 2;
    } else if constexpr (format == QKVFormat::BSH || format == QKVFormat::SBH) {
      id[1] -= size[1] / 2;
    }
  }
};

template <typename dtype, int tile_size, bool causal, QKVFormat out_format, QKVFormat lse_format,
          int max_tensors>
__global__ void fused_out_correction_kernel(dtype *out, TensorList<max_tensors> tensors, float *lse,
                                            int *cu_seqlens, int batch, int num_heads,
                                            int dim_per_head, int lse_seqlen, int cp_size, int rank,
                                            int start) {
  extern __shared__ int cu_seqlens_s[];
  int full_num;
  int num_total_tokens;

  if constexpr (out_format == QKVFormat::TH) {
    for (int i = threadIdx.x; i <= batch; i += blockDim.x) {
      cu_seqlens_s[i] = cu_seqlens[i];
    }
    __syncthreads();
    num_total_tokens = cu_seqlens_s[batch];
  } else if constexpr (out_format == QKVFormat::SBH || out_format == QKVFormat::BSH) {
    num_total_tokens = lse_seqlen * batch;
  }

  if constexpr (causal) {
    full_num = min(start + tensors.start_tensor_this_launch, max(rank + 1, start));
  } else {
    full_num = start + tensors.start_tensor_this_launch;
  }

  int size[4] = {batch, lse_seqlen, num_heads, lse_seqlen};
  // Since the formats of out and lse are often different, create two TensorFormat objects to calculate the address
  TensorFormat<out_format> out_full(size, cu_seqlens_s);
  TensorFormat<lse_format> lse_full(size);

  int tile_id = (blockIdx.x * blockDim.x + threadIdx.x) / tile_size;
  int lane_id = threadIdx.x % tile_size;
  int num_tiles = (blockDim.x * gridDim.x) / tile_size;
  int num_loops_per_head = dim_per_head * sizeof(dtype) / sizeof(float4);

  size_t idx_out_full, idx_lse_full, idx_out_half, idx_lse_half;

  for (int token_id = tile_id; token_id < num_total_tokens; token_id += num_tiles) {
    int head_id = blockIdx.y;
    int id[4];
    if constexpr (out_format == QKVFormat::TH) {
      id[0] = binary_search(token_id, cu_seqlens_s, batch + 1);
      id[1] = token_id - cu_seqlens_s[id[0]];
    } else if constexpr (out_format == QKVFormat::BSH) {
      id[0] = token_id / lse_seqlen;
      id[1] = token_id - id[0] * lse_seqlen;
    } else if constexpr (out_format == QKVFormat::SBH) {
      id[1] = token_id / batch;
      id[0] = token_id - id[1] * batch;
    }
    id[2] = head_id;
    id[3] = token_id;

    // calculate the address using the index
    idx_out_full = out_full.compute_address(id);
    idx_lse_full = lse_full.compute_address(id);

    dtype *cur_out = out + idx_out_full * dim_per_head;
    float lse_temp = lse[idx_lse_full];

    int end = full_num;

    // The number of times the current thread participates in the computation is determined by start and end
    // If a causal mask is applied, the current thread will not participate in the full computation if id[0] < 0
    if (start + tensors.start_tensor_this_launch > full_num) {
      out_full.compute_half_right(id);
      if (id[1] >= 0) {
        int size_half[4] = {batch, lse_seqlen / 2, num_heads, lse_seqlen / 2};
        TensorFormat<out_format> out_half(size_half);
        TensorFormat<lse_format> lse_half(size_half);
        idx_out_half = out_half.compute_address(id);
        idx_lse_half = lse_half.compute_address(id);
        end = start + tensors.start_tensor_this_launch;
      }
    }

    for (int j = lane_id; j < num_loops_per_head; j += tile_size) {
      float4 data = reinterpret_cast<float4 *>(cur_out)[j];
      dtype *p = reinterpret_cast<dtype *>(&data);

      for (int i = start; i < end; i++) {
        size_t idx_out;
        size_t idx_lse;
        if (causal && id[1] >= 0 && i > rank) {
          idx_out = idx_out_half;
          idx_lse = idx_lse_half;
        } else {
          idx_out = idx_out_full;
          idx_lse = idx_lse_full;
        }
        dtype *cur_out_per_step =
            reinterpret_cast<dtype *>(tensors.addresses_out[i]) + idx_out * dim_per_head;
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
