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
      float lse_corrected_exp = expf(lse_per_step[idx_per_step] - lse[idx]);

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
          p[k] = p[k] +
                 (p_per_step[k] == static_cast<dtype>(0.f)
                      ? static_cast<dtype>(0.f)
                      : static_cast<dtype>(static_cast<float>(p_per_step[k]) * lse_corrected_exp));
        }
        reinterpret_cast<float4 *>(cur_out)[j] = data;
      }
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
  int hidden_size_in_bytes = num_heads * dim_per_head * typeToSize(tensor.dtype());

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
 * Support THD format for Context Parallel: Out correction in forward
 **************************************************************************************************/

template <typename dtype, int only_second_half>
static void thd_out_correction_helper(Tensor out, const Tensor &out_per_step, const Tensor &lse,
                                      const Tensor &lse_per_step, const Tensor &cu_seqlens,
                                      bool lse_packed, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(out.dtype() == out_per_step.dtype());
  NVTE_CHECK(lse.dtype() == DType::kFloat32);
  NVTE_CHECK(lse_per_step.dtype() == DType::kFloat32);
  NVTE_CHECK(cu_seqlens.dtype() == DType::kInt32);

  auto out_shape = out.shape();
  auto lse_shape = lse.shape();
  auto out_per_step_shape = out_per_step.shape();
  auto lse_per_step_shape = lse_per_step.shape();
  auto cu_seqlens_shape = cu_seqlens.shape();

  int total_tokens = out_shape[0];
  int num_heads = out_shape[1];
  int dim_per_head = out_shape[2];

  NVTE_CHECK(out_per_step_shape[0] == total_tokens / (only_second_half + 1));
  NVTE_CHECK(out_per_step_shape[1] == num_heads);
  NVTE_CHECK(out_per_step_shape[2] == dim_per_head);

  int batch, lse_seqlen, lse_per_step_seqlen;
  if (lse_packed) {
    batch = cu_seqlens_shape[0] - 1;
    lse_seqlen = lse_shape[1];
    lse_per_step_seqlen = lse_per_step_shape[1];

    NVTE_CHECK(lse_shape[0] == num_heads);
    NVTE_CHECK(lse_seqlen >= total_tokens);
    NVTE_CHECK(lse_per_step_shape[0] == num_heads);
    NVTE_CHECK(lse_per_step_seqlen >= lse_seqlen / (only_second_half + 1));
  } else {
    batch = lse_shape[0];
    lse_seqlen = lse_shape[2];
    lse_per_step_seqlen = lse_per_step_shape[2];

    NVTE_CHECK(lse_shape[1] == num_heads);
    NVTE_CHECK(lse_per_step_shape[0] == batch);
    NVTE_CHECK(lse_per_step_shape[1] == num_heads);
    NVTE_CHECK(lse_per_step_seqlen == lse_seqlen / (only_second_half + 1));
    NVTE_CHECK(cu_seqlens_shape[0] == batch + 1);
  }

  constexpr int tile = 16;
  constexpr int block = 512;
  unsigned int grid_x =
      (static_cast<size_t>(total_tokens) / (only_second_half + 1) * tile + block - 1) / block;
  dim3 grid = {grid_x, (unsigned int)num_heads};

  if (lse_packed) {
    thd_out_correction_kernel<dtype, only_second_half, tile, true>
        <<<grid, block, sizeof(int) * (batch + 1), stream>>>(
            reinterpret_cast<dtype *>(out.data.dptr),
            reinterpret_cast<dtype *>(out_per_step.data.dptr),
            reinterpret_cast<float *>(lse.data.dptr),
            reinterpret_cast<float *>(lse_per_step.data.dptr),
            reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, num_heads, dim_per_head,
            lse_seqlen, lse_per_step_seqlen);
  } else {
    thd_out_correction_kernel<dtype, only_second_half, tile, false>
        <<<grid, block, sizeof(int) * (batch + 1), stream>>>(
            reinterpret_cast<dtype *>(out.data.dptr),
            reinterpret_cast<dtype *>(out_per_step.data.dptr),
            reinterpret_cast<float *>(lse.data.dptr),
            reinterpret_cast<float *>(lse_per_step.data.dptr),
            reinterpret_cast<int *>(cu_seqlens.data.dptr), batch, num_heads, dim_per_head,
            lse_seqlen, lse_per_step_seqlen);
  }
}

void thd_out_correction(Tensor out, const Tensor &out_per_step, const Tensor &lse,
                        const Tensor &lse_per_step, const Tensor &cu_seqlens, bool only_second_half,
                        bool lse_packed, cudaStream_t stream) {
  using namespace transformer_engine;
  if (only_second_half) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        out.dtype(), dtype,
        thd_out_correction_helper<dtype, 1>(out, out_per_step, lse, lse_per_step, cu_seqlens,
                                            lse_packed, stream););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
        out.dtype(), dtype,
        thd_out_correction_helper<dtype, 0>(out, out_per_step, lse, lse_per_step, cu_seqlens,
                                            lse_packed, stream););
  }
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
  NVTE_CHECK((hidden_size * typeToSize(grad.dtype())) % 16 == 0);

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

  context_parallel::thd_read_half_tensor(*reinterpret_cast<Tensor *>(tensor),
                                         *reinterpret_cast<Tensor *>(cu_seqlens),
                                         *reinterpret_cast<Tensor *>(half), half_idx, stream);
}

void nvte_cp_thd_second_half_lse_correction(NVTETensor lse, const NVTETensor &lse_per_step,
                                            const NVTETensor &cu_seqlens, int lse_packed,
                                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_second_half_lse_correction);
  using namespace transformer_engine;

  context_parallel::thd_second_half_lse_correction(
      *reinterpret_cast<Tensor *>(lse), *reinterpret_cast<Tensor *>(lse_per_step),
      *reinterpret_cast<Tensor *>(cu_seqlens), lse_packed, stream);
}

void nvte_cp_thd_read_second_half_lse(const NVTETensor &lse, const NVTETensor &cu_seqlens,
                                      NVTETensor half_lse, int lse_packed,
                                      int second_half_lse_seqlen, cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_read_second_half_lse);
  using namespace transformer_engine;

  context_parallel::thd_read_second_half_lse(
      *reinterpret_cast<Tensor *>(lse), *reinterpret_cast<Tensor *>(cu_seqlens),
      *reinterpret_cast<Tensor *>(half_lse), lse_packed, second_half_lse_seqlen, stream);
}

void nvte_cp_thd_out_correction(NVTETensor out, const NVTETensor &out_per_step,
                                const NVTETensor &lse, const NVTETensor &lse_per_step,
                                const NVTETensor &cu_seqlens, int only_second_half, int lse_packed,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_out_correction);
  using namespace transformer_engine;

  context_parallel::thd_out_correction(
      *reinterpret_cast<Tensor *>(out), *reinterpret_cast<Tensor *>(out_per_step),
      *reinterpret_cast<Tensor *>(lse), *reinterpret_cast<Tensor *>(lse_per_step),
      *reinterpret_cast<Tensor *>(cu_seqlens), only_second_half, lse_packed, stream);
}

void nvte_cp_thd_grad_correction(NVTETensor grad, const NVTETensor &grad_per_step,
                                 const NVTETensor &cu_seqlens, const char *first_half,
                                 const char *second_half, cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_grad_correction);
  using namespace transformer_engine;

  std::string first_half_str(first_half);
  std::string second_half_str(second_half);

  context_parallel::thd_grad_correction(
      *reinterpret_cast<Tensor *>(grad), *reinterpret_cast<Tensor *>(grad_per_step),
      *reinterpret_cast<Tensor *>(cu_seqlens), first_half_str, second_half_str, stream);
}

void nvte_cp_thd_get_partitioned_indices(const NVTETensor &cu_seqlens, NVTETensor output,
                                         int total_tokens, int world_size, int rank,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_thd_get_partitioned_indices);
  using namespace transformer_engine;

  context_parallel::thd_get_partitioned_indices(*reinterpret_cast<Tensor *>(cu_seqlens),
                                                *reinterpret_cast<Tensor *>(output), total_tokens,
                                                world_size, rank, stream);
}
