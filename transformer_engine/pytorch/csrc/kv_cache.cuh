/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_KV_CACHE_CUH_
#define TRANSFORMER_ENGINE_FUSED_ATTN_KV_CACHE_CUH_

namespace transformer_engine {
namespace fused_attn {
template <typename scalar_t>
__global__ void reshape_q_kernel(scalar_t *new_q, scalar_t *q_buffer, int *cu_new_lens, int h_q,
                                 int d_q, int b, int max_seq_len) {
  // new_q: thd; q_buffer: bshd;
  // cu_new_lens: [b + 1]
  for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
    int num_elts = (cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx]) * h_q * d_q;
    int new_token_offset = cu_new_lens[batch_idx] * h_q * d_q;
    int cache_offset = batch_idx * max_seq_len * h_q * d_q;
    scalar_t *new_q_token = new_q + new_token_offset;
    scalar_t *q_buffer_token = q_buffer + cache_offset;
    for (int i = threadIdx.x; i < num_elts; i += blockDim.x) {
      *(q_buffer_token + i) = *(new_q_token + i);
    }
  }
}

template <typename scalar_t>
__global__ void reshape_o_kernel(scalar_t *output, scalar_t *output_buffer, int *cu_new_lens,
                                 int h_o, int d_o, int b, int max_seq_len,
                                 bool is_output_right_aligned) {
  // output: bshd; output_buffer: thd;
  // cu_new_lens: [b + 1]
  for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
    int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
    int num_elts = new_len * h_o * d_o;
    int output_offset = batch_idx * max_seq_len * h_o * d_o;
    if (is_output_right_aligned) {
      output_offset = ((batch_idx + 1) * max_seq_len - new_len) * h_o * d_o;
    }
    int output_buffer_offset = cu_new_lens[batch_idx] * h_o * d_o;
    scalar_t *output_token = output + output_offset;
    scalar_t *output_buffer_token = output_buffer + output_buffer_offset;
    for (int i = threadIdx.x; i < num_elts; i += blockDim.x) {
      *(output_buffer_token + i) = *(output_token + i);
    }
  }
}

template <typename scalar_t>
__global__ void reindex_kv_cache_kernel(scalar_t *k_cache, scalar_t *v_cache, int *batch_indices,
                                        int *cu_new_lens, int *cu_cached_lens, int h_kv, int d_k,
                                        int d_v, int b, int max_seq_len) {
  // k_cache, v_cache: bshd
  // batch_indices: [b]; cu_new_lens, cu_cached_lens: [b + 1]
  int actual_b = b;
  for (int i = 0; i < b - 1; i++) {
    if (batch_indices[i + 1] < batch_indices[i]) {
      actual_b = i + 1;
    }
  }
  for (int batch_idx = 0; batch_idx < actual_b; batch_idx++) {
    int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
    int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
    for (int token_idx = blockIdx.x; token_idx < cached_len - new_len; token_idx += gridDim.x) {
      int num_elts_k = h_kv * d_k;
      int num_elts_v = h_kv * d_v;
      int k_cache_src_offset = (batch_indices[batch_idx] * max_seq_len + token_idx) * h_kv * d_k;
      int k_cache_des_offset = (batch_idx * max_seq_len + token_idx) * h_kv * d_k;
      int v_cache_src_offset = (batch_indices[batch_idx] * max_seq_len + token_idx) * h_kv * d_v;
      int v_cache_des_offset = (batch_idx * max_seq_len + token_idx) * h_kv * d_v;
      for (int i = threadIdx.x; i < num_elts_k; i += blockDim.x) {
        *(k_cache + k_cache_des_offset + i) = *(k_cache + k_cache_src_offset + i);
      }
      for (int i = threadIdx.x; i < num_elts_v; i += blockDim.x) {
        *(v_cache + v_cache_des_offset + i) = *(v_cache + v_cache_src_offset + i);
      }
    }
  }
}

template <typename scalar_t>
__global__ void copy_to_kv_cache_kernel(scalar_t *new_k, scalar_t *new_v, scalar_t *k_cache,
                                        scalar_t *v_cache, int *page_table, int *cu_new_lens,
                                        int *cu_cached_lens, NVTE_QKV_Format qkv_format, int h_kv,
                                        int d_k, int d_v, int b, int max_ctx_len, int max_seq_len,
                                        int max_pages_per_seq, bool is_non_paged) {
  // new_k, new_v: qkv_format; k_cache, v_cache: bshd
  // cu_new_lens, cu_cached_lens: [b + 1]
  // page_table: [b, max_pages_per_seq]
  int page_size = max_seq_len / max_pages_per_seq;
  if (qkv_format == NVTE_QKV_Format::NVTE_BSHD) {
    for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
      int *page_list = is_non_paged ? nullptr : page_table + batch_idx * max_pages_per_seq;
      int new_token_offset = batch_idx * max_ctx_len;
      int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
      int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
      for (int i = threadIdx.x; i < new_len; i += blockDim.x) {
        int page_idx = is_non_paged ? batch_idx : page_list[(cached_len - new_len + i) / page_size];
        int token_idx = page_idx * page_size + (cached_len - new_len + i) % page_size;
        for (int j = 0; j < h_kv * d_k; j++) {
          *(k_cache + token_idx * h_kv * d_k + j) =
              *(new_k + (new_token_offset + i) * h_kv * d_k + j);
        }
        for (int j = 0; j < h_kv * d_v; j++) {
          *(v_cache + token_idx * h_kv * d_v + j) =
              *(new_v + (new_token_offset + i) * h_kv * d_v + j);
        }
      }
    }
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
      int *page_list = is_non_paged ? nullptr : page_table + batch_idx * max_pages_per_seq;
      int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
      int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
      for (int i = threadIdx.x; i < new_len; i += blockDim.x) {
        int page_idx = is_non_paged ? batch_idx : page_list[(cached_len - new_len + i) / page_size];
        int token_idx = page_idx * page_size + (cached_len - new_len + i) % page_size;
        for (int j = 0; j < h_kv * d_k; j++) {
          *(k_cache + token_idx * h_kv * d_k + j) = *(new_k + (i * b + batch_idx) * h_kv * d_k + j);
        }
        for (int j = 0; j < h_kv * d_v; j++) {
          *(v_cache + token_idx * h_kv * d_v + j) = *(new_v + (i * b + batch_idx) * h_kv * d_v + j);
        }
      }
    }
  } else if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
      int *page_list = is_non_paged ? nullptr : page_table + batch_idx * max_pages_per_seq;
      int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
      int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
      for (int i = threadIdx.x; i < new_len; i += blockDim.x) {
        int page_idx = is_non_paged ? batch_idx : page_list[(cached_len - new_len + i) / page_size];
        int token_idx = page_idx * page_size + (cached_len - new_len + i) % page_size;
        for (int j = 0; j < h_kv * d_k; j++) {
          *(k_cache + token_idx * h_kv * d_k + j) =
              *(new_k + (cu_new_lens[batch_idx] + i) * h_kv * d_k + j);
        }
        for (int j = 0; j < h_kv * d_v; j++) {
          *(v_cache + token_idx * h_kv * d_v + j) =
              *(new_v + (cu_new_lens[batch_idx] + i) * h_kv * d_v + j);
        }
      }
    }
  }
}
}  // namespace fused_attn
}  // namespace transformer_engine
#endif
