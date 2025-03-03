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
__global__ void convert_thd_to_bshd_kernel(scalar_t *tensor, scalar_t *new_tensor, int *cu_seqlens,
                                           int b, int max_seq_len, int h, int d) {
  // tensor: thd; new_tensor: bshd
  // cu_seqlens: [b + 1]
  for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
    int num_elts = (cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]) * h * d;
    int thd_offset = cu_seqlens[batch_idx] * h * d;
    int bshd_offset = batch_idx * max_seq_len * h * d;
    scalar_t *thd_token = tensor + thd_offset;
    scalar_t *bshd_token = new_tensor + bshd_offset;
    for (int i = threadIdx.x; i < num_elts; i += blockDim.x) {
      *(bshd_token + i) = *(thd_token + i);
    }
  }
}

template <typename scalar_t>
__global__ void convert_bshd_to_thd_kernel(scalar_t *tensor, scalar_t *new_tensor, int *cu_seqlens,
                                           int b, int max_seq_len, int h, int d) {
  // tensor: bshd; new_tensor: thd
  // cu_seqlens: [b + 1]
  for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
    int seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx];
    int num_elts = seqlen * h * d;
    int bshd_offset = batch_idx * max_seq_len * h * d;
    int thd_offset = cu_seqlens[batch_idx] * h * d;
    scalar_t *bshd_token = tensor + bshd_offset;
    scalar_t *thd_token = new_tensor + thd_offset;
    for (int i = threadIdx.x; i < num_elts; i += blockDim.x) {
      *(thd_token + i) = *(bshd_token + i);
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
