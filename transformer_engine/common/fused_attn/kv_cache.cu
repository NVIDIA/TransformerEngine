/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace kv_cache {

constexpr int block_size = 1024;

template <typename dtype>
__global__ void reindex_kv_cache_kernel(dtype *k_cache, dtype *v_cache, int *batch_indices,
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
  bool flag = (batch_indices[0] != 0);
  for (int batch_idx = 0; batch_idx < actual_b; batch_idx++) {
    if (flag || ((batch_indices[batch_idx] - batch_indices[0]) != batch_idx)) {
      int num_tokens = (cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx]) -
                       (cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx]);
      int num_elts_k = h_kv * d_k;
      int num_elts_v = h_kv * d_v;
      int num_elts = max(num_elts_k, num_elts_v);
      for (int token_idx = blockIdx.x; token_idx < num_tokens; token_idx += gridDim.x) {
        int src_offset = batch_indices[batch_idx] * max_seq_len + token_idx;
        int des_offset = batch_idx * max_seq_len + token_idx;
        dtype *k_cache_src_offset = k_cache + src_offset * num_elts_k;
        dtype *k_cache_des_offset = k_cache + des_offset * num_elts_k;
        dtype *v_cache_src_offset = v_cache + src_offset * num_elts_v;
        dtype *v_cache_des_offset = v_cache + des_offset * num_elts_v;
        for (int i = threadIdx.x; i < num_elts; i += blockDim.x) {
          if (i < num_elts_k) {
            *(k_cache_des_offset + i) = *(k_cache_src_offset + i);
          }
          if (i < num_elts_v) {
            *(v_cache_des_offset + i) = *(v_cache_src_offset + i);
          }
        }
      }
    }
  }
}

template <typename dtype>
__global__ void copy_to_kv_cache_kernel(dtype *new_k, dtype *new_v, dtype *k_cache, dtype *v_cache,
                                        int *page_table, int *cu_new_lens, int *cu_cached_lens,
                                        NVTE_QKV_Format qkv_format, int h_kv, int d_k, int d_v,
                                        int b, int max_ctx_len, int max_seq_len,
                                        int max_pages_per_seq, bool is_non_paged) {
  // new_k, new_v: qkv_format; k_cache, v_cache: bshd
  // cu_new_lens, cu_cached_lens: [b + 1]
  // page_table: [b, max_pages_per_seq]
  int page_size = max_seq_len / max_pages_per_seq;
  if (qkv_format == NVTE_QKV_Format::NVTE_BSHD) {
    for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
      int *page_list = is_non_paged ? nullptr : page_table + batch_idx * max_pages_per_seq;
      int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
      int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
      int num_elts_k = h_kv * d_k;
      int num_elts_v = h_kv * d_v;
      int hd = h_kv * max(d_k, d_v);
      for (int i = blockIdx.y; i < new_len; i += gridDim.y) {
        int page_idx = is_non_paged ? batch_idx : page_list[(cached_len - new_len + i) / page_size];
        dtype *new_token_id_k = new_k + (batch_idx * max_ctx_len + i) * num_elts_k;
        dtype *new_token_id_v = new_v + (batch_idx * max_ctx_len + i) * num_elts_v;
        dtype *token_id_k =
            k_cache + (page_idx * page_size + (cached_len - new_len + i) % page_size) * num_elts_k;
        dtype *token_id_v =
            v_cache + (page_idx * page_size + (cached_len - new_len + i) % page_size) * num_elts_v;
        for (int j = threadIdx.x; j < hd; j += blockDim.x) {
          if (j < num_elts_k) {
            *(token_id_k + j) = *(new_token_id_k + j);
          }
          if (j < num_elts_v) {
            *(token_id_v + j) = *(new_token_id_v + j);
          }
        }
      }
    }
  } else if (qkv_format == NVTE_QKV_Format::NVTE_SBHD) {
    for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
      int *page_list = is_non_paged ? nullptr : page_table + batch_idx * max_pages_per_seq;
      int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
      int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
      int num_elts_k = h_kv * d_k;
      int num_elts_v = h_kv * d_v;
      int hd = h_kv * max(d_k, d_v);
      for (int i = blockIdx.y; i < new_len; i += gridDim.y) {
        int page_idx = is_non_paged ? batch_idx : page_list[(cached_len - new_len + i) / page_size];
        dtype *new_token_id_k = new_k + (i * b + batch_idx) * num_elts_k;
        dtype *new_token_id_v = new_v + (i * b + batch_idx) * num_elts_v;
        dtype *token_id_k =
            k_cache + (page_idx * page_size + (cached_len - new_len + i) % page_size) * num_elts_k;
        dtype *token_id_v =
            v_cache + (page_idx * page_size + (cached_len - new_len + i) % page_size) * num_elts_v;
        for (int j = threadIdx.x; j < hd; j += blockDim.x) {
          if (j < num_elts_k) {
            *(token_id_k + j) = *(new_token_id_k + j);
          }
          if (j < num_elts_v) {
            *(token_id_v + j) = *(new_token_id_v + j);
          }
        }
      }
    }
  } else if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
      int *page_list = is_non_paged ? nullptr : page_table + batch_idx * max_pages_per_seq;
      int cached_len = cu_cached_lens[batch_idx + 1] - cu_cached_lens[batch_idx];
      int new_len = cu_new_lens[batch_idx + 1] - cu_new_lens[batch_idx];
      int num_elts_k = h_kv * d_k;
      int num_elts_v = h_kv * d_v;
      int hd = h_kv * max(d_k, d_v);
      for (int i = blockIdx.y; i < new_len; i += gridDim.y) {
        int page_idx = is_non_paged ? batch_idx : page_list[(cached_len - new_len + i) / page_size];
        dtype *new_token_id_k = new_k + (cu_new_lens[batch_idx] + i) * num_elts_k;
        dtype *new_token_id_v = new_v + (cu_new_lens[batch_idx] + i) * num_elts_v;
        dtype *token_id_k =
            k_cache + (page_idx * page_size + (cached_len - new_len + i) % page_size) * num_elts_k;
        dtype *token_id_v =
            v_cache + (page_idx * page_size + (cached_len - new_len + i) % page_size) * num_elts_v;
        for (int j = threadIdx.x; j < hd; j += blockDim.x) {
          if (j < num_elts_k) {
            *(token_id_k + j) = *(new_token_id_k + j);
          }
          if (j < num_elts_v) {
            *(token_id_v + j) = *(new_token_id_v + j);
          }
        }
      }
    }
  }
}

template <typename dtype>
void copy_to_kv_cache_launcher(Tensor new_k, Tensor new_v, Tensor k_cache, Tensor v_cache,
                               Tensor page_table, Tensor cu_new_lens, Tensor cu_cached_lens,
                               NVTE_QKV_Format qkv_format, int h_kv, int d_k, int d_v, int b,
                               int max_ctx_len, int max_seq_len, int max_pages_per_seq,
                               bool is_non_paged, cudaStream_t stream) {
  if (new_k.has_data() && new_v.has_data() && k_cache.has_data() && v_cache.has_data()) {
    if (is_non_paged) {
      reindex_kv_cache_kernel<<<max_seq_len, block_size, 0, stream>>>(
          reinterpret_cast<dtype *>(k_cache.data.dptr),
          reinterpret_cast<dtype *>(v_cache.data.dptr),
          reinterpret_cast<int *>(page_table.data.dptr),
          reinterpret_cast<int *>(cu_new_lens.data.dptr),
          reinterpret_cast<int *>(cu_cached_lens.data.dptr), h_kv, d_k, d_v, b, max_seq_len);
      NVTE_CHECK_CUDA(cudaGetLastError());
    }
    dim3 grid_size(b, max_ctx_len);
    copy_to_kv_cache_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<dtype *>(new_k.data.dptr), reinterpret_cast<dtype *>(new_v.data.dptr),
        reinterpret_cast<dtype *>(k_cache.data.dptr), reinterpret_cast<dtype *>(v_cache.data.dptr),
        reinterpret_cast<int *>(page_table.data.dptr),
        reinterpret_cast<int *>(cu_new_lens.data.dptr),
        reinterpret_cast<int *>(cu_cached_lens.data.dptr), qkv_format, h_kv, d_k, d_v, b,
        max_ctx_len, max_seq_len, max_pages_per_seq, is_non_paged);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void copy_to_kv_cache(Tensor new_k, Tensor new_v, Tensor k_cache, Tensor v_cache, Tensor page_table,
                      Tensor cu_new_lens, Tensor cu_cached_lens, NVTE_QKV_Format qkv_format, int b,
                      int max_ctx_len, int max_seq_len, int max_pages_per_seq, bool is_non_paged,
                      cudaStream_t stream) {
  int h_kv = new_k.shape()[new_k.dim() - 2];
  int d_k = new_k.shape()[new_k.dim() - 1];
  int d_v = new_v.shape()[new_v.dim() - 1];
  NVTE_CHECK(k_cache.dtype() == v_cache.dtype() && new_k.dtype() == new_v.dtype() &&
                 new_k.dtype() == k_cache.dtype(),
             "new_k, new_v, k_cache and v_cache must be of the same data type.");
  NVTE_CHECK(qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD ||
                 qkv_format == NVTE_QKV_Format::NVTE_THD,
             "qkv_format must be {BSHD, SBHD, THD}.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_FLOAT(
      k_cache.dtype(), dtype,
      copy_to_kv_cache_launcher<dtype>(new_k, new_v, k_cache, v_cache, page_table, cu_new_lens,
                                       cu_cached_lens, qkv_format, h_kv, d_k, d_v, b, max_ctx_len,
                                       max_seq_len, max_pages_per_seq, is_non_paged, stream););
}

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
void convert_thd_to_bshd_launcher(Tensor tensor, Tensor new_tensor, Tensor cu_seqlens, int b,
                                  int max_seq_len, int h, int d, cudaStream_t stream) {
  using namespace transformer_engine;
  convert_thd_to_bshd_kernel<<<16, 256, 0, stream>>>(
      reinterpret_cast<scalar_t *>(tensor.data.dptr),
      reinterpret_cast<scalar_t *>(new_tensor.data.dptr),
      reinterpret_cast<int *>(cu_seqlens.data.dptr), b, max_seq_len, h, d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void convert_thd_to_bshd(Tensor tensor, Tensor cu_seqlens, Tensor new_tensor, int b,
                         int max_seq_len, cudaStream_t stream) {
  using namespace transformer_engine;

  auto tensor_shape = tensor.shape();
  TRANSFORMER_ENGINE_TYPE_SWITCH_FLOAT(
      new_tensor.dtype(), dtype,
      convert_thd_to_bshd_launcher<dtype>(tensor, new_tensor, cu_seqlens, b, max_seq_len,
                                          tensor_shape[1], tensor_shape[2], stream););
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
void convert_bshd_to_thd_launcher(Tensor tensor, Tensor new_tensor, Tensor cu_seqlens, int b,
                                  int max_seq_len, int h, int d, cudaStream_t stream) {
  using namespace transformer_engine;
  convert_bshd_to_thd_kernel<<<16, 256, 0, stream>>>(
      reinterpret_cast<scalar_t *>(tensor.data.dptr),
      reinterpret_cast<scalar_t *>(new_tensor.data.dptr),
      reinterpret_cast<int *>(cu_seqlens.data.dptr), b, max_seq_len, h, d);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void convert_bshd_to_thd(Tensor tensor, Tensor cu_seqlens, Tensor new_tensor, int t,
                         cudaStream_t stream) {
  using namespace transformer_engine;

  auto tensor_shape = tensor.shape();
  TRANSFORMER_ENGINE_TYPE_SWITCH_FLOAT(
      tensor.dtype(), dtype,
      convert_bshd_to_thd_launcher<dtype>(tensor, new_tensor, cu_seqlens, tensor_shape[0],
                                          tensor_shape[1], tensor_shape[2], tensor_shape[3],
                                          stream););
}

}  // namespace kv_cache
}  // namespace transformer_engine

/***************************************************************************************************
 * KV Cache: Copy new KV tokens to the KV cache
 *   1. new_k and new_v are in qkv_format; k_cache and v_cache are in 'bshd' format
 *   2. cu_new_lens and cu_cached_lens are in shape [b + 1]; cu_cached_lens include the added lens
 *      in current step
 *   3. Non-paged KV cache is a special case of paged KV cache, with page_table = [b, 1] and
 *      max_pages_per_seq = 1. We use the same underlying kernel for both non-paged and paged.
 *      Set is_non_paged = True/False to indicate as such.
 *   4. is_non_paged = True also re-indexes the KV cache, e.g. the initial batch indices [0, 3, 1, 2]
 *      becomes [0, 1, 1, 2]. The page_table = batch_indices.unsqueeze(1) is however unchanged.
 *      batch_indices_post can be used for monotonical indexing, i.e. [0, 1, 2, 3]. batch_indices is
 *      preserved for the next layer in the same iteration.
 *   5. Only supports same page_table for k_cache and v_cache
 *   6. Only pad_between_seqs = False when qkv_format = thd, i.e. there should be no pad tokens
 *      between sequences in new_k and new_v such as [a a a 0..0 b b 0..0 c 0..0].
 **************************************************************************************************/

void nvte_copy_to_kv_cache(NVTETensor new_k, NVTETensor new_v, NVTETensor k_cache,
                           NVTETensor v_cache, NVTETensor page_table, NVTETensor cu_new_lens,
                           NVTETensor cu_cached_lens, NVTE_QKV_Format qkv_format, int b,
                           int max_ctx_len, int max_seq_len, int max_pages_per_seq,
                           int is_non_paged, cudaStream_t stream) {
  NVTE_API_CALL(nvte_copy_to_kv_cache);
  using namespace transformer_engine;

  kv_cache::copy_to_kv_cache(*convertNVTETensorCheck(new_k), *convertNVTETensorCheck(new_v),
                             *convertNVTETensorCheck(k_cache), *convertNVTETensorCheck(v_cache),
                             *convertNVTETensorCheck(page_table),
                             *convertNVTETensorCheck(cu_new_lens),
                             *convertNVTETensorCheck(cu_cached_lens), qkv_format, b, max_ctx_len,
                             max_seq_len, max_pages_per_seq, is_non_paged, stream);
}

/***************************************************************************************************
 * KV Cache: Convert a tensor from qkv_format = thd to qkv_format = bshd
 **************************************************************************************************/

void nvte_convert_thd_to_bshd(NVTETensor tensor, NVTETensor cu_seqlens, NVTETensor new_tensor,
                              int b, int max_seq_len, cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_thd_to_bshd);
  using namespace transformer_engine;

  kv_cache::convert_thd_to_bshd(*convertNVTETensorCheck(tensor),
                                *convertNVTETensorCheck(cu_seqlens),
                                *convertNVTETensorCheck(new_tensor), b, max_seq_len, stream);
}

/***************************************************************************************************
 * KV Cache: Convert a tensor from qkv_format = bshd to qkv_format = thd
 **************************************************************************************************/

void nvte_convert_bshd_to_thd(NVTETensor tensor, NVTETensor cu_seqlens, NVTETensor new_tensor,
                              int t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_bshd_to_thd);
  using namespace transformer_engine;

  kv_cache::convert_bshd_to_thd(*convertNVTETensorCheck(tensor),
                                *convertNVTETensorCheck(cu_seqlens),
                                *convertNVTETensorCheck(new_tensor), t, stream);
}
