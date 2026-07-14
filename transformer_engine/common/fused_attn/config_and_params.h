/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file config_and_params.h
 *  \brief Internal backing objects for fused-attention config and parameter handles.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_

#include <tuple>

#include "common/common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {

struct FusedAttnConfig {
  // basic attention knobs
  bool is_training = false;
  bool deterministic = false;
  bool cuda_graph = false;
  bool return_max_logit = false;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  float dropout = 0.0f;

  // data types
  NVTEDType qkv_dtype = kNVTEBFloat16;
  NVTEDType o_dtype = kNVTEBFloat16;
  NVTEDType do_dtype = kNVTEBFloat16;
  NVTEDType dqkv_dtype = kNVTEBFloat16;

  // data and scale layout
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Layout dqkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_scale_inv_format = NVTE_QKV_Format_NOT_SET;

  // attention scaling
  float attn_scale = 1.0f;

  // tensor dimensions
  size_t batch_size = 0;
  size_t num_attn_heads = 0;
  size_t num_gqa_groups = 0;
  size_t head_dim_qk = 0;
  size_t head_dim_v = 0;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
  size_t num_tokens_q = 0;
  size_t num_tokens_kv = 0;

  // derived tensor dimensions (internal only)
  size_t bucketed_batch_size = 0;
  size_t bucketed_num_tokens_q = 0;
  size_t bucketed_num_tokens_kv = 0;

  // paged KV dimensions
  size_t num_pages_k = 0;
  size_t num_pages_v = 0;
  size_t page_size_k = 0;
  size_t page_size_v = 0;
  size_t max_pages_per_seq_k = 0;
  size_t max_pages_per_seq_v = 0;

  // bias dimensions
  size_t bias_batch_size = 0;
  size_t bias_num_heads = 0;
  size_t bias_seqlen_q = 0;
  size_t bias_seqlen_kv = 0;

  static constexpr size_t attr_sizes[] = {
      // basic attention knobs
      sizeof(uint8_t),            // is_training
      sizeof(uint8_t),            // deterministic
      sizeof(uint8_t),            // cuda_graph
      sizeof(uint8_t),            // return_max_logit
      sizeof(NVTE_Mask_Type),     // attn_mask_type
      sizeof(NVTE_Bias_Type),     // bias_type
      sizeof(int64_t),            // window_size_left
      sizeof(int64_t),            // window_size_right
      sizeof(uint8_t),            // bottom_right_diagonal
      sizeof(NVTE_Softmax_Type),  // softmax_type
      sizeof(NVTEScalingMode),    // scaling_mode
      sizeof(float),              // dropout
      // data types
      sizeof(NVTEDType),  // qkv_dtype
      sizeof(NVTEDType),  // o_dtype
      sizeof(NVTEDType),  // do_dtype
      sizeof(NVTEDType),  // dqkv_dtype
      // data and scale layout
      sizeof(NVTE_QKV_Layout),  // qkv_layout
      sizeof(NVTE_QKV_Format),  // o_format
      sizeof(NVTE_QKV_Format),  // do_format
      sizeof(NVTE_QKV_Layout),  // dqkv_layout
      sizeof(NVTE_QKV_Format),  // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),  // do_scale_inv_format
      // attention scaling
      sizeof(float),  // attn_scale
      // tensor dimensions
      sizeof(size_t),  // batch_size
      sizeof(size_t),  // num_attn_heads
      sizeof(size_t),  // num_gqa_groups
      sizeof(size_t),  // head_dim_qk
      sizeof(size_t),  // head_dim_v
      sizeof(size_t),  // max_seqlen_q
      sizeof(size_t),  // max_seqlen_kv
      sizeof(size_t),  // num_tokens_q
      sizeof(size_t),  // num_tokens_kv
      // paged KV dimensions
      sizeof(size_t),  // num_pages_k
      sizeof(size_t),  // num_pages_v
      sizeof(size_t),  // page_size_k
      sizeof(size_t),  // page_size_v
      sizeof(size_t),  // max_pages_per_seq_k
      sizeof(size_t),  // max_pages_per_seq_v
      // bias dimensions
      sizeof(size_t),  // bias_batch_size
      sizeof(size_t),  // bias_num_heads
      sizeof(size_t),  // bias_seqlen_q
      sizeof(size_t),  // bias_seqlen_kv
  };

  bool operator<(const FusedAttnConfig &rhs) const {
    return std::tie(is_training, deterministic, cuda_graph, return_max_logit, attn_mask_type,
                    bias_type, window_size_left, window_size_right, bottom_right_diagonal,
                    softmax_type, scaling_mode, dropout, qkv_dtype, o_dtype, do_dtype, dqkv_dtype,
                    qkv_layout, o_format, do_format, dqkv_layout, qkv_scale_inv_format,
                    do_scale_inv_format, attn_scale, batch_size, num_attn_heads, num_gqa_groups,
                    head_dim_qk, head_dim_v, max_seqlen_q, max_seqlen_kv, num_tokens_q,
                    num_tokens_kv, bucketed_batch_size, bucketed_num_tokens_q,
                    bucketed_num_tokens_kv, num_pages_k, num_pages_v, page_size_k, page_size_v,
                    max_pages_per_seq_k, max_pages_per_seq_v, bias_batch_size, bias_num_heads,
                    bias_seqlen_q, bias_seqlen_kv) <
           std::tie(rhs.is_training, rhs.deterministic, rhs.cuda_graph, rhs.return_max_logit,
                    rhs.attn_mask_type, rhs.bias_type, rhs.window_size_left, rhs.window_size_right,
                    rhs.bottom_right_diagonal, rhs.softmax_type, rhs.scaling_mode, rhs.dropout,
                    rhs.qkv_dtype, rhs.o_dtype, rhs.do_dtype, rhs.dqkv_dtype, rhs.qkv_layout,
                    rhs.o_format, rhs.do_format, rhs.dqkv_layout, rhs.qkv_scale_inv_format,
                    rhs.do_scale_inv_format, rhs.attn_scale, rhs.batch_size, rhs.num_attn_heads,
                    rhs.num_gqa_groups, rhs.head_dim_qk, rhs.head_dim_v, rhs.max_seqlen_q,
                    rhs.max_seqlen_kv, rhs.num_tokens_q, rhs.num_tokens_kv, rhs.bucketed_batch_size,
                    rhs.bucketed_num_tokens_q, rhs.bucketed_num_tokens_kv, rhs.num_pages_k,
                    rhs.num_pages_v, rhs.page_size_k, rhs.page_size_v, rhs.max_pages_per_seq_k,
                    rhs.max_pages_per_seq_v, rhs.bias_batch_size, rhs.bias_num_heads,
                    rhs.bias_seqlen_q, rhs.bias_seqlen_kv);
  }
};

inline FusedAttnConfig make_default_fused_attn_config() { return FusedAttnConfig{}; }

void populate_fused_attn_config(FusedAttnConfig *cfg);

// Normalize cfg into the graph-cache key form used by cuDNN graph caching (ragged bucketing,
// bottom-right mask folding). Call after populate_fused_attn_config(). Pass is_forward=true when
// keying a forward graph and is_forward=false for a backward graph; each key drops the fields the
// corresponding graph does not consume so it is not fragmented by them: a training forward key
// drops the dO/dQKV dtypes and the (backward-only) deterministic flag, and a backward key drops
// the (forward-only) return_max_logit flag.
FusedAttnConfig make_fused_attn_graph_cache_config(const FusedAttnConfig &cfg, bool is_forward);

inline const FusedAttnConfig *get_fused_attn_config(NVTEFusedAttnConfig config) {
  NVTE_CHECK(config != nullptr, "NVTEFusedAttnConfig must not be NULL.");
  return reinterpret_cast<const FusedAttnConfig *>(config);
}

inline FusedAttnConfig *get_fused_attn_config_mutable(NVTEFusedAttnConfig config) {
  NVTE_CHECK(config != nullptr, "NVTEFusedAttnConfig must not be NULL.");
  return reinterpret_cast<FusedAttnConfig *>(config);
}

struct FusedAttnFwdParams {
  NVTETensor Q = nullptr;
  NVTETensor K = nullptr;
  NVTETensor V = nullptr;
  NVTETensor Bias = nullptr;
  NVTETensor SoftmaxOffset = nullptr;
  NVTETensor cu_seqlens_q = nullptr;
  NVTETensor cu_seqlens_kv = nullptr;
  NVTETensor cu_seqlens_q_padded = nullptr;
  NVTETensor cu_seqlens_kv_padded = nullptr;
  NVTETensor page_table_k = nullptr;
  NVTETensor page_table_v = nullptr;
  NVTETensor rng_state = nullptr;
  NVTETensor S = nullptr;
  NVTETensor O = nullptr;
  NVTETensorPack *Aux_CTX_Tensors = nullptr;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  float attn_scale = 1.0f;
  float dropout = 0.0f;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  bool is_training = false;
  bool return_max_logit = false;
  bool cuda_graph = false;
  NVTETensor workspace = nullptr;
  cudaStream_t stream = nullptr;

  static constexpr size_t attr_sizes[] = {
      sizeof(NVTETensor),         // Q
      sizeof(NVTETensor),         // K
      sizeof(NVTETensor),         // V
      sizeof(NVTETensor),         // Bias
      sizeof(NVTETensor),         // SoftmaxOffset
      sizeof(NVTETensor),         // cu_seqlens_q
      sizeof(NVTETensor),         // cu_seqlens_kv
      sizeof(NVTETensor),         // cu_seqlens_q_padded
      sizeof(NVTETensor),         // cu_seqlens_kv_padded
      sizeof(NVTETensor),         // page_table_k
      sizeof(NVTETensor),         // page_table_v
      sizeof(NVTETensor),         // rng_state
      sizeof(NVTETensor),         // S
      sizeof(NVTETensor),         // O
      sizeof(NVTETensorPack *),   // Aux_CTX_Tensors
      sizeof(size_t),             // max_seqlen_q
      sizeof(size_t),             // max_seqlen_kv
      sizeof(NVTE_QKV_Layout),    // qkv_layout
      sizeof(NVTE_QKV_Format),    // o_format
      sizeof(NVTE_QKV_Format),    // qkv_scale_inv_format
      sizeof(NVTE_Bias_Type),     // bias_type
      sizeof(NVTE_Mask_Type),     // attn_mask_type
      sizeof(NVTE_Softmax_Type),  // softmax_type
      sizeof(float),              // attn_scale
      sizeof(float),              // dropout
      sizeof(int64_t),            // window_size_left
      sizeof(int64_t),            // window_size_right
      sizeof(uint8_t),            // bottom_right_diagonal
      sizeof(uint8_t),            // is_training
      sizeof(uint8_t),            // return_max_logit
      sizeof(uint8_t),            // cuda_graph
      sizeof(NVTETensor),         // workspace
      sizeof(cudaStream_t),       // stream
  };
};

struct FusedAttnBwdParams {
  NVTETensor Q = nullptr;
  NVTETensor K = nullptr;
  NVTETensor V = nullptr;
  NVTETensor O = nullptr;
  NVTETensor dO = nullptr;
  NVTETensor S = nullptr;
  NVTETensor dP = nullptr;
  const NVTETensorPack *Aux_CTX_Tensors = nullptr;
  NVTETensor dQ = nullptr;
  NVTETensor dK = nullptr;
  NVTETensor dV = nullptr;
  NVTETensor dBias = nullptr;
  NVTETensor dSoftmaxOffset = nullptr;
  NVTETensor cu_seqlens_q = nullptr;
  NVTETensor cu_seqlens_kv = nullptr;
  NVTETensor cu_seqlens_q_padded = nullptr;
  NVTETensor cu_seqlens_kv_padded = nullptr;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Layout dqkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  float attn_scale = 1.0f;
  float dropout = 0.0f;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  bool deterministic = false;
  bool cuda_graph = false;
  NVTETensor workspace = nullptr;
  cudaStream_t stream = nullptr;

  static constexpr size_t attr_sizes[] = {
      sizeof(NVTETensor),              // Q
      sizeof(NVTETensor),              // K
      sizeof(NVTETensor),              // V
      sizeof(NVTETensor),              // O
      sizeof(NVTETensor),              // dO
      sizeof(NVTETensor),              // S
      sizeof(NVTETensor),              // dP
      sizeof(const NVTETensorPack *),  // Aux_CTX_Tensors
      sizeof(NVTETensor),              // dQ
      sizeof(NVTETensor),              // dK
      sizeof(NVTETensor),              // dV
      sizeof(NVTETensor),              // dBias
      sizeof(NVTETensor),              // dSoftmaxOffset
      sizeof(NVTETensor),              // cu_seqlens_q
      sizeof(NVTETensor),              // cu_seqlens_kv
      sizeof(NVTETensor),              // cu_seqlens_q_padded
      sizeof(NVTETensor),              // cu_seqlens_kv_padded
      sizeof(size_t),                  // max_seqlen_q
      sizeof(size_t),                  // max_seqlen_kv
      sizeof(NVTE_QKV_Layout),         // qkv_layout
      sizeof(NVTE_QKV_Format),         // o_format
      sizeof(NVTE_QKV_Format),         // do_format
      sizeof(NVTE_QKV_Layout),         // dqkv_layout
      sizeof(NVTE_QKV_Format),         // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),         // do_scale_inv_format
      sizeof(NVTE_Bias_Type),          // bias_type
      sizeof(NVTE_Mask_Type),          // attn_mask_type
      sizeof(NVTE_Softmax_Type),       // softmax_type
      sizeof(float),                   // attn_scale
      sizeof(float),                   // dropout
      sizeof(int64_t),                 // window_size_left
      sizeof(int64_t),                 // window_size_right
      sizeof(uint8_t),                 // bottom_right_diagonal
      sizeof(uint8_t),                 // deterministic
      sizeof(uint8_t),                 // cuda_graph
      sizeof(NVTETensor),              // workspace
      sizeof(cudaStream_t),            // stream
  };
};

inline FusedAttnFwdParams make_default_fused_attn_fwd_params() { return FusedAttnFwdParams{}; }

inline FusedAttnBwdParams make_default_fused_attn_bwd_params() { return FusedAttnBwdParams{}; }

// Build a FusedAttnConfig from the scalar "knobs" carried by the fwd/bwd params (mask/bias/softmax
// type, scales, dropout, window, layout/format fields, flags). The tensor-derived fields (dtypes,
// dims, scaling_mode, paged-KV / bias dims, token counts) are left at their defaults and must be
// filled in by the caller from the actual tensors.
FusedAttnConfig make_fused_attn_config(const FusedAttnFwdParams &params);
FusedAttnConfig make_fused_attn_config(const FusedAttnBwdParams &params);

inline const FusedAttnFwdParams *get_fused_attn_fwd_params(NVTEFusedAttnFwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnFwdParams must not be NULL.");
  return reinterpret_cast<const FusedAttnFwdParams *>(params);
}

inline FusedAttnFwdParams *get_fused_attn_fwd_params_mutable(NVTEFusedAttnFwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnFwdParams must not be NULL.");
  return reinterpret_cast<FusedAttnFwdParams *>(params);
}

inline const FusedAttnBwdParams *get_fused_attn_bwd_params(NVTEFusedAttnBwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnBwdParams must not be NULL.");
  return reinterpret_cast<const FusedAttnBwdParams *>(params);
}

inline FusedAttnBwdParams *get_fused_attn_bwd_params_mutable(NVTEFusedAttnBwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnBwdParams must not be NULL.");
  return reinterpret_cast<FusedAttnBwdParams *>(params);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_
