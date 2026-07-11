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

#include "common/common.h"
#include "transformer_engine/fused_attn.h"

#include <tuple>

namespace transformer_engine {

struct FusedAttnConfig {
  bool is_training = false;
  bool deterministic = false;
  bool cuda_graph = false;
  bool return_max_logit = false;
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Layout dqkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  float attn_scale = 0.0f;
  float dropout = 0.0f;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = false;
  NVTEDType qkv_dtype = kNVTEFloat32;
  NVTEDType o_dtype = kNVTEFloat32;
  NVTEDType do_dtype = kNVTEFloat32;
  NVTEDType dqkv_dtype = kNVTEFloat32;
  size_t batch_size = 0;
  size_t num_attn_heads = 0;
  size_t num_gqa_groups = 0;
  size_t head_dim_qk = 0;
  size_t head_dim_v = 0;
  size_t num_pages_k = 0;
  size_t num_pages_v = 0;
  size_t page_size_k = 0;
  size_t page_size_v = 0;
  size_t max_pages_per_seq_k = 0;
  size_t max_pages_per_seq_v = 0;
  size_t bias_batch_size = 0;
  size_t bias_num_heads = 0;
  size_t bias_seqlen_q = 0;
  size_t bias_seqlen_kv = 0;
  size_t num_tokens_q = 0;
  size_t num_tokens_kv = 0;
  size_t bucketed_batch_size = 0;
  size_t bucketed_num_tokens_q = 0;
  size_t bucketed_num_tokens_kv = 0;

  static constexpr size_t attr_sizes[] = {
      sizeof(uint8_t),              // is_training
      sizeof(uint8_t),              // deterministic
      sizeof(uint8_t),              // cuda_graph
      sizeof(uint8_t),              // return_max_logit
      sizeof(NVTE_QKV_Layout),      // qkv_layout
      sizeof(NVTE_QKV_Format),      // o_format
      sizeof(NVTE_QKV_Format),      // do_format
      sizeof(NVTE_QKV_Layout),      // dqkv_layout
      sizeof(NVTE_QKV_Format),      // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),      // do_scale_inv_format
      sizeof(NVTE_Bias_Type),       // bias_type
      sizeof(NVTE_Mask_Type),       // attn_mask_type
      sizeof(NVTE_Softmax_Type),    // softmax_type
      sizeof(NVTEScalingMode),      // scaling_mode
      sizeof(float),                // attn_scale
      sizeof(float),                // dropout
      sizeof(size_t),               // max_seqlen_q
      sizeof(size_t),               // max_seqlen_kv
      sizeof(int64_t),              // window_size_left
      sizeof(int64_t),              // window_size_right
      sizeof(uint8_t),              // bottom_right_diagonal
      sizeof(NVTEDType),            // qkv_dtype
      sizeof(NVTEDType),            // o_dtype
      sizeof(NVTEDType),            // do_dtype
      sizeof(NVTEDType),            // dqkv_dtype
      sizeof(size_t),               // batch_size
      sizeof(size_t),               // num_attn_heads
      sizeof(size_t),               // num_gqa_groups
      sizeof(size_t),               // head_dim_qk
      sizeof(size_t),               // head_dim_v
      sizeof(size_t),               // num_pages_k
      sizeof(size_t),               // num_pages_v
      sizeof(size_t),               // page_size_k
      sizeof(size_t),               // page_size_v
      sizeof(size_t),               // max_pages_per_seq_k
      sizeof(size_t),               // max_pages_per_seq_v
      sizeof(size_t),               // bias_batch_size
      sizeof(size_t),               // bias_num_heads
      sizeof(size_t),               // bias_seqlen_q
      sizeof(size_t),               // bias_seqlen_kv
      sizeof(size_t),               // num_tokens_q
      sizeof(size_t),               // num_tokens_kv
      sizeof(size_t),               // bucketed_batch_size
      sizeof(size_t),               // bucketed_num_tokens_q
      sizeof(size_t),               // bucketed_num_tokens_kv
  };

  bool operator<(const FusedAttnConfig &rhs) const {
    return std::tie(is_training, deterministic, cuda_graph, return_max_logit, qkv_layout, o_format,
                    do_format, dqkv_layout, qkv_scale_inv_format, do_scale_inv_format, bias_type,
                    attn_mask_type, softmax_type, scaling_mode, attn_scale, dropout, max_seqlen_q,
                    max_seqlen_kv, window_size_left, window_size_right, bottom_right_diagonal,
                    qkv_dtype, o_dtype, do_dtype, dqkv_dtype, batch_size, num_attn_heads,
                    num_gqa_groups, head_dim_qk, head_dim_v, num_pages_k, num_pages_v, page_size_k,
                    page_size_v, max_pages_per_seq_k, max_pages_per_seq_v, bias_batch_size,
                    bias_num_heads, bias_seqlen_q, bias_seqlen_kv, num_tokens_q, num_tokens_kv,
                    bucketed_batch_size, bucketed_num_tokens_q, bucketed_num_tokens_kv) <
           std::tie(rhs.is_training, rhs.deterministic, rhs.cuda_graph, rhs.return_max_logit,
                    rhs.qkv_layout, rhs.o_format, rhs.do_format, rhs.dqkv_layout,
                    rhs.qkv_scale_inv_format, rhs.do_scale_inv_format, rhs.bias_type,
                    rhs.attn_mask_type, rhs.softmax_type, rhs.scaling_mode, rhs.attn_scale,
                    rhs.dropout, rhs.max_seqlen_q, rhs.max_seqlen_kv, rhs.window_size_left,
                    rhs.window_size_right, rhs.bottom_right_diagonal, rhs.qkv_dtype, rhs.o_dtype,
                    rhs.do_dtype, rhs.dqkv_dtype, rhs.batch_size, rhs.num_attn_heads,
                    rhs.num_gqa_groups, rhs.head_dim_qk, rhs.head_dim_v, rhs.num_pages_k,
                    rhs.num_pages_v, rhs.page_size_k, rhs.page_size_v, rhs.max_pages_per_seq_k,
                    rhs.max_pages_per_seq_v, rhs.bias_batch_size, rhs.bias_num_heads,
                    rhs.bias_seqlen_q, rhs.bias_seqlen_kv, rhs.num_tokens_q, rhs.num_tokens_kv,
                    rhs.bucketed_batch_size, rhs.bucketed_num_tokens_q, rhs.bucketed_num_tokens_kv);
  }
};

inline FusedAttnConfig make_default_fused_attn_config() { return FusedAttnConfig{}; }

void populate_fused_attn_config(FusedAttnConfig *cfg);

// Normalize cfg into the graph-cache key form used by cuDNN graph caching (ragged bucketing,
// bottom-right mask folding). Call after populate_fused_attn_config().
FusedAttnConfig make_fused_attn_graph_cache_config(const FusedAttnConfig &cfg);

inline const FusedAttnConfig *get_fused_attn_config(NVTEFusedAttnConfig config) {
  NVTE_CHECK(config != nullptr, "NVTEFusedAttnConfig must not be NULL.");
  return reinterpret_cast<const FusedAttnConfig *>(config);
}

inline FusedAttnConfig *get_fused_attn_config_mutable(NVTEFusedAttnConfig config) {
  NVTE_CHECK(config != nullptr, "NVTEFusedAttnConfig must not be NULL.");
  return reinterpret_cast<FusedAttnConfig *>(config);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_
