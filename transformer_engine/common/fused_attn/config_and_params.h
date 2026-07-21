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
namespace fused_attn {

// Single source of truth for the graph-cache-relevant fields of FusedAttnConfig, in the SAME order
// as NVTEFusedAttnConfigAttribute / attr_sizes[]. Each row is
//   X(member, wire_type, printf_fmt, printf_cast, debug_label)
// where wire_type is the field's 1-byte-exact serialization type (bool is serialized as uint8_t),
// and printf_fmt / printf_cast / debug_label drive the [FUSED-ATTN-CACHE] debug dump. operator<,
// attr_sizes[], and that debug dump are all generated from this one list so they cannot drift apart.
// When adding/removing a cache-relevant field, edit ONLY this list -- and the public
// NVTEFusedAttnConfigAttribute enum, whose entry count the static_assert below cross-checks.
#define TE_FUSED_ATTN_CACHE_KEY_FIELDS(X)                                     \
  /* basic attention settings */                                         \
  X(is_training, uint8_t, "%d", int, "train")                            \
  X(deterministic, uint8_t, "%d", int, "det")                            \
  X(cuda_graph, uint8_t, "%d", int, "cg")                                \
  X(return_max_logit, uint8_t, "%d", int, "maxlogit")                    \
  X(attn_mask_type, NVTE_Mask_Type, "%lld", long long, "mask")           \
  X(bias_type, NVTE_Bias_Type, "%lld", long long, "bias")                \
  X(window_size_left, int64_t, "%lld", long long, "wl")                  \
  X(window_size_right, int64_t, "%lld", long long, "wr")                 \
  X(bottom_right_diagonal, uint8_t, "%d", int, "brd")                    \
  X(softmax_type, NVTE_Softmax_Type, "%lld", long long, "softmax")       \
  X(scaling_mode, NVTEScalingMode, "%lld", long long, "scale_mode")      \
  X(dropout, float, "%g", double, "dropout")                             \
  X(attn_scale, float, "%g", double, "attn_scale")                       \
  /* tensor types */                                                     \
  X(qkv_dtype, NVTEDType, "%lld", long long, "qkv_dt")                   \
  X(o_dtype, NVTEDType, "%lld", long long, "o_dt")                       \
  X(do_dtype, NVTEDType, "%lld", long long, "do_dt")                     \
  X(dqkv_dtype, NVTEDType, "%lld", long long, "dqkv_dt")                 \
  /* tensor layouts */                                                   \
  X(qkv_layout, NVTE_QKV_Layout, "%lld", long long, "qkv_lay")           \
  X(o_format, NVTE_QKV_Format, "%lld", long long, "o_fmt")               \
  X(do_format, NVTE_QKV_Format, "%lld", long long, "do_fmt")             \
  X(dqkv_layout, NVTE_QKV_Layout, "%lld", long long, "dqkv_lay")         \
  X(qkv_scale_inv_format, NVTE_QKV_Format, "%lld", long long, "qkv_sif") \
  X(do_scale_inv_format, NVTE_QKV_Format, "%lld", long long, "do_sif")   \
  /* tensor dimensions */                                                \
  X(batch_size, size_t, "%lld", long long, "b")                          \
  X(num_attn_heads, size_t, "%lld", long long, "h")                      \
  X(num_gqa_groups, size_t, "%lld", long long, "hg")                     \
  X(head_dim_qk, size_t, "%lld", long long, "dqk")                       \
  X(head_dim_v, size_t, "%lld", long long, "dv")                         \
  X(max_seqlen_q, size_t, "%lld", long long, "sq")                       \
  X(max_seqlen_kv, size_t, "%lld", long long, "skv")                     \
  X(num_tokens_q, size_t, "%lld", long long, "tq")                       \
  X(num_tokens_kv, size_t, "%lld", long long, "tkv")                     \
  /* paged KV dimensions */                                              \
  X(num_pages_k, size_t, "%lld", long long, "npk")                       \
  X(num_pages_v, size_t, "%lld", long long, "npv")                       \
  X(page_size_k, size_t, "%lld", long long, "psk")                       \
  X(page_size_v, size_t, "%lld", long long, "psv")                       \
  X(max_pages_per_seq_k, size_t, "%lld", long long, "mppk")              \
  X(max_pages_per_seq_v, size_t, "%lld", long long, "mppv")              \
  /* bias dimensions */                                                  \
  X(bias_batch_size, size_t, "%lld", long long, "bias_b")                \
  X(bias_num_heads, size_t, "%lld", long long, "bias_h")                 \
  X(bias_seqlen_q, size_t, "%lld", long long, "bias_sq")                 \
  X(bias_seqlen_kv, size_t, "%lld", long long, "bias_skv")

struct FusedAttnConfig {
  // basic attention settings
  bool is_training = true;
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
  float attn_scale = 1.0f;

  // tensor types
  NVTEDType qkv_dtype = kNVTEBFloat16;
  NVTEDType o_dtype = kNVTEBFloat16;
  NVTEDType do_dtype = kNVTEBFloat16;
  NVTEDType dqkv_dtype = kNVTEBFloat16;

  // tensor layouts
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Layout dqkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_scale_inv_format = NVTE_QKV_Format_NOT_SET;

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

  // Internal-only fields: never part of attribute serialization, operator<, or the graph cache key.
  // Filled by derive() or set by caller (i.e. is_forward). Added for convinence purposes and do not
  // represent graph properties.

  // Direction to build the cuDNN graph for; steers make_cache_key() normalization.
  bool is_forward = false;
  // THD batch/token counts; make_cache_key() folds these into batch_size/max_seqlen_*.
  size_t bucketed_batch_size = 0;
  size_t bucketed_num_tokens_q = 0;
  size_t bucketed_num_tokens_kv = 0;
  // Uses cu_seqlens or actual_seqlens.
  bool uses_cu_seqlens_directly = false;
  // Convinence fields to avoid recompute.
  NVTE_QKV_Format q_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format kv_format = NVTE_QKV_Format_NOT_SET;
  bool is_ragged_q = false;
  bool is_ragged_kv = false;
  bool is_paged_kv = false;
  bool is_padding = false;
  bool is_causal = false;
  bool is_causal_bottom_right = false;

  // Generated from TE_FUSED_ATTN_CACHE_KEY_FIELDS so the per-attribute serialized sizes stay in lockstep
  // with the field list (and, via the static_assert below, with NVTEFusedAttnConfigAttribute).
  static constexpr size_t attr_sizes[] = {
#define TE_FUSED_ATTN_CACHE_KEY_FIELD_SIZE(member, wire, fmt, cast, label) sizeof(wire),
      TE_FUSED_ATTN_CACHE_KEY_FIELDS(TE_FUSED_ATTN_CACHE_KEY_FIELD_SIZE)
#undef TE_FUSED_ATTN_CACHE_KEY_FIELD_SIZE
  };

  // Tuple of all cache-relevant fields, generated from TE_FUSED_ATTN_CACHE_KEY_FIELDS. The trailing 0
  // sentinel absorbs the macro's trailing comma; it is identical on both operands so it never
  // affects ordering. Used by operator< so the comparison can never omit a field.
  auto cache_key_tuple() const {
    return std::make_tuple(
#define TE_FUSED_ATTN_CACHE_KEY_FIELD_VALUE(member, wire, fmt, cast, label) member,
        TE_FUSED_ATTN_CACHE_KEY_FIELDS(TE_FUSED_ATTN_CACHE_KEY_FIELD_VALUE)
#undef TE_FUSED_ATTN_CACHE_KEY_FIELD_VALUE
        0);
  }

  bool operator<(const FusedAttnConfig &rhs) const {
    return cache_key_tuple() < rhs.cache_key_tuple();
  }

  // Derive fields such as bucketed batch_size or num_tokens for THD, based on input fields
  // that have been set by the caller.
  void derive();

  // Return a normalized copy of this config to be used as a key for the cuDNN graph cache.
  // It drops fields that are invariant (e.g. attn_scale) or irrelevant (e.g. dO/dQKV dtypes
  // and `deterministic` for forward, and `return_max_logit` for backward) to the corresponding graph.
  // This helps avoid redundant graph builds and cache misses.
  FusedAttnConfig make_cache_key() const;
};

// Cross-check the generated field list against the public attribute enum: if a cache-relevant field
// is added to TE_FUSED_ATTN_CACHE_KEY_FIELDS without a matching NVTEFusedAttnConfigAttribute entry (or
// vice versa), this fails to compile instead of silently corrupting attribute (de)serialization.
static_assert(sizeof(FusedAttnConfig::attr_sizes) / sizeof(FusedAttnConfig::attr_sizes[0]) ==
                  kNVTEFusedAttnConfigNumAttributes,
              "TE_FUSED_ATTN_CACHE_KEY_FIELDS is out of sync with NVTEFusedAttnConfigAttribute; "
              "update both together.");

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
  bool is_training = true;
  bool cuda_graph = false;
  bool return_max_logit = false;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  float dropout = 0.0f;
  float attn_scale = 1.0f;
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
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
      sizeof(uint8_t),            // is_training
      sizeof(uint8_t),            // cuda_graph
      sizeof(uint8_t),            // return_max_logit
      sizeof(NVTE_Mask_Type),     // attn_mask_type
      sizeof(NVTE_Bias_Type),     // bias_type
      sizeof(NVTE_Softmax_Type),  // softmax_type
      sizeof(int64_t),            // window_size_left
      sizeof(int64_t),            // window_size_right
      sizeof(uint8_t),            // bottom_right_diagonal
      sizeof(float),              // dropout
      sizeof(float),              // attn_scale
      sizeof(NVTE_QKV_Layout),    // qkv_layout
      sizeof(NVTE_QKV_Format),    // o_format
      sizeof(NVTE_QKV_Format),    // qkv_scale_inv_format
      sizeof(size_t),             // max_seqlen_q
      sizeof(size_t),             // max_seqlen_kv
      sizeof(NVTETensor),         // workspace
      sizeof(cudaStream_t),       // stream
  };

  // Build a FusedAttnConfig from the scalar "knobs" carried here (e.g. attn_mask_type, bias_type)
  // and the fields derived from the tensor handles (dtypes, dims, scaling mode, paged-KV and bias
  // broadcast shapes). Returns the real execution config; call FusedAttnConfig::make_cache_key on
  // it to obtain the normalized cuDNN graph-cache key.
  FusedAttnConfig make_config() const;
};

inline const FusedAttnFwdParams *get_fused_attn_fwd_params(NVTEFusedAttnFwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnFwdParams must not be NULL.");
  return reinterpret_cast<const FusedAttnFwdParams *>(params);
}

inline FusedAttnFwdParams *get_fused_attn_fwd_params_mutable(NVTEFusedAttnFwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnFwdParams must not be NULL.");
  return reinterpret_cast<FusedAttnFwdParams *>(params);
}

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
  bool cuda_graph = false;
  bool deterministic = false;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  float dropout = 0.0f;
  float attn_scale = 1.0f;
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Layout dqkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
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
      sizeof(uint8_t),                 // cuda_graph
      sizeof(uint8_t),                 // deterministic
      sizeof(NVTE_Mask_Type),          // attn_mask_type
      sizeof(NVTE_Bias_Type),          // bias_type
      sizeof(NVTE_Softmax_Type),       // softmax_type
      sizeof(int64_t),                 // window_size_left
      sizeof(int64_t),                 // window_size_right
      sizeof(uint8_t),                 // bottom_right_diagonal
      sizeof(float),                   // dropout
      sizeof(float),                   // attn_scale
      sizeof(NVTE_QKV_Layout),         // qkv_layout
      sizeof(NVTE_QKV_Format),         // o_format
      sizeof(NVTE_QKV_Format),         // do_format
      sizeof(NVTE_QKV_Layout),         // dqkv_layout
      sizeof(NVTE_QKV_Format),         // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),         // do_scale_inv_format
      sizeof(size_t),                  // max_seqlen_q
      sizeof(size_t),                  // max_seqlen_kv
      sizeof(NVTETensor),              // workspace
      sizeof(cudaStream_t),            // stream
  };

  // Build a FusedAttnConfig from the scalar "knobs" carried here (e.g. attn_mask_type, bias_type)
  // and the fields derived from the tensor handles (e.g. dtypes, dims, scaling mode and bias broadcast
  // shape). Returns the real execution config; call FusedAttnConfig::make_cache_key on it to
  // obtain the normalized cuDNN graph-cache key.
  FusedAttnConfig make_config() const;
};

inline const FusedAttnBwdParams *get_fused_attn_bwd_params(NVTEFusedAttnBwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnBwdParams must not be NULL.");
  return reinterpret_cast<const FusedAttnBwdParams *>(params);
}

inline FusedAttnBwdParams *get_fused_attn_bwd_params_mutable(NVTEFusedAttnBwdParams params) {
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnBwdParams must not be NULL.");
  return reinterpret_cast<FusedAttnBwdParams *>(params);
}

}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_
