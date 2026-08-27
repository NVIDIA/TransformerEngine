/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file config_and_params.h
 *  \brief Internal objects for fused-attention config and parameter handles.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_CONFIG_AND_PARAMS_H_

#include <string>
#include <tuple>

#include "common/common.h"
#include "transformer_engine/fused_attn.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn {

enum class Backend { F16, FP8 };
enum class Pass { Fwd, Bwd };

inline constexpr const char *backend_name(Backend b) { return b == Backend::F16 ? "f16" : "fp8"; }
inline constexpr const char *pass_name(Pass p) { return p == Pass::Fwd ? "fwd" : "bwd"; }

struct FusedAttnConfig {
  // Basic attention settings
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

  // Tensor types
  NVTEDType qkv_dtype = kNVTEBFloat16;
  NVTEDType o_dtype = kNVTEBFloat16;
  NVTEDType do_dtype = kNVTEBFloat16;
  NVTEDType dqkv_dtype = kNVTEBFloat16;

  // Tensor layouts
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Layout dqkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format do_scale_inv_format = NVTE_QKV_Format_NOT_SET;

  // Tensor dimensions
  size_t batch_size = 0;
  size_t num_attn_heads = 0;
  size_t num_gqa_groups = 0;
  size_t head_dim_qk = 0;
  size_t head_dim_v = 0;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
  size_t num_tokens_q = 0;
  size_t num_tokens_kv = 0;

  // Paged KV dimensions
  size_t num_pages_k = 0;
  size_t num_pages_v = 0;
  size_t page_size_k = 0;
  size_t page_size_v = 0;
  size_t max_pages_per_seq_k = 0;
  size_t max_pages_per_seq_v = 0;

  // Bias dimensions
  size_t bias_batch_size = 0;
  size_t bias_num_heads = 0;
  size_t bias_seqlen_q = 0;
  size_t bias_seqlen_kv = 0;

  // ============================================================================
  // Internal fields: not part of attr_sizes[] or NVTEFusedAttnConfigAttribute, and
  // unreachable from nvte_set_fused_attn_config_attribute, hence "internal".
  //
  // - Keyed: do participate in operator<, and can distinguish graphs in the cache,
  //   e.g. device_id;
  // - Unkeyed: do not participate in operator<, and can not distinguish graphs
  //   in the cache; used for convenience purposes only; run derive() to fill them
  // ============================================================================

  // Keyed:
  // - distinguish graphs on different GPUs in a single-process run
  int device_id = -1;

  // Unkeyed:
  // - support query directions
  bool check_for_forward_support = true;
  bool check_for_backward_support = true;
  // - whether derived fields have been filled
  bool is_derived = false;
  // - common attributes
  NVTE_QKV_Format qkv_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format q_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format kv_format = NVTE_QKV_Format_NOT_SET;
  bool is_ragged_q = false;
  bool is_ragged_kv = false;
  bool is_paged_kv = false;
  bool is_padding = false;
  bool is_causal = false;
  bool is_causal_bottom_right = false;
  bool is_bias = false;
  bool is_alibi = false;
  bool is_softmax_offset = false;
  bool is_dropout = false;
  // - FP8 recipes
  bool is_o_in_fp8 = false;
  bool is_dqkv_in_fp8 = false;
  bool is_o_in_f16 = false;
  bool is_dqkv_in_f16 = false;
  bool is_tensor_scaling = false;
  bool is_mxfp8 = false;
  bool is_delayed_scaling_fwd = false;
  bool is_delayed_scaling_bwd = false;
  bool is_current_scaling_fwd = false;
  bool is_current_scaling_bwd = false;
  bool is_mxfp8_fwd = false;
  bool is_mxfp8_bwd = false;
  // - cu_seqlens vs actual_seqlens for THD or padding masks
  bool uses_cu_seqlens_directly = false;
  // - bucket the batch size and token counts for THD
  size_t bucketed_batch_size = 0;
  size_t bucketed_num_tokens_q = 0;
  size_t bucketed_num_tokens_kv = 0;
  // - packed (TH1) vs dense (BHS1) graphs and stats
  bool uses_ragged_graph = false;
  bool uses_ragged_stats = false;
  size_t graph_batch_size_fwd = 0;
  size_t graph_batch_size_bwd = 0;
  size_t graph_max_seqlen_q = 0;
  size_t graph_max_seqlen_kv = 0;
  // - ragged offset widths and multipliers
  bool needs_64bit_ragged_offset = false;
  DType ragged_offset_type_fwd = DType::kInt32;
  DType ragged_offset_type_bwd = DType::kInt32;
  RaggedOffsetMultipliers ragged_offset_mults;

  static constexpr size_t attr_sizes[] = {
      // Basic attention settings
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
      sizeof(float),              // attn_scale
      // Tensor types
      sizeof(NVTEDType),  // qkv_dtype
      sizeof(NVTEDType),  // o_dtype
      sizeof(NVTEDType),  // do_dtype
      sizeof(NVTEDType),  // dqkv_dtype
      // Tensor layouts
      sizeof(NVTE_QKV_Layout),  // qkv_layout
      sizeof(NVTE_QKV_Format),  // o_format
      sizeof(NVTE_QKV_Format),  // do_format
      sizeof(NVTE_QKV_Layout),  // dqkv_layout
      sizeof(NVTE_QKV_Format),  // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),  // do_scale_inv_format
      // Tensor dimensions
      sizeof(size_t),  // batch_size
      sizeof(size_t),  // num_attn_heads
      sizeof(size_t),  // num_gqa_groups
      sizeof(size_t),  // head_dim_qk
      sizeof(size_t),  // head_dim_v
      sizeof(size_t),  // max_seqlen_q
      sizeof(size_t),  // max_seqlen_kv
      sizeof(size_t),  // num_tokens_q
      sizeof(size_t),  // num_tokens_kv
      // Paged KV dimensions
      sizeof(size_t),  // num_pages_k
      sizeof(size_t),  // num_pages_v
      sizeof(size_t),  // page_size_k
      sizeof(size_t),  // page_size_v
      sizeof(size_t),  // max_pages_per_seq_k
      sizeof(size_t),  // max_pages_per_seq_v
      // Bias dimensions
      sizeof(size_t),  // bias_batch_size
      sizeof(size_t),  // bias_num_heads
      sizeof(size_t),  // bias_seqlen_q
      sizeof(size_t),  // bias_seqlen_kv
  };

  static_assert(sizeof(attr_sizes) / sizeof(attr_sizes[0]) == kNVTEFusedAttnConfigNumAttributes,
                "attr_sizes must have one entry per NVTEFusedAttnConfigAttribute; add the size of "
                "the new attribute alongside its enumerator.");

  bool operator<(const FusedAttnConfig &rhs) const {
    return std::tie(is_training, deterministic, cuda_graph, return_max_logit, attn_mask_type,
                    bias_type, window_size_left, window_size_right, bottom_right_diagonal,
                    softmax_type, scaling_mode, dropout, attn_scale, qkv_dtype, o_dtype, do_dtype,
                    dqkv_dtype, qkv_layout, o_format, do_format, dqkv_layout, qkv_scale_inv_format,
                    do_scale_inv_format, batch_size, num_attn_heads, num_gqa_groups, head_dim_qk,
                    head_dim_v, max_seqlen_q, max_seqlen_kv, num_tokens_q, num_tokens_kv,
                    num_pages_k, num_pages_v, page_size_k, page_size_v, max_pages_per_seq_k,
                    max_pages_per_seq_v, bias_batch_size, bias_num_heads, bias_seqlen_q,
                    bias_seqlen_kv, device_id) <
           std::tie(rhs.is_training, rhs.deterministic, rhs.cuda_graph, rhs.return_max_logit,
                    rhs.attn_mask_type, rhs.bias_type, rhs.window_size_left, rhs.window_size_right,
                    rhs.bottom_right_diagonal, rhs.softmax_type, rhs.scaling_mode, rhs.dropout,
                    rhs.attn_scale, rhs.qkv_dtype, rhs.o_dtype, rhs.do_dtype, rhs.dqkv_dtype,
                    rhs.qkv_layout, rhs.o_format, rhs.do_format, rhs.dqkv_layout,
                    rhs.qkv_scale_inv_format, rhs.do_scale_inv_format, rhs.batch_size,
                    rhs.num_attn_heads, rhs.num_gqa_groups, rhs.head_dim_qk, rhs.head_dim_v,
                    rhs.max_seqlen_q, rhs.max_seqlen_kv, rhs.num_tokens_q, rhs.num_tokens_kv,
                    rhs.num_pages_k, rhs.num_pages_v, rhs.page_size_k, rhs.page_size_v,
                    rhs.max_pages_per_seq_k, rhs.max_pages_per_seq_v, rhs.bias_batch_size,
                    rhs.bias_num_heads, rhs.bias_seqlen_q, rhs.bias_seqlen_kv, rhs.device_id);
  }

  // Derive relevant fields based on input fields that have been set by the caller. They are
  // read by the graph build, cache lookup, and support query.
  void derive();

  // Assert that derive() has run, for code about to read a derived field.
  void check_derived() const {
    NVTE_CHECK(is_derived,
               "FusedAttnConfig's derived fields are not set. Please run "
               "FusedAttnConfig::derive() first.");
  }

  // Return a normalized copy of this config to be used as a key for the cuDNN graph cache.
  // It drops fields that are either invariant (e.g. attn_scale) or irrelevant (e.g. dO/dQKV dtypes
  // and `deterministic` for forward, and `return_max_logit` for backward).
  FusedAttnConfig make_cache_key(Pass pass) const;

  // Return a string representation of this config for level-2 cache diagnostics.
  std::string to_string() const;
};

inline const FusedAttnConfig *get_fused_attn_config(NVTEFusedAttnConfig config) {
  NVTE_CHECK(config != nullptr, "NVTEFusedAttnConfig must not be NULL.");
  return reinterpret_cast<const FusedAttnConfig *>(config);
}

inline FusedAttnConfig *get_fused_attn_config_mutable(NVTEFusedAttnConfig config) {
  NVTE_CHECK(config != nullptr, "NVTEFusedAttnConfig must not be NULL.");
  return reinterpret_cast<FusedAttnConfig *>(config);
}

struct FusedAttnFwdParams {
  // Input tensors
  NVTETensor Q = nullptr;
  NVTETensor K = nullptr;
  NVTETensor V = nullptr;
  NVTETensor Bias = nullptr;
  NVTETensor SoftmaxOffset = nullptr;
  // Intermediate tensors
  NVTETensor S = nullptr;
  // Output tensor
  NVTETensor O = nullptr;
  // Auxiliary context tensor pack
  NVTETensorPack *Aux_CTX_Tensors = nullptr;
  // Miscellaneous tensors
  NVTETensor cu_seqlens_q = nullptr;
  NVTETensor cu_seqlens_kv = nullptr;
  NVTETensor cu_seqlens_q_padded = nullptr;
  NVTETensor cu_seqlens_kv_padded = nullptr;
  NVTETensor page_table_k = nullptr;
  NVTETensor page_table_v = nullptr;
  NVTETensor rng_state = nullptr;
  // Scalars
  bool is_training = true;
  bool cuda_graph = false;
  bool return_max_logit = false;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
  float dropout = 0.0f;
  float attn_scale = 1.0f;
  NVTE_QKV_Layout qkv_layout = NVTE_QKV_Layout_NOT_SET;
  NVTE_QKV_Format o_format = NVTE_QKV_Format_NOT_SET;
  NVTE_QKV_Format qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET;
  size_t max_seqlen_q = 0;
  size_t max_seqlen_kv = 0;
  // Workspace and stream
  NVTETensor workspace = nullptr;
  cudaStream_t stream = nullptr;

  static constexpr size_t attr_sizes[] = {
      // Tensor handles
      sizeof(NVTETensor),        // Q
      sizeof(NVTETensor),        // K
      sizeof(NVTETensor),        // V
      sizeof(NVTETensor),        // Bias
      sizeof(NVTETensor),        // SoftmaxOffset
      sizeof(NVTETensor),        // S
      sizeof(NVTETensor),        // O
      sizeof(NVTETensorPack *),  // Aux_CTX_Tensors
      sizeof(NVTETensor),        // cu_seqlens_q
      sizeof(NVTETensor),        // cu_seqlens_kv
      sizeof(NVTETensor),        // cu_seqlens_q_padded
      sizeof(NVTETensor),        // cu_seqlens_kv_padded
      sizeof(NVTETensor),        // page_table_k
      sizeof(NVTETensor),        // page_table_v
      sizeof(NVTETensor),        // rng_state
      // Configuration knobs
      sizeof(uint8_t),            // is_training
      sizeof(uint8_t),            // cuda_graph
      sizeof(uint8_t),            // return_max_logit
      sizeof(NVTE_Mask_Type),     // attn_mask_type
      sizeof(NVTE_Bias_Type),     // bias_type
      sizeof(int64_t),            // window_size_left
      sizeof(int64_t),            // window_size_right
      sizeof(uint8_t),            // bottom_right_diagonal
      sizeof(NVTE_Softmax_Type),  // softmax_type
      sizeof(float),              // dropout
      sizeof(float),              // attn_scale
      sizeof(NVTE_QKV_Layout),    // qkv_layout
      sizeof(NVTE_QKV_Format),    // o_format
      sizeof(NVTE_QKV_Format),    // qkv_scale_inv_format
      sizeof(size_t),             // max_seqlen_q
      sizeof(size_t),             // max_seqlen_kv
      // Workspace and stream
      sizeof(NVTETensor),    // workspace
      sizeof(cudaStream_t),  // stream
  };

  static_assert(sizeof(attr_sizes) / sizeof(attr_sizes[0]) == kNVTEFusedAttnFwdParamsNumAttributes,
                "attr_sizes must have one entry per NVTEFusedAttnFwdParamsAttribute; add the size "
                "of the new attribute alongside its enumerator.");

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
  // Input tensors
  NVTETensor Q = nullptr;
  NVTETensor K = nullptr;
  NVTETensor V = nullptr;
  NVTETensor O = nullptr;
  NVTETensor dO = nullptr;
  NVTETensor S = nullptr;
  NVTETensor dP = nullptr;
  const NVTETensorPack *Aux_CTX_Tensors = nullptr;
  // Output tensors
  NVTETensor dQ = nullptr;
  NVTETensor dK = nullptr;
  NVTETensor dV = nullptr;
  NVTETensor dBias = nullptr;
  NVTETensor dSoftmaxOffset = nullptr;
  // Miscellaneous tensors
  NVTETensor cu_seqlens_q = nullptr;
  NVTETensor cu_seqlens_kv = nullptr;
  NVTETensor cu_seqlens_q_padded = nullptr;
  NVTETensor cu_seqlens_kv_padded = nullptr;
  // Scalars
  bool deterministic = false;
  bool cuda_graph = false;
  NVTE_Mask_Type attn_mask_type = NVTE_NO_MASK;
  NVTE_Bias_Type bias_type = NVTE_NO_BIAS;
  int64_t window_size_left = -1;
  int64_t window_size_right = -1;
  bool bottom_right_diagonal = true;
  NVTE_Softmax_Type softmax_type = NVTE_VANILLA_SOFTMAX;
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
  // Workspace and stream
  NVTETensor workspace = nullptr;
  cudaStream_t stream = nullptr;

  static constexpr size_t attr_sizes[] = {
      // Tensor handles
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
      // Configuration knobs
      sizeof(uint8_t),            // deterministic
      sizeof(uint8_t),            // cuda_graph
      sizeof(NVTE_Mask_Type),     // attn_mask_type
      sizeof(NVTE_Bias_Type),     // bias_type
      sizeof(int64_t),            // window_size_left
      sizeof(int64_t),            // window_size_right
      sizeof(uint8_t),            // bottom_right_diagonal
      sizeof(NVTE_Softmax_Type),  // softmax_type
      sizeof(float),              // dropout
      sizeof(float),              // attn_scale
      sizeof(NVTE_QKV_Layout),    // qkv_layout
      sizeof(NVTE_QKV_Format),    // o_format
      sizeof(NVTE_QKV_Format),    // do_format
      sizeof(NVTE_QKV_Layout),    // dqkv_layout
      sizeof(NVTE_QKV_Format),    // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),    // do_scale_inv_format
      sizeof(size_t),             // max_seqlen_q
      sizeof(size_t),             // max_seqlen_kv
      // Workspace and stream
      sizeof(NVTETensor),    // workspace
      sizeof(cudaStream_t),  // stream
  };

  static_assert(sizeof(attr_sizes) / sizeof(attr_sizes[0]) == kNVTEFusedAttnBwdParamsNumAttributes,
                "attr_sizes must have one entry per NVTEFusedAttnBwdParamsAttribute; add the size "
                "of the new attribute alongside its enumerator.");

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
