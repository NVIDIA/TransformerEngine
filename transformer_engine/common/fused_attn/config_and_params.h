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

// The pair that names one build site: whose graphs, and which of the two a config is being turned
// into. A site keeps its own graph cache and its own counters, so both halves travel together.
//
// Declared here, with the config, rather than with the cache or its diagnostics: they are the
// vocabulary those two share, and the config is where their consumers start -- make_cache_key()
// below turns a config into one site's key, and derive() fills the dimensions built from it.
//
// Backend::F16 is the arbitrary-seqlen backend; the max512 one keeps no graph cache, so it has no
// site here. Narrower than the public NVTE_Fused_Attn_Backend, and not a substitute for it: this
// names only the backends that build graphs.
enum class Backend { F16, FP8 };

// Passed in rather than derived, because a config cannot say which graph is being built from it.
// check_for_forward_support and check_for_backward_support state which directions a caller wants
// probed, and a backend query arriving from a framework has both set, so they answer a different
// question -- see the comment on them below.
enum class Pass { Fwd, Bwd };

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

  // device ID: not part of attribute serialization, but part of operator< and used to
  // differentiate graphs built for different devices in multi-GPU single-process runs
  int device_id = -1;

  // Internal-only fields: never part of attribute serialization, operator<, or the graph cache key.
  // Filled by derive() or set by caller (i.e. check_for_forward_support). Added for convinence
  // purposes and do not represent any graph properties.
  //
  // The two below say which directions nvte_get_fused_attn_backend_v2 should probe, and nothing
  // else: not which graph is being built, which is what Pass names. They default to true and the
  // attribute API cannot reach them, so a backend query from a framework asks about both
  // directions, while the execution entry points set the one they are about to run.
  bool check_for_forward_support = true;
  bool check_for_backward_support = true;
  // Whether derive() has run, i.e. whether the fields below hold anything. Every consumer of a
  // derived field needs them filled -- an unfilled config yields a graph with the wrong shapes
  // and a cache key that collides with unrelated configs, neither of which announces itself --
  // so this exists to let those consumers assert rather than trust. Not a cached-result marker:
  // derive() recomputes unconditionally, so a config whose inputs change can simply be re-derived.
  bool is_derived = false;
  // THD batch/token counts, the raw buckets. The graph dimensions built out of them are
  // graph_max_seqlen_* below and, because the batch is direction-dependent, graph_batch_size_*.
  size_t bucketed_batch_size = 0;
  size_t bucketed_num_tokens_q = 0;
  size_t bucketed_num_tokens_kv = 0;
  // Uses cu_seqlens or actual_seqlens. One answer per backend, because the same question has two:
  // the FP8 graphs need newer cuDNN and frontend versions for it than the F16 ones, so a config
  // that can hand cu_seqlens straight to one cannot necessarily hand them to the other. Each
  // backend reads its own and no more; nothing reads both.
  bool uses_cu_seqlens_directly = false;
  bool fp8_uses_cu_seqlens_directly = false;
  // Whether a ragged (THD) graph is built at packed token-count dimensions with ragged Stats/LSE,
  // rather than at dense max_seqlen ones. Held here rather than asked for at each of the places
  // that need it -- graph_max_seqlen_* and graph_batch_size_* below -- because the key and the
  // graph have to be built at the same dimensions, and two independent queries are two chances to
  // disagree. Unlike the flags above, this one depends on the device as
  // well as the cuDNN version, so a config carries the answer for the device it was derived on;
  // every entry point derives immediately before use, and the cache key records device_id.
  bool uses_packed_ragged_graph = false;
  // Whether the graph's Stats/LSE tensor is the packed, token-indexed one. Ragged Q is necessary
  // but not sufficient, since the packed representation also needs an architecture that supports
  // it. Derived because three unrelated places read it -- the graph build, the pointer binding at
  // execution, and the Stats/Max shapes reported back to the framework -- and they are describing
  // one buffer, so they cannot be allowed to disagree about its shape.
  bool uses_ragged_stats = false;
  // The sequence lengths the graph is built at: max_seqlen_* for a dense graph, and the bucketed
  // token counts where a ragged layout is packed. Held here because the cache key has to name the
  // dimensions the graph was built with -- a key that says otherwise is a hit on a graph of the
  // wrong shape -- and stating the substitution once is what keeps make_cache_key() and the graph
  // builders from drifting. Both passes build at the same sequence lengths; the batch size is the
  // one dimension they disagree on, which is why that one is a pair below rather than a single
  // field here.
  size_t graph_max_seqlen_q = 0;
  size_t graph_max_seqlen_kv = 0;
  // The batch size the graph is built at and the width it expects ragged (THD) offsets in, one of
  // each per direction. Pairs rather than one value apiece because the passes disagree on both,
  // and a config is derived once and then probed and built for either direction, so no single
  // field could answer: the selector derives a config and asks about forward and backward off that
  // one copy. Whoever reads them names the direction they are building or keying for -- the graph
  // builders, the code that binds runtime pointers to the built graph, and make_cache_key(), all
  // of which have to agree, since a disagreement is a graph whose bound pointers do not describe
  // the dimensions it was built at. All four are set together by the one condition in derive().
  size_t graph_batch_size_fwd = 0;
  size_t graph_batch_size_bwd = 0;
  DType ragged_offset_type_fwd = DType::kInt32;
  DType ragged_offset_type_bwd = DType::kInt32;
  // Elements per token for each ragged tensor, from the layout group and the head dimensions.
  // Shared with the cu_seqlens_padded_to_offsets kernel, so the offsets the graph is told to
  // expect and the offsets that are written cannot drift apart.
  RaggedOffsetMultipliers ragged_offset_mults;
  // Convinence fields to avoid recompute.
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
  bool is_mxfp8 = false;
  // Whether the graph has a dropout node. The is_training term is what makes this one worth having
  // as a field: a backward graph is only ever built for training, so the two directions used to
  // spell this differently -- forward with the term, backward without -- and agreed only because
  // every config that reaches a backward build has is_training set. One field states the rule the
  // forward way, which is the safe way round: if that ever stops holding, a backward graph loses
  // its dropout node rather than gaining one the cache key does not name.
  bool is_dropout = false;
  // Whether what each pass stores is itself quantized: O for a forward graph, dQKV for a backward
  // one. Only the FP8 backend asks, and for it this is the whole of what separates the two
  // tensor-scaling recipes -- FP8 out means the scale is known before the graph is built, F16 out
  // means the graph has to compute it. Derived rather than asked at each build site so that the
  // pairing of a pass with the tensor it writes is stated once.
  //
  // The FP8 builders read "not FP8" as "F16", which holds only because
  // nvte_get_fused_attn_backend_v2 refuses an FP8 config whose output is neither before any graph
  // is built; that refusal and these two fields are the same rule read from its two ends.
  bool o_is_fp8 = false;
  bool dqkv_is_fp8 = false;

  static constexpr size_t attr_sizes[] = {
      // basic attention settings
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
      // tensor types
      sizeof(NVTEDType),  // qkv_dtype
      sizeof(NVTEDType),  // o_dtype
      sizeof(NVTEDType),  // do_dtype
      sizeof(NVTEDType),  // dqkv_dtype
      // tensor layouts
      sizeof(NVTE_QKV_Layout),  // qkv_layout
      sizeof(NVTE_QKV_Format),  // o_format
      sizeof(NVTE_QKV_Format),  // do_format
      sizeof(NVTE_QKV_Layout),  // dqkv_layout
      sizeof(NVTE_QKV_Format),  // qkv_scale_inv_format
      sizeof(NVTE_QKV_Format),  // do_scale_inv_format
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

  // The public header asks contributors to append to NVTEFusedAttnConfigAttribute, and the
  // accessors index attr_sizes[attr] after checking only that attr is below the sentinel. An
  // enumerator added without its size here would therefore read one past the end of this array,
  // silently and only for the new attribute. Tying the two together turns that into a build
  // failure at the line that has to change.
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

  // Derive fields such as bucketed batch_size or num_tokens for THD, based on input fields
  // that have been set by the caller. Call once, after the last input field is set and before
  // the config reaches a graph build, a cache lookup, or a support query -- all of which read
  // derived fields.
  //
  // Called by whoever owns the config, at the point it stops being edited: the execution entry
  // points (nvte_fused_attn_fwd_v2 and its backward counterpart) on the config they go on to run,
  // and nvte_get_fused_attn_backend_v2() on a copy of the caller's, so that asking whether a
  // configuration is supported does not modify it. Nothing further in is expected to derive
  // again, and check_derived() is what holds them to that. Idempotent, so a config that is
  // derived and then re-derived is unharmed.
  //
  // Throws for combinations of input fields that no graph can serve, so that all four graph
  // builders inherit the rule from one place. Those same combinations are stated as rejection
  // rules in nvte_get_fused_attn_backend_v2(), ahead of its derive() call, so that asking whether
  // such a configuration is supported gets an answer instead of an exception.
  void derive();

  // Return a normalized copy of this config to be used as a key for the cuDNN graph cache.
  // Requires a config that has been through derive(), whose fields the normalizations read.
  // It drops fields that are invariant (e.g. attn_scale) or irrelevant (e.g. dO/dQKV dtypes
  // and `deterministic` for forward, and `return_max_logit` for backward) to the corresponding graph.
  // This helps avoid redundant graph builds and cache misses.
  //
  // `pass` is which graph the key is for. It decides both direction-dependent normalizations --
  // which fields are dropped, and whether the batch is bucketed -- so a key built for one pass
  // cannot be handed to the other's cache.
  FusedAttnConfig make_cache_key(Pass pass) const;

  // One line for the graph cache's level-2 trace: every field operator< compares, in its order,
  // less device_id, which the trace's own prefix prints as the dev column. Abbreviated and terse
  // on purpose, the reason to print a key at all being that two of these lines diff cleanly,
  // naming the fields that cost an extra graph build.
  //
  // Four fields it prints that operator< does not compare, because a line without them cannot
  // account for the values beside them: check_for_forward_support, which make_cache_key() sets
  // from the pass and so is what says which direction's key this is, and the three bucketed_*
  // inputs, which are what the normalization substituted into batch_size and the token counts.
  //
  // Defined here rather than with the diagnostics that print it because it is the third
  // enumeration of these fields, after attr_sizes and operator< above. A field added to the key
  // without being added here does not fail to build, it just stops appearing in the trace, so the
  // three lists are kept where one change can see all of them.
  std::string key_debug_string() const;
};

// Assert that `cfg` has been through derive(), for code about to read a derived field. Worth
// asserting rather than assuming because the failure is silent: an unset bucketed_batch_size or
// q_format reads as zero, which is a legal value that yields a graph of the wrong shape and a
// key that collides with unrelated configs. Deriving happens at the library's entry points rather
// than here, where it would be needed, so this is what keeps a new path into the builders from
// quietly skipping it. It catches a config that was never derived and nothing else: a config
// derived and then edited passes, so callers that change an input field re-derive rather than rely
// on this, which derive() being idempotent makes cheap.
inline void check_derived(const FusedAttnConfig &cfg) {
  NVTE_CHECK(cfg.is_derived,
             "FusedAttnConfig reached a graph build with its derived fields unset. Every config "
             "must pass through FusedAttnConfig::derive() first; see the entry points in "
             "fused_attn.cpp.");
}

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

  // See FusedAttnConfig::attr_sizes: an enumerator appended without a size here reads past the
  // end of this array.
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

  // See FusedAttnConfig::attr_sizes: an enumerator appended without a size here reads past the
  // end of this array.
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
