/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "config_and_params.h"

#include <cudnn.h>
#include <cudnn_frontend_version.h>

#include <cstring>

#include "../common.h"
#include "../util/cuda_runtime.h"

namespace {

void bool_to_uint8(bool in, void *out) {
  *reinterpret_cast<uint8_t *>(out) = static_cast<uint8_t>(in);
}

void uint8_to_bool(const void *in, bool &out) {
  out = static_cast<bool>(*reinterpret_cast<const uint8_t *>(in));
}

}  // namespace

namespace transformer_engine {

namespace fused_attn {
// Forward declarations from fused_attn/utils.h. Declared here to avoid pulling the heavy
// cuDNN frontend header into this plain C++ translation unit.
size_t get_max_batch_size(size_t batch_size);
size_t get_max_tokens(size_t num_tokens);
}  // namespace fused_attn

void FusedAttnConfig::derive() {
  const int64_t b = static_cast<int64_t>(batch_size);
  const int64_t sq = static_cast<int64_t>(max_seqlen_q);
  const int64_t skv = static_cast<int64_t>(max_seqlen_kv);

  // convenience fields
  q_format = nvte_get_q_format(qkv_layout);
  kv_format = nvte_get_kv_format(qkv_layout);
  const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  is_paged_kv = (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD);
  is_ragged_q = (q_format == NVTE_QKV_Format::NVTE_THD);
  is_ragged_kv = (kv_format == NVTE_QKV_Format::NVTE_THD);
  is_padding = (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
               (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) ||
               (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);
  is_causal = (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
              (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK);
  is_causal_bottom_right =
      (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK) ||
      (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);

  // bucket the THD (ragged) batch and token counts
  const size_t tokens_q = num_tokens_q != 0 ? num_tokens_q : static_cast<size_t>(b * sq);
  const size_t tokens_kv = num_tokens_kv != 0 ? num_tokens_kv : static_cast<size_t>(b * skv);
  bucketed_batch_size =
      (is_ragged_q || is_ragged_kv) ? fused_attn::get_max_batch_size(batch_size) : 0;
  bucketed_num_tokens_q = is_ragged_q ? fused_attn::get_max_tokens(tokens_q) : 0;
  bucketed_num_tokens_kv = is_ragged_kv ? fused_attn::get_max_tokens(tokens_kv) : 0;

  // use of cu_seqlens vs actual_seqlens
  const size_t cudnn_runtime_version = cudnnGetVersion();
  const bool is_dropout = is_training && dropout != 0.0f;
  uses_cu_seqlens_directly = CUDNN_FRONTEND_VERSION >= 12500 &&
                             (CUDNN_VERSION >= 92400 && cudnn_runtime_version >= 92400) &&
                             !is_dropout;

  // paged KV dimensions
  if (is_paged_kv) {
    if (num_pages_k == 0) {
      num_pages_k = static_cast<size_t>(b);
    }
    if (num_pages_v == 0) {
      num_pages_v = static_cast<size_t>(b);
    }
    if (page_size_k == 0) {
      page_size_k = static_cast<size_t>(skv);
    }
    if (page_size_v == 0) {
      page_size_v = static_cast<size_t>(skv);
    }
    if (max_pages_per_seq_k == 0) {
      max_pages_per_seq_k = 1;
    }
    if (max_pages_per_seq_v == 0) {
      max_pages_per_seq_v = 1;
    }
  }
}

FusedAttnConfig FusedAttnConfig::make_cache_key() const {
  FusedAttnConfig cache_cfg = *this;

  // Normalize bottom_right_diagonal (the cuDNN diagonal alignment). The impl only turns it into a
  // real causal band under `is_causal || is_causal_bottom_right` or a sliding window; otherwise the
  // alignment is inert, so canonicalize it (like attn_scale) to false. This keeps the backend
  // support probe (which passes a possibly-different brd, e.g. default false) and the real op on a
  // single cached graph.
  const bool has_window = cache_cfg.window_size_left != -1 || cache_cfg.window_size_right != -1;
  if (!cache_cfg.is_causal && !cache_cfg.is_causal_bottom_right && !has_window) {
    cache_cfg.bottom_right_diagonal = false;
  } else if (cache_cfg.is_causal_bottom_right &&
             cache_cfg.max_seqlen_q == cache_cfg.max_seqlen_kv && !cache_cfg.is_padding) {
    // square bottom-right causal collapses to top-left causal (mirrors the impl).
    cache_cfg.bottom_right_diagonal = false;
  }

  // Bucket THD (ragged) batch and token counts
  if (cache_cfg.is_ragged_q || cache_cfg.is_ragged_kv) {
    const auto cudnn_runtime_version = cudnnGetVersion();
    const int sm_arch_ = cuda::sm_arch(cuda::current_device());
    if (cudnn_runtime_version >= 90600 && sm_arch_ != 120) {
      if (cache_cfg.is_ragged_q) {
        cache_cfg.max_seqlen_q = cache_cfg.bucketed_num_tokens_q;
      }
      if (cache_cfg.is_ragged_kv) {
        cache_cfg.max_seqlen_kv = cache_cfg.bucketed_num_tokens_kv;
      }
      cache_cfg.num_tokens_q = 0;
      cache_cfg.num_tokens_kv = 0;
      const bool bucket_batch = !is_forward || !cache_cfg.uses_cu_seqlens_directly;
      if (bucket_batch) {
        cache_cfg.batch_size = cache_cfg.bucketed_batch_size;
      }
    }
  }

  // attn_scale is a pass-by-value graph input and different scales can share the same cached graph
  cache_cfg.attn_scale = 1.0f;

  // Restrict each direction's key to the fields its graph actually consumes, so
  // no redundant graphs are built and no cache misses either
  if (is_forward) {
    cache_cfg.do_dtype = kNVTEBFloat16;
    cache_cfg.dqkv_dtype = kNVTEBFloat16;
    cache_cfg.do_format = NVTE_QKV_Format_NOT_SET;
    cache_cfg.dqkv_layout = NVTE_QKV_Layout_NOT_SET;
    cache_cfg.do_scale_inv_format = NVTE_QKV_Format_NOT_SET;
    cache_cfg.deterministic = false;
  } else {
    cache_cfg.return_max_logit = false;
  }

  return cache_cfg;
}

FusedAttnConfig FusedAttnFwdParams::make_config() const {
  const FusedAttnFwdParams &params = *this;
  FusedAttnConfig cfg{};
  cfg.is_forward = true;
  cfg.is_training = params.is_training;
  cfg.deterministic = false;
  cfg.cuda_graph = params.cuda_graph;
  cfg.return_max_logit = params.return_max_logit;
  cfg.qkv_layout = params.qkv_layout;
  cfg.o_format = params.o_format;
  cfg.qkv_scale_inv_format = params.qkv_scale_inv_format;
  cfg.bias_type = params.bias_type;
  cfg.attn_mask_type = params.attn_mask_type;
  cfg.softmax_type = params.softmax_type;
  cfg.attn_scale = params.attn_scale;
  cfg.dropout = params.dropout;
  cfg.max_seqlen_q = params.max_seqlen_q;
  cfg.max_seqlen_kv = params.max_seqlen_kv;
  cfg.window_size_left = params.window_size_left;
  cfg.window_size_right = params.window_size_right;
  cfg.bottom_right_diagonal = params.bottom_right_diagonal;

  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(params.cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(params.cu_seqlens_kv);
  const Tensor *input_page_table_k = convertNVTETensorCheck(params.page_table_k);
  const Tensor *input_page_table_v = convertNVTETensorCheck(params.page_table_v);
  const Tensor *input_Q = convertNVTETensorCheck(params.Q);
  const Tensor *input_K = convertNVTETensorCheck(params.K);
  const Tensor *input_V = convertNVTETensorCheck(params.V);
  const Tensor *input_Bias = convertNVTETensorCheck(params.Bias);
  const Tensor *output_O = convertNVTETensorCheck(params.O);

  const NVTE_QKV_Format q_format = nvte_get_q_format(params.qkv_layout);
  const NVTE_QKV_Format kv_format = nvte_get_kv_format(params.qkv_layout);
  auto *q_dims = input_Q->data.shape.data();
  auto *k_dims = input_K->data.shape.data();
  auto *v_dims = input_V->scaling_mode != NVTE_MXFP8_1D_SCALING
                     ? input_V->data.shape.data()
                     : input_V->columnwise_data.shape.data();
  AttentionShape q_shape(q_format, q_dims);
  AttentionShape k_shape(kv_format, k_dims);
  AttentionShape v_shape(kv_format, v_dims);
  size_t b = q_shape.b(), h_q = q_shape.h(), d_qk = q_shape.d(), t_q = q_shape.t();
  size_t h_kv = k_shape.h(), t_kv = k_shape.t(), d_v = v_shape.d();
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    b = input_cu_seqlens_q->data.shape[0] - 1;
  } else if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    b = input_cu_seqlens_kv->data.shape[0] - 1;
  }

  int64_t num_pages_k = 0, num_pages_v = 0, page_size_k = 0, page_size_v = 0;
  int64_t max_pages_per_seq_k = 0, max_pages_per_seq_v = 0;
  if (input_page_table_k->data.dptr != nullptr) {
    max_pages_per_seq_k = input_page_table_k->data.shape[1];
  }
  if (input_page_table_v->data.dptr != nullptr) {
    max_pages_per_seq_v = input_page_table_v->data.shape[1];
  }
  const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(params.qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) {
    const NVTE_QKV_Format paged_kv_format = nvte_get_kv_format(params.qkv_layout);
    if (paged_kv_format == NVTE_QKV_Format::NVTE_BSHD) {
      num_pages_k = input_K->data.shape[0];
      page_size_k = input_K->data.shape[1];
      num_pages_v = input_V->data.shape[0];
      page_size_v = input_V->data.shape[1];
    } else if (paged_kv_format == NVTE_QKV_Format::NVTE_SBHD) {
      num_pages_k = input_K->data.shape[1];
      page_size_k = input_K->data.shape[0];
      num_pages_v = input_V->data.shape[1];
      page_size_v = input_V->data.shape[0];
    }
  }

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  NVTE_CHECK(Q_type == KV_type, "Q and KV must have the same data type.");

  cfg.scaling_mode = input_Q->scaling_mode;
  cfg.qkv_dtype = Q_type;
  cfg.o_dtype = static_cast<NVTEDType>(output_O->data.dtype);
  cfg.batch_size = b;
  cfg.num_attn_heads = h_q;
  cfg.num_gqa_groups = h_kv;
  cfg.head_dim_qk = d_qk;
  cfg.head_dim_v = d_v;
  cfg.num_pages_k = static_cast<size_t>(num_pages_k);
  cfg.num_pages_v = static_cast<size_t>(num_pages_v);
  cfg.page_size_k = static_cast<size_t>(page_size_k);
  cfg.page_size_v = static_cast<size_t>(page_size_v);
  cfg.max_pages_per_seq_k = static_cast<size_t>(max_pages_per_seq_k);
  cfg.max_pages_per_seq_v = static_cast<size_t>(max_pages_per_seq_v);
  cfg.num_tokens_q = t_q;
  cfg.num_tokens_kv = t_kv;

  if ((params.bias_type != NVTE_NO_BIAS) && (params.bias_type != NVTE_ALIBI) &&
      input_Bias->data.shape.size() >= 4) {
    cfg.bias_batch_size = input_Bias->data.shape[0];
    cfg.bias_num_heads = input_Bias->data.shape[1];
    cfg.bias_seqlen_q = input_Bias->data.shape[2];
    cfg.bias_seqlen_kv = input_Bias->data.shape[3];
  }
  return cfg;
}

FusedAttnConfig FusedAttnBwdParams::make_config() const {
  const FusedAttnBwdParams &params = *this;
  FusedAttnConfig cfg{};
  cfg.is_training = true;
  cfg.deterministic = params.deterministic;
  cfg.cuda_graph = params.cuda_graph;
  cfg.return_max_logit = false;
  cfg.qkv_layout = params.qkv_layout;
  cfg.o_format = params.o_format;
  cfg.do_format = params.do_format;
  cfg.dqkv_layout = params.dqkv_layout;
  cfg.qkv_scale_inv_format = params.qkv_scale_inv_format;
  cfg.do_scale_inv_format = params.do_scale_inv_format;
  cfg.bias_type = params.bias_type;
  cfg.attn_mask_type = params.attn_mask_type;
  cfg.softmax_type = params.softmax_type;
  cfg.attn_scale = params.attn_scale;
  cfg.dropout = params.dropout;
  cfg.max_seqlen_q = params.max_seqlen_q;
  cfg.max_seqlen_kv = params.max_seqlen_kv;
  cfg.window_size_left = params.window_size_left;
  cfg.window_size_right = params.window_size_right;
  cfg.bottom_right_diagonal = params.bottom_right_diagonal;

  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(params.cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(params.cu_seqlens_kv);
  const Tensor *input_Q = convertNVTETensorCheck(params.Q);
  const Tensor *input_K = convertNVTETensorCheck(params.K);
  const Tensor *input_V = convertNVTETensorCheck(params.V);
  const Tensor *input_O = convertNVTETensorCheck(params.O);
  const Tensor *input_dO = convertNVTETensorCheck(params.dO);
  const Tensor *output_dQ = convertNVTETensorCheck(params.dQ);
  const Tensor *output_dBias = convertNVTETensorCheck(params.dBias);

  const NVTE_QKV_Format q_format = nvte_get_q_format(params.qkv_layout);
  const NVTE_QKV_Format kv_format = nvte_get_kv_format(params.qkv_layout);
  auto *q_dims = input_Q->data.shape.data();
  auto *k_dims = input_K->data.shape.data();
  auto *v_dims = input_V->data.shape.data();
  AttentionShape q_shape(q_format, q_dims);
  AttentionShape k_shape(kv_format, k_dims);
  AttentionShape v_shape(kv_format, v_dims);
  size_t b = q_shape.b(), h_q = q_shape.h(), d_qk = q_shape.d(), t_q = q_shape.t();
  size_t h_kv = k_shape.h(), t_kv = k_shape.t(), d_v = v_shape.d();
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    b = input_cu_seqlens_q->data.shape[0] - 1;
  } else if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    b = input_cu_seqlens_kv->data.shape[0] - 1;
  }

  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  NVTE_CHECK(Q_type == KV_type, "Q and KV must have the same data type.");

  cfg.scaling_mode = input_Q->scaling_mode;
  cfg.qkv_dtype = Q_type;
  cfg.o_dtype = static_cast<NVTEDType>(input_O->data.dtype);
  cfg.do_dtype = static_cast<NVTEDType>(input_dO->data.dtype);
  cfg.dqkv_dtype = static_cast<NVTEDType>(output_dQ->data.dtype);
  cfg.batch_size = b;
  cfg.num_attn_heads = h_q;
  cfg.num_gqa_groups = h_kv;
  cfg.head_dim_qk = d_qk;
  cfg.head_dim_v = d_v;
  cfg.num_tokens_q = t_q;
  cfg.num_tokens_kv = t_kv;

  if ((params.bias_type != NVTE_NO_BIAS) && (params.bias_type != NVTE_ALIBI) &&
      output_dBias->data.shape.size() >= 4) {
    cfg.bias_batch_size = output_dBias->data.shape[0];
    cfg.bias_num_heads = output_dBias->data.shape[1];
    cfg.bias_seqlen_q = output_dBias->data.shape[2];
    cfg.bias_seqlen_kv = output_dBias->data.shape[3];
  }
  return cfg;
}

}  // namespace transformer_engine

NVTEFusedAttnConfig nvte_create_fused_attn_config() {
  return new transformer_engine::FusedAttnConfig{};
}

void nvte_destroy_fused_attn_config(NVTEFusedAttnConfig config) {
  delete transformer_engine::get_fused_attn_config_mutable(config);
}

void nvte_get_fused_attn_config_attribute(NVTEFusedAttnConfig config,
                                          NVTEFusedAttnConfigAttribute attr, void *buf,
                                          size_t size_in_bytes, size_t *size_written) {
  using namespace transformer_engine;

  NVTE_CHECK(attr < kNVTEFusedAttnConfigNumAttributes, "Invalid NVTEFusedAttnConfigAttribute (got ",
             static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnConfig::attr_sizes[attr];
  if (size_written != nullptr) {
    *size_written = attr_size;
  }
  if (buf == nullptr) {
    return;
  }
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for fused attention config attribute (attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ", size_in_bytes,
             " bytes)");

  const auto &cfg = *get_fused_attn_config(config);
  switch (attr) {
    case kNVTEFusedAttnConfigIsTraining:
      bool_to_uint8(cfg.is_training, buf);
      break;
    case kNVTEFusedAttnConfigDeterministic:
      bool_to_uint8(cfg.deterministic, buf);
      break;
    case kNVTEFusedAttnConfigCudaGraph:
      bool_to_uint8(cfg.cuda_graph, buf);
      break;
    case kNVTEFusedAttnConfigReturnMaxLogit:
      bool_to_uint8(cfg.return_max_logit, buf);
      break;
    case kNVTEFusedAttnConfigAttnMaskType:
      std::memcpy(buf, &cfg.attn_mask_type, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasType:
      std::memcpy(buf, &cfg.bias_type, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeLeft:
      std::memcpy(buf, &cfg.window_size_left, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeRight:
      std::memcpy(buf, &cfg.window_size_right, attr_size);
      break;
    case kNVTEFusedAttnConfigBottomRightDiagonal:
      bool_to_uint8(cfg.bottom_right_diagonal, buf);
      break;
    case kNVTEFusedAttnConfigSoftmaxType:
      std::memcpy(buf, &cfg.softmax_type, attr_size);
      break;
    case kNVTEFusedAttnConfigScalingMode:
      std::memcpy(buf, &cfg.scaling_mode, attr_size);
      break;
    case kNVTEFusedAttnConfigDropout:
      std::memcpy(buf, &cfg.dropout, attr_size);
      break;
    case kNVTEFusedAttnConfigAttnScale:
      std::memcpy(buf, &cfg.attn_scale, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVDtype:
      std::memcpy(buf, &cfg.qkv_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigODtype:
      std::memcpy(buf, &cfg.o_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigDODtype:
      std::memcpy(buf, &cfg.do_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVDtype:
      std::memcpy(buf, &cfg.dqkv_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVLayout:
      std::memcpy(buf, &cfg.qkv_layout, attr_size);
      break;
    case kNVTEFusedAttnConfigOFormat:
      std::memcpy(buf, &cfg.o_format, attr_size);
      break;
    case kNVTEFusedAttnConfigDOFormat:
      std::memcpy(buf, &cfg.do_format, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVLayout:
      std::memcpy(buf, &cfg.dqkv_layout, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVScaleInvFormat:
      std::memcpy(buf, &cfg.qkv_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnConfigDOScaleInvFormat:
      std::memcpy(buf, &cfg.do_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnConfigBatchSize:
      std::memcpy(buf, &cfg.batch_size, attr_size);
      break;
    case kNVTEFusedAttnConfigNumAttnHeads:
      std::memcpy(buf, &cfg.num_attn_heads, attr_size);
      break;
    case kNVTEFusedAttnConfigNumGQAGroups:
      std::memcpy(buf, &cfg.num_gqa_groups, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimQK:
      std::memcpy(buf, &cfg.head_dim_qk, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimV:
      std::memcpy(buf, &cfg.head_dim_v, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenQ:
      std::memcpy(buf, &cfg.max_seqlen_q, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenKV:
      std::memcpy(buf, &cfg.max_seqlen_kv, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensQ:
      std::memcpy(buf, &cfg.num_tokens_q, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensKV:
      std::memcpy(buf, &cfg.num_tokens_kv, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesK:
      std::memcpy(buf, &cfg.num_pages_k, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesV:
      std::memcpy(buf, &cfg.num_pages_v, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeK:
      std::memcpy(buf, &cfg.page_size_k, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeV:
      std::memcpy(buf, &cfg.page_size_v, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqK:
      std::memcpy(buf, &cfg.max_pages_per_seq_k, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqV:
      std::memcpy(buf, &cfg.max_pages_per_seq_v, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasBatchSize:
      std::memcpy(buf, &cfg.bias_batch_size, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasNumHeads:
      std::memcpy(buf, &cfg.bias_num_heads, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenQ:
      std::memcpy(buf, &cfg.bias_seqlen_q, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenKV:
      std::memcpy(buf, &cfg.bias_seqlen_kv, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_set_fused_attn_config_attribute(NVTEFusedAttnConfig config,
                                          NVTEFusedAttnConfigAttribute attr, const void *buf,
                                          size_t size_in_bytes) {
  using namespace transformer_engine;

  NVTE_CHECK(attr < kNVTEFusedAttnConfigNumAttributes, "Invalid NVTEFusedAttnConfigAttribute (got ",
             static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnConfig::attr_sizes[attr];
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for fused attention config attribute (attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ", size_in_bytes,
             " bytes)");
  NVTE_CHECK(buf != nullptr, "Invalid buffer (got NULL)");

  auto &cfg = *get_fused_attn_config_mutable(config);
  switch (attr) {
    case kNVTEFusedAttnConfigIsTraining:
      uint8_to_bool(buf, cfg.is_training);
      break;
    case kNVTEFusedAttnConfigDeterministic:
      uint8_to_bool(buf, cfg.deterministic);
      break;
    case kNVTEFusedAttnConfigCudaGraph:
      uint8_to_bool(buf, cfg.cuda_graph);
      break;
    case kNVTEFusedAttnConfigReturnMaxLogit:
      uint8_to_bool(buf, cfg.return_max_logit);
      break;
    case kNVTEFusedAttnConfigAttnMaskType:
      std::memcpy(&cfg.attn_mask_type, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasType:
      std::memcpy(&cfg.bias_type, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeLeft:
      std::memcpy(&cfg.window_size_left, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeRight:
      std::memcpy(&cfg.window_size_right, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBottomRightDiagonal:
      uint8_to_bool(buf, cfg.bottom_right_diagonal);
      break;
    case kNVTEFusedAttnConfigSoftmaxType:
      std::memcpy(&cfg.softmax_type, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigScalingMode:
      std::memcpy(&cfg.scaling_mode, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDropout:
      std::memcpy(&cfg.dropout, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigAttnScale:
      std::memcpy(&cfg.attn_scale, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVDtype:
      std::memcpy(&cfg.qkv_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigODtype:
      std::memcpy(&cfg.o_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDODtype:
      std::memcpy(&cfg.do_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVDtype:
      std::memcpy(&cfg.dqkv_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVLayout:
      std::memcpy(&cfg.qkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigOFormat:
      std::memcpy(&cfg.o_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDOFormat:
      std::memcpy(&cfg.do_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVLayout:
      std::memcpy(&cfg.dqkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVScaleInvFormat:
      std::memcpy(&cfg.qkv_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDOScaleInvFormat:
      std::memcpy(&cfg.do_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBatchSize:
      std::memcpy(&cfg.batch_size, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumAttnHeads:
      std::memcpy(&cfg.num_attn_heads, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumGQAGroups:
      std::memcpy(&cfg.num_gqa_groups, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimQK:
      std::memcpy(&cfg.head_dim_qk, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimV:
      std::memcpy(&cfg.head_dim_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenQ:
      std::memcpy(&cfg.max_seqlen_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenKV:
      std::memcpy(&cfg.max_seqlen_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensQ:
      std::memcpy(&cfg.num_tokens_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensKV:
      std::memcpy(&cfg.num_tokens_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesK:
      std::memcpy(&cfg.num_pages_k, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesV:
      std::memcpy(&cfg.num_pages_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeK:
      std::memcpy(&cfg.page_size_k, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeV:
      std::memcpy(&cfg.page_size_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqK:
      std::memcpy(&cfg.max_pages_per_seq_k, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqV:
      std::memcpy(&cfg.max_pages_per_seq_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasBatchSize:
      std::memcpy(&cfg.bias_batch_size, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasNumHeads:
      std::memcpy(&cfg.bias_num_heads, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenQ:
      std::memcpy(&cfg.bias_seqlen_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenKV:
      std::memcpy(&cfg.bias_seqlen_kv, buf, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

NVTEFusedAttnFwdParams nvte_create_fused_attn_fwd_params() {
  return new transformer_engine::FusedAttnFwdParams{};
}

void nvte_destroy_fused_attn_fwd_params(NVTEFusedAttnFwdParams params) {
  delete transformer_engine::get_fused_attn_fwd_params_mutable(params);
}

void nvte_get_fused_attn_fwd_params_attribute(NVTEFusedAttnFwdParams params,
                                              NVTEFusedAttnFwdParamsAttribute attr, void *buf,
                                              size_t size_in_bytes, size_t *size_written) {
  using namespace transformer_engine;
  NVTE_CHECK(attr < kNVTEFusedAttnFwdParamsNumAttributes,
             "Invalid NVTEFusedAttnFwdParamsAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnFwdParams::attr_sizes[attr];
  if (size_written != nullptr) {
    *size_written = attr_size;
  }
  if (buf == nullptr) {
    return;
  }
  NVTE_CHECK(size_in_bytes >= attr_size, "Buffer is too small for attribute (need ", attr_size,
             ", got ", size_in_bytes, ")");
  const auto &p = *get_fused_attn_fwd_params(params);
  switch (attr) {
    case kNVTEFusedAttnFwdParamsQ:
      std::memcpy(buf, &p.Q, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsK:
      std::memcpy(buf, &p.K, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsV:
      std::memcpy(buf, &p.V, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsBias:
      std::memcpy(buf, &p.Bias, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsSoftmaxOffset:
      std::memcpy(buf, &p.SoftmaxOffset, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensQ:
      std::memcpy(buf, &p.cu_seqlens_q, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensKV:
      std::memcpy(buf, &p.cu_seqlens_kv, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensQPadded:
      std::memcpy(buf, &p.cu_seqlens_q_padded, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensKVPadded:
      std::memcpy(buf, &p.cu_seqlens_kv_padded, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsPageTableK:
      std::memcpy(buf, &p.page_table_k, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsPageTableV:
      std::memcpy(buf, &p.page_table_v, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsRngState:
      std::memcpy(buf, &p.rng_state, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsS:
      std::memcpy(buf, &p.S, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsO:
      std::memcpy(buf, &p.O, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsAuxCtxTensors:
      std::memcpy(buf, &p.Aux_CTX_Tensors, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsIsTraining:
      bool_to_uint8(p.is_training, buf);
      break;
    case kNVTEFusedAttnFwdParamsCudaGraph:
      bool_to_uint8(p.cuda_graph, buf);
      break;
    case kNVTEFusedAttnFwdParamsReturnMaxLogit:
      bool_to_uint8(p.return_max_logit, buf);
      break;
    case kNVTEFusedAttnFwdParamsAttnMaskType:
      std::memcpy(buf, &p.attn_mask_type, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsBiasType:
      std::memcpy(buf, &p.bias_type, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsSoftmaxType:
      std::memcpy(buf, &p.softmax_type, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsWindowSizeLeft:
      std::memcpy(buf, &p.window_size_left, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsWindowSizeRight:
      std::memcpy(buf, &p.window_size_right, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsBottomRightDiagonal:
      bool_to_uint8(p.bottom_right_diagonal, buf);
      break;
    case kNVTEFusedAttnFwdParamsDropout:
      std::memcpy(buf, &p.dropout, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsAttnScale:
      std::memcpy(buf, &p.attn_scale, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsQKVLayout:
      std::memcpy(buf, &p.qkv_layout, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsOFormat:
      std::memcpy(buf, &p.o_format, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsQKVScaleInvFormat:
      std::memcpy(buf, &p.qkv_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsMaxSeqlenQ:
      std::memcpy(buf, &p.max_seqlen_q, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsMaxSeqlenKV:
      std::memcpy(buf, &p.max_seqlen_kv, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsWorkspace:
      std::memcpy(buf, &p.workspace, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsStream:
      std::memcpy(buf, &p.stream, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnFwdParamsAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_set_fused_attn_fwd_params_attribute(NVTEFusedAttnFwdParams params,
                                              NVTEFusedAttnFwdParamsAttribute attr, const void *buf,
                                              size_t size_in_bytes) {
  using namespace transformer_engine;
  NVTE_CHECK(attr < kNVTEFusedAttnFwdParamsNumAttributes,
             "Invalid NVTEFusedAttnFwdParamsAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnFwdParams::attr_sizes[attr];
  NVTE_CHECK(buf != nullptr, "Input buffer must not be NULL.");
  NVTE_CHECK(size_in_bytes >= attr_size, "Buffer is too small for attribute (need ", attr_size,
             ", got ", size_in_bytes, ")");
  auto &p = *get_fused_attn_fwd_params_mutable(params);
  switch (attr) {
    case kNVTEFusedAttnFwdParamsQ:
      std::memcpy(&p.Q, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsK:
      std::memcpy(&p.K, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsV:
      std::memcpy(&p.V, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsBias:
      std::memcpy(&p.Bias, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsSoftmaxOffset:
      std::memcpy(&p.SoftmaxOffset, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensQ:
      std::memcpy(&p.cu_seqlens_q, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensKV:
      std::memcpy(&p.cu_seqlens_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensQPadded:
      std::memcpy(&p.cu_seqlens_q_padded, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsCuSeqlensKVPadded:
      std::memcpy(&p.cu_seqlens_kv_padded, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsPageTableK:
      std::memcpy(&p.page_table_k, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsPageTableV:
      std::memcpy(&p.page_table_v, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsRngState:
      std::memcpy(&p.rng_state, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsS:
      std::memcpy(&p.S, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsO:
      std::memcpy(&p.O, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsAuxCtxTensors:
      std::memcpy(&p.Aux_CTX_Tensors, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsIsTraining:
      uint8_to_bool(buf, p.is_training);
      break;
    case kNVTEFusedAttnFwdParamsCudaGraph:
      uint8_to_bool(buf, p.cuda_graph);
      break;
    case kNVTEFusedAttnFwdParamsReturnMaxLogit:
      uint8_to_bool(buf, p.return_max_logit);
      break;
    case kNVTEFusedAttnFwdParamsAttnMaskType:
      std::memcpy(&p.attn_mask_type, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsBiasType:
      std::memcpy(&p.bias_type, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsSoftmaxType:
      std::memcpy(&p.softmax_type, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsWindowSizeLeft:
      std::memcpy(&p.window_size_left, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsWindowSizeRight:
      std::memcpy(&p.window_size_right, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsBottomRightDiagonal:
      uint8_to_bool(buf, p.bottom_right_diagonal);
      break;
    case kNVTEFusedAttnFwdParamsDropout:
      std::memcpy(&p.dropout, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsAttnScale:
      std::memcpy(&p.attn_scale, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsQKVLayout:
      std::memcpy(&p.qkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsOFormat:
      std::memcpy(&p.o_format, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsQKVScaleInvFormat:
      std::memcpy(&p.qkv_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsMaxSeqlenQ:
      std::memcpy(&p.max_seqlen_q, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsMaxSeqlenKV:
      std::memcpy(&p.max_seqlen_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsWorkspace:
      std::memcpy(&p.workspace, buf, attr_size);
      break;
    case kNVTEFusedAttnFwdParamsStream:
      std::memcpy(&p.stream, buf, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnFwdParamsAttribute (got ", static_cast<int>(attr), ")");
  }
}

NVTEFusedAttnBwdParams nvte_create_fused_attn_bwd_params() {
  return new transformer_engine::FusedAttnBwdParams{};
}

void nvte_destroy_fused_attn_bwd_params(NVTEFusedAttnBwdParams params) {
  delete transformer_engine::get_fused_attn_bwd_params_mutable(params);
}

void nvte_get_fused_attn_bwd_params_attribute(NVTEFusedAttnBwdParams params,
                                              NVTEFusedAttnBwdParamsAttribute attr, void *buf,
                                              size_t size_in_bytes, size_t *size_written) {
  using namespace transformer_engine;
  NVTE_CHECK(attr < kNVTEFusedAttnBwdParamsNumAttributes,
             "Invalid NVTEFusedAttnBwdParamsAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnBwdParams::attr_sizes[attr];
  if (size_written != nullptr) {
    *size_written = attr_size;
  }
  if (buf == nullptr) {
    return;
  }
  NVTE_CHECK(size_in_bytes >= attr_size, "Buffer is too small for attribute (need ", attr_size,
             ", got ", size_in_bytes, ")");
  const auto &p = *get_fused_attn_bwd_params(params);
  switch (attr) {
    case kNVTEFusedAttnBwdParamsQ:
      std::memcpy(buf, &p.Q, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsK:
      std::memcpy(buf, &p.K, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsV:
      std::memcpy(buf, &p.V, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsO:
      std::memcpy(buf, &p.O, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDO:
      std::memcpy(buf, &p.dO, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsS:
      std::memcpy(buf, &p.S, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDP:
      std::memcpy(buf, &p.dP, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsAuxCtxTensors:
      std::memcpy(buf, &p.Aux_CTX_Tensors, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDQ:
      std::memcpy(buf, &p.dQ, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDK:
      std::memcpy(buf, &p.dK, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDV:
      std::memcpy(buf, &p.dV, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDBias:
      std::memcpy(buf, &p.dBias, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDSoftmaxOffset:
      std::memcpy(buf, &p.dSoftmaxOffset, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensQ:
      std::memcpy(buf, &p.cu_seqlens_q, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensKV:
      std::memcpy(buf, &p.cu_seqlens_kv, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensQPadded:
      std::memcpy(buf, &p.cu_seqlens_q_padded, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensKVPadded:
      std::memcpy(buf, &p.cu_seqlens_kv_padded, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCudaGraph:
      bool_to_uint8(p.cuda_graph, buf);
      break;
    case kNVTEFusedAttnBwdParamsDeterministic:
      bool_to_uint8(p.deterministic, buf);
      break;
    case kNVTEFusedAttnBwdParamsAttnMaskType:
      std::memcpy(buf, &p.attn_mask_type, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsBiasType:
      std::memcpy(buf, &p.bias_type, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsSoftmaxType:
      std::memcpy(buf, &p.softmax_type, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsWindowSizeLeft:
      std::memcpy(buf, &p.window_size_left, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsWindowSizeRight:
      std::memcpy(buf, &p.window_size_right, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsBottomRightDiagonal:
      bool_to_uint8(p.bottom_right_diagonal, buf);
      break;
    case kNVTEFusedAttnBwdParamsDropout:
      std::memcpy(buf, &p.dropout, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsAttnScale:
      std::memcpy(buf, &p.attn_scale, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsQKVLayout:
      std::memcpy(buf, &p.qkv_layout, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsOFormat:
      std::memcpy(buf, &p.o_format, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDOFormat:
      std::memcpy(buf, &p.do_format, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDQKVLayout:
      std::memcpy(buf, &p.dqkv_layout, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsQKVScaleInvFormat:
      std::memcpy(buf, &p.qkv_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDOScaleInvFormat:
      std::memcpy(buf, &p.do_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsMaxSeqlenQ:
      std::memcpy(buf, &p.max_seqlen_q, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsMaxSeqlenKV:
      std::memcpy(buf, &p.max_seqlen_kv, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsWorkspace:
      std::memcpy(buf, &p.workspace, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsStream:
      std::memcpy(buf, &p.stream, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnBwdParamsAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_set_fused_attn_bwd_params_attribute(NVTEFusedAttnBwdParams params,
                                              NVTEFusedAttnBwdParamsAttribute attr, const void *buf,
                                              size_t size_in_bytes) {
  using namespace transformer_engine;
  NVTE_CHECK(attr < kNVTEFusedAttnBwdParamsNumAttributes,
             "Invalid NVTEFusedAttnBwdParamsAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnBwdParams::attr_sizes[attr];
  NVTE_CHECK(buf != nullptr, "Input buffer must not be NULL.");
  NVTE_CHECK(size_in_bytes >= attr_size, "Buffer is too small for attribute (need ", attr_size,
             ", got ", size_in_bytes, ")");
  auto &p = *get_fused_attn_bwd_params_mutable(params);
  switch (attr) {
    case kNVTEFusedAttnBwdParamsQ:
      std::memcpy(&p.Q, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsK:
      std::memcpy(&p.K, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsV:
      std::memcpy(&p.V, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsO:
      std::memcpy(&p.O, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDO:
      std::memcpy(&p.dO, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsS:
      std::memcpy(&p.S, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDP:
      std::memcpy(&p.dP, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsAuxCtxTensors:
      std::memcpy(&p.Aux_CTX_Tensors, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDQ:
      std::memcpy(&p.dQ, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDK:
      std::memcpy(&p.dK, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDV:
      std::memcpy(&p.dV, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDBias:
      std::memcpy(&p.dBias, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDSoftmaxOffset:
      std::memcpy(&p.dSoftmaxOffset, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensQ:
      std::memcpy(&p.cu_seqlens_q, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensKV:
      std::memcpy(&p.cu_seqlens_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensQPadded:
      std::memcpy(&p.cu_seqlens_q_padded, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCuSeqlensKVPadded:
      std::memcpy(&p.cu_seqlens_kv_padded, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsCudaGraph:
      uint8_to_bool(buf, p.cuda_graph);
      break;
    case kNVTEFusedAttnBwdParamsDeterministic:
      uint8_to_bool(buf, p.deterministic);
      break;
    case kNVTEFusedAttnBwdParamsAttnMaskType:
      std::memcpy(&p.attn_mask_type, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsBiasType:
      std::memcpy(&p.bias_type, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsSoftmaxType:
      std::memcpy(&p.softmax_type, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsWindowSizeLeft:
      std::memcpy(&p.window_size_left, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsWindowSizeRight:
      std::memcpy(&p.window_size_right, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsBottomRightDiagonal:
      uint8_to_bool(buf, p.bottom_right_diagonal);
      break;
    case kNVTEFusedAttnBwdParamsDropout:
      std::memcpy(&p.dropout, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsAttnScale:
      std::memcpy(&p.attn_scale, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsQKVLayout:
      std::memcpy(&p.qkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsOFormat:
      std::memcpy(&p.o_format, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDOFormat:
      std::memcpy(&p.do_format, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDQKVLayout:
      std::memcpy(&p.dqkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsQKVScaleInvFormat:
      std::memcpy(&p.qkv_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsDOScaleInvFormat:
      std::memcpy(&p.do_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsMaxSeqlenQ:
      std::memcpy(&p.max_seqlen_q, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsMaxSeqlenKV:
      std::memcpy(&p.max_seqlen_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsWorkspace:
      std::memcpy(&p.workspace, buf, attr_size);
      break;
    case kNVTEFusedAttnBwdParamsStream:
      std::memcpy(&p.stream, buf, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnBwdParamsAttribute (got ", static_cast<int>(attr), ")");
  }
}
