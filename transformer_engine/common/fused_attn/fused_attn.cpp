/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "config_and_params.h"
#include "fused_attn_f16_arbitrary_seqlen.h"
#include "fused_attn_fp8.h"
#include "utils.h"

namespace transformer_engine {

std::string to_string(NVTE_QKV_Layout layout) {
  switch (layout) {
    case NVTE_SB3HD:
      return "NVTE_SB3HD";
    case NVTE_SBH3D:
      return "NVTE_SBH3D";
    case NVTE_SBHD_SB2HD:
      return "NVTE_SBHD_SB2HD";
    case NVTE_SBHD_SBH2D:
      return "NVTE_SBHD_SBH2D";
    case NVTE_SBHD_SBHD_SBHD:
      return "NVTE_SBHD_SBHD_SBHD";
    case NVTE_BS3HD:
      return "NVTE_BS3HD";
    case NVTE_BSH3D:
      return "NVTE_BSH3D";
    case NVTE_BSHD_BS2HD:
      return "NVTE_BSHD_BS2HD";
    case NVTE_BSHD_BSH2D:
      return "NVTE_BSHD_BSH2D";
    case NVTE_BSHD_BSHD_BSHD:
      return "NVTE_BSHD_BSHD_BSHD";
    case NVTE_T3HD:
      return "NVTE_T3HD";
    case NVTE_TH3D:
      return "NVTE_TH3D";
    case NVTE_THD_T2HD:
      return "NVTE_THD_T2HD";
    case NVTE_THD_TH2D:
      return "NVTE_THD_TH2D";
    case NVTE_THD_THD_THD:
      return "NVTE_THD_THD_THD";
    case NVTE_SBHD_BSHD_BSHD:
      return "NVTE_SBHD_BSHD_BSHD";
    case NVTE_BSHD_SBHD_SBHD:
      return "NVTE_BSHD_SBHD_SBHD";
    case NVTE_THD_BSHD_BSHD:
      return "NVTE_THD_BSHD_BSHD";
    case NVTE_THD_SBHD_SBHD:
      return "NVTE_THD_SBHD_SBHD";
    case NVTE_Paged_KV_BSHD_BSHD_BSHD:
      return "NVTE_Paged_KV_BSHD_BSHD_BSHD";
    case NVTE_Paged_KV_BSHD_SBHD_SBHD:
      return "NVTE_Paged_KV_BSHD_SBHD_SBHD";
    case NVTE_Paged_KV_SBHD_BSHD_BSHD:
      return "NVTE_Paged_KV_SBHD_BSHD_BSHD";
    case NVTE_Paged_KV_SBHD_SBHD_SBHD:
      return "NVTE_Paged_KV_SBHD_SBHD_SBHD";
    case NVTE_Paged_KV_THD_BSHD_BSHD:
      return "NVTE_Paged_KV_THD_BSHD_BSHD";
    case NVTE_Paged_KV_THD_SBHD_SBHD:
      return "NVTE_Paged_KV_THD_SBHD_SBHD";
    default:
      return "UNKNOWN_QKV_LAYOUT(" + std::to_string(static_cast<int>(layout)) + ")";
  }
}

std::string to_string(NVTE_QKV_Format format) {
  switch (format) {
    case NVTE_SBHD:
      return "NVTE_SBHD";
    case NVTE_BSHD:
      return "NVTE_BSHD";
    case NVTE_THD:
      return "NVTE_THD";
    case NVTE_BSHD_2SBHD:
      return "NVTE_BSHD_2SBHD";
    case NVTE_SBHD_2BSHD:
      return "NVTE_SBHD_2BSHD";
    case NVTE_THD_2BSHD:
      return "NVTE_THD_2BSHD";
    case NVTE_THD_2SBHD:
      return "NVTE_THD_2SBHD";
    default:
      return "UNKNOWN_QKV_FORMAT(" + std::to_string(static_cast<int>(format)) + ")";
  }
}

}  // namespace transformer_engine

// map NVTE_QKV_Layout to NVTE_QKV_Layout_Group
NVTE_QKV_Layout_Group nvte_get_qkv_layout_group(NVTE_QKV_Layout qkv_layout) {
  switch (qkv_layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_T3HD:
      return NVTE_QKV_Layout_Group::NVTE_3HD;
    case NVTE_QKV_Layout::NVTE_SBH3D:
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_TH3D:
      return NVTE_QKV_Layout_Group::NVTE_H3D;
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
      return NVTE_QKV_Layout_Group::NVTE_HD_2HD;
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
      return NVTE_QKV_Layout_Group::NVTE_HD_H2D;
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
    case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD:
      return NVTE_QKV_Layout_Group::NVTE_HD_HD_HD;
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD:
      return NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD;
    case NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD:
      return NVTE_QKV_Layout_Group::NVTE_SD_SD_SD;
    default:
      NVTE_ERROR("Unsupported qkv_layout ", transformer_engine::to_string(qkv_layout),
                 " in nvte_get_qkv_layout_group.");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format
NVTE_QKV_Format nvte_get_qkv_format(NVTE_QKV_Layout qkv_layout) {
  switch (qkv_layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
    case NVTE_QKV_Layout::NVTE_SBH3D:
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Layout::NVTE_T3HD:
    case NVTE_QKV_Layout::NVTE_TH3D:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
      return NVTE_QKV_Format::NVTE_THD;
    case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_SBHD_2BSHD;
    case NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_BSHD_2SBHD;
    case NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD:
      return NVTE_QKV_Format::NVTE_THD_2BSHD;
    case NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD:
      return NVTE_QKV_Format::NVTE_THD_2SBHD;
    case NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD:
      return NVTE_QKV_Format::NVTE_BHSD;
    default:
      NVTE_ERROR("Unsupported qkv_layout ", transformer_engine::to_string(qkv_layout),
                 " in nvte_get_qkv_format.");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format for Q
NVTE_QKV_Format nvte_get_q_format(NVTE_QKV_Layout qkv_layout) {
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  switch (qkv_format) {
    case NVTE_QKV_Format::NVTE_SBHD:
    case NVTE_QKV_Format::NVTE_SBHD_2BSHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Format::NVTE_BSHD:
    case NVTE_QKV_Format::NVTE_BSHD_2SBHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Format::NVTE_THD:
    case NVTE_QKV_Format::NVTE_THD_2BSHD:
    case NVTE_QKV_Format::NVTE_THD_2SBHD:
      return NVTE_QKV_Format::NVTE_THD;
    case NVTE_QKV_Format::NVTE_BHSD:
      return NVTE_QKV_Format::NVTE_BHSD;
    default:
      NVTE_ERROR("Unsupported qkv_format ", transformer_engine::to_string(qkv_format),
                 " in nvte_get_q_format.");
  }
}

// map NVTE_QKV_Layout to NVTE_QKV_Format for KV
NVTE_QKV_Format nvte_get_kv_format(NVTE_QKV_Layout qkv_layout) {
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  switch (qkv_format) {
    case NVTE_QKV_Format::NVTE_SBHD:
    case NVTE_QKV_Format::NVTE_BSHD_2SBHD:
    case NVTE_QKV_Format::NVTE_THD_2SBHD:
      return NVTE_QKV_Format::NVTE_SBHD;
    case NVTE_QKV_Format::NVTE_BSHD:
    case NVTE_QKV_Format::NVTE_SBHD_2BSHD:
    case NVTE_QKV_Format::NVTE_THD_2BSHD:
      return NVTE_QKV_Format::NVTE_BSHD;
    case NVTE_QKV_Format::NVTE_THD:
      return NVTE_QKV_Format::NVTE_THD;
    case NVTE_QKV_Format::NVTE_BHSD:
      return NVTE_QKV_Format::NVTE_BHSD;
    default:
      NVTE_ERROR("Unsupported qkv_format ", transformer_engine::to_string(qkv_format),
                 " in nvte_get_kv_format.");
  }
}

namespace {

// per-thread storage for the diagnostic string
// re-used (cleared + re-populated) on every call to nvte_get_fused_attn_backend_v2 on this thread
thread_local std::string fused_attn_backend_message_buffer;

// Stash `reason` in the thread-local buffer and, if the caller asked for a diagnostic,
// publish a NUL-terminated pointer to it via `*message`. Safe to call with `message == nullptr`.
void set_message(const char **message, std::string reason) {
  fused_attn_backend_message_buffer = std::move(reason);
  if (message != nullptr) {
    *message = fused_attn_backend_message_buffer.c_str();
  }
}

}  // namespace

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(NVTEFusedAttnConfig config,
                                                       const char **message) {
  using namespace transformer_engine;
  const FusedAttnConfig &cfg = *get_fused_attn_config(config);
  set_message(message, "");

  cudnnHandle_t handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(cfg.qkv_layout);
  const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(cfg.qkv_layout);
  const auto cudnn_runtime_version = cudnnGetVersion();

  // THD + 64-bit ragged offsets require cuDNN >= 9.5
  const bool requires_64bit_ragged_offset =
      (qkv_format == NVTE_THD &&
       fused_attn::get_ragged_offset_dtype(layout_group, cfg.num_attn_heads, cfg.num_gqa_groups,
                                           cfg.max_seqlen_q, cfg.max_seqlen_kv, cfg.head_dim_qk,
                                           cfg.head_dim_v) == DType::kInt64);
  if (requires_64bit_ragged_offset && cudnn_runtime_version < 90500) {
    set_message(message,
                "Configuration requires 64-bit ragged offsets, which require "
                "cuDNN >= 9.5.");
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  // THD requires padding-style mask
  if (qkv_format == NVTE_QKV_Format::NVTE_THD &&
      cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
      cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
      cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) {
    set_message(message,
                "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  const bool is_fp8 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat8E4M3 || cfg.qkv_dtype == NVTEDType::kNVTEFloat8E5M2);
  const bool is_f16_or_bf16 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat16 || cfg.qkv_dtype == NVTEDType::kNVTEBFloat16);

  if (is_fp8) {
    if (cfg.return_max_logit) {
      set_message(message, "FP8 fused attention does not support return_max_logit=True.");
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    if (qkv_format != NVTE_QKV_Format::NVTE_BSHD && qkv_format != NVTE_QKV_Format::NVTE_SBHD &&
        qkv_format != NVTE_QKV_Format::NVTE_BHSD) {
      set_message(message, "FP8 fused attention supports BSHD/SBHD/BHSD formats, found " +
                               std::to_string(static_cast<int>(qkv_format)) + ".");
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    std::string fwd_reason = is_supported_fp8_fwd(cfg, handle);
    if (!fwd_reason.empty()) {
      set_message(message, std::move(fwd_reason));
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    if (cfg.is_training) {
      std::string bwd_reason = is_supported_fp8_bwd(cfg, handle);
      if (!bwd_reason.empty()) {
        set_message(message, std::move(bwd_reason));
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
    }
    return NVTE_Fused_Attn_Backend::NVTE_FP8;
  }

  if (is_f16_or_bf16) {
    if (cudnn_runtime_version <= 91500 && cfg.is_training &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (cfg.max_seqlen_kv % 128 != 0) && cfg.cuda_graph &&
        cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
        cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
        cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) {
      set_message(message, "Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN.");
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    std::string fwd_reason = is_supported_f16_fwd(cfg, handle);
    if (!fwd_reason.empty()) {
      set_message(message, std::move(fwd_reason));
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    if (cfg.is_training) {
      std::string bwd_reason = is_supported_f16_bwd(cfg, handle);
      if (!bwd_reason.empty()) {
        set_message(message, std::move(bwd_reason));
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
    }
    return NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  }

  set_message(message, "Unsupported QKV dtype qkv_dtype=" + std::to_string(cfg.qkv_dtype) + " .");
  return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
}

// Deprecated: thin wrapper preserving the historical narrow signature. New callers should
// construct an NVTEFusedAttnConfig and call nvte_get_fused_attn_backend_v2 directly to access
// the additional fields (attn_scale, format/layout fields, scaling_mode, paged-KV/bias shape,
// dO/dQKV dtypes, etc.) that this wrapper cannot express.
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) {
  transformer_engine::FusedAttnConfig cfg = transformer_engine::make_default_fused_attn_config();
  cfg.qkv_layout = qkv_layout;
  cfg.bias_type = bias_type;
  cfg.attn_mask_type = attn_mask_type;
  cfg.softmax_type = softmax_type;
  cfg.attn_scale = attn_scale;
  cfg.dropout = dropout;
  cfg.max_seqlen_q = max_seqlen_q;
  cfg.max_seqlen_kv = max_seqlen_kv;
  cfg.window_size_left = window_size_left;
  cfg.window_size_right = window_size_right;
  cfg.cuda_graph = cuda_graph;
  NVTE_CHECK(q_dtype == kv_dtype, "Q and KV must have the same data type.");
  cfg.qkv_dtype = q_dtype;
  cfg.o_dtype = q_dtype;
  cfg.do_dtype = q_dtype;
  cfg.dqkv_dtype = q_dtype;
  cfg.num_attn_heads = num_attn_heads;
  cfg.num_gqa_groups = num_gqa_groups;
  cfg.head_dim_qk = head_dim_qk;
  cfg.head_dim_v = head_dim_v;
  cfg.is_training = is_training;
  cfg.return_max_logit = return_max_logit;
  cfg.deterministic = deterministic;
  return nvte_get_fused_attn_backend_v2(reinterpret_cast<NVTEFusedAttnConfig>(&cfg),
                                        /*message=*/nullptr);
}

void nvte_fused_attn_fwd_v2(NVTEFusedAttnFwdParams params) {
  NVTE_API_CALL(nvte_fused_attn_fwd_v2);
  using namespace transformer_engine;
  const FusedAttnFwdParams &p = *get_fused_attn_fwd_params(params);
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(p.cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(p.cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(p.cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(p.cu_seqlens_kv_padded);
  const Tensor *input_page_table_k = convertNVTETensorCheck(p.page_table_k);
  const Tensor *input_page_table_v = convertNVTETensorCheck(p.page_table_v);
  const Tensor *input_rng_state = convertNVTETensorCheck(p.rng_state);
  const Tensor *input_Q = convertNVTETensorCheck(p.Q);
  const Tensor *input_K = convertNVTETensorCheck(p.K);
  const Tensor *input_V = convertNVTETensorCheck(p.V);
  const Tensor *input_Bias = convertNVTETensorCheck(p.Bias);
  const Tensor *input_SoftmaxOffset = convertNVTETensorCheck(p.SoftmaxOffset);
  Tensor *input_output_S = convertNVTETensorCheck(p.S);
  Tensor *output_O = convertNVTETensorCheck(p.O);
  Tensor *wkspace = convertNVTETensor(p.workspace);

  NVTE_QKV_Format q_format = nvte_get_q_format(p.qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(p.qkv_layout);
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

  int64_t num_pages_k = 0;
  int64_t num_pages_v = 0;
  int64_t page_size_k = 0;
  int64_t page_size_v = 0;
  int64_t max_pages_per_seq_k = 0;
  int64_t max_pages_per_seq_v = 0;
  if (input_page_table_k->data.dptr != nullptr) {
    max_pages_per_seq_k = input_page_table_k->data.shape[1];
  }
  if (input_page_table_v->data.dptr != nullptr) {
    max_pages_per_seq_v = input_page_table_v->data.shape[1];
  }
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(p.qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) {
    NVTE_QKV_Format paged_kv_format = nvte_get_kv_format(p.qkv_layout);
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

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  NVTE_CHECK(Q_type == KV_type, "Q and KV must have the same data type.");
  const NVTEDType O_type = static_cast<NVTEDType>(output_O->data.dtype);
  const NVTEScalingMode scaling_mode = input_Q->scaling_mode;

  size_t bias_b = 0, bias_h = 0, bias_sq = 0, bias_skv = 0;
  if ((p.bias_type != NVTE_NO_BIAS) && (p.bias_type != NVTE_ALIBI) &&
      input_Bias->data.dptr != nullptr && input_Bias->data.shape.size() >= 4) {
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
    bias_sq = input_Bias->data.shape[2];
    bias_skv = input_Bias->data.shape[3];
  }

  FusedAttnConfig cfg = make_fused_attn_config(p);
  cfg.scaling_mode = scaling_mode;
  cfg.qkv_dtype = Q_type;
  cfg.o_dtype = O_type;
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
  cfg.bias_batch_size = bias_b;
  cfg.bias_num_heads = bias_h;
  cfg.bias_seqlen_q = bias_sq;
  cfg.bias_seqlen_kv = bias_skv;
  cfg.num_tokens_q = t_q;
  cfg.num_tokens_kv = t_kv;
  NVTE_Fused_Attn_Backend fused_attention_backend =
      nvte_get_fused_attn_backend_v2(reinterpret_cast<NVTEFusedAttnConfig>(&cfg),
                                     /*message=*/nullptr);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    fused_attn_arbitrary_seqlen_fwd(cfg, input_Q, input_K, input_V, input_Bias, input_SoftmaxOffset,
                                    output_O, p.Aux_CTX_Tensors, input_cu_seqlens_q,
                                    input_cu_seqlens_kv, input_cu_seqlens_q_padded,
                                    input_cu_seqlens_kv_padded, input_page_table_k,
                                    input_page_table_v, input_rng_state, wkspace, p.stream, handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    fused_attn_fp8_fwd(cfg, input_Q, input_K, input_V, input_SoftmaxOffset, input_output_S,
                       output_O, p.Aux_CTX_Tensors, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_rng_state, wkspace, p.stream, handle);
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}

// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, const NVTETensor SoftmaxOffset, NVTETensor S,
                         NVTETensor O, NVTETensorPack *Aux_CTX_Tensors,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, const NVTETensor page_table_k,
                         const NVTETensor page_table_v, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         bool return_max_logit, bool cuda_graph, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
                         NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
                         int64_t window_size_left, int64_t window_size_right,
                         bool bottom_right_diagonal, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd);
  transformer_engine::FusedAttnFwdParams p =
      transformer_engine::make_default_fused_attn_fwd_params();
  p.Q = Q;
  p.K = K;
  p.V = V;
  p.Bias = Bias;
  p.SoftmaxOffset = SoftmaxOffset;
  p.S = S;
  p.O = O;
  p.Aux_CTX_Tensors = Aux_CTX_Tensors;
  p.cu_seqlens_q = cu_seqlens_q;
  p.cu_seqlens_kv = cu_seqlens_kv;
  p.cu_seqlens_q_padded = cu_seqlens_q_padded;
  p.cu_seqlens_kv_padded = cu_seqlens_kv_padded;
  p.page_table_k = page_table_k;
  p.page_table_v = page_table_v;
  p.rng_state = rng_state;
  p.max_seqlen_q = max_seqlen_q;
  p.max_seqlen_kv = max_seqlen_kv;
  p.is_training = is_training;
  p.return_max_logit = return_max_logit;
  p.cuda_graph = cuda_graph;
  p.attn_scale = attn_scale;
  p.dropout = dropout;
  p.qkv_layout = qkv_layout;
  p.o_format = o_format;
  p.qkv_scale_inv_format = qkv_scale_inv_format;
  p.bias_type = bias_type;
  p.attn_mask_type = attn_mask_type;
  p.softmax_type = softmax_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.workspace = workspace;
  p.stream = stream;
  nvte_fused_attn_fwd_v2(reinterpret_cast<NVTEFusedAttnFwdParams>(&p));
}

void nvte_fused_attn_bwd_v2(NVTEFusedAttnBwdParams params) {
  NVTE_API_CALL(nvte_fused_attn_bwd_v2);
  using namespace transformer_engine;
  const FusedAttnBwdParams &p = *get_fused_attn_bwd_params(params);
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(p.cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(p.cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(p.cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(p.cu_seqlens_kv_padded);
  const Tensor *input_Q = convertNVTETensorCheck(p.Q);
  const Tensor *input_K = convertNVTETensorCheck(p.K);
  const Tensor *input_V = convertNVTETensorCheck(p.V);
  const Tensor *input_O = convertNVTETensorCheck(p.O);
  const Tensor *input_dO = convertNVTETensorCheck(p.dO);
  const Tensor *input_S = convertNVTETensorCheck(p.S);
  Tensor *input_output_dP = convertNVTETensorCheck(p.dP);
  Tensor *output_dQ = convertNVTETensorCheck(p.dQ);
  Tensor *output_dK = convertNVTETensorCheck(p.dK);
  Tensor *output_dV = convertNVTETensorCheck(p.dV);
  Tensor *output_dBias = convertNVTETensorCheck(p.dBias);
  Tensor *output_dSoftmaxOffset = convertNVTETensorCheck(p.dSoftmaxOffset);
  Tensor *wkspace = convertNVTETensor(p.workspace);

  NVTE_QKV_Format q_format = nvte_get_q_format(p.qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(p.qkv_layout);
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

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  NVTE_CHECK(Q_type == KV_type, "Q and KV must have the same data type.");
  const NVTEDType O_type = static_cast<NVTEDType>(input_O->data.dtype);
  const NVTEDType dO_type = static_cast<NVTEDType>(input_dO->data.dtype);
  const NVTEDType dQKV_type = static_cast<NVTEDType>(output_dQ->data.dtype);
  const NVTEScalingMode scaling_mode = input_Q->scaling_mode;

  size_t bias_b = 0, bias_h = 0, bias_sq = 0, bias_skv = 0;
  if ((p.bias_type != NVTE_NO_BIAS) && (p.bias_type != NVTE_ALIBI) &&
      output_dBias->data.shape.size() >= 4) {
    bias_b = output_dBias->data.shape[0];
    bias_h = output_dBias->data.shape[1];
    bias_sq = output_dBias->data.shape[2];
    bias_skv = output_dBias->data.shape[3];
  }

  FusedAttnConfig cfg = make_fused_attn_config(p);
  cfg.scaling_mode = scaling_mode;
  cfg.qkv_dtype = Q_type;
  cfg.o_dtype = O_type;
  cfg.do_dtype = dO_type;
  cfg.dqkv_dtype = dQKV_type;
  cfg.batch_size = b;
  cfg.num_attn_heads = h_q;
  cfg.num_gqa_groups = h_kv;
  cfg.head_dim_qk = d_qk;
  cfg.head_dim_v = d_v;
  cfg.bias_batch_size = bias_b;
  cfg.bias_num_heads = bias_h;
  cfg.bias_seqlen_q = bias_sq;
  cfg.bias_seqlen_kv = bias_skv;
  cfg.num_tokens_q = t_q;
  cfg.num_tokens_kv = t_kv;
  NVTE_Fused_Attn_Backend fused_attention_backend =
      nvte_get_fused_attn_backend_v2(reinterpret_cast<NVTEFusedAttnConfig>(&cfg),
                                     /*message=*/nullptr);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    size_t i = 0;
    Tensor *output_S = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_rng_state = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_Bias, *input_SoftmaxOffset;
    if ((p.bias_type != NVTE_NO_BIAS) && (p.bias_type != NVTE_ALIBI)) {
      input_Bias = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    if (p.softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_arbitrary_seqlen_bwd(
        cfg, input_Q, input_K, input_V, input_O, input_dO, input_Bias, input_SoftmaxOffset,
        output_S, output_dQ, output_dK, output_dV, output_dBias, output_dSoftmaxOffset,
        input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded,
        input_cu_seqlens_kv_padded, input_rng_state, wkspace, p.stream, handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    size_t i = 0;
    const Tensor *input_M = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_rng_state = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_SoftmaxOffset = nullptr;
    if (p.softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    const Tensor *input_dO_f16 = nullptr;
    if (input_dO->scaling_mode == NVTE_MXFP8_1D_SCALING) {
      input_dO_f16 = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_fp8_bwd(cfg, input_Q, input_K, input_V, input_O, input_dO, input_dO_f16, input_M,
                       input_S, input_SoftmaxOffset, input_output_dP, output_dQ, output_dK,
                       output_dV, output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_rng_state, wkspace, p.stream, handle);
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}

// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, NVTETensor dSoftmaxOffset,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
                         NVTE_QKV_Format do_format, NVTE_QKV_Layout dqkv_layout,
                         NVTE_QKV_Format qkv_scale_inv_format, NVTE_QKV_Format do_scale_inv_format,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTE_Softmax_Type softmax_type, int64_t window_size_left,
                         int64_t window_size_right, bool bottom_right_diagonal, bool deterministic,
                         bool cuda_graph, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd);
  transformer_engine::FusedAttnBwdParams p =
      transformer_engine::make_default_fused_attn_bwd_params();
  p.Q = Q;
  p.K = K;
  p.V = V;
  p.O = O;
  p.dO = dO;
  p.S = S;
  p.dP = dP;
  p.Aux_CTX_Tensors = Aux_CTX_Tensors;
  p.dQ = dQ;
  p.dK = dK;
  p.dV = dV;
  p.dBias = dBias;
  p.dSoftmaxOffset = dSoftmaxOffset;
  p.cu_seqlens_q = cu_seqlens_q;
  p.cu_seqlens_kv = cu_seqlens_kv;
  p.cu_seqlens_q_padded = cu_seqlens_q_padded;
  p.cu_seqlens_kv_padded = cu_seqlens_kv_padded;
  p.max_seqlen_q = max_seqlen_q;
  p.max_seqlen_kv = max_seqlen_kv;
  p.attn_scale = attn_scale;
  p.dropout = dropout;
  p.qkv_layout = qkv_layout;
  p.o_format = o_format;
  p.do_format = do_format;
  p.dqkv_layout = dqkv_layout;
  p.qkv_scale_inv_format = qkv_scale_inv_format;
  p.do_scale_inv_format = do_scale_inv_format;
  p.bias_type = bias_type;
  p.attn_mask_type = attn_mask_type;
  p.softmax_type = softmax_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.deterministic = deterministic;
  p.cuda_graph = cuda_graph;
  p.workspace = workspace;
  p.stream = stream;
  nvte_fused_attn_bwd_v2(reinterpret_cast<NVTEFusedAttnBwdParams>(&p));
}

uint32_t nvte_get_runtime_num_segments(NVTETensor cu_seqlen, NVTETensor workspace, size_t len,
                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_get_runtime_num_segments);
  using namespace transformer_engine::fused_attn;
  return GetRuntimeNumSegments(cu_seqlen, workspace, len, stream);
}

void nvte_populate_rng_state_async(NVTETensor rng_state_dst, const NVTETensor seed,
                                   size_t q_max_seqlen, size_t kv_max_seqlen,
                                   NVTE_Fused_Attn_Backend backend, cudaStream_t stream) {
  NVTE_API_CALL(nvte_populate_rng_state_async);
  using namespace transformer_engine::fused_attn;
  PopulateRngStateAsync(rng_state_dst, seed, q_max_seqlen, kv_max_seqlen, backend, stream);
}
