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
  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
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
  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
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

// The per-thread storage for the diagnostic string
thread_local std::string fused_attn_backend_message_buffer;

// Records `reason` in the per-thread buffer and `message`; returns with "no backend"
[[nodiscard]] NVTE_Fused_Attn_Backend reject(const char **message, std::string reason) {
  if (message != nullptr) {
    fused_attn_backend_message_buffer = std::move(reason);
    *message = fused_attn_backend_message_buffer.c_str();
  }
  return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
}

}  // namespace

// Fused attention backend query: runs TE's specific rules first, then cuDNN's. First rejection wins.
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(NVTEFusedAttnConfig config,
                                                       const char **message) {
  NVTE_API_CALL(nvte_get_fused_attn_backend_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
  FusedAttnConfig cfg = *get_fused_attn_config(config);
  cfg.derive();
  if (message != nullptr) *message = "";

  cudnnHandle_t handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const auto cudnn_runtime_version = cudnnGetVersion();
  const int sm_arch = cuda::sm_arch(cuda::current_device());

  // THD + 64-bit ragged offsets require cuDNN >= 9.5
  if (cfg.needs_64bit_ragged_offset && cudnn_runtime_version < 90500) {
    return reject(
        message,
        "This config requires 64-bit ragged offsets, which is only supported by cuDNN >= 9.5.");
  }

  // THD input requires a padding mask
  if ((cfg.is_ragged_q || cfg.is_ragged_kv) && !cfg.is_padding) {
    return reject(
        message,
        "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
  }

  if ((cfg.is_ragged_q && cfg.num_tokens_q == 0) || (cfg.is_ragged_kv && cfg.num_tokens_kv == 0)) {
    return reject(message,
                  "THD format requires num_tokens_q / num_tokens_kv to be set for the ragged "
                  "inputs.");
  }

  // Paged KV requires a padding mask
  if (cfg.is_paged_kv && !cfg.is_padding) {
    return reject(message,
                  "Paged KV requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
  }

  // Paged KV requires cache dimensions to be set
  if (cfg.is_paged_kv &&
      (cfg.num_pages_k == 0 || cfg.num_pages_v == 0 || cfg.page_size_k == 0 ||
       cfg.page_size_v == 0 || cfg.max_pages_per_seq_k == 0 || cfg.max_pages_per_seq_v == 0)) {
    return reject(message,
                  "Paged KV requires num_pages, page_size and max_pages_per_seq to be set for both "
                  "K and V.");
  }

  // Fused-attention does not support pre-scale bias
  if (cfg.bias_type == NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS) {
    return reject(message, "Fused attention does not support pre-scale bias.");
  }

  const bool is_fp8 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat8E4M3 || cfg.qkv_dtype == NVTEDType::kNVTEFloat8E5M2);
  const bool is_f16_or_bf16 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat16 || cfg.qkv_dtype == NVTEDType::kNVTEBFloat16);

  auto each_pass = [&](auto &&verdict) -> std::string {
    if (cfg.check_for_forward_support) {
      std::string reason = verdict(Pass::Fwd);
      if (!reason.empty()) return reason;
    }
    if (cfg.is_training && cfg.check_for_backward_support) {
      std::string reason = verdict(Pass::Bwd);
      if (!reason.empty()) return reason;
    }
    return "";
  };

  // F16/BF16 support checks
  if (is_f16_or_bf16) {
    if ((cfg.is_ragged_q || cfg.is_ragged_kv) && sm_arch < 90) {
      return reject(message, "F16/BF16 fused attention with THD format requires sm90 or later.");
    }
    const bool has_sliding_window = !(cfg.window_size_left == -1 &&
                                      (cfg.window_size_right == -1 || cfg.window_size_right == 0));
    if (cfg.is_causal_bottom_right && has_sliding_window && cfg.max_seqlen_q != cfg.max_seqlen_kv &&
        cudnn_runtime_version <= 90700 && sm_arch >= 100) {
      return reject(message,
                    "Known cuDNN <= 9.7.0 issue with bottom-right causal masking and a sliding "
                    "window for cross-attention on sm100. Please upgrade cuDNN.");
    }
    if (cudnn_runtime_version <= 91500 && cfg.is_training &&
        (cfg.qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
         cfg.qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (cfg.max_seqlen_kv % 128 != 0) && cfg.cuda_graph && !cfg.is_padding) {
      return reject(message, "Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN.");
    }

    // Run cuDNN support checks
    std::string cudnn_reason =
        each_pass([&](Pass pass) { return support_verdict_f16(cfg, pass, handle); });
    if (!cudnn_reason.empty()) return reject(message, std::move(cudnn_reason));
    return NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  }

  // FP8 support checks
  if (is_fp8) {
    if (cfg.return_max_logit) {
      return reject(message, "FP8 fused attention does not support return_max_logit=True.");
    }
    if (cfg.qkv_format != NVTE_QKV_Format::NVTE_BSHD &&
        cfg.qkv_format != NVTE_QKV_Format::NVTE_SBHD &&
        cfg.qkv_format != NVTE_QKV_Format::NVTE_BHSD &&
        cfg.qkv_format != NVTE_QKV_Format::NVTE_THD) {
      return reject(message, "FP8 fused attention supports BSHD/SBHD/BHSD/THD formats, found " +
                                 std::to_string(static_cast<int>(cfg.qkv_format)) + ".");
    }
    // Backward writes dQ/dK/dV through dqkv_layout, which a forward-only caller may leave unset;
    // nvte_get_qkv_format() rejects NVTE_QKV_Layout_NOT_SET, so only classify a layout that is set.
    if (cfg.is_training && cfg.check_for_backward_support &&
        cfg.dqkv_layout != NVTE_QKV_Layout_NOT_SET) {
      const NVTE_QKV_Format dqkv_format = nvte_get_qkv_format(cfg.dqkv_layout);
      if (dqkv_format != NVTE_QKV_Format::NVTE_BSHD && dqkv_format != NVTE_QKV_Format::NVTE_SBHD &&
          dqkv_format != NVTE_QKV_Format::NVTE_BHSD && dqkv_format != NVTE_QKV_Format::NVTE_THD) {
        return reject(message,
                      "FP8 fused attention supports BSHD/SBHD/BHSD/THD gradient formats, found " +
                          std::to_string(static_cast<int>(dqkv_format)) + ".");
      }
    }
    if (cfg.qkv_format == NVTE_QKV_Format::NVTE_THD) {
      if (cudnn_runtime_version < 92300) {
        return reject(message,
                      "FP8 fused attention with THD format requires cuDNN 9.23.0 or later!");
      }
      if (cfg.is_training && cfg.check_for_backward_support && sm_arch < 100) {
        return reject(message,
                      "FP8 fused attention with THD format supports backward on sm100+ only!");
      }
      if (sm_arch >= 100 && (cfg.head_dim_qk > 128 || cfg.head_dim_v > 128)) {
        return reject(message,
                      "FP8 fused attention with THD format supports head dimensions up to 128 on "
                      "sm100+ only!");
      }
    }
    if (cfg.is_bias) {
      return reject(message, "FP8 fused attention does not support pre/post_scale_bias yet!");
    }
    if (cfg.is_alibi) {
      return reject(message, "FP8 fused attention does not support ALiBi yet!");
    }
    const char *const recipe_reason =
        "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 "
        "recipes!";
    if (cfg.check_for_forward_support &&
        !(cfg.is_delayed_scaling_fwd || cfg.is_current_scaling_fwd || cfg.is_mxfp8_fwd)) {
      return reject(message, recipe_reason);
    }
    if (cfg.is_training && cfg.check_for_backward_support &&
        !(cfg.is_delayed_scaling_bwd || cfg.is_current_scaling_bwd || cfg.is_mxfp8_bwd)) {
      return reject(message, recipe_reason);
    }
    if (cfg.is_mxfp8 && cudnn_runtime_version < 92100) {
      return reject(message, "MXFP8 fused attention requires cuDNN 9.21.0 or later!");
    }

    // Run cuDNN support checks
    std::string cudnn_reason =
        each_pass([&](Pass pass) { return support_verdict_fp8(cfg, pass, handle); });
    if (!cudnn_reason.empty()) return reject(message, std::move(cudnn_reason));
    return NVTE_Fused_Attn_Backend::NVTE_FP8;
  }

  // Unsupported dtype
  return reject(message, "Unsupported QKV dtype qkv_dtype=" + std::to_string(cfg.qkv_dtype) + " .");
}

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) {
  NVTE_API_CALL(nvte_get_fused_attn_backend);
  transformer_engine::fused_attn::FusedAttnConfig cfg{};
  cfg.qkv_layout = qkv_layout;
  cfg.bias_type = bias_type;
  cfg.attn_mask_type = attn_mask_type;
  cfg.softmax_type = softmax_type;
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
  // fill in the missing fields with the most common use case;
  // otherwise it would return NVTE_No_Backend always
  cfg.batch_size = 1;
  cfg.num_tokens_q = cfg.batch_size * max_seqlen_q;
  cfg.num_tokens_kv = cfg.batch_size * max_seqlen_kv;
  cfg.o_format = nvte_get_q_format(qkv_layout);
  cfg.do_format = cfg.o_format;
  cfg.dqkv_layout = qkv_layout;
  if (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
    cfg.bias_batch_size = cfg.batch_size;
    cfg.bias_num_heads = num_attn_heads;
    cfg.bias_seqlen_q = max_seqlen_q;
    cfg.bias_seqlen_kv = max_seqlen_kv;
  }

  return nvte_get_fused_attn_backend_v2(reinterpret_cast<NVTEFusedAttnConfig>(&cfg),
                                        /*message=*/nullptr);
}

// Fused attention forward: create a config based on the params, check which backend supports it,
// and run that backend's implementation.
void nvte_fused_attn_fwd_v2(NVTEFusedAttnFwdParams params) {
  NVTE_API_CALL(nvte_fused_attn_fwd_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
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

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  FusedAttnConfig cfg = p.make_config();
  cfg.derive();
  const char *fused_attn_reject_reason = nullptr;
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend_v2(
      reinterpret_cast<NVTEFusedAttnConfig>(&cfg), &fused_attn_reject_reason);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    fused_attn_arbitrary_seqlen_fwd(cfg, input_Q, input_K, input_V, input_Bias, input_SoftmaxOffset,
                                    output_O, p.Aux_CTX_Tensors, input_cu_seqlens_q,
                                    input_cu_seqlens_kv, input_cu_seqlens_q_padded,
                                    input_cu_seqlens_kv_padded, input_page_table_k,
                                    input_page_table_v, input_rng_state, wkspace, p.stream, handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    fused_attn_fp8_fwd(cfg, input_Q, input_K, input_V, input_SoftmaxOffset, input_output_S,
                       output_O, p.Aux_CTX_Tensors, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state,
                       wkspace, p.stream, handle);
  } else {
    NVTE_ERROR("Fused attention is not supported for the user configuration: ",
               fused_attn_reject_reason);
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
  NVTE_API_CALL(nvte_fused_attn_fwd);
  transformer_engine::fused_attn::FusedAttnFwdParams p{};
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
  p.is_training = is_training;
  p.cuda_graph = cuda_graph;
  p.return_max_logit = return_max_logit;
  p.attn_mask_type = attn_mask_type;
  p.bias_type = bias_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.softmax_type = softmax_type;
  p.dropout = dropout;
  p.attn_scale = attn_scale;
  p.qkv_layout = qkv_layout;
  p.o_format = o_format;
  p.qkv_scale_inv_format = qkv_scale_inv_format;
  p.max_seqlen_q = max_seqlen_q;
  p.max_seqlen_kv = max_seqlen_kv;
  p.workspace = workspace;
  p.stream = stream;
  nvte_fused_attn_fwd_v2(reinterpret_cast<NVTEFusedAttnFwdParams>(&p));
}

// Fused attention backward. Same shape as nvte_fused_attn_fwd_v2, whose comment sketches the path;
// this one asks the selector for backward support and probes the backward builders.
void nvte_fused_attn_bwd_v2(NVTEFusedAttnBwdParams params) {
  NVTE_API_CALL(nvte_fused_attn_bwd_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
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

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  FusedAttnConfig cfg = p.make_config();
  // Derived here, not by the query below: the query works on its own copy, and it is this config
  // that goes on to the backend and must arrive with its derived fields filled in.
  cfg.derive();
  const char *fused_attn_reject_reason = nullptr;
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend_v2(
      reinterpret_cast<NVTEFusedAttnConfig>(&cfg), &fused_attn_reject_reason);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    size_t i = 0;
    Tensor *output_S = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_rng_state = convertNVTETensorCheck(p.Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_Bias = nullptr, *input_SoftmaxOffset = nullptr;
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
                       input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state,
                       wkspace, p.stream, handle);
  } else {
    NVTE_ERROR("Fused attention is not supported for this configuration: ",
               fused_attn_reject_reason);
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
  NVTE_API_CALL(nvte_fused_attn_bwd);
  transformer_engine::fused_attn::FusedAttnBwdParams p{};
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
  p.deterministic = deterministic;
  p.cuda_graph = cuda_graph;
  p.attn_mask_type = attn_mask_type;
  p.bias_type = bias_type;
  p.window_size_left = window_size_left;
  p.window_size_right = window_size_right;
  p.bottom_right_diagonal = bottom_right_diagonal;
  p.softmax_type = softmax_type;
  p.dropout = dropout;
  p.attn_scale = attn_scale;
  p.qkv_layout = qkv_layout;
  p.o_format = o_format;
  p.do_format = do_format;
  p.dqkv_layout = dqkv_layout;
  p.qkv_scale_inv_format = qkv_scale_inv_format;
  p.do_scale_inv_format = do_scale_inv_format;
  p.max_seqlen_q = max_seqlen_q;
  p.max_seqlen_kv = max_seqlen_kv;
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
