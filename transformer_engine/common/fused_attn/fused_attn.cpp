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

// The per-thread storage for the diagnostic string; it is re-used (cleared + re-populated)
// on every call to nvte_get_fused_attn_backend_v2 on the same thread.
thread_local std::string fused_attn_backend_message_buffer;

// Stash `reason` in the thread-local buffer and, if the caller asked for a diagnostic,
// publish a NUL-terminated pointer to it via `*message`. Safe to call with `message == nullptr`.
void set_message(const char **message, std::string reason) {
  if (message == nullptr) return;
  fused_attn_backend_message_buffer = std::move(reason);
  *message = fused_attn_backend_message_buffer.c_str();
}

// Records `reason` and answers with the backend that means "none", so that a rejection reads as
// the one statement it is: `if (cond) return reject(message, "why");`. Every rejection in
// nvte_get_fused_attn_backend_v2 goes through here, which is what keeps a reason attached to
// each: the value cannot be produced without one. nodiscard because dropping the value would
// leave the message set and the rejection unreturned, and the function would carry on.
[[nodiscard]] NVTE_Fused_Attn_Backend reject(const char **message, std::string reason) {
  set_message(message, std::move(reason));
  return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
}

}  // namespace

// select a backend for fused attention; the diagnostic message is based on the first failure, not cumulative.
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(NVTEFusedAttnConfig config,
                                                       const char **message) {
  NVTE_API_CALL(nvte_get_fused_attn_backend_v2);
  using namespace transformer_engine;
  using namespace transformer_engine::fused_attn;
  // Derived on a copy, leaving the caller's config untouched: this function answers a question
  // about a configuration and has no business editing one, and a query that wrote to its argument
  // could not be asked about the same config from two threads at once. The copy costs nothing that
  // matters here, since deriving is a version check and some arithmetic.
  //
  // The execution path derives its own config before calling this (see nvte_fused_attn_fwd_v2),
  // and still reuses whatever graph the query builds: both derive the same fields from the same
  // inputs, so make_cache_key() lands on the same entry. Deriving is idempotent, so re-deriving
  // an already-derived config here changes nothing.
  FusedAttnConfig cfg = *get_fused_attn_config(config);
  set_message(message, "");

  cudnnHandle_t handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const auto qkv_format = nvte_get_qkv_format(cfg.qkv_layout);
  const auto layout_group = nvte_get_qkv_layout_group(cfg.qkv_layout);
  const auto cudnn_runtime_version = cudnnGetVersion();
  // Read from attn_mask_type rather than from cfg.is_padding, because the two rules that need it
  // are stated before derive() runs; see the derive() call below.
  const bool has_padding_mask =
      cfg.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
      cfg.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
      cfg.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK;

  // THD + 64-bit ragged offsets require cuDNN >= 9.5
  const bool requires_64bit_ragged_offset =
      (qkv_format == NVTE_THD &&
       fused_attn::get_ragged_offset_dtype(layout_group, cfg.num_attn_heads, cfg.num_gqa_groups,
                                           cfg.max_seqlen_q, cfg.max_seqlen_kv, cfg.head_dim_qk,
                                           cfg.head_dim_v) == DType::kInt64);
  if (requires_64bit_ragged_offset && cudnn_runtime_version < 90500) {
    return reject(message,
                  "Configuration requires 64-bit ragged offsets, which require cuDNN >= 9.5.");
  }

  // THD requires padding-style mask
  if (qkv_format == NVTE_QKV_Format::NVTE_THD && !has_padding_mask) {
    return reject(
        message,
        "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
  }

  // Paged KV requires padding-style mask, for the same reason THD does: the graph is built at
  // padded dimensions and the mask is what tells cuDNN where the real tokens end.
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD && !has_padding_mask) {
    return reject(message,
                  "Paged KV requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
  }

  // Derived here rather than above, so that the two rules stated above are answered rather than
  // thrown. Both are invariants derive() asserts, and an assertion that fired first would leave
  // this function no chance to report them as an unsupported configuration.
  cfg.derive();

  // Ragged Q/KV requires sm90+, the rule the hand-written support matrix this function replaced
  // carried as `qkv_format == NVTE_THD && sm_arch_ >= 90`. Below sm90 the only graph we can build
  // is the dense max_seqlen one -- cfg.uses_packed_ragged_graph is false -- so SDPA_backward
  // never gets max_total_seq_len_q/kv and its dQ/dK/dV come back wrong.
  //
  // This is ours to state because it is a wrong-result rejection, and check_support answers a
  // different question: whether cuDNN can run the graph, not whether the graph computes what we
  // asked for. cuDNN's own answer has moved, which is what makes the distinction worth spelling
  // out here. Its frontend gates ragged SDPA on `sm < 90 && cudnn < 9.18.1`, so through 9.18.0 it
  // would have refused this configuration for us and the rule below is redundant; from 9.18.1 it
  // accepts sm80/sm89 ragged and the rule is the only thing standing between a THD model on an
  // A100 and silently wrong gradients. Lifting it is a change to the graphs TE builds -- packed
  // ragged shapes, and the Stats/LSE layouts that go with them, which cuDNN documents as
  // differing on sm8x -- not a change to this condition, and that work is deliberately not part
  // of this refactor.
  //
  // sm120 takes that same dense path and is left enabled, as it was before this refactor;
  // whether it has the same problem is a separate question from restoring the sm90 rule.
  if ((cfg.is_ragged_q || cfg.is_ragged_kv) && cuda::sm_arch(cuda::current_device()) < 90) {
    return reject(message, "Ragged (THD) Q or KV requires compute capability 9.0 or higher.");
  }

  // TE's cuDNN fused-attention graph does not represent pre-scale bias.
  if (cfg.bias_type == NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS) {
    return reject(message, "Fused attention does not support pre-scale bias.");
  }

  const bool is_fp8 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat8E4M3 || cfg.qkv_dtype == NVTEDType::kNVTEFloat8E5M2);
  const bool is_f16_or_bf16 =
      (cfg.qkv_dtype == NVTEDType::kNVTEFloat16 || cfg.qkv_dtype == NVTEDType::kNVTEBFloat16);

  // Ask `verdict` about each direction the caller wants, and report the first refusal: the empty
  // string means every direction asked about is served. Stated once here because every rule that
  // is direction-dependent has to be asked this same way, and two copies of the gating would be
  // two chances to probe a direction the caller never asked about.
  //
  // Forward is asked first because a config that cannot run forward cannot train either, and the
  // forward refusal is the more useful of the two to report. Backward is skipped for inference,
  // where no backward graph is ever built.
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

  // cuDNN's own verdict on `backend`. The two backends differ only in which set of graphs gets
  // built, so they share this; what is theirs alone are the rules in each branch below.
  //
  // Each backend answers for both directions from its own translation unit, the only place that can
  // name the graph builders, so choosing between them is all the dispatch left to do here.
  auto probe = [&](Backend backend) -> std::string {
    return each_pass([&](Pass pass) {
      return backend == Backend::FP8 ? support_verdict_fp8(cfg, pass, handle)
                                     : support_verdict_f16(cfg, pass, handle);
    });
  };

  if (is_fp8) {
    if (cfg.return_max_logit) {
      return reject(message, "FP8 fused attention does not support return_max_logit=True.");
    }
    if (qkv_format != NVTE_QKV_Format::NVTE_BSHD && qkv_format != NVTE_QKV_Format::NVTE_SBHD &&
        qkv_format != NVTE_QKV_Format::NVTE_BHSD) {
      return reject(message, "FP8 fused attention supports BSHD/SBHD/BHSD formats, found " +
                                 std::to_string(static_cast<int>(qkv_format)) + ".");
    }
    // The rest of what the FP8 graphs cannot represent: bias, ALiBi, and the quantization recipes
    // they are not written for. TE's rules rather than cuDNN's, and stated here rather than in the
    // build path for the reason all the rules above are: a rejection stated here is an answer
    // carrying its reason, where the same rule inside a graph build would have to travel out as an
    // exception.
    if (cfg.bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
      return reject(message, "FP8 fused attention does not support pre/post_scale_bias yet!");
    }
    if (cfg.bias_type == NVTE_Bias_Type::NVTE_ALIBI) {
      return reject(message, "FP8 fused attention does not support ALiBi yet!");
    }

    // Whether the config names a recipe the FP8 graphs are written for at all. Delayed scaling
    // writes FP8 out and keeps its scale, current scaling writes F16 and computes one, MXFP8 writes
    // F16 with block scales; every other pairing of scaling mode and output dtype is refused here.
    //
    // Per direction, because the pairing is read off what each pass stores, which is also how the
    // graph builders read which of the three they are building for -- off cfg.o_is_fp8 and
    // cfg.dqkv_is_fp8, taking "not FP8" to mean F16. That reading is sound only because this
    // refusal has already happened, so the two belong to each other: a pairing accepted here must
    // be one they read the same way, and anything added to either belongs in both.
    std::string recipe_reason = each_pass([&](Pass pass) -> std::string {
      const NVTEDType out_dtype = (pass == Pass::Fwd) ? cfg.o_dtype : cfg.dqkv_dtype;
      const bool out_is_fp8 = (pass == Pass::Fwd) ? cfg.o_is_fp8 : cfg.dqkv_is_fp8;
      const bool out_is_f16 = (out_dtype == kNVTEFloat16 || out_dtype == kNVTEBFloat16);
      const bool serves_this_output =
          (cfg.scaling_mode == NVTE_DELAYED_TENSOR_SCALING && (out_is_fp8 || out_is_f16)) ||
          (cfg.scaling_mode == NVTE_MXFP8_1D_SCALING && out_is_f16);
      if (!serves_this_output) {
        return "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 "
               "recipes!";
      }
      return "";
    });
    if (!recipe_reason.empty()) return reject(message, std::move(recipe_reason));

    // Asked after the pairing above, not with it: a config that names no recipe at all should hear
    // that rather than be sent to upgrade cuDNN for a recipe it was not asking for.
    if (cfg.scaling_mode == NVTE_MXFP8_1D_SCALING && cudnn_runtime_version < 92100) {
      return reject(message, "MXFP8 fused attention requires cuDNN 9.21.0 or later!");
    }

    std::string reason = probe(Backend::FP8);
    if (!reason.empty()) return reject(message, std::move(reason));
    return NVTE_Fused_Attn_Backend::NVTE_FP8;
  }

  if (is_f16_or_bf16) {
    // TODO(cyanguwa): re-validate BRCM + cross-attention on sm100 with cuDNN <= 9.7. The
    // hand-written support matrix this function replaced rejected bottom-right-diagonal masks
    // with max_seqlen_q != max_seqlen_kv there, for a cuDNN bug fixed in 9.7. cuDNN's own
    // check_support is the authority now, so the guard is gone; it needs to come back as an
    // explicit rejection here, like the CUDA-graph one below, if that bug is a wrong-result
    // bug rather than a support gap check_support reports for itself.
    if (cudnn_runtime_version <= 91500 && cfg.is_training &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (cfg.max_seqlen_kv % 128 != 0) && cfg.cuda_graph &&
        cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
        cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
        cfg.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) {
      return reject(message, "Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN.");
    }
    std::string reason = probe(Backend::F16);
    if (!reason.empty()) return reject(message, std::move(reason));
    return NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  }

  return reject(message, "Unsupported QKV dtype qkv_dtype=" + std::to_string(cfg.qkv_dtype) + " .");
}

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) {
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
  // fill in missing fields so it doesn't always return NVTE_No_Backend
  cfg.batch_size = 1;
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

// Fused attention forward: derive the config, ask the selector which backend can run it, and run
// that backend's implementation.
//
// Support is decided by building the graph rather than by consulting a table of rules, and the
// support query and the execution path reach the same cache through the same accessor. That is
// what the HIT below means: by the time a backend has been selected, the entry the implementation
// needs has already been built and inserted by the probe that selected it, so what was checked is
// what runs. The rules the selector does state for itself are the ones cuDNN cannot answer: either
// about whether the graph computes what was asked for rather than whether cuDNN can run it, or
// about what TE's graphs can represent in the first place, which is where the FP8 recipes come in.
// Stating them here rather than inside a build is what lets each one answer with its reason.
//
//   nvte_fused_attn_fwd_v2
//     |
//     +-- cfg = p.make_config(), which sets check_for_forward_support; cfg.derive()
//     |
//     +-- nvte_get_fused_attn_backend_v2                            the support query
//     |     |
//     |     +-- TE's own rules: THD and paged KV need a padding mask, no pre-scale bias,
//     |     |     ragged Q/KV needs sm90+, the cuDNN 9.15-and-older CUDA-graph bug, and for FP8
//     |     |     bias, ALiBi and the quantization recipes its graphs are not written for
//     |     |     `-- reject -> NVTE_No_Backend + reason -> the NVTE_ERROR below
//     |     |
//     |     `-- probe(backend) -> support_verdict_f16 / support_verdict_fp8, with Pass::Fwd
//     |           `-- support_verdict<F16, Fwd, ...>()
//     |                 `-- get_graph<F16, Fwd, ...>(): builds and inserts the entry, or throws,
//     |                       in which case cuDNN's message becomes the reason for the refusal
//     |
//     `-- fused_attn_arbitrary_seqlen_fwd -> ..._fwd_impl           the selected backend
//           |
//           +-- get_graph<F16, Fwd, ...>() HIT: the entry the query above just built
//           +-- build_plans()                the kernel compilation, once per entry
//           `-- bind device pointers, graph.execute()
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
  // Derived here, not by the query below: the query works on its own copy, and it is this config
  // that goes on to the backend and must arrive with its derived fields filled in.
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
                       input_rng_state, wkspace, p.stream, handle);
  } else {
    const char *const reject_reason =
        (fused_attn_reject_reason != nullptr && fused_attn_reject_reason[0] != '\0')
            ? fused_attn_reject_reason
            : "no cuDNN fused-attention backend supports the requested parameters";
    NVTE_ERROR("Fused attention is not supported for this configuration: ", reject_reason);
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

// Fused attention backward. The same shape as nvte_fused_attn_fwd_v2, which sketches the path from
// an entry point through the selector to the cache; the only differences are that the config asks
// the selector for backward support and that the backward builders are the ones the probe runs.
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
                       input_rng_state, wkspace, p.stream, handle);
  } else {
    const char *const reject_reason =
        (fused_attn_reject_reason != nullptr && fused_attn_reject_reason[0] != '\0')
            ? fused_attn_reject_reason
            : "no cuDNN fused-attention backend supports the requested parameters";
    NVTE_ERROR("Fused attention is not supported for this configuration: ", reject_reason);
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
