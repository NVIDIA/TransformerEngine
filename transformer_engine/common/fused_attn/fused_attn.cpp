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
// re-used (cleared + re-populated) on every call to nvte_get_fused_attn_backend on this thread
thread_local std::string fused_attn_backend_message_buffer;

void set_message(const char **message, const std::string &reason) {
  if (message == nullptr) return;
  fused_attn_backend_message_buffer = reason;
  *message = fused_attn_backend_message_buffer.c_str();
}

}  // namespace

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, size_t batch_size, NVTEDType q_dtype, NVTEDType kv_dtype, NVTEDType o_dtype,
    NVTEScalingMode scaling_mode, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type, float dropout,
    size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q, size_t max_seqlen_kv,
    size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left, int64_t window_size_right,
    bool bottom_right_diagonal, bool return_max_logit, bool cuda_graph, bool deterministic,
    const char **message) {
  using namespace transformer_engine;
  set_message(message, "");
  NVTE_CHECK(q_dtype == kv_dtype, "Q and KV must have the same data type.");

  cudnnHandle_t handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  const auto cudnn_runtime_version = cudnnGetVersion();

  // THD + 64-bit ragged offsets require cuDNN >= 9.5
  const bool requires_64bit_ragged_offset =
      (qkv_format == NVTE_THD && fused_attn::get_ragged_offset_dtype(
                                     layout_group, num_attn_heads, num_gqa_groups, max_seqlen_q,
                                     max_seqlen_kv, head_dim_qk, head_dim_v) == DType::kInt64);
  if (requires_64bit_ragged_offset && cudnn_runtime_version < 90500) {
    set_message(message,
                "Configuration requires 64-bit ragged offsets, which require "
                "cuDNN >= 9.5.");
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  // THD requires padding-style mask
  if (qkv_format == NVTE_QKV_Format::NVTE_THD &&
      attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
      attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
      attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) {
    set_message(message,
                "THD format requires PADDING / PADDING_CAUSAL / PADDING_CAUSAL_BOTTOM_RIGHT mask.");
    return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }

  const bool is_fp8 =
      (q_dtype == NVTEDType::kNVTEFloat8E4M3 || q_dtype == NVTEDType::kNVTEFloat8E5M2);
  const bool is_f16_or_bf16 =
      (q_dtype == NVTEDType::kNVTEFloat16 || q_dtype == NVTEDType::kNVTEBFloat16);

  if (is_fp8) {
    if (return_max_logit) {
      set_message(message, "FP8 fused attention does not support return_max_logit=True.");
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    if (qkv_format != NVTE_QKV_Format::NVTE_BSHD && qkv_format != NVTE_QKV_Format::NVTE_SBHD &&
        qkv_format != NVTE_QKV_Format::NVTE_BHSD) {
      set_message(message, "FP8 fused attention supports BSHD/SBHD/BHSD formats, found " +
                               std::to_string(static_cast<int>(qkv_format)) + ".");
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    const DType qkv_t = static_cast<DType>(q_dtype);
    const DType o_t = static_cast<DType>(o_dtype);
    std::string fwd_reason = is_supported_fp8_fwd(
        batch_size, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk,
        head_dim_v, is_training, dropout, qkv_layout, bias_type, attn_mask_type, softmax_type,
        window_size_left, window_size_right, bottom_right_diagonal, qkv_t, o_t, scaling_mode,
        handle);
    if (!fwd_reason.empty()) {
      set_message(message, fwd_reason);
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    if (is_training) {
      std::string bwd_reason = is_supported_fp8_bwd(
          batch_size, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk,
          head_dim_v, dropout, qkv_layout, bias_type, attn_mask_type, softmax_type,
          window_size_left, window_size_right, bottom_right_diagonal, deterministic, qkv_t,
          o_t, scaling_mode, handle);
      if (!bwd_reason.empty()) {
        set_message(message, bwd_reason);
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
    }
    return NVTE_Fused_Attn_Backend::NVTE_FP8;
  }

  if (is_f16_or_bf16) {
    if (cudnn_runtime_version <= 91500 && is_training &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (max_seqlen_kv % 128 != 0) && cuda_graph &&
        attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
        attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
        attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) {
      set_message(message, "Known cuDNN <= 9.15 issue with CUDA graph. Please upgrade cuDNN.");
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    const DType qkv_t = static_cast<DType>(q_dtype);
    std::string fwd_reason = is_supported_f16_fwd(
        batch_size, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk,
        head_dim_v, is_training, return_max_logit, dropout, qkv_layout, bias_type, attn_mask_type,
        softmax_type, window_size_left, window_size_right, bottom_right_diagonal, qkv_t,
        handle);
    if (!fwd_reason.empty()) {
      set_message(message, fwd_reason);
      return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
    }
    if (is_training) {
      std::string bwd_reason = is_supported_f16_bwd(
          batch_size, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk,
          head_dim_v, dropout, qkv_layout, bias_type, attn_mask_type, softmax_type,
          window_size_left, window_size_right, bottom_right_diagonal, deterministic, qkv_t,
          handle);
      if (!bwd_reason.empty()) {
        set_message(message, bwd_reason);
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
    }
    return NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
  }

  set_message(message, "Unsupported QKV dtype qkv_dtype=" + std::to_string(q_dtype) + " .");
  return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
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
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(cu_seqlens_kv_padded);
  const Tensor *input_page_table_k = convertNVTETensorCheck(page_table_k);
  const Tensor *input_page_table_v = convertNVTETensorCheck(page_table_v);
  const Tensor *input_rng_state = convertNVTETensorCheck(rng_state);
  const Tensor *input_Q = convertNVTETensorCheck(Q);
  const Tensor *input_K = convertNVTETensorCheck(K);
  const Tensor *input_V = convertNVTETensorCheck(V);
  const Tensor *input_Bias = convertNVTETensorCheck(Bias);
  const Tensor *input_SoftmaxOffset = convertNVTETensorCheck(SoftmaxOffset);
  Tensor *input_output_S = convertNVTETensorCheck(S);
  Tensor *output_O = convertNVTETensorCheck(O);
  Tensor *wkspace = convertNVTETensor(workspace);

  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
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
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) {
    NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
    if (kv_format == NVTE_QKV_Format::NVTE_BSHD) {
      num_pages_k = input_K->data.shape[0];
      page_size_k = input_K->data.shape[1];
      num_pages_v = input_V->data.shape[0];
      page_size_v = input_V->data.shape[1];
    } else if (kv_format == NVTE_QKV_Format::NVTE_SBHD) {
      num_pages_k = input_K->data.shape[1];
      page_size_k = input_K->data.shape[0];
      num_pages_v = input_V->data.shape[1];
      page_size_v = input_V->data.shape[0];
    }
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);
  const NVTEDType O_type = static_cast<NVTEDType>(output_O->data.dtype);
  const NVTEScalingMode scaling_mode = input_Q->scaling_mode;

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      is_training, b, Q_type, KV_type, O_type, scaling_mode, qkv_layout, bias_type, attn_mask_type,
      softmax_type, dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, window_size_left,
      window_size_right, bottom_right_diagonal, return_max_logit, cuda_graph,
      /*deterministic=*/false, /*message=*/nullptr);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    fused_attn_arbitrary_seqlen_fwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, num_pages_k, num_pages_v,
        page_size_k, page_size_v, max_pages_per_seq_k, max_pages_per_seq_v, is_training,
        return_max_logit, attn_scale, dropout, qkv_layout, o_format, bias_type, attn_mask_type,
        softmax_type, window_size_left, window_size_right, bottom_right_diagonal, input_Q, input_K,
        input_V, input_Bias, input_SoftmaxOffset, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
        input_cu_seqlens_kv, input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded,
        input_page_table_k, input_page_table_v, input_rng_state, wkspace, stream, handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    fused_attn_fp8_fwd(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, is_training,
                       attn_scale, dropout, qkv_layout, o_format, qkv_scale_inv_format, bias_type,
                       attn_mask_type, softmax_type, window_size_left, window_size_right,
                       bottom_right_diagonal, input_Q, input_K, input_V, input_SoftmaxOffset,
                       input_output_S, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
                       input_cu_seqlens_kv, input_rng_state, wkspace, stream, handle);
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
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(cu_seqlens_kv_padded);
  const Tensor *input_Q = convertNVTETensorCheck(Q);
  const Tensor *input_K = convertNVTETensorCheck(K);
  const Tensor *input_V = convertNVTETensorCheck(V);
  const Tensor *input_O = convertNVTETensorCheck(O);
  const Tensor *input_dO = convertNVTETensorCheck(dO);
  const Tensor *input_S = convertNVTETensorCheck(S);
  Tensor *input_output_dP = convertNVTETensorCheck(dP);
  Tensor *output_dQ = convertNVTETensorCheck(dQ);
  Tensor *output_dK = convertNVTETensorCheck(dK);
  Tensor *output_dV = convertNVTETensorCheck(dV);
  Tensor *output_dBias = convertNVTETensorCheck(dBias);
  Tensor *output_dSoftmaxOffset = convertNVTETensorCheck(dSoftmaxOffset);
  Tensor *wkspace = convertNVTETensor(workspace);

  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
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
  const NVTEDType O_type = static_cast<NVTEDType>(input_O->data.dtype);
  const NVTEScalingMode scaling_mode = input_Q->scaling_mode;

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      /*is_training=*/true, b, Q_type, KV_type, O_type, scaling_mode, qkv_layout, bias_type,
      attn_mask_type, softmax_type, dropout, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v,
      window_size_left, window_size_right, bottom_right_diagonal, /*return_max_logit=*/false,
      cuda_graph, deterministic, /*message=*/nullptr);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    size_t i = 0;
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    Tensor *input_Bias, *input_SoftmaxOffset;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_Bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_arbitrary_seqlen_bwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, attn_scale, dropout,
        qkv_layout, o_format, do_format, dqkv_layout, bias_type, attn_mask_type, softmax_type,
        window_size_left, window_size_right, bottom_right_diagonal, deterministic, input_Q, input_K,
        input_V, input_O, input_dO, input_Bias, input_SoftmaxOffset, output_S, output_dQ, output_dK,
        output_dV, output_dBias, output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv,
        input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded, input_rng_state, wkspace, stream,
        handle);
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
    size_t i = 0;
    const Tensor *input_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    const Tensor *input_SoftmaxOffset = nullptr;
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      input_SoftmaxOffset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    const Tensor *input_dO_f16 = nullptr;
    if (input_dO->scaling_mode == NVTE_MXFP8_1D_SCALING) {
      input_dO_f16 = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    }
    fused_attn_fp8_bwd(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, attn_scale, dropout,
                       qkv_layout, o_format, do_format, dqkv_layout, qkv_scale_inv_format,
                       do_scale_inv_format, bias_type, attn_mask_type, softmax_type,
                       window_size_left, window_size_right, bottom_right_diagonal, deterministic,
                       input_Q, input_K, input_V, input_O, input_dO, input_dO_f16, input_M, input_S,
                       input_SoftmaxOffset, input_output_dP, output_dQ, output_dK, output_dV,
                       output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_rng_state, wkspace, stream, handle);
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
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
