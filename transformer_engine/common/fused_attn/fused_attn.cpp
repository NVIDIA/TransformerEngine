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

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(const NVTEFusedAttnConfig *cfg) {
  using namespace transformer_engine;
  NVTE_CHECK(cfg != nullptr, "NVTEFusedAttnConfig pointer must not be null.");
  NVTE_CHECK(cfg->struct_size >= sizeof(NVTEFusedAttnConfig),
             "NVTEFusedAttnConfig::struct_size is smaller than the library expects; "
             "did you forget NVTE_FUSED_ATTN_CONFIG_INIT?");

  // Bind config fields to the local names that the original implementation
  // below was written against, so the body can stay as-is.
  const bool is_training = cfg->is_training;
  const NVTEDType q_dtype = cfg->q_dtype;
  const NVTEDType kv_dtype = cfg->kv_dtype;
  const NVTE_QKV_Layout qkv_layout = cfg->qkv_layout;
  const NVTE_Bias_Type bias_type = cfg->bias_type;
  const NVTE_Mask_Type attn_mask_type = cfg->attn_mask_type;
  const NVTE_Softmax_Type softmax_type = cfg->softmax_type;
  const float dropout = cfg->dropout;
  const size_t num_attn_heads = cfg->num_attn_heads;
  const size_t num_gqa_groups = cfg->num_gqa_groups;
  const size_t max_seqlen_q = cfg->max_seqlen_q;
  const size_t max_seqlen_kv = cfg->max_seqlen_kv;
  const size_t head_dim_qk = cfg->head_dim_qk;
  const size_t head_dim_v = cfg->head_dim_v;
  const int64_t window_size_left = cfg->window_size_left;
  const int64_t window_size_right = cfg->window_size_right;
  const bool return_max_logit = cfg->return_max_logit;
  const bool cuda_graph = cfg->cuda_graph;
  const bool deterministic = cfg->deterministic;

  NVTE_Fused_Attn_Backend backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);
  NVTE_CHECK(q_dtype == kv_dtype, "Q and KV must have the same data type.");
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  auto cudnn_runtime_version = cudnnGetVersion();

  // For ragged offsets we only support 32-bit prior to cuDNN 9.5
  // Only used when THD format is requested.
  const bool requires_64bit_ragged_offset =
      (qkv_format == NVTE_THD && fused_attn::get_ragged_offset_dtype(
                                     layout_group, num_attn_heads, num_gqa_groups, max_seqlen_q,
                                     max_seqlen_kv, head_dim_qk, head_dim_v) == DType::kInt64);
  const bool supported_ragged_offset_size =
      (!requires_64bit_ragged_offset || cudnn_runtime_version >= 90500);

  if ((q_dtype == NVTEDType::kNVTEFloat8E4M3 || q_dtype == NVTEDType::kNVTEFloat8E5M2) &&
      sm_arch_ >= 90 && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
      (
          // 9.2.1: {bshd, sbhd}, any seqlen, d=128, {no_mask, causal}
          (cudnn_runtime_version >= 90201 && sm_arch_ < 100 && max_seqlen_q % 128 == 0 &&
           max_seqlen_kv % 128 == 0 && head_dim_qk == 128 && head_dim_v == 128 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) ||
          // 9.7: {bshd, sbhd}, any seqlen, d<=256 for sm90 and d<=128 for sm100, {padding, padding_causal}
          (cudnn_runtime_version >= 90700 &&
           // TODO (cyang): add is_training to nvte_get_fused_attn_backend
           // sm90: fwd d<=256, bwd d=128 only
           // sm100: fwd d<=128, bwd d<=128
           ((sm_arch_ < 100 && (!is_training) && head_dim_qk <= 256 && head_dim_v <= 256) ||
            (sm_arch_ < 100 && is_training && head_dim_qk == 128 && head_dim_v == 128) ||
            (sm_arch_ >= 100 && head_dim_qk <= 128 && head_dim_v <= 128)) &&
           head_dim_qk % 16 == 0 && head_dim_v % 16 == 0 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)) ||
          // 9.21: d_qk=192, d_v=128
          (cudnn_runtime_version >= 92100 && sm_arch_ >= 100 && head_dim_qk <= 192 &&
           head_dim_v <= 128 && head_dim_qk % 16 == 0 && head_dim_v % 16 == 0 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK))) &&
      // pre-9.21: {bshd, sbhd}, {vanilla}
      // 9.21+: {bshd, sbhd, bhsd}, {vanilla, off-by-one, learnable}
      ((cudnn_runtime_version < 92100 &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        softmax_type == NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX) ||
       (cudnn_runtime_version >= 92100 &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD ||
         qkv_format == NVTE_QKV_Format::NVTE_BHSD))) &&
      !requires_64bit_ragged_offset &&
      // 9.10.0: known bugs with SDPA FP8
      (cudnn_runtime_version != 91000) && !return_max_logit) {
    backend = NVTE_Fused_Attn_Backend::NVTE_FP8;
  } else if ((q_dtype == NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16)) {
    bool flag_arb = false;
    if (
        // TODO(cyang): replace with cudnn-frontend check_support for cleaner logic and better error messaging
        // architecture
        ((cudnn_runtime_version < 8903 && (sm_arch_ == 80 || sm_arch_ == 90)) ||
         (cudnn_runtime_version >= 8903 && sm_arch_ >= 80 && sm_arch_ < 100) ||
         (cudnn_runtime_version >= 90700 && sm_arch_ >= 100)) &&
        // sequence length
        ((cudnn_runtime_version < 90000 && max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0) ||
         (cudnn_runtime_version >= 90000)) &&
        // number of heads
        ((cudnn_runtime_version < 8907 && num_attn_heads == num_gqa_groups) ||
         (cudnn_runtime_version >= 8907)) &&
        // head dimension
        // multiples of 8
        (head_dim_qk % 8 == 0 && head_dim_v % 8 == 0 &&
         // <= 128
         ((head_dim_qk <= 128 && head_dim_v <= 128) ||
          // 9.1: <= 256 + Hopper + fprop
          // 9.5: <= 256 + Hopper + bprop
          (head_dim_qk <= 256 && head_dim_v <= 256 &&
           ((!is_training && sm_arch_ == 90 && cudnn_runtime_version >= 90100) ||
            (is_training && sm_arch_ == 90 && cudnn_runtime_version >= 90500))) ||
          // 9.9: any head_dim + Blackwell + fprop + non_paged + sq > 1
          (!is_training && sm_arch_ >= 100 && cudnn_runtime_version >= 90900 && max_seqlen_q > 1 &&
           layout_group != NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) ||
          // 9.10.2: any head_dim + any arch + fprop + paged
          // 9.10.2: any head_dim + any arch + fprop + non_paged + sq > 1
          // 9.10.2: any head_dim + any arch + fprop + non_paged + sq = 1 + {no_mask, padding, BRCM, padding_BRCM}
          (!is_training && cudnn_runtime_version >= 91002 &&
           (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD || max_seqlen_q > 1 ||
            (max_seqlen_q == 1 && attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_MASK &&
             attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK))) ||
          // 9.11: d_qk = 192, d_v = 128 + Blackwell + bprop + non-paged
          (head_dim_qk == 192 && head_dim_v == 128 && is_training && sm_arch_ >= 100 &&
           cudnn_runtime_version >= 91100)) &&
         // 9.11+ bug: 128 < d_qk <= 256, 128 < d_v <= 256 + Hopper + bprop + MLA
         // Conditional to temporarily use blanket cudnn_runtime_version >= 9.11 until fixed
         (!((cudnn_runtime_version >= 91100) && is_training && sm_arch_ == 90 &&
            head_dim_qk >= 128 && head_dim_v >= 128 && !(head_dim_qk == 192 && head_dim_v == 128) &&
            head_dim_qk != head_dim_v))) &&
        // bias type
        ((cudnn_runtime_version < 8906 && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) ||
         (cudnn_runtime_version >= 8906 &&
          (bias_type == NVTE_Bias_Type::NVTE_NO_BIAS ||
           (bias_type == NVTE_Bias_Type::NVTE_ALIBI &&
            attn_mask_type != NVTE_Mask_Type::NVTE_NO_MASK &&
            attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
            attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
            attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
            sm_arch_ >= 90) ||
           (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && sm_arch_ >= 90))) ||
         (cudnn_runtime_version >= 90000 &&
          (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && sm_arch_ >= 80))) &&
        // mask type
        // pre-8.9.6: causal
        ((cudnn_runtime_version < 8906 && attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
         // 8.9.6: {bshd, sbhd} + {no_mask, causal, padding, padding_causal}
         (cudnn_runtime_version >= 8906 &&
          (qkv_format == NVTE_QKV_Format::NVTE_SBHD || qkv_format == NVTE_QKV_Format::NVTE_BSHD) &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) ||
         // 9.1: adds thd + {padding, padding_causal}
         (cudnn_runtime_version >= 90100 && qkv_format == NVTE_QKV_Format::NVTE_THD &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)) ||
         // 9.3: adds {bshd, sbhd} + causal_bottom_right + self/cross-attn (sq <= skv)
         (cudnn_runtime_version >= 90300 &&
          (qkv_format == NVTE_QKV_Format::NVTE_SBHD || qkv_format == NVTE_QKV_Format::NVTE_BSHD) &&
          attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
          max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0 && max_seqlen_q <= max_seqlen_kv &&
          bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
         // 9.5: adds {paged_kv_bshd, paged_kv_sbhd} + {padding, padding_causal, padding_causal_bottom_right}
         (cudnn_runtime_version >= 90500 &&
          layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
           (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
            max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0 && max_seqlen_q <= max_seqlen_kv)) &&
          bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
         // 9.6: adds {bshd, sbhd, thd} + padding_causal_bottom_right + self/cross-attn (sq <= skv)
         (cudnn_runtime_version >= 90600 &&
          attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
          max_seqlen_q % 64 == 0 && max_seqlen_kv % 64 == 0 && max_seqlen_q <= max_seqlen_kv &&
          bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
         // 9.7: removes s_q/s_kv % 64 = 0 for {causal_bottom_right, padding_causal_bottom_right}
         // for any q_format/kv_format, and paged/non-paged
         (cudnn_runtime_version >= 90700 &&
          (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
           attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
           ((attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
            bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && dropout == 0.0) ||
           ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
            max_seqlen_q <= max_seqlen_kv)))) &&
        // bias + mask combination
        (!(cudnn_runtime_version >= 8906 &&
           (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
            attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) &&
           bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)) &&
        // qkv format
        (qkv_format == NVTE_QKV_Format::NVTE_SBHD || qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
         qkv_format == NVTE_QKV_Format::NVTE_BHSD ||
         (qkv_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90 &&
          ((cudnn_runtime_version >= 90100 && num_attn_heads == num_gqa_groups) ||
           cudnn_runtime_version >= 90600)) ||
         ((q_format == NVTE_QKV_Format::NVTE_SBHD || q_format == NVTE_QKV_Format::NVTE_BSHD ||
           q_format == NVTE_QKV_Format::NVTE_BHSD ||
           (q_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90) ||
           kv_format == NVTE_QKV_Format::NVTE_SBHD || kv_format == NVTE_QKV_Format::NVTE_BSHD ||
           kv_format == NVTE_QKV_Format::NVTE_BHSD ||
           (kv_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90)) &&
          cudnn_runtime_version >= 90700)) &&
        // sliding window
        // pre-9.2: full attn, causal
        ((cudnn_runtime_version < 90200 && window_size_left == -1 &&
          (window_size_right == -1 || window_size_right == 0)) ||
         // 9.2: SWA (left, 0) + top-left diagonal + {bshd, sbhd}
         (cudnn_runtime_version >= 90200 &&
          ((window_size_left == -1 && window_size_right == -1 &&
            attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK) ||
           ((window_size_left == -1 || window_size_left >= 0) && window_size_right == 0 &&
            (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
             (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
              max_seqlen_q == max_seqlen_kv)) &&
            max_seqlen_q <= max_seqlen_kv && dropout == 0.0 &&
            bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
            (qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
             qkv_format == NVTE_QKV_Format::NVTE_SBHD)))) ||
         // 9.6: SWA (left, 0) + top-left/bottom-right diagonal + {bshd, sbhd, thd}
         (cudnn_runtime_version >= 90600 &&
          ((window_size_left == -1 && (window_size_right == -1 || window_size_right == 0)) ||
           ((window_size_left >= 0 || window_size_left == -1) &&
            (window_size_right >= 0 || window_size_right == -1) &&
            ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
              // TODO(cyang): fix bug for BRCM + cross-attention on sm100
              (sm_arch_ < 100 || (sm_arch_ >= 100 && ((max_seqlen_q == max_seqlen_kv &&
                                                       cudnn_runtime_version <= 90700) ||
                                                      cudnn_runtime_version > 90700)))) ||
             attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
             attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
             (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
              (sm_arch_ < 100 || (sm_arch_ >= 100 && ((max_seqlen_q == max_seqlen_kv &&
                                                       cudnn_runtime_version <= 90700) ||
                                                      cudnn_runtime_version > 90700))))) &&
            max_seqlen_q <= max_seqlen_kv && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
            dropout == 0.0)))) &&
        // check 64-bit ragged offset support
        (supported_ragged_offset_size) &&
        // 9.10.0/9.10.1: known bugs with SDPA F16
        (cudnn_runtime_version != 91000) && (cudnn_runtime_version != 91001) &&
        // softmax type
        // pre-9.13.1: vanilla
        // 9.13.1+: vanilla, off-by-one, learnable
        (cudnn_runtime_version >= 91301 ||
         (cudnn_runtime_version < 91301 &&
          softmax_type == NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX)) &&
        // determinism on Blackwell
        // pre-9.18.1: fwd: deterministic; bwd: non-deterministic
        // 9.18.1+: fwd: deterministic; bwd: non-deterministic/deterministic
        (sm_arch_ < 100 ||
         (sm_arch_ >= 100 && (!is_training ||
                              (is_training && !deterministic &&
                               (dropout == 0.0 || bias_type == NVTE_Bias_Type::NVTE_NO_BIAS)) ||
                              (is_training && deterministic && cudnn_runtime_version >= 91801 &&
                               dropout == 0.0 && bias_type == NVTE_Bias_Type::NVTE_NO_BIAS))))) {
      flag_arb = true;
    }
    if (flag_arb) {
      backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
    }
    if (cudnn_runtime_version < 8900 &&
        backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP16/BF16 fused attention is supported by cuDNN 8.9.0+."
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if ((cudnn_runtime_version == 91400) && (max_seqlen_kv > 1024) && (window_size_left != -1) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_MASK) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK)) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: Given combination of attention mask (non-causal) and "
                   "max_seqlen_kv (> 1024) does not support fused attention for cuDNN 9.14.0. "
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if ((cudnn_runtime_version <= 91500) && is_training &&
        (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
        (max_seqlen_kv % 128 != 0) && cuda_graph &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) &&
        (attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK)) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: Given combination of attention mask (non-padding),"
                   " max_seqlen_kv (not divisible by 128), and qkv_format (BSHD/SBHD) for"
                   " backward fused attention with graph capture requires cuDNN 9.15.1+. "
                   "Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen && sm_arch_ == 120) {
      if (cudnn_runtime_version < 91801) {
        backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
        std::cout << "Warning: Given combination of sm_arch_ == 120 and cudnn_runtime_version < "
                     "91801 is not supported. "
                  << " Please upgrade your cuDNN version if possible." << std::endl;
      } else if (deterministic && is_training) {
        backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
        std::cout << "Warning: Deterministic fused attention on SM120 is not supported."
                  << std::endl;
      } else {
        // Known missing support for T3HD/TH3D layouts on SM120
        const bool is_t3hd_or_th3d =
            (qkv_layout == NVTE_QKV_Layout::NVTE_T3HD || qkv_layout == NVTE_QKV_Layout::NVTE_TH3D);
        if (is_t3hd_or_th3d) {
          backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
          std::cout << "Warning: Given combination of T3HD/TH3D layouts on SM120 is not supported. "
                    << " Please consider using other THD layouts if possible." << std::endl;
        }
      }
    }
  } else {
    backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  return backend;
}

// Deprecated: forwards to nvte_get_fused_attn_backend_v2.
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic) {
  NVTEFusedAttnConfig cfg = NVTE_FUSED_ATTN_CONFIG_INIT;
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
  cfg.q_dtype = q_dtype;
  cfg.kv_dtype = kv_dtype;
  cfg.num_attn_heads = num_attn_heads;
  cfg.num_gqa_groups = num_gqa_groups;
  cfg.head_dim_qk = head_dim_qk;
  cfg.head_dim_v = head_dim_v;
  cfg.is_training = is_training;
  cfg.return_max_logit = return_max_logit;
  cfg.deterministic = deterministic;
  return nvte_get_fused_attn_backend_v2(&cfg);
}

// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd_v2(const NVTEFusedAttnFwdParams *params) {
  NVTE_API_CALL(nvte_flash_attn_fwd);
  using namespace transformer_engine;
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnFwdParams pointer must not be null.");
  NVTE_CHECK(params->struct_size >= sizeof(NVTEFusedAttnFwdParams),
             "NVTEFusedAttnFwdParams::struct_size is smaller than the library expects; "
             "did you forget NVTE_FUSED_ATTN_FWD_PARAMS_INIT?");

  // Bind struct fields to the local names that the original implementation
  // below was written against.
  const NVTE_QKV_Layout qkv_layout = params->qkv_layout;
  const NVTE_QKV_Format o_format = params->o_format;
  const NVTE_QKV_Format qkv_scale_inv_format = params->qkv_scale_inv_format;
  const NVTE_Bias_Type bias_type = params->bias_type;
  const NVTE_Mask_Type attn_mask_type = params->attn_mask_type;
  const NVTE_Softmax_Type softmax_type = params->softmax_type;
  const size_t max_seqlen_q = params->max_seqlen_q;
  const size_t max_seqlen_kv = params->max_seqlen_kv;
  const float attn_scale = params->attn_scale;
  const float dropout = params->dropout;
  const int64_t window_size_left = params->window_size_left;
  const int64_t window_size_right = params->window_size_right;
  const bool bottom_right_diagonal = params->bottom_right_diagonal;
  const bool is_training = params->is_training;
  const bool return_max_logit = params->return_max_logit;
  const bool cuda_graph = params->cuda_graph;
  NVTETensorPack *Aux_CTX_Tensors = params->Aux_CTX_Tensors;
  cudaStream_t stream = params->stream;

  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(params->cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(params->cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(params->cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(params->cu_seqlens_kv_padded);
  const Tensor *input_page_table_k = convertNVTETensorCheck(params->page_table_k);
  const Tensor *input_page_table_v = convertNVTETensorCheck(params->page_table_v);
  const Tensor *input_rng_state = convertNVTETensorCheck(params->rng_state);
  const Tensor *input_Q = convertNVTETensorCheck(params->Q);
  const Tensor *input_K = convertNVTETensorCheck(params->K);
  const Tensor *input_V = convertNVTETensorCheck(params->V);
  const Tensor *input_Bias = convertNVTETensorCheck(params->Bias);
  const Tensor *input_SoftmaxOffset = convertNVTETensorCheck(params->SoftmaxOffset);
  Tensor *input_output_S = convertNVTETensorCheck(params->S);
  Tensor *output_O = convertNVTETensorCheck(params->O);
  Tensor *wkspace = convertNVTETensor(params->workspace);

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

  size_t bias_batch_size = 0, bias_num_heads = 0, bias_seqlen_q = 0, bias_seqlen_kv = 0;
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) &&
      input_Bias->data.dptr != nullptr && input_Bias->data.shape.size() >= 4) {
    bias_batch_size = input_Bias->data.shape[0];
    bias_num_heads = input_Bias->data.shape[1];
    bias_seqlen_q = input_Bias->data.shape[2];
    bias_seqlen_kv = input_Bias->data.shape[3];
  }

  NVTEFusedAttnConfig backend_cfg = NVTE_FUSED_ATTN_CONFIG_INIT;
  backend_cfg.qkv_layout = qkv_layout;
  backend_cfg.o_format = o_format;
  backend_cfg.qkv_scale_inv_format = qkv_scale_inv_format;
  backend_cfg.bias_type = bias_type;
  backend_cfg.attn_mask_type = attn_mask_type;
  backend_cfg.softmax_type = softmax_type;
  backend_cfg.attn_scale = attn_scale;
  backend_cfg.dropout = dropout;
  backend_cfg.max_seqlen_q = max_seqlen_q;
  backend_cfg.max_seqlen_kv = max_seqlen_kv;
  backend_cfg.window_size_left = window_size_left;
  backend_cfg.window_size_right = window_size_right;
  backend_cfg.bottom_right_diagonal = bottom_right_diagonal;
  backend_cfg.cuda_graph = cuda_graph;
  backend_cfg.q_dtype = Q_type;
  backend_cfg.kv_dtype = KV_type;
  backend_cfg.o_dtype = O_type;
  backend_cfg.batch_size = b;
  backend_cfg.num_attn_heads = h_q;
  backend_cfg.num_gqa_groups = h_kv;
  backend_cfg.head_dim_qk = d_qk;
  backend_cfg.head_dim_v = d_v;
  backend_cfg.num_pages_k = num_pages_k;
  backend_cfg.num_pages_v = num_pages_v;
  backend_cfg.page_size_k = page_size_k;
  backend_cfg.page_size_v = page_size_v;
  backend_cfg.max_pages_per_seq_k = max_pages_per_seq_k;
  backend_cfg.max_pages_per_seq_v = max_pages_per_seq_v;
  backend_cfg.bias_batch_size = bias_batch_size;
  backend_cfg.bias_num_heads = bias_num_heads;
  backend_cfg.bias_seqlen_q = bias_seqlen_q;
  backend_cfg.bias_seqlen_kv = bias_seqlen_kv;
  backend_cfg.is_training = is_training;
  backend_cfg.return_max_logit = return_max_logit;
  backend_cfg.deterministic = false;
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend_v2(&backend_cfg);

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

// Deprecated: forwards to nvte_fused_attn_fwd_v2.
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
  NVTEFusedAttnFwdParams params = NVTE_FUSED_ATTN_FWD_PARAMS_INIT;
  params.Q = Q;
  params.K = K;
  params.V = V;
  params.Bias = Bias;
  params.SoftmaxOffset = SoftmaxOffset;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_kv = cu_seqlens_kv;
  params.cu_seqlens_q_padded = cu_seqlens_q_padded;
  params.cu_seqlens_kv_padded = cu_seqlens_kv_padded;
  params.page_table_k = page_table_k;
  params.page_table_v = page_table_v;
  params.rng_state = rng_state;
  params.S = S;
  params.O = O;
  params.Aux_CTX_Tensors = Aux_CTX_Tensors;
  params.max_seqlen_q = max_seqlen_q;
  params.max_seqlen_kv = max_seqlen_kv;
  params.qkv_layout = qkv_layout;
  params.o_format = o_format;
  params.qkv_scale_inv_format = qkv_scale_inv_format;
  params.bias_type = bias_type;
  params.attn_mask_type = attn_mask_type;
  params.softmax_type = softmax_type;
  params.attn_scale = attn_scale;
  params.dropout = dropout;
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.bottom_right_diagonal = bottom_right_diagonal;
  params.is_training = is_training;
  params.return_max_logit = return_max_logit;
  params.cuda_graph = cuda_graph;
  params.workspace = workspace;
  params.stream = stream;
  nvte_fused_attn_fwd_v2(&params);
}
// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd_v2(const NVTEFusedAttnBwdParams *params) {
  NVTE_API_CALL(nvte_flash_attn_bwd);
  using namespace transformer_engine;
  NVTE_CHECK(params != nullptr, "NVTEFusedAttnBwdParams pointer must not be null.");
  NVTE_CHECK(params->struct_size >= sizeof(NVTEFusedAttnBwdParams),
             "NVTEFusedAttnBwdParams::struct_size is smaller than the library expects; "
             "did you forget NVTE_FUSED_ATTN_BWD_PARAMS_INIT?");

  // Bind struct fields to the local names that the original implementation
  // below was written against.
  const NVTE_QKV_Layout qkv_layout = params->qkv_layout;
  const NVTE_QKV_Layout dqkv_layout = params->dqkv_layout;
  const NVTE_QKV_Format o_format = params->o_format;
  const NVTE_QKV_Format do_format = params->do_format;
  const NVTE_QKV_Format qkv_scale_inv_format = params->qkv_scale_inv_format;
  const NVTE_QKV_Format do_scale_inv_format = params->do_scale_inv_format;
  const NVTE_Bias_Type bias_type = params->bias_type;
  const NVTE_Mask_Type attn_mask_type = params->attn_mask_type;
  const NVTE_Softmax_Type softmax_type = params->softmax_type;
  const size_t max_seqlen_q = params->max_seqlen_q;
  const size_t max_seqlen_kv = params->max_seqlen_kv;
  const float attn_scale = params->attn_scale;
  const float dropout = params->dropout;
  const int64_t window_size_left = params->window_size_left;
  const int64_t window_size_right = params->window_size_right;
  const bool bottom_right_diagonal = params->bottom_right_diagonal;
  const bool deterministic = params->deterministic;
  const bool cuda_graph = params->cuda_graph;
  const NVTETensorPack *Aux_CTX_Tensors = params->Aux_CTX_Tensors;
  cudaStream_t stream = params->stream;

  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(params->cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(params->cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(params->cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(params->cu_seqlens_kv_padded);
  const Tensor *input_Q = convertNVTETensorCheck(params->Q);
  const Tensor *input_K = convertNVTETensorCheck(params->K);
  const Tensor *input_V = convertNVTETensorCheck(params->V);
  const Tensor *input_O = convertNVTETensorCheck(params->O);
  const Tensor *input_dO = convertNVTETensorCheck(params->dO);
  const Tensor *input_S = convertNVTETensorCheck(params->S);
  Tensor *input_output_dP = convertNVTETensorCheck(params->dP);
  Tensor *output_dQ = convertNVTETensorCheck(params->dQ);
  Tensor *output_dK = convertNVTETensorCheck(params->dK);
  Tensor *output_dV = convertNVTETensorCheck(params->dV);
  Tensor *output_dBias = convertNVTETensorCheck(params->dBias);
  Tensor *output_dSoftmaxOffset = convertNVTETensorCheck(params->dSoftmaxOffset);
  Tensor *wkspace = convertNVTETensor(params->workspace);

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
  const NVTEDType dO_type = static_cast<NVTEDType>(input_dO->data.dtype);
  const NVTEDType dQKV_type = static_cast<NVTEDType>(output_dQ->data.dtype);

  size_t bias_batch_size = 0, bias_num_heads = 0, bias_seqlen_q = 0, bias_seqlen_kv = 0;
  if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI) &&
      output_dBias->data.dptr != nullptr && output_dBias->data.shape.size() >= 4) {
    bias_batch_size = output_dBias->data.shape[0];
    bias_num_heads = output_dBias->data.shape[1];
    bias_seqlen_q = output_dBias->data.shape[2];
    bias_seqlen_kv = output_dBias->data.shape[3];
  }

  NVTEFusedAttnConfig backend_cfg = NVTE_FUSED_ATTN_CONFIG_INIT;
  backend_cfg.qkv_layout = qkv_layout;
  backend_cfg.o_format = o_format;
  backend_cfg.do_format = do_format;
  backend_cfg.dqkv_layout = dqkv_layout;
  backend_cfg.qkv_scale_inv_format = qkv_scale_inv_format;
  backend_cfg.do_scale_inv_format = do_scale_inv_format;
  backend_cfg.bias_type = bias_type;
  backend_cfg.attn_mask_type = attn_mask_type;
  backend_cfg.softmax_type = softmax_type;
  backend_cfg.attn_scale = attn_scale;
  backend_cfg.dropout = dropout;
  backend_cfg.max_seqlen_q = max_seqlen_q;
  backend_cfg.max_seqlen_kv = max_seqlen_kv;
  backend_cfg.window_size_left = window_size_left;
  backend_cfg.window_size_right = window_size_right;
  backend_cfg.bottom_right_diagonal = bottom_right_diagonal;
  backend_cfg.cuda_graph = cuda_graph;
  backend_cfg.q_dtype = Q_type;
  backend_cfg.kv_dtype = KV_type;
  backend_cfg.o_dtype = O_type;
  backend_cfg.do_dtype = dO_type;
  backend_cfg.dqkv_dtype = dQKV_type;
  backend_cfg.batch_size = b;
  backend_cfg.num_attn_heads = h_q;
  backend_cfg.num_gqa_groups = h_kv;
  backend_cfg.head_dim_qk = d_qk;
  backend_cfg.head_dim_v = d_v;
  backend_cfg.bias_batch_size = bias_batch_size;
  backend_cfg.bias_num_heads = bias_num_heads;
  backend_cfg.bias_seqlen_q = bias_seqlen_q;
  backend_cfg.bias_seqlen_kv = bias_seqlen_kv;
  backend_cfg.is_training = true;
  backend_cfg.return_max_logit = false;
  backend_cfg.deterministic = deterministic;
  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend_v2(&backend_cfg);

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

// Deprecated: forwards to nvte_fused_attn_bwd_v2.
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
  NVTEFusedAttnBwdParams params = NVTE_FUSED_ATTN_BWD_PARAMS_INIT;
  params.Q = Q;
  params.K = K;
  params.V = V;
  params.O = O;
  params.dO = dO;
  params.S = S;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_kv = cu_seqlens_kv;
  params.cu_seqlens_q_padded = cu_seqlens_q_padded;
  params.cu_seqlens_kv_padded = cu_seqlens_kv_padded;
  params.Aux_CTX_Tensors = Aux_CTX_Tensors;
  params.dP = dP;
  params.dQ = dQ;
  params.dK = dK;
  params.dV = dV;
  params.dBias = dBias;
  params.dSoftmaxOffset = dSoftmaxOffset;
  params.max_seqlen_q = max_seqlen_q;
  params.max_seqlen_kv = max_seqlen_kv;
  params.qkv_layout = qkv_layout;
  params.dqkv_layout = dqkv_layout;
  params.o_format = o_format;
  params.do_format = do_format;
  params.qkv_scale_inv_format = qkv_scale_inv_format;
  params.do_scale_inv_format = do_scale_inv_format;
  params.bias_type = bias_type;
  params.attn_mask_type = attn_mask_type;
  params.softmax_type = softmax_type;
  params.attn_scale = attn_scale;
  params.dropout = dropout;
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;
  params.bottom_right_diagonal = bottom_right_diagonal;
  params.deterministic = deterministic;
  params.cuda_graph = cuda_graph;
  params.workspace = workspace;
  params.stream = stream;
  nvte_fused_attn_bwd_v2(&params);
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
