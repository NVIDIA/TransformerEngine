/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_f16_arbitrary_seqlen.h"
#include "fused_attn_f16_max512_seqlen.h"
#include "fused_attn_fp8.h"
#include "utils.h"

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
    default:
      NVTE_ERROR("qkv_layout not supported!");
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
    default:
      NVTE_ERROR("qkv_layout not supported!");
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
    default:
      NVTE_ERROR("qkv_layout not supported!");
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
    default:
      NVTE_ERROR("qkv_layout not supported!");
  }
}

// select a backend for fused attention
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, float dropout, size_t num_attn_heads,
    size_t num_gqa_groups, size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim_qk,
    size_t head_dim_v, int64_t window_size_left, int64_t window_size_right) {
  using namespace transformer_engine;
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
      // 8.9: t3hd, max_s=512, d=64, padding
      ((cudnn_runtime_version >= 8900 && sm_arch_ < 100 &&
        qkv_layout == NVTE_QKV_Layout::NVTE_T3HD && max_seqlen_q == max_seqlen_kv &&
        max_seqlen_q <= 512 && head_dim_qk == 64 && head_dim_v == 64 &&
        attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
       // 9.2: {bshd, sbhd}, any seqlen, d=128, {no_mask, causal}
       (cudnn_runtime_version >= 90201 && sm_arch_ < 100 && max_seqlen_q % 128 == 0 &&
        max_seqlen_kv % 128 == 0 && head_dim_qk == 128 && head_dim_v == 128 &&
        (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
         attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) ||
       // 9.7: {bshd, sbhd}, any seqlen, d<=256 for sm90 and d<=128 for sm100, {padding, padding_causal}
       (cudnn_runtime_version >= 90700 &&
        // TODO (cyang): add is_training to nvte_get_fused_attn_backend
        // sm90: fwd d<=256, bwd d=128 only
        // sm100: fwd d<=128, bwd d<=128
        ((sm_arch_ < 100 && head_dim_qk <= 256 && head_dim_v <= 256) ||
         (sm_arch_ >= 100 && head_dim_qk <= 128 && head_dim_v <= 128)) &&
        head_dim_qk % 16 == 0 && head_dim_v % 16 == 0 &&
        (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
         attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
         attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
         attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK))) &&
      (qkv_format == NVTE_QKV_Format::NVTE_BSHD || qkv_format == NVTE_QKV_Format::NVTE_SBHD) &&
      !requires_64bit_ragged_offset &&
      // 9.10.0: known bugs with SDPA FP8
      (cudnn_runtime_version != 91000)) {
    if (cudnn_runtime_version >= 8900) {
      backend = NVTE_Fused_Attn_Backend::NVTE_FP8;
    } else {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP8 fused attention is supported by cuDNN 8.9.0+."
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
  } else if ((q_dtype == NVTEDType::kNVTEFloat16) || (q_dtype == NVTEDType::kNVTEBFloat16)) {
    bool flag_m512 = false;
    bool flag_arb = false;
    if ((sm_arch_ == 80 || sm_arch_ == 90) && (max_seqlen_q <= 512 && max_seqlen_q % 64 == 0) &&
        (max_seqlen_kv <= 512 && max_seqlen_kv % 64 == 0) && (head_dim_qk == 64) &&
        (head_dim_v == 64) && (num_attn_heads == num_gqa_groups) &&
        ((bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) ||
         (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)) &&
        ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
         (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
         (attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
          max_seqlen_q == max_seqlen_kv) ||
         (attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) &&
        ((qkv_layout == NVTE_QKV_Layout::NVTE_SB3HD) ||
         (qkv_layout == NVTE_QKV_Layout::NVTE_SBHD_SB2HD) ||
         (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) ||
         (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) ||
         (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD)) &&
        ((window_size_left == -1) && (window_size_right == -1 || window_size_right == 0)) &&
        !requires_64bit_ragged_offset) {
      flag_m512 = true;
    }
    if (
        // TODO(cyang): replace with cudnn-frontend check_support for cleaner logic and better error messaging
        // architecture
        ((cudnn_runtime_version < 8903 && (sm_arch_ == 80 || sm_arch_ == 90)) ||
         (cudnn_runtime_version >= 8903 && sm_arch_ >= 80 && sm_arch_ < 100) ||
         (cudnn_runtime_version >= 90700 && sm_arch_ >= 80)) &&
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
         // 9.11/9.12 bug: 128 < d_qk <= 256, 128 < d_v <= 256 + Hopper + bprop + MLA
         (!((cudnn_runtime_version == 91100 || cudnn_runtime_version == 91200 ||
             cudnn_runtime_version == 91300) &&
            is_training && sm_arch_ == 90 && head_dim_qk >= 128 && head_dim_v >= 128 &&
            !(head_dim_qk == 192 && head_dim_v == 128) && head_dim_qk != head_dim_v))) &&
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
         (qkv_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90 &&
          ((cudnn_runtime_version >= 90100 && num_attn_heads == num_gqa_groups) ||
           cudnn_runtime_version >= 90600)) ||
         ((q_format == NVTE_QKV_Format::NVTE_SBHD || q_format == NVTE_QKV_Format::NVTE_BSHD ||
           (q_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90) ||
           kv_format == NVTE_QKV_Format::NVTE_SBHD || kv_format == NVTE_QKV_Format::NVTE_BSHD ||
           (kv_format == NVTE_QKV_Format::NVTE_THD && sm_arch_ >= 90)) &&
          cudnn_runtime_version >= 90700)) &&
        // sliding window
        // pre-9.2: full attn, causal
        ((cudnn_runtime_version < 90200 && window_size_left == -1 &&
          (window_size_right == -1 || window_size_right == 0)) ||
         // 9.2: SWA (left, 0) + top-left diagonal + {bshd, sbhd}
         (cudnn_runtime_version >= 90200 &&
          ((window_size_left == -1 && (window_size_right == -1 || window_size_right == 0)) ||
           ((window_size_left >= 0 || window_size_left == -1) && window_size_right == 0 &&
            (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
             (attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
              max_seqlen_q == max_seqlen_kv)) &&
            max_seqlen_q <= max_seqlen_kv && dropout == 0.0 &&
            bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
            (qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
             qkv_format == NVTE_QKV_Format::NVTE_SBHD)))) ||
         // 9.6: SWA (left, 0) + top-left/bottom-right diagonal + {bshd, sbhd, thd}
         (cudnn_runtime_version >= 90600 &&
          ((window_size_left == -1 && (window_size_right == -1 || window_size_right == 0)) ||
           ((window_size_left >= 0 || window_size_left == -1) && window_size_right == 0 &&
            ((attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
              // TODO(cyang): fix bug for BRCM + cross-attention on sm100
              (sm_arch_ < 100 || (sm_arch_ >= 100 && ((max_seqlen_q == max_seqlen_kv &&
                                                       cudnn_runtime_version <= 90700) ||
                                                      cudnn_runtime_version > 90700)))) ||
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
        (cudnn_runtime_version != 91000) && (cudnn_runtime_version != 91001)) {
      flag_arb = true;
    }
    if (((max_seqlen_q > 512) || (max_seqlen_kv > 512)) && (flag_arb == true)) {
      backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
    }
    if ((max_seqlen_q <= 512) && (max_seqlen_kv <= 512)) {
      if (flag_arb == true) {
        backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
      } else if ((flag_arb == false) && (flag_m512 == true)) {
        backend = NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen;
      }
      int env_backend = static_cast<int>(backend);
      env_backend = transformer_engine::getenv<int>("NVTE_FUSED_ATTN_BACKEND", env_backend);
      if (((env_backend == static_cast<int>(NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)) &&
           flag_m512) ||
          ((env_backend == static_cast<int>(NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)) &&
           flag_arb)) {
        backend = static_cast<NVTE_Fused_Attn_Backend>(env_backend);
      }
    }
    if (cudnn_runtime_version < 8901 &&
        backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP16/BF16 fused attention is supported by cuDNN 8.9.1+."
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
    if (cudnn_runtime_version < 8900 &&
        backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
      backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      std::cout << "Warning: FP16/BF16 fused attention is supported by cuDNN 8.9.0+."
                   " Please upgrade your cuDNN version if possible."
                << std::endl;
    }
  } else {
    backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
  }
  return backend;
}

// NVTE fused attention FWD with packed QKV
void nvte_fused_attn_fwd_qkvpacked(const NVTETensor QKV, const NVTETensor Bias, NVTETensor S,
                                   NVTETensor O, NVTETensorPack *Aux_CTX_Tensors,
                                   const NVTETensor cu_seqlens, const NVTETensor cu_seqlens_padded,
                                   const NVTETensor rng_state, size_t max_seqlen, bool is_training,
                                   float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                   int64_t window_size_left, int64_t window_size_right,
                                   NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = convertNVTETensorCheck(cu_seqlens);
  const Tensor *input_cu_seqlens_padded = convertNVTETensorCheck(cu_seqlens_padded);
  const Tensor *input_rng_state = convertNVTETensorCheck(rng_state);
  const Tensor *input_QKV = convertNVTETensorCheck(QKV);
  const Tensor *input_Bias = convertNVTETensorCheck(Bias);
  Tensor *input_output_S = convertNVTETensorCheck(S);
  Tensor *output_O = convertNVTETensorCheck(O);
  Tensor *wkspace = convertNVTETensor(workspace);

  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    h = input_QKV->data.shape[ndim - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    h = input_QKV->data.shape[ndim - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_qkvpacked only supports H3D and 3HD layouts!");
  }
  size_t d = input_QKV->data.shape[ndim - 1];
  size_t t = 0;
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    t = input_QKV->data.shape[0];
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      is_training, QKV_type, QKV_type, qkv_layout, bias_type, attn_mask_type, dropout, h, h,
      max_seqlen, max_seqlen, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
    fused_attn_max_512_fwd_qkvpacked(b, h, max_seqlen, d, is_training, attn_scale, dropout,
                                     qkv_layout, bias_type, attn_mask_type, input_QKV, input_Bias,
                                     output_O, Aux_CTX_Tensors, input_cu_seqlens, input_rng_state,
                                     wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_arbitrary_seqlen_fwd_qkvpacked(
        b, h, max_seqlen, d, t, is_training, attn_scale, dropout, qkv_layout, bias_type,
        attn_mask_type, window_size_left, window_size_right, input_QKV, input_Bias, output_O,
        Aux_CTX_Tensors, input_cu_seqlens, input_cu_seqlens_padded, input_rng_state, wkspace,
        stream, handle);
#else
    NVTE_ERROR(
        "cuDNN 8.9.0 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_fp8_fwd_qkvpacked(b, h, max_seqlen, d, is_training, attn_scale, dropout, qkv_layout,
                                 bias_type, attn_mask_type, input_QKV, input_output_S, output_O,
                                 Aux_CTX_Tensors, input_cu_seqlens, input_rng_state, wkspace,
                                 stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention BWD with packed QKV
void nvte_fused_attn_bwd_qkvpacked(const NVTETensor QKV, const NVTETensor O, const NVTETensor dO,
                                   const NVTETensor S, NVTETensor dP,
                                   const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQKV,
                                   NVTETensor dBias, const NVTETensor cu_seqlens,
                                   const NVTETensor cu_seqlens_padded, size_t max_seqlen,
                                   float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                   int64_t window_size_left, int64_t window_size_right,
                                   bool deterministic, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_qkvpacked);
  using namespace transformer_engine;

  const Tensor *input_cu_seqlens = convertNVTETensorCheck(cu_seqlens);
  const Tensor *input_cu_seqlens_padded = convertNVTETensorCheck(cu_seqlens_padded);
  const Tensor *input_QKV = convertNVTETensorCheck(QKV);
  const Tensor *input_O = convertNVTETensorCheck(O);
  const Tensor *input_dO = convertNVTETensorCheck(dO);
  const Tensor *input_S = convertNVTETensorCheck(S);
  Tensor *input_output_dP = convertNVTETensorCheck(dP);
  Tensor *output_dQKV = convertNVTETensorCheck(dQKV);
  Tensor *output_dBias = convertNVTETensorCheck(dBias);
  Tensor *wkspace = convertNVTETensor(workspace);

  auto ndim = input_QKV->data.shape.size();
  size_t b = input_cu_seqlens->data.shape[0] - 1;
  size_t h = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    h = input_QKV->data.shape[ndim - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    h = input_QKV->data.shape[ndim - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_qkvpacked only supports H3D and 3HD layouts!");
  }
  size_t d = input_QKV->data.shape[ndim - 1];
  size_t t = 0;
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    t = input_QKV->data.shape[0];
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType QKV_type = static_cast<NVTEDType>(input_QKV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      true, QKV_type, QKV_type, qkv_layout, bias_type, attn_mask_type, dropout, h, h, max_seqlen,
      max_seqlen, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    fused_attn_max_512_bwd_qkvpacked(
        b, h, max_seqlen, d, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type, input_QKV,
        input_dO, output_S, output_dQKV, output_dBias, input_cu_seqlens, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    Tensor *input_Bias, *input_rng_state;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      input_Bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    } else {
      input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    }
    fused_attn_arbitrary_seqlen_bwd_qkvpacked(
        b, h, max_seqlen, d, t, attn_scale, dropout, qkv_layout, bias_type, attn_mask_type,
        window_size_left, window_size_right, deterministic, input_QKV, input_O, input_dO,
        input_Bias, output_S, output_dQKV, output_dBias, input_cu_seqlens, input_cu_seqlens_padded,
        input_rng_state, wkspace, stream, handle);
#else
    const char *err_msg =
        "cuDNN 8.9.0 is required for BF16/FP16 fused attention "
        "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    const Tensor *input_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    fused_attn_fp8_bwd_qkvpacked(b, h, max_seqlen, d, attn_scale, dropout, qkv_layout, bias_type,
                                 attn_mask_type, input_QKV, input_O, input_dO, input_M, input_ZInv,
                                 input_S, input_output_dP, output_dQKV, input_cu_seqlens,
                                 input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention FWD with packed KV
void nvte_fused_attn_fwd_kvpacked(
    const NVTETensor Q, const NVTETensor KV, const NVTETensor Bias, NVTETensor S, NVTETensor O,
    NVTETensorPack *Aux_CTX_Tensors, const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
    const NVTETensor cu_seqlens_q_padded, const NVTETensor cu_seqlens_kv_padded,
    const NVTETensor page_table_k, const NVTETensor page_table_v, const NVTETensor rng_state,
    size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training, float attn_scale, float dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    int64_t window_size_left, int64_t window_size_right, NVTETensor workspace,
    cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(cu_seqlens_kv_padded);
  const Tensor *input_page_table_k = convertNVTETensorCheck(page_table_k);
  const Tensor *input_page_table_v = convertNVTETensorCheck(page_table_v);
  const Tensor *input_rng_state = convertNVTETensorCheck(rng_state);
  const Tensor *input_Q = convertNVTETensorCheck(Q);
  const Tensor *input_KV = convertNVTETensorCheck(KV);
  const Tensor *input_Bias = convertNVTETensorCheck(Bias);
  Tensor *input_output_S = convertNVTETensorCheck(S);
  Tensor *output_O = convertNVTETensorCheck(O);
  Tensor *wkspace = convertNVTETensor(workspace);

  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  auto ndim = input_Q->data.shape.size();
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];
  auto ndim_kv = input_KV->data.shape.size();
  size_t h_kv = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    h_kv = input_KV->data.shape[ndim_kv - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    h_kv = input_KV->data.shape[ndim_kv - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_kvpacked only supports HD_H2D and HD_2HD layouts!");
  }
  size_t t_q = 0;
  size_t t_kv = 0;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    t_q = input_Q->data.shape[0];
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    t_kv = input_KV->data.shape[0];
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
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) {
    NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
    if (kv_format == NVTE_QKV_Format::NVTE_BSHD) {
      num_pages_k = input_KV->data.shape[0];
      page_size_k = input_KV->data.shape[1];
      num_pages_v = num_pages_v;
      page_size_v = page_size_v;
    } else if (kv_format == NVTE_QKV_Format::NVTE_SBHD) {
      num_pages_k = input_KV->data.shape[1];
      page_size_k = input_KV->data.shape[0];
      num_pages_v = num_pages_v;
      page_size_v = page_size_v;
    }
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      is_training, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv,
      max_seqlen_q, max_seqlen_kv, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
    fused_attn_max_512_fwd_kvpacked(
        b, h_q, max_seqlen_q, max_seqlen_kv, d, is_training, attn_scale, dropout, qkv_layout,
        bias_type, attn_mask_type, input_Q, input_KV, input_Bias, output_O, Aux_CTX_Tensors,
        input_cu_seqlens_q, input_cu_seqlens_kv, input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8903)
    fused_attn_arbitrary_seqlen_fwd_kvpacked(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, t_q, t_kv, num_pages_k, num_pages_v,
        page_size_k, page_size_v, max_pages_per_seq_k, max_pages_per_seq_v, is_training, attn_scale,
        dropout, qkv_layout, bias_type, attn_mask_type, window_size_left, window_size_right,
        input_Q, input_KV, input_Bias, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
        input_cu_seqlens_kv, input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded,
        input_page_table_k, input_page_table_v, input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR(
        "cuDNN 8.9.3 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_fp8_fwd_kvpacked(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, is_training, attn_scale, dropout, qkv_layout,
        bias_type, attn_mask_type, input_Q, input_KV, input_output_S, output_O, Aux_CTX_Tensors,
        input_cu_seqlens_q, input_cu_seqlens_kv, input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention BWD with packed KV
void nvte_fused_attn_bwd_kvpacked(
    const NVTETensor Q, const NVTETensor KV, const NVTETensor O, const NVTETensor dO,
    const NVTETensor S, NVTETensor dP, const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ,
    NVTETensor dKV, NVTETensor dBias, const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
    const NVTETensor cu_seqlens_q_padded, const NVTETensor cu_seqlens_kv_padded,
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    int64_t window_size_left, int64_t window_size_right, bool deterministic, NVTETensor workspace,
    cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = convertNVTETensorCheck(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = convertNVTETensorCheck(cu_seqlens_kv);
  const Tensor *input_cu_seqlens_q_padded = convertNVTETensorCheck(cu_seqlens_q_padded);
  const Tensor *input_cu_seqlens_kv_padded = convertNVTETensorCheck(cu_seqlens_kv_padded);
  const Tensor *input_Q = convertNVTETensorCheck(Q);
  const Tensor *input_KV = convertNVTETensorCheck(KV);
  const Tensor *input_O = convertNVTETensorCheck(O);
  const Tensor *input_dO = convertNVTETensorCheck(dO);
  const Tensor *input_S = convertNVTETensorCheck(S);
  Tensor *input_output_dP = convertNVTETensorCheck(dP);
  Tensor *output_dQ = convertNVTETensorCheck(dQ);
  Tensor *output_dKV = convertNVTETensorCheck(dKV);
  Tensor *output_dBias = convertNVTETensorCheck(dBias);
  Tensor *wkspace = convertNVTETensor(workspace);

  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  auto ndim = input_Q->data.shape.size();
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t d = input_Q->data.shape[ndim - 1];
  auto ndim_kv = input_KV->data.shape.size();
  size_t h_kv = 0;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    h_kv = input_KV->data.shape[ndim_kv - 2];
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    h_kv = input_KV->data.shape[ndim_kv - 3];
  } else {
    NVTE_ERROR("nvte_fused_attn_fwd_kvpacked only supports HD_H2D and HD_2HD layouts!");
  }
  size_t t_q = 0;
  size_t t_kv = 0;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    t_q = input_Q->data.shape[0];
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    t_kv = input_KV->data.shape[0];
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_KV->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      true, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv,
      max_seqlen_q, max_seqlen_kv, d, d, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    fused_attn_max_512_bwd_kvpacked(
        b, h_q, max_seqlen_q, max_seqlen_kv, d, attn_scale, dropout, qkv_layout, bias_type,
        attn_mask_type, input_Q, input_KV, input_dO, output_S, output_dQ, output_dKV, output_dBias,
        input_cu_seqlens_q, input_cu_seqlens_kv, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8903)
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    Tensor *input_Bias, *input_rng_state;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      input_Bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    } else {
      input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    }
    fused_attn_arbitrary_seqlen_bwd_kvpacked(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, t_q, t_kv, attn_scale, dropout, qkv_layout,
        bias_type, attn_mask_type, window_size_left, window_size_right, deterministic, input_Q,
        input_KV, input_O, input_dO, input_Bias, output_S, output_dQ, output_dKV, output_dBias,
        input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded,
        input_cu_seqlens_kv_padded, input_rng_state, wkspace, stream, handle);
#else
    const char *err_msg =
        "cuDNN 8.9.3 is required for BF16/FP16 fused attention "
        "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    const Tensor *input_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    fused_attn_fp8_bwd_kvpacked(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d, attn_scale, dropout,
                                qkv_layout, bias_type, attn_mask_type, input_Q, input_KV, input_O,
                                input_dO, input_M, input_ZInv, input_S, input_output_dP, output_dQ,
                                output_dKV, input_cu_seqlens_q, input_cu_seqlens_kv,
                                input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention FWD with separate Q, K and V
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, NVTETensor S, NVTETensor O,
                         NVTETensorPack *Aux_CTX_Tensors, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, const NVTETensor page_table_k,
                         const NVTETensor page_table_v, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         int64_t window_size_left, int64_t window_size_right, NVTETensor workspace,
                         cudaStream_t stream) {
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
  Tensor *input_output_S = convertNVTETensorCheck(S);
  Tensor *output_O = convertNVTETensorCheck(O);
  Tensor *wkspace = convertNVTETensor(workspace);

  auto ndim = input_Q->data.shape.size();
  auto ndim_kv = input_K->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t h_kv = input_K->data.shape[ndim_kv - 2];
  size_t d_qk = input_Q->data.shape[ndim - 1];
  size_t d_v = input_V->data.shape[ndim_kv - 1];
  size_t t_q = 0;
  size_t t_kv = 0;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    t_q = input_Q->data.shape[0];
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    t_kv = input_K->data.shape[0];
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

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      is_training, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv,
      max_seqlen_q, max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
    fused_attn_max_512_fwd(b, h_q, max_seqlen_q, max_seqlen_kv, d_qk, is_training, attn_scale,
                           dropout, qkv_layout, bias_type, attn_mask_type, input_Q, input_K,
                           input_V, input_Bias, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
                           input_cu_seqlens_kv, input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_arbitrary_seqlen_fwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, num_pages_k, num_pages_v,
        page_size_k, page_size_v, max_pages_per_seq_k, max_pages_per_seq_v, is_training, attn_scale,
        dropout, qkv_layout, bias_type, attn_mask_type, window_size_left, window_size_right,
        input_Q, input_K, input_V, input_Bias, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
        input_cu_seqlens_kv, input_cu_seqlens_q_padded, input_cu_seqlens_kv_padded,
        input_page_table_k, input_page_table_v, input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR(
        "cuDNN 8.9.0 is required for BF16/FP16 fused attention with arbitrary sequence length. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    fused_attn_fp8_fwd(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, is_training, attn_scale,
                       dropout, qkv_layout, bias_type, attn_mask_type, input_Q, input_K, input_V,
                       input_output_S, output_O, Aux_CTX_Tensors, input_cu_seqlens_q,
                       input_cu_seqlens_kv, input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length for fused attention. \n");
  }
}
// NVTE fused attention BWD with separate Q, K and V
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, int64_t window_size_left,
                         int64_t window_size_right, bool deterministic, NVTETensor workspace,
                         cudaStream_t stream) {
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
  Tensor *wkspace = convertNVTETensor(workspace);

  auto ndim = input_Q->data.shape.size();
  auto ndim_kv = input_K->data.shape.size();
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h_q = input_Q->data.shape[ndim - 2];
  size_t h_kv = input_K->data.shape[ndim_kv - 2];
  size_t d_qk = input_Q->data.shape[ndim - 1];
  size_t d_v = input_V->data.shape[ndim_kv - 1];
  size_t t_q = 0;
  size_t t_kv = 0;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    t_q = input_Q->data.shape[0];
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    t_kv = input_K->data.shape[0];
  }

  auto handle = cudnnExecutionPlanManager::Instance().GetHandle();
  const NVTEDType Q_type = static_cast<NVTEDType>(input_Q->data.dtype);
  const NVTEDType KV_type = static_cast<NVTEDType>(input_K->data.dtype);

  NVTE_Fused_Attn_Backend fused_attention_backend = nvte_get_fused_attn_backend(
      true, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, dropout, h_q, h_kv,
      max_seqlen_q, max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right);

  if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
#if (CUDNN_VERSION >= 8901)
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    fused_attn_max_512_bwd(b, h_q, max_seqlen_q, max_seqlen_kv, d_qk, attn_scale, dropout,
                           qkv_layout, bias_type, attn_mask_type, input_Q, input_K, input_V,
                           input_dO, output_S, output_dQ, output_dK, output_dV, output_dBias,
                           input_cu_seqlens_q, input_cu_seqlens_kv, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.1 is required for BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
#if (CUDNN_VERSION >= 8900)
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    Tensor *input_Bias, *input_rng_state;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      input_Bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    } else {
      input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    }
    fused_attn_arbitrary_seqlen_bwd(
        b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, t_q, t_kv, attn_scale, dropout,
        qkv_layout, bias_type, attn_mask_type, window_size_left, window_size_right, deterministic,
        input_Q, input_K, input_V, input_O, input_dO, input_Bias, output_S, output_dQ, output_dK,
        output_dV, output_dBias, input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded,
        input_cu_seqlens_kv_padded, input_rng_state, wkspace, stream, handle);
#else
    const char *err_msg =
        "cuDNN 8.9.0 is required for BF16/FP16 fused attention "
        "with arbitrary sequence length. \n";
    NVTE_ERROR(err_msg);
#endif
  } else if (fused_attention_backend == NVTE_Fused_Attn_Backend::NVTE_FP8) {
#if (CUDNN_VERSION >= 8900)
    const Tensor *input_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    fused_attn_fp8_bwd(b, h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, attn_scale, dropout,
                       qkv_layout, bias_type, attn_mask_type, input_Q, input_K, input_V, input_O,
                       input_dO, input_M, input_ZInv, input_S, input_output_dP, output_dQ,
                       output_dK, output_dV, input_cu_seqlens_q, input_cu_seqlens_kv,
                       input_rng_state, wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9.0 is required for FP8 fused attention. \n");
#endif
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
