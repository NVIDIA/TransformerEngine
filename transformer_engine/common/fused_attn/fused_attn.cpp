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

namespace {
  struct BackendSelectionContext {
    bool is_training;
    NVTEDType q_dtype;
    NVTE_QKV_Layout qkv_layout;
    NVTE_Bias_Type bias_type;
    NVTE_Mask_Type attn_mask_type;
    NVTE_Softmax_Type softmax_type;
    float dropout;
    size_t num_attn_heads;
    size_t num_gqa_groups;
    size_t max_seqlen_q;
    size_t max_seqlen_kv;
    size_t head_dim_qk;
    size_t head_dim_v;
    int64_t window_size_left;
    int64_t window_size_right;
    
    int sm_arch;
    int cudnn_version;
    NVTE_QKV_Format qkv_format;
    NVTE_QKV_Format q_format;
    NVTE_QKV_Format kv_format;
    NVTE_QKV_Layout_Group layout_group;
    bool requires_64bit_ragged_offset;
    bool supported_ragged_offset_size;
    
    std::string error_msg;
    
    void set_error(const std::string& msg) {
      error_msg = msg;
    }
  };
  
  bool checks_for_fp8(BackendSelectionContext& ctx) {
    // Check dtype
    if (ctx.q_dtype != NVTEDType::kNVTEFloat8E4M3 && 
        ctx.q_dtype != NVTEDType::kNVTEFloat8E5M2) {
      ctx.set_error("FP8 backend requires FP8E4M3 or FP8E5M2 dtype");
      return false;
    }
    
    // Check architecture
    if (ctx.sm_arch < 90) {
      ctx.set_error("FP8 backend requires SM90 (Hopper) or newer, got SM" + 
                    std::to_string(ctx.sm_arch));
      return false;
    }
    
    // Check bias
    if (ctx.bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) {
      ctx.set_error("FP8 backend requires NVTE_NO_BIAS");
      return false;
    }
    
    bool version_has_support = false;
    // cuDNN 8.9: t3hd, max_s=512, d=64, padding
    if (ctx.cudnn_version >= 8900 && ctx.sm_arch < 100 &&
        ctx.qkv_layout == NVTE_QKV_Layout::NVTE_T3HD && 
        ctx.max_seqlen_q == ctx.max_seqlen_kv &&
        ctx.max_seqlen_q <= 512 && 
        ctx.head_dim_qk == 64 && ctx.head_dim_v == 64 &&
        ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) {
      version_has_support = true;
    }
    // cuDNN 9.2: {bshd, sbhd}, any seqlen, d=128, {no_mask, causal}
    if (ctx.cudnn_version >= 90201 && ctx.sm_arch < 100 && 
        ctx.max_seqlen_q % 128 == 0 && ctx.max_seqlen_kv % 128 == 0 && 
        ctx.head_dim_qk == 128 && ctx.head_dim_v == 128 &&
        (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
         ctx.attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) {
      version_has_support = true;
    }
    // cuDNN 9.7: {bshd, sbhd}, any seqlen, d<=256 for sm90 and d<=128 for sm100, {padding, padding_causal}
    if (ctx.cudnn_version >= 90700) {
      bool head_dim_ok = false;
      if (ctx.sm_arch < 100 && !ctx.is_training && 
          ctx.head_dim_qk <= 256 && ctx.head_dim_v <= 256) {
        head_dim_ok = true;
      } else if (ctx.sm_arch < 100 && ctx.is_training && 
                 ctx.head_dim_qk == 128 && ctx.head_dim_v == 128) {
        head_dim_ok = true;
      } else if (ctx.sm_arch >= 100 && 
                 ctx.head_dim_qk <= 128 && ctx.head_dim_v <= 128) {
        head_dim_ok = true;
      }
      if (head_dim_ok && 
          ctx.head_dim_qk % 16 == 0 && ctx.head_dim_v % 16 == 0 &&
          (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
           ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
           ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
           ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)) {
        version_has_support = true;
      }
    }
    if (!version_has_support) {
      ctx.set_error("FP8 backend: cuDNN version" + std::to_string(ctx.cudnn_version) + " does not support provided head_dim, seqlen, or mask");
      return false;
    }
    
    // Check common constraints
    if (ctx.qkv_format != NVTE_QKV_Format::NVTE_BSHD && 
        ctx.qkv_format != NVTE_QKV_Format::NVTE_SBHD) {
      ctx.set_error("FP8 backend requires BSHD or SBHD format");
      return false;
    }
    if (ctx.requires_64bit_ragged_offset) {
      ctx.set_error("FP8 backend does not support 64-bit ragged offsets");
      return false;
    }
    if (ctx.softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX) {
      ctx.set_error("FP8 backend requires vanilla softmax");
      return false;
    }
    if (ctx.cudnn_version == 91000) {
      ctx.set_error("FP8 backend has known bugs in cuDNN 9.10.0");
      return false;
    }
    
    return true;
  }
  
  bool checks_for_max512(BackendSelectionContext& ctx) {
    // Check dtype
    if (ctx.q_dtype != NVTEDType::kNVTEFloat16 && 
        ctx.q_dtype != NVTEDType::kNVTEBFloat16) {
      ctx.set_error("Max512 backend requires FP16 or BF16 dtype");
      return false;
    }
    
    // Check architecture
    if (ctx.sm_arch != 80 && ctx.sm_arch != 90) {
      ctx.set_error("Max512 backend requires sm80 or sm90, got sm" + 
                    std::to_string(ctx.sm_arch));
      return false;
    }
    
    // Check sequence length
    if (ctx.max_seqlen_q > 512 || ctx.max_seqlen_kv > 512) {
      ctx.set_error("Max512 backend requires seqlen <= 512, got q=" + 
                    std::to_string(ctx.max_seqlen_q) + ", kv=" + 
                    std::to_string(ctx.max_seqlen_kv));
      return false;
    }
    if (ctx.max_seqlen_q % 64 != 0 || ctx.max_seqlen_kv % 64 != 0) {
      ctx.set_error("Max512 backend requires seqlen % 64 == 0");
      return false;
    }

    // Check head dimension
    if (ctx.head_dim_qk != 64 || ctx.head_dim_v != 64) {
      ctx.set_error("Max512 backend requires head_dim=64");
      return false;
    }
    
    // Check GQA
    if (ctx.num_attn_heads != ctx.num_gqa_groups) {
      ctx.set_error("Max512 backend does not support GQA");
      return false;
    }
    
    // Check bias type
    if (ctx.bias_type != NVTE_Bias_Type::NVTE_NO_BIAS &&
        ctx.bias_type != NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
      ctx.set_error("Max512 backend requires NO_BIAS or POST_SCALE_BIAS");
      return false;
    }
    
    // Check mask type
    bool mask_ok = false;
    if (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
        ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
        ctx.attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK) {
      mask_ok = true;
    } else if (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
               ctx.max_seqlen_q == ctx.max_seqlen_kv) {
      mask_ok = true;
    }
    if (!mask_ok) {
      ctx.set_error("Max512 backend: unsupported mask type");
      return false;
    }
    
    // Check layout
    if (ctx.qkv_layout != NVTE_QKV_Layout::NVTE_SB3HD &&
        ctx.qkv_layout != NVTE_QKV_Layout::NVTE_SBHD_SB2HD &&
        ctx.qkv_layout != NVTE_QKV_Layout::NVTE_BS3HD &&
        ctx.qkv_layout != NVTE_QKV_Layout::NVTE_BSHD_BS2HD &&
        ctx.qkv_layout != NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
      ctx.set_error("Max512 backend: unsupported QKV layout");
      return false;
    }
    
    // Check window size
    if (ctx.window_size_left != -1 || 
        (ctx.window_size_right != -1 && ctx.window_size_right != 0)) {
      ctx.set_error("Max512 backend requires does not support sliding window");
      return false;
    }
    
    // Check ragged offset
    if (!ctx.supported_ragged_offset_size) {
      ctx.set_error("Max512 backend does not support 64-bit ragged offsets");
      return false;
    }
    
    // Check softmax type
    if (ctx.softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX) {
      ctx.set_error("Max512 backend requires vanilla softmax type");
      return false;
    }
    
    return true;
  }
  
  bool checks_for_arbitrary_seqlen(BackendSelectionContext& ctx) {
    // Check dtype
    if (ctx.q_dtype != NVTEDType::kNVTEFloat16 && 
        ctx.q_dtype != NVTEDType::kNVTEBFloat16) {
      ctx.set_error("ArbitrarySeqlen backend requires FP16 or BF16 dtype");
      return false;
    }
    
    // Check architecture
    bool arch_ok = false;
    if (ctx.cudnn_version < 8903 && (ctx.sm_arch == 80 || ctx.sm_arch == 90)) {
      arch_ok = true;
    } else if (ctx.cudnn_version >= 8903 && ctx.sm_arch >= 80 && ctx.sm_arch < 100) {
      arch_ok = true;
    } else if (ctx.cudnn_version >= 90700 && ctx.sm_arch >= 80) {
      arch_ok = true;
    }
    if (!arch_ok) {
      ctx.set_error("ArbitrarySeqlen backend: unsupported sm" + std::to_string(ctx.sm_arch) + 
                    " with cuDNN " + std::to_string(ctx.cudnn_version));
      return false;
    }
    
    // Check sequence length
    if (ctx.cudnn_version < 90000) {
      if (ctx.max_seqlen_q % 64 != 0 || ctx.max_seqlen_kv % 64 != 0) {
        ctx.set_error("ArbitrarySeqlen backend (cuDNN < 9.0) requires seqlen % 64 == 0");
        return false;
      }
    }
    
    // Check GQA
    if (ctx.cudnn_version < 8907) {
      if (ctx.num_attn_heads != ctx.num_gqa_groups) {
        ctx.set_error("ArbitrarySeqlen backend (cuDNN < 8.9.7) does not support GQA");
        return false;
      }
    }
    
    // Check head dimension
    if (ctx.head_dim_qk % 8 != 0 || ctx.head_dim_v % 8 != 0) {
      ctx.set_error("ArbitrarySeqlen backend requires head_dim % 8 == 0");
      return false;
    }
    bool head_dim_ok = false;
    // <= 128
    if (ctx.head_dim_qk <= 128 && ctx.head_dim_v <= 128) {
      head_dim_ok = true;
    }
    // 9.1: <= 256 + Hopper + fprop
    else if (ctx.head_dim_qk <= 256 && ctx.head_dim_v <= 256 &&
             !ctx.is_training && ctx.sm_arch == 90 && ctx.cudnn_version >= 90100) {
      head_dim_ok = true;
    }
    // 9.5: <= 256 + Hopper + bprop
    else if (ctx.head_dim_qk <= 256 && ctx.head_dim_v <= 256 &&
             ctx.is_training && ctx.sm_arch == 90 && ctx.cudnn_version >= 90500) {
      head_dim_ok = true;
    }
    // 9.9: any head_dim + Blackwell + fprop + non_paged + sq > 1
    else if (!ctx.is_training && ctx.sm_arch >= 100 && ctx.cudnn_version >= 90900 && 
             ctx.max_seqlen_q > 1 &&
             ctx.layout_group != NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD) {
      head_dim_ok = true;
    }
    // 9.10.2: any head_dim + any arch + fprop + paged
    // 9.10.2: any head_dim + any arch + fprop + non_paged + sq > 1
    // 9.10.2: any head_dim + any arch + fprop + non_paged + sq = 1 + {no_mask, padding, BRCM, padding_BRCM}
    else if (!ctx.is_training && ctx.cudnn_version >= 91002 &&
             (ctx.layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD || 
              ctx.max_seqlen_q > 1 ||
              (ctx.max_seqlen_q == 1 && 
               ctx.attn_mask_type != NVTE_Mask_Type::NVTE_CAUSAL_MASK &&
               ctx.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK))) {
      head_dim_ok = true;
    }
    // 9.11: d_qk=192, d_v=128 + Blackwell + bprop + non-paged
    else if (ctx.head_dim_qk == 192 && ctx.head_dim_v == 128 && 
             ctx.is_training && ctx.sm_arch >= 100 && ctx.cudnn_version >= 91100) {
      head_dim_ok = true;
    }    
    // 9.11+ bug: 128 < d_qk <= 256, 128 < d_v <= 256 + Hopper + bprop + MLA
    if (ctx.cudnn_version >= 91100 && ctx.is_training && ctx.sm_arch == 90 &&
        ctx.head_dim_qk >= 128 && ctx.head_dim_v >= 128 && 
        !(ctx.head_dim_qk == 192 && ctx.head_dim_v == 128) &&
        ctx.head_dim_qk != ctx.head_dim_v) {
      ctx.set_error("ArbitrarySeqlen backend: known cuDNN 9.11+ bug for sm90 bprop with MLA");
      return false;
    }
    if (!head_dim_ok) {
      ctx.set_error("ArbitrarySeqlen backend: unsupported head_dim (qk=" + 
                    std::to_string(ctx.head_dim_qk) + ", v=" + std::to_string(ctx.head_dim_v) + ")");
      return false;
    }
    
    // Check bias type
    bool bias_ok = false;
    if (ctx.cudnn_version < 8906 && 
        ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) {
      bias_ok = true;
    }
    else if (ctx.cudnn_version >= 8906) {
      if (ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS) {
        bias_ok = true;
      } else if (ctx.bias_type == NVTE_Bias_Type::NVTE_ALIBI &&
                 ctx.attn_mask_type != NVTE_Mask_Type::NVTE_NO_MASK &&
                 ctx.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_MASK &&
                 ctx.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK &&
                 ctx.attn_mask_type != NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
                 ctx.sm_arch >= 90) {
        bias_ok = true;
      } else if (ctx.bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && 
                 ctx.sm_arch >= 90) {
        bias_ok = true;
      }
    }
    if (ctx.cudnn_version >= 90000 &&
        ctx.bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS && 
        ctx.sm_arch >= 80) {
      bias_ok = true;
    }    
    if (!bias_ok) {
      ctx.set_error("ArbitrarySeqlen backend: unsupported bias type");
      return false;
    }
    
    // Check mask type
    bool mask_ok = false;
    // Pre-8.9.6: causal
    if (ctx.cudnn_version < 8906 && 
        ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) {
      mask_ok = true;
    }
    // 8.9.6: {bshd, sbhd} + {no_mask, causal, padding, padding_causal}
    else if (ctx.cudnn_version >= 8906 &&
             (ctx.qkv_format == NVTE_QKV_Format::NVTE_SBHD || 
              ctx.qkv_format == NVTE_QKV_Format::NVTE_BSHD) &&
             (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
              ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
              ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
              ctx.attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK)) {
      mask_ok = true;
    }
    // 9.1: adds thd + {padding, padding_causal}
    else if (ctx.cudnn_version >= 90100 && 
             ctx.qkv_format == NVTE_QKV_Format::NVTE_THD &&
             (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
              ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)) {
      mask_ok = true;
    }
    // 9.3: adds {bshd, sbhd} + causal_bottom_right + self/cross-attn (sq <= skv)
    else if (ctx.cudnn_version >= 90300 &&
             (ctx.qkv_format == NVTE_QKV_Format::NVTE_SBHD || 
              ctx.qkv_format == NVTE_QKV_Format::NVTE_BSHD) &&
             ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
             ctx.max_seqlen_q % 64 == 0 && ctx.max_seqlen_kv % 64 == 0 && 
             ctx.max_seqlen_q <= ctx.max_seqlen_kv &&
             ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && ctx.dropout == 0.0) {
      mask_ok = true;
    }
    // 9.5: adds {paged_kv_bshd, paged_kv_sbhd} + {padding, padding_causal, padding_causal_bottom_right}
    else if (ctx.cudnn_version >= 90500 &&
             ctx.layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD &&
             (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
              ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
              (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
               ctx.max_seqlen_q % 64 == 0 && ctx.max_seqlen_kv % 64 == 0 && 
               ctx.max_seqlen_q <= ctx.max_seqlen_kv)) &&
             ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && ctx.dropout == 0.0) {
      mask_ok = true;
    }
    // 9.6: adds {bshd, sbhd, thd} + padding_causal_bottom_right + self/cross-attn (sq <= skv)
    else if (ctx.cudnn_version >= 90600 &&
             ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
             ctx.max_seqlen_q % 64 == 0 && ctx.max_seqlen_kv % 64 == 0 && 
             ctx.max_seqlen_q <= ctx.max_seqlen_kv &&
             ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && ctx.dropout == 0.0) {
      mask_ok = true;
    }
    // 9.7: removes s_q/s_kv % 64 = 0 for {causal_bottom_right, padding_causal_bottom_right}
    // for any q_format/kv_format, and paged/non-paged
    else if (ctx.cudnn_version >= 90700) {
      if (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_NO_MASK ||
          ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) {
        mask_ok = true;
      } else if ((ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
                  ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK ||
                  ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
                 ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS && ctx.dropout == 0.0) {
        mask_ok = true;
      } else if ((ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK ||
                  ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK) &&
                 ctx.max_seqlen_q <= ctx.max_seqlen_kv) {
        mask_ok = true;
      }
    }
    if (!mask_ok) {
      ctx.set_error("ArbitrarySeqlen backend: unsupported mask type for cuDNN" + std::to_string(ctx.cudnn_version));
      return false;
    }
    
    // Check bias + mask combination
    if (ctx.cudnn_version >= 8906 &&
        (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK ||
         ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) &&
        ctx.bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS) {
      ctx.set_error("ArbitrarySeqlen backend: POST_SCALE_BIAS not supported with PADDING masks");
      return false;
    }
    
    // Check QKV format
    bool format_ok = false;
    if (ctx.qkv_format == NVTE_QKV_Format::NVTE_SBHD || 
        ctx.qkv_format == NVTE_QKV_Format::NVTE_BSHD) {
      format_ok = true;
    } else if (ctx.qkv_format == NVTE_QKV_Format::NVTE_THD && ctx.sm_arch >= 90 &&
               ((ctx.cudnn_version >= 90100 && ctx.num_attn_heads == ctx.num_gqa_groups) ||
                ctx.cudnn_version >= 90600)) {
      format_ok = true;
    } else if (ctx.cudnn_version >= 90700 &&
               ((ctx.q_format == NVTE_QKV_Format::NVTE_SBHD || 
                 ctx.q_format == NVTE_QKV_Format::NVTE_BSHD ||
                 (ctx.q_format == NVTE_QKV_Format::NVTE_THD && ctx.sm_arch >= 90)) ||
                (ctx.kv_format == NVTE_QKV_Format::NVTE_SBHD || 
                 ctx.kv_format == NVTE_QKV_Format::NVTE_BSHD ||
                 (ctx.kv_format == NVTE_QKV_Format::NVTE_THD && ctx.sm_arch >= 90)))) {
      format_ok = true;
    }
    if (!format_ok) {
      ctx.set_error("ArbitrarySeqlen backend: unsupported QKV format");
      return false;
    }
    
    // Check sliding window
    bool window_ok = false;
    // Pre-9.2: full attention, causal
    if (ctx.cudnn_version < 90200 && 
        ctx.window_size_left == -1 &&
        (ctx.window_size_right == -1 || ctx.window_size_right == 0)) {
      window_ok = true;
    }
    // 9.2: SWA (left, 0) + top-left diagonal + {bshd, sbhd}
    else if (ctx.cudnn_version >= 90200) {
      if (ctx.window_size_left == -1 && 
          (ctx.window_size_right == -1 || ctx.window_size_right == 0)) {
        window_ok = true;
      } else if ((ctx.window_size_left >= 0 || ctx.window_size_left == -1) && 
                 ctx.window_size_right == 0 &&
                 (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK ||
                  (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
                   ctx.max_seqlen_q == ctx.max_seqlen_kv)) &&
                 ctx.max_seqlen_q <= ctx.max_seqlen_kv && 
                 ctx.dropout == 0.0 &&
                 ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
                 (ctx.qkv_format == NVTE_QKV_Format::NVTE_BSHD ||
                  ctx.qkv_format == NVTE_QKV_Format::NVTE_SBHD)) {
        window_ok = true;
      }
    }
    // 9.6: SWA (left, 0) + top-left/bottom-right diagonal + {bshd, sbhd, thd}
    if (ctx.cudnn_version >= 90600) {
      if (ctx.window_size_left == -1 && 
          (ctx.window_size_right == -1 || ctx.window_size_right == 0)) {
        window_ok = true;
      } else if ((ctx.window_size_left >= 0 || ctx.window_size_left == -1) && 
                 ctx.window_size_right == 0) {
        bool mask_and_arch_ok = false;
        
        if (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK &&
            (ctx.sm_arch < 100 || 
             (ctx.sm_arch >= 100 && 
              ((ctx.max_seqlen_q == ctx.max_seqlen_kv && ctx.cudnn_version <= 90700) ||
               ctx.cudnn_version > 90700)))) {
          mask_and_arch_ok = true;
        } else if (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) {
          mask_and_arch_ok = true;
        } else if (ctx.attn_mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK &&
                   (ctx.sm_arch < 100 || 
                    (ctx.sm_arch >= 100 && 
                     ((ctx.max_seqlen_q == ctx.max_seqlen_kv && ctx.cudnn_version <= 90700) ||
                      ctx.cudnn_version > 90700)))) {
          mask_and_arch_ok = true;
        }
        if (mask_and_arch_ok && 
            ctx.max_seqlen_q <= ctx.max_seqlen_kv && 
            ctx.bias_type == NVTE_Bias_Type::NVTE_NO_BIAS &&
            ctx.dropout == 0.0) {
          window_ok = true;
        }
      }
    }
    if (!window_ok) {
      ctx.set_error("ArbitrarySeqlen backend: unsupported sliding window configuration");
      return false;
    }
    
    // Check ragged offset
    if (!ctx.supported_ragged_offset_size) {
      ctx.set_error("ArbitrarySeqlen backend does not support 64-bit ragged offset");
      return false;
    }
    
    // Check known bugs
    if (ctx.cudnn_version == 91000 || ctx.cudnn_version == 91001) {
      ctx.set_error("ArbitrarySeqlen backend: known bugs with SDPA F16 in cuDNN 9.10.0/9.10.1");
      return false;
    }
    
    // Check softmax type
    if (ctx.cudnn_version >= 91301) {
      // 9.13.1+: vanilla, off-by-one, learnable
    } else {
      // pre-9.13.1: vanilla
      if (ctx.softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX) {
        ctx.set_error("ArbitrarySeqlen backend (cuDNN < 9.13.1) requires vanilla softmax type");
        return false;
      }
    }
    
    return true;
  }  
  }
  
  NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
      bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
      NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
      float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
      size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
      int64_t window_size_right) {
    
    using namespace transformer_engine;
    NVTE_CHECK(q_dtype == kv_dtype, "Q and KV must have the same data type.");
    
    BackendSelectionContext ctx;
    ctx.is_training = is_training;
    ctx.q_dtype = q_dtype;
    ctx.qkv_layout = qkv_layout;
    ctx.bias_type = bias_type;
    ctx.attn_mask_type = attn_mask_type;
    ctx.softmax_type = softmax_type;
    ctx.dropout = dropout;
    ctx.num_attn_heads = num_attn_heads;
    ctx.num_gqa_groups = num_gqa_groups;
    ctx.max_seqlen_q = max_seqlen_q;
    ctx.max_seqlen_kv = max_seqlen_kv;
    ctx.head_dim_qk = head_dim_qk;
    ctx.head_dim_v = head_dim_v;
    ctx.window_size_left = window_size_left;
    ctx.window_size_right = window_size_right;
    
    const int device_id = cuda::current_device();
    ctx.sm_arch = cuda::sm_arch(device_id);
    ctx.cudnn_version = cudnnGetVersion();
    ctx.qkv_format = nvte_get_qkv_format(qkv_layout);
    ctx.q_format = nvte_get_q_format(qkv_layout);
    ctx.kv_format = nvte_get_kv_format(qkv_layout);
    ctx.layout_group = nvte_get_qkv_layout_group(qkv_layout);
    ctx.requires_64bit_ragged_offset = 
        (ctx.qkv_format == NVTE_THD && 
         fused_attn::get_ragged_offset_dtype(ctx.layout_group, num_attn_heads, num_gqa_groups,
                                            max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v) == DType::kInt64);
    ctx.supported_ragged_offset_size = 
        (!ctx.requires_64bit_ragged_offset || ctx.cudnn_version >= 90500);
    
    // Try FP8 backend
    if (checks_for_fp8(ctx)) {
      if (ctx.cudnn_version >= 8900) {
        return NVTE_Fused_Attn_Backend::NVTE_FP8;
      } else {
        std::cout << "Warning: FP8 fused attention requires cuDNN 8.9.0+. "
                  << "Please upgrade your cuDNN version." << std::endl;
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
    }
    
    // Try F16/BF16 backends
    if (q_dtype == NVTEDType::kNVTEFloat16 || q_dtype == NVTEDType::kNVTEBFloat16) {
      bool can_use_max512 = checks_for_max512(ctx);
      std::string max512_error = ctx.error_msg;
      
      bool can_use_arbitrary = checks_for_arbitrary(ctx);
      std::string arbitrary_error = ctx.error_msg;
      
      // Select backend based on seqlen and availability
      NVTE_Fused_Attn_Backend backend = NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      
      if (max_seqlen_q > 512 || max_seqlen_kv > 512) {
        // Must use arbitrary
        if (can_use_arbitrary) {
          backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
        } else {
          std::cout << "Warning: No fused attention backend available. " 
                    << arbitrary_error << std::endl;
        }
      } else {
        // seqlen <= 512: prefer arbitrary, fallback to max512
        if (can_use_arbitrary) {
          backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
        } else if (can_use_max512) {
          backend = NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen;
        } else {
          std::cout << "Warning: No fused attention backend available." << std::endl;
          std::cout << "  Max512: " << max512_error << std::endl;
          std::cout << "  Arbitrary: " << arbitrary_error << std::endl;
        }
        
        // Environment variable override
        int env_backend = static_cast<int>(backend);
        env_backend = transformer_engine::getenv<int>("NVTE_FUSED_ATTN_BACKEND", env_backend);
        
        if ((env_backend == static_cast<int>(NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) && can_use_max512) ||
            (env_backend == static_cast<int>(NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) && can_use_arbitrary)) {
          backend = static_cast<NVTE_Fused_Attn_Backend>(env_backend);
        }
      }
      
      // Validate cuDNN version for selected backend
      if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen && ctx.cudnn_version < 8901) {
        std::cout << "Warning: FP16/BF16 fused attention (max512) requires cuDNN 8.9.1+. "
                  << "Please upgrade your cuDNN version." << std::endl;
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
      
      if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen && ctx.cudnn_version < 8900) {
        std::cout << "Warning: FP16/BF16 fused attention (arbitrary) requires cuDNN 8.9.0+. "
                  << "Please upgrade your cuDNN version." << std::endl;
        return NVTE_Fused_Attn_Backend::NVTE_No_Backend;
      }
      
      return backend;
    }
    
    // No backend available
    std::cout << "Warning: No fused attention backend available for the given configuration." << std::endl;
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
                         float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTE_Softmax_Type softmax_type, int64_t window_size_left,
                         int64_t window_size_right, NVTETensor workspace, cudaStream_t stream) {
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
      is_training, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, softmax_type, dropout,
      h_q, h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right);

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
        dropout, qkv_layout, bias_type, attn_mask_type, softmax_type, window_size_left,
        window_size_right, input_Q, input_K, input_V, input_Bias, input_SoftmaxOffset, output_O,
        Aux_CTX_Tensors, input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded,
        input_cu_seqlens_kv_padded, input_page_table_k, input_page_table_v, input_rng_state,
        wkspace, stream, handle);
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
                         NVTETensor dV, NVTETensor dBias, NVTETensor dSoftmaxOffset,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
                         int64_t window_size_left, int64_t window_size_right, bool deterministic,
                         NVTETensor workspace, cudaStream_t stream) {
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
      true, Q_type, KV_type, qkv_layout, bias_type, attn_mask_type, softmax_type, dropout, h_q,
      h_kv, max_seqlen_q, max_seqlen_kv, d_qk, d_v, window_size_left, window_size_right);

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
        qkv_layout, bias_type, attn_mask_type, softmax_type, window_size_left, window_size_right,
        deterministic, input_Q, input_K, input_V, input_O, input_dO, input_Bias,
        input_SoftmaxOffset, output_S, output_dQ, output_dK, output_dV, output_dBias,
        output_dSoftmaxOffset, input_cu_seqlens_q, input_cu_seqlens_kv, input_cu_seqlens_q_padded,
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
