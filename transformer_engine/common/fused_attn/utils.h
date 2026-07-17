/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_UTILS_H_

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>

#include <cstdint>
#include <mutex>

#include "../common.h"
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

enum NVTE_QKV_Matrix {
  NVTE_Q_Matrix = 0,            // queries
  NVTE_K_Matrix = 1,            // keys
  NVTE_K_Matrix_Transpose = 2,  // keys transposed
  NVTE_V_Matrix = 3,            // values
  NVTE_V_Matrix_Transpose = 4,  // values transposed
  NVTE_S_Matrix = 5,            // output of GEMM1
  NVTE_O_Matrix = 6,            // final output
};

// Padded sizes for MXFP8 layout (s_q/s_kv/d_qk/d_v and their scaled dimensions)
struct MXFP8PaddedSizes {
  int64_t s_q_padded;
  int64_t s_kv_padded;
  int64_t s_q_scale;
  int64_t s_kv_scale;
  int64_t s_q_scale_padded;
  int64_t s_kv_scale_padded;
  int64_t d_qk_padded;
  int64_t d_v_padded;
  int64_t d_qk_scale;
  int64_t d_v_scale;
  int64_t d_qk_scale_padded;
  int64_t d_v_scale_padded;
};

// Pad s and d for MXFP8 quantization
inline MXFP8PaddedSizes pad_s_d_for_mxfp8(int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v) {
  constexpr int64_t block_size = 32;
  MXFP8PaddedSizes p;
  p.s_q_padded = DIVUP_TO_MULTIPLE(s_q, 128);
  p.s_kv_padded = DIVUP_TO_MULTIPLE(s_kv, 128);
  p.s_q_scale = DIVUP(s_q, block_size);
  p.s_kv_scale = DIVUP(s_kv, block_size);
  p.s_q_scale_padded = DIVUP_TO_MULTIPLE(p.s_q_scale, 4);
  p.s_kv_scale_padded = DIVUP_TO_MULTIPLE(p.s_kv_scale, 4);
  p.d_qk_padded = DIVUP_TO_MULTIPLE(d_qk, 128);
  p.d_v_padded = DIVUP_TO_MULTIPLE(d_v, 128);
  p.d_qk_scale = DIVUP(d_qk, block_size);
  p.d_v_scale = DIVUP(d_v, block_size);
  p.d_qk_scale_padded = DIVUP_TO_MULTIPLE(p.d_qk_scale, 4);
  p.d_v_scale_padded = DIVUP_TO_MULTIPLE(p.d_v_scale, 4);
  return p;
}

// Get matrix strides for a 4D tensor [batch_size, num_heads, sequence_len, head_dim] given a QKV format.
// strides must point to at least 4 int64_t elements.
inline void generateMatrixStridesWithFormat(int64_t b, int64_t h, int64_t s, int64_t d,
                                            int64_t *strides, NVTE_QKV_Format format) {
  constexpr int b_dim = 0;
  constexpr int h_dim = 1;
  constexpr int s_dim = 2;
  constexpr int d_dim = 3;

  switch (format) {
    case NVTE_QKV_Format::NVTE_BSHD:
    case NVTE_QKV_Format::NVTE_THD:
      strides[b_dim] = s * h * d;
      strides[h_dim] = d;
      strides[s_dim] = h * d;
      strides[d_dim] = 1;
      break;
    case NVTE_QKV_Format::NVTE_SBHD:
      strides[b_dim] = h * d;
      strides[h_dim] = d;
      strides[s_dim] = b * h * d;
      strides[d_dim] = 1;
      break;
    case NVTE_QKV_Format::NVTE_BHSD:
      strides[b_dim] = h * s * d;
      strides[h_dim] = s * d;
      strides[s_dim] = d;
      strides[d_dim] = 1;
      break;
    default:
      NVTE_CHECK(false, "Invalid format.");
      break;
  }
}

// get matrix strides based on layout and matrix type
inline void generateMatrixStridesWithLayout(int64_t b, int64_t h, int64_t hg, int64_t s_q,
                                            int64_t s_kv, int64_t d_qk, int64_t d_v,
                                            int64_t *q_strides, int64_t *k_strides,
                                            int64_t *v_strides, NVTE_QKV_Layout layout) {
  constexpr int b_dim = 0;
  constexpr int h_dim = 1;
  constexpr int s_dim = 2;
  constexpr int d_dim = 3;
  const NVTE_QKV_Format q_format = nvte_get_q_format(layout);
  const NVTE_QKV_Format kv_format = nvte_get_kv_format(layout);

  switch (layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
      q_strides[b_dim] = 3 * h * d_qk;
      q_strides[h_dim] = d_qk;
      q_strides[s_dim] = b * 3 * h * d_qk;
      q_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        k_strides[i] = v_strides[i] = q_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBH3D:
      q_strides[b_dim] = 3 * h * d_qk;
      q_strides[h_dim] = 3 * d_qk;
      q_strides[s_dim] = b * 3 * h * d_qk;
      q_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        k_strides[i] = v_strides[i] = q_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
      generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_strides, q_format);
      k_strides[b_dim] = 2 * hg * d_qk;
      k_strides[h_dim] = d_qk;
      k_strides[s_dim] = b * 2 * hg * d_qk;
      k_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        v_strides[i] = k_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
      generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_strides, q_format);
      k_strides[b_dim] = 2 * hg * d_qk;
      k_strides[h_dim] = 2 * d_qk;
      k_strides[s_dim] = b * 2 * hg * d_qk;
      k_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        v_strides[i] = k_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_T3HD:
      q_strides[b_dim] = s_q * 3 * h * d_qk;
      q_strides[h_dim] = d_qk;
      q_strides[s_dim] = 3 * h * d_qk;
      q_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        k_strides[i] = v_strides[i] = q_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_TH3D:
      q_strides[b_dim] = s_q * 3 * h * d_qk;
      q_strides[h_dim] = 3 * d_qk;
      q_strides[s_dim] = 3 * h * d_qk;
      q_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        k_strides[i] = v_strides[i] = q_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
      generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_strides, q_format);
      k_strides[b_dim] = s_kv * 2 * hg * d_qk;
      k_strides[h_dim] = d_qk;
      k_strides[s_dim] = 2 * hg * d_qk;
      k_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        v_strides[i] = k_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
      generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_strides, q_format);
      k_strides[b_dim] = s_kv * 2 * hg * d_qk;
      k_strides[h_dim] = 2 * d_qk;
      k_strides[s_dim] = 2 * hg * d_qk;
      k_strides[d_dim] = 1;
      for (int i = 0; i < 4; i++) {
        v_strides[i] = k_strides[i];
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
    case NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD:
      generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_strides, q_format);
      generateMatrixStridesWithFormat(b, hg, s_kv, d_qk, k_strides, kv_format);
      generateMatrixStridesWithFormat(b, hg, s_kv, d_v, v_strides, kv_format);
      break;
    default:
      NVTE_CHECK(false, "Invalid layout.");
      break;
  }
}

void generateMatrixStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                           int64_t *strideA, NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix);

bool allowAllConfig(cudnnBackendDescriptor_t engine_config);

cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const *dim,
                                     int64_t const *stride, bool is_virtual, bool is_value);

cudnn_frontend::Tensor tensor_create_with_offset(
    cudnnDataType_t type, int64_t id, int64_t const *dim, int64_t const *stride, bool is_virtual,
    bool is_value, std::shared_ptr<cudnn_frontend::Tensor> raggedOffset);

cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type, cudnnPointwiseMode_t mode);

cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                             cudnn_frontend::Tensor const &yDesc,
                                             cudnn_frontend::PointWiseDesc const &pwDesc);

cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                              cudnn_frontend::Tensor const &bDesc,
                                              cudnn_frontend::Tensor const &yDesc,
                                              cudnn_frontend::PointWiseDesc const &pwDesc);

cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                               cudnn_frontend::Tensor const &bDesc,
                                               cudnn_frontend::Tensor const &tDesc,
                                               cudnn_frontend::Tensor const &yDesc,
                                               cudnn_frontend::PointWiseDesc const &pwDesc);

// Per-tensor scale factors relating cu_seqlens_padded (token units) to tensor-element
// ragged offsets, as a function of the QKV layout group. Single source of truth shared
// by the cu_seqlens_padded_to_offsets conversion kernel and the direct-seqlens path
// (which passes them to cuDNN as ragged offset multipliers).
struct RaggedOffsetMultipliers {
  RaggedOffsetMultipliers(NVTE_QKV_Layout_Group layout_group, int64_t h, int64_t hg, int64_t d_qk,
                          int64_t d_v)
      : q(h * d_qk), k(hg * d_qk), v(hg * d_v), o(h * d_v), stats(h), kv_from_q(false) {
    switch (layout_group) {
      case NVTE_QKV_Layout_Group::NVTE_3HD:
      case NVTE_QKV_Layout_Group::NVTE_H3D:
        q = k = v = 3 * h * d_qk;
        kv_from_q = true;
        break;
      case NVTE_QKV_Layout_Group::NVTE_HD_2HD:
      case NVTE_QKV_Layout_Group::NVTE_HD_H2D:
        k = v = 2 * hg * d_qk;
        break;
      default:
        break;
    }
  }

  int64_t q;
  int64_t k;
  int64_t v;
  int64_t o;
  int64_t stats;
  // K/V offsets scale the Q-side cu_seqlens_padded (interleaved QKV layouts)
  bool kv_from_q;
};

__global__ void cu_seqlens_to_actual_seqlens(int64_t actual_b, int64_t max_b,
                                             int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens, int32_t *q_seqlens,
                                             int32_t *kv_seqlens);

__global__ void cu_seqlens_padded_to_offsets(RaggedOffsetMultipliers mults, int64_t actual_b,
                                             int64_t max_b, const int32_t *cu_seqlens_q_padded,
                                             const int32_t *cu_seqlens_kv_padded,
                                             DType offset_dtype, void *offsets_q, void *offsets_k,
                                             void *offsets_v, void *offsets_o, void *offsets_s);

DType get_ragged_offset_dtype(NVTE_QKV_Layout_Group layout_group, int64_t num_attn_heads,
                              int64_t num_gqa_groups, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                              int64_t head_dim_qk, int64_t head_dim_v);

size_t get_max_batch_size(size_t batch_size);
size_t get_max_tokens(size_t num_tokens);

class FusedAttnOffsetManager {
 public:
  static FusedAttnOffsetManager &Instance() {
    static thread_local FusedAttnOffsetManager instance;
    return instance;
  }

  size_t GetAndUpdateOffset(size_t increment) {
    size_t ret = offset_;
    offset_ += increment;
    return ret;
  }

  FusedAttnOffsetManager(FusedAttnOffsetManager const &) = delete;
  void operator=(FusedAttnOffsetManager const &) = delete;

 private:
  FusedAttnOffsetManager() {}
  size_t offset_ = 0;
};

__global__ void populate_rng_state_kernel(int64_t *rng_state_dst, const int64_t *const seed,
                                          int64_t offset);

__global__ void get_runtime_num_segments_kernel(int32_t *cu_seqlen, size_t len, uint32_t *out);

void PopulateRngStateAsync(void *rng_state_dst, const void *const seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream);

uint32_t GetRuntimeNumSegments(void *cu_seqlen, void *workspace, size_t len, cudaStream_t stream);

}  // namespace fused_attn
}  // namespace transformer_engine

#endif
