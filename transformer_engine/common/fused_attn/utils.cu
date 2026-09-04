/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "transformer_engine/fused_attn.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

// get matrix strides based on matrix type
void generateMatrixStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                           int64_t *strideA, NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix) {
  constexpr int batch_dim_idx = 0;
  constexpr int head_dim_idx = 1;
  constexpr int seqlen_dim_idx = 2;
  constexpr int hidden_dim_idx = 3;

  constexpr int seqlen_transpose_dim_idx = 3;
  constexpr int hidden_transpose_dim_idx = 2;

  constexpr int seqlen_q_dim_idx = 2;
  constexpr int seqlen_kv_dim_idx = 3;

  switch (layout) {
    case NVTE_QKV_Layout::NVTE_SB3HD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * 3 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = b * 3 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBH3D:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = 3 * d;
        strideA[seqlen_dim_idx] = b * 3 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = 3 * h * d;
        strideA[head_dim_idx] = 3 * d;
        strideA[seqlen_transpose_dim_idx] = b * 3 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_SB2HD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * 2 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = b * 2 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_SBH2D:
      if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = 2 * h * d;
        strideA[head_dim_idx] = 2 * d;
        strideA[seqlen_dim_idx] = b * 2 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = 2 * h * d;
        strideA[head_dim_idx] = 2 * d;
        strideA[seqlen_transpose_dim_idx] = b * 2 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = b * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BS3HD:
    case NVTE_QKV_Layout::NVTE_T3HD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = s_q * 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_q * 3 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = 3 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSH3D:
    case NVTE_QKV_Layout::NVTE_TH3D:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = s_q * 3 * h * d;
        strideA[head_dim_idx] = 3 * d;
        strideA[seqlen_dim_idx] = 3 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_q * 3 * h * d;
        strideA[head_dim_idx] = 3 * d;
        strideA[seqlen_transpose_dim_idx] = 3 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSHD_BS2HD:
    case NVTE_QKV_Layout::NVTE_THD_T2HD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = 2 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = 2 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSHD_BSH2D:
    case NVTE_QKV_Layout::NVTE_THD_TH2D:
      if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
        strideA[head_dim_idx] = 2 * d;
        strideA[seqlen_dim_idx] = 2 * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_kv * 2 * h * d;
        strideA[head_dim_idx] = 2 * d;
        strideA[seqlen_transpose_dim_idx] = 2 * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_THD_THD_THD:
    case NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = s_kv * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_kv * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = s_kv * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = s_kv * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD:
    case NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = b * h * d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_transpose_dim_idx] = b * h * d;
        strideA[hidden_transpose_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = s_q * h * d;
        strideA[head_dim_idx] = d;
        strideA[seqlen_dim_idx] = h * d;
        strideA[hidden_dim_idx] = 1;
      }
      break;
    case NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD:
      if ((matrix == NVTE_QKV_Matrix::NVTE_Q_Matrix) ||
          (matrix == NVTE_QKV_Matrix::NVTE_O_Matrix)) {
        strideA[batch_dim_idx] = h * s_q * d;
        strideA[head_dim_idx] = s_q * d;
        strideA[seqlen_dim_idx] = d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix)) {
        strideA[batch_dim_idx] = h * s_kv * d;
        strideA[head_dim_idx] = s_kv * d;
        strideA[seqlen_dim_idx] = d;
        strideA[hidden_dim_idx] = 1;
      } else if ((matrix == NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose) ||
                 (matrix == NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose)) {
        strideA[batch_dim_idx] = h * s_kv * d;
        strideA[head_dim_idx] = s_kv * d;
        strideA[seqlen_transpose_dim_idx] = d;
        strideA[hidden_transpose_dim_idx] = 1;
      }
      break;
  }

  if (matrix == NVTE_QKV_Matrix::NVTE_S_Matrix) {
    strideA[seqlen_kv_dim_idx] = 1;
    strideA[seqlen_q_dim_idx] = s_kv;
    strideA[head_dim_idx] = s_q * s_kv;
    strideA[batch_dim_idx] = h * s_q * s_kv;
  }
}

// convert cu_seqlens to actual_seqlens
__global__ void cu_seqlens_to_actual_seqlens(int64_t actual_b, int64_t max_b,
                                             int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens, int32_t *q_seqlens,
                                             int32_t *kv_seqlens) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < actual_b) {
    q_seqlens[tid] = q_cu_seqlens[tid + 1] - q_cu_seqlens[tid];
    kv_seqlens[tid] = kv_cu_seqlens[tid + 1] - kv_cu_seqlens[tid];
  } else if (tid < max_b) {
    q_seqlens[tid] = 0;
    kv_seqlens[tid] = 0;
  }
}

// convert cu_seqlens_padded to offsets
template <class OFFSETS_T>
__device__ void cu_seqlens_padded_to_offsets_impl(
    const RaggedOffsetMultipliers &mults, int64_t actual_b, int64_t max_b,
    const int32_t *cu_seqlens_q_padded, const int32_t *cu_seqlens_kv_padded, OFFSETS_T *offsets_q,
    OFFSETS_T *offsets_k, OFFSETS_T *offsets_v, OFFSETS_T *offsets_o, OFFSETS_T *offsets_s) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto cu_seqlens_id = min(tid, actual_b);
  if (tid <= max_b) {
    if (offsets_s != nullptr) {
      offsets_s[tid] = mults.stats * cu_seqlens_q_padded[cu_seqlens_id];
    }
    if (offsets_q != nullptr && offsets_o != nullptr) {
      offsets_q[tid] = mults.q * cu_seqlens_q_padded[cu_seqlens_id];
      offsets_o[tid] = mults.o * cu_seqlens_q_padded[cu_seqlens_id];
    }
    if (offsets_k != nullptr && offsets_v != nullptr) {
      const int32_t *cu_seqlens_kv_src =
          mults.kv_from_q ? cu_seqlens_q_padded : cu_seqlens_kv_padded;
      offsets_k[tid] = mults.k * cu_seqlens_kv_src[cu_seqlens_id];
      offsets_v[tid] = mults.v * cu_seqlens_kv_src[cu_seqlens_id];
    }
  }
}

__global__ void cu_seqlens_padded_to_offsets(RaggedOffsetMultipliers mults, int64_t actual_b,
                                             int64_t max_b, const int32_t *cu_seqlens_q_padded,
                                             const int32_t *cu_seqlens_kv_padded,
                                             DType offset_dtype, void *offsets_q, void *offsets_k,
                                             void *offsets_v, void *offsets_o, void *offsets_s) {
  if (offset_dtype == DType::kInt32) {
    cu_seqlens_padded_to_offsets_impl<int32_t>(
        mults, actual_b, max_b, cu_seqlens_q_padded, cu_seqlens_kv_padded,
        reinterpret_cast<int32_t *>(offsets_q), reinterpret_cast<int32_t *>(offsets_k),
        reinterpret_cast<int32_t *>(offsets_v), reinterpret_cast<int32_t *>(offsets_o),
        reinterpret_cast<int32_t *>(offsets_s));
  } else {
    assert(offset_dtype == DType::kInt64 && "expect int64");
    cu_seqlens_padded_to_offsets_impl<int64_t>(
        mults, actual_b, max_b, cu_seqlens_q_padded, cu_seqlens_kv_padded,
        reinterpret_cast<int64_t *>(offsets_q), reinterpret_cast<int64_t *>(offsets_k),
        reinterpret_cast<int64_t *>(offsets_v), reinterpret_cast<int64_t *>(offsets_o),
        reinterpret_cast<int64_t *>(offsets_s));
  }
}

DType get_ragged_offset_dtype(NVTE_QKV_Layout_Group layout_group, int64_t num_attn_heads,
                              int64_t num_gqa_groups, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                              int64_t head_dim_qk, int64_t head_dim_v) {
  std::array<int64_t, 4> offsets_qkvo{};
  switch (layout_group) {
    case NVTE_QKV_Layout_Group::NVTE_HD_HD_HD:
    case NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD:
      offsets_qkvo[0] = num_attn_heads * head_dim_qk * max_seqlen_q;
      offsets_qkvo[1] = num_gqa_groups * head_dim_qk * max_seqlen_kv;
      offsets_qkvo[2] = num_gqa_groups * head_dim_v * max_seqlen_kv;
      break;
    case NVTE_QKV_Layout_Group::NVTE_3HD:
    case NVTE_QKV_Layout_Group::NVTE_H3D:
      offsets_qkvo[0] = 3 * num_attn_heads * head_dim_qk * max_seqlen_q;
      offsets_qkvo[1] = offsets_qkvo[0];
      offsets_qkvo[2] = offsets_qkvo[0];
      break;
    case NVTE_QKV_Layout_Group::NVTE_HD_2HD:
    case NVTE_QKV_Layout_Group::NVTE_HD_H2D:
      offsets_qkvo[0] = num_attn_heads * head_dim_qk * max_seqlen_q;
      offsets_qkvo[1] = 2 * num_gqa_groups * head_dim_qk * max_seqlen_kv;
      offsets_qkvo[2] = offsets_qkvo[1];
      break;
  }

  offsets_qkvo[3] = num_attn_heads * head_dim_qk * max_seqlen_q;

  size_t max_offset = *std::max_element(offsets_qkvo.begin(), offsets_qkvo.end());
  if (max_offset > std::numeric_limits<int32_t>::max()) {
    return DType::kInt64;
  }

  return DType::kInt32;
}

// quantize batch size
size_t get_max_batch_size(size_t batch_size) {
  if (batch_size == 0) return 0;  // guard: log2(0) = -inf, casting to size_t is UB
  size_t max_b = batch_size;
  size_t log2_b = ceil(log2(batch_size));
  // batch size is expected to be 10s-100s
  // b = 1, ..., 32   -> max_b = 32
  // b = 33, ..., 512 -> max_b = next power of 2
  // b = 513, ...     -> max_b = increment by 512
  if (log2_b <= 5) {
    max_b = 32;
  } else if (log2_b <= 9) {
    max_b = pow(2, log2_b);
  } else {
    max_b = (batch_size + 511) / 512 * 512;
  }
  return max_b;
}

// quantize token count
size_t get_max_tokens(size_t num_tokens) {
  if (num_tokens == 0) return 0;  // guard: log2(0) = -inf, casting to size_t is UB
  // token count is expected to be 1k's-100k's
  // t = 0, ..., 1024   -> max_t = 1024
  // t = 1025, ..., 32k -> max_t = next power of 2
  // t = 32k+1, ...     -> max_t = increment by 32k
  size_t log2_t = ceil(log2(num_tokens));
  size_t max_t = 0;
  if (log2_t <= 10) {
    max_t = 1024;
  } else if (log2_t <= 15) {
    max_t = pow(2, log2_t);
  } else {
    max_t = (num_tokens + 32767) / 32768 * 32768;
  }
  return max_t;
}

__global__ void populate_rng_state_kernel(int64_t *rng_state_dst, const int64_t *const seed,
                                          int64_t offset) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > 0) return;
  rng_state_dst[0] = seed[0];
  rng_state_dst[1] = offset;
}

__global__ void get_runtime_num_segments_kernel(int32_t *cu_seqlen, size_t len, uint32_t *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= len) return;

  if (cu_seqlen[tid] > 0) {
    // atomicAdd only support 32 bits dtype
    atomicAdd(out, 1);
  }
}

void PopulateRngStateAsync(void *rng_state_dst, const void *seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream) {
  size_t increment = 0;
  if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
    increment = 16;
  } else {
    constexpr int threads_per_cta = 128;
    increment = (q_max_seqlen * kv_max_seqlen + threads_per_cta - 1) / threads_per_cta;
  }
  auto offset = FusedAttnOffsetManager::Instance().GetAndUpdateOffset(increment);
  populate_rng_state_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<int64_t *>(rng_state_dst),
                                                 reinterpret_cast<const int64_t *>(seed), offset);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

uint32_t GetRuntimeNumSegments(void *cu_seqlen, void *workspace, size_t len, cudaStream_t stream) {
  // workspace size requires 4 bytes
  uint32_t *dout = static_cast<uint32_t *>(workspace);
  uint32_t hout{};
  NVTE_CHECK_CUDA(cudaMemsetAsync(dout, 0, sizeof(uint32_t), stream));
  constexpr int threads = 128;
  const int blocks = (len - 1) / threads + 1;
  get_runtime_num_segments_kernel<<<blocks, threads, 0, stream>>>(static_cast<int32_t *>(cu_seqlen),
                                                                  len, dout);
  NVTE_CHECK_CUDA(cudaGetLastError());
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&hout, dout, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  return hout;
}

__global__ void extract_seed_and_offset(int64_t *rng_state_ptr, bool captured, int64_t *seed_ptr,
                                        uint64_t seed_val, int64_t *offset_ptr, uint64_t offset_val,
                                        uint32_t offset_intragraph) {
  if (captured) {
    rng_state_ptr[0] = *seed_ptr;
    rng_state_ptr[1] = static_cast<int64_t>(*offset_ptr + static_cast<int64_t>(offset_intragraph));
  } else {
    rng_state_ptr[0] = static_cast<int64_t>(seed_val);
    rng_state_ptr[1] = static_cast<int64_t>(offset_val);
  }
}

}  // namespace fused_attn
}  // namespace transformer_engine

void nvte_extract_seed_and_offset(int64_t *rng_state_ptr, int captured, int64_t *seed_ptr,
                                  uint64_t seed_val, int64_t *offset_ptr, uint64_t offset_val,
                                  uint32_t offset_intragraph, cudaStream_t stream) {
  NVTE_API_CALL(nvte_extract_seed_and_offset);
  using namespace transformer_engine;

  fused_attn::extract_seed_and_offset<<<1, 1, 0, stream>>>(
      rng_state_ptr, captured, seed_ptr, seed_val, offset_ptr, offset_val, offset_intragraph);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
