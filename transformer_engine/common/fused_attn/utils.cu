/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>

#include "../common.h"
#include "../cudnn_utils.h"
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
  }

  if (matrix == NVTE_QKV_Matrix::NVTE_S_Matrix) {
    strideA[seqlen_kv_dim_idx] = 1;
    strideA[seqlen_q_dim_idx] = s_kv;
    strideA[head_dim_idx] = s_q * s_kv;
    strideA[batch_dim_idx] = h * s_q * s_kv;
  }
}

bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
  (void)engine_config;
  return false;
}

cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const *dim,
                                     int64_t const *stride, bool is_virtual, bool is_value) {
  int nbDims = 4;
  auto tensor_created =
      cudnn_frontend::TensorBuilder()
          .setDim(nbDims, dim)
          .setStride(nbDims, stride)
          .setId(id)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(type)
          .setVirtual(is_virtual)
          .setByValue(is_value)
          .build();
  return tensor_created;
}

cudnn_frontend::Tensor tensor_create_with_offset(
    cudnnDataType_t type, int64_t id, int64_t const *dim, int64_t const *stride, bool is_virtual,
    bool is_value, std::shared_ptr<cudnn_frontend::Tensor> raggedOffset) {
  int nbDims = 4;
  auto tensor_created =
      cudnn_frontend::TensorBuilder()
          .setDim(nbDims, dim)
          .setStride(nbDims, stride)
          .setId(id)
          .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
          .setDataType(type)
          .setVirtual(is_virtual)
          .setByValue(is_value)
          .setRaggedOffset(raggedOffset)
          .build();
  return tensor_created;
}

cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type, cudnnPointwiseMode_t mode) {
  auto pw_desc_created =
      cudnn_frontend::PointWiseDescBuilder().setMode(mode).setComputeType(type).build();
  return pw_desc_created;
}

cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                             cudnn_frontend::Tensor const &yDesc,
                                             cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(xDesc)
          .setyDesc(yDesc)
          .setpwDesc(pwDesc)
          .build();
  return pw_op_created;
}

cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                              cudnn_frontend::Tensor const &bDesc,
                                              cudnn_frontend::Tensor const &yDesc,
                                              cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(xDesc)
          .setbDesc(bDesc)
          .setyDesc(yDesc)
          .setpwDesc(pwDesc)
          .build();
  return pw_op_created;
}

cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                               cudnn_frontend::Tensor const &bDesc,
                                               cudnn_frontend::Tensor const &tDesc,
                                               cudnn_frontend::Tensor const &yDesc,
                                               cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created =
      cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(xDesc)
          .setbDesc(bDesc)
          .settDesc(tDesc)
          .setyDesc(yDesc)
          .setpwDesc(pwDesc)
          .build();
  return pw_op_created;
}

// convert cu_seqlens_q to qkv/o_ragged_offset and actual_seqlens_q
__global__ void cu_seqlens_to_offsets(int64_t b, int64_t h, int64_t d, int32_t *cu_seqlens_q,
                                      int32_t *actual_seqlens_q, int32_t *qkv_ragged_offset,
                                      int32_t *o_ragged_offset) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < b) {
    actual_seqlens_q[tid] = cu_seqlens_q[tid + 1] - cu_seqlens_q[tid];
  }
  if (tid < b + 1) {
    qkv_ragged_offset[tid] = cu_seqlens_q[tid] * 3 * h * d;
    o_ragged_offset[tid] = cu_seqlens_q[tid] * h * d;
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
    NVTE_QKV_Layout_Group layout_group, int64_t actual_b, int64_t max_b, int64_t h, int64_t hg,
    int64_t d_qk, int64_t d_v, const int32_t *cu_seqlens_q_padded,
    const int32_t *cu_seqlens_kv_padded, OFFSETS_T *offsets_q, OFFSETS_T *offsets_k,
    OFFSETS_T *offsets_v, OFFSETS_T *offsets_o, OFFSETS_T *offsets_s) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto cu_seqlens_id = min(tid, actual_b);
  if (tid <= max_b) {
    if (offsets_s != nullptr) {
      offsets_s[tid] = h * cu_seqlens_q_padded[cu_seqlens_id];
    }
    if (offsets_q != nullptr && offsets_o != nullptr) {
      offsets_o[tid] = h * d_v * cu_seqlens_q_padded[cu_seqlens_id];
      switch (layout_group) {
        case NVTE_QKV_Layout_Group::NVTE_HD_HD_HD:
        case NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD:
          offsets_q[tid] = h * d_qk * cu_seqlens_q_padded[cu_seqlens_id];
          break;
        case NVTE_QKV_Layout_Group::NVTE_3HD:
        case NVTE_QKV_Layout_Group::NVTE_H3D:
          offsets_q[tid] = 3 * h * d_qk * cu_seqlens_q_padded[cu_seqlens_id];
          break;
        case NVTE_QKV_Layout_Group::NVTE_HD_2HD:
        case NVTE_QKV_Layout_Group::NVTE_HD_H2D:
          offsets_q[tid] = h * d_qk * cu_seqlens_q_padded[cu_seqlens_id];
          break;
      }
    }
    if (offsets_k != nullptr && offsets_v != nullptr) {
      switch (layout_group) {
        case NVTE_QKV_Layout_Group::NVTE_HD_HD_HD:
        case NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD:
          offsets_k[tid] = hg * d_qk * cu_seqlens_kv_padded[cu_seqlens_id];
          offsets_v[tid] = hg * d_v * cu_seqlens_kv_padded[cu_seqlens_id];
          break;
        case NVTE_QKV_Layout_Group::NVTE_3HD:
        case NVTE_QKV_Layout_Group::NVTE_H3D:
          offsets_k[tid] = 3 * h * d_qk * cu_seqlens_q_padded[cu_seqlens_id];
          offsets_v[tid] = offsets_k[cu_seqlens_id];
          break;
        case NVTE_QKV_Layout_Group::NVTE_HD_2HD:
        case NVTE_QKV_Layout_Group::NVTE_HD_H2D:
          offsets_k[tid] = 2 * hg * d_qk * cu_seqlens_kv_padded[cu_seqlens_id];
          offsets_v[tid] = offsets_k[cu_seqlens_id];
          break;
      }
    }
  }
}

__global__ void cu_seqlens_padded_to_offsets(NVTE_QKV_Layout_Group layout_group, int64_t actual_b,
                                             int64_t max_b, int64_t h, int64_t hg, int64_t d_qk,
                                             int64_t d_v, const int32_t *cu_seqlens_q_padded,
                                             const int32_t *cu_seqlens_kv_padded,
                                             DType offset_dtype, void *offsets_q, void *offsets_k,
                                             void *offsets_v, void *offsets_o, void *offsets_s) {
  if (offset_dtype == DType::kInt32) {
    cu_seqlens_padded_to_offsets_impl<int32_t>(
        layout_group, actual_b, max_b, h, hg, d_qk, d_v, cu_seqlens_q_padded, cu_seqlens_kv_padded,
        reinterpret_cast<int32_t *>(offsets_q), reinterpret_cast<int32_t *>(offsets_k),
        reinterpret_cast<int32_t *>(offsets_v), reinterpret_cast<int32_t *>(offsets_o),
        reinterpret_cast<int32_t *>(offsets_s));
  } else {
    assert(offset_dtype == DType::kInt64 && "expect int64");
    cu_seqlens_padded_to_offsets_impl<int64_t>(
        layout_group, actual_b, max_b, h, hg, d_qk, d_v, cu_seqlens_q_padded, cu_seqlens_kv_padded,
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
  size_t max_b = batch_size;
  size_t log2_b = ceil(log2(batch_size));
  // batch size is expected to be 10s-100s
  // b = 1, ..., 32   -> max_b = 32
  // b = 33, ..., 512 -> max_b = next power of 2
  // otherwise        -> max_b = b
  if (log2_b <= 5) {
    max_b = 32;
  } else if (log2_b <= 9) {
    max_b = pow(2, log2_b);
  }
  return max_b;
}

// quantize token count
size_t get_max_tokens(size_t num_tokens) {
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
  cudaMemsetAsync(dout, 0, sizeof(uint32_t), stream);
  constexpr int threads = 128;
  const int blocks = (len - 1) / threads + 1;
  get_runtime_num_segments_kernel<<<blocks, threads, 0, stream>>>(static_cast<int32_t *>(cu_seqlen),
                                                                  len, dout);
  cudaMemcpyAsync(&hout, dout, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
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
}
