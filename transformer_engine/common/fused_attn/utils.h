/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  NVTE_V_Matrix_Transpose = 4,  // value matrix transposed
  NVTE_S_Matrix = 5,            // output of GEMM1
  NVTE_O_Matrix = 6,            // final output
};

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

struct FADescriptor {
  std::int64_t b;
  std::int64_t h;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d;
  float attnScale;
  bool isTraining;
  float dropoutProbability;
  NVTE_QKV_Layout layout;
  NVTE_Bias_Type bias_type;
  NVTE_Mask_Type mask_type;
  cudnnDataType_t tensor_type;
  bool use_workspace_opt;

  bool operator<(const FADescriptor &rhs) const {
    return std::tie(b, h, s_q, s_kv, d, attnScale, isTraining, dropoutProbability, layout,
                    mask_type, bias_type, tensor_type, use_workspace_opt) <
           std::tie(rhs.b, rhs.h, rhs.s_q, rhs.s_kv, rhs.d, rhs.attnScale, rhs.isTraining,
                    rhs.dropoutProbability, rhs.layout, rhs.mask_type, rhs.bias_type,
                    rhs.tensor_type, rhs.use_workspace_opt);
  }
};

struct FADescriptor_v1 {
  std::int64_t b;
  std::int64_t h;
  std::int64_t hg;
  std::int64_t s_q;
  std::int64_t s_kv;
  std::int64_t d_qk;
  std::int64_t d_v;
  std::int64_t bias_b;
  std::int64_t bias_h;
  float attnScale;
  bool isTraining;
  float dropoutProbability;
  NVTE_QKV_Layout layout;
  NVTE_Bias_Type bias_type;
  NVTE_Mask_Type mask_type;
  std::int64_t window_size_left;
  std::int64_t window_size_right;
  bool deterministic;
  cudnn_frontend::DataType_t fwd_tensor_type;
  cudnn_frontend::DataType_t bwd_tensor_type;

  bool operator<(const FADescriptor_v1 &rhs) const {
    return std::tie(b, h, hg, s_q, s_kv, d_qk, d_v, bias_b, bias_h, attnScale, isTraining,
                    dropoutProbability, layout, mask_type, window_size_left, window_size_right,
                    deterministic, bias_type, fwd_tensor_type, bwd_tensor_type) <
           std::tie(rhs.b, rhs.h, rhs.hg, rhs.s_q, rhs.s_kv, rhs.d_qk, rhs.d_v, rhs.bias_b,
                    rhs.bias_h, rhs.attnScale, rhs.isTraining, rhs.dropoutProbability, rhs.layout,
                    rhs.mask_type, rhs.window_size_left, rhs.window_size_right, rhs.deterministic,
                    rhs.bias_type, rhs.fwd_tensor_type, rhs.bwd_tensor_type);
  }
};

__global__ void cu_seqlens_to_offsets(int64_t b, int64_t h, int64_t d, int32_t *cu_seqlens_q,
                                      int32_t *actual_seqlens_q, int32_t *qkv_ragged_offset,
                                      int32_t *o_ragged_offset);

__global__ void cu_seqlens_to_actual_seqlens(int64_t actual_b, int64_t max_b,
                                             int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens, int32_t *q_seqlens,
                                             int32_t *kv_seqlens);

__global__ void cu_seqlens_padded_to_offsets(NVTE_QKV_Layout_Group layout_group, int64_t actual_b,
                                             int64_t max_b, int64_t h, int64_t hg, int64_t d_qk,
                                             int64_t d_v, const int32_t *cu_seqlens_q_padded,
                                             const int32_t *cu_seqlens_kv_padded,
                                             DType offset_dtype, void *offsets_q, void *offsets_k,
                                             void *offsets_v, void *offsets_o, void *offsets_s);

DType get_ragged_offset_dtype(NVTE_QKV_Layout_Group layout_group, int64_t num_attn_heads,
                              int64_t num_gqa_groups, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                              int64_t head_dim_qk, int64_t head_dim_v);

size_t get_max_batch_size(size_t batch_size);
size_t get_max_tokens(size_t num_tokens);

}  // namespace fused_attn
}  // namespace transformer_engine

#endif
