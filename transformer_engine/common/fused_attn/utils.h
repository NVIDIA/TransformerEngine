/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  std::int64_t d;
  std::int64_t bias_b;
  std::int64_t bias_h;
  float attnScale;
  bool isTraining;
  float dropoutProbability;
  NVTE_QKV_Layout layout;
  NVTE_Bias_Type bias_type;
  NVTE_Mask_Type mask_type;
  cudnn_frontend::DataType_t fwd_tensor_type;
  cudnn_frontend::DataType_t bwd_tensor_type;

  bool operator<(const FADescriptor_v1 &rhs) const {
    return std::tie(b, h, hg, s_q, s_kv, d, bias_b, bias_h, attnScale, isTraining,
                    dropoutProbability, layout, mask_type, bias_type, fwd_tensor_type,
                    bwd_tensor_type) <
           std::tie(rhs.b, rhs.h, rhs.hg, rhs.s_q, rhs.s_kv, rhs.d, rhs.bias_b, rhs.bias_h,
                    rhs.attnScale, rhs.isTraining, rhs.dropoutProbability, rhs.layout,
                    rhs.mask_type, rhs.bias_type, rhs.fwd_tensor_type, rhs.bwd_tensor_type);
  }
};

__global__ void cu_seqlens_to_offsets(size_t b, size_t h, size_t d, int32_t *cu_seqlens_q,
                                      int32_t *actual_seqlens_q, int32_t *qkv_ragged_offset,
                                      int32_t *o_ragged_offset);

__global__ void cu_seqlens_to_actual_seqlens(size_t b, int32_t const *const q_cu_seqlens,
                                             int32_t const *const kv_cu_seqlens, int32_t *q_seqlens,
                                             int32_t *kv_seqlens);

__global__ void cu_seqlens_padded_to_offsets(NVTE_QKV_Layout_Group layout_group, size_t b, size_t h,
                                             size_t hg, size_t d, int32_t *cu_seqlens_q_padded,
                                             int32_t *cu_seqlens_kv_padded, int32_t *offsets_q,
                                             int32_t *offsets_k, int32_t *offsets_v,
                                             int32_t *offsets_o);
}  // namespace fused_attn

cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t);
cudnn_frontend::DataType_t get_cudnn_fe_dtype(const transformer_engine::DType t);

class cudnnExecutionPlanManager {
 public:
  static cudnnExecutionPlanManager &Instance() {
    static thread_local cudnnExecutionPlanManager instance;
    return instance;
  }

  cudnnHandle_t GetCudnnHandle() {
    static thread_local std::once_flag flag;
    std::call_once(flag, [&] { cudnnCreate(&handle_); });
    return handle_;
  }

  ~cudnnExecutionPlanManager() {}

 private:
  cudnnHandle_t handle_ = nullptr;
};
}  // namespace transformer_engine

#endif
