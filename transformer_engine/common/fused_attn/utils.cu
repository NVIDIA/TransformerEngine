/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

// get matrix strides based on matrix type
void generateMatrixStrides(
            int64_t b, int64_t h,
            int64_t s_q, int64_t s_kv,
            int64_t d, int64_t* strideA,
            NVTE_QKV_Layout layout, NVTE_QKV_Matrix matrix) {
    constexpr int batch_dim_idx   = 0;
    constexpr int head_dim_idx    = 1;
    constexpr int seqlen_dim_idx  = 2;
    constexpr int hidden_dim_idx  = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix) {
        case NVTE_QKV_Matrix::NVTE_Q_Matrix:
            if (layout == NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * 3 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_q * h * d;
            }
            break;
        case NVTE_QKV_Matrix::NVTE_K_Matrix:
            if (layout == NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == NVTE_QKV_Layout::NVTE_KV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[seqlen_dim_idx] = h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case NVTE_QKV_Matrix::NVTE_K_Matrix_Transpose:
            if (layout == NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == NVTE_QKV_Layout::NVTE_KV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case NVTE_QKV_Matrix::NVTE_V_Matrix:
            if (layout == NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == NVTE_QKV_Layout::NVTE_KV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 2* h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case NVTE_QKV_Matrix::NVTE_V_Matrix_Transpose:
            if (layout == NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED) {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * 3 * h * d;
                } else if (layout == NVTE_QKV_Layout::NVTE_KV_INTERLEAVED) {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = 2* h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * 2 * h * d;
                } else {
                    strideA[hidden_transpose_dim_idx] = 1;
                    strideA[seqlen_transpose_dim_idx] = h * d;
                    strideA[head_dim_idx] = d;
                    strideA[batch_dim_idx] = s_kv * h * d;
                }
            break;
        case NVTE_QKV_Matrix::NVTE_S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
        case NVTE_QKV_Matrix::NVTE_O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = h * d;
            strideA[head_dim_idx] = d;
            strideA[batch_dim_idx] = s_q * h * d;
            break;
    }
}

// get qkv pointer offsets based on qkv layout
void get_qkv_offset(void** devPtrQ, void** devPtrK, void** devPtrV,
            int64_t num_heads, int64_t head_dim, NVTE_QKV_Layout qkv_layout,
            NVTEDType qkv_dtype) {

    size_t num_bytes = 0;
    switch (qkv_dtype) {
            case kNVTEByte:
            case kNVTEFloat8E4M3:
            case kNVTEFloat8E5M2:
                    num_bytes = 1;
                    break;
            case kNVTEInt32:
            case kNVTEFloat32:
                    num_bytes = 4;
                    break;
            case kNVTEInt64:
                    num_bytes = 8;
                    break;
            case kNVTEFloat16:
            case kNVTEBFloat16:
                    num_bytes = 2;
                    break;
            default:
                    NVTE_ERROR("NVTEDType not supported!");
    }
    printf("get offset qkv: num_bytes %d\n", num_bytes);

    int layout_mod = (int)qkv_layout % 5; 
    size_t stride = 0;
    //void *devPtrQ = static_cast<void *>(devPtrQKV);
    //void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
    //void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);
    switch (layout_mod) {
        case 0:
            stride = num_bytes * num_heads * head_dim;
	    printf("---------- Im case 0, %d bytes, stide %d\n", num_bytes, stride);
            *devPtrK = static_cast<void *>(static_cast<int8_t *>(*devPtrQ) + stride);
            *devPtrV = static_cast<void *>(static_cast<int8_t *>(*devPtrQ) + 2 * stride);
            break;
        case 1:
            stride = num_bytes * head_dim;
            *devPtrK = static_cast<void *>(static_cast<int8_t *>(*devPtrQ) + stride);
            *devPtrV = static_cast<void *>(static_cast<int8_t *>(*devPtrQ) + 2 * stride);
            break;
        case 2:
            stride = num_bytes * num_heads * head_dim;
            *devPtrV = static_cast<void *>(static_cast<int8_t *>(*devPtrK) + stride);
            break;
        case 3:
            stride = num_bytes * head_dim;
            *devPtrV = static_cast<void *>(static_cast<int8_t *>(*devPtrK) + stride);
            break;
        case 4:
            break;
        default:
            NVTE_ERROR("QKV Layout not supported!");
    }
}

bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
  (void)engine_config;
  return false;
}

cudnn_frontend::Tensor tensor_create(
                cudnnDataType_t type, int64_t id,
                int64_t const * dim, int64_t const * stride,
                bool is_virtual, bool is_value) {
  int nbDims = 4;
  auto tensor_created = cudnn_frontend::TensorBuilder()
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
                cudnnDataType_t type, int64_t id,
                int64_t const * dim, int64_t const * stride,
                bool is_virtual, bool is_value,
                std::shared_ptr<cudnn_frontend::Tensor> raggedOffset) {
  int nbDims = 4;
  auto tensor_created = cudnn_frontend::TensorBuilder()
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

cudnn_frontend::PointWiseDesc pw_desc_create(
                cudnnDataType_t type, cudnnPointwiseMode_t mode) {
  auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
          .setMode(mode)
          .setComputeType(type)
          .build();
  return pw_desc_created;
}

cudnn_frontend::Operation unary_pw_op_create(
                cudnn_frontend::Tensor const &xDesc,
                cudnn_frontend::Tensor const &yDesc,
                cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(xDesc)
                      .setyDesc(yDesc)
                      .setpwDesc(pwDesc)
                      .build();
  return pw_op_created;
}

cudnn_frontend::Operation binary_pw_op_create(
                cudnn_frontend::Tensor const &xDesc,
                cudnn_frontend::Tensor const &bDesc,
                cudnn_frontend::Tensor const &yDesc,
                cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(xDesc)
                      .setbDesc(bDesc)
                      .setyDesc(yDesc)
                      .setpwDesc(pwDesc)
                      .build();
  return pw_op_created;
}

cudnn_frontend::Operation ternary_pw_op_create(
    cudnn_frontend::Tensor const &xDesc, cudnn_frontend::Tensor const &bDesc,
    cudnn_frontend::Tensor const &tDesc, cudnn_frontend::Tensor const &yDesc,
    cudnn_frontend::PointWiseDesc const &pwDesc) {
  auto pw_op_created = cudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(xDesc)
                           .setbDesc(bDesc)
                           .settDesc(tDesc)
                           .setyDesc(yDesc)
                           .setpwDesc(pwDesc)
                           .build();
  return pw_op_created;
}

// convert cu_seqlens_q to qkv/o_ragged_offset and actual_seqlens_q
__global__ void cu_seqlens_to_offsets(size_t b, size_t h, size_t d,
                int32_t *cu_seqlens_q, int32_t *actual_seqlens_q,
                int32_t *qkv_ragged_offset, int32_t *o_ragged_offset) {
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
__global__ void cu_seqlens_to_actual_seqlens(size_t b,
                int32_t const * const q_cu_seqlens,
                int32_t const * const kv_cu_seqlens,
                int32_t *q_seqlens, int32_t *kv_seqlens) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < b) {
    q_seqlens[tid] = q_cu_seqlens[tid + 1] - q_cu_seqlens[tid];
    kv_seqlens[tid] = kv_cu_seqlens[tid + 1] - kv_cu_seqlens[tid];
  }
}
}  // namespace fused_attn

// get cuDNN data type
cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kInt32:
      return CUDNN_DATA_INT32;
    case DType::kInt64:
      return CUDNN_DATA_INT64;
    case DType::kFloat16:
      return CUDNN_DATA_HALF;
    case DType::kFloat32:
      return CUDNN_DATA_FLOAT;
    case DType::kBFloat16:
      return CUDNN_DATA_BFLOAT16;
    case DType::kFloat8E4M3:
      return CUDNN_DATA_FP8_E4M3;
    case DType::kFloat8E5M2:
      return CUDNN_DATA_FP8_E5M2;
    default:
      NVTE_ERROR("Invalid cuDNN data type. \n");
  }
}
}  // namespace transformer_engine
