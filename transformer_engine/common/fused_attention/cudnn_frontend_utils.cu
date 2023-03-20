/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cudnn_frontend_utils.h"

cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t) {
    using namespace transformer_engine;
    if (debug) printf("cudnn get type %d\n", static_cast<int>(t));
    switch (t) {
        case DType::kFloat16:
            return CUDNN_DATA_HALF;
        case DType::kFloat32:
            return CUDNN_DATA_FLOAT;
        case DType::kBFloat16:
            return CUDNN_DATA_BFLOAT16;
        case DType::kFloat8E4M3:
            if (debug) printf(" type DType::kFloat8E4M3 \n");
            return CUDNN_DATA_FP8_E4M3;
        case DType::kFloat8E5M2:
            if (debug) printf(" type DType::kFloat8E5M2 \n");
            return CUDNN_DATA_FP8_E5M2;
        default:
            NVTE_ERROR("Invalid type");
    }
}

MHA_Layout get_mha_layout(const int layout) {
    switch (layout) {
        case 0:
            return MHA_Layout::NOT_INTERLEAVED;
        case 1:
            return MHA_Layout::QKV_INTERLEAVED;
        case 2:
            return MHA_Layout::KV_INTERLEAVED;
        default:
            NVTE_ERROR("Invalid layout");
    }
}

namespace transformer_engine {
namespace fused_attention {

using namespace transformer_engine;

void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                        int64_t *strideA, MHA_Layout layout, MHA_Matrix matrix) {
    CUDNN_FRONTEND_UNUSED(b);
    constexpr int batch_dim_idx = 0;
    constexpr int head_dim_idx = 1;
    constexpr int seqlen_dim_idx = 2;
    constexpr int hidden_dim_idx = 3;

    constexpr int seqlen_transpose_dim_idx = 3;
    constexpr int hidden_transpose_dim_idx = 2;

    constexpr int seqlen_q_dim_idx = 2;
    constexpr int seqlen_kv_dim_idx = 3;

    switch (matrix) {
        case MHA_Matrix::Q_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
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
        case MHA_Matrix::K_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[hidden_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
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
        case MHA_Matrix::K_Matrix_Transpose:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
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
        case MHA_Matrix::V_Matrix:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = 2 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[hidden_dim_idx] = 1;
                strideA[seqlen_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::V_Matrix_Transpose:
            if (layout == MHA_Layout::QKV_INTERLEAVED) {
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[seqlen_transpose_dim_idx] = 3 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 3 * h * d;
            } else if (layout == MHA_Layout::KV_INTERLEAVED) {
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[seqlen_transpose_dim_idx] = 2 * h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * 2 * h * d;
            } else {
                strideA[hidden_transpose_dim_idx] = 1;
                strideA[seqlen_transpose_dim_idx] = h * d;
                strideA[head_dim_idx] = d;
                strideA[batch_dim_idx] = s_kv * h * d;
            }
            break;
        case MHA_Matrix::S_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = s_kv;
            strideA[head_dim_idx] = s_q * s_kv;
            strideA[batch_dim_idx] = h * s_q * s_kv;
            break;
        case MHA_Matrix::O_Matrix:
            strideA[seqlen_kv_dim_idx] = 1;
            strideA[seqlen_q_dim_idx] = h * d;
            strideA[head_dim_idx] = d;
            strideA[batch_dim_idx] = s_q * h * d;
            break;
    }
}

bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const *dim,
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
    if (debug) std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
}

#if (CUDNN_VERSION >= 8900)
static cudnn_frontend::Tensor tensor_create_with_offset(
    cudnnDataType_t type, int64_t id, int64_t const *dim, int64_t const *stride, bool is_virtual,
    bool is_value, std::shared_ptr<cudnn_frontend::Tensor> const &raggedOffset) {
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
    if (debug) std::cout << tensor_created.describe() << std::endl;
    return tensor_created;
}
#endif

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type,
                                                    cudnnPointwiseMode_t mode) {
    auto pw_desc_created =
        cudnn_frontend::PointWiseDescBuilder().setMode(mode).setComputeType(type).build();

    if (debug) std::cout << pw_desc_created.describe() << std::endl;
    return pw_desc_created;
}

static cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                                    cudnn_frontend::Tensor const &yDesc,
                                                    cudnn_frontend::PointWiseDesc const &pwDesc) {
    auto pw_op_created =
        cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(xDesc)
            .setyDesc(yDesc)
            .setpwDesc(pwDesc)
            .build();
    if (debug) std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
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
    if (debug) std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

static cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
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
    if (debug) std::cout << pw_op_created.describe() << std::endl;
    return pw_op_created;
}

}  // namespace fused_attention
}  // namespace transformer_engine
