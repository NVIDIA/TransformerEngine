/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTENTION_CUDNN_FRONTEND_UTILS_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTENTION_CUDNN_FRONTEND_UTILS_H_

#include "../common.h"
#include "transformer_engine/fused_attention.h"

#include "cudnn_frontend.h"

constexpr static bool debug = false;

cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t);

MHA_Layout get_mha_layout(const int layout);

namespace transformer_engine {
namespace fused_attention {

using namespace transformer_engine;

void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                        int64_t *strideA, MHA_Layout layout, MHA_Matrix matrix);

bool allowAllConfig(cudnnBackendDescriptor_t engine_config);

static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type, int64_t id, int64_t const *dim,
                                            int64_t const *stride, bool is_virtual, bool is_value);

#if (CUDNN_VERSION >= 8900)
static cudnn_frontend::Tensor tensor_create_with_offset(
    cudnnDataType_t type, int64_t id, int64_t const *dim, int64_t const *stride, bool is_virtual,
    bool is_value, std::shared_ptr<cudnn_frontend::Tensor> const &raggedOffset);
#endif

static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type,
                                                    cudnnPointwiseMode_t mode);

static cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                                    cudnn_frontend::Tensor const &yDesc,
                                                    cudnn_frontend::PointWiseDesc const &pwDesc);

static cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                                     cudnn_frontend::Tensor const &bDesc,
                                                     cudnn_frontend::Tensor const &yDesc,
                                                     cudnn_frontend::PointWiseDesc const &pwDesc);

static cudnn_frontend::Operation ternary_pw_op_create(cudnn_frontend::Tensor const &xDesc,
                                                      cudnn_frontend::Tensor const &bDesc,
                                                      cudnn_frontend::Tensor const &tDesc,
                                                      cudnn_frontend::Tensor const &yDesc,
                                                      cudnn_frontend::PointWiseDesc const &pwDesc);

}  // namespace fused_attention
}  // namespace transformer_engine

#endif
