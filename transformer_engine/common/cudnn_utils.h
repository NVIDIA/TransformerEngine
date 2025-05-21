/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_CUDNN_UTILS_H_
#define TRANSFORMER_ENGINE_CUDNN_UTILS_H_

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>
#include <cudnn_graph.h>

#include "transformer_engine/transformer_engine.h"
#include "util/handle_manager.h"

namespace transformer_engine {

namespace detail {

void CreateCuDNNHandle(cudnnHandle_t* handle);

}  // namespace detail

cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t);

cudnn_frontend::DataType_t get_cudnn_fe_dtype(const transformer_engine::DType t);

using cudnnExecutionPlanManager = detail::HandleManager<cudnnHandle_t, detail::CreateCuDNNHandle>;

}  // namespace transformer_engine

#endif  //  TRANSFORMER_ENGINE_CUDNN_UTILS_H_
