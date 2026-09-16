/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Minimum cuDNN version supported by Transformer Engine, encoded as cudnnGetVersion() reports it.
// Keep in sync with MIN_CUDNN_VERSION in transformer_engine/common/__init__.py.
constexpr size_t kMinCudnnVersion = 91200;

static_assert(static_cast<size_t>(CUDNN_VERSION) >= kMinCudnnVersion,
              "Transformer Engine must be built against cuDNN 9.12.0 or later headers.");

namespace detail {

void CreateCuDNNHandle(cudnnHandle_t* handle);

}  // namespace detail

cudnnDataType_t get_cudnn_dtype(const transformer_engine::DType t);

cudnn_frontend::DataType_t get_cudnn_fe_dtype(const transformer_engine::DType t);

using cudnnExecutionPlanManager = detail::HandleManager<cudnnHandle_t, detail::CreateCuDNNHandle>;

}  // namespace transformer_engine

#endif  //  TRANSFORMER_ENGINE_CUDNN_UTILS_H_
