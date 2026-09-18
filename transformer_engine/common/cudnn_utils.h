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

#include "cudnn_min_version.h"
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

// Handle owned by the common library, shared with the framework extensions. Declared extern "C" at
// global scope so the nvte_* rule in libtransformer_engine.version exports it: the extensions
// cannot instantiate cudnnExecutionPlanManager themselves, since HandleManager depends on symbols
// that are hidden from the shared object.
extern "C" cudnnHandle_t nvte_get_cudnn_handle();

#endif  //  TRANSFORMER_ENGINE_CUDNN_UTILS_H_
