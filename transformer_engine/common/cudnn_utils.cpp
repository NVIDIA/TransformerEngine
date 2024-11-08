/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cudnn_utils.h"

#include "./util/logging.h"
#include "transformer_engine/cudnn.h"

namespace transformer_engine {

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

// get cuDNN data type
cudnn_frontend::DataType_t get_cudnn_fe_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kInt32:
      return cudnn_frontend::DataType_t::INT32;
    case DType::kInt64:
      return cudnn_frontend::DataType_t::INT64;
    case DType::kFloat16:
      return cudnn_frontend::DataType_t::HALF;
    case DType::kFloat32:
      return cudnn_frontend::DataType_t::FLOAT;
    case DType::kBFloat16:
      return cudnn_frontend::DataType_t::BFLOAT16;
    case DType::kFloat8E4M3:
      return cudnn_frontend::DataType_t::FP8_E4M3;
    case DType::kFloat8E5M2:
      return cudnn_frontend::DataType_t::FP8_E5M2;
    default:
      NVTE_ERROR("Invalid cuDNN data type. \n");
  }
}

void nvte_cudnn_handle_init() {
  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
}

}  // namespace transformer_engine

namespace cudnn_frontend {

// This is needed to define the symbol `cudnn_dlhandle`
// When using the flag NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
// to enable dynamic loading.
void *cudnn_dlhandle = nullptr;

}  // namespace cudnn_frontend
