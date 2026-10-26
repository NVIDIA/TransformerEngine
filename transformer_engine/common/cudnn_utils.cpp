/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cudnn_utils.h"

#include <string>

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

void nvte_cudnn_handle_init() { auto _ = nvte_get_cudnn_handle(); }

namespace detail {

namespace {

std::string format_cudnn_version(size_t version) {
  // cuDNN 8 encoded versions as MAJOR * 1000 + MINOR * 100 + PATCH, cuDNN 9 and later use
  // MAJOR * 10000 + MINOR * 100 + PATCH.
  const size_t major_magnitude = version < 90000 ? 1000 : 10000;
  const size_t remainder = version % major_magnitude;
  return std::to_string(version / major_magnitude) + "." + std::to_string(remainder / 100) + "." +
         std::to_string(remainder % 100);
}

}  // namespace

void CreateCuDNNHandle(cudnnHandle_t* handle) {
  // cuDNN is loaded dynamically, so the runtime version can be older than the headers TE was
  // built against. This is the single point every cuDNN-backed path in every framework reaches.
  const size_t version = cudnnGetVersion();
  NVTE_CHECK(version >= kMinCudnnVersion, "Transformer Engine requires cuDNN ",
             format_cudnn_version(kMinCudnnVersion), " or later, but the cuDNN runtime is ",
             format_cudnn_version(version), ".");
  NVTE_CHECK_CUDNN(cudnnCreate(handle));
}

}  // namespace detail

}  // namespace transformer_engine

extern "C" cudnnHandle_t nvte_get_cudnn_handle() {
  return transformer_engine::cudnnExecutionPlanManager::Instance().GetHandle();
}

namespace cudnn_frontend {

// This is needed to define the symbol `cudnn_dlhandle`
// When using the flag NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
// to enable dynamic loading.
void* cudnn_dlhandle = nullptr;

}  // namespace cudnn_frontend
