/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <nvrtc.h>

#include <stdexcept>

#include "../util/string.h"

#define NVTE_ERROR(...)                                              \
  do {                                                               \
    throw ::std::runtime_error(::transformer_engine::concat_strings( \
        __FILE__ ":", __LINE__, " in function ", __func__, ": ",     \
        ::transformer_engine::concat_strings(__VA_ARGS__)));         \
  } while (false)

#define NVTE_CHECK(expr, ...)                                        \
  do {                                                               \
    if (!(expr)) {                                                   \
      NVTE_ERROR("Assertion failed: " #expr ". ",                    \
                 ::transformer_engine::concat_strings(__VA_ARGS__)); \
    }                                                                \
  } while (false)

#define NVTE_CHECK_CUDA(expr)                                                 \
  do {                                                                        \
    const cudaError_t status_NVTE_CHECK_CUDA = (expr);                        \
    if (status_NVTE_CHECK_CUDA != cudaSuccess) {                              \
      NVTE_ERROR("CUDA Error: ", cudaGetErrorString(status_NVTE_CHECK_CUDA)); \
    }                                                                         \
  } while (false)

#define NVTE_CHECK_CUBLAS(expr)                                                      \
  do {                                                                               \
    const cublasStatus_t status_NVTE_CHECK_CUBLAS = (expr);                          \
    if (status_NVTE_CHECK_CUBLAS != CUBLAS_STATUS_SUCCESS) {                         \
      NVTE_ERROR("cuBLAS Error: ", cublasGetStatusString(status_NVTE_CHECK_CUBLAS)); \
    }                                                                                \
  } while (false)

#define NVTE_CHECK_CUDNN(expr)                                                  \
  do {                                                                          \
    const cudnnStatus_t status_NVTE_CHECK_CUDNN = (expr);                       \
    if (status_NVTE_CHECK_CUDNN != CUDNN_STATUS_SUCCESS) {                      \
      NVTE_ERROR("cuDNN Error: ", cudnnGetErrorString(status_NVTE_CHECK_CUDNN), \
                 ". "                                                           \
                 "For more information, enable cuDNN error logging "            \
                 "by setting CUDNN_LOGERR_DBG=1 and "                           \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");               \
    }                                                                           \
  } while (false)

#define NVTE_CHECK_CUDNN_FE(expr)                                    \
  do {                                                               \
    const auto error = (expr);                                       \
    if (error.is_bad()) {                                            \
      NVTE_ERROR("cuDNN Error: ", error.err_msg,                     \
                 ". "                                                \
                 "For more information, enable cuDNN error logging " \
                 "by setting CUDNN_LOGERR_DBG=1 and "                \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");    \
    }                                                                \
  } while (false)

#define NVTE_CHECK_NVRTC(expr)                                                   \
  do {                                                                           \
    const nvrtcResult status_NVTE_CHECK_NVRTC = (expr);                          \
    if (status_NVTE_CHECK_NVRTC != NVRTC_SUCCESS) {                              \
      NVTE_ERROR("NVRTC Error: ", nvrtcGetErrorString(status_NVTE_CHECK_NVRTC)); \
    }                                                                            \
  } while (false)

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
