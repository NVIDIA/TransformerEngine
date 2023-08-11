/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_LOGGING_H_
#define TRANSFORMER_ENGINE_LOGGING_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <nvrtc.h>
#include <stdexcept>
#include <string>

#define NVTE_ERROR(x)                                                          \
  do {                                                                         \
    throw std::runtime_error(std::string(__FILE__ ":") +                       \
                             std::to_string(__LINE__) + " in function " +      \
                             __func__ + ": " + x);                             \
  } while (false)

#define NVTE_CHECK(x, ...)                                                     \
  do {                                                                         \
    if (!(x)) {                                                                \
      NVTE_ERROR(std::string("Assertion failed: " #x ". ") +                   \
                 std::string(__VA_ARGS__));                                    \
    }                                                                          \
  } while (false)

#define NVTE_CHECK_CUDA(status)                                                \
  do {                                                                         \
    if (status != cudaSuccess) {                                               \
      NVTE_ERROR("CUDA Error: " + std::string(cudaGetErrorString(status)));    \
    }                                                                          \
  } while (false)

#define NVTE_CHECK_CUBLAS(status)                                              \
  do {                                                                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::string message;                                                     \
      message.reserve(1024);                                                   \
      message += "CUBLAS Error: ";                                             \
      message += cublasGetStatusString(status);                                \
      message += (". "                                                         \
                  "For more information, increase CUBLASLT_LOG_LEVEL, "        \
                  "by setting CUBLASLT_LOG_LEVEL=N [0-5] "                     \
                  "in the environment.");                                      \
      NVTE_ERROR(message);                                                     \
    }                                                                          \
  } while (false)

#define NVTE_CHECK_CUDNN(status)                                               \
  do {                                                                         \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::string message;                                                     \
      message.reserve(1024);                                                   \
      message += "CUDNN Error: ";                                              \
      message += cudnnGetErrorString(status);                                  \
      message += (". "                                                         \
                  "For more information, enable cuDNN error logging "          \
                  "by setting CUDNN_LOGERR_DBG=1 and "                         \
                  "CUDNN_LOGDEST_DBG=stderr in the environment.");             \
      NVTE_ERROR(message);                                                     \
    }                                                                          \
  } while (false)

#define NVTE_CHECK_NVRTC(status)                                               \
  do {                                                                         \
    if (status != NVRTC_SUCCESS) {                                             \
      NVTE_ERROR("NVRTC Error: " + std::string(nvrtcGetErrorString(status)));  \
    }                                                                          \
  } while (false)

#endif // TRANSFORMER_ENGINE_LOGGING_H_
