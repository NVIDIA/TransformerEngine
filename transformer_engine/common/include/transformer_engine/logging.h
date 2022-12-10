/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_LOGGING_H_
#define TRANSFORMER_ENGINE_LOGGING_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

#define NVTE_ERROR(x)                                                               \
  do {                                                                              \
    throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) + \
                             " in function " + __func__ + ": " + x);                \
  } while (false)

#define NVTE_CHECK(x, ...)                                                              \
  do {                                                                                  \
    if (!(x)) {                                                                         \
      NVTE_ERROR(std::string("Assertion failed: " #x ". ") + std::string(__VA_ARGS__)); \
    }                                                                                   \
  } while (false)

namespace {

inline void check_cuda_(cudaError_t status) {
  if (status != cudaSuccess) {
    NVTE_ERROR("CUDA Error: " + std::string(cudaGetErrorString(status)));
  }
}

inline void check_cublas_(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    NVTE_ERROR("CUBLAS Error: " + std::string(cublasGetStatusString(status)));
  }
}

}  // namespace

#define NVTE_CHECK_CUDA(ans) \
  { check_cuda_(ans); }

#define NVTE_CHECK_CUBLAS(ans) \
  { check_cublas_(ans); }

#endif  // TRANSFORMER_ENGINE_LOGGING_H_
