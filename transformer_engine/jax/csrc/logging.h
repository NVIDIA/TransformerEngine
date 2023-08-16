/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_LOGGING_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_LOGGING_H_

#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>

#define NVTE_ERROR(message)                                             \
  do {                                                                  \
    throw std::runtime_error(std::string(__FILE__ ":")                  \
                             + std::to_string(__LINE__)                 \
                             + " in function " + __func__ + ": "        \
                             + message);                                \
  } while (false)

#define NVTE_CHECK(expr, ...)                                   \
  do {                                                          \
    if (!(expr)) {                                              \
      NVTE_ERROR(std::string("Assertion failed: " #expr ". ")   \
                 + std::string(__VA_ARGS__));                   \
    }                                                           \
  } while (false)

#define NVTE_CHECK_CUDA(expr)                                           \
  do {                                                                  \
    const cudaError_t status_NVTE_CHECK_CUDA = (expr);                  \
    if (status_NVTE_CHECK_CUDA != cudaSuccess) {                        \
      NVTE_ERROR(std::string("CUDA Error: ")                            \
                 + cudaGetErrorString(status_NVTE_CHECK_CUDA));         \
    }                                                                   \
  } while (false)

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_LOGGING_H_
