/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdexcept>
#include <dpct/blas_utils.hpp>

#include "../util/string.h"

#define NVTE_ERROR(...)                                         \
  do {                                                          \
    throw ::std::runtime_error(                                 \
      ::transformer_engine::concat_strings(                     \
        __FILE__ ":", __LINE__,                                 \
        " in function ", __func__, ": ",                        \
        ::transformer_engine::concat_strings(__VA_ARGS__)));    \
  } while (false)

#define NVTE_CHECK(expr, ...)                                           \
  do {                                                                  \
    if (!(expr)) {                                                      \
      NVTE_ERROR("Assertion failed: " #expr ". ",                       \
                 ::transformer_engine::concat_strings(__VA_ARGS__));    \
    }                                                                   \
  } while (false)

#define NVTE_CHECK_CUDA(expr)                                                  \
    do {                                                                       \
        const dpct::err0 status_NVTE_CHECK_CUDA = (expr);                      \
                                                                               \
    } while (false)

#define NVTE_CHECK_CUBLAS(expr)                                                                            \
    do {                                                                                                   \
        const int status_NVTE_CHECK_CUBLAS = (expr);                                                       \
        if (status_NVTE_CHECK_CUBLAS != 0) {                                                               \
            NVTE_ERROR("cuBLAS Error: ", "cublasGetStatusString is not "                                   \
                                         "supported" /*cublasGetStatusString(status_NVTE_CHECK_CUBLAS)*/); \
        }                                                                                                  \
    } while (false)

#define NVTE_CHECK_CUDNN(expr)                                                                          \
    do {                                                                                                \
        const dpct::err1 status_NVTE_CHECK_CUDNN = (expr);                                              \
        if (status_NVTE_CHECK_CUDNN != 0) {                                                             \
            NVTE_ERROR(                                                                                 \
                "cuDNN Error: ",                                                                        \
                "cudnnGetErrorString is not supported" /*cudnnGetErrorString(status_NVTE_CHECK_CUDNN)*/ \
                ,                                                                                       \
                ". "                                                                                    \
                "For more information, enable cuDNN error logging "                                     \
                "by setting CUDNN_LOGERR_DBG=1 and "                                                    \
                "CUDNN_LOGDEST_DBG=stderr in the environment.");                                        \
        }                                                                                               \
    } while (false)

#define NVTE_CHECK_CUDNN_FE(expr)                                       \
  do {                                                                  \
    const auto error = (expr);                                          \
    if (error.is_bad()) {                                               \
      NVTE_ERROR("cuDNN Error: ",                                       \
                 error.err_msg,                                         \
                 ". "                                                   \
                 "For more information, enable cuDNN error logging "    \
                 "by setting CUDNN_LOGERR_DBG=1 and "                   \
                 "CUDNN_LOGDEST_DBG=stderr in the environment.");       \
    }                                                                   \
  } while (false)

#define NVTE_CHECK_NVRTC(expr)                                  \
  do {                                                          \
    const nvrtcResult status_NVTE_CHECK_NVRTC = (expr);         \
    if (status_NVTE_CHECK_NVRTC != NVRTC_SUCCESS) {             \
      NVTE_ERROR("NVRTC Error: ",                               \
                 nvrtcGetErrorString(status_NVTE_CHECK_NVRTC)); \
    }                                                           \
  } while (false)

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_LOGGING_H_
