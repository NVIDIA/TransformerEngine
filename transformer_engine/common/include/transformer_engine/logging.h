/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_LOGGING_H_
#define TRANSFORMER_ENGINE_LOGGING_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <nvrtc.h>
#include <string>
#include <stdexcept>

#define NVTE_ERROR(x) \
    do { \
        throw std::runtime_error(std::string(__FILE__ ":") + std::to_string(__LINE__) +            \
                                 " in function " + __func__ + ": " + x);                           \
    } while (false)

#define NVTE_LIBRARY_ERROR(x, file, line, func) \
    do { \
        throw std::runtime_error(std::string(file ":") + std::to_string(line) +            \
                                 " in function " + func + ": " + x);                           \
    } while (false)

#define NVTE_CHECK(x, ...)                                                                         \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            NVTE_ERROR(std::string("Assertion failed: "  #x ". ") + std::string(__VA_ARGS__));     \
        }                                                                                          \
    } while (false)

namespace {

inline void check_cuda_(cudaError_t status, const char *file, const int line, const char *func) {
    if ( status != cudaSuccess ) {
        NVTE_LIBRARY_ERROR("CUDA Error: " + std::string(cudaGetErrorString(status)), file, line, func);
    }
}

inline void check_cublas_(cublasStatus_t status, const char *file, const int line, const char *func) {
    if ( status != CUBLAS_STATUS_SUCCESS ) {
        NVTE_LIBRARY_ERROR("CUBLAS Error: " + std::string(cublasGetStatusString(status)), file, line, func);
    }
}

inline void check_cudnn_(cudnnStatus_t status, const char *file, const int line, const char *func) {
    if ( status != CUDNN_STATUS_SUCCESS ) {
        std::string message;
        message.reserve(1024);
        message += "CUDNN Error: ";
        message += cudnnGetErrorString(status);
        message += (". "
                    "For more information, enable cuDNN error logging "
                    "by setting CUDNN_LOGERR_DBG=1 and "
                    "CUDNN_LOGDEST_DBG=stderr in the environment.");
        NVTE_LIBRARY_ERROR(message, file, line, func);
    }
}

inline void check_nvrtc_(nvrtcResult status, const char *file, const int line, const char *func) {
    if ( status != NVRTC_SUCCESS ) {
        NVTE_LIBRARY_ERROR("NVRTC Error: " + std::string(nvrtcGetErrorString(status)), file, line, func);
    }
}

}  // namespace

#define NVTE_CHECK_CUDA(ans) { check_cuda_(ans, __FILE__, __LINE__, __func__); }

#define NVTE_CHECK_CUBLAS(ans) { check_cublas_(ans, __FILE__, __LINE__, __func__); }

#define NVTE_CHECK_CUDNN(ans) { check_cudnn_(ans, __FILE__, __LINE__, __func__); }

#define NVTE_CHECK_NVRTC(ans) { check_nvrtc_(ans, __FILE__, __LINE__, __func__); }

#endif  // TRANSFORMER_ENGINE_LOGGING_H_
