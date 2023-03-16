/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <mutex>  // NOLINT [build/c++11]
#include <stdexcept>
#include <string>
#include <type_traits>
#include "transformer_engine/logging.h"

namespace transformer_engine {
namespace jax {

class cublasLtMetaManager {
 public:
    static cublasLtMetaManager &Instance() {
        static thread_local cublasLtMetaManager instance;
        return instance;
    }

    cublasLtMetaManager() {}
    ~cublasLtMetaManager() { Clear_(); }

    void *GetWorkspace(size_t size = 4194304) {
        ReallocateIfNeed_(size);
        return workspace_;
    }

 private:
    void *workspace_ = nullptr;
    size_t size_ = 0;

    void Clear_() {
        if (workspace_ != nullptr) {
            NVTE_CHECK_CUDA(cudaFree(workspace_));
        }
        workspace_ = nullptr;
        size_ = 0;
    }

    void Allocate_(size_t new_size) {
        NVTE_CHECK_CUDA(cudaMalloc(&workspace_, new_size));
        size_ = new_size;
    }

    void ReallocateIfNeed_(size_t new_size) {
        if (new_size > size_) {
            Clear_();
            Allocate_(new_size);
        }
    }
};

class cudaDevicePropertiesManager {
 public:
    static cudaDevicePropertiesManager &Instance() {
        static thread_local cudaDevicePropertiesManager instance;
        return instance;
    }

    int GetMultiProcessorCount() {
        if (!prop_queried_) {
            int device_id;
            NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
            cudaGetDeviceProperties(&prop_, device_id);
            prop_queried_ = true;
        }
        return prop_.multiProcessorCount;
    }

 private:
    bool prop_queried_ = false;
    cudaDeviceProp prop_;
};

class cudnnExecutionPlanManager {
 public:
    static cudnnExecutionPlanManager &Instance() {
        static thread_local cudnnExecutionPlanManager instance;
        return instance;
    }

    cudnnHandle_t GetCudnnHandle() {
        static std::once_flag flag;
        std::call_once(flag, [&] { cudnnCreate(&handle_); });
        return handle_;
    }

 private:
    cudnnHandle_t handle_;
};

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
