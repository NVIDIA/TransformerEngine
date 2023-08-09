/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/logging.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion();
int GetDeviceComputeCapability(int gpu_id);

void PopulateRngStateAsync(void *rng_state_dst, const void *const seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream);

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

    int GetMajor() {
        if (!prop_queried_) {
            int device_id;
            NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
            cudaGetDeviceProperties(&prop_, device_id);
            prop_queried_ = true;
        }
        return prop_.major;
    }

 private:
    bool prop_queried_ = false;
    cudaDeviceProp prop_;
};

class FusedAttnOffsetManager {
 public:
    static FusedAttnOffsetManager &Instance() {
        static thread_local FusedAttnOffsetManager instance;
        return instance;
    }

    size_t GetAndUpdateOffset(size_t increment) {
        size_t ret = offset_;
        offset_ += increment;
        return ret;
    }

    FusedAttnOffsetManager(FusedAttnOffsetManager const &) = delete;
    void operator=(FusedAttnOffsetManager const &) = delete;

 private:
    FusedAttnOffsetManager() {}
    size_t offset_ = 0;
};

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
