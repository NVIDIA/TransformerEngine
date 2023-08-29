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
#include <vector>
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/logging.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion();
int GetDeviceComputeCapability(int gpu_id);

void PopulateRngStateAsync(void *rng_state_dst, const void *const seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream);

class WorkspaceManager {
 public:
    static WorkspaceManager &Instance() {
        static thread_local WorkspaceManager instance;
        return instance;
    }

    WorkspaceManager() {}
    ~WorkspaceManager() { Clear_(); }

    void *GetWorkspace(size_t size = 4194304) {
        ReallocateIfNeed_(size);
        return workspace_;
    }

    std::vector<void *> GetWorkspace(std::vector<size_t> sizes) {
        size_t total_size = 0;
        for (int i = 0; i < sizes.size(); i++) {
            sizes[i] = PadSize_(sizes[i]);
            total_size += sizes[i];
        }

        ReallocateIfNeed_(total_size);
        std::vector<void *> ptrs(sizes.size(), workspace_);

        size_t accumulated = 0;
        for (int i = 0; i < ptrs.size(); i++) {
            ptrs[i] = static_cast<char *>(workspace_) + accumulated;
            accumulated += sizes[i];
        }

        return ptrs;
    }

 private:
    void *workspace_ = nullptr;
    size_t size_ = 0;

    size_t PadSize_(size_t size) {
        constexpr size_t alignment = 128;
        return ((size + alignment - 1) / alignment) * alignment;
    }

    void Clear_() {
        if (workspace_ != nullptr) {
            NVTE_CHECK_CUDA(cudaFree(workspace_));
        }
        workspace_ = nullptr;
        size_ = 0;
    }

    void Allocate_(size_t new_size) {
        new_size = PadSize_(new_size);
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
