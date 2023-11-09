/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <pybind11/pybind11.h>

#include "common/util/logging.h"
#include <transformer_engine/fused_attn.h>

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

    template <typename... Args>
    inline auto GetWorkspace(Args... args) {
        auto asks = std::array<size_t, sizeof...(Args)>{args...};
        std::array<size_t, sizeof...(Args) + 1> offsets = {0};
        std::array<void *, sizeof...(Args)> workspaces = {nullptr};
        std::transform_inclusive_scan(
            asks.cbegin(), asks.cend(), offsets.begin() + 1, std::plus<size_t>{},
            [=](auto x) { return PadSize_(x); }, 0);
        auto *workspace = GetWorkspace(offsets.back());
        std::transform(offsets.cbegin(), offsets.cend() - 1, workspaces.begin(),
                       [workspace](auto x) { return static_cast<char *>(workspace) + x; });
        return workspaces;
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
