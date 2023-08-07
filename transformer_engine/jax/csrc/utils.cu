/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime_api.h>
#include <cassert>

#include "utils.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion() {
    int ver = 0;
    NVTE_CHECK_CUDA(cudaRuntimeGetVersion(&ver));
    return ver;
}

int GetDeviceComputeCapability(int gpu_id) {
    int max_num_gpu = 0;
    NVTE_CHECK_CUDA(cudaGetDeviceCount(&max_num_gpu));
    assert(gpu_id < max_num_gpu);

    int major = 0;
    NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu_id));

    int minor = 0;
    NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, gpu_id));

    int gpu_arch = major * 10 + minor;
    return gpu_arch;
}

__global__ void populate_rng_state_kernel(int64_t *rng_state_dst, const int64_t *const seed,
                                          int64_t offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0) return;
    rng_state_dst[0] = seed[0];
    rng_state_dst[1] = offset;
}

void PopulateRngStateAsync(void *rng_state_dst, const void *const seed, size_t q_max_seqlen,
                           size_t kv_max_seqlen, NVTE_Fused_Attn_Backend backend,
                           cudaStream_t stream) {
    size_t increment = 0;
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
        increment = 16;
    } else {
        constexpr int threads_per_cta = 128;
        increment = (q_max_seqlen * kv_max_seqlen + threads_per_cta - 1) / threads_per_cta;
    }
    auto offset = FusedAttnOffsetManager::Instance().GetAndUpdateOffset(increment);
    populate_rng_state_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<int64_t *>(rng_state_dst),
                                                   reinterpret_cast<const int64_t *>(seed), offset);
    NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace jax
}  // namespace transformer_engine
