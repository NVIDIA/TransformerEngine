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

}  // namespace jax
}  // namespace transformer_engine
