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

template <typename OutT, typename InT>
__global__ void cast_kernel(OutT *out, const InT *in, size_t num_elem) {
    // This kernel is not optimized as the num_elem is small now
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elem) return;
    out[tid] = static_cast<OutT>(in[tid]);
}

template <typename OutT, typename InT>
void CastAsync(void *out, const void *in, size_t num_elem, cudaStream_t stream) {
    constexpr size_t nthreads_per_block = 128;
    const size_t grid = (num_elem + nthreads_per_block - 1) / nthreads_per_block;

    cast_kernel<<<grid, nthreads_per_block, 0, stream>>>(
        reinterpret_cast<OutT *>(out), reinterpret_cast<const InT *>(in), num_elem);
    NVTE_CHECK_CUDA(cudaGetLastError());
}
template void CastAsync<int64_t, uint32_t>(void *, const void *, size_t, cudaStream_t);

}  // namespace jax
}  // namespace transformer_engine
