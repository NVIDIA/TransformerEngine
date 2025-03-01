/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime_api.h>

#include <cassert>

#include "common/util/cuda_runtime.h"
#include "utils.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion() {
  int ver = 0;
  NVTE_CHECK_CUDA(cudaRuntimeGetVersion(&ver));
  return ver;
}

size_t GetCudnnRuntimeVersion() { return cudnnGetVersion(); }

int GetDeviceComputeCapability(int gpu_id) { return transformer_engine::cuda::sm_arch(gpu_id); }

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

__global__ void get_runtime_num_segments_kernel(int32_t *cu_seqlen, size_t len, uint32_t *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= len) return;

  if (cu_seqlen[tid] > 0) {
    // atomicAdd only support 32 bits dtype
    atomicAdd(out, 1);
  }
}

uint32_t GetRuntimeNumSegments(void *cu_seqlen, void *workspace, size_t len, cudaStream_t stream) {
  // workspace size requires 4 bytes
  uint32_t *dout = static_cast<uint32_t *>(workspace);
  uint32_t hout{};
  cudaMemsetAsync(dout, 0, sizeof(uint32_t), stream);
  constexpr int threads = 128;
  const int blocks = (len - 1) / threads + 1;
  get_runtime_num_segments_kernel<<<blocks, threads, 0, stream>>>(static_cast<int32_t *>(cu_seqlen),
                                                                  len, dout);
  cudaMemcpyAsync(&hout, dout, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return hout;
}

}  // namespace jax
}  // namespace transformer_engine
