/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine {
namespace jax {
namespace {

__global__ void PopulateFusedAttnRngStateKernel(int64_t *rng_state, const int64_t *seed,
                                                uint64_t offset) {
  rng_state[0] = seed[0];
  rng_state[1] = static_cast<int64_t>(offset);
}

}  // namespace

void PopulateFusedAttnRngState(void *rng_state, const void *seed, uint64_t offset,
                               cudaStream_t stream) {
  PopulateFusedAttnRngStateKernel<<<1, 1, 0, stream>>>(static_cast<int64_t *>(rng_state),
                                                       static_cast<const int64_t *>(seed), offset);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace jax
}  // namespace transformer_engine
