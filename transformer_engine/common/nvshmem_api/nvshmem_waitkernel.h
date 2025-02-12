/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H
#define TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H

namespace transformer_engine {
void nvshmem_wait_on_stream(uint64_t* sig_addr, int wait_kind, cudaStream_t stream);
}
#endif  // TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H
