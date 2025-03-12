/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H
#define TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H

namespace transformer_engine {
/*! \brief Wait on a signal until a certain condition is met.
 *
 *  \param[in]     sig_addr        The address of the signal to wait on.
 *  \param[in]     wait_kind       The kind of wait to perform.
 *  \param[in]     stream          The stream to wait on.
 */
void nvshmem_wait_on_stream(uint64_t* sig_addr, int wait_kind, cudaStream_t stream);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H
