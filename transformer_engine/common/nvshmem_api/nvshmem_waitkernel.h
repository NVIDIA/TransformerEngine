/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H
#define TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

/*! \enum WaitKind
 *  \brief Types of wait operations that can be performed.
 */
enum class WaitKind {
  KERNEL_WAIT = 0,  /*!< Wait using a CUDA kernel */
  NVSHMEM_WAIT = 1, /*!< Wait using NVSHMEM wait operation */
  STREAM_WAIT = 2   /*!< Wait using CUDA stream synchronization */
};

/*! \brief Wait on a signal until a certain condition is met.
 *
 *  \param[in]     sig_addr        The address of the signal to wait on.
 *  \param[in]     wait_kind       The kind of wait to perform.
 *  \param[in]     stream          The stream to wait on.
 */
void nvshmem_wait_on_stream(uint64_t* sig_addr, WaitKind wait_kind, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMMON_NVSHMEM_WAITKERNEL_H
