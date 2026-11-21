/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file utils.h
 *  \brief Utility functions (e.g. host-to-device value stores).
 */

#ifndef TRANSFORMER_ENGINE_UTILS_H_
#define TRANSFORMER_ENGINE_UTILS_H_

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Copy a small host buffer into device memory via kernel arguments.
 *
 *  The host buffer may be modified or freed after this call returns.
 *  This is compatible with CUDA Graphs.
 *
 *  \param[in]     host_ptr     Source in host memory.
 *  \param[out]    device_ptr   Destination in device memory.
 *  \param[in]     num_bytes    Size of the value in bytes.
 *  \param[in]     stream       CUDA stream for the operation.
 */
void nvte_copy_host_to_device_via_kernel(const void *host_ptr, void *device_ptr, size_t num_bytes,
                                         cudaStream_t stream);

/*! \deprecated Use nvte_copy_host_to_device_via_kernel instead.
 *
 *  \brief Copy an array of device pointers (held on host) into a device tensor.
 */
void nvte_convert_pointers_to_tensor(const uint64_t *host_ptrs, NVTETensor output, int64_t count,
                                     cudaStream_t stream);

/*! \brief Convert cumulative Q and KV sequence lengths to individual lengths.
 *
 *  All tensors are contiguous, one-dimensional INT32 device tensors. The inputs
 *  each contain batch_size + 1 cumulative lengths. The outputs have the same
 *  size, at least batch_size, and any extra entries are set to zero.
 *
 *  \param[in]     cu_seqlens_q    Cumulative Q sequence lengths.
 *  \param[in]     cu_seqlens_kv   Cumulative KV sequence lengths.
 *  \param[out]    seqlens_q       Individual Q sequence lengths.
 *  \param[out]    seqlens_kv      Individual KV sequence lengths.
 *  \param[in]     stream          CUDA stream for the operation.
 */
void nvte_cu_seqlens_to_actual_seqlens(const NVTETensor cu_seqlens_q,
                                       const NVTETensor cu_seqlens_kv, NVTETensor seqlens_q,
                                       NVTETensor seqlens_kv, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_UTILS_H_
