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

/*! \brief Extract a CUDA generator's seed and offset into a device RNG-state buffer.
 *
 *  When a CUDA graph is being captured, the seed and offset are read from device pointers and
 *  the graph-local offset is added. Otherwise the provided host values are stored directly.
 *
 *  \param[out]    rng_state_ptr       A two-element device array containing seed and offset.
 *  \param[in]     captured            Whether CUDA graph capture is active.
 *  \param[in]     seed_ptr            Device pointer to the seed used during capture.
 *  \param[in]     seed_val            Seed value used outside capture.
 *  \param[in]     offset_ptr          Device pointer to the offset used during capture.
 *  \param[in]     offset_val          Offset value used outside capture.
 *  \param[in]     offset_intragraph   Offset to add within a captured graph.
 *  \param[in]     stream              CUDA stream for the operation.
 */
void nvte_extract_seed_and_offset(int64_t *rng_state_ptr, int captured, int64_t *seed_ptr,
                                  uint64_t seed_val, int64_t *offset_ptr, uint64_t offset_val,
                                  uint32_t offset_intragraph, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_UTILS_H_
