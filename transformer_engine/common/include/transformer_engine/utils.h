/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file utils.h
 *  \brief Utility functions (e.g. host-to-device pointer copies).
 */

#ifndef TRANSFORMER_ENGINE_UTILS_H_
#define TRANSFORMER_ENGINE_UTILS_H_

#include <cuda_runtime.h>
#include <stdint.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Copy an array of device pointers (held on host) into a device tensor.
 *
 *  \param[in]     host_ptrs    Host array of device pointer values cast to uint64_t.
 *  \param[out]    output       NVTETensor whose rowwise data buffer receives the pointer values.
 *  \param[in]     count        Number of pointers.
 *  \param[in]     stream       CUDA stream used for the operation.
 */
void nvte_convert_pointers_to_tensor(const uint64_t *host_ptrs, NVTETensor output, int64_t count,
                                     cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_UTILS_H_
