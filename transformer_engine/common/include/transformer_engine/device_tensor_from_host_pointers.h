/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file device_tensor_from_host_pointers.h
 *  \brief Function to copy host-side device pointers into a device tensor.
 */

#ifndef TRANSFORMER_ENGINE_DEVICE_TENSOR_FROM_HOST_POINTERS_H_
#define TRANSFORMER_ENGINE_DEVICE_TENSOR_FROM_HOST_POINTERS_H_

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Copy an array of device pointers (held on host) into a device buffer.
 *
 *  \param[in]     host_ptrs    Host array of device pointer values cast to int64_t.
 *  \param[out]    output       Device buffer to write pointer values into.
 *  \param[in]     count        Number of pointers.
 *  \param[in]     stream       CUDA stream used for the operation.
 */
void nvte_convert_pointers_to_tensor(const int64_t *host_ptrs, int64_t *output, int64_t count,
                                     cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_DEVICE_TENSOR_FROM_HOST_POINTERS_H_
