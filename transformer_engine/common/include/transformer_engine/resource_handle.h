/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file resource_handle.h
 *  \brief Optional, caller-owned resources for common operations.
 */

#ifndef TRANSFORMER_ENGINE_RESOURCE_HANDLE_H_
#define TRANSFORMER_ENGINE_RESOURCE_HANDLE_H_

#include <cuda_runtime_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Opaque owner of device resources used by common operations.
 *
 * Creation does not allocate device memory. The first grouped MXFP8 quantization needing
 * tensor-map descriptors allocates storage for its group count. Descriptor contents
 * are refreshed for each quantization because tensor addresses and shapes may change.
 * On first use, the handle binds to the current CUDA context and device.
 * It owns separate descriptor storage for each calling host thread and CUDA stream,
 * so calls from different threads or streams can enqueue work concurrently.
 * Reserve and use storage from the same host thread and stream. Keep the handle
 * alive while submitted kernels, captured graphs, or graph executables use its
 * storage. Do not destroy it concurrently with host calls, and complete all
 * submitted work before destruction.
 */
typedef struct NVTEResourceHandleOpaque *NVTEResourceHandle;

/*! \brief Create an empty resource handle without allocating device memory. */
NVTEResourceHandle nvte_create_resource_handle(void);

/*! \brief Resource selected for explicit reservation.
 *
 * Additional resource types can be added without introducing another reserve API.
 */
typedef enum {
  NVTE_RESOURCE_GROUPED_TMA_DESCRIPTORS = 0,
} NVTEResourceType;

/*! \brief Reserve capacity for one resource type before CUDA graph capture.
 *
 * Reservation is optional: an operation may allocate its resource lazily when
 * needed. For NVTE_RESOURCE_GROUPED_TMA_DESCRIPTORS, capacity is the number of
 * tensor groups. Reserve on each host thread and CUDA stream that will use the
 * handle during capture. Growing storage during stream capture is unsupported.
 * Passing zero is a no-op for a valid resource type. Unknown types are rejected.
 */
void nvte_resource_handle_reserve(NVTEResourceHandle handle, NVTEResourceType resource_type,
                                  size_t capacity, cudaStream_t stream);

/*! \brief Release resources in the bound CUDA context and device.
 *  All submitted uses must be complete. Graphs and graph executables using
 *  the handle must also be destroyed.
 */
void nvte_destroy_resource_handle(NVTEResourceHandle handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_RESOURCE_HANDLE_H_
