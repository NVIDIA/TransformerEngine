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
 * Creation does not allocate device memory. The first grouped MXFP8 operation needing
 * tensor-map descriptors allocates storage for its group count. Descriptor contents
 * are refreshed for each operation because tensor addresses and shapes may change.
 * A handle is bound to the CUDA context and device on which it first allocates
 * descriptors. Operations using the same handle must be ordered in one stream, or explicitly
 * synchronized by the caller across streams.
 * Access from multiple host threads must also be serialized. Distinct handles may be used
 * concurrently. Keep the handle alive while captured CUDA graphs use it.
 */
typedef struct NVTEResourceHandleOpaque *NVTEResourceHandle;

/*! \brief Create an empty resource handle without allocating device memory. */
NVTEResourceHandle nvte_create_resource_handle(void);

/*! \brief Reserve grouped TMA descriptor storage before CUDA graph capture.
 *
 * Optional: a grouped operation will reserve storage lazily if needed. Growing storage
 * during stream capture is unsupported; call this before capture when the group count
 * is known. Passing zero is a no-op.
 */
void nvte_resource_handle_reserve_grouped_tma(NVTEResourceHandle handle, size_t num_tensors,
                                               cudaStream_t stream);

/*! \brief Release resources in the bound CUDA context. All submitted uses must be complete. */
void nvte_destroy_resource_handle(NVTEResourceHandle handle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_RESOURCE_HANDLE_H_
