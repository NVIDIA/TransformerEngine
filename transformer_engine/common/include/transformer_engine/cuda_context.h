/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cuda_context.h
 *  \brief Explicit invalidation of common-library state after CUDA context loss.
 */

#ifndef TRANSFORMER_ENGINE_CUDA_CONTEXT_H_
#define TRANSFORMER_ENGINE_CUDA_CONTEXT_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! Invalidate registered TE resources after the primary context of device has
 *  been destroyed externally. The caller must serialize this call with ALL TE
 *  calls, including destruction, in every host thread. The caller also owns
 *  context teardown and recovery of framework allocators and device tensors.
 *
 *  This call does not destroy the context, synchronize the GPU, or
 *  create GPU resources. Host plan descriptors are released via dependency APIs.
 *  Registered resources abandon handles already destroyed
 *  with the context, and are recreated lazily. Calling this while the old
 *  context is alive leaks those resources and is not supported.
 *
 *  Only registered resources participate. See the recovery documentation for
 *  the supported subsystems; this is not a reset of the complete training stack.
 *  Device must be a nonnegative CUDA ordinal, supplied by the caller because
 *  CUDA may be unavailable during recovery. Repeated resets are safe.
 */
void nvte_reset_cuda_state(int device);

/*! Host-only generation for a device, initially one. Reset publishes its new
 *  generation with release ordering after invalidation. A generation change
 *  invalidates externally cached TE stream/event handles too. Reading a
 *  generation does not pin resources or permit racing reset with TE operations.
 */
uint64_t nvte_get_cuda_context_generation(int device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CUDA_CONTEXT_H_
