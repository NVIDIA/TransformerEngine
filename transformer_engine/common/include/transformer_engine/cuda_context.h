/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cuda_context.h
 *  \brief Explicit release and invalidation of registered CUDA-dependent state.
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

/*! Release TE resources while the device's primary CUDA context is still
 *  usable. This supports in-process restarts which preserve the CUDA context,
 *  and optional cleanup before externally destroying a context. Stop all TE
 *  calls, finish GPU work, retire captured graphs, and release framework-owned
 *  TE resources before calling. Communicators must remain alive through cleanup.
 *  This call does not reset CUDA, framework allocators, model tensors, or RNG.
 *  Operations recreate released resources lazily. Cleanup errors propagate;
 *  after such an error the caller must finish cleanup before resuming work.
 */
void nvte_release_cuda_state(int device);

/*! Process-wide release/acquire notification, advanced after release or reset
 *  mutates state on any device, including a release reporting partial failure.
 *  Comparing this is a cheap way to decide
 *  whether a per-device generation query is needed. It is not a resource pin.
 */
uint64_t nvte_get_cuda_state_version(void);

/*! Callbacks for framework bindings which own context-dependent state. All
 *  callbacks run at an externally serialized lifecycle boundary; they must not
 *  register new resources or re-enter lifecycle APIs. Check callbacks must not
 *  mutate state. C++ check/release callbacks may throw; invalidate must not.
 *  Release order is ascending (plans precede library handles and streams).
 */
typedef void (*NVTECudaStateCallback)(void *user_data);
typedef struct {
  NVTECudaStateCallback check_context_loss;
  NVTECudaStateCallback check_release;
  NVTECudaStateCallback release;
  NVTECudaStateCallback invalidate;
  int release_order;
} NVTECudaStateCallbacks;
typedef struct NVTECudaStateRegistrationImpl *NVTECudaStateRegistration;

/*! Register stable user_data without querying CUDA. Keep it alive until
 *  unregister returns. Registration/destruction is serialized with TE calls.
 *  Callbacks may unregister other resources they destroy during cleanup.
 */
NVTECudaStateRegistration nvte_register_cuda_state(int device, NVTECudaStateCallbacks callbacks,
                                                   void *user_data);
void nvte_unregister_cuda_state(NVTECudaStateRegistration registration);

/*! Host-only resource generation for a device, initially one. Release/reset
 *  publishes its new generation after cleanup. A generation change
 *  invalidates externally cached TE stream/event handles too. Reading a
 *  generation does not pin resources or permit racing reset with TE operations.
 */
uint64_t nvte_get_cuda_context_generation(int device);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CUDA_CONTEXT_H_
