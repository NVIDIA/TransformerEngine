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

/*! Invalidate all registered TE CUDA-dependent state after external context loss.
 *  Every primary context backing a registered resource must already be destroyed.
 *  This is a process-wide boundary, not selective recovery of one device.
 *  Calling reset with any such context still alive abandons live resources and
 *  is unsupported. Use release while contexts are usable instead.
 *
 *  Stop ALL TE calls and object construction/destruction in every host thread.
 *  The caller owns context teardown/recreation, framework allocator recovery,
 *  and replacement of device tensors and captured graphs. Reset is host-only:
 *  it does not query CUDA, synchronize, or destroy dead stream/event/module
 *  handles. Supported resources are recreated lazily after recovery.
 *  Opaque vendor host-state cleanup may require dependency support.
 */
void nvte_reset_cuda_state(void);

/*! Release all registered TE CUDA-dependent state while its contexts and
 *  borrowed communicators remain usable. Stop all TE calls, finish GPU work,
 *  retire captured graphs, and release frontend-owned TE state first. This is
 *  process-wide and does not destroy CUDA contexts, model tensors, or RNG state.
 *  Device-affine callbacks run with their existing primary context current;
 *  unused devices are not initialized. The calling thread's context is restored.
 *  A callback registered with device=-1 manages its own device selection.
 *  Cleanup errors propagate after remaining callbacks run. Finish cleanup before
 *  resuming work after an error. Resources are recreated lazily on next use.
 */
void nvte_release_cuda_state(void);

/*! Single process-wide generation, initially one. A completed release/reset
 *  publishes an increment with release/acquire semantics. Rejected preflight
 *  does not advance it; a partially failed release does. This host-only query
 *  is a cheap notification, not a resource pin or permission to race cleanup.
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

/*! Register stable user_data in the process-wide registry without querying CUDA.
 *  Device is a live-cleanup affinity, not a reset scope. Use -1 for host-only or
 *  multi-device owners; those callbacks select their own devices when needed.
 *  Keep user_data alive until unregister returns. Registration/destruction is
 *  serialized with lifecycle calls. Cleanup may unregister destroyed children.
 */
NVTECudaStateRegistration nvte_register_cuda_state(int device, NVTECudaStateCallbacks callbacks,
                                                   void *user_data);
void nvte_unregister_cuda_state(NVTECudaStateRegistration registration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CUDA_CONTEXT_H_
