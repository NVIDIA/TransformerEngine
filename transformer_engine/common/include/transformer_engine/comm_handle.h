/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file comm_handle.h
 *  \brief Generic peer-handle annotation on NVTETensor.
 *
 *  A peer handle is an opaque, comm-backend-specific reference that lets a
 *  consumer initiate one-sided remote-memory operations against a peer's
 *  buffer (e.g. NCCL symmetric-memory window, NVSHMEM pointer, CUDA-IPC
 *  handle). The annotation is borrowed; the resource outlives the call and
 *  the tensor never owns it.
 *
 *  Backends register their setter/getter under a dedicated header (e.g.
 *  ``nccl_comm.h`` for NCCL windows). This header exposes only the kind tag
 *  and detach, both of which are payload-agnostic.
 */

#ifndef TRANSFORMER_ENGINE_COMM_HANDLE_H_
#define TRANSFORMER_ENGINE_COMM_HANDLE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Kind tag identifying which comm backend owns a tensor's peer handle. */
typedef enum {
  NVTE_PEER_HANDLE_NONE = 0,
  NVTE_PEER_HANDLE_NCCL_WINDOW = 1,
  /* Reserved for future backends: NVSHMEM_PTR, CUDA_IPC, UCX_RKEY, ... */
} NVTEPeerHandleKind;

/*! \brief Return the peer-handle kind currently attached to ``t``. */
NVTEPeerHandleKind nvte_tensor_peer_handle_kind(const NVTETensor t);

/*! \brief Clear any peer handle attached to ``t``; no-op when none is set. */
void nvte_tensor_detach_peer_handle(NVTETensor t);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_COMM_HANDLE_H_
