/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file comm_handle.h
 *  \brief Generic peer-handle annotation on NVTETensor for one-sided RMA.
 *
 *  The annotation is borrowed; the tensor never owns the underlying resource.
 *  Per-backend setters/getters live in dedicated headers (e.g. ``nccl_comm.h``).
 */

#ifndef TRANSFORMER_ENGINE_COMM_HANDLE_H_
#define TRANSFORMER_ENGINE_COMM_HANDLE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Comm backend that owns a tensor's peer handle. */
typedef enum {
  NVTE_PEER_HANDLE_NONE = 0,
  NVTE_PEER_HANDLE_NCCL_WINDOW = 1,
} NVTEPeerHandleKind;

/*! \brief Peer-handle kind attached to ``t``. */
NVTEPeerHandleKind nvte_tensor_peer_handle_kind(const NVTETensor t);

/*! \brief Clear any peer handle attached to ``t``. */
void nvte_tensor_detach_peer_handle(NVTETensor t);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_COMM_HANDLE_H_
