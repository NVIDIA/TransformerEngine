/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nccl_comm.h
 *  \brief NCCL-backed peer-handle setter/getter for NVTETensor.
 *
 *  Attaches a registered NCCL symmetric-memory window + byte offset onto a
 *  tensor so consumers (e.g. the EP backend) can issue one-sided put/get over
 *  the window instead of staging through the raw data pointer. The window is
 *  caller-owned; ``attach`` does not register or rendezvous it.
 */

#ifndef TRANSFORMER_ENGINE_NCCL_COMM_H_
#define TRANSFORMER_ENGINE_NCCL_COMM_H_

#include "comm_handle.h"
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Attach an NCCL window + byte offset to ``t``.
 *
 *  Sets the tensor's peer-handle kind to ``NVTE_PEER_HANDLE_NCCL_WINDOW``. The
 *  window must stay registered for the tensor's lifetime. Pass ``window=NULL``
 *  to detach (equivalent to ``nvte_tensor_detach_peer_handle``).
 *
 *  \param[in,out] t      Tensor to annotate.
 *  \param[in]     window Opaque ncclWindow_t (caller-owned), or NULL to clear.
 *  \param[in]     offset Byte offset into the window where this tensor starts.
 */
void nvte_tensor_attach_nccl_window(NVTETensor t, void* window, uint64_t offset);

/*! \brief Read the NCCL window + offset attached to ``t``.
 *
 *  ``*window`` is set to ``NULL`` and ``*offset`` to 0 when no NCCL window is
 *  attached (including when the tensor carries a different peer-handle kind).
 *  Either out-pointer may be ``NULL`` to skip that field.
 */
void nvte_tensor_nccl_window(const NVTETensor t, void** window, uint64_t* offset);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_NCCL_COMM_H_
