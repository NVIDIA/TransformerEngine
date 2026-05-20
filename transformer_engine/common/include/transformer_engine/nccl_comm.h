/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nccl_comm.h
 *  \brief Attach a registered NCCL symmetric-memory window to an NVTETensor.
 *
 *  The window is caller-owned and must outlive the tensor; ``attach`` does
 *  not register or rendezvous it.
 */

#ifndef TRANSFORMER_ENGINE_NCCL_COMM_H_
#define TRANSFORMER_ENGINE_NCCL_COMM_H_

#include "comm_handle.h"
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Attach an NCCL window + byte offset to ``t``. Pass ``window=NULL`` to detach.
 *
 *  \param[in,out] t      Tensor to annotate.
 *  \param[in]     window Opaque ncclWindow_t (caller-owned), or NULL to clear.
 *  \param[in]     offset Byte offset into the window where this tensor starts.
 */
void nvte_tensor_attach_nccl_window(NVTETensor t, void* window, uint64_t offset);

/*! \brief Read the NCCL window + offset attached to ``t``; yields (NULL, 0) when unset.
 *         Either out-pointer may be NULL to skip that field.
 */
void nvte_tensor_nccl_window(const NVTETensor t, void** window, uint64_t* offset);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_NCCL_COMM_H_
