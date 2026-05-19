/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/comm_handle.h"

#include "transformer_engine/nccl_comm.h"

#include "common.h"
#include "util/logging.h"

using transformer_engine::convertNVTETensor;

NVTEPeerHandleKind nvte_tensor_peer_handle_kind(const NVTETensor t) {
  const auto* tensor = convertNVTETensor(t);
  return tensor != nullptr ? tensor->peer_handle_kind : NVTE_PEER_HANDLE_NONE;
}

void nvte_tensor_detach_peer_handle(NVTETensor t) {
  auto* tensor = convertNVTETensor(t);
  if (tensor == nullptr) return;
  tensor->peer_handle_kind = NVTE_PEER_HANDLE_NONE;
  tensor->peer_handle_data = nullptr;
  tensor->peer_handle_offset = 0;
}

void nvte_tensor_attach_nccl_window(NVTETensor t, void* window, uint64_t offset) {
  auto* tensor = convertNVTETensor(t);
  NVTE_CHECK(tensor != nullptr, "nvte_tensor_attach_nccl_window: invalid NVTETensor handle");
  if (window == nullptr) {
    tensor->peer_handle_kind = NVTE_PEER_HANDLE_NONE;
    tensor->peer_handle_data = nullptr;
    tensor->peer_handle_offset = 0;
    return;
  }
  tensor->peer_handle_kind = NVTE_PEER_HANDLE_NCCL_WINDOW;
  tensor->peer_handle_data = window;
  tensor->peer_handle_offset = offset;
}

void nvte_tensor_nccl_window(const NVTETensor t, void** window, uint64_t* offset) {
  const auto* tensor = convertNVTETensor(t);
  const bool has_nccl =
      tensor != nullptr && tensor->peer_handle_kind == NVTE_PEER_HANDLE_NCCL_WINDOW;
  if (window != nullptr) *window = has_nccl ? tensor->peer_handle_data : nullptr;
  if (offset != nullptr) *offset = has_nccl ? tensor->peer_handle_offset : 0;
}
