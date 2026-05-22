/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_api.cpp
 *  \brief nvte_ep_* C API: thin delegations to the EPBackend singleton.
 */

#include <nccl.h>
#include <transformer_engine/ep.h>

#include "../common.h"
#include "../util/logging.h"
#include "ep_backend.h"

using transformer_engine::ep::EPBackend;

void nvte_ep_initialize(void* ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  EPBackend::initialize(static_cast<ncclComm_t>(ep_comm), group_config);
}

void nvte_ep_shutdown(void) { EPBackend::shutdown(); }

uint64_t nvte_ep_register_layer(NVTEEpLayerConfig layer_config, size_t* handle_mem_size) {
  NVTE_CHECK(handle_mem_size != nullptr, "handle_mem_size must not be null");
  return EPBackend::get().register_layer(layer_config, handle_mem_size);
}

void nvte_ep_prepare(NVTEEpHandle handle, NVTETensor topk_idx, NVTETensor token_counts,
                     size_t dispatch_output_per_expert_alignment, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle.mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().prepare(handle.id, topk_idx, token_counts, mem_ptr,
                           dispatch_output_per_expert_alignment, stream);
}

void nvte_ep_dispatch(NVTEEpHandle handle, NVTETensor topk_idx, NVTETensor tokens,
                      NVTECommWindow tokens_win, NVTETensor topk_weights,
                      NVTECommWindow topk_weights_win, NVTETensor recv_tokens,
                      NVTECommWindow recv_tokens_win, NVTETensor recv_topk_weights,
                      NVTECommWindow recv_topk_weights_win, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle.mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().dispatch(handle.id, mem_ptr, topk_idx, tokens, tokens_win, topk_weights,
                            topk_weights_win, recv_tokens, recv_tokens_win, recv_topk_weights,
                            recv_topk_weights_win, stream);
}

void nvte_ep_combine(NVTEEpHandle handle, NVTETensor expert_out, NVTECommWindow expert_out_win,
                     NVTETensor result, cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle.mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().combine(handle.id, mem_ptr, expert_out, expert_out_win, result, stream);
}

void nvte_ep_dispatch_bwd(NVTEEpHandle handle, NVTETensor grad, NVTECommWindow grad_win,
                          NVTETensor g_recv_topk_weights, NVTECommWindow g_recv_topk_weights_win,
                          NVTETensor grad_tokens, NVTETensor grad_topk_weights,
                          cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle.mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().dispatch_bwd(handle.id, mem_ptr, grad, grad_win, g_recv_topk_weights,
                                g_recv_topk_weights_win, grad_tokens, grad_topk_weights, stream);
}

void nvte_ep_combine_bwd(NVTEEpHandle handle, NVTETensor grad, NVTECommWindow grad_win,
                         NVTETensor grad_expert_out, NVTECommWindow grad_expert_out_win,
                         cudaStream_t stream) {
  void* mem_ptr = nvte_tensor_data(handle.mem);
  NVTE_CHECK(mem_ptr != nullptr, "handle_mem tensor data must not be null");
  EPBackend::get().combine_bwd(handle.id, mem_ptr, grad, grad_win, grad_expert_out,
                               grad_expert_out_win, stream);
}
