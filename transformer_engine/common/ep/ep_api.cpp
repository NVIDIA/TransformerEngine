/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_api.cpp
 *  \brief nvte_ep_* C API: thin delegations to the EPBackend singleton.
 *
 *  When NVTE_WITH_NCCL_EP is undefined, the entry points become throwing
 *  stubs so framework bindings still link without NCCL EP support.
 */

#include <transformer_engine/ep.h>

#include "../util/logging.h"

#if defined(NVTE_WITH_NCCL_EP)

#include <nccl.h>

#include "../common.h"
#include "ep_backend.h"

using transformer_engine::ep::EPBackend;

void nvte_ep_initialize(void* ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  EPBackend::initialize(static_cast<ncclComm_t>(ep_comm), group_config);
}

void nvte_ep_shutdown(void) { EPBackend::shutdown(); }

size_t nvte_ep_handle_mem_size(NVTEEpLayerConfig layer_cfg) {
  return EPBackend::get().handle_mem_size(layer_cfg);
}

namespace {
inline void* handle_mem_ptr(NVTETensor mem) {
  void* p = nvte_tensor_data(mem);
  NVTE_CHECK(p != nullptr, "handle_mem tensor data must not be null");
  return p;
}
}  // namespace

void nvte_ep_prepare(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor token_counts,
                     NVTEEpLayerConfig layer_cfg, cudaStream_t stream) {
  EPBackend::get().prepare(handle_mem_ptr(handle_mem), topk_idx, token_counts, layer_cfg, stream);
}

void nvte_ep_dispatch(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor tokens,
                      NVTECommWindow tokens_win, NVTETensor topk_weights,
                      NVTECommWindow topk_weights_win, NVTETensor recv_tokens,
                      NVTECommWindow recv_tokens_win, NVTETensor recv_topk_weights,
                      NVTECommWindow recv_topk_weights_win, cudaStream_t stream) {
  EPBackend::get().dispatch(handle_mem_ptr(handle_mem), topk_idx, tokens, tokens_win, topk_weights,
                            topk_weights_win, recv_tokens, recv_tokens_win, recv_topk_weights,
                            recv_topk_weights_win, stream);
}

void nvte_ep_combine(NVTETensor handle_mem, NVTETensor expert_out, NVTECommWindow expert_out_win,
                     NVTETensor result, cudaStream_t stream) {
  EPBackend::get().combine(handle_mem_ptr(handle_mem), expert_out, expert_out_win, result, stream);
}

void nvte_ep_dispatch_bwd(NVTETensor handle_mem, NVTETensor grad, NVTECommWindow grad_win,
                          NVTETensor g_recv_topk_weights, NVTECommWindow g_recv_topk_weights_win,
                          NVTETensor grad_tokens, NVTETensor grad_topk_weights,
                          cudaStream_t stream) {
  EPBackend::get().dispatch_bwd(handle_mem_ptr(handle_mem), grad, grad_win, g_recv_topk_weights,
                                g_recv_topk_weights_win, grad_tokens, grad_topk_weights, stream);
}

void nvte_ep_combine_bwd(NVTETensor handle_mem, NVTETensor grad, NVTECommWindow grad_win,
                         NVTETensor grad_expert_out, NVTECommWindow grad_expert_out_win,
                         cudaStream_t stream) {
  EPBackend::get().combine_bwd(handle_mem_ptr(handle_mem), grad, grad_win, grad_expert_out,
                               grad_expert_out_win, stream);
}

#else  // !NVTE_WITH_NCCL_EP — throwing stubs.

namespace {
[[noreturn]] void ep_not_built() {
  NVTE_ERROR(
      "NCCL EP is not built into this TransformerEngine. Rebuild TE with "
      "NVTE_BUILD_WITH_NCCL_EP=1 and CUDA arch >= 90 (e.g. NVTE_CUDA_ARCHS=\"90\").");
}
}  // namespace

void nvte_ep_initialize(void* /*ep_comm*/, NVTEEpGroupConfig /*group_config*/) { ep_not_built(); }

void nvte_ep_shutdown(void) {}

size_t nvte_ep_handle_mem_size(NVTEEpLayerConfig /*layer_cfg*/) { ep_not_built(); }

void nvte_ep_prepare(NVTETensor /*handle_mem*/, NVTETensor /*topk_idx*/,
                     NVTETensor /*token_counts*/, NVTEEpLayerConfig /*layer_cfg*/,
                     cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_dispatch(NVTETensor /*handle_mem*/, NVTETensor /*topk_idx*/, NVTETensor /*tokens*/,
                      NVTECommWindow /*tokens_win*/, NVTETensor /*topk_weights*/,
                      NVTECommWindow /*topk_weights_win*/, NVTETensor /*recv_tokens*/,
                      NVTECommWindow /*recv_tokens_win*/, NVTETensor /*recv_topk_weights*/,
                      NVTECommWindow /*recv_topk_weights_win*/, cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_combine(NVTETensor /*handle_mem*/, NVTETensor /*expert_out*/,
                     NVTECommWindow /*expert_out_win*/, NVTETensor /*result*/,
                     cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_dispatch_bwd(NVTETensor /*handle_mem*/, NVTETensor /*grad*/,
                          NVTECommWindow /*grad_win*/, NVTETensor /*g_recv_topk_weights*/,
                          NVTECommWindow /*g_recv_topk_weights_win*/, NVTETensor /*grad_tokens*/,
                          NVTETensor /*grad_topk_weights*/, cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_combine_bwd(NVTETensor /*handle_mem*/, NVTETensor /*grad*/,
                         NVTECommWindow /*grad_win*/, NVTETensor /*grad_expert_out*/,
                         NVTECommWindow /*grad_expert_out_win*/, cudaStream_t /*stream*/) {
  ep_not_built();
}

#endif  // NVTE_WITH_NCCL_EP
