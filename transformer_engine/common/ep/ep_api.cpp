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

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "../util/logging.h"

#if defined(NVTE_WITH_NCCL_EP)

#include <nccl.h>

#include "../common.h"
#include "ep_backend.h"

using transformer_engine::ep::EPBackend;

namespace {
// Smallest accepted struct_size: covers the base (required) fields. Frozen;
// never raise these. Later fields are read only when struct_size covers them.
constexpr size_t kGroupConfigMinSize = offsetof(NVTEEpGroupConfig, zero_copy) + sizeof(int);
constexpr size_t kLayerConfigMinSize =
    offsetof(NVTEEpLayerConfig, dispatch_output_per_expert_alignment) + sizeof(size_t);

// Copy a caller's versioned config into a full current-layout struct: fields
// the caller did not provide stay zero, extra trailing fields are dropped.
// struct_size 0 is read as the base layout. Requires a size_t struct_size
// first member.
template <typename Cfg>
Cfg normalize_ep_config(const Cfg* user, size_t min_size, const char* name) {
  NVTE_CHECK(user != nullptr, name, " must not be null");
  const size_t want = (user->struct_size == 0) ? min_size : user->struct_size;
  NVTE_CHECK(want >= min_size, name, ".struct_size (", user->struct_size,
             ") is below the required minimum ", min_size,
             "; zero-initialize the struct or set struct_size via NVTE_EP_*_CONFIG_INIT");
  Cfg cfg{};
  std::memcpy(&cfg, user, std::min(want, sizeof(Cfg)));
  cfg.struct_size = sizeof(Cfg);
  return cfg;
}

inline void* handle_mem_ptr(NVTETensor mem) {
  void* p = nvte_tensor_data(mem);
  NVTE_CHECK(p != nullptr, "handle_mem tensor data must not be null");
  return p;
}
}  // namespace

void nvte_ep_initialize(void* ep_comm, const NVTEEpGroupConfig* group_config) {
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");
  NVTEEpGroupConfig cfg = normalize_ep_config(group_config, kGroupConfigMinSize, "group_config");
  EPBackend::initialize(static_cast<ncclComm_t>(ep_comm), cfg);
}

void nvte_ep_shutdown(void) { EPBackend::shutdown(); }

size_t nvte_ep_handle_mem_size(const NVTEEpLayerConfig* layer_cfg) {
  NVTEEpLayerConfig cfg = normalize_ep_config(layer_cfg, kLayerConfigMinSize, "layer_cfg");
  return EPBackend::get().handle_mem_size(cfg);
}

void nvte_ep_prepare(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor recv_tokens_per_expert,
                     NVTETensor total_recv_tokens_per_rank, const NVTEEpLayerConfig* layer_cfg,
                     cudaStream_t stream) {
  NVTEEpLayerConfig cfg = normalize_ep_config(layer_cfg, kLayerConfigMinSize, "layer_cfg");
  EPBackend::get().prepare(handle_mem_ptr(handle_mem), topk_idx, recv_tokens_per_expert,
                           total_recv_tokens_per_rank, cfg, stream);
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

#else  // !NVTE_WITH_NCCL_EP - throwing stubs.

namespace {
[[noreturn]] void ep_not_built() {
  NVTE_ERROR(
      "NCCL EP is not built into this TransformerEngine. Rebuild TE with "
      "NVTE_WITH_NCCL_EP=1 and CUDA arch >= 90 (e.g. NVTE_CUDA_ARCHS=\"90\").");
}
}  // namespace

void nvte_ep_initialize(void* /*ep_comm*/, const NVTEEpGroupConfig* /*group_config*/) {
  ep_not_built();
}

void nvte_ep_shutdown(void) {}

size_t nvte_ep_handle_mem_size(const NVTEEpLayerConfig* /*layer_cfg*/) { ep_not_built(); }

void nvte_ep_prepare(NVTETensor /*handle_mem*/, NVTETensor /*topk_idx*/,
                     NVTETensor /*recv_tokens_per_expert*/,
                     NVTETensor /*total_recv_tokens_per_rank*/,
                     const NVTEEpLayerConfig* /*layer_cfg*/, cudaStream_t /*stream*/) {
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
