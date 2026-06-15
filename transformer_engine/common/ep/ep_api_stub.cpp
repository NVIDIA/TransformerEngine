/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_api_stub.cpp
 *  \brief Throwing nvte_ep_* stubs compiled when NVTE_WITH_NCCL_EP=OFF.
 */

#include <transformer_engine/ep.h>

#include "../util/logging.h"

namespace {
[[noreturn]] void ep_not_built() {
  NVTE_ERROR(
      "NCCL EP is not built into this TransformerEngine. Rebuild TE with "
      "NVTE_BUILD_WITH_NCCL_EP=1 and CUDA arch >= 90 (e.g. NVTE_CUDA_ARCHS=\"90\").");
}
}  // namespace

void nvte_ep_initialize(void* /*ep_comm*/, NVTEEpGroupConfig /*group_config*/) { ep_not_built(); }

void nvte_ep_shutdown(void) {}

uint64_t nvte_ep_register_layer(NVTEEpLayerConfig /*layer_config*/, size_t* /*handle_mem_size*/) {
  ep_not_built();
}

void nvte_ep_prepare(NVTEEpHandle /*handle*/, NVTETensor /*topk_idx*/, NVTETensor /*token_counts*/,
                     size_t /*dispatch_output_per_expert_alignment*/, cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_dispatch(NVTEEpHandle /*handle*/, NVTETensor /*topk_idx*/, NVTETensor /*tokens*/,
                      NVTECommWindow /*tokens_win*/, NVTETensor /*topk_weights*/,
                      NVTECommWindow /*topk_weights_win*/, NVTETensor /*recv_tokens*/,
                      NVTECommWindow /*recv_tokens_win*/, NVTETensor /*recv_topk_weights*/,
                      NVTECommWindow /*recv_topk_weights_win*/, cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_combine(NVTEEpHandle /*handle*/, NVTETensor /*expert_out*/,
                     NVTECommWindow /*expert_out_win*/, NVTETensor /*result*/,
                     cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_dispatch_bwd(NVTEEpHandle /*handle*/, NVTETensor /*grad*/, NVTECommWindow /*grad_win*/,
                          NVTETensor /*g_recv_topk_weights*/,
                          NVTECommWindow /*g_recv_topk_weights_win*/, NVTETensor /*grad_tokens*/,
                          NVTETensor /*grad_topk_weights*/, cudaStream_t /*stream*/) {
  ep_not_built();
}

void nvte_ep_combine_bwd(NVTEEpHandle /*handle*/, NVTETensor /*grad*/, NVTECommWindow /*grad_win*/,
                         NVTETensor /*grad_expert_out*/, NVTECommWindow /*grad_expert_out_win*/,
                         cudaStream_t /*stream*/) {
  ep_not_built();
}
