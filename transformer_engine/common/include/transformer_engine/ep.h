/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep.h
 *  \brief Public C API for Expert Parallelism. Per-step ops are
 *         allocation-free and CUDA graph-capturable.
 *
 *  Per layer: call nvte_ep_handle_mem_size(layer_cfg) for the buffer size;
 *  allocate handle_mem as a kByte NVTETensor. Per step: nvte_ep_prepare seeds
 *  routing, then nvte_ep_dispatch / nvte_ep_combine / _bwd consume it.
 *  Cache cap: NVTE_EP_HANDLE_CACHE_SIZE (default 4096; -1 disables eviction).
 */

#ifndef TRANSFORMER_ENGINE_EP_H_
#define TRANSFORMER_ENGINE_EP_H_

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>
#include <transformer_engine/comm_window.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -- Config structs ------------------------------------------------------- */
/* Each config begins with struct_size so the API can add fields without
 * breaking ABI. The backend reads only the bytes struct_size covers and
 * zero-defaults the rest; struct_size 0 means the base layout. Append new
 * fields at the end only; never reorder, resize, or remove existing ones. */

/*! \brief Group-level EP configuration (fixed for the EP group lifetime). */
typedef struct {
  /*! Struct size in bytes, or 0 for the base layout. Set to
   *  sizeof(NVTEEpGroupConfig) to include fields added in newer versions. */
  size_t struct_size;
  /*! EP world size. */
  int ep_size;
  /*! Total experts across all ranks. */
  int num_experts;
  /*! Upper bound on tokens this rank sends per dispatch. */
  int max_tokens_per_rank;
  /*! Upper bound on tokens this rank receives per dispatch (must be > 0). */
  int max_recv_tokens_per_rank;
  /*! Token hidden dimension. */
  int hidden_dim;
  /*! Max SMs for NCCL EP dispatch/combine kernels. 0 = auto. */
  int num_comm_sms;
  /*! Widest token dtype the group will dispatch; sizes staging buffers.
   *  Required (no default): must be set to a real token dtype. Per-dispatch
   *  tensors may use any dtype with element size <= this. */
  NVTEDType max_token_dtype;
  /*! Zero-copy dispatch/combine. When nonzero, payload tensors must be backed
   *  by NVTECommWindow handles and transfer in place (no staging copies);
   *  0 (default) = staged. */
  int zero_copy;
} NVTEEpGroupConfig;

/*! \brief Per-layer configuration consumed by nvte_ep_handle_mem_size and
 *         nvte_ep_prepare. Reserved for future per-call options (fp8 scale,
 *         overflow policy, ...).
 */
typedef struct {
  /*! Struct size in bytes, or 0 for the base layout. Set to
   *  sizeof(NVTEEpLayerConfig) to include fields added in newer versions. */
  size_t struct_size;
  /*! Per-token expert fan-out (> 0). */
  int top_k;
  /*! Per-expert recv-slab alignment in tokens (power of two; 0/1 disables).
   *  When > 1, each expert's slab in recv_tokens is zero-padded up to a
   *  multiple of this for downstream per-expert GEMM alignment. */
  size_t dispatch_output_per_expert_alignment;
} NVTEEpLayerConfig;

/* Zero-init a config with struct_size set to the current layout:
 *   NVTEEpGroupConfig cfg = NVTE_EP_GROUP_CONFIG_INIT;
 *   cfg.ep_size = ...; */
#define NVTE_EP_GROUP_CONFIG_INIT {sizeof(NVTEEpGroupConfig)}
#define NVTE_EP_LAYER_CONFIG_INIT {sizeof(NVTEEpLayerConfig)}

/* -- Bootstrap ------------------------------------------------------------ */

/*! \brief Bootstrap the EP backend from an existing NCCL EP sub-communicator.
 *         Requires SM>=90.
 *
 *  ep_comm is borrowed and must span exactly group_config.ep_size ranks. The
 *  caller retains ownership and must keep it alive until nvte_ep_shutdown()
 *  returns. Re-init after shutdown is allowed; double-init throws. One EP
 *  group per process, bound to the current CUDA device.
 *
 *  \param[in] ep_comm      Opaque ncclComm_t for the EP sub-group.
 *  \param[in] group_config Group-level EP configuration (struct_size set).
 */
void nvte_ep_initialize(void* ep_comm, const NVTEEpGroupConfig* group_config);

/*! \brief Tear down the EP backend. Idempotent. Does not destroy ep_comm. */
void nvte_ep_shutdown(void);

/* -- Layer sizing (host-only) --------------------------------------------- */

/*! \brief Report the handle_mem byte size required for the given layer config.
 *
 *  handle_mem is a per-layer kByte routing-state buffer; allocate once and
 *  thread the same pointer through every prepare/dispatch/combine/_bwd call
 *  for that layer (the backend keys its cache on the pointer). Host-only;
 *  size is stable for a given (group, layer) pair.
 *
 *  \param[in] layer_cfg  Per-call layer configuration (struct_size set).
 *  \return size in bytes for the handle_mem buffer.
 */
size_t nvte_ep_handle_mem_size(const NVTEEpLayerConfig* layer_cfg);

/* -- Per-step ops (all allocation-free, CUDA graph-capturable) ------------ */

/*! \brief Seed handle_mem with this step's routing plan.
 *
 *  AllGathers topk_idx across the EP group and stages per-expert offsets and
 *  counts into handle_mem so the matching dispatch/combine/_bwd can run with
 *  no further routing computation. Must precede every dispatch/combine/_bwd
 *  that uses this handle_mem. recv_tokens_per_expert becomes host-valid after a
 *  stream sync.
 *
 *  \param[in]     handle_mem                 uint8 routing-state buffer.
 *  \param[in]     topk_idx                   [T, top_k] int64 routing indices.
 *  \param[out]    recv_tokens_per_expert     [num_local_experts] int32 counts.
 *  \param[out]    total_recv_tokens_per_rank Reserved placeholder; may be null. Unused for now.
 *  \param[in]     layer_cfg                  Per-call layer configuration (struct_size set).
 *  \param[in]     stream                     CUDA stream.
 */
void nvte_ep_prepare(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor recv_tokens_per_expert,
                     NVTETensor total_recv_tokens_per_rank, const NVTEEpLayerConfig* layer_cfg,
                     cudaStream_t stream);

/*! \brief Dispatch tokens (and routing weights) to expert ranks.
 *
 *  Each local token is sent to its top_k destinations; recv_tokens is laid out
 *  expert-major (contiguous per-expert slabs, padded per layer_cfg). The
 *  *_win arguments enable zero-copy via symmem windows; pass NVTECommWindow{}
 *  when unused. Requires a prior nvte_ep_prepare on this handle_mem.
 *
 *  \param[in]     handle_mem             uint8 routing-state buffer (from prepare).
 *  \param[in]     topk_idx               [T, top_k] int64 sparse routing indices.
 *  \param[in]     tokens                 [T, hidden_dim] input tokens.
 *  \param[in]     tokens_win             Optional symmem window for tokens.
 *  \param[in]     topk_weights           [T, top_k] float32 weights, or null in backward.
 *  \param[in]     topk_weights_win       Optional symmem window for topk_weights.
 *  \param[out]    recv_tokens            [recv_T, hidden_dim] received tokens.
 *  \param[in]     recv_tokens_win        Optional symmem window for recv_tokens.
 *  \param[out]    recv_topk_weights      [recv_T] float32 per-slot weights, or null in backward.
 *  \param[in]     recv_topk_weights_win  Optional symmem window for recv_topk_weights.
 *  \param[in]     stream                 CUDA stream.
 */
void nvte_ep_dispatch(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor tokens,
                      NVTECommWindow tokens_win, NVTETensor topk_weights,
                      NVTECommWindow topk_weights_win, NVTETensor recv_tokens,
                      NVTECommWindow recv_tokens_win, NVTETensor recv_topk_weights,
                      NVTECommWindow recv_topk_weights_win, cudaStream_t stream);

/*! \brief Scatter-sum expert outputs back to originating ranks.
 *
 *  Inverse of dispatch: the top_k destination slots for token t are summed
 *  into result[t]. Sums are unweighted; pre-scale expert_out by
 *  recv_topk_weights (and the valid-slot mask) before calling. Requires a
 *  prior nvte_ep_prepare on this handle_mem.
 *
 *  \param[in]  handle_mem      uint8 routing-state buffer (from prepare).
 *  \param[in]  expert_out      [recv_T, hidden_dim] pre-weighted expert outputs.
 *  \param[in]  expert_out_win  Optional symmem window for expert_out.
 *  \param[out] result          [T, hidden_dim] combined output.
 *  \param[in]  stream          CUDA stream.
 */
void nvte_ep_combine(NVTETensor handle_mem, NVTETensor expert_out, NVTECommWindow expert_out_win,
                     NVTETensor result, cudaStream_t stream);

/*! \brief Backward of dispatch: route per-recv-slot grads back to source.
 *
 *  Sums the top_k recv-slot grads into grad_tokens[t]; scatters per-slot
 *  recv-weight grads into grad_topk_weights[t, k]. Padded recv slots
 *  contribute nothing. Requires a prior nvte_ep_prepare on this handle_mem.
 *
 *  \param[in]  handle_mem               uint8 routing-state buffer (from prepare).
 *  \param[in]  grad                     [recv_capacity, hidden_dim] grad w.r.t. recv_tokens.
 *  \param[in]  grad_win                 Optional symmem window for grad.
 *  \param[in]  g_recv_topk_weights      [recv_capacity] f32 grad w.r.t. recv_topk_weights.
 *  \param[in]  g_recv_topk_weights_win  Optional symmem window for g_recv_topk_weights.
 *  \param[out] grad_tokens              [T, hidden_dim] grad w.r.t. tokens.
 *  \param[out] grad_topk_weights        [T, top_k] f32 grad w.r.t. topk_weights.
 *  \param[in]  stream                   CUDA stream.
 */
void nvte_ep_dispatch_bwd(NVTETensor handle_mem, NVTETensor grad, NVTECommWindow grad_win,
                          NVTETensor g_recv_topk_weights, NVTECommWindow g_recv_topk_weights_win,
                          NVTETensor grad_tokens, NVTETensor grad_topk_weights,
                          cudaStream_t stream);

/*! \brief Backward of combine: replicate each source-token grad to its recv
 *         slots from the forward.
 *
 *  Padded recv slots in grad_expert_out are zeroed. Requires a prior
 *  nvte_ep_prepare on this handle_mem.
 *
 *  \param[in]  handle_mem           uint8 routing-state buffer (from prepare).
 *  \param[in]  grad                 [T, hidden_dim] grad w.r.t. result.
 *  \param[in]  grad_win             Optional symmem window for grad.
 *  \param[out] grad_expert_out      [recv_capacity, hidden_dim] grad w.r.t. expert_out.
 *  \param[in]  grad_expert_out_win  Optional symmem window for grad_expert_out.
 *  \param[in]  stream               CUDA stream.
 */
void nvte_ep_combine_bwd(NVTETensor handle_mem, NVTETensor grad, NVTECommWindow grad_win,
                         NVTETensor grad_expert_out, NVTECommWindow grad_expert_out_win,
                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_EP_H_
