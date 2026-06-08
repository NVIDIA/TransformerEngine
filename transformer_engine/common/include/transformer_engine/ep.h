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
/* TODO: add a struct_size/version field to these configs (and align with other
 *       TE public structs) once a TE-wide convention for ABI versioning lands. */

/*! \brief Group-level EP configuration (fixed for the EP group lifetime). */
typedef struct {
  int ep_size;             /*!< EP world size. */
  int num_experts;         /*!< Total experts across all ranks. */
  int max_tokens_per_rank; /*!< Upper bound on tokens this rank sends per dispatch. */
  /*! Upper bound on tokens received per dispatch (worst-case top_k fan-out; must be > 0). */
  int max_recv_tokens_per_rank;
  int hidden_dim;  /*!< Token hidden dimension. */
  int max_num_sms; /*!< Max SMs for EP kernels. 0 = auto. */
  /*! Widest token dtype the group will dispatch. Sizes NCCL EP staging buffers
   *  at group create. Tensors passed to nvte_ep_dispatch may use any dtype whose
   *  element size is <= sizeof(max_token_dtype). */
  NVTEDType max_token_dtype;
} NVTEEpGroupConfig;

/*! \brief Per-layer configuration consumed by nvte_ep_handle_mem_size and
 *         nvte_ep_prepare. Reserved for future per-call options (fp8 scale,
 *         overflow policy, ...).
 */
typedef struct {
  int top_k; /*!< Per-token expert fan-out (> 0). */
  /*! Per-expert zone alignment in tokens (pow2; 0/1 = none). */
  size_t dispatch_output_per_expert_alignment;
} NVTEEpLayerConfig;

/* -- Bootstrap ------------------------------------------------------------ */

/*! \brief Bootstrap from an existing NCCL EP sub-communicator. Requires SM>=90.
 *
 *  ep_comm is borrowed and must span exactly group_config.ep_size ranks.
 *  The caller retains ownership and must keep ep_comm alive until
 *  nvte_ep_shutdown() returns; destroying it earlier is undefined behavior.
 *  Re-init after shutdown is allowed; double-init throws.
 *
 *  One EP group per process, bound to the current CUDA device at initialize
 *  time. Multiple GPUs per process are not supported.
 *
 *  \param[in] ep_comm      Opaque ncclComm_t for the EP sub-group.
 *  \param[in] group_config Group-level EP configuration.
 */
void nvte_ep_initialize(void* ep_comm, NVTEEpGroupConfig group_config);

/*! \brief Tear down the EP backend. Idempotent. Does not destroy ep_comm. */
void nvte_ep_shutdown(void);

/* -- Layer sizing (host-only) --------------------------------------------- */

/*! \brief Report the handle_mem byte size required for the given layer config.
 *         Host-only; cheap to call. The caller allocates the buffer and passes
 *         it back to ep ops as the handle_mem argument. top_k comes from the
 *         active NVTEEpGroupConfig.
 *
 *  \param[in] layer_cfg  Per-call layer configuration.
 *  \return size in bytes for the handle_mem buffer.
 */
size_t nvte_ep_handle_mem_size(NVTEEpLayerConfig layer_cfg);

/* -- Per-step ops (all allocation-free, CUDA graph-capturable) ------------ */

/*! \brief AllGather the routing map; write per-expert counts and cache routing
 *         metadata in handle_mem for the subsequent dispatch/combine.
 *
 *  \param[in]     handle_mem    uint8 routing-state buffer.
 *  \param[in]     topk_idx      [T, top_k] int64 routing indices.
 *  \param[out]    token_counts  [num_local_experts] int32 counts.
 *  \param[in]     layer_cfg     Per-call layer configuration.
 *  \param[in]     stream        CUDA stream.
 */
void nvte_ep_prepare(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor token_counts,
                     NVTEEpLayerConfig layer_cfg, cudaStream_t stream);

/*! \brief Dispatch tokens (and routing weights) to expert ranks. Requires a
 *         prior nvte_ep_prepare.
 *
 *  \param[in]     handle_mem             uint8 routing-state buffer (from prepare).
 *  \param[in]     topk_idx               [T, top_k] int64 sparse routing indices.
 *  \param[in]     tokens                 [T, hidden_dim] input tokens.
 *  \param[in]     tokens_win             Optional symmem window for ``tokens``.
 *  \param[in]     topk_weights           [T, top_k] float32 weights, or null in backward.
 *  \param[in]     topk_weights_win       Optional symmem window for ``topk_weights``.
 *  \param[out]    recv_tokens            [recv_T, hidden_dim] received tokens.
 *  \param[in]     recv_tokens_win        Optional symmem window for ``recv_tokens``.
 *  \param[out]    recv_topk_weights      [recv_T] float32 per-slot weights, or null in backward.
 *  \param[in]     recv_topk_weights_win  Optional symmem window for ``recv_topk_weights``.
 *  \param[in]     stream                 CUDA stream.
 */
void nvte_ep_dispatch(NVTETensor handle_mem, NVTETensor topk_idx, NVTETensor tokens,
                      NVTECommWindow tokens_win, NVTETensor topk_weights,
                      NVTECommWindow topk_weights_win, NVTETensor recv_tokens,
                      NVTECommWindow recv_tokens_win, NVTETensor recv_topk_weights,
                      NVTECommWindow recv_topk_weights_win, cudaStream_t stream);

/*! \brief Scatter-sum expert outputs back to originating ranks. Unweighted;
 *         caller must pre-multiply expert_out by recv_topk_weights (and the
 *         valid-slot mask) before calling. Requires a prior nvte_ep_prepare.
 *
 *  \param[in]  handle_mem      uint8 routing-state buffer (from prepare).
 *  \param[in]  expert_out      [recv_T, hidden_dim] pre-weighted expert outputs.
 *  \param[in]  expert_out_win  Optional symmem window for ``expert_out``.
 *  \param[out] result          [T, hidden_dim] combined output.
 *  \param[in]  stream          CUDA stream.
 */
void nvte_ep_combine(NVTETensor handle_mem, NVTETensor expert_out, NVTECommWindow expert_out_win,
                     NVTETensor result, cudaStream_t stream);

/*! \brief Backward of dispatch; routes token and weight grads back to source.
 *         Requires a prior nvte_ep_prepare.
 *
 *  \param[in]  handle_mem               uint8 routing-state buffer (from prepare).
 *  \param[in]  grad                     [recv_capacity, hidden_dim] grad w.r.t. recv_tokens.
 *  \param[in]  grad_win                 Optional symmem window for ``grad``.
 *  \param[in]  g_recv_topk_weights      [recv_capacity] f32 grad w.r.t. recv_topk_weights.
 *  \param[in]  g_recv_topk_weights_win  Optional symmem window for ``g_recv_topk_weights``.
 *  \param[out] grad_tokens              [T, hidden_dim] grad w.r.t. tokens.
 *  \param[out] grad_topk_weights        [T, top_k] f32 grad w.r.t. topk_weights.
 *  \param[in]  stream                   CUDA stream.
 */
void nvte_ep_dispatch_bwd(NVTETensor handle_mem, NVTETensor grad, NVTECommWindow grad_win,
                          NVTETensor g_recv_topk_weights, NVTECommWindow g_recv_topk_weights_win,
                          NVTETensor grad_tokens, NVTETensor grad_topk_weights,
                          cudaStream_t stream);

/*! \brief Backward of combine. Padded slots in grad_expert_out are zeroed.
 *         Requires a prior nvte_ep_prepare.
 *
 *  \param[in]  handle_mem           uint8 routing-state buffer (from prepare).
 *  \param[in]  grad                 [T, hidden_dim] grad w.r.t. result.
 *  \param[in]  grad_win             Optional symmem window for ``grad``.
 *  \param[out] grad_expert_out      [recv_capacity, hidden_dim] grad w.r.t. expert_out.
 *  \param[in]  grad_expert_out_win  Optional symmem window for ``grad_expert_out``.
 *  \param[in]  stream               CUDA stream.
 */
void nvte_ep_combine_bwd(NVTETensor handle_mem, NVTETensor grad, NVTECommWindow grad_win,
                         NVTETensor grad_expert_out, NVTECommWindow grad_expert_out_win,
                         cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_EP_H_
