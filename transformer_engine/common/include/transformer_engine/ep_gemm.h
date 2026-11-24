/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_gemm.h
 *  \brief Expert-parallel grouped GEMMs overlapped with the NCCL EP dispatch and combine.
 *
 *  Each call overlaps the grouped GEMMs of one expert-MLP stage with an NCCL EP call: the EP call
 *  signals its progress in chunks and the GEMM follows chunk by chunk. The backward calls also
 *  run the stage's weight-gradient GEMM, weight_grad[e] = grad^T wgrad_input over the rows of
 *  expert e, beside the EP call when weight_grad is given (null skips it).
 *
 *    entry                      GEMMs                          EP call       SMs of the EP call
 *    nvte_ep_dispatch_gemm_fwd  FC1 + SwiGLU                   dispatch      dispatch_fwd_num_sms
 *      [dispatch on its SMs] || [FC1 + SwiGLU on the rest, tile by tile as rows arrive]
 *    nvte_ep_combine_gemm_fwd   FC2                            combine       combine_fwd_num_sms
 *      [FC2 on the rest, chunk by chunk] || [combine on its SMs]
 *    nvte_ep_combine_gemm_bwd   FC2 dgrad + SwiGLU bwd, wgrad  dispatch bwd  dispatch_bwd_num_sms
 *      [FC2 dgrad + SwiGLU bwd (+ wgrad) on the rest] || [dispatch bwd on its SMs]
 *    nvte_ep_dispatch_gemm_bwd  FC1 dgrad, wgrad               combine bwd   combine_bwd_num_sms
 *      [FC1 dgrad (+ wgrad) on the rest] || [combine bwd on its SMs]
 *
 *  The SM counts are NVTEEpGroupConfig fields. A GEMM runs on the SMs its EP call leaves and, in
 *  its tail, on the SMs the EP call frees as it ends. A combine readiness flag spans
 *  1024 hidden columns when that divides hidden_dim, else the whole row.
 *
 *  Each call takes a caller-owned uint8 workspace of nvte_ep_overlap_workspace_size bytes for its
 *  readiness flags and GEMM state; it is read only during the call, so the caller may allocate it
 *  per call or reuse one buffer across the calls of a group.
 *
 *  Requirements: SM 10.0, 10.3 or 10.7, EP within one NVLink domain, num_comm_sms > 0, zero_copy
 *  off, dispatch_output_per_expert_alignment 256, 128 (not MXFP8 on SM 10.7) or (bfloat16 only)
 *  64, and NVTE_ENABLE_CUTEDSL_BACKEND=1 with NVTE_MOE_OVERLAP. MXFP8 needs hidden_dim a multiple
 *  of 512 and ffn_dim a multiple of 256. The first call per shape compiles the GEMM: make it at
 *  warmup, outside CUDA graph capture. Issue the calls of one EP group on one stream, in the same
 *  order on every rank. A call throws when a requirement does not hold.
 *
 *  A symm-mem window (tokens_win, grad_result_win) lets peers read the input in place. Either every
 *  rank passes a window, at the same data and scale offsets, or none does. The input must not be
 *  written until the call's dispatch completes, and then only from a launch with no programmatic
 *  dependency on the stream.
 *
 *  Row-sized tensors have recv_capacity rows, a multiple of the alignment. Under MXFP8: tokens and
 *  received rows carry compact rowwise scales; weights and GEMM inputs carry GEMM-swizzled scales;
 *  the backward reads the weights and wgrad operands columnwise.
 */

#ifndef TRANSFORMER_ENGINE_EP_GEMM_H_
#define TRANSFORMER_ENGINE_EP_GEMM_H_

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <transformer_engine/ep.h>
#include <transformer_engine/transformer_engine.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Dispatch overlapped with the first grouped GEMM and SwiGLU.
 *
 *  Per expert slab e: gemm_out = recv_tokens @ weight[e]^T and
 *  act_out = silu(gate) * up * recv_topk_weights, gate and up being alternating 32-column blocks of
 *  gemm_out. MXFP8 tokens: tokens, recv_tokens, weight and act_out are MXFP8 and act_out is
 *  silu(gate) * up, the routing weights left to nvte_ep_combine_gemm_fwd.
 *
 *  \param[in,out] handle_mem             uint8 routing-state buffer.
 *  \param[in]     topk_idx               [T, top_k] int32 or int64 routing indices.
 *  \param[in]     tokens                 [T, hidden_dim] input tokens.
 *  \param[in]     tokens_win             Optional symm-mem window for tokens (MXFP8: and, in
 *                                        scale_window, for its scales); NVTECommWindow{} stages
 *                                        them.
 *  \param[in]     topk_weights           [T, top_k] float32 routing weights.
 *  \param[in]     weight                 [num_local_experts, 2 * ffn_dim, hidden_dim] weights.
 *  \param[out]    recv_tokens            [recv_capacity, hidden_dim] received tokens.
 *  \param[out]    recv_topk_weights      [recv_capacity] float32 per-row routing weights.
 *  \param[out]    gemm_out               [recv_capacity, 2 * ffn_dim] GEMM output.
 *  \param[out]    act_out                [recv_capacity, ffn_dim] SwiGLU output.
 *  \param[in,out] recv_tokens_per_expert [num_local_experts] int32 padded counts. Eager mode
 *                                        provides them to build the expert layout. Capacity mode
 *                                        writes them: the dispatch without drops and with at most
 *                                        2048 local experts, otherwise the routing prepare.
 *  \param[out]    expert_token_offsets   [num_local_experts + 1] int32 expert layout in tokens
 *                                        (rows): expert 0's start (0), then each expert's padded
 *                                        row end, the GEMM operand. expert_token_offsets[1:] must
 *                                        start 16-byte aligned, so front-pad the allocation by 3
 *                                        int32 elements. Capacity mode writes it; eager mode builds
 *                                        it from recv_tokens_per_expert. The last element is the
 *                                        recv total; the per-expert counts are its diffs.
 *  \param[out]    total_recv_tokens      [1] int32 recv total before any overflow drop. The routing
 *                                        prepare writes it; null when the dispatch writes
 *                                        expert_token_offsets, where expert_token_offsets[-1] is
 *                                        the padded total of the kept rows.
 *  \param[in,out] workspace              uint8 scratch, nvte_ep_overlap_workspace_size bytes.
 *  \param[in]     layer_cfg              Per-layer EP configuration.
 *  \param[in]     stream                 CUDA stream.
 */
void nvte_ep_dispatch_gemm_fwd(NVTETensor handle_mem, const NVTETensor topk_idx,
                               const NVTETensor tokens, NVTECommWindow tokens_win,
                               const NVTETensor topk_weights, const NVTETensor weight,
                               NVTETensor recv_tokens, NVTETensor recv_topk_weights,
                               NVTETensor gemm_out, NVTETensor act_out,
                               NVTETensor recv_tokens_per_expert, NVTETensor expert_token_offsets,
                               NVTETensor total_recv_tokens, NVTETensor workspace,
                               const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream);

/*! \brief Second grouped GEMM overlapped with the combine.
 *
 *  Per expert slab e: expert_out = (gemm_in @ weight[e]^T) * recv_topk_weights; result is the
 *  unweighted combine of expert_out. BF16 applies the routing weights in the dispatch GEMM's
 *  activation, so recv_topk_weights is null here; MXFP8 leaves them off the activation and applies
 *  them to this FC2 output.
 *
 *  \param[in,out] handle_mem             uint8 routing-state buffer.
 *  \param[in]     gemm_in                [recv_capacity, ffn_dim] act_out of
 *                                        nvte_ep_dispatch_gemm_fwd, bfloat16 or MXFP8.
 *  \param[in]     recv_topk_weights      [recv_capacity] float32 per-row routing weights; null for
 *                                        unit weights (bfloat16 act_out carries them).
 *  \param[in]     expert_token_offsets   [num_local_experts + 1] int32 expert layout from
 *                                        nvte_ep_dispatch_gemm_fwd (the GEMM reads its row ends);
 *                                        expert_token_offsets[1:] must start 16-byte aligned.
 *  \param[in]     weight                 [num_local_experts, hidden_dim, ffn_dim] weights.
 *  \param[out]    expert_out             [recv_capacity, hidden_dim] GEMM output.
 *  \param[out]    result                 [T, hidden_dim] combined output.
 *  \param[in,out] workspace              uint8 scratch, nvte_ep_overlap_workspace_size bytes.
 *  \param[in]     layer_cfg              Per-layer EP configuration.
 *  \param[in]     stream                 CUDA stream.
 */
void nvte_ep_combine_gemm_fwd(NVTETensor handle_mem, const NVTETensor gemm_in,
                              const NVTETensor recv_topk_weights,
                              const NVTETensor expert_token_offsets, const NVTETensor weight,
                              NVTETensor expert_out, NVTETensor result, NVTETensor workspace,
                              const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream);

/*! \brief Backward of nvte_ep_combine_gemm_fwd and of the SwiGLU: the combine backward
 *         overlapped with the second GEMM's data gradient, SwiGLU backward in its epilogue.
 *
 *  The combine backward scatters grad_result to the expert rows as grad_expert_out (dy). SwiGLU
 *  backward over gemm_out = [gate | up] (gate and up in alternating 32-column blocks) gives
 *  grad_gemm_out, scaled by recv_topk_weights (MXFP8) or carrying it from the activation (BF16).
 *  Over expert e's rows, with W2 its forward weight and wgrad_input the route-weighted activation:
 *
 *      grad_recv_topk_weights = sum((dy @ W2) * (silu(gate) * up), over ffn)   one value per row
 *      weight_grad[e]         = dy^T @ wgrad_input                            [hidden_dim, ffn_dim]
 *
 *  grad_recv_topk_weights uses the unweighted activation silu(gate) * up. weight_grad is written
 *  only when non-null; accumulate_weight_grad adds to it instead of overwriting it.
 *
 *  \param[in,out] handle_mem             uint8 routing-state buffer.
 *  \param[in]     grad_result            [T, hidden_dim] gradient of result.
 *  \param[in]     grad_result_win        Optional symm-mem window for grad_result (MXFP8: and,
 *                                        in scale_window, for its scales); NVTECommWindow{}
 *                                        stages them.
 *  \param[in]     gemm_out               [recv_capacity, 2 * ffn_dim] gemm_out of
 *                                        nvte_ep_dispatch_gemm_fwd.
 *  \param[in]     recv_topk_weights      [recv_capacity] float32 per-row routing weights.
 *  \param[in]     recv_tokens_per_expert [num_local_experts] int32 padded counts.
 *  \param[in]     weight                 [num_local_experts, hidden_dim, ffn_dim] forward weights.
 *  \param[in]     wgrad_input            [recv_capacity, ffn_dim] bfloat16 act_out of
 *                                        nvte_ep_dispatch_gemm_fwd, or MXFP8 act_out of this
 *                                        call; with weight_grad only.
 *  \param[out]    grad_expert_out        [recv_capacity, hidden_dim] received gradient of
 *                                        expert_out; MXFP8: rowwise with compact scales and, when
 *                                        its columnwise data is set, also columnwise, its scales
 *                                        GEMM-swizzled per expert slab, for the weight gradient.
 *  \param[out]    grad_gemm_out          [recv_capacity, 2 * ffn_dim] gradient of gemm_out; MXFP8:
 *                                        rowwise and columnwise.
 *  \param[out]    grad_recv_topk_weights [recv_capacity] float32 gradient of recv_topk_weights.
 *  \param[out]    act_out                [recv_capacity, ffn_dim] columnwise MXFP8
 *                                        recv_topk_weights * silu(gate) * up, the input of the
 *                                        second GEMM's weight gradient; MXFP8 only, else null.
 *  \param[in,out] weight_grad            [num_local_experts, hidden_dim, ffn_dim] bfloat16 or
 *                                        float32 gradient of weight, or null.
 *  \param[in]     accumulate_weight_grad Add to weight_grad instead of writing it.
 *  \param[in,out] workspace              uint8 scratch, nvte_ep_overlap_workspace_size bytes.
 *  \param[in]     layer_cfg              Per-layer EP configuration.
 *  \param[in]     stream                 CUDA stream.
 */
void nvte_ep_combine_gemm_bwd(NVTETensor handle_mem, const NVTETensor grad_result,
                              NVTECommWindow grad_result_win, const NVTETensor gemm_out,
                              const NVTETensor recv_topk_weights,
                              const NVTETensor recv_tokens_per_expert, const NVTETensor weight,
                              const NVTETensor wgrad_input, NVTETensor grad_expert_out,
                              NVTETensor grad_gemm_out, NVTETensor grad_recv_topk_weights,
                              NVTETensor act_out, NVTETensor weight_grad,
                              int accumulate_weight_grad, NVTETensor workspace,
                              const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream);

/*! \brief Backward of nvte_ep_dispatch_gemm_fwd: the first GEMM's data gradient overlapped with
 *         the dispatch backward.
 *
 *  \param[in,out] handle_mem             uint8 routing-state buffer.
 *  \param[in]     grad_gemm_out          [recv_capacity, 2 * ffn_dim] from
 *                                        nvte_ep_combine_gemm_bwd; MXFP8 with weight_grad: also
 *                                        columnwise.
 *  \param[in]     grad_recv_topk_weights [recv_capacity] float32 from nvte_ep_combine_gemm_bwd.
 *  \param[in]     recv_tokens_per_expert [num_local_experts] int32 padded counts.
 *  \param[in]     weight                 [num_local_experts, 2 * ffn_dim, hidden_dim] forward
 *                                        weights.
 *  \param[in,out] wgrad_input            [recv_capacity, hidden_dim] recv_tokens of
 *                                        nvte_ep_dispatch_gemm_fwd, bfloat16 or columnwise
 *                                        MXFP8; with weight_grad only. MXFP8 with its rowwise
 *                                        data and compact scales also set: the call writes the
 *                                        columnwise data and scales from them.
 *  \param[out]    grad_recv_tokens       [recv_capacity, hidden_dim] gradient of recv_tokens.
 *  \param[out]    grad_tokens            [T, hidden_dim] gradient of tokens.
 *  \param[out]    grad_topk_weights      [T, top_k] float32 gradient of topk_weights.
 *  \param[in,out] weight_grad            [num_local_experts, 2 * ffn_dim, hidden_dim] bfloat16 or
 *                                        float32 gradient of weight, or null.
 *  \param[in]     accumulate_weight_grad Add to weight_grad instead of writing it.
 *  \param[in,out] workspace              uint8 scratch, nvte_ep_overlap_workspace_size bytes.
 *  \param[in]     layer_cfg              Per-layer EP configuration.
 *  \param[in]     stream                 CUDA stream.
 */
void nvte_ep_dispatch_gemm_bwd(NVTETensor handle_mem, const NVTETensor grad_gemm_out,
                               const NVTETensor grad_recv_topk_weights,
                               const NVTETensor recv_tokens_per_expert, const NVTETensor weight,
                               const NVTETensor wgrad_input, NVTETensor grad_recv_tokens,
                               NVTETensor grad_tokens, NVTETensor grad_topk_weights,
                               NVTETensor weight_grad, int accumulate_weight_grad,
                               NVTETensor workspace, const NVTEEpLayerConfig* layer_cfg,
                               cudaStream_t stream);

/*! \brief Bytes of a workspace that fits every overlapped GEMM call of a group. Queryable once the
 *         EP group is bootstrapped; call it at warmup, as it compiles the group's GEMMs.
 *
 *  \param[in] hidden_dim        Hidden size.
 *  \param[in] ffn_dim           Expert FFN size.
 *  \param[in] num_local_experts Experts held by this rank.
 *  \param[in] recv_capacity     Rows of the receive buffers (recv_capacity).
 *  \param[in] with_weight_grad  Nonzero to fit the backward weight-gradient workspace.
 *  \param[in] mxfp8             Nonzero for MXFP8 operands instead of bfloat16.
 *  \param[in] layer_cfg         Per-layer EP configuration.
 *  \return Required workspace size in bytes.
 */
size_t nvte_ep_overlap_workspace_size(size_t hidden_dim, size_t ffn_dim, size_t num_local_experts,
                                      size_t recv_capacity, int with_weight_grad, int mxfp8,
                                      const NVTEEpLayerConfig* layer_cfg);

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_EP_GEMM_H_
