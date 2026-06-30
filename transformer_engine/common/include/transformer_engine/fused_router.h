/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Output format of the routing_map tensor.
 *
 *  BYTEMAP   — bool/uint8 tensor of shape [num_tokens, num_experts]; one byte
 *              per (token, expert) pair, 0 or 1.
 *  BITMAP_U8 — uint8 tensor of shape [num_tokens, ceil(num_experts/8)]; bit
 *              (e % 8) of byte (e / 8) of row t is 1 iff token t routes to
 *              expert e (little-endian / LSB-first packing along the expert
 *              axis).
 */
typedef enum {
  NVTE_ROUTING_MAP_FORMAT_BYTEMAP = 0,
  NVTE_ROUTING_MAP_FORMAT_BITMAP_U8 = 1,
} NVTERoutingMapFormat;

/*! \brief Apply topk + softmax/sigmoid to the input tensor. Grouped topk is supported (deprecated).
 *
 *  \deprecated This function has been deprecated in favor of
 *              nvte_fused_topk_with_score_function_forward_v2, which adds support
 *              for the NVTE_ROUTING_MAP_FORMAT_BITMAP_U8 routing_map layout. This
 *              entry point assumes NVTE_ROUTING_MAP_FORMAT_BYTEMAP.
 *
 *  \param[in]     logits          Logits from the gating GEMM.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     use_pre_softmax Whether to use softmax before topk.
 *  \param[in]     num_groups      Number of groups in grouped topk.
 *  \param[in]     group_topk      Grouped topk value.
 *  \param[in]     scaling_factor  Scaling factor.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[in]     expert_bias     Expert bias. (Used at the sigmoid/sqrtsoftplus cases)
 *  \param[out]    probs           Output tensor for probabilities.
 *  \param[out]    routing_map     Output tensor for routing map (BYTEMAP layout).
 *  \param[out]    intermediate_output  Output tensor for intermediate output. (Softmax/sigmoid output)
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, cudaStream_t stream);

/*! \brief Apply topk + softmax/sigmoid to the input tensor. Grouped topk is supported.
 *
 *  \param[in]     logits          Logits from the gating GEMM.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     use_pre_softmax Whether to use softmax before topk.
 *  \param[in]     num_groups      Number of groups in grouped topk.
 *  \param[in]     group_topk      Grouped topk value.
 *  \param[in]     scaling_factor  Scaling factor.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[in]     expert_bias     Expert bias. (Used at the sigmoid/sqrtsoftplus cases)
 *  \param[out]    probs           Output tensor for probabilities.
 *  \param[out]    routing_map     Output tensor for routing map. Shape depends on
 *                                 routing_map_format (see NVTERoutingMapFormat).
 *  \param[in]     routing_map_format NVTERoutingMapFormat value selecting the routing_map
 *                                    output layout. The caller is responsible for
 *                                    allocating routing_map with the matching shape.
 *  \param[out]    intermediate_output  Output tensor for intermediate output. (Softmax/sigmoid output)
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_topk_with_score_function_forward_v2(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTERoutingMapFormat routing_map_format, NVTETensor intermediate_output, cudaStream_t stream);

/*! \brief Backward pass for fused topk + softmax/sigmoid (deprecated).
 *
 *  \deprecated This function has been deprecated in favor of
 *              nvte_fused_topk_with_score_function_backward_v2. This entry point
 *              assumes NVTE_ROUTING_MAP_FORMAT_BYTEMAP.
 *
 *  \param[in]     routing_map     Routing map (BYTEMAP layout).
 *  \param[in]     intermediate_output  Intermediate output from the forward pass. (Softmax/sigmoid output)
 *  \param[in]     grad_probs      Gradient of probs.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     use_pre_softmax Whether to use softmax before topk.
 *  \param[in]     scaling_factor  Scaling factor.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[out]    grad_logits     Gradient of logits.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, cudaStream_t stream);

/*! \brief Backward pass for fused topk + softmax/sigmoid.
 *
 *  \param[in]     routing_map     Routing map (same layout as produced by forward).
 *  \param[in]     routing_map_format NVTERoutingMapFormat value matching the layout of routing_map.
 *  \param[in]     intermediate_output  Intermediate output from the forward pass. (Softmax/sigmoid output)
 *  \param[in]     grad_probs      Gradient of probs.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     use_pre_softmax Whether to use softmax before topk.
 *  \param[in]     scaling_factor  Scaling factor.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[out]    grad_logits     Gradient of logits.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_topk_with_score_function_backward_v2(const NVTETensor routing_map,
                                                     NVTERoutingMapFormat routing_map_format,
                                                     const NVTETensor intermediate_output,
                                                     const NVTETensor grad_probs, int num_tokens,
                                                     int num_experts, int topk, int use_pre_softmax,
                                                     float scaling_factor, int score_function,
                                                     NVTETensor grad_logits, cudaStream_t stream);

/*! \brief Forward pass for computing scores/routing map for auxiliary loss (deprecated).
 *
 *  \deprecated This function has been deprecated in favor of
 *              nvte_fused_score_for_moe_aux_loss_forward_v2. This entry point
 *              assumes NVTE_ROUTING_MAP_FORMAT_BYTEMAP.
 *
 *  \param[in]     logits          Logits from the gating GEMM.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[out]    scores          Output tensor for scores.
 *  \param[out]    routing_map     Output tensor for routing map (BYTEMAP layout).
 *  \param[in]     intermediate_output  Intermediate output from the forward pass. (Softmax/sigmoid output)
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               cudaStream_t stream);

/*! \brief Forward pass for computing scores/routing map for auxiliary loss.
 *
 *  \param[in]     logits          Logits from the gating GEMM.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[out]    scores          Output tensor for scores.
 *  \param[out]    routing_map     Output tensor for routing map. Shape depends on
 *                                 routing_map_format (see NVTERoutingMapFormat).
 *  \param[in]     routing_map_format NVTERoutingMapFormat value selecting the routing_map
 *                                    output layout.
 *  \param[in]     intermediate_output  Intermediate output from the forward pass. (Softmax/sigmoid output)
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_score_for_moe_aux_loss_forward_v2(const NVTETensor logits, int num_tokens,
                                                  int num_experts, int topk, int score_function,
                                                  NVTETensor scores, NVTETensor routing_map,
                                                  NVTERoutingMapFormat routing_map_format,
                                                  const NVTETensor intermediate_output,
                                                  cudaStream_t stream);

/*! \brief Backward pass for computing scores/routing map for auxiliary loss.
 *
 *  \param[in]     intermediate_output  Intermediate output from the forward pass. (Softmax/sigmoid output)
 *  \param[in]     grad_scores      Gradient of scores.
 *  \param[in]     num_tokens       Number of tokens.
 *  \param[in]     num_experts      Number of experts.
 *  \param[in]     topk             Topk value.
 *  \param[in]     score_function   Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[out]    grad_logits      Gradient of logits.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_fused_score_for_moe_aux_loss_backward(const NVTETensor intermediate_output,
                                                const NVTETensor grad_scores, int num_tokens,
                                                int num_experts, int topk, int score_function,
                                                NVTETensor grad_logits, cudaStream_t stream);

/*! \brief Forward pass for auxiliary loss. Host-int total_num_tokens path:
 *  the coefficient is folded on the host and passed as a kernel argument.
 *  Prefer this path when total_num_tokens is statically known and the call
 *  is not captured into a CUDA Graph.
 *
 *  \param[in]     probs              Probabilities from the forward pass.
 *  \param[in]     tokens_per_expert  Number of tokens per expert.
 *  \param[in]     total_num_tokens   Number of total tokens. Used in seq/global aux loss.
 *  \param[in]     num_experts        Number of experts.
 *  \param[in]     num_rows           Number of rows of probs.
 *  \param[in]     num_cols           Number of columns of probs.
 *  \param[in]     topk               Topk value.
 *  \param[in]     coeff              Coefficient.
 *  \param[out]    aux_loss           Output GPU scalar for auxiliary loss.
 *  \param[out]    Const_buf          Output GPU scalar for temporary constant buffer for backward
 *                                    pass.
 *  \param[in]     stream             CUDA stream used for the operation.
 */
void nvte_fused_moe_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                     int total_num_tokens, int num_experts, int num_rows,
                                     int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                     NVTETensor Const_buf, cudaStream_t stream);

/*! \brief Forward pass for auxiliary loss. Device-tensor total_num_tokens path:
 *  the coefficient is computed on device from a 0-dim int64 GPU tensor so its
 *  value stays dynamic across CUDA Graph replays. Prefer this path when the
 *  caller needs CUDA-graph-safe semantics with a dynamic token count.
 *
 *  \param[in]     total_num_tokens   0-dim int64 GPU tensor with the total token count.
 *  Other parameters as in :c:func:`nvte_fused_moe_aux_loss_forward`.
 */
void nvte_fused_moe_aux_loss_forward_graph_safe(const NVTETensor probs,
                                                const NVTETensor tokens_per_expert,
                                                const NVTETensor total_num_tokens, int num_experts,
                                                int num_rows, int num_cols, int topk, float coeff,
                                                NVTETensor aux_loss, NVTETensor Const_buf,
                                                cudaStream_t stream);

/*! \brief Backward pass for auxiliary loss.
 *
 *  \param[in]     Const_buf       Constant buffer from the forward pass.
 *  \param[in]     tokens_per_expert  Number of tokens per expert.
 *  \param[in]     num_rows        Number of rows of probs.
 *  \param[in]     num_cols        Number of columns of probs.
 *  \param[in]     grad_aux_loss   Gradient of auxiliary loss.
 *  \param[out]    grad_probs      Gradient of probs.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_moe_aux_loss_backward(const NVTETensor Const_buf,
                                      const NVTETensor tokens_per_expert, int num_rows,
                                      int num_cols, NVTETensor grad_aux_loss, NVTETensor grad_probs,
                                      cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
