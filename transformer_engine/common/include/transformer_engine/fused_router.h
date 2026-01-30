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
 *  \param[in]     expert_bias     Expert bias. (Only used at the sigmoid case)
 *  \param[out]    probs           Output tensor for probabilities.
 *  \param[out]    routing_map     Output tensor for routing map.
 *  \param[out]    intermediate_output  Output tensor for intermediate output. (Softmax/sigmoid output)
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, cudaStream_t stream);

/*! \brief Backward pass for fused topk + softmax/sigmoid.
 *
 *  \param[in]     routing_map     Routing map.
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

/*! \brief Forward pass for computing scores/routing map for auxiliary loss.
 *
 *  \param[in]     logits          Logits from the gating GEMM.
 *  \param[in]     num_tokens      Number of tokens.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     topk            Topk value.
 *  \param[in]     score_function  Score function, 0: sigmoid, 1: softmax, 2: sqrtsoftplus.
 *  \param[out]    scores          Output tensor for scores.
 *  \param[in]     routing_map     Routing map.
 *  \param[in]     intermediate_output  Intermediate output from the forward pass. (Softmax/sigmoid output)
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, const NVTETensor routing_map,
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

/*! \brief Forward pass for auxiliary loss.
 *
 *  \param[in]     probs           Probabilities from the forward pass.
 *  \param[in]     tokens_per_expert  Number of tokens per expert.
 *  \param[in]     total_num_tokens   Number of total tokens. Will be used in seq/global aux loss.
 *  \param[in]     num_experts     Number of experts.
 *  \param[in]     num_rows        Number of rows of probs.
 *  \param[in]     num_cols        Number of columns of probs.
 *  \param[in]     topk            Topk value.
 *  \param[in]     coeff           Coefficient.
 *  \param[out]    aux_loss        Output GPU scalar for auxiliary loss.
 *  \param[out]    Const_buf       Output GPU scalar for temporary constant buffer for backward pass.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_moe_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                     int total_num_tokens, int num_experts, int num_rows,
                                     int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                     NVTETensor Const_buf, cudaStream_t stream);

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
