/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_fp16_bf16_max_seqlen_512.h
 *  \brief Functions for fused attention for half precision with seqlen <= 512
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_MAX_512_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_MAX_512_H_

#include <cudnn.h>

#include "common/common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
#if (CUDNN_VERSION >= 8901)
void fused_attn_max_512_fwd_qkvpacked(size_t batch, size_t num_head, size_t max_seqlen,
                                      size_t head_size, bool is_training, float attn_scale,
                                      float p_dropout, NVTE_QKV_Layout qkv_layout,
                                      NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                      const Tensor *input_QKV, const Tensor *input_Bias,
                                      Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                                      const Tensor *cu_seqlens, const Tensor *rng_state,
                                      Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_max_512_fwd_kvpacked(size_t batch, size_t num_head, size_t q_max_seqlen,
                                     size_t kv_max_seqlen, size_t head_dim, bool is_training,
                                     float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                                     NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                     const Tensor *input_Q, const Tensor *input_KV,
                                     const Tensor *input_Bias, Tensor *output_O,
                                     NVTETensorPack *Aux_CTX_Tensors, const Tensor *q_cu_seqlens,
                                     const Tensor *kv_cu_seqlens, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_max_512_fwd(size_t batch, size_t num_head, size_t q_max_seqlen,
                            size_t kv_max_seqlen, size_t head_dim, bool is_training,
                            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                            NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                            const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V,
                            const Tensor *input_Bias, Tensor *output_O,
                            NVTETensorPack *Aux_CTX_Tensors, const Tensor *q_cu_seqlens,
                            const Tensor *kv_cu_seqlens, const Tensor *rng_state, Tensor *workspace,
                            cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_max_512_bwd_qkvpacked(size_t batch, size_t num_head, size_t max_seqlen,
                                      size_t head_dim, float attn_scale, float p_dropout,
                                      NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                      NVTE_Mask_Type mask_type, const Tensor *input_QKV,
                                      const Tensor *input_dO, Tensor *output_S, Tensor *output_dQKV,
                                      Tensor *output_dBias, const Tensor *cu_seqlens,
                                      Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_max_512_bwd_kvpacked(size_t batch, size_t num_head, size_t q_max_seqlen,
                                     size_t kv_max_seqlen, size_t head_dim, float attn_scale,
                                     float p_dropout, NVTE_QKV_Layout qkv_layout,
                                     NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                     const Tensor *input_Q, const Tensor *input_KV,
                                     const Tensor *input_dO, Tensor *output_S, Tensor *output_dQ,
                                     Tensor *output_dKV, Tensor *output_dBias,
                                     const Tensor *q_cu_seqlens, const Tensor *kv_cu_seqlens,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_max_512_bwd(size_t batch, size_t num_head, size_t q_max_seqlen,
                            size_t kv_max_seqlen, size_t head_dim, float attn_scale,
                            float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                            NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
                            const Tensor *input_V, const Tensor *input_dO, Tensor *output_S,
                            Tensor *output_dQ, Tensor *output_dK, Tensor *output_dV,
                            Tensor *output_dBias, const Tensor *q_cu_seqlens,
                            const Tensor *kv_cu_seqlens, Tensor *workspace, cudaStream_t stream,
                            cudnnHandle_t handle);
#endif  // CUDNN_VERSION >= 8901
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_MAX_512_H_
