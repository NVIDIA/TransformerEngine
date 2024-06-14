/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_fp8.h
 *  \brief Functions for fused attention for FP8 with seqlen <= 512
 */

#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
#if (CUDNN_VERSION >= 8900)
// fused attention FWD FP8 with packed QKV
void fused_attn_fp8_fwd_qkvpacked(size_t batch, size_t num_attn_heads, size_t max_seqlen,
                                  size_t head_dim, bool is_training, float attn_scale,
                                  float p_dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
                                  const Tensor *input_QKV, Tensor *input_output_S, Tensor *output_O,
                                  NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens,
                                  const Tensor *rng_state, Tensor *workspace, cudaStream_t stream,
                                  cudnnHandle_t handle);

// fused attention BWD FP8 with packed QKV
void fused_attn_fp8_bwd_qkvpacked(
    size_t batch, size_t num_attn_heads, size_t max_seqlen, size_t head_dim, float attn_scale,
    float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    const Tensor *input_QKV, const Tensor *input_O, const Tensor *input_dO, const Tensor *input_M,
    const Tensor *input_ZInv, const Tensor *input_S, Tensor *input_output_dP,
    const Tensor *output_dQKV, const Tensor *cu_seqlens, const Tensor *rng_state, Tensor *workspace,
    cudaStream_t stream, cudnnHandle_t handle);

// fused attention FWD FP8 with packed KV
void fused_attn_fp8_fwd_kvpacked(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                                 size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                                 bool is_training, float attn_scale, float p_dropout,
                                 NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                 NVTE_Mask_Type mask_type, const Tensor *input_Q,
                                 const Tensor *input_KV, Tensor *input_output_S, Tensor *output_O,
                                 NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q,
                                 const Tensor *cu_seqlens_kv, const Tensor *rng_state,
                                 Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// fused attention BWD FP8 with packed KV
void fused_attn_fp8_bwd_kvpacked(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    const Tensor *input_Q, const Tensor *input_KV, const Tensor *input_O, const Tensor *input_dO,
    const Tensor *input_M, const Tensor *input_ZInv, const Tensor *input_S, Tensor *input_output_dP,
    const Tensor *output_dQ, const Tensor *output_dKV, const Tensor *cu_seqlens_q,
    const Tensor *cu_seqlens_kv, const Tensor *rng_state, Tensor *workspace, cudaStream_t stream,
    cudnnHandle_t handle);

// fused attention FWD FP8 with separate Q, K, V
void fused_attn_fp8_fwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                        size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                        bool is_training, float attn_scale, float p_dropout,
                        NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                        NVTE_Mask_Type mask_type, const Tensor *input_Q, const Tensor *input_K,
                        const Tensor *input_V, Tensor *input_output_S, Tensor *output_O,
                        NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q,
                        const Tensor *cu_seqlens_kv, const Tensor *rng_state, Tensor *workspace,
                        cudaStream_t stream, cudnnHandle_t handle);

// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                        size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                        float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
                        NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, const Tensor *input_Q,
                        const Tensor *input_K, const Tensor *input_V, const Tensor *input_O,
                        const Tensor *input_dO, const Tensor *input_M, const Tensor *input_ZInv,
                        const Tensor *input_S, Tensor *input_output_dP, const Tensor *output_dQ,
                        const Tensor *output_dK, const Tensor *output_dV,
                        const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                        const Tensor *rng_state, Tensor *workspace, cudaStream_t stream,
                        cudnnHandle_t handle);
#endif  // end of CUDNN>=8900
}  // namespace transformer_engine
