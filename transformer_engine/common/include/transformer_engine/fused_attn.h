/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NVTE_QKV_Layout {
/*!< separate Q, K, V tensors:
     Q: [total_seqs_q, num_heads, head_dim]
                      | Q   Q   Q        ...       Q
                      | \___________  _____________/
        total_seqs_q <|             \/ 
                      |   num_heads * head_dim
     K: [total_seqs_kv, num_heads, head_dim]
                       | K   K   K        ...       K
                       | \___________  _____________/
        total_seqs_kv <|             \/ 
                       |   num_heads * head_dim
     V: [total_seqs_kv, num_heads, head_dim]
                       | V   V   V        ...       V
                       | \___________  _____________/
        total_seqs_kv <|             \/ 
                       |   num_heads * head_dim
 */
    NVTE_NOT_INTERLEAVED = 0,

/*!< packed QKV tensor:
     QKV: [total_seqs, 3, num_heads, head_dim]
                 | Q   Q   Q        ...       Q K K K ... K V V V ... V 
                 | \___________  _____________/
     total_seqs <|             \/ 
                 |   num_heads * head_dim
 */
    NVTE_QKV_INTERLEAVED = 1,

/*!< Q and packed KV tensor:
     Q: [total_seqs_q, num_heads, head_dim]
                      | Q   Q   Q        ...       Q
                      | \___________  _____________/
        total_seqs_q <|             \/ 
                      |   num_heads * head_dim
     KV: [total_seqs_kv, 2, num_heads, head_dim]
                        | K   K   K        ...       K V V V ... V 
                        | \___________  _____________/
         total_seqs_kv <|             \/ 
                        |   num_heads * head_dim
 */
    NVTE_KV_INTERLEAVED = 2
};

enum NVTE_Bias_Type {
    NVTE_NO_BIAS = 0,  /*!< no bias */
    NVTE_PRE_SCALE_BIAS = 1,  /*!< bias before scale */
    NVTE_POST_SCALE_BIAS = 2  /*!< bias after scale */
};

enum NVTE_Mask_Type {
    NVTE_PADDING_MASK = 0,  /*!< padding attention mask */
    NVTE_CAUSAL_MASK = 1,  /*!< causal attention mask */
    NVTE_NO_MASK = 2  /*!< no masking */
};

/*! \brief Compute dot product attention with packed QKV input.
 *
 * Computes:
 *  - P = Q * K.T + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * V.T
 *
 * Support Matrix:
 *  | precision |    qkv layout   |  bias   |  mask   | sequence length |  head_dim  |
 *  |    FP8    | QKV_INTERLEAVED | NO_BIAS | PADDING |   <= 512        |      64    |
 *
 *
 *  \param[in]     QKV                      The QKV tensor in packed format,
 *                                          [total_seqs, 3, num_heads, head_dim].
 *  \param[in]     Bias                     The Bias tensor.
 *  \param[in,out] S                        The S tensor.
 *  \param[out]    O                        The output O tensor.
 *  \param[out]    Aux_Output_Tensors       Auxiliary output tensors when training, e.g. M, ZInv.
 *  \param[in]     cu_seqlens               Accumulative sequence lengths, [batch_size + 1].
 *  \param[in]     rng_state                Seed and offset of CUDA random number generator.
 *  \param[in]     max_seqlen               Max sequence length used for computing,
 *                                          it may be >= max(cu_seqlens). 
 *  \param[in]     is_training              Whether this is in training mode or inference.
 *  \param[in]     attn_scale               Scaling factor for Q * K.T.
 *  \param[in]     dropout                  Dropout probability.
 *  \param[in]     qkv_layout               QKV tensor's layout.
 *  \param[in]     bias_type                Bias type.
 *  \param[in]     attn_mask_type           Attention mask type.
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     stream                   CUDA stream used for this operation.
 *  \param[in]     return_softmax           FlashAttention backend: Whether to return softmax tensor.
 *  \param[in]     num_split                FlashAttention backend: Number of kernels.
 *  \param[in]     fused_attention_backend  Which backend to use.
 */
void nvte_fused_attn_fwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            const NVTETensor cu_seqlens,
            const NVTETensor rng_state,
            size_t max_seqlen,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            bool return_softmax,
            int num_split,
            int fused_attention_backend);

/*! \brief Compute the backward of the dot product attention with packed QKV input.
 *
 * Support Matrix:
 *  | precision |    qkv layout   |  bias   |  mask   | sequence length |  head_dim  |
 *  |    FP8    | QKV_INTERLEAVED | NO_BIAS | PADDING |   <= 512        |      64    |
 *
 *
 *  \param[in]     QKV                      The QKV tensor in packed format,
 *                                          [total_seqs, 3, num_heads, head_dim].
 *  \param[in]     O                        The O tensor from forward.
 *  \param[in]     dO                       The gradient of the O tensor.
 *  \param[in]     S                        The S tensor.
 *  \param[in,out] dP                       The gradient of the P tensor.
 *  \param[in]     Aux_CTX_Tensors          Auxiliary tensors from forward when in training mode.
 *  \param[out]    dQKV                     The gradient of the QKV tensor.
 *  \param[out]    dBias                    The gradient of the Bias tensor.
 *  \param[in]     cu_seqlens               Accumulative sequence lengths, [batch_size + 1].
 *  \param[in]     max_seqlen               Max sequence length used for computing,
 *                                          it may be >= max(cu_seqlens). 
 *  \param[in]     attn_scale               Scaling factor for Q * K.T.
 *  \param[in]     dropout                  Dropout probability.
 *  \param[in]     qkv_layout               QKV tensor's layout.
 *  \param[in]     bias_type                Bias type.
 *  \param[in]     attn_mask_type           Attention mask type.
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     stream                   CUDA stream used for this operation.
 *  \param[in]     num_split                FlashAttention backend: Number of kernels.
 *  \param[in]     fused_attention_backend  Which backend to use.
 */
void nvte_fused_attn_bwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQKV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens,
            size_t max_seqlen,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            int num_split,
            int fused_attention_backend);

/*! \brief Compute dot product attention with packed KV input.
 *
 * Computes:
 *  - P = Q * K.T + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * V.T
 *
 *  \param[in]     Q                        The Q tensor, [total_seqs_q, num_heads, head_dim].
 *  \param[in]     KV                       The KV tensor, [total_seqs_kv, 2, num_heads, head_dim].
 *  \param[in]     Bias                     The Bias tensor.
 *  \param[in,out] S                        The S tensor.
 *  \param[out]    O                        The output O tensor.
 *  \param[out]    Aux_Output_Tensors       Auxiliary output tensors when training, e.g. M, ZInv.
 *  \param[in]     cu_seqlens_q             Accumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv            Accumulative sequence lengths for KV, [batch_size + 1].
 *  \param[in]     rng_state                Seed and offset of CUDA random number generator.
 *  \param[in]     max_seqlen_q             Max sequence length used for computing for Q.  
 *                                          it may be >= max(cu_seqlens_q). 
 *  \param[in]     max_seqlen_kv            Max sequence length used for computing for KV.  
 *                                          it may be >= max(cu_seqlens_kv). 
 *  \param[in]     is_training              Whether this is in training mode or inference.
 *  \param[in]     attn_scale               Scaling factor for Q * K.T.
 *  \param[in]     dropout                  Dropout probability.
 *  \param[in]     qkv_layout               QKV tensor's layout.
 *  \param[in]     bias_type                Bias type.
 *  \param[in]     attn_mask_type           Attention mask type.
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     stream                   CUDA stream used for this operation.
 *  \param[in]     return_softmax           FlashAttention backend: Whether to return softmax tensor.
 *  \param[in]     num_split                FlashAttention backend: Number of kernels.
 *  \param[in]     fused_attention_backend  Which backend to use.
 */
void nvte_fused_attn_fwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            bool return_softmax,
            int num_split,
            int fused_attention_backend);

/*! \brief Compute the backward of the dot product attention with packed KV input.
 *
 *  \param[in]     Q                        The Q tensor, [total_seqs_q, num_heads, head_dim].
 *  \param[in]     KV                       The KV tensor, [total_seqs_kv, 2, num_heads, head_dim].
 *  \param[in]     O                        The O tensor from forward.
 *  \param[in]     dO                       The gradient of the O tensor.
 *  \param[in]     S                        The S tensor.
 *  \param[in,out] dP                       The gradient of the P tensor.
 *  \param[in]     Aux_CTX_Tensors          Auxiliary tensors from forward when in training mode.
 *  \param[out]    dQ                       The gradient of the Q tensor.
 *  \param[out]    dKV                      The gradient of the KV tensor.
 *  \param[out]    dBias                    The gradient of the Bias tensor.
 *  \param[in]     cu_seqlens_q             Accumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv            Accumulative sequence lengths for KV, [batch_size + 1].
 *  \param[in]     max_seqlen_q             Max sequence length used for computing for Q.  
 *                                          it may be >= max(cu_seqlens_q). 
 *  \param[in]     max_seqlen_kv            Max sequence length used for computing for KV.  
 *                                          it may be >= max(cu_seqlens_kv). 
 *  \param[in]     attn_scale               Scaling factor for Q * K.T.
 *  \param[in]     dropout                  Dropout probability.
 *  \param[in]     qkv_layout               QKV tensor's layout.
 *  \param[in]     bias_type                Bias type.
 *  \param[in]     attn_mask_type           Attention mask type.
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     stream                   CUDA stream used for this operation.
 *  \param[in]     num_split                FlashAttention backend: Number of kernels.
 *  \param[in]     fused_attention_backend  Which backend to use.
 */
void nvte_fused_attn_bwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dKV,
            NVTETensor dBias,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream,
            int num_split,
            int fused_attention_backend);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
