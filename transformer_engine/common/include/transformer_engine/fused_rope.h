/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROPE_H_
#define TRANSFORMER_ENGINE_FUSED_ROPE_H_

#include "fused_attn.h"
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Apply rotary positional embedding to the input tensor.
 *
 *  \param[in]     input           Input tensor for fused rope.
 *  \param[in]     cu_seqlens      The cumulative sum of sequence lengths tensor.
 *                                 (Required for the thd format, empty tensor for other formats)
 *  \param[in]     freqs           The freqs tensor.
 *  \param[in]     start_positions The beginning offsets for applying RoPE embeddings.
 *  \param[out]    output          Output tensor.
 *  \param[in]     qkv_format      QKV format.
 *  \param[in]     interleaved     Whether to use interleaved rotary position embedding.
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Length of the s dimension of input.
 *  \param[in]     b               Length of the b dimension of input.
 *  \param[in]     h               Length of the h dimension of input.
 *  \param[in]     d               Length of the d dimension of input.
 *  \param[in]     d2              Length of the d dimension of freqs.
 *  \param[in]     stride_s_or_t   Stride of the s (sbhd/bshd)/t (thd) dimension of input.
 *  \param[in]     stride_b        Stride of the b dimension of input. (0 for thd).
 *  \param[in]     stride_h        Stride of the h dimension of input.
 *  \param[in]     stride_d        Stride of the d dimension of input.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                             const NVTETensor freqs, const NVTETensor start_positions,
                             NVTETensor output, const NVTE_QKV_Format qkv_format,
                             const bool interleaved, const int cp_size, const int cp_rank,
                             const int s, const int b, const int h, const int d, const int d2,
                             const int stride_s_or_t, const int stride_b, const int stride_h,
                             const int stride_d, cudaStream_t stream);

/*! \brief Compute the backward of the fused rope.
 *
 *  \param[in]     output_grads    Incoming gradient tensor for backward.
 *  \param[in]     cu_seqlens      The cumulative sum of sequence lengths tensor.
 *                                 (Required for the thd format, empty tensor for other formats)
 *  \param[in]     freqs           The freqs tensor.
 *  \param[in]     start_positions The beginning offsets for applying RoPE embeddings.
 *  \param[out]    input_grads     Input gradient tensor to calculate.
 *  \param[in]     qkv_format      QKV format.
 *  \param[in]     interleaved     Whether to use interleaved rotary position embedding.
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Length of the s dimension of output_grads.
 *  \param[in]     b               Length of the b dimension of output_grads.
 *  \param[in]     h               Length of the h dimension of output_grads.
 *  \param[in]     d               Length of the d dimension of output_grads.
 *  \param[in]     d2              Length of the d dimension of freqs.
 *  \param[in]     stride_s_or_t   Stride of the s (sbhd/bshd)/t (thd) dimension of output_grads.
 *  \param[in]     stride_b        Stride of the b dimension of output_grads. (0 for thd).
 *  \param[in]     stride_h        Stride of the h dimension of output_grads.
 *  \param[in]     stride_d        Stride of the d dimension of output_grads.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_rope_backward(const NVTETensor output_grads, const NVTETensor cu_seqlens,
                              const NVTETensor freqs, const NVTETensor start_positions,
                              NVTETensor input_grads, const NVTE_QKV_Format qkv_format,
                              const bool interleaved, const int cp_size, const int cp_rank,
                              const int s, const int b, const int h, const int d, const int d2,
                              const int stride_s_or_t, const int stride_b, const int stride_h,
                              const int stride_d, cudaStream_t stream);

/*! \brief Apply rotary positional embedding to the combined QKV input tensor.
 *
 *  \param[in]     qkv_input       Combined QKV input tensor for fused rope.
 *  \param[in]     q_freqs         The freqs tensor for Q.
 *  \param[in]     k_freqs         The freqs tensor for K.
 *  \param[in]     start_positions The beginning offsets for applying RoPE embeddings.
 *  \param[out]    q_out           Output tensor for Q.
 *  \param[out]    k_out           Output tensor for K.
 *  \param[out]    v_out           Output tensor for V.
 *  \param[in]     qkv_format      QKV format.
 *  \param[in]     interleaved     Whether to use interleaved rotary position embedding.
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Length of the s dimension of input.
 *  \param[in]     b               Length of the b dimension of input.
 *  \param[in]     h               Length of the h dimension of input.
 *  \param[in]     d               Length of the d dimension of input.
 *  \param[in]     d2              Length of the d dimension of freqs.
 *  \param[in]     qkv_split_arg_list_0  The hidden size for Q.
 *  \param[in]     qkv_split_arg_list_1  The hidden size for K.
 *  \param[in]     qkv_split_arg_list_2  The hidden size for V.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_qkv_rope_forward(const NVTETensor qkv_input, const NVTETensor q_freqs,
                                 const NVTETensor k_freqs, const NVTETensor start_positions,
                                 NVTETensor q_out, NVTETensor k_out, NVTETensor v_out,
                                 const NVTE_QKV_Format qkv_format, const bool interleaved,
                                 const int cp_size, const int cp_rank, const int s, const int b,
                                 const int h, const int d, const int d2,
                                 const int qkv_split_arg_list_0, const int qkv_split_arg_list_1,
                                 const int qkv_split_arg_list_2, cudaStream_t stream);

/*! \brief Compute the backward of the fused qkv rope.
 *
 *  \param[in]     q_grad_out      Incoming gradient tensor for Q.
 *  \param[in]     k_grad_out      Incoming gradient tensor for K.
 *  \param[in]     v_grad_out      Incoming gradient tensor for V.
 *  \param[in]     q_freqs         The freqs tensor for Q.
 *  \param[in]     k_freqs         The freqs tensor for K.
 *  \param[out]    qkv_grad_input  Input gradient tensor to calculate.
 *  \param[in]     qkv_format      QKV format.
 *  \param[in]     interleaved     Whether to use interleaved rotary position embedding.
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Length of the s dimension of input.
 *  \param[in]     b               Length of the b dimension of input.
 *  \param[in]     h               Length of the h dimension of input.
 *  \param[in]     d               Length of the d dimension of input.
 *  \param[in]     d2              Length of the d dimension of freqs.
 *  \param[in]     qkv_split_arg_list_0  The hidden size for Q.
 *  \param[in]     qkv_split_arg_list_1  The hidden size for K.
 *  \param[in]     qkv_split_arg_list_2  The hidden size for V.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_qkv_rope_backward(const NVTETensor q_grad_out, const NVTETensor k_grad_out,
                                  const NVTETensor v_grad_out, const NVTETensor q_freqs,
                                  const NVTETensor k_freqs, NVTETensor qkv_grad_input,
                                  const NVTE_QKV_Format qkv_format, const bool interleaved,
                                  const int cp_size, const int cp_rank, const int s, const int b,
                                  const int h, const int d, const int d2,
                                  const int qkv_split_arg_list_0, const int qkv_split_arg_list_1,
                                  const int qkv_split_arg_list_2, cudaStream_t stream);

/*! \brief Apply MLA YARN rotary positional embedding to the query tensor.
 *
 *  YARN rotation is applied to the *tail* emb_dim elements of each head,
 *  leaving the first qk_head_dim elements untouched.
 *
 *  \param[in]     input           Input Q tensor. Shape: [s,b,h,d] / [b,s,h,d] / [t,h,d].
 *  \param[in]     cu_seqlens      Cumulative sequence lengths (THD only, empty otherwise).
 *  \param[in]     cos_table       Pre-computed cosine table [max_seq_len, emb_dim].
 *  \param[in]     sin_table       Pre-computed sine table   [max_seq_len, emb_dim].
 *  \param[out]    output          Output tensor (same shape as input).
 *  \param[in]     qkv_format      QKV format (SBHD / BSHD / THD).
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Sequence length (or max_s for THD).
 *  \param[in]     b               Batch size (or num sequences for THD).
 *  \param[in]     h               Number of heads.
 *  \param[in]     d               Head dimension = qk_head_dim + emb_dim.
 *  \param[in]     qk_head_dim     Compressed latent dimension (untouched prefix).
 *  \param[in]     emb_dim         Rotary embedding dimension (must be divisible by 4).
 *  \param[in]     stride_s_or_t   Stride of s/t dimension.
 *  \param[in]     stride_b        Stride of b dimension (0 for THD).
 *  \param[in]     stride_h        Stride of h dimension.
 *  \param[in]     stream          CUDA stream.
 */
void nvte_mla_rope_q_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                             const NVTETensor cos_table, const NVTETensor sin_table,
                             NVTETensor output, const NVTE_QKV_Format qkv_format,
                             const int cp_size, const int cp_rank, const int s, const int b,
                             const int h, const int d, const int qk_head_dim,
                             const int emb_dim, const int stride_s_or_t, const int stride_b,
                             const int stride_h, cudaStream_t stream);

/*! \brief Backward of MLA YARN RoPE for the query tensor. */
void nvte_mla_rope_q_backward(const NVTETensor grad_out, const NVTETensor cu_seqlens,
                              const NVTETensor cos_table, const NVTETensor sin_table,
                              NVTETensor grad_in, const NVTE_QKV_Format qkv_format,
                              const int cp_size, const int cp_rank, const int s, const int b,
                              const int h, const int d, const int qk_head_dim,
                              const int emb_dim, const int stride_s_or_t, const int stride_b,
                              const int stride_h, cudaStream_t stream);

/*! \brief Apply MLA YARN RoPE to key-value, splitting KV and broadcasting rotated pos emb.
 *
 *  \param[in]     kv              Combined KV tensor [s,b,h,k_dim+v_dim] / [t,h,k_dim+v_dim].
 *  \param[in]     k_pos_emb       Positional embedding [s,b,emb_dim] / [t,emb_dim] (single-head).
 *  \param[in]     cos_table       Pre-computed cosine table [max_seq_len, emb_dim].
 *  \param[in]     sin_table       Pre-computed sine table   [max_seq_len, emb_dim].
 *  \param[out]    o_key           Output key   [s,b,h,k_dim+emb_dim] / [t,h,k_dim+emb_dim].
 *  \param[out]    o_value         Output value [s,b,h,v_dim]         / [t,h,v_dim].
 *  \param[in]     cu_seqlens      Cumulative sequence lengths (THD only, empty otherwise).
 *  \param[in]     qkv_format      QKV format.
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Sequence length (or max_s for THD).
 *  \param[in]     b               Batch size (or num sequences for THD).
 *  \param[in]     h               Number of heads.
 *  \param[in]     k_dim           Compressed key dimension.
 *  \param[in]     v_dim           Value dimension.
 *  \param[in]     emb_dim         Rotary embedding dimension (must be divisible by 4).
 *  \param[in]     stride_kv_s     KV stride of s/t dimension.
 *  \param[in]     stride_kv_b     KV stride of b dimension (0 for THD).
 *  \param[in]     stride_kv_h     KV stride of h dimension.
 *  \param[in]     stride_emb_s    K_POS_EMB stride of s/t dimension.
 *  \param[in]     stride_emb_b    K_POS_EMB stride of b dimension (0 for THD).
 *  \param[in]     stream          CUDA stream.
 */
void nvte_mla_rope_kv_forward(
    const NVTETensor kv, const NVTETensor k_pos_emb, const NVTETensor cos_table,
    const NVTETensor sin_table, NVTETensor o_key, NVTETensor o_value, const NVTETensor cu_seqlens,
    const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank, const int s,
    const int b, const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int stride_kv_s, const int stride_kv_b, const int stride_kv_h, const int stride_emb_s,
    const int stride_emb_b, cudaStream_t stream);

/*! \brief Backward of MLA YARN RoPE for key-value.
 *
 *  \param[in]     dk              Gradient of key   [s,b,h,k_dim+emb_dim] / [t,h,k_dim+emb_dim].
 *  \param[in]     dv              Gradient of value  [s,b,h,v_dim]        / [t,h,v_dim].
 *  \param[in]     cos_table       Pre-computed cosine table [max_seq_len, emb_dim].
 *  \param[in]     sin_table       Pre-computed sine table   [max_seq_len, emb_dim].
 *  \param[out]    dkv             Gradient of KV    [s,b,h,k_dim+v_dim]  / [t,h,k_dim+v_dim].
 *  \param[out]    d_emb           Gradient of K_POS_EMB [s,b,emb_dim]    / [t,emb_dim].
 *  \param[in]     cu_seqlens      Cumulative sequence lengths (THD only, empty otherwise).
 *  \param[in]     qkv_format      QKV format.
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Sequence length.
 *  \param[in]     b               Batch size.
 *  \param[in]     h               Number of heads.
 *  \param[in]     k_dim           Compressed key dimension.
 *  \param[in]     v_dim           Value dimension.
 *  \param[in]     emb_dim         Rotary embedding dimension (must be divisible by 4).
 *  \param[in]     stride_dk_s     dK stride of s/t dimension.
 *  \param[in]     stride_dk_b     dK stride of b dimension (0 for THD).
 *  \param[in]     stride_dk_h     dK stride of h dimension.
 *  \param[in]     stride_dv_s     dV stride of s/t dimension.
 *  \param[in]     stride_dv_b     dV stride of b dimension (0 for THD).
 *  \param[in]     stride_dv_h     dV stride of h dimension.
 *  \param[in]     o_demb_stride_s d_emb stride of s/t dimension.
 *  \param[in]     o_demb_stride_b d_emb stride of b dimension (0 for THD).
 *  \param[in]     stream          CUDA stream.
 */
void nvte_mla_rope_kv_backward(
    const NVTETensor dk, const NVTETensor dv, const NVTETensor cos_table,
    const NVTETensor sin_table, NVTETensor dkv, NVTETensor d_emb, const NVTETensor cu_seqlens,
    const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank, const int s,
    const int b, const int h, const int k_dim, const int v_dim, const int emb_dim,
    const int stride_dk_s, const int stride_dk_b, const int stride_dk_h, const int stride_dv_s,
    const int stride_dv_b, const int stride_dv_h, const int o_demb_stride_s,
    const int o_demb_stride_b, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
