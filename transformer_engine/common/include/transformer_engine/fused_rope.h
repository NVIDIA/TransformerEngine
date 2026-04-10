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

/*! \brief Apply YARN RoPE to MLA query tensor (forward).
 *
 *  Reads the last emb_dim elements interleaved, applies YARN rotation with
 *  split cos/sin, and writes de-interleaved. First qk_head_dim elements are
 *  copied unchanged. Input is [total_seqlen, h, d] (flattened from SBHD or THD).
 *
 *  \param[in]     q_input         Input Q tensor.
 *  \param[in]     cos             Pre-computed cosine tensor [max_s, emb_dim].
 *  \param[in]     sin             Pre-computed sine tensor [max_s, emb_dim].
 *  \param[out]    q_output        Output Q tensor (same shape as input).
 *  \param[in]     cu_seqlens      Cumulative sequence lengths for THD (empty for SBHD).
 *  \param[in]     qk_head_dim     Non-RoPE prefix dimension per head.
 *  \param[in]     emb_dim         RoPE embedding dimension.
 *  \param[in]     h               Number of heads.
 *  \param[in]     d               Total head dimension (qk_head_dim + emb_dim).
 *  \param[in]     total_seqlen    Total tokens (s*b for SBHD, total_t for THD).
 *  \param[in]     s               Sequence length (SBHD) or max_s (THD).
 *  \param[in]     b               Batch size (SBHD) or num_seqs (THD).
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     stream          CUDA stream.
 */
void nvte_fused_mla_rope_q_forward(const NVTETensor q_input, const NVTETensor cos,
                                   const NVTETensor sin, NVTETensor q_output,
                                   const NVTETensor cu_seqlens, const int qk_head_dim,
                                   const int emb_dim, const int h, const int d,
                                   const int total_seqlen, const int s, const int b,
                                   const int cp_size, const int cp_rank, cudaStream_t stream);

/*! \brief Backward of YARN RoPE for MLA query tensor. */
void nvte_fused_mla_rope_q_backward(const NVTETensor grad_output, const NVTETensor cos,
                                    const NVTETensor sin, NVTETensor grad_input,
                                    const NVTETensor cu_seqlens, const int qk_head_dim,
                                    const int emb_dim, const int h, const int d,
                                    const int total_seqlen, const int s, const int b,
                                    const int cp_size, const int cp_rank, cudaStream_t stream);

/*! \brief Apply YARN RoPE to MLA key-value tensor (forward).
 *
 *  Splits KV into key and value, applies YARN rotation to k_pos_emb (shared
 *  across heads), concatenates rotated embedding to each head of output key.
 *
 *  \param[in]     kv_input        Input KV tensor [total_t, h, k_dim+v_dim].
 *  \param[in]     k_pos_emb       Positional embedding [total_t, emb_dim].
 *  \param[in]     cos             Pre-computed cosine [max_s, emb_dim].
 *  \param[in]     sin             Pre-computed sine [max_s, emb_dim].
 *  \param[out]    o_key           Output key [total_t, h, k_dim+emb_dim].
 *  \param[out]    o_value         Output value [total_t, h, v_dim].
 *  \param[in]     cu_seqlens      Cumulative sequence lengths for THD (empty for SBHD).
 *  \param[in]     emb_dim         RoPE embedding dimension.
 *  \param[in]     k_dim           Key dimension per head (from KV).
 *  \param[in]     v_dim           Value dimension per head (from KV).
 *  \param[in]     h               Number of heads.
 *  \param[in]     total_seqlen    Total tokens.
 *  \param[in]     s               Sequence length (SBHD) or max_s (THD).
 *  \param[in]     b               Batch size (SBHD) or num_seqs (THD).
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     stream          CUDA stream.
 */
void nvte_fused_mla_rope_kv_forward(const NVTETensor kv_input, const NVTETensor k_pos_emb,
                                    const NVTETensor cos, const NVTETensor sin, NVTETensor o_key,
                                    NVTETensor o_value, const NVTETensor cu_seqlens,
                                    const int emb_dim, const int k_dim, const int v_dim,
                                    const int h, const int total_seqlen, const int s, const int b,
                                    const int cp_size, const int cp_rank, cudaStream_t stream);

/*! \brief Backward of YARN RoPE for MLA key-value tensor. */
void nvte_fused_mla_rope_kv_backward(const NVTETensor dk, const NVTETensor dv, const NVTETensor cos,
                                     const NVTETensor sin, NVTETensor d_kv, NVTETensor d_emb,
                                     const NVTETensor cu_seqlens, const int emb_dim,
                                     const int k_dim, const int v_dim, const int h,
                                     const int total_seqlen, const int s, const int b,
                                     const int cp_size, const int cp_rank, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
