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

/*! \brief Apply YARN MLA RoPE to query tensor (forward).
 *
 *  Reads interleaved emb region (x[2i], x[2i+1]), writes non-interleaved
 *  (first_half | second_half). The leading qk_head_dim elements per head are
 *  passed through unchanged.
 *
 *  \param[in]     input           Q tensor.
 *  \param[in]     cu_seqlens      Cumulative seq lengths (required for thd, else empty).
 *  \param[in]     cos             Precomputed cosines [max_s, 1, 1, emb_dim], float32.
 *  \param[in]     sin             Precomputed sines  [max_s, 1, 1, emb_dim], float32.
 *  \param[out]    output          Output Q tensor (same shape as input).
 *  \param[in]     qkv_format      QKV format (SBHD or THD).
 *  \param[in]     cp_size         Context parallel world size.
 *  \param[in]     cp_rank         Context parallel rank.
 *  \param[in]     s               Seq length (or max_s for THD).
 *  \param[in]     b               Batch size (or num_seqs for THD).
 *  \param[in]     h               Number of heads.
 *  \param[in]     qk_head_dim     Non-rotary (nope) head dim.
 *  \param[in]     emb_dim         Rotary embedding dim.
 *  \param[in]     stride_s_or_t   Token/seq stride of input.
 *  \param[in]     stride_b        Batch stride of input (0 for thd).
 *  \param[in]     stride_h        Head stride of input.
 *  \param[in]     stream          CUDA stream.
 */
void nvte_mla_rope_q_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                              const NVTETensor cos, const NVTETensor sin, NVTETensor output,
                              const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                              const int s, const int b, const int h, const int qk_head_dim,
                              const int emb_dim, const int stride_s_or_t, const int stride_b,
                              const int stride_h, cudaStream_t stream);

/*! \brief Backward of nvte_mla_rope_q_forward. */
void nvte_mla_rope_q_backward(const NVTETensor grad, const NVTETensor cu_seqlens,
                               const NVTETensor cos, const NVTETensor sin, NVTETensor dst,
                               const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                               const int s, const int b, const int h, const int qk_head_dim,
                               const int emb_dim, const int stride_s_or_t, const int stride_b,
                               const int stride_h, cudaStream_t stream);

/*! \brief Apply YARN MLA RoPE to packed KV tensor (forward).
 *
 *  Splits kv into key (k_dim) and value (v_dim) per head.
 *  Rotates k_pos_emb (interleaved, h=1) and appends to key.
 *  Output key width = k_dim + emb_dim; value is unchanged.
 *
 *  \param[in]     kv                  Packed KV tensor.
 *  \param[in]     k_pos_emb           Per-token positional embedding (h=1, interleaved).
 *  \param[in]     cu_seqlens          Cumulative seq lengths (required for thd, else empty).
 *  \param[in]     cos                 Cosines [max_s, 1, 1, emb_dim], float32.
 *  \param[in]     sin                 Sines   [max_s, 1, 1, emb_dim], float32.
 *  \param[out]    key_out             Key output tensor (k_dim + emb_dim per head).
 *  \param[out]    val_out             Value output tensor (v_dim per head).
 *  \param[in]     qkv_format          QKV format (SBHD or THD).
 *  \param[in]     cp_size             Context parallel world size.
 *  \param[in]     cp_rank             Context parallel rank.
 *  \param[in]     s                   Seq length (or max_s for THD).
 *  \param[in]     b                   Batch size (or num_seqs for THD).
 *  \param[in]     h                   Number of heads.
 *  \param[in]     k_dim               Non-rotary key dim.
 *  \param[in]     v_dim               Value dim.
 *  \param[in]     emb_dim             Rotary embedding dim.
 *  \param[in]     kv_stride_s_or_t    Token/seq stride of kv.
 *  \param[in]     kv_stride_b         Batch stride of kv (0 for thd).
 *  \param[in]     kv_stride_h         Head stride of kv.
 *  \param[in]     emb_stride_s_or_t   Token/seq stride of k_pos_emb.
 *  \param[in]     emb_stride_b        Batch stride of k_pos_emb (0 for thd).
 *  \param[in]     stream              CUDA stream.
 */
void nvte_mla_rope_kv_forward(const NVTETensor kv, const NVTETensor k_pos_emb,
                               const NVTETensor cu_seqlens, const NVTETensor cos,
                               const NVTETensor sin, NVTETensor key_out, NVTETensor val_out,
                               const NVTE_QKV_Format qkv_format, const int cp_size, const int cp_rank,
                               const int s, const int b, const int h,
                               const int k_dim, const int v_dim, const int emb_dim,
                               const int kv_stride_s_or_t, const int kv_stride_b,
                               const int kv_stride_h,
                               const int emb_stride_s_or_t, const int emb_stride_b,
                               cudaStream_t stream);

/*! \brief Backward of nvte_mla_rope_kv_forward. */
void nvte_mla_rope_kv_backward(const NVTETensor dk, const NVTETensor dv,
                                const NVTETensor cu_seqlens, const NVTETensor cos,
                                const NVTETensor sin, NVTETensor d_kv_out, NVTETensor d_emb_out,
                                const NVTE_QKV_Format qkv_format, const int cp_size,
                                const int cp_rank, const int s, const int b, const int h,
                                const int k_dim, const int v_dim, const int emb_dim,
                                const int dk_stride_s_or_t, const int dk_stride_b,
                                const int dk_stride_h, const int dv_stride_s_or_t,
                                const int dv_stride_b, const int dv_stride_h,
                                const int d_emb_stride_s_or_t, const int d_emb_stride_b,
                                cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
