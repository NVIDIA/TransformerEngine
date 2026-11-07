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

/*! \brief Split multi-latent attention's key/value, apply RoPE to the key and quantize key and
 *         value to MXFP8.
 *
 *  Each head of kv holds 128 key dimensions without RoPE followed by 128 value dimensions.
 *  The key is those 128 dimensions followed by the 64 dimensions of k_rope with RoPE
 *  applied, the same for all heads. For token t, batch entry b and i < 32, with
 *  (x1, x2) = (k_rope[t, b, 2i], k_rope[t, b, 2i + 1]):
 *
 *      key[t, b, h, 128 + i]      = x1 * cos[t, i]      - x2 * sin[t, i]
 *      key[t, b, h, 128 + 32 + i] = x2 * cos[t, 32 + i] + x1 * sin[t, 32 + i]
 *
 *  This is the RoPE of Megatron-LM's MLA kernels (interleaved pairs in, the two halves out),
 *  with their roundings: with BF16 cos and sin, fma(x1, cos, -(x2 * sin)) and
 *  fma(x1, sin, x2 * cos) with every operation rounded to BF16; with FP32 cos and sin, the
 *  same in FP32, rounded once to BF16.
 *
 *  key and value are written as the MXFP8 quantization of their [s, b * h * d] views, row-wise
 *  (32 elements of a head) and column-wise (32 tokens), with E4M3 data and compact E8M0 scaling
 *  factors: the bytes the MXFP8 quantizer produces for the BF16 key and value.
 *
 *  Requires SM 10.0+, an even number of heads and a sequence length that is a multiple of 32.
 *
 *  \param[in]     kv                Key/value, [s, b, h, 256] in BF16.
 *  \param[in]     k_rope            Key RoPE dimensions shared by the heads, s * b rows
 *                                   (token-major) of 64 BF16 values.
 *  \param[in]     k_rope_stride     Elements from one row of k_rope to the next, a multiple
 *                                   of 8.
 *  \param[in]     cos               Cosines, [max_s, 64] in BF16 or FP32; row t is used for
 *                                   token t.
 *  \param[in]     sin               Sines, same shape and type as cos.
 *  \param[in,out] key               MXFP8 key of shape [s, b * h * 192], with row-wise and
 *                                   column-wise data.
 *  \param[in,out] value             MXFP8 value of shape [s, b * h * 128], with row-wise and
 *                                   column-wise data.
 *  \param[in]     stream            CUDA stream used for the operation.
 */
void nvte_fused_mla_kv_rope_mxfp8(const NVTETensor kv, const NVTETensor k_rope,
                                  const int k_rope_stride, const NVTETensor cos,
                                  const NVTETensor sin, NVTETensor key, NVTETensor value,
                                  cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
