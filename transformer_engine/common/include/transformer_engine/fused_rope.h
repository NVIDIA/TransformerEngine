/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROPE_H_
#define TRANSFORMER_ENGINE_FUSED_ROPE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Apply rotary positional embedding to the input tensor.
 *
 *  \param[in]     input           Input tensor for fused rope.
 *  \param[in]     freqs           The freqs tensor.
 *  \param[out]    output          Output tensor.
 *  \param[in]     s               Length of the s dimension of input.
 *  \param[in]     b               Length of the b dimension of input.
 *  \param[in]     h               Length of the h dimension of input.
 *  \param[in]     d               Length of the d dimension of input.
 *  \param[in]     d2              Length of the d dimension of freqs.
 *  \param[in]     stride_s        Stride of the s dimension of input.
 *  \param[in]     stride_b        Stride of the b dimension of input.
 *  \param[in]     stride_h        Stride of the h dimension of input.
 *  \param[in]     stride_d        Stride of the d dimension of input.
 *  \param[in]     o_stride_s      Stride of the s dimension of output.
 *  \param[in]     o_stride_b      Stride of the b dimension of output.
 *  \param[in]     o_stride_h      Stride of the h dimension of output.
 *  \param[in]     o_stride_d      Stride of the d dimension of output.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor freqs, NVTETensor output,
                             const int s, const int b, const int h, const int d, const int d2,
                             const int stride_s, const int stride_b, const int stride_h,
                             const int stride_d, const int o_stride_s, const int o_stride_b,
                             const int o_stride_h, const int o_stride_d, cudaStream_t stream);

/*! \brief Compute the backward of the fused rope.
 *
 *  \param[in]     output_grads    Incoming gradient tensor for backward.
 *  \param[in]     freqs           The freqs tensor.
 *  \param[out]    input_grads     Input gradient tensor to calculate.
 *  \param[in]     s               Length of the s dimension of output_grads.
 *  \param[in]     b               Length of the b dimension of output_grads.
 *  \param[in]     h               Length of the h dimension of output_grads.
 *  \param[in]     d               Length of the d dimension of output_grads.
 *  \param[in]     d2              Length of the d dimension of freqs.
 *  \param[in]     stride_s        Stride of the s dimension of output_grads.
 *  \param[in]     stride_b        Stride of the b dimension of output_grads.
 *  \param[in]     stride_h        Stride of the h dimension of output_grads.
 *  \param[in]     stride_d        Stride of the d dimension of output_grads.
 *  \param[in]     o_stride_s      Stride of the s dimension of input_grads.
 *  \param[in]     o_stride_b      Stride of the b dimension of input_grads.
 *  \param[in]     o_stride_h      Stride of the h dimension of input_grads.
 *  \param[in]     o_stride_d      Stride of the d dimension of input_grads.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_rope_backward(const NVTETensor output_grads, const NVTETensor freqs,
                              NVTETensor input_grads, const int s, const int b, const int h,
                              const int d, const int d2, const int stride_s, const int stride_b,
                              const int stride_h, const int stride_d, const int o_stride_s,
                              const int o_stride_b, const int o_stride_h, const int o_stride_d,
                              cudaStream_t stream);

/*! \brief Apply rotary positional embedding to the input tensor in thd format.
 *
 *  \param[in]     input         Input tensor for fused rope.
 *  \param[in]     cu_seqlens    The cumulative sum of sequence lengths tensor.
 *  \param[in]     freqs         The freqs tensor.
 *  \param[out]    output        Output tensor.
 *  \param[in]     max_s         Max sequence length.
 *  \param[in]     b             Batch size.
 *  \param[in]     h             Length of the h dimension of input.
 *  \param[in]     d             Length of the d dimension of input.
 *  \param[in]     d2            Length of the d dimension of freqs.
 *  \param[in]     stride_t      Stride of the t dimension of input.
 *  \param[in]     stride_h      Stride of the h dimension of input.
 *  \param[in]     stride_d      Stride of the d dimension of input.
 *  \param[in]     o_stride_t    Stride of the t dimension of output.
 *  \param[in]     o_stride_h    Stride of the h dimension of output.
 *  \param[in]     o_stride_d    Stride of the d dimension of output.
 *  \param[in]     stream        CUDA stream used for the operation.
 */
void nvte_fused_rope_thd_forward(const NVTETensor input, const NVTETensor cu_seqlens,
                                 const NVTETensor freqs, NVTETensor output, const int max_s,
                                 const int b, const int h, const int d, const int d2,
                                 const int stride_t, const int stride_h, const int stride_d,
                                 const int o_stride_t, const int o_stride_h, const int o_stride_d,
                                 cudaStream_t stream);

/*! \brief Compute the backward of the fused rope in thd format.
 *
 *  \param[in]     output_grads  Incoming gradient tensor for backward.
 *  \param[in]     cu_seqlens    The cumulative sum of sequence lengths tensor.
 *  \param[in]     freqs         The freqs tensor.
 *  \param[out]    input_grads   Input gradient to calculate.
 *  \param[in]     max_s         Max sequence length.
 *  \param[in]     b             Batch size.
 *  \param[in]     h             Length of the h dimension of output_grads.
 *  \param[in]     d             Length of the d dimension of output_grads.
 *  \param[in]     d2            Length of the d dimension of freqs.
 *  \param[in]     stride_t      Stride of the t dimension of output_grads.
 *  \param[in]     stride_h      Stride of the h dimension of output_grads.
 *  \param[in]     stride_d      Stride of the d dimension of output_grads.
 *  \param[in]     o_stride_t    Stride of the t dimension of input_grads.
 *  \param[in]     o_stride_h    Stride of the h dimension of input_grads.
 *  \param[in]     o_stride_d    Stride of the d dimension of input_grads.
 *  \param[in]     stream        CUDA stream used for the operation.
 */
void nvte_fused_rope_thd_backward(const NVTETensor output_grads, const NVTETensor cu_seqlens,
                                  const NVTETensor freqs, NVTETensor input_grads, const int max_s,
                                  const int b, const int h, const int d, const int d2,
                                  const int stride_t, const int stride_h, const int stride_d,
                                  const int o_stride_t, const int o_stride_h, const int o_stride_d,
                                  cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
