/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
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
void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor freqs,
                             NVTETensor output, const int s, const int b,
                             const int h, const int d, const int d2,
                             const int stride_s, const int stride_b,
                             const int stride_h, const int stride_d,
                             const int o_stride_s, const int o_stride_b,
                             const int o_stride_h, const int o_stride_d,
                             cudaStream_t stream);

/*! \brief Compute the backward of the fused rope.
 *
 *  \param[in]     incoming_grads  Input gradient tensor for backward.
 *  \param[in]     freqs           The freqs tensor.
 *  \param[out]    output_grads    Output gradient tensor.
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
void nvte_fused_rope_backward(const NVTETensor incoming_grads,
                              const NVTETensor freqs, NVTETensor output_grads,
                              const int s, const int b, const int h,
                              const int d, const int d2, const int stride_s,
                              const int stride_b, const int stride_h,
                              const int stride_d, const int o_stride_s,
                              const int o_stride_b, const int o_stride_h,
                              const int o_stride_d, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
