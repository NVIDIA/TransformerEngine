/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation.h
 *  \brief Activation functions.
 */

#ifndef TRANSFORMER_ENGINE_ACTIVATION_H_
#define TRANSFORMER_ENGINE_ACTIVATION_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Supported activations: GeLU, SiLU, ReLU, QuickGeLU, SquaredReLU */

/*! \brief Computes activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */

enum class NVTE_Activation_Type {
  GELU,
  GEGLU,
  GLU,
  SILU,
  SWIGLU,
  RELU,
  REGLU,
  QGELU,
  QGEGLU,
  SRELU,
  SREGLU,
  CLAMPED_SWIGLU
};

/*! \brief Computes the GeLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_gelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the GeLU activation of the grouped input.
 *         If the scaling mode of the grouped output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_gelu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream);

/*! \brief Computes the SiLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the SiLU activation of the grouped input.
 *         If the scaling mode of the grouped output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_silu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream);

/*! \brief Computes the ReLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_relu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the ReLU activation of the grouped input.
 *         If the scaling mode of the grouped output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_relu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream);

/*! \brief Computes the Quick GeLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_qgelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the Quick GeLU activation of the grouped input.
 *         If the scaling mode of the grouped output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_qgelu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream);

/*! \brief Computes the Squared ReLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_srelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the Squared ReLU activation of the grouped input.
 *         If the scaling mode of the grouped output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_srelu(const NVTEGroupedTensor input, NVTEGroupedTensor output, cudaStream_t stream);

/*! \brief Computes the GeLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream);

/*! \brief Computes the GeLU activation gradient of the grouped input.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     grad      Incoming grouped gradient.
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_dgelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTETensor output, cudaStream_t stream);

/*! \brief Computes the SiLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dsilu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream);

/*! \brief Computes the SiLU activation gradient of the grouped input.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     grad      Incoming grouped gradient.
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_dsilu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTETensor output, cudaStream_t stream);

/*! \brief Computes the ReLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_drelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream);

/*! \brief Computes the ReLU activation gradient of the grouped input.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     grad      Incoming grouped gradient.
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_drelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                      NVTETensor output, cudaStream_t stream);

/*! \brief Computes the Quick GeLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dqgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

/*! \brief Computes the Quick GeLU activation gradient of the grouped input.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     grad      Incoming grouped gradient.
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_dqgelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                       NVTETensor output, cudaStream_t stream);

/*! \brief Computes the Squared ReLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dsrelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

/*! \brief Computes the Squared ReLU activation gradient of the grouped input.
 *         If the scaling mode of the output grouped tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *         For grouped tensors with a varying last dimension, the last dimension must be a multiple of 128.
 *
 *  \param[in]     grad      Incoming grouped gradient.
 *  \param[in]     input     Input grouped tensor for activation.
 *  \param[in,out] output    Output grouped tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_group_dsrelu(const NVTEGroupedTensor grad, const NVTEGroupedTensor input,
                       NVTETensor output, cudaStream_t stream);

/*! \brief Computes the GLU (Gated Linear Unit) activation of the input.
 *         GLU(a,b) = sigmoid(a) * b
 *         See "Language Modeling with Gated Convolutional Networks" (arXiv:1612.08083)
 *         and "GLU Variants Improve Transformer" (arXiv:2002.05202).
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes sigmoid(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_glu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the GLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
               cudaStream_t stream);

/*! \brief Computes the gated GeLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_geglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Swish activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Swish activation of the input used in GPT OSS.
 *
 *  \deprecated This function has been deprecated in favor of nvte_clamped_swiglu_v2,
 *              which exposes a configurable offset for the linear (gate) component.
 *              This API is preserved for backward compatibility and is equivalent to
 *              calling nvte_clamped_swiglu_v2 with glu_linear_offset = 1.0.
 *
 *        See https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250
 *        This Gated activation has two differences compared to the original SwiGLU
 *           1. Both gate and pre-activations are clipped based on parameter limit.
 *           2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation inspired
 *           by original GELU paper https://arxiv.org/pdf/1606.08415
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     limit     Clipping limits for gate and pre-activation.
 *  \param[in]     alpha     Scaling factor for the sigmoid function used in the activation.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_clamped_swiglu(const NVTETensor input, NVTETensor output, float limit, float alpha,
                         cudaStream_t stream);

/*! \brief Computes the gated Swish activation of the input used in GPT OSS, with a configurable
 *         offset for the linear (gate) component after clamping.
 *
 *        See https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250
 *        This Gated activation has two differences compared to the original SwiGLU
 *           1. Both gate and pre-activations are clipped based on parameter limit.
 *           2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation inspired
 *           by original GELU paper https://arxiv.org/pdf/1606.08415
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input              Input tensor of shape [N, H * 2].
 *  \param[in,out] output             Output tensor of shape [N, H].
 *                                    It computes Act(input[N, :H]) x (input[N, H:] + glu_linear_offset)
 *  \param[in]     limit              Clipping limits for gate and pre-activation.
 *  \param[in]     alpha              Scaling factor for the sigmoid function used in the activation.
 *  \param[in]     glu_linear_offset  Offset added to the linear component after clamping (typically 1.0).
 *  \param[in]     stream             CUDA stream used for the operation.
 */
void nvte_clamped_swiglu_v2(const NVTETensor input, NVTETensor output, float limit, float alpha,
                            float glu_linear_offset, cudaStream_t stream);

/*! \brief Computes ScaledSwiGLU without materializing GLU deinterleave.
 *
 *  Computes output = SwiGLU(input) * act_scales[:, None].
 *  If glu_interleave_size > 0, input is interpreted as interleaved
 *  [activation_block, linear_block] chunks of that size.
 *
 *  \param[in]     input                Input tensor of shape [N, H * 2].
 *  \param[in]     act_scales           Row-wise activation scales of shape [N].
 *  \param[in,out] output               Output tensor of shape [N, H].
 *  \param[in]     glu_interleave_size  GLU interleave chunk size, or 0 for non-interleaved layout.
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_scaled_swiglu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                        int64_t glu_interleave_size, cudaStream_t stream);

/*! \brief Computes ScaledClampedSwiGLU without materializing GLU deinterleave.
 *
 *  Computes output = ClampedSwiGLU(input) * act_scales[:, None].
 *  This uses the same clamping, alpha, and linear-offset semantics as
 *  nvte_clamped_swiglu_v2.
 *
 *  \param[in]     input                Input tensor of shape [N, H * 2].
 *  \param[in]     act_scales           Row-wise activation scales of shape [N].
 *  \param[in,out] output               Output tensor of shape [N, H].
 *  \param[in]     limit                Clipping limit.
 *  \param[in]     alpha                Activation sigmoid alpha.
 *  \param[in]     glu_linear_offset    Offset added to linear component after clamping.
 *  \param[in]     glu_interleave_size  GLU interleave chunk size, or 0 for non-interleaved layout.
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_scaled_clamped_swiglu(const NVTETensor input, const NVTETensor act_scales,
                                NVTETensor output, float limit, float alpha,
                                float glu_linear_offset, int64_t glu_interleave_size,
                                cudaStream_t stream);

/*! \brief Computes the gated ReLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_reglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Quick GeLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_qgeglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Squared ReLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_sreglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated GeLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

/*! \brief Computes the gated Swish activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream);

/*! \brief Computes the gradient of gated Swish activation of the input used in GPT OSS.
 *
 *  \deprecated This function has been deprecated in favor of nvte_clamped_dswiglu_v2,
 *              which exposes a configurable offset for the linear (gate) component.
 *              This API is preserved for backward compatibility and is equivalent to
 *              calling nvte_clamped_dswiglu_v2 with glu_linear_offset = 1.0.
 *
 *        https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250
 *        This activation has two differences compared to the original SwiGLU
 *           1. Both gate and pre-activations are clipped based on parameter limit.
 *           2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation inspired
 *           by original GELU paper https://arxiv.org/pdf/1606.08415
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     limit     Clipping limits for gate and pre-activation.
 *  \param[in]     alpha     Scaling factor for the sigmoid function used in the activation.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_clamped_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                          float limit, float alpha, cudaStream_t stream);

/*! \brief Computes the gradient of gated Swish activation of the input used in GPT OSS, with a
 *         configurable offset for the linear (gate) component after clamping.
 *
 *        https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250
 *        This activation has two differences compared to the original SwiGLU
 *           1. Both gate and pre-activations are clipped based on parameter limit.
 *           2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation inspired
 *           by original GELU paper https://arxiv.org/pdf/1606.08415
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad               Incoming gradient of shape [N, H].
 *  \param[in]     input              Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output             Outgoing gradient of shape [N, H * 2].
 *  \param[in]     limit              Clipping limits for gate and pre-activation.
 *  \param[in]     alpha              Scaling factor for the sigmoid function used in the activation.
 *  \param[in]     glu_linear_offset  Offset added to the linear component after clamping (typically 1.0).
 *  \param[in]     stream             CUDA stream used for the operation.
 */
void nvte_clamped_dswiglu_v2(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                             float limit, float alpha, float glu_linear_offset,
                             cudaStream_t stream);

/*! \brief Computes ScaledSwiGLU backward without materializing GLU deinterleave.
 *
 *  The optional grad_act_scales tensor may be null. When present, it receives
 *  sum(dY * SwiGLU(input), dim=-1).
 *
 *  \param[in]     grad                 Incoming gradient of shape [N, H].
 *  \param[in]     input                Forward input tensor of shape [N, H * 2].
 *  \param[in]     act_scales           Row-wise activation scales of shape [N].
 *  \param[in,out] grad_input           Outgoing gradient of shape [N, H * 2].
 *  \param[in,out] grad_act_scales      Optional row-wise scale gradient of shape [N], or null.
 *  \param[in]     glu_interleave_size  GLU interleave chunk size, or 0 for non-interleaved layout.
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_scaled_dswiglu(const NVTETensor grad, const NVTETensor input,
                         const NVTETensor act_scales, NVTETensor grad_input,
                         NVTETensor grad_act_scales, int64_t glu_interleave_size,
                         cudaStream_t stream);

/*! \brief Computes ScaledClampedSwiGLU backward without materializing GLU deinterleave.
 *
 *  The optional grad_act_scales tensor may be null. When present, it receives
 *  sum(dY * ClampedSwiGLU(input), dim=-1).
 *
 *  \param[in]     grad                 Incoming gradient of shape [N, H].
 *  \param[in]     input                Forward input tensor of shape [N, H * 2].
 *  \param[in]     act_scales           Row-wise activation scales of shape [N].
 *  \param[in,out] grad_input           Outgoing gradient of shape [N, H * 2].
 *  \param[in,out] grad_act_scales      Optional row-wise scale gradient of shape [N], or null.
 *  \param[in]     limit                Clipping limit.
 *  \param[in]     alpha                Activation sigmoid alpha.
 *  \param[in]     glu_linear_offset    Offset added to linear component after clamping.
 *  \param[in]     glu_interleave_size  GLU interleave chunk size, or 0 for non-interleaved layout.
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_scaled_clamped_dswiglu(const NVTETensor grad, const NVTETensor input,
                                 const NVTETensor act_scales, NVTETensor grad_input,
                                 NVTETensor grad_act_scales, float limit, float alpha,
                                 float glu_linear_offset, int64_t glu_interleave_size,
                                 cudaStream_t stream);

/*! \brief Computes the gated ReLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dreglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

/*! \brief Computes the gated Quick GeLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dqgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream);

/*! \brief Computes the gated Squared ReLU activation gradient.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dsreglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream);

/*! \brief Computes ScaledSReLU.
 *
 *  Computes output = SReLU(input) * act_scales[:, None].
 *
 *  \param[in]     input       Input tensor for activation.
 *  \param[in]     act_scales  Row-wise activation scales of shape [N].
 *  \param[in,out] output      Output tensor.
 *  \param[in]     stream      CUDA stream used for the operation.
 */
void nvte_scaled_srelu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                       cudaStream_t stream);

/*! \brief Computes ScaledSReLU backward.
 *
 *  The optional grad_act_scales tensor may be null. When present, it receives
 *  sum(dY * SReLU(input), dim=-1).
 *
 *  \param[in]     grad             Incoming gradient.
 *  \param[in]     input            Forward input tensor.
 *  \param[in]     act_scales       Row-wise activation scales of shape [N].
 *  \param[in,out] grad_input       Outgoing input gradient.
 *  \param[in,out] grad_act_scales  Optional row-wise scale gradient of shape [N], or null.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_scaled_dsrelu(const NVTETensor grad, const NVTETensor input,
                        const NVTETensor act_scales, NVTETensor grad_input,
                        NVTETensor grad_act_scales, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ACTIVATION_H_
