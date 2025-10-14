/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \brief Computes the SiLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the ReLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_relu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the Quick GeLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_qgelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Computes the Squared ReLU activation of the input.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_srelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ACTIVATION_H_
