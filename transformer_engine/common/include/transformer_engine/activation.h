/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \brief Compute activation of the input.
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
};

void nvte_gelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_silu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_relu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_qgelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_srelu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Compute activation gradient.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream);

void nvte_dsilu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream);

void nvte_drelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                cudaStream_t stream);

void nvte_dqgelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

void nvte_dsrelu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

/*! \brief Compute gated activation of the input.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Act(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_geglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_swiglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_reglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_qgeglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

void nvte_sreglu(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Compute gated activation gradient.
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

void nvte_dswiglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream);

void nvte_dreglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                 cudaStream_t stream);

void nvte_dqgeglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream);

void nvte_dsreglu(const NVTETensor grad, const NVTETensor input, NVTETensor output,
                  cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ACTIVATION_H_
