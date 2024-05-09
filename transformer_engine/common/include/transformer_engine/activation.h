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

/*! \enum NVTE_Activation_Type
 *  \brief TE activation function type.
 */
enum class NVTE_Activation_Type {
  GELU      = 0,  /*!< Gaussian Error Linear Unit */
  GEGLU     = 1,  /*!< Gated Gaussian Error Linear Unit */
  SILU      = 2,  /*!< Sigmoid Linear Unit */
  SWIGLU    = 3,  /*!< Gated Sigmoid Linear Unit */
  RELU      = 4,  /*!< Rectified Linear Unit */
  REGLU     = 5,  /*!< Gated Rectified Linear Unit */
  QGELU     = 6,  /*!< Quick-GeLU Approximation */
  QGEGLU    = 7,  /*!< Gated Quick-GeLU Approximation */
  SRELU     = 8,  /*!< S-Shaped ReLU */
  SREGLU    = 9,  /*!< Gated S-Shaped ReLU */
};


/*! \brief Compute GELU activation of the input.
 *
 *  \param[in]     input     Input tensor for GELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream);

/*! \brief Compute GELU activation gradient.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for GELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgelu(const NVTETensor grad,
                const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream);

/*! \brief Compute GeGLU of the input.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes GELU(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_geglu(const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream);

/*! \brief Compute GeGLU gradient.
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgeglu(const NVTETensor grad,
                 const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream);

/*! \brief Compute SiLU activation of the input.
 *
 *  \param[in]     input     Input tensor for GELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_swish(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream);

/*! \brief Compute Swish activation gradient.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for Swish activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dswish(const NVTETensor grad,
                const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream);

/*! \brief Compute SwiGLU activation of the input.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes Swish(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_swiglu(const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream);

/*! \brief Compute SwiGLU gradient.
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dswiglu(const NVTETensor grad,
                  const NVTETensor input,
                  NVTETensor output,
                  cudaStream_t stream);


/*! \brief Compute RELU activation of the input.
 *
 *  \param[in]     input     Input tensor for RELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_relu(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream);

/*! \brief Compute RELU activation gradient.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for RELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_drelu(const NVTETensor grad,
                const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream);

/*! \brief Compute ReGLU activation of the input.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H].
 *                           It computes ReLU(input[N, :H]) x input[N, H:]
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_reglu(const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream);

/*! \brief Compute ReGLU gradient.
 *  \param[in]     grad      Incoming gradient of shape [N, H].
 *  \param[in]     input     Forward input tensor of shape [N, H * 2].
 *  \param[in,out] output    Outgoing gradient of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dreglu(const NVTETensor grad,
                 const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream);

/*! \brief Compute QuickGELU activation of the input.
 *
 *  \param[in]     input     Input tensor for QuickGELU activation.
 *  \param[in,out] output    Output tensor. Approximates GELU as input x sigmoid(1.702 x input).
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_qgelu(const NVTETensor input,
                NVTETensor output,
                cudaStream_t stream);

/*! \brief Compute QuickGELU activation gradient.
 *
 *  \param[in]     grad      Incoming gradient.
 *  \param[in]     input     Input tensor for QuickGELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dqgelu(const NVTETensor grad,
                 const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ACTIVATION_H_
