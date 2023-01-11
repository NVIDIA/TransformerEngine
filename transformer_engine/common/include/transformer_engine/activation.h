/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \brief Compute GELU activation of the input.
 *
 *  \param[in]     input     Input tensor for GELU activation.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               cudaStream_t stream);

/*! \brief Compute GeGLU of the input.
 *
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *                           It computes GELU([N, :H]) x [N, H:]
 *  \param[in,out] output    Output tensor of shape [N, H].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_geglu(const NVTETensor input,
                      NVTETensor output,
                      cudaStream_t stream);

/*! \brief Compute GeGLU gradient.
 *  \param[in]     grad      Input tensor of shape [N, H].
 *  \param[in]     input     Input tensor of shape [N, H * 2].
 *  \param[in,out] output    Output tensor of shape [N, H * 2].
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dgeglu(const NVTETensor grad,
                 const NVTETensor input,
                 NVTETensor output,
                 cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ACTIVATION_H_
