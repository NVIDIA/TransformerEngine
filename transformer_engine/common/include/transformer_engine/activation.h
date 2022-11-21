/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *  \param[out]    output    Output tensor.
 *  \param[in]     scale     Scaling factor of the output tensor.
 *  \param[in,out] amax      AMAX value of the output tensor.
 *  \param[out]    scale_inv Inverse of the output's scaling factor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_gelu(const NVTETensor input,
               NVTETensor output,
               const NVTETensor scale,
               NVTETensor amax,
               NVTETensor scale_inv,
               cudaStream_t stream);

void nvte_gated_gelu(const NVTETensor input,
                     NVTETensor output,
                     const NVTETensor scale,
                     NVTETensor amax,
                     NVTETensor scale_inv,
                     cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_ACTIVATION_H_
