/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dropout.h
 *  \brief Enums and functions for fused attention.
 */

#ifndef TRANSFORMER_ENGINE_DROPOUT_FP8_H_
#define TRANSFORMER_ENGINE_DROPOUT_FP8_H_

#include "stdint.h"
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!  \brief Dropout forward kernel.
 *
 *  \param[in]     input            Input tensor.
 *  \param[out]    output           Output tensor.
 *  \param[out]    mask             Mask tensor.
 *  \param[in]     rng_state        RNG engine inputs.
 *  \param[in]     dropout_probability Dropout probability.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_dropout_fwd(NVTETensor input, NVTETensor output, NVTETensor mask,
                      NVTETensor rng_state, float dropout_probability,
                      cudaStream_t stream);

/*!  \brief Dropout forward kernel with FP8 input.
 *
 *  \param[in]     input            Input tensor (FP8).
 *  \param[out]    output           Output tensor (dequantized to FP16/BF16).
 *  \param[out]    mask             Mask tensor.
 *  \param[in]     rng_state        RNG engine inputs.
 *  \param[in]     dropout_probability Dropout probability.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_dropout_fwd_fp8(NVTETensor input, NVTETensor output, NVTETensor mask,
                          NVTETensor rng_state, float dropout_probability,
                          cudaStream_t stream);

/*!  \brief Dropout backward kernel.
 *
 *  \param[in]     grad_output      Gradient of output tensor.
 *  \param[in]     mask             Mask tensor.
 *  \param[out]    grad_input       Gradient of input tensor.
 *  \param[in]     dropout_probability Dropout probability.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_dropout_bwd(NVTETensor grad_output, NVTETensor mask, NVTETensor grad_input,
                      float dropout_probability, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
