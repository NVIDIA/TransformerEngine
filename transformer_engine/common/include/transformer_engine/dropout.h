/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dropout.h
 *  \brief Functions for dropout.
 */

#ifndef TRANSFORMER_ENGINE_DROPOUT_FP8_H_
#define TRANSFORMER_ENGINE_DROPOUT_FP8_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!  \brief Dropout forward kernel.
 *
 *  \param[in]     input            Input tensor.
 *  \param[out]    output           Output tensor.
 *  \param[out]    mask             Mask tensor. Each bit corresponds to an
 *                                  output tensor entry. Ones indicate kept
 *                                  entries and zeros indicate dropped entries.
 *  \param[in]     rng_state        RNG engine inputs.
 *  \param[in]     dropout_probability Dropout probability.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_dropout_fwd(const NVTETensor input, NVTETensor output, NVTETensor mask,
                      NVTETensor rng_state, float dropout_probability, cudaStream_t stream);

/*!  \brief Dropout backward kernel.
 *
 *  \param[in]     grad_output      Gradient of output tensor.
 *  \param[out]    mask             Mask tensor. Each bit corresponds to an
 *                                  output tensor entry. Ones indicate kept
 *                                  entries and zeros indicate dropped entries.
 *  \param[out]    grad_input       Gradient of input tensor.
 *  \param[in]     dropout_probability Dropout probability.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_dropout_bwd(const NVTETensor grad_output, const NVTETensor mask, NVTETensor grad_input,
                      float dropout_probability, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
