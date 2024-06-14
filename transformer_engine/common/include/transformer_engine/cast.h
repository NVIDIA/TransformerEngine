/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_H_
#define TRANSFORMER_ENGINE_CAST_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Cast tensor to FP8.
 *
 *  \param[in]     input     Input tensor to be cast.
 *  \param[in,out] output    Output FP8 tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Cast tensor from FP8.
 *
 *  \param[in]     input     Input tensor to be cast.
 *  \param[out]    output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
