/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8.
 */

#ifndef TRANSFORMER_ENGINE_SWIZZLE_H_
#define TRANSFORMER_ENGINE_SWIZZLE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Swizzling scaling factors into the required interleaved layout for GEMM
 *
 *  \param[in]     input        Input tensor with non-swizzled scale_inv.
 *  \param[in,out] output       Output tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantitized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_swizzle_scaling_factors(const NVTETensor input, NVTETensor output, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SWIZZLE_H_
