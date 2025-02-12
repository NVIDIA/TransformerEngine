/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file transpose_with_noop.h
 *  \brief Functions handling transposes with no-op.
 */

#ifndef TRANSFORMER_ENGINE_CAST_TRANSPOSE_WITH_NOOP_H_
#define TRANSFORMER_ENGINE_CAST_TRANSPOSE_WITH_NOOP_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Transposes the input, providing the option to immediately exit the kernel
 *         based on the value of the 'noop' tensor.
 *
 *  \param[in]     input     Input tensor.
 *  \param[in]     noop      Noop tensor.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_transpose_with_noop(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                              cudaStream_t stream);

/*! \brief Casts and transposes the input, providing the option to immediately exit the kernel
 *         based on the value of the 'noop' tensor.
 *
 *  \param[in]     input     Input tensor.
 *  \param[in]     noop      Noop tensor.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_cast_transpose_with_noop(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                                   cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_TRANSPOSE_WITH_NOOP_H_
