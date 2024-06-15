/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void nvte_transpose_with_noop(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                              cudaStream_t stream);

void nvte_cast_transpose_with_noop(const NVTETensor input, const NVTETensor noop,
                                   NVTETensor cast_output, NVTETensor transposed_output,
                                   cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_TRANSPOSE_WITH_NOOP_H_
