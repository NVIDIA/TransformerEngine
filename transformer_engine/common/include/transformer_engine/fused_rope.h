/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROPE_H_
#define TRANSFORMER_ENGINE_FUSED_ROPE_H_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Apply rotary positional embedding to the input tensor.
 *
 *  \param[in]     input           Input tensor for fused rope.
 *  \param[in]     cos             The cos tensor.
 *  \param[in]     sin             The sin tensor.
 *  \param[out]    output          Output tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor cos,
                             const NVTETensor sin, NVTETensor output,
                             cudaStream_t stream);

/*! \brief Compute the backward of the fused rope.
 *
 *  \param[in]     incoming_grads  Input gradient tensor for backward.
 *  \param[in]     cos             The cos tensor.
 *  \param[in]     sin             The sin tensor.
 *  \param[out]    output_grads    Output gradient tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_fused_rope_backward(const NVTETensor incoming_grads,
                              const NVTETensor cos, const NVTETensor sin,
                              NVTETensor output_grads, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FUSED_ROPE_H_
