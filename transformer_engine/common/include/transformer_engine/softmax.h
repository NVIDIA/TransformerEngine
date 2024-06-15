/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_SOFTMAX_H_
#define TRANSFORMER_ENGINE_SOFTMAX_H_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute scaled softmax activation on the input.
 *
 *  \param[in]     input           Input tensor for softmax.
 *  \param[out]    softmax_results Output tensor.
 *  \param[in]     scale_factor    Scalar for the input tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_softmax_forward(const NVTETensor input, NVTETensor softmax_results,
                                 float scale_factor, cudaStream_t stream);

/*! \brief Compute the backward of the scaled softmax activation.
 *
 *  - `incoming_grads` is the input tensor containing the gradients received from the following layer.
 *  - `softmax_results` is the output tensor of the corresponding forward softmax operation.
 *  - `output_grads` is the output tensor containing the computed gradients.
 *
 *  \param[in]     incoming_grads  Input gradient tensor for backward.
 *  \param[in]     softmax_results Output tensor of softmax forward.
 *  \param[out]    output_grads    Output tensor.
 *  \param[in]     scale_factor    Scalar for the output tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_softmax_backward(const NVTETensor incoming_grads, const NVTETensor softmax_results,
                                  NVTETensor output_grads, float scale_factor, cudaStream_t stream);

/*! \brief Compute scaled masked softmax activation on the input.
 *
 *  \param[in]     input           Input tensor for softmax.
 *  \param[in]     mask            Mask for the input tensor.
 *  \param[out]    softmax_results Output tensor.
 *  \param[in]     scale_factor    Scalar for the input tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_masked_softmax_forward(const NVTETensor input, const NVTETensor mask,
                                        NVTETensor softmax_results, float scale_factor,
                                        cudaStream_t stream);

/*! \brief Compute the backward of the scaled masked softmax activation.
 *
 *  - `incoming_grads` is the input tensor containing the gradients received from the following layer.
 *  - `softmax_results` is the output tensor of the corresponding forward softmax operation.
 *  - `output_grads` is the output tensor containing the computed gradients.
 *
 *  \param[in]     incoming_grads  Input gradient tensor for backward.
 *  \param[in]     softmax_results Output tensor of softmax forward.
 *  \param[out]    output_grads    Output tensor.
 *  \param[in]     scale_factor    Scalar for the output tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_masked_softmax_backward(const NVTETensor incoming_grads,
                                         const NVTETensor softmax_results, NVTETensor output_grads,
                                         float scale_factor, cudaStream_t stream);

/*! \brief Compute scaled softmax activation using a 2D upper triangular mask on the input.
 *
 *  \param[in]     input           Input tensor for softmax.
 *  \param[out]    softmax_results Output tensor.
 *  \param[in]     scale_factor    Scalar for the input tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_upper_triang_masked_softmax_forward(const NVTETensor input,
                                                     NVTETensor softmax_results, float scale_factor,
                                                     cudaStream_t stream);

/*! \brief Compute the backward of the scaled softmax activation using a 2D upper triangular mask.
 *
 *  - `incoming_grads` is the input tensor containing the gradients received from the following layer.
 *  - `softmax_results` is the output tensor of the corresponding forward softmax operation.
 *  - `output_grads` is the output tensor containing the computed gradients.
 *
 *  \param[in]     incoming_grads  Input gradient tensor for backward.
 *  \param[in]     softmax_results Output tensor of softmax forward.
 *  \param[out]    output_grads    Output tensor.
 *  \param[in]     scale_factor    Scalar for the output tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_upper_triang_masked_softmax_backward(const NVTETensor incoming_grads,
                                                      const NVTETensor softmax_results,
                                                      NVTETensor output_grads, float scale_factor,
                                                      cudaStream_t stream);

/*! \brief Compute scaled softmax activation using an implicit 2D mask aligned to the bottom right corner of the input matrix.
 *
 *  \param[in]     input           Input tensor for softmax.
 *  \param[out]    softmax_results Output tensor.
 *  \param[in]     scale_factor    Scalar for the input tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_aligned_causal_masked_softmax_forward(const NVTETensor input,
                                                       NVTETensor softmax_results,
                                                       float scale_factor, cudaStream_t stream);

/*! \brief Compute the backward pass of the scaled softmax activation using an implicit 2D mask aligned to the bottom right corner of the input matrix.
 *
 *  - `incoming_grads` is the input tensor containing the gradients received from the following layer.
 *  - `softmax_results` is the output tensor of the corresponding forward softmax operation.
 *  - `output_grads` is the output tensor containing the computed gradients.
 *
 *  \param[in]     incoming_grads  Input gradient tensor for backward.
 *  \param[in]     softmax_results Output tensor of softmax forward.
 *  \param[out]    output_grads    Output tensor.
 *  \param[in]     scale_factor    Scalar for the output tensor.
 *  \param[in]     stream          CUDA stream used for the operation.
 */
void nvte_scaled_aligned_causal_masked_softmax_backward(const NVTETensor incoming_grads,
                                                        const NVTETensor softmax_results,
                                                        NVTETensor output_grads, float scale_factor,
                                                        cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SOFTMAX_H_
