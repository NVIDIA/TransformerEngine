/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file hadamard_transform.h
 *  \brief Functions for Hadamard transforms.
 */

#ifndef TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
#define TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Perform a randomized Hadamard transform on the input tensor.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input              Input tensor to apply Hadamard transform.
 *  \param[in,out]  output             Output tensor.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_hadamard_transform(const NVTETensor input, NVTETensor output, int random_sign_mask,
                             int random_sign_mask_t, cudaStream_t stream);

/*! \brief Perform the absolute maximum reduction on the input tensor with/without
 *         randomized hadamard transform. The rowwise result is the absolute maximum
 *         of the input tensor. The columnwise result is the absolute maximum of the
 *         input tensor transposed and applied randomized hadamard transformation.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input              Input tensor to apply Hadamard transform.
 *  \param[in,out]  output             Output tensor.
 *  \param[in]      random_sign_mask   16-bit sign mask.
 *  \param[in]      random_sign_mask_t 16-bit sign mask.
 *  \param[in]      stream             CUDA stream used for the operation.
 */
void nvte_hadamard_transform_amax(const NVTETensor input, NVTETensor output, int random_sign_mask,
                                  int random_sign_mask_t, cudaStream_t stream);

/*! \brief Perform the columnwise hadamard transform cast fusion.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input           Input tensor to apply Hadamard transform.
 *  \param[in,out]  output          Output tensor.
 *  \param[in]      hadamard_matrix Hadamard matrix.
 *  \param[in]      quant_config    Quantization configuration.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_hadamard_transform_cast_fusion_columnwise(const NVTETensor input, NVTETensor output,
                                                    const NVTETensor hadamard_matrix,
                                                    const NVTEQuantizationConfig quant_config,
                                                    cudaStream_t stream);

/*!
 * \brief Perform multi-tensor Hadamard transform absolute maximum reduction (amax) with optional randomized Hadamard transform.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input                Input tensor to apply Hadamard transform, assumed contiguous in memory and split on dimension 0.
 *  \param[in,out]  outputs              Array of output tensors.
 *  \param[in]      split_sections       Array of splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors          Number of output tensors, must be > 0.
 *  \param[in]      random_sign_mask     16-bit (int) sign mask for transform.
 *  \param[in]      random_sign_mask_t   16-bit (int) sign mask for transform (transposed).
 *  \param[in]      stream               CUDA stream used for the operation.
 */
void nvte_multi_hadamard_transform_amax(const NVTETensor input, NVTETensor* outputs,
                                        const int* split_sections, const size_t num_tensors,
                                        int random_sign_mask, int random_sign_mask_t,
                                        cudaStream_t stream);

/*!
 * \brief Perform multi-tensor absolute maximum reduction (amax) without Hadamard transform.
 *
 *  This function is experimental and the API is not stable.
 *
 *  \param[in]      input           Input tensor, assumed contiguous in memory and split on dimension 0.
 *  \param[in,out]  outputs         Array of output tensors.
 *  \param[in]      split_sections  Array specifying splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors     Number of output tensors, must be > 0.
 *  \param[in]      stream          CUDA stream used for the operation.
 */
void nvte_multi_tensor_amax(const NVTETensor input, NVTETensor* outputs, const int* split_sections,
                            const size_t num_tensors, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
