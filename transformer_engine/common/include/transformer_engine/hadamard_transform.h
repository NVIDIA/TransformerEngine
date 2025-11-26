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

/*! \brief Split a tensor along dimension 0 and compute RHT amaxes for each split.
 *
 *  This function is experimental and the API is not stable.
 *
 *  This is intended for quantizing to NVFP4 with random Hadamard
 *  transforms (RHT). For each tensor split, compute the maximum
 *  absolute value (amax) and populate the row-wise amax of the
 *  corresponding output tensor. Also, compute the amax after a
 *  transposed RHT and populate the column-wise amax of the
 *  corresponding output tensor.
 *
 *  \param[in]      input               Input tensor.
 *  \param[in,out]  outputs             Array of NVFP4 output tensors. Only the row-wise and
 *                                      column-wise amaxes are updated.
 *  \param[in]      split_sections      Size of each tensor split along dimension 0.
 *  \param[in]      num_tensors         Number of tensor splits.
 *  \param[in]      random_sign_mask    16-bit sign mask for RHT.
 *  \param[in]      random_sign_mask_t  16-bit sign mask for transposed RHT.
 *  \param[in]      stream              CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_amax(const NVTETensor input, NVTETensor* outputs,
                                        const size_t* split_sections, size_t num_tensors,
                                        int random_sign_mask, int random_sign_mask_t,
                                        cudaStream_t stream);

/*!
 * \brief Perform the grouped-tensor columnwise Hadamard transform cast fusion operation.
 *
 *  This function is experimental and the API is not stable. Group_ prefix means contiguous input concatenated
 *
 *  \param[in]      input             Input tensor to apply Hadamard transform.
 *  \param[in,out]  outputs           Array of output tensors.
 *  \param[in]      hadamard_matrix   Hadamard matrix to use for transformation.
 *  \param[in]      split_sections    Array specifying splits in dimension 0 for each output tensor.
 *  \param[in]      num_tensors       Number of output tensors, must be > 0.
 *  \param[in]      quant_config      Quantization configuration.
 *  \param[in]      stream            CUDA stream used for the operation.
 */
void nvte_group_hadamard_transform_cast_fusion_columnwise(
    const NVTETensor input, NVTETensor* outputs, const NVTETensor hadamard_matrix,
    const size_t* split_sections, size_t num_tensors, const NVTEQuantizationConfig quant_config,
    cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_H_
