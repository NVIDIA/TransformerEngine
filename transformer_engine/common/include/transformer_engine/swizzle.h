/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file swizzle.h
 *  \brief Functions to convert scaling factors into format expected by GEMM.
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
 *  - data is quantized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_swizzle_scaling_factors(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Swizzling scaling factors into the required interleaved layout for GEMM
 *
 *  \param[in]     inputs       Input tensors with non-swizzled scale_inv.
 *  \param[in,out] outputs      Output tensors which hosts swizzled scale_inv.
 *  \param[in]     num_tensors  Number of input and output tensors.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_multi_tensor_swizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                               const size_t num_tensors, cudaStream_t stream);

/*! \brief Unswizzling scaling factors from the interleaved layout used by GEMM back to row-major
 *
 *  \param[in]     input        Input tensor with swizzled scale_inv.
 *  \param[in,out] output       Output tensor which hosts non-swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major in output.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_unswizzle_scaling_factors(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Unswizzling scaling factors from the interleaved layout used by GEMM back to row-major
 *
 *  \param[in]     inputs       Input tensors with swizzled scale_inv.
 *  \param[in,out] outputs      Output tensors which hosts non-swizzled scale_inv.
 *  \param[in]     num_tensors  Number of input and output tensors.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major in output.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_multi_tensor_unswizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                                 const size_t num_tensors, cudaStream_t stream);

/*! \brief Swizzling FP8 block scaling scaling factors into mxfp8 interleaved layout for GEMM
 *
 *  \param[in]     input        Input FP8 block-scaled tensor.
 *  \param[in,out] output       Output mxfp8 tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  This function is used for emulating the FP8 block scaling recipe on Blackwell and newer as it
 *  not natively supported by cublasLt on architectures other than Hopper.

 *  Requirements:
 *  - input is an FP8 block scaling tensor
 *  - input has rowwise usage
 *  - output is an MXFP8 tensor
 *  - output has rowwise usage
 *  - output.scale_inv has appropriate shape
 *  */
void nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(const NVTETensor input, NVTETensor output,
                                                         cudaStream_t stream);
/*! \brief Swizzling scaling factors into the required interleaved layout for GEMM (grouped tensor)
 *
 *  \param[in]     input        Input grouped tensor with non-swizzled scale_inv.
 *  \param[in,out] output       Output grouped tensor which hosts swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements(for now, more features will be added later):
 *  - scaling mode must be MXFP8 1D scaling.
 *  - scale_inv is stored in row-major per group.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantitized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 *  - all tensors in the grouped tensor must have the same shape.
 */
void nvte_swizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                          cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SWIZZLE_H_
