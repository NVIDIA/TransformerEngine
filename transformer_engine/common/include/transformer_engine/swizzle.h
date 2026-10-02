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
 *  \param[in]     inputs                  Input tensors with non-swizzled scale_inv.
 *  \param[in,out] outputs                 Output tensors which hosts swizzled scale_inv.
 *  \param[in]     num_tensors             Number of input and output tensors.
 *  \param[in]     stream                  CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scale_inv is stored in row-major.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 */
void nvte_multi_tensor_swizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                               const size_t num_tensors, cudaStream_t stream);

/*! \brief Same as nvte_multi_tensor_swizzle_scaling_factors, but skips
 *         scale_inv shape/padding validation.
 *
 *  Use this variant when the data and scale_inv tensors intentionally have
 *  different shapes, e.g. when scale_invs have been transposed for attention.
 */
void nvte_multi_tensor_swizzle_scaling_factors_unchecked(const NVTETensor* inputs,
                                                         NVTETensor* outputs,
                                                         const size_t num_tensors,
                                                         cudaStream_t stream);

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

/*! \brief Swizzle the FP8 block-scaling scaling factors of a grouped tensor into the MXFP8
 *         interleaved layout for GEMM (grouped variant of
 *         nvte_swizzle_block_scaling_to_mxfp8_scaling_factors).
 *
 *  \param[in]     input        Input FP8 block-scaled grouped tensor.
 *  \param[in,out] output       Output MXFP8 grouped tensor which hosts the swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Used to emulate the FP8 block scaling recipe with MXFP8 grouped GEMM on Blackwell and newer.
 *  Per-tensor dimensions are read on the device, so the operation is CUDA-graph safe when they
 *  change between replays.
 *
 *  Requirements:
 *  - input is an FP8 block scaling (1D or 2D) grouped tensor with rowwise data and FP32 rowwise
 *    scale_inv in the compact per-tensor layout; per-tensor first dims or last dims (int64) may
 *    vary, but not both
 *  - output is an MXFP8 grouped tensor with the same rowwise data pointer, dims and logical
 *    shape, with_gemm_swizzled_scales set, and E8M0 rowwise scale_inv. Tensor t with rowwise
 *    data [f, l] is written at the cumulative offset of
 *    roundup(f, 128) * ceil(l / 128) * 4 bytes.
 *  - output scale_inv holds at least the following number of bytes, where [F, L] is the logical
 *    shape and n the number of tensors (0 bytes if F * L == 0). The bound depends only on the
 *    logical shape, so the buffer can be sized without reading per-tensor dims:
 *    - uniform dims:        n * roundup(F / n, 128) * ceil(L / 128) * 4
 *    - varying first dims:  (F + 127 * n) * ceil(L / 128) * 4
 *    - varying last dims:   roundup(F, 128) * 4 * (ceil(L / 128) + n)
 *  - 24 * n + 8 bytes (per-tensor offset tables) fit in the device's shared memory per block
 */
void nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors(const NVTEGroupedTensor input,
                                                                 NVTEGroupedTensor output,
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

/*! \brief Unswizzling scaling factors from the interleaved GEMM layout back to row-major (grouped)
 *
 *  \param[in]     input        Input grouped tensor with swizzled scale_inv.
 *  \param[in,out] output       Output grouped tensor which hosts non-swizzled scale_inv.
 *  \param[in]     stream       CUDA stream used for the operation.
 *
 *  Requirements:
 *  - scaling mode must be MXFP8 1D scaling.
 *  - scale_inv is stored in row-major in output.
 *  - scale_inv size is padded to 128x4 for row-scale and 4x128 for col-scale.
 *  - data is quantized along K-dimension, i.e. 1D-scaling block lies along the K-dimension.
 *  - all tensors in the grouped tensor must have the same shape.
 */
void nvte_unswizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                            cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_SWIZZLE_H_
