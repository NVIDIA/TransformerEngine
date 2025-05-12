/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast.h
 *  \brief Functions to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_H_
#define TRANSFORMER_ENGINE_CAST_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Quantize the tensor
 *
 *  The type of quantized tensor in the output depends on the scaling mode of the output
 *  tensor.
 *
 *  Supported formats are:
 *
 *  1) MXFP8 scaling (for compute capability 10.0 or newer)
 *
 *  The MXFP8 implementation is per the microscaling format MXFP8 defined by the OCP specification:
 *  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *
 *
 *  Supported modes of MXFP8 scaling (live scaling) for scaling mode NVTE_MXFP8_1D_SCALING
 *      a) Rowwise scaling (along the dim=0) computes one set of the output data, which includes:
 *          - the scaled output tensor
 *          - the corresponding scaling factors
 *         The scaling factors are computed for blocks of the shape [1,32]
 *         (i.e., each scaling factor spans 32 contiguous elements along rows).
 *
 *      b) Columwise scaling (along the dim=1) computes one set of the output data.
 *         The scaling factors are computed for blocks of the shape [32,1]
 *         (i.e., each scaling factor spans 32 contiguous elements along columns).
 *
 *      c) Both rowwise AND columnwise scaling (along the dim=0 and the dim=1)
 *         computes two sets of the output data: both 1) and 2).
 *
 *  The shape of the MX block must be specified in the 'output' argument,
 *  and can be either [1,32] or [32,1] as no other shapes are currently supported.
 *
 *  To cast the input tensor to the MXFP8, the scaling_mode.delayed_scaling parameter
 *  of the output tensor should be set to 0.
 *
 *  2) NVTE_DELAYED_TENSOR_SCALING that quantize the entire tensor
 *  using a single scaling factor. The absolute maximum value of the tensor should
 *  be precalculated either online (current scaling) or based on a tensor history
 *  (delayed scaling). The calls to nvte_quantize scale based on that data value.
 *  Note the NVTE_DELAYED_TENSOR_SCALING NVTEScalingMode is reused for online
 *  per tensor scaling.
 *
 *
 *  3) FP8 block scaling formats NVTE_BLOCK_SCALING_1D and NVTE_BLOCK_SCALING_2D
 *  for compute capability of at least 9.0. These modes quantize the tensor by blocks
 *  of size 1x128 (with columnwise mode of 128x1) and 128x128 respectively.
 *
 *  The supported modes are:
 *      a) Rowwise scaling yields output data:
 *          - the scaled output tensor in fp8 coefficients with identical shape to the
 *            input tensor.
 *          - Scale factors which are computed for either 1D 1x128 or 2D 128x128 blocks.
 *      b) Columnwise scaling yields output data:
 *          - the scaled output tensor in fp8 coefficients with a shape equivalent to
 *            the transpose of the input tensor.
 *          - Scale factors which are calculated for either 1D 128x1 or 2D 128x128 blocks
 *            of the input tensor.
 *      c) Both: In which both tensors and both scales are calculated.
 *
 *  This quantization mode includes both the calculation of the scaling factors
 *  per-tile and quantization of the row and/or columnwise tiles. No precalculated
 *  absolute max is required. The scaling factors are also rounded to powers of 2.
 */

/*! \brief Casts input tensor to FP8/MXFP8/BlockwiseFP8.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8/BlockwiseFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts input tensor to FP8/MXFP8/BlockwiseFP8, providing the option to immediately exit the kernel
 *         based on the value of the 'noop' tensor.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output quantized tensor.
 *  \param[out]     noop             Noop tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream);

/*! \brief Casts input tensor to quantized output tensor, with advanced quantization options.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output quantized tensor.
 *  \param[in]      quant_config     Quantization configuration.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream);

/*! \brief Casts input tensor to MXFP8. Additionally, reduces the input along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                         NVTETensor workplace, cudaStream_t stream);

/*! \brief Computes backward of GeLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the GeLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of SiLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the SiLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of ReLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the ReLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_drelu(const NVTETensor input, const NVTETensor act_input,
                               NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                               cudaStream_t stream);

/*! \brief Computes backward of Quick GeLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Quick GeLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream);

/*! \brief Computes backward of Squared ReLU operation on the input, then casts to FP8/MXFP8.
 *         Additionally, reduces the result of the Squared ReLU backward along columns.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with the workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream);

/*! \brief Casts input tensor from reduced to higher precision.
 *         If the scaling mode of the input tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block dequantization (MXFP8) of the specified shape of the block will be used.
 *         In case of the MXFP8 dequantization, the dequantized values are stored to the rowwise
 *         data of the output tensor, regardless of whether the row- or columnwise scaling is used.
 *
 *  \param[in]     input     Input FP8/MXFP8 tensor to be cast.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts a group of input tensors to FP8/MXFP8/BlockwiseFP8.
 *         The type of quantized tensor in the output depends on the scaling mode of the output
 *         tensor. See file level comments.
 *
 *  \param[in]     input            List of input tensor to be cast.
 *  \param[in,out] output           List of output FP8/MXFP8/BlockwiseFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_grouped_quantize(const NVTETensor *input, NVTETensor *output, const int num_groups,
                           cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
