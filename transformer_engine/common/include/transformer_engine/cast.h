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

/*  Cast the tensor to FP8 (or microscaling FP8 if the compute capability of the device is 10.0 or newer)
 *  The implementation is per the microscaling format MXFP8 defined by the OCP specification:
 *  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
 *
 *  Supported modes of scaling (live scaling):
 *      1) Rowwise scaling (along the dim=0) computes one set of the output data, which includes:
 *          - the scaled output tensor
 *          - the corresponding scaling factors
 *         The scaling factors are computed for blocks of the shape [1,32]
 *         (i.e., each scaling factor spans 32 contiguous elements along rows).
 *
 *      2) Columwise scaling (along the dim=1) computes one set of the output data.
 *         The scaling factors are computed for blocks of the shape [32,1]
 *         (i.e., each scaling factor spans 32 contiguous elements along columns).
 *
 *      3) Both rowwise AND columnwise scaling (along the dim=0 and the dim=1)
 *         computes two sets of the output data: both 1) and 2).
 *
 *  The shape of the MX block must be specified in the 'output' argument,
 *  and can be either [1,32] or [32,1] as no other shapes are currently supported.
 *
 *  To cast the input tensor to the MXFP8, the scaling_mode.delayed_scaling parameter
 *  of the output tensor should be set to 0.
 */

/*! \brief Casts input tensor to FP8/MXFP8.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Casts input tensor to FP8/MXFP8, providing the option to immediately exit the kernel
 *         based on the value of the 'noop' tensor.
 *         If the scaling mode of the output tensor is set to NVTE_MXFP8_1D_SCALING,
 *         the block quantization (MXFP8) of the specified shape of the block will be used.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output FP8/MXFP8 tensor.
 *  \param[out]     noop             Noop tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream);

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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
