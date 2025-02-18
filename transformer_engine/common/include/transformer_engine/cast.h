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

/*  Cast the tensor to FP8 (or MXFP8 if the compute capability of the device is 10.0 or newer)
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
 *  The required shape of the MX block must be specified in the 'output' argument,
 *  where the shapes [1,32] and [32,1] are currently supported.
 *
 *  To cast the input tensor to the MXFP8, the scaling_mode.delayed_scaling parameter
 *  of the output tensor should be set equal to 0.
 */

/*! \brief Casts the tensor to FP8 (or MXFP8 if the compute capability of the device is 10.0 or newer).
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in,out] output           Output FP8/MXFP8 tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Cast tensor to MXFP8. Additionally, reduce the input along columns.
 *
 * This function casts the input and produces 2 results:
 *  - `output` is the result of the cast including the scaling factors
 *  - `dbias` is the result of the reduction of the input along columns.
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]      input            Input tensor to be cast.
 *  \param[in,out]  output           Output MXFP8 tensor.
 *  \param[out]     dbias            Result of the reduction of the input along columns.
 *  \param[out]     workspace        Workspace tensor.
 *  \param[in]      stream           CUDA stream used for the operation.
 */
void nvte_fp8_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                             NVTETensor workplace, cudaStream_t stream);

/*! \brief Compute backward of ActLU operation on the input, then cast to MXFP8.
 *         Additionally, reduce the result of the ActLU backward along columns.
 *         Supported by the devices with the compute capability 10.0 or newer.
 *
 * This function produces 2 results:
 *  - `output` is equal to `cast(dact(input))`
 *  - `dbias` is equal to `reduce(dact(input), dim=1)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input            Input tensor to be cast.
 *  \param[in]     act_input        Activation input tensor.
 *  \param[in,out] output           Output MXFP8 tensor.
 *  \param[out]    dbias            Result of the reduction of the input along columns.
 *  \param[out]    workspace        Workspace tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 *
 *  Supported activations: GeLU, SiLU, ReLU, QuickGeLU, SquaredReLU
 */
void nvte_fp8_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream);
void nvte_fp8_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor act_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream);
void nvte_fp8_quantize_dbias_drelu(const NVTETensor input, const NVTETensor act_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream);
void nvte_fp8_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor act_input,
                                    NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                    cudaStream_t stream);
void nvte_fp8_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor act_input,
                                    NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                    cudaStream_t stream);

/*! \brief Compute backward of ActLU operation on the input, then cast to MXFP8.
 *         Additionally, reduce the result of the ActLU backward along columns.
 *         Supported by the devices with the compute capability 10.0 or newer.
 *
 *  Produces two sets of output data:
 *  1) Scaled rows + row-wise scaling factors, AND
 *  2) Scaled columns + column-wise scaling factors
 *
 *  \param[in]     input                Input tensor to be cast.
 *  \param[in,out] output_rowwise       Output MXFP8 tensor scaled along rows.
 *  \param[in,out] output_columnwise    Output MXFP8 tensor scaled along columns.
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_fp8_quantize_x2(const NVTETensor input, NVTETensor output_rowwise,
                          NVTETensor output_columnwise, cudaStream_t stream);

/*! \brief Cast tensor to MXFP8 along both dimensions.
 *         Additionally, reduce the input along columns.
 *         Supported by the devices with the compute capability 10.0 or newer.
 *
 *  Produces 3 sets of output data:
 *  1) Scaled rows + row-wise scaling factors, AND
 *  2) Scaled columns + column-wise scaling factors
 *  3) dBias - the result of the reduction of the input along columns.
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input                Input tensor to be cast.
 *  \param[in,out] output_rowwise       Output MXFP8 tensor scaled along rows.
 *  \param[in,out] output_columnwise    Output MXFP8 tensor scaled along columns.
 *  \param[out]    dbias                Result of the reduction of the input along columns.
 *  \param[out]    workspace            Workspace tensor.
 *  \param[in]     stream               CUDA stream used for the operation.
 */
void nvte_fp8_quantize_dbias_x2(const NVTETensor input, NVTETensor output_rowwise,
                                NVTETensor output_columnwise, NVTETensor dbias,
                                NVTETensor workplace, cudaStream_t stream);

/*! \brief Compute backward of ActLU operation on the input, then cast to MXFP8.
 *         Additionally, reduce the result of the ActLU backward along columns.
 *         Supported by the devices with the compute capability 10.0 or newer.
 *
 *  Produces 3 sets of output data:
 *  1) Scaled rows + row-wise scaling factors, AND
 *  2) Scaled columns + column-wise scaling factors
 *  3) dBias - the result of the reduction of the input along columns.
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input                Input tensor to be cast.
 *  \param[in]     act_input            Activation input tensor.
 *  \param[in,out] output_rowwise       Output MXFP8 tensor scaled along rows.
 *  \param[in,out] output_columnwise    Output MXFP8 tensor scaled along columns.
 *  \param[out]    dbias                Result of the reduction of the input along columns.
 *  \param[out]    workspace            Workspace tensor.
 *  \param[in]     stream               CUDA stream used for the operation.
 *
 *  Supported activations: GeLU, SiLU, ReLU, QuickGeLU, SquaredReLU
 */
void nvte_fp8_quantize_dbias_dgelu_x2(const NVTETensor input, const NVTETensor act_input,
                                      NVTETensor output_rowwise, NVTETensor output_columnwise,
                                      NVTETensor dbias, NVTETensor workplace, cudaStream_t stream);
void nvte_fp8_quantize_dbias_dsilu_x2(const NVTETensor input, const NVTETensor act_input,
                                      NVTETensor output_rowwise, NVTETensor output_columnwise,
                                      NVTETensor dbias, NVTETensor workplace, cudaStream_t stream);
void nvte_fp8_quantize_dbias_drelu_x2(const NVTETensor input, const NVTETensor act_input,
                                      NVTETensor output_rowwise, NVTETensor output_columnwise,
                                      NVTETensor dbias, NVTETensor workplace, cudaStream_t stream);
void nvte_fp8_quantize_dbias_dqgelu_x2(const NVTETensor input, const NVTETensor act_input,
                                       NVTETensor output_rowwise, NVTETensor output_columnwise,
                                       NVTETensor dbias, NVTETensor workplace, cudaStream_t stream);
void nvte_fp8_quantize_dbias_dsrelu_x2(const NVTETensor input, const NVTETensor act_input,
                                       NVTETensor output_rowwise, NVTETensor output_columnwise,
                                       NVTETensor dbias, NVTETensor workplace, cudaStream_t stream);

/*! \brief Cast tensor from FP8.
 *
 *  \param[in]     input     Input tensor to be cast.
 *  \param[in,out] output    Output tensor.
 *  \param[in]     stream    CUDA stream used for the operation.
 */
void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Compute activation of the input, casting the output to FP8/MXFP8.
 *
 *  \param[in]     grad             Input tensor of shape [N, H].
 *  \param[in]     gated_input      Tensor used as input to the forward of SwiGLU operation.
 *                                  Shape [N, H * 2].
 *  \param[in,out] output           Result of the cast. Shape: [N, H * 2].
 *  \param[in]     stream           CUDA stream used for the operation.

  Supported activations: SiLU
*/
void nvte_fp8_quantize_swiglu(const NVTETensor grad, const NVTETensor gated_input,
                              NVTETensor output, cudaStream_t stream);

/*! \brief Compute activation of the input, casting the output to FP8/MXFP8.
 *
 *  \param[in]     grad             Input tensor of shape [N, H].
 *  \param[in]     gated_input      Tensor used as input to the forward of SwiGLU operation.
 *                                  Shape [N, H * 2].
 *  \param[in,out] output_rowwise   Result of the cast along row axis. Shape: [N, H * 2].
 *  \param[in,out] output_colwise   Result of the cast along column axis. Shape: [N, H * 2].
 *  \param[in]     stream           CUDA stream used for the operation.

  Supported activations: SiLU
*/
void nvte_fp8_quantize_swiglu_x2(const NVTETensor grad, const NVTETensor gated_input,
                                 NVTETensor output_rowwise, NVTETensor output_colwise,
                                 cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_CAST_H_
