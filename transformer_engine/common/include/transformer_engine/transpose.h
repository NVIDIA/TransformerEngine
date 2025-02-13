/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file transpose.h
 *  \brief Functions handling transposes.
 */

#ifndef TRANSFORMER_ENGINE_TRANSPOSE_H_
#define TRANSFORMER_ENGINE_TRANSPOSE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Cast and transpose the input.
 *
 * This function casts the input and produces 2 results:
 *  - rowwise data in `output` is the result of the cast
 *  - columnwise data in `output` is the transposed result of the cast.
 *
 *  \param[in]     input          Input tensor of shape [N, H].
 *  \param[in,out] output         Result of the cast and transpose.
 *                                Shape of the rowwise data: [N, H].
 *                                Shape of the columnwise data: [H, N]
 *  \param[in]     stream         CUDA stream used for the operation.
 */
void nvte_cast_transpose(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Transpose the input.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[out]    transposed_output   Result of the transpose. Shape: [H, N].
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_transpose(const NVTETensor input, NVTETensor transposed_output, cudaStream_t stream);

/*! \brief Cast and transpose the input. Additionally, reduce the input along the first dimension.
 *
 * This function casts the input and produces 2 results:
 *  - `output` is the result of the cast (rowwise data) and transposed cast (columnwise data)
 *  - `dbias` is the result of the reduction of the input along the first dimension.
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input          Input tensor of shape [N, H].
 *  \param[in,out] output         Result of the cast and transpose.
 *                                Shape of the rowwise data: [N, H].
 *                                Shape of the columnwise data: [H, N]
 *  \param[out]    dbias          Result of the reduction of the input along the
 *                                first dimension. Shape: [H].
 *  \param[out]    workspace      Workspace tensor.
 *  \param[in]     stream         CUDA stream used for the operation.
 */
void nvte_cast_transpose_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                               NVTETensor workspace, cudaStream_t stream);

/*! \brief Transpose the FP8 input. Additionally, reduce the input along the first dimension.
 *
 * This function takes FP8 input and produces 2 results:
 *  - `transposed_output` is the transposed result of the input.
 *  - `dbias` is the result of the reduction of the input along the first dimension.
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in,out] transposed_output   Result of the transpose. Shape: [H, N].
 *  \param[out]    dbias               Result of the reduction of the input along the
 *                                     first dimension. Shape: [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_fp8_transpose_dbias(const NVTETensor input, NVTETensor transposed_output,
                              NVTETensor dbias, NVTETensor workspace, cudaStream_t stream);

/*! \brief Cast and transpose multiple tensors.
 *
 *  \param[in]     num_tensors         Number of tensors.
 *  \param[in]     input_list          List of 2D input tensors.
 *  \param[in,out] output_list         List of casted tensors. Dimensions
 *                                     of their rowwise data members match
 *                                     tensors in input_list. Dimensions of
 *                                     their columnwise data members are
 *                                     transposed.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_multi_cast_transpose(size_t num_tensors, const NVTETensor* input_list,
                               NVTETensor* output_list, cudaStream_t stream);

/*! \brief Compute backward of GeLU operation on the input, then cast and transpose.
 *         Additionally, reduce the result of the GeLU backward along the first dimension.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *  - `dbias` is equal to `reduce(dact(input), axis=0)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     act_input           Tensor used as input for the operation of forward activation.
 *                                     Shape [N, H].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H].
 *                                     Shape of columnwise data: [H, N].
 *  \param[out]    dbias               Result of the reduction of the dact(input) along the
 *                                     first dimension. Shape: [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_cast_transpose_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                                     NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Compute backward of SiLU operation on the input, then cast and transpose.
 *         Additionally, reduce the result of the SiLU backward along the first dimension.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *  - `dbias` is equal to `reduce(dact(input), axis=0)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     act_input           Tensor used as input for the operation of forward activation.
 *                                     Shape [N, H].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H].
 *                                     Shape of columnwise data: [H, N].
 *  \param[out]    dbias               Result of the reduction of the dact(input) along the
 *                                     first dimension. Shape: [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_cast_transpose_dbias_dsilu(const NVTETensor input, const NVTETensor act_input,
                                     NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Compute backward of ReLU operation on the input, then cast and transpose.
 *         Additionally, reduce the result of the ReLU backward along the first dimension.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *  - `dbias` is equal to `reduce(dact(input), axis=0)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     act_input           Tensor used as input for the operation of forward activation.
 *                                     Shape [N, H].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H].
 *                                     Shape of columnwise data: [H, N].
 *  \param[out]    dbias               Result of the reduction of the dact(input) along the
 *                                     first dimension. Shape: [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_cast_transpose_dbias_drelu(const NVTETensor input, const NVTETensor act_input,
                                     NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                     cudaStream_t stream);

/*! \brief Compute backward of the Quick GeLU operation on the input, then cast and transpose.
 *         Additionally, reduce the result of the Quick GeLU backward along the first dimension.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *  - `dbias` is equal to `reduce(dact(input), axis=0)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     act_input           Tensor used as input for the operation of forward activation.
 *                                     Shape [N, H].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H].
 *                                     Shape of columnwise data: [H, N].
 *  \param[out]    dbias               Result of the reduction of the dact(input) along the
 *                                     first dimension. Shape: [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_cast_transpose_dbias_dqgelu(const NVTETensor input, const NVTETensor act_input,
                                      NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                      cudaStream_t stream);

/*! \brief Compute backward of the Squared ReLU operation on the input, then cast and transpose.
 *         Additionally, reduce the result of the Squared ReLU backward along the first dimension.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *  - `dbias` is equal to `reduce(dact(input), axis=0)`
 *
 *  Calling this function with workspace being an empty tensor will not perform the operation,
 *  but instead set the shape and type of the workspace tensor to the required values.
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     act_input           Tensor used as input for the operation of forward activation.
 *                                     Shape [N, H].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H].
 *                                     Shape of columnwise data: [H, N].
 *  \param[out]    dbias               Result of the reduction of the dact(input) along the
 *                                     first dimension. Shape: [H].
 *  \param[out]    workspace           Workspace tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_cast_transpose_dbias_dsrelu(const NVTETensor input, const NVTETensor act_input,
                                      NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                      cudaStream_t stream);

/*! \brief Computes the gated GeLU activation of the input, additionally casts and transposes
 *         the output.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     gated_act_input     Tensor used as input to the forward of
 *                                     gated activation operation.
 *                                     Shape [N, H * 2].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H * 2].
 *                                     Shape of columnwise data: [H * 2, N].
 *  \param[in]     stream              CUDA stream used for the operation.
*/
void nvte_dgeglu_cast_transpose(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Swish activation of the input,
 *         additionally casts and transposes the output.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     gated_act_input     Tensor used as input to the forward of
 *                                     gated activation operation.
 *                                     Shape [N, H * 2].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H * 2].
 *                                     Shape of columnwise data: [H * 2, N].
 *  \param[in]     stream              CUDA stream used for the operation.
*/
void nvte_dswiglu_cast_transpose(const NVTETensor input, const NVTETensor act_input,
                                 NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated ReLU activation of the input,
 *         additionally casts and transposes the output.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     gated_act_input     Tensor used as input to the forward of
 *                                     gated activation operation.
 *                                     Shape [N, H * 2].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H * 2].
 *                                     Shape of columnwise data: [H * 2, N].
 *  \param[in]     stream              CUDA stream used for the operation.
*/
void nvte_dreglu_cast_transpose(const NVTETensor input, const NVTETensor act_input,
                                NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Quick GeLU activation of the input,
 *         additionally casts and transposes the output.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     gated_act_input     Tensor used as input to the forward of
 *                                     gated activation operation.
 *                                     Shape [N, H * 2].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H * 2].
 *                                     Shape of columnwise data: [H * 2, N].
 *  \param[in]     stream              CUDA stream used for the operation.
*/
void nvte_dqgeglu_cast_transpose(const NVTETensor input, const NVTETensor act_input,
                                 NVTETensor output, cudaStream_t stream);

/*! \brief Computes the gated Squared ReLU activation of the input,
 *         additionally casts and transposes the output.
 *
 * This function produces 2 results:
 *  - rowwise data of `output` is equal to `cast(dact(input))`
 *  - columnwise data of `output` is equal to `transpose(cast(dact(input)))`
 *
 *  \param[in]     input               Input tensor of shape [N, H].
 *  \param[in]     gated_act_input     Tensor used as input to the forward of
 *                                     gated activation operation.
 *                                     Shape [N, H * 2].
 *  \param[in,out] output              Result of the cast.
 *                                     Shape of rowwise data: [N, H * 2].
 *                                     Shape of columnwise data: [H * 2, N].
 *  \param[in]     stream              CUDA stream used for the operation.
*/
void nvte_dsreglu_cast_transpose(const NVTETensor input, const NVTETensor act_input,
                                 NVTETensor output, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_TRANSPOSE_H_
