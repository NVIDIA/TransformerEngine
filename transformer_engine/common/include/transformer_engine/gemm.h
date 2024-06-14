/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gemm.h
 *  \brief Functions for matrix multiplication.
 */

#ifndef TRANSFORMER_ENGINE_GEMM_H_
#define TRANSFORMER_ENGINE_GEMM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Compute matrix multiplication of 2 matrices, potentially fused with other operations.
 *
 * Computes:
 *  - `D = AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  \param[in]     A                     The A matrix.
 *  \param[in]     B                     The B matrix.
 *  \param[in,out] D                     Output matrix.
 *  \param[in]     bias                  Bias tensor.
 *  \param[in,out] pre_gelu_out          Output matrix before GELU activation.
 *  \param[in]     transa                Whether A matrix is transposed.
 *  \param[in]     transb                Whether B matrix is transposed.
 *  \param[in]     grad                  Whether this operation is part of the
 *                                       gradient computation.
 *  \param[out]    workspace             Workspace tensor.
 *  \param[in]     accumulate            Whether to accumulate the result into the D matrix.
 *  \param[in]     use_split_accumulator Whether to use split accumulator in the FP8 GEMM.
 *  \param[in]     math_sm_count         Number of GPU SMs to use (default=0: use cuBLAS heuristics)
 *  \param[in]     stream                CUDA stream used for the operation.
 */
void nvte_cublas_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D, const NVTETensor bias,
                      NVTETensor pre_gelu_out, bool transa, bool transb, bool grad,
                      NVTETensor workspace, bool accumulate, bool use_split_accumulator,
                      int math_sm_count, cudaStream_t stream);

/*! \brief Compute matrix multiplication of 2 matrices with chunking and atomic counters.
 *
 * \warning   Cublas atomic gemm uses a beta API and is not tested for all use cases.
 *
 * Computes:
 *  - `D = AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  \param[in]     A                     The A matrix.
 *  \param[in]     B                     The B matrix.
 *  \param[in,out] D                     Output matrix.
 *  \param[in]     bias                  Bias tensor.
 *  \param[in,out] pre_gelu_out          Output matrix before GELU activation.
 *  \param[in]     transa                Whether A matrix is transposed.
 *  \param[in]     transb                Whether B matrix is transposed.
 *  \param[in]     grad                  Whether this operation is part of the
 *                                       gradient computation.
 *  \param[out]    workspace             Workspace tensor.
 *  \param[in]     accumulate            Whether to accumulate the result into the D matrix.
 *  \param[in]     use_split_accumulator Whether to use split accumulator in the FP8 GEMM.
 *  \param[in]     math_sm_count         Number of GPU SMs to use (default=0: use cuBLAS heuristics)
 *  \param[in]     m_split               Number of chunks/splits along m-dimension for Atomic GEMM.
 *  \param[in]     n_split               Number of chunks/splits along n-dimension for Atomic GEMM.
 *  \param[in]     gemm_producer         Whether Atomic GEMM is the producer or consumer.
 *  \param[in,out] counter               counter[chunk_i]=0 indicates chunk_i has been produced.
 *  \param[in]     stream                CUDA stream used for the operation.
 */
void nvte_cublas_atomic_gemm(const NVTETensor A, const NVTETensor B, NVTETensor D,
                             const NVTETensor bias, NVTETensor pre_gelu_out, bool transa,
                             bool transb, bool grad, NVTETensor workspace, bool accumulate,
                             bool use_split_accumulator, int math_sm_count, int m_split,
                             int n_split, bool gemm_producer, const NVTETensor counter,
                             cudaStream_t stream);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_GEMM_H_
