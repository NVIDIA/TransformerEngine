/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file comm_gemm.h
 *  \brief Functions for distributed (multi-GPU) matrix multiplication.
 *
 *  This API is a TE-native binding to cuBLASMp library.
 *  Refer here: https://docs.nvidia.com/cuda/cublasmp/usage/tp.html for specific
 *  patterns, which allow communication-computation overlap.
 *
 *  All GEMM functions here have the same computation semantic, as expressed
 *  on global matrices, similar to nvte_cublas_gemm call:
 *  - `D = AB` if both `bias` and `pre_gelu_out` are empty tensors
 *  - `D = AB + bias` if `pre_gelu_out` is empty and `bias` is not empty
 *  - `D = GELU(AB + bias)` if both `bias` and `pre_gelu_out` are not empty tensors
 *
 *  Functions differ in matrix distribution patterns
 */

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_

#include <nccl.h>
#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

typedef struct NVTECommGemmCtx NVTECommGemmCtx;

enum NVTECommGemmAlgoType {
  kNVTECommGemmAlgoDefault = 0,
  kNVTECommGemmAlgoSplitP2P = 1,
  kNVTECommGemmAlgoSplitMulticast = 2,
  kNVTECommGemmAlgoAtomicP2P = 3,
  kNVTECommGemmAlgoAtomicMulticast = 4
};

/*! \brief Create a comm-gemm context.
 *
 *  \param[in]  comm          NCCL communicator.
 *  \param[in]  nranks        Number of ranks.
 *  \param[in]  rank          Local rank.
 */
NVTECommGemmCtx* nvte_comm_gemm_ctx_create(ncclComm_t comm, int nranks, int rank);

/*! \brief Destroy a comm-gemm context.
 *
 *  \param[in]  ctx  Context to destroy.
 */
void nvte_comm_gemm_ctx_destroy(NVTECommGemmCtx* ctx);

/*! \brief Perform AllGather communication followed by GEMM
 *
 *  Gathers distributed data from all ranks, then computes matrix multiplication.
 *
 *  \param[in]     ctx           Comm-GEMM context.
 *  \param[in]     m             Global m dimension.
 *  \param[in]     n             Global n dimension.
 *  \param[in]     k             Global k dimension.
 *  \param[in]     a             Local part of A matrix.
 *  \param[in]     b             Local part of B matrix.
 *  \param[in,out] d             Local part of D matrix.
 *  \param[in]     bias          Bias tensor.
 *  \param[in,out] pre_act_out   Local part of output matrix before GELU activation.
 *  \param[in]     transa        Whether A matrix is transposed.
 *  \param[in]     transb        Whether B matrix is transposed.
 *  \param[in]     grad          Whether this operation is part of gradient computation.
 *  \param[in]     accumulate    Whether to accumulate the result into the D matrix.
 *  \param[in]     comm_sm_count Number of GPU SMs to use for communication (default=0: use heuristics)
 *  \param[in]     main_stream   CUDA stream used for computation.
 *  \param[in]     algo          Algorithm to use.
 */
void nvte_all_gather_gemm(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                          NVTECommGemmAlgoType algo);

/*! \brief Perform GEMM followed by ReduceScatter communication
 *
 *  Computes matrix multiplication, then distributes results across ranks with reduction.
 *
 *  \param[in]     ctx           Comm-GEMM context.
 *  \param[in]     m             Global m dimension.
 *  \param[in]     n             Global n dimension.
 *  \param[in]     k             Global k dimension.
 *  \param[in]     a             Local part of A matrix.
 *  \param[in]     b             Local part of B matrix.
 *  \param[in,out] d             Local part of D matrix.
 *  \param[in]     bias          Bias tensor.
 *  \param[in,out] pre_act_out   Local part of output matrix before GELU activation.
 *  \param[in]     transa        Whether A matrix is transposed.
 *  \param[in]     transb        Whether B matrix is transposed.
 *  \param[in]     grad          Whether this operation is part of gradient computation.
 *  \param[in]     accumulate    Whether to accumulate the result into the D matrix.
 *  \param[in]     comm_sm_count Number of GPU SMs to use for communication (default=0: use heuristics)
 *  \param[in]     main_stream   CUDA stream used for computation.
 *  \param[in]     algo          Algorithm to use.
 */
void nvte_gemm_reduce_scatter(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k,
                              const NVTETensor a, const NVTETensor b, const NVTETensor d,
                              const NVTETensor bias, const NVTETensor pre_act_out, bool transa,
                              bool transb, bool grad, bool accumulate, int comm_sm_count,
                              cudaStream_t main_stream, NVTECommGemmAlgoType algo);

/*! \brief Perform GEMM followed by AllReduce communication
 *
 *  Computes matrix multiplication, then reduces results across all ranks.
 *
 *  \param[in]     ctx           Comm-GEMM context.
 *  \param[in]     m             Global m dimension.
 *  \param[in]     n             Global n dimension.
 *  \param[in]     k             Global k dimension.
 *  \param[in]     a             Local part of A matrix.
 *  \param[in]     b             Local part of B matrix.
 *  \param[in,out] d             Local part of D matrix.
 *  \param[in]     bias          Bias tensor.
 *  \param[in,out] pre_act_out   Local part of output matrix before GELU activation.
 *  \param[in]     transa        Whether A matrix is transposed.
 *  \param[in]     transb        Whether B matrix is transposed.
 *  \param[in]     grad          Whether this operation is part of gradient computation.
 *  \param[in]     accumulate    Whether to accumulate the result into the D matrix.
 *  \param[in]     comm_sm_count Number of GPU SMs to use for communication (default=0: use heuristics)
 *  \param[in]     main_stream   CUDA stream used for computation.
 *  \param[in]     algo          Algorithm to use.
 */
void nvte_gemm_all_reduce(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                          NVTECommGemmAlgoType algo);

/*! \brief Get local number of rows or columns.
 *
 *  Utility function to get local dimension.
 *  Block size, nranks and local rank is derived from the context ctx.
 *
 *  \param[in]  ctx          Comm-GEMM context.
 *  \param[in]  global_size  Global dimension.
 */
int64_t nvte_comm_gemm_numroc(NVTECommGemmCtx* ctx, int64_t global_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMM_GEMM_H_
