/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_

#include <cublasmp.h>
#include <nccl.h>
#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NVTECommGemmCtx NVTECommGemmCtx;

/*! \brief Create a comm-gemm context.
 */
NVTECommGemmCtx* nvte_comm_gemm_ctx_create(ncclComm_t comm, int nranks, int rank, int local_device);

/*! \brief Destroy a comm-gemm context.
 */
void nvte_comm_gemm_ctx_destroy(NVTECommGemmCtx* ctx);

/*
 * Refer here: https://docs.nvidia.com/cuda/cublasmp/usage/tp.html for additional details.
 */

/*! \brief AllGather-GEMM
 *
 * m, n, k - Global GEMM dimensions.
 * Tensors and boolean flags have the same meaning as in nvte_cublas_gemm.
 */
void nvte_all_gather_gemm(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                          cublasMpMatmulAlgoType_t algo = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT);

/*! \brief GEMM-ReduceScatter.
 *
 * m, n, k - Global GEMM dimensions.
 * Tensors and boolean flags have the same meaning as in nvte_cublas_gemm.
 */
void nvte_gemm_reduce_scatter(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                              const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                              const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                              bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                              cublasMpMatmulAlgoType_t algo = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT);

/*! \brief GEMM-AllReduce.
 *
 * m, n, k - Global GEMM dimensions.
 * Tensors and boolean flags have the same meaning as in nvte_cublas_gemm.
 */
void nvte_gemm_all_reduce(NVTECommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream,
                          cublasMpMatmulAlgoType_t algo = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT);

/*! \brief Get local number of rows or columns.
 */
int64_t nvte_comm_gemm_numroc(NVTECommGemmCtx* ctx, int64_t global_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMM_GEMM_H_
