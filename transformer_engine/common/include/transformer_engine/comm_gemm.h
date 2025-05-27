/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_

#include <nccl.h>
#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CommGemmCtx CommGemmCtx;

CommGemmCtx* nvte_comm_gemm_ctx_create(ncclComm_t comm, int nranks, int rank, int local_device);
void nvte_comm_gemm_ctx_destroy(CommGemmCtx* ctx);

void nvte_all_gather_gemm(CommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream);

void nvte_gemm_reduce_scatter(CommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                              const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                              const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                              bool accumulate, int comm_sm_count, cudaStream_t main_stream);

void nvte_gemm_all_reduce(CommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                          const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                          const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                          bool accumulate, int comm_sm_count, cudaStream_t main_stream);

int64_t nvte_comm_gemm_numroc(CommGemmCtx* ctx, int64_t global_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMM_GEMM_H_
