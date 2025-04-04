/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_

#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CommGemmCtx CommGemmCtx;

CommGemmCtx* nvte_comm_gemm_ctx_create(int nranks, int rank, int local_device);
void nvte_comm_gemm_ctx_destroy(CommGemmCtx* ctx) noexcept;

void nvte_comm_gemm(CommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                    const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                    const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                    bool accumulate, int comm_sm_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMM_GEMM_H_
