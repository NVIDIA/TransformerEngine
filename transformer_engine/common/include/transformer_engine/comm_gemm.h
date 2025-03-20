/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_
#define TRANSFORMER_ENGINE_COMMON_COMM_GEMM_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CommGemmCtx CommGemmCtx;

CommGemmCtx* nvte_comm_gemm_ctx_create();
void nvte_comm_gemm_ctx_destroy(CommGemmCtx* ctx);

void nvte_comm_gemm(CommGemmCtx* ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMM_GEMM_H_
