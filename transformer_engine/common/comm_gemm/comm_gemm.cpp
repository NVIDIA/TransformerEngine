/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/comm_gemm.h"

#include "../util/logging.h"

struct CommGemmCtx {
};

CommGemmCtx* nvte_comm_gemm_ctx_create() {
  return nullptr;
}

void nvte_comm_gemm_ctx_destroy(CommGemmCtx* ctx) {
  delete ctx;
}

void nvte_comm_gemm(CommGemmCtx* ctx) {
  // TODO:
}
