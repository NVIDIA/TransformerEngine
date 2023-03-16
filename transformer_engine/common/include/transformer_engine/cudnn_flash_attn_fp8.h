/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_CUDNN_FLASH_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_CUDNN_FLASH_ATTN_FP8_H_

#include "transformer_engine.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2
};

enum class MHA_Matrix {
    Q_Matrix            = 0, // queries
    K_Matrix            = 1, // keys
    K_Matrix_Transpose  = 2, // keys transposed
    V_Matrix            = 3, // values
    V_Matrix_Transpose  = 4, // value matrix transposed
    S_Matrix            = 5, // output of GEMM1
    O_Matrix            = 6, // final output
};

void nvte_cudnn_flash_attn_fwd(
                int64_t b, int64_t max_seq_len,
                int64_t total_seqs, int64_t h, int64_t d,
                float scale_q_k, float p_dropout, int qkv_layout,
                const NVTETensor QKV,
                const NVTETensor M,
                const NVTETensor ZInv,
                const NVTETensor S,
                const NVTETensor O,
                int64_t *QKVRaggedOffset,
                int64_t *ORaggedOffset,
                uint64_t *PhiloxUnpacked,
                NVTETensor workspace,
                cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
