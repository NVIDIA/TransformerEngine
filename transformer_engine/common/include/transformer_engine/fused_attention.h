/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ATTENTION_H_
#define TRANSFORMER_ENGINE_FUSED_ATTENTION_H_

#include "../common.h"
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

enum class MHA_Layout { NOT_INTERLEAVED = 0, QKV_INTERLEAVED = 1, KV_INTERLEAVED = 2 };

enum class MHA_Matrix {
    Q_Matrix = 0,            // queries
    K_Matrix = 1,            // keys
    K_Matrix_Transpose = 2,  // keys transposed
    V_Matrix = 3,            // values
    V_Matrix_Transpose = 4,  // value matrix transposed
    S_Matrix = 5,            // output of GEMM1
    O_Matrix = 6,            // final output
};

enum class MHA_Bias_Type { NO_BIAS = 0, PRE_SCALE_BIAS = 1, POST_SCALE_BIAS = 2 };

void nvte_cudnn_fused_attention_fwd(int64_t batch, int64_t max_q_seqlen, int64_t max_kv_seqlen,
                                    int64_t total_seqs, int64_t num_head, int64_t head_size,
                                    float scale_qk, float p_dropout, bool is_causal_masking,
                                    bool is_training, MHA_Layout qkv_layout,
                                    MHA_Bias_Type bias_type, NVTETensor qkv, NVTETensor m,
                                    NVTETensor z_inv, NVTETensor softmax_aux, NVTETensor output,
                                    NVTETensor bias, NVTETensor q_ragged_offset,
                                    NVTETensor kv_ragged_offset, NVTETensor actual_q_seqlen,
                                    NVTETensor actual_kv_seqlen, NVTETensor philox_unpack,
                                    NVTETensor workspace, cudaStream_t stream);

void nvte_cudnn_fused_attention_bwd(int64_t batch, int64_t max_q_seqlen, int64_t max_kv_seqlen,
                                    int64_t total_seqs, int64_t num_head, int64_t head_size,
                                    float scale_qk, float p_dropout, bool is_causal_masking,
                                    MHA_Layout qkv_layout, NVTETensor qkv, NVTETensor m,
                                    NVTETensor z_inv, NVTETensor softmax_aux, NVTETensor output,
                                    NVTETensor doutput, NVTETensor dqkv, NVTETensor dsoftmax,
                                    NVTETensor q_ragged_offset, NVTETensor kv_ragged_offset,
                                    NVTETensor actual_q_seqlen, NVTETensor actual_kv_seqlen,
                                    NVTETensor philox_unpack, NVTETensor workspace,
                                    cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
