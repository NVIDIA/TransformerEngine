/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fmha.h
 *  \brief Functions for fused multi-head attention
 */

#ifndef TRANSFORMER_ENGINE_FMHA_H_
#define TRANSFORMER_ENGINE_FMHA_H_

// #include <cudnn_frontend.h>
#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

enum class MHA_Layout {
  NOT_INTERLEAVED = 0,
  QKV_INTERLEAVED = 1,
  KV_INTERLEAVED = 2
};

enum class MHA_Matrix {
  Q_Matrix = 0,  // queries
  K_Matrix = 1,  // keys
  V_Matrix = 2,  // values
  S_Matrix = 3,  // output of GEMM1
  O_Matrix = 4   // final output
};

enum class MHA_Bias_Type {
  NO_BIAS = 0,
  PRE_SCALE_BIAS = 1,
  POST_SCALE_BIAS = 2
};

void nvte_fmha_fwd(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                   int64_t seed, MHA_Layout layout, float scaling_factor,
                   double dropout_probability, MHA_Bias_Type bias_type,
                   bool is_causal_masking, void *devPtrQ, void *devPtrK,
                   void *devPtrV, void *devPtrS, void *devPtrO,
                   void *devPtrBias, void *devActualSeqlenQ,
                   void *devActualSeqlenK, cudnnDataType_t tensorType,
                   cudaStream_t stream, cudnnHandle_t handle_);

void nvte_fmha_bwd(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
                   MHA_Layout layout, float scaling_factor,
                   float dropout_probability, bool is_causal_masking,
                   void *devPtrQ, void *devPtrK, void *devPtrV, void *devPtrS,
                   void *devPtrdQ, void *devPtrdK, void *devPtrdV,
                   void *devPtrdO, void *devPtrdS, void *devActualSeqlenQ,
                   void *devActualSeqlenK, cudnnDataType_t tensorType,
                   cudaStream_t stream, cudnnHandle_t handle_);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_FMHA_H_
