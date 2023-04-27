/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn.h
 *  \brief Functions for fused multi-head attention
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_H_

#include "transformer_engine/fused_attn.h"

#include "common/common.h"

#include <cudnn.h>

void fused_attn_max_512_fwd_impl(
  int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
  int64_t seed, NVTE_QKV_Layout layout, float scaling_factor,
  float dropout_probability, NVTE_Bias_Type bias_type,
  NVTE_Mask_Type mask_type, void *devPtrQ, void *devPtrK,
  void *devPtrV, void *devPtrS, void *devPtrO,
  void *devPtrBias, void *devCuSeqlenQ, void *devCuSeqlenK,
  void* workspace, size_t *workspace_size, cudnnDataType_t tensorType,
  cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_max_512_bwd_impl(
  int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
  NVTE_QKV_Layout layout, float scaling_factor,
  float dropout_probability, NVTE_Mask_Type mask_type, NVTE_Bias_Type bias_type,
  void *devPtrQ, void *devPtrK, void *devPtrV, void *devPtrS,
  void *devPtrdQ, void *devPtrdK, void *devPtrdV,
  void *devPtrdO, void *devPtrdS, void *devPtrdBias,
  void *devCuSeqlenQ, void *devCuSeqlenK,
  void* workspace, size_t *workspace_size, cudnnDataType_t tensorType,
  cudaStream_t stream, cudnnHandle_t handle);

namespace transformer_engine {
void fused_attn_max_512_fwd_qkvpacked(
    size_t batch,
    size_t max_seqlen,
    size_t num_head,
    size_t head_size,
    bool is_training,
    float attn_scale,
    float p_dropout,
    NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type,
    const Tensor *input_QKV,
    const Tensor *input_Bias,
    Tensor *output_O,
    NVTETensorPack* Aux_Output_Tensors,
    const Tensor *cu_seqlens,
    const Tensor *rng_state,
    Tensor *workspace,
    cudaStream_t stream,
    cudnnHandle_t handle);

void fused_attn_max_512_fwd_kvpacked(
    size_t batch,
    size_t q_max_seqlen,
    size_t kv_max_seqlen,
    size_t num_head,
    size_t head_dim,
    bool is_training,
    float attn_scale,
    float p_dropout,
    NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type,
    const Tensor *input_Q,
    const Tensor *input_KV,
    const Tensor *input_Bias,
    Tensor *output_O,
    NVTETensorPack* Aux_Output_Tensors,
    const Tensor *q_cu_seqlens,
    const Tensor *kv_cu_seqlens,
    const Tensor *rng_state,
    Tensor *workspace,
    cudaStream_t stream,
    cudnnHandle_t handle);

void fused_attn_max_512_bwd_qkvpacked(
  size_t batch,
  size_t max_seqlen,
  size_t num_head,
  size_t head_dim,
  float attn_scale,
  float p_dropout,
  NVTE_QKV_Layout qkv_layout,
  NVTE_Bias_Type bias_type,
  NVTE_Mask_Type mask_type,
  const Tensor *input_QKV,
  const Tensor *input_dO,
  const NVTETensorPack* Aux_CTX_Tensors,
  Tensor *output_dQKV,
  Tensor *output_dBias,
  const Tensor *cu_seqlens,
  Tensor *workspace,
  cudaStream_t stream,
  cudnnHandle_t handle);
}

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_H_
