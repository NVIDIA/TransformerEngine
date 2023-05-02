/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"
#include "../common.h"
#include "utils.h"
#include "fused_attn_fp8.h"
#include "fused_attn_max_512.h"

// NVTE fused attention FWD FP8 with packed QKV
void nvte_fused_attn_fwd_qkvpacked(
            const NVTETensor QKV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            const NVTETensor cu_seqlens,
            const NVTETensor rng_state,
            size_t max_seqlen,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_qkvpacked);
  using namespace transformer_engine;

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  const DType QKV_type = input_QKV->data.dtype;

  if (((QKV_type == DType::kFloat8E4M3) || (QKV_type == DType::kFloat8E5M2))
                  && (max_seqlen <= 512)) {
    // QKV shape is [total_seqs, 3, h, d]
    size_t b = input_cu_seqlens->data.shape[0] - 1;
    size_t h = input_QKV->data.shape[2];
    size_t d = input_QKV->data.shape[3];

#if (CUDNN_VERSION >= 8900)
    // FP8 API doesn't use input_Bias, bias_type or attn_mask_type
    fused_attn_fwd_fp8_qkvpacked(
            b, max_seqlen, h, d,
            is_training, attn_scale, dropout, qkv_layout,
            input_QKV, input_output_S, output_O,
            Aux_Output_Tensors,
            input_cu_seqlens,
            input_rng_state,
            wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9 is required to run FP8 fused attention. \n");
#endif
  } else if (((QKV_type == DType::kFloat16) || (QKV_type == DType::kBFloat16))
                  && (max_seqlen <= 512)) {
#if (CUDNN_VERSION >= 8901)
    auto ndim = input_QKV->data.shape.size();

    // QKV shape is [b, s, 3, h, d]
    size_t batch = input_QKV->data.shape[0];
    size_t max_seqlen = input_QKV->data.shape[1];
    size_t num_head = input_QKV->data.shape[ndim - 2];
    size_t head_dim = input_QKV->data.shape[ndim - 1];

    fused_attn_max_512_fwd_qkvpacked(
      batch,
      max_seqlen,
      num_head,
      head_dim,
      is_training,
      attn_scale,
      dropout,
      qkv_layout,
      bias_type,
      attn_mask_type,
      input_QKV,
      input_Bias,
      output_O,
      Aux_Output_Tensors,
      input_cu_seqlens,
      input_rng_state,
      wkspace,
      stream,
      handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.1 is required to run BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (max_seqlen > 512) {
    NVTE_ERROR("TBD: No support for fused attention with >512 seqlence length currently. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length! \n");
  }
}
// NVTE fused attention BWD FP8 with packed QKV
void nvte_fused_attn_bwd_qkvpacked(
            const NVTETensor QKV,
            NVTETensor dBias,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQKV,
            const NVTETensor cu_seqlens,
            size_t max_seqlen,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_qkvpacked);
  using namespace transformer_engine;

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();

  const Tensor *input_cu_seqlens = reinterpret_cast<const Tensor*>(cu_seqlens);
  const Tensor *input_QKV = reinterpret_cast<const Tensor*>(QKV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQKV = reinterpret_cast<Tensor*>(dQKV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  const DType QKV_type = input_QKV->data.dtype;

  if (((QKV_type == DType::kFloat8E4M3) || (QKV_type == DType::kFloat8E5M2))
                  && (max_seqlen <= 512)) {
    // QKV shape is [total_seqs, 3, h, d]
    size_t b = input_cu_seqlens->data.shape[0] - 1;
    size_t h = input_QKV->data.shape[2];
    size_t d = input_QKV->data.shape[3];
#if (CUDNN_VERSION >= 8900)
    // Aux_CTX_Tensors contain [M, ZInv, rng_state] generated by the forward pass
    const Tensor *input_M = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[0]);
    const Tensor *input_ZInv = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[1]);
    const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(Aux_CTX_Tensors->tensors[2]);
    auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();
    // FP8 API doesn't use input_dBias, bias_type or attn_mask_type
    fused_attn_bwd_fp8_qkvpacked(
                    b, max_seqlen, h, d,
                    attn_scale, dropout, qkv_layout,
                    input_QKV, input_O, input_dO,
                    input_M, input_ZInv,
                    input_S, input_output_dP,
                    output_dQKV,
                    input_cu_seqlens,
                    input_rng_state,
                    wkspace, stream, handle);
#else
    NVTE_ERROR("cuDNN 8.9 is required to run FP8 fused attention. \n");
#endif
  } else if (((QKV_type == DType::kFloat16) || (QKV_type == DType::kBFloat16))
                  && (max_seqlen <= 512)) {
#if (CUDNN_VERSION >= 8901)
    auto ndim = input_QKV->data.shape.size();
    // QKV shape is [b, s, 3, h, d]
    size_t batch = input_QKV->data.shape[0];
    size_t max_seqlen = input_QKV->data.shape[1];
    size_t num_head = input_QKV->data.shape[ndim - 2];
    size_t head_dim = input_QKV->data.shape[ndim - 1];

    // NVTE_ERROR("TBD: No support for BF16/FP16 fused attention currently. \n");
    fused_attn_max_512_bwd_qkvpacked(
      batch,
      max_seqlen,
      num_head,
      head_dim,
      attn_scale,
      dropout,
      qkv_layout,
      bias_type,
      attn_mask_type,
      input_QKV,
      input_dO,
      Aux_CTX_Tensors,
      output_dQKV,
      output_dBias,
      input_cu_seqlens,
      wkspace,
      stream,
      handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.1 is required to run BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if (max_seqlen > 512) {
    NVTE_ERROR("TBD: No support for fused attention with >512 seqlence length currently. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length! \n");
  }
}
// NVTE fused attention FWD FP8 with packed KV
void nvte_fused_attn_fwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            const NVTETensor Bias,
            NVTETensor S,
            NVTETensor O,
            NVTETensorPack* Aux_Output_Tensors,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            const NVTETensor rng_state,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            bool is_training, float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_fwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_rng_state = reinterpret_cast<const Tensor*>(rng_state);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  const Tensor *input_Bias = reinterpret_cast<const Tensor*>(Bias);
  Tensor *input_output_S = reinterpret_cast<Tensor*>(S);
  Tensor *output_O = reinterpret_cast<Tensor*>(O);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // Q shape is [total_seqs, h, d]
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h = input_Q->data.shape[1];
  size_t d = input_Q->data.shape[2];
  const DType QKV_type = input_Q->data.dtype;

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();

  if (((QKV_type == DType::kFloat8E4M3) || (QKV_type == DType::kFloat8E5M2))
                  && (max_seqlen_q <= 512) && (max_seqlen_kv <= 512)) {
    NVTE_ERROR("The FP8 fused attention API only supports packed QKV input. \n");
  } else if (((QKV_type == DType::kFloat16) || (QKV_type == DType::kBFloat16))
                  && (max_seqlen_q <= 512) && (max_seqlen_kv <= 512)) {
#if (CUDNN_VERSION >= 8901)
    auto ndim = input_Q->data.shape.size();
    // Q shape is [b, s_q, h, d]
    // KV shape is [b, s_kv, 2, h, d]
    size_t batch = input_Q->data.shape[0];
    size_t q_max_seqlen = input_Q->data.shape[1];
    size_t kv_max_seqlen = input_KV->data.shape[1];
    size_t num_head = input_Q->data.shape[ndim - 2];
    size_t head_dim = input_Q->data.shape[ndim - 1];

    fused_attn_max_512_fwd_kvpacked(
      batch,
      q_max_seqlen,
      kv_max_seqlen,
      num_head,
      head_dim,
      is_training,
      attn_scale,
      dropout,
      qkv_layout,
      bias_type,
      attn_mask_type,
      input_Q,
      input_KV,
      input_Bias,
      output_O,
      Aux_Output_Tensors,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      input_rng_state,
      wkspace,
      stream,
      handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.1 is required to run BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if ((max_seqlen_q > 512) || (max_seqlen_kv > 512)) {
    NVTE_ERROR("TBD: No support for fused attention with >512 seqlence length currently. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length! \n");
  }
}
// NVTE fused attention BWD FP8 with packed KV
void nvte_fused_attn_bwd_kvpacked(
            const NVTETensor Q,
            const NVTETensor KV,
            NVTETensor dBias,
            const NVTETensor O,
            const NVTETensor dO,
            const NVTETensor S,
            NVTETensor dP,
            const NVTETensorPack* Aux_CTX_Tensors,
            NVTETensor dQ,
            NVTETensor dKV,
            const NVTETensor cu_seqlens_q,
            const NVTETensor cu_seqlens_kv,
            size_t max_seqlen_q, size_t max_seqlen_kv,
            float attn_scale, float dropout,
            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
            NVTE_Mask_Type attn_mask_type,
            NVTETensor workspace,
            cudaStream_t stream) {
  NVTE_API_CALL(nvte_flash_attn_bwd_kvpacked);
  using namespace transformer_engine;
  const Tensor *input_cu_seqlens_q = reinterpret_cast<const Tensor*>(cu_seqlens_q);
  const Tensor *input_cu_seqlens_kv = reinterpret_cast<const Tensor*>(cu_seqlens_kv);
  const Tensor *input_Q = reinterpret_cast<const Tensor*>(Q);
  const Tensor *input_KV = reinterpret_cast<const Tensor*>(KV);
  const Tensor *input_O = reinterpret_cast<const Tensor*>(O);
  const Tensor *input_dO = reinterpret_cast<const Tensor*>(dO);
  const Tensor *input_S = reinterpret_cast<const Tensor*>(S);
  Tensor *input_output_dP = reinterpret_cast<Tensor*>(dP);
  Tensor *output_dQ = reinterpret_cast<Tensor*>(dQ);
  Tensor *output_dKV = reinterpret_cast<Tensor*>(dKV);
  Tensor *output_dBias = reinterpret_cast<Tensor*>(dBias);
  Tensor *wkspace = reinterpret_cast<Tensor*>(workspace);

  // Q shape is [total_seqs, h, d]
  size_t b = input_cu_seqlens_q->data.shape[0] - 1;
  size_t h = input_Q->data.shape[1];
  size_t d = input_Q->data.shape[2];
  const DType QKV_type = input_Q->data.dtype;

  auto handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();

  if (((QKV_type == DType::kFloat8E4M3) || (QKV_type == DType::kFloat8E5M2))
                  && (max_seqlen_q <= 512) && (max_seqlen_kv <= 512)) {
    NVTE_ERROR("The FP8 fused attention API only supports packed QKV input. \n");
  } else if (((QKV_type == DType::kFloat16) || (QKV_type == DType::kBFloat16))
                  && (max_seqlen_q <= 512) && (max_seqlen_kv <= 512)) {
#if (CUDNN_VERSION >= 8901)
    auto ndim = input_Q->data.shape.size();
    // Q shape is [b, s_q, h, d]
    // KV shape is [b, s_kv, 2, h, d]
    size_t batch = input_Q->data.shape[0];
    size_t q_max_seqlen = input_Q->data.shape[1];
    size_t kv_max_seqlen = input_KV->data.shape[1];
    size_t num_head = input_Q->data.shape[ndim - 2];
    size_t head_dim = input_Q->data.shape[ndim - 1];

    fused_attn_max_512_bwd_kvpacked(
      batch,
      q_max_seqlen,
      kv_max_seqlen,
      num_head,
      head_dim,
      attn_scale,
      dropout,
      qkv_layout,
      bias_type,
      attn_mask_type,
      input_Q,
      input_KV,
      input_dO,
      Aux_CTX_Tensors,
      output_dQ,
      output_dKV,
      output_dBias,
      input_cu_seqlens_q,
      input_cu_seqlens_kv,
      wkspace,
      stream,
      handle);
#else
    NVTE_ERROR(
      "cuDNN 8.9.1 is required to run BF16/FP16 fused attention with max_seqlen<=512. \n");
#endif
  } else if ((max_seqlen_q > 512) || (max_seqlen_kv > 512)) {
    NVTE_ERROR("TBD: No support for fused attention with >512 seqlence length currently. \n");
  } else {
    NVTE_ERROR("Invalid combination of data type and sequence length! \n");
  }
}
