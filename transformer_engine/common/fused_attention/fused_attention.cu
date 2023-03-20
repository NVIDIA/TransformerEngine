/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attention.h"

namespace transformer_engine {
namespace fused_attention {

void cudnn_fused_attention_fwd(int64_t batch, int64_t max_q_seqlen, int64_t max_kv_seqlen,
                               int64_t total_seqs, int64_t num_head, int64_t head_size,
                               float scale_qk, float p_dropout, bool is_causal_masking,
                               bool is_training, MHA_Layout qkv_layout, MHA_Bias_Type bias_type,
                               Tensor *qkv, Tensor *m, Tensor *z_inv, Tensor *softmax_aux,
                               Tensor *output, Tensor *bias, Tensor *q_ragged_offset,
                               Tensor *kv_ragged_offset, Tensor *actual_q_seqlen,
                               Tensor *actual_kv_seqlen, Tensor *philox_unpack, Tensor *workspace,
                               cudaStream_t stream) {
    // TODO: add more checking
    const DType qkv_type = qkv->data.dtype;

    if (is_fp8_dtype(qkv_type) && max_q_seqlen <= 512 && max_kv_seqlen <= 512) {
        // TODD(cyanguwa): flash attention w/ fp8
    } else if (!is_fp8_dtype(qkv_type) && max_q_seqlen <= 512 && max_kv_seqlen <= 512) {
        // TODO(rewang): fused multi-head attention w/ bf16/fp16
    } else if (max_q_seqlen > 512 || max_kv_seqlen > 512) {
        NVTE_ERROR("cudnn frontend doesn't support seqlen > 512 for now. \n");
    } else {
        NVTE_ERROR("Invalid combination of data type and sequence length! \n");
    }
}

void cudnn_fused_attention_bwd(int64_t batch, int64_t max_q_seqlen, int64_t max_kv_seqlen,
                               int64_t total_seqs, int64_t num_head, int64_t head_size,
                               float scale_qk, float p_dropout, bool is_causal_masking,
                               MHA_Layout qkv_layout, Tensor *qkv, Tensor *m, Tensor *z_inv,
                               Tensor *softmax_aux, Tensor *output, Tensor *doutput, Tensor *dqkv,
                               Tensor *dsoftmax, Tensor *q_ragged_offset, Tensor *kv_ragged_offset,
                               Tensor *actual_q_seqlen, Tensor *actual_kv_seqlen,
                               Tensor *philox_unpack, Tensor *workspace, cudaStream_t stream) {
    // TODO: add more checking
    const DType qkv_type = qkv->data.dtype;

    if (is_fp8_dtype(qkv_type) && max_q_seqlen <= 512 && max_kv_seqlen <= 512) {
        // TODD(cyanguwa): flash attention w/ fp8
    } else if (!is_fp8_dtype(qkv_type) && max_q_seqlen <= 512 && max_kv_seqlen <= 512) {
        // TODO(rewang): fused multi-head attention w/ bf16/fp16
    } else if (max_q_seqlen > 512 || max_kv_seqlen > 512) {
        NVTE_ERROR("cudnn frontend doesn't support seqlen > 512 for now. \n");
    } else {
        NVTE_ERROR("Invalid combination of data type and sequence length! \n");
    }
}

}  // namespace fused_attention
}  // namespace transformer_engine

void nvte_cudnn_fused_attention_fwd(
    int64_t batch, int64_t max_q_seqlen, int64_t max_kv_seqlen, int64_t total_seqs,
    int64_t num_head, int64_t head_size, float scale_qk, float p_dropout, bool is_causal_masking,
    bool is_training, MHA_Layout qkv_layout, MHA_Bias_Type bias_type, const NVTETensor qkv,
    const NVTETensor m, const NVTETensor z_inv, const NVTETensor softmax_aux,
    const NVTETensor output, const NVTETensor bias, const NVTETensor q_ragged_offset,
    const NVTETensor kv_ragged_offset, const NVTETensor actual_q_seqlen,
    const NVTETensor actual_kv_seqlen, const NVTETensor philox_unpack, NVTETensor workspace,
    cudaStream_t stream) {
    NVTE_API_CALL(nvte_cudnn_fused_attention_fwd);
    using namespace transformer_engine;

    Tensor *qkv_tensor = reinterpret_cast<Tensor *>(qkv);
    Tensor *m_tensor = reinterpret_cast<Tensor *>(m);
    Tensor *z_inv_tensor = reinterpret_cast<Tensor *>(z_inv);
    Tensor *softmax_aux_tensor = reinterpret_cast<Tensor *>(softmax_aux);
    Tensor *output_tensor = reinterpret_cast<Tensor *>(output);
    Tensor *bias_tensor = reinterpret_cast<Tensor *>(bias);
    Tensor *q_ragged_offset_tensor = reinterpret_cast<Tensor *>(q_ragged_offset);
    Tensor *kv_ragged_offset_tensor = reinterpret_cast<Tensor *>(kv_ragged_offset);
    Tensor *actual_q_seqlen_tensor = reinterpret_cast<Tensor *>(actual_q_seqlen);
    Tensor *actual_kv_seqlen_tensor = reinterpret_cast<Tensor *>(actual_kv_seqlen);
    Tensor *philox_unpack_tensor = reinterpret_cast<Tensor *>(philox_unpack);
    Tensor *workspace_tensor = reinterpret_cast<Tensor *>(workspace);

    fused_attention::cudnn_fused_attention_fwd(
        batch, max_q_seqlen, max_kv_seqlen, total_seqs, num_head, head_size, scale_qk, p_dropout,
        is_causal_masking, is_training, qkv_layout, bias_type, qkv_tensor, m_tensor, z_inv_tensor,
        softmax_aux_tensor, output_tensor, bias_tensor, q_ragged_offset_tensor,
        kv_ragged_offset_tensor, actual_q_seqlen_tensor, actual_kv_seqlen_tensor,
        philox_unpack_tensor, workspace_tensor, stream);
}

void nvte_cudnn_fused_attention_bwd(
    int64_t batch, int64_t max_q_seqlen, int64_t max_kv_seqlen, int64_t total_seqs,
    int64_t num_head, int64_t head_size, float scale_qk, float p_dropout, bool is_causal_masking,
    MHA_Layout qkv_layout, const NVTETensor qkv, const NVTETensor m, const NVTETensor z_inv,
    const NVTETensor softmax_aux, const NVTETensor output, const NVTETensor doutput,
    const NVTETensor dqkv, const NVTETensor dsoftmax, const NVTETensor q_ragged_offset,
    const NVTETensor kv_ragged_offset, const NVTETensor actual_q_seqlen,
    const NVTETensor actual_kv_seqlen, const NVTETensor philox_unpack, NVTETensor workspace,
    cudaStream_t stream) {
    NVTE_API_CALL(nvte_cudnn_fused_attention_bwd);
    using namespace transformer_engine;

    Tensor *qkv_tensor = reinterpret_cast<Tensor *>(qkv);
    Tensor *m_tensor = reinterpret_cast<Tensor *>(m);
    Tensor *z_inv_tensor = reinterpret_cast<Tensor *>(z_inv);
    Tensor *softmax_aux_tensor = reinterpret_cast<Tensor *>(softmax_aux);
    Tensor *output_tensor = reinterpret_cast<Tensor *>(output);
    Tensor *doutput_tensor = reinterpret_cast<Tensor *>(doutput);
    Tensor *dqkv_tensor = reinterpret_cast<Tensor *>(dqkv);
    Tensor *dsoftmax_tensor = reinterpret_cast<Tensor *>(dsoftmax);
    Tensor *q_ragged_offset_tensor = reinterpret_cast<Tensor *>(q_ragged_offset);
    Tensor *kv_ragged_offset_tensor = reinterpret_cast<Tensor *>(kv_ragged_offset);
    Tensor *actual_q_seqlen_tensor = reinterpret_cast<Tensor *>(actual_q_seqlen);
    Tensor *actual_kv_seqlen_tensor = reinterpret_cast<Tensor *>(actual_kv_seqlen);
    Tensor *philox_unpack_tensor = reinterpret_cast<Tensor *>(philox_unpack);
    Tensor *workspace_tensor = reinterpret_cast<Tensor *>(workspace);

    fused_attention::cudnn_fused_attention_bwd(
        batch, max_q_seqlen, max_kv_seqlen, total_seqs, num_head, head_size, scale_qk, p_dropout,
        is_causal_masking, qkv_layout, qkv_tensor, m_tensor, z_inv_tensor, softmax_aux_tensor,
        output_tensor, doutput_tensor, dqkv_tensor, dsoftmax_tensor, q_ragged_offset_tensor,
        kv_ragged_offset_tensor, actual_q_seqlen_tensor, actual_kv_seqlen_tensor,
        philox_unpack_tensor, workspace_tensor, stream);
}
