/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
#if (CUDNN_VERSION >= 8900)
// fused attention FWD FP8 with packed QKV
void fused_attn_fwd_fp8_qkvpacked(
            size_t b, size_t max_seqlen,
            size_t h, size_t d,
            bool is_training, float attn_scale,
            float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_QKV,
            Tensor *input_output_S,
            Tensor *output_O,
            NVTETensorPack* Aux_Output_Tensors,
            const Tensor *cu_seqlens,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle);

// fused attention BWD FP8 with packed QKV
void fused_attn_bwd_fp8_qkvpacked(
            size_t b, size_t max_seqlen,
            size_t h, size_t d,
            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_QKV,
            const Tensor *input_O,
            const Tensor *input_dO,
            const Tensor *input_M,
            const Tensor *input_ZInv,
            const Tensor *input_S,
            Tensor *input_output_dP,
            const Tensor *output_dQKV,
            const Tensor *cu_seqlens,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle);
#endif  // end of CUDNN>=8900
}  // namespace transformer_engine
