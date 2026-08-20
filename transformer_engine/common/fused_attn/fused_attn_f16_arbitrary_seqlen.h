/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_arbitrary_seqlen.h
 *  \brief Functions for fused attention with seqlen > 512
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_F16_ARBITRARY_SEQLEN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_F16_ARBITRARY_SEQLEN_H_

#include <cudnn.h>

#include <string>

#include "common/common.h"
#include "config_and_params.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
void fused_attn_arbitrary_seqlen_fwd(const fused_attn::FusedAttnConfig &cfg, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_Bias, const Tensor *input_SoftmaxOffset,
                                     Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                                     const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                                     const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *page_table_k,
                                     const Tensor *page_table_v, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd(const fused_attn::FusedAttnConfig &cfg, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_O, const Tensor *input_dO,
                                     const Tensor *input_Bias, const Tensor *input_SoftmaxOffset,
                                     Tensor *output_S, Tensor *output_dQ, Tensor *output_dK,
                                     Tensor *output_dV, Tensor *output_dBias,
                                     Tensor *output_dSoftmaxOffset, const Tensor *cu_seqlens_q,
                                     const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// cuDNN's verdict on this config's F16/BF16 graph for `pass`: an empty string if it can run,
// otherwise a diagnostic message explaining why not. A verdict of "supported" leaves the graph in
// the cache, where the execution path finds it.
//
// The direction is a runtime argument, not two functions, because the graph builder it selects is
// local to this translation unit -- so this is the only place that can map one to the other.
std::string support_verdict_f16(const fused_attn::FusedAttnConfig &cfg, fused_attn::Pass pass,
                                cudnnHandle_t handle);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_F16_ARBITRARY_SEQLEN_H_
