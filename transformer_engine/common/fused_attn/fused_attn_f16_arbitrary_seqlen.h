/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_arbitrary_seqlen.h
 *  \brief Functions for fused attention with seqlen > 512
 */

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_

#include <cudnn.h>

#include <string>

#include "common/common.h"
#include "config_and_params.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
void fused_attn_arbitrary_seqlen_fwd(const FusedAttnConfig &cfg, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_Bias, const Tensor *input_SoftmaxOffset,
                                     Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                                     const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                                     const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *page_table_k,
                                     const Tensor *page_table_v, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

void fused_attn_arbitrary_seqlen_bwd(const FusedAttnConfig &cfg, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_O, const Tensor *input_dO,
                                     const Tensor *input_Bias, const Tensor *input_SoftmaxOffset,
                                     Tensor *output_S, Tensor *output_dQ, Tensor *output_dK,
                                     Tensor *output_dV, Tensor *output_dBias,
                                     Tensor *output_dSoftmaxOffset, const Tensor *cu_seqlens_q,
                                     const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// check if a given configuration is supported for F16/BF16 forward;
// if it is, cache the graph built for this config, and return an empty string;
// if not, return a diagnostic message explaining why it is not supported.
std::string is_supported_f16_fwd(const FusedAttnConfig &cfg, cudnnHandle_t handle);

// check if a given configuration is supported for F16/BF16 backward;
// if it is, cache the graph built for this config, and return an empty string;
// if not, return a diagnostic message explaining why it is not supported.
std::string is_supported_f16_bwd(const FusedAttnConfig &cfg, cudnnHandle_t handle);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_FUSED_ATTN_ARBITRARY_SEQLEN_H_
