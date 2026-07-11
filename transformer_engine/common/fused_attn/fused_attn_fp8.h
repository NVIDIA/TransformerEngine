/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn_fp8.h
 *  \brief Functions for fused attention for FP8
 */

#include <string>

#include "config_and_params.h"
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
// fused attention FWD FP8 with separate Q, K, V
void fused_attn_fp8_fwd(
    const FusedAttnConfig &cfg, const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V,
    const Tensor *input_SoftmaxOffset, Tensor *input_output_S, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *rng_state, Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(
    const FusedAttnConfig &cfg, const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V,
    const Tensor *input_O, const Tensor *input_dO, const Tensor *input_dO_f16, const Tensor *input_M,
    const Tensor *input_S, const Tensor *input_SoftmaxOffset, Tensor *input_output_dP,
    const Tensor *output_dQ, const Tensor *output_dK, const Tensor *output_dV,
    Tensor *output_dSoftmaxOffset, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *rng_state, Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle);

// check if a given configuration is supported for FP8 forward;
// if it is, cache the graph built for this config, and return an empty string;
// if not, return a diagnostic message in the form of a string.
std::string is_supported_fp8_fwd(const FusedAttnConfig &cfg, cudnnHandle_t handle);

// check if a given configuration is supported for FP8 backward;
// if it is, cache the graph built for this config, and return an empty string;
// if not, return a diagnostic message in the form of a string.
std::string is_supported_fp8_bwd(const FusedAttnConfig &cfg, cudnnHandle_t handle);
}  // namespace transformer_engine
