/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <transformer_engine/normalization.h>
#include <transformer_engine/transformer_engine.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/util/logging.h"
#include "extensions/ffi.h"
#include "extensions/misc.h"
#include "extensions/utils.h"
#include "transformer_engine/activation.h"

// ENUM_ATTR and DICT_ATTR recoding need to be registered in the global namespace
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(transformer_engine::jax::JAXX_Scaling_Mode);

namespace transformer_engine {
namespace jax {

inline bool use_fp8(DType type) { return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2; }

// Activation

XLA_FFI_DECLARE_HANDLER_SYMBOL(ActLuHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DActLuDBiasQuantizeHandler);

pybind11::tuple GetDActDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType out_dtype,
                                                   JAXX_Scaling_Mode scaling_mode, bool is_2x);

// Normalization
XLA_FFI_DECLARE_HANDLER_SYMBOL(NormForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(NormBackwardHandler);

pybind11::tuple GetNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size, DType in_dtype,
                                             DType w_dtype, DType out_dtype,
                                             NVTE_Norm_Type norm_type,
                                             JAXX_Scaling_Mode scaling_mode,
                                             bool zero_centered_gamma, float epsilon, int sm_margin,
                                             bool is_training);

pybind11::tuple GetNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size, DType in_dtype,
                                              DType w_dtype, NVTE_Norm_Type norm_type,
                                              bool zero_centered_gamma, int sm_margin);

// Quantization
XLA_FFI_DECLARE_HANDLER_SYMBOL(DBiasQuantizeHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedQuantizeHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DequantizeHandler);

pybind11::tuple GetDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                               DType in_dtype, DType out_dtype,
                                               JAXX_Scaling_Mode scaling_mode,
                                               QuantizeLayout q_layout);

// Softmax
XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledSoftmaxForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledSoftmaxBackwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledMaskedSoftmaxForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledMaskedSoftmaxBackwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledUpperTriangMaskedSoftmaxForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledUpperTriangMaskedSoftmaxBackwardHandler);

// Attention
XLA_FFI_DECLARE_HANDLER_SYMBOL(FusedAttnForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(FusedAttnBackwardHandler);

NVTE_Fused_Attn_Backend GetFusedAttnBackend(DType q_dtype, DType kv_dtype,
                                            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                            NVTE_Mask_Type mask_type, float dropout_probability,
                                            size_t q_num_heads, size_t kv_num_heads,
                                            size_t q_max_seqlen, size_t kv_max_seqlen,
                                            size_t head_dim, int64_t window_size_left,
                                            int64_t window_size_right);

pybind11::tuple GetFusedAttnForwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training,
    size_t max_segments_per_seq, int64_t window_size_left, int64_t window_size_right);

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training,
    bool deterministic, size_t max_segments_per_seq, int64_t window_size_left,
    int64_t window_size_right);

// Grouped GEMM
XLA_FFI_DECLARE_HANDLER_SYMBOL(GroupedGemmHandler);

// Cudnn helpers
XLA_FFI_DECLARE_HANDLER_SYMBOL(CudnnHandleInitHandler);

// CuBLAS helpers
XLA_FFI_DECLARE_HANDLER_SYMBOL(CublasHandleInitHandler);

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
