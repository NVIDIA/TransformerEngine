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
#include "transformer_engine/activation.h"
#include "utils.h"

namespace transformer_engine {
namespace jax {

inline bool use_fp8(DType type) { return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2; }

// Activation

XLA_FFI_DECLARE_HANDLER_SYMBOL(ActLuHandler);

// Normalization
XLA_FFI_DECLARE_HANDLER_SYMBOL(NormForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(NormBackwardHandler);

pybind11::tuple GetNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size, DType in_dtype,
                                             DType w_dtype, DType out_dtype,
                                             NVTE_Norm_Type norm_type, int scaling_mode,
                                             bool zero_centered_gamma, float epsilon, int sm_margin,
                                             bool is_training);

pybind11::tuple GetNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size, DType in_dtype,
                                              DType w_dtype, NVTE_Norm_Type norm_type,
                                              bool zero_centered_gamma, int sm_margin);

// Quantization
XLA_FFI_DECLARE_HANDLER_SYMBOL(DBiasQuantizeHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DequantizeHandler);

pybind11::tuple GetDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                               DType in_dtype, DType out_dtype);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DActLuDBiasQuantizeHandler);

pybind11::tuple GetDActDBiasQuantizeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType out_dtype,
                                                   int scaling_mode, bool is_2x);

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

// GEMM

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Buffer_Type out_amax, Buffer_Type out_scale, Result_Type out,
                   Result_Type out_amax_updated, Result_Type out_scale_updated,
                   Result_Type pre_gelu_out, Result_Type bias_grad, Result_Type workspace,
                   bool lhs_trans, bool rhs_trans, bool fuse_gelu, bool fuse_bias, bool grad,
                   bool accumulate, bool use_split_accumulator);

XLA_FFI_DECLARE_HANDLER_SYMBOL(GemmHandler);

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
