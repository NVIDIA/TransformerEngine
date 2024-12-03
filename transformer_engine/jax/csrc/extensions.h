/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_H_

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <transformer_engine/comm_gemm_overlap.h>
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

// Phuong: These 3 functions need to stay in the header file for compilation purpose
// 1.
inline bool use_fp8(DType type) { return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2; }
// 2.
template <typename T>
pybind11::bytes PackOpaque(const T &descriptor) {
  auto str = std::string(reinterpret_cast<const char *>(&descriptor), sizeof(T));
  return pybind11::bytes(str);
}
// 3.
template <typename T>
const T *UnpackOpaque(const char *opaque, size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return reinterpret_cast<const T *>(opaque);
}

// Packing

struct CustomCallCommonDescriptor {
  Shape shape;
  DType in_dtype;
  DType out_dtype;
  size_t act_enum;
};

pybind11::bytes PackCustomCallCommonDescriptor(const std::vector<size_t> &shape, DType in_dtype,
                                               DType out_dtype, size_t act_enum = 0);

struct CustomCallCommonWkDescriptor {
  Shape shape;
  Shape wkshape;
  DType in_dtype;
  DType out_dtype;
  DType wk_dtype;
  size_t act_enum;
};

pybind11::bytes PackCustomCallCommonWkDescriptor(const std::vector<size_t> &shape,
                                                 const std::vector<size_t> &wkshape, DType in_dtype,
                                                 DType out_dtype, DType wk_dtype,
                                                 size_t act_enum = 0);

struct CustomCallNormDescriptor {
  size_t batch_size;
  size_t hidden_size;
  size_t wkspace_size;
  size_t barrier_size;
  Shape dgamma_part_shape;
  Shape dbeta_part_shape;
  DType x_dtype;
  DType w_dtype;
  DType wkspace_dtype;
  DType barrier_dtype;
  DType dgamma_part_dtype;
  DType dbeta_part_dtype;
  bool zero_centered_gamma;
  float eps;
  int sm_margin;
};

pybind11::bytes PackCustomCallNormDescriptor(
    size_t batch_size, size_t hidden_size, size_t wkspace_size, size_t barrier_size,
    const std::vector<size_t> &dgamma_part_shape, const std::vector<size_t> &dbeta_part_shape,
    DType x_dtype, DType w_dtype, DType wkspace_dtype, DType barrier_dtype, DType dgamma_part_dtype,
    DType dbeta_part_dtype, bool zero_centered_gamma, float eps, int sm_margin);

struct SoftmaxDescriptor {
  size_t batch_size;
  size_t padding_size;
  size_t head_dim;
  size_t q_seqlen;
  size_t k_seqlen;
  DType dtype;
  float scale_factor;
};

pybind11::bytes PackCustomCallSoftmaxDescriptor(size_t batch_size, size_t padding_size,
                                                size_t head_dim, size_t q_seqlen, size_t k_seqlen,
                                                DType dtype, float scale_factor);

struct CustomCallFusedAttnDescriptor {
  size_t input_batch;
  size_t bias_batch;
  size_t q_max_seqlen;
  size_t kv_max_seqlen;
  size_t attn_heads;
  size_t num_gqa_groups;
  size_t bias_heads;
  size_t head_dim;
  size_t max_segments_per_seq;
  size_t wkspace_size;
  float scaling_factor;
  float dropout_probability;
  NVTE_Bias_Type bias_type;
  NVTE_Mask_Type mask_type;
  NVTE_QKV_Layout qkv_layout;
  DType dtype;
  DType wkspace_dtype;
  bool is_training;
  bool deterministic;
  int64_t window_size_left;
  int64_t window_size_right;
};

pybind11::bytes PackCustomCallFusedAttnDescriptor(
    size_t input_batch, size_t batch_size, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    size_t max_segments_per_seq, size_t wkspace_size, float scaling_factor,
    float dropout_probability, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_QKV_Layout qkv_layout, DType dtype, DType wkspace_dtype, bool is_training,
    bool deterministic, int64_t window_size_left, int64_t window_size_right);

struct CustomCallGemmDescriptor {
  size_t m;
  size_t k;
  size_t n;
  size_t workspace_size;
  DType operand_dtype;
  DType bias_dtype;
  DType out_dtype;
  bool lhs_trans;
  bool rhs_trans;
  bool fuse_gelu;
  bool fuse_bias;
  bool grad;
  bool accumulate;
  bool use_split_accumulator;
};

pybind11::bytes PackCustomCallGemmDescriptor(size_t m, size_t n, size_t k, size_t workspace_size,
                                             DType operand_dtype, DType out_dtype, DType bias_dtype,
                                             bool lhs_trans, bool rhs_trans, bool fuse_gelu,
                                             bool fuse_bias, bool grad, bool accumulate,
                                             bool use_split_accumulator);

// Transpose

void Transpose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(TransposeHandler);

void CastTranspose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(CastTransposeHandler);

pybind11::tuple GetDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                    DType in_dtype, DType out_dtype);

void DBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DBiasCastTransposeHandler);

// Activation

size_t get_activation_len(NVTE_Activation_Type activation_enum);

void ActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ActLuHandler);

void ActLuFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ActLuFP8Handler);

void DActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DActLuHandler);

pybind11::tuple GetDActDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                        DType in_dtype, DType out_dtype);

void DActLuDBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                              size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DActLuDBiasCastTransposeHandler);

void DGatedActLuCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                              size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DGatedActLuCastTransposeHandler);

// Normalization

pybind11::tuple GetLayerNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                  DType in_dtype, DType w_dtype, DType out_dtype,
                                                  bool is_layer_norm, bool zero_centered_gamma,
                                                  float eps, int sm_margin);

void LayerNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(LayerNormForwardHandler);

void LayerNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque,
                         size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(LayerNormForwardFP8Handler);

pybind11::tuple GetLayerNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType w_dtype,
                                                   bool is_layer_norm, bool zero_centered_gamma,
                                                   float eps, int sm_margin);

void LayerNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(LayerNormBackwardHandler);

void RMSNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormForwardHandler);

void RMSNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormForwardFP8Handler);

void RMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(RMSNormBackwardHandler);

// Quantization

void Quantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(QuantizeHandler);

void Dequantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DequantizeHandler);

// Softmax

void ScaledSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                          std::size_t opaque_len);

void ScaledSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                           std::size_t opaque_len);

void ScaledMaskedSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                                std::size_t opaque_len);

void ScaledMaskedSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                                 std::size_t opaque_len);

void ScaledUpperTriangMaskedSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                                           std::size_t opaque_len);

void ScaledUpperTriangMaskedSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                                            std::size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledSoftmaxForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledSoftmaxBackwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledMaskedSoftmaxForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledMaskedSoftmaxBackwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledUpperTriangMaskedSoftmaxForwardHandler);

XLA_FFI_DECLARE_HANDLER_SYMBOL(ScaledUpperTriangMaskedSoftmaxBackwardHandler);

// Attention

// Cudnn helpers
XLA_FFI_DECLARE_HANDLER_SYMBOL(CudnnHandleInitHandler);

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

void FusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(FusedAttnForwardHandler);

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training,
    bool deterministic, size_t max_segments_per_seq, int64_t window_size_left,
    int64_t window_size_right);

void FusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

XLA_FFI_DECLARE_HANDLER_SYMBOL(FusedAttnBackwardHandler);

// GEMM

XLA_FFI_DECLARE_HANDLER_SYMBOL(CublasltHandleInitHandler);

void Gemm(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

Error_Type GemmFFI(
    cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
    Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input, Buffer_Type out,
    Buffer_Type out_amax, Buffer_Type out_scale, Buffer_Type dummy_in, Result_Type out_updated,
    Result_Type out_amax_updated, Result_Type out_scale_updated, Result_Type pre_gelu_out,
    Result_Type bias_grad, Result_Type dummy_out, Result_Type workspace, bool lhs_trans,
    bool rhs_trans, bool fuse_gelu, bool fuse_bias, bool grad, bool accumulate,
    bool use_split_accumulator);

XLA_FFI_DECLARE_HANDLER_SYMBOL(GemmHandler);

// Comm+GEMM Overlap

bool OverlapBufferIsFp8(const std::string &name);

pybind11::object GetOverlapBuffer(const std::string &name, bool sharded);

void SetOverlapBufferScaleInverse(const std::string &name, pybind11::object scale_inv, bool grad);

void BootstrapCommGemmOverlap(
    const std::vector<size_t> &buffer_shape, DType buffer_dtype, const std::string &name,
    const std::string &method, CommOverlapType comm_type, int64_t myrank, int64_t numranks,
    int64_t tp_size, int64_t num_splits, int64_t num_max_streams, int64_t cga_size,
    int64_t num_comm_sm, bool set_sm_margin, bool use_ce, bool atomic_gemm, bool aggregate,
    bool pipeline_rs_overlap_first_gemm);

Error_Type BootstrapCommGemmOverlapFFI(
    cudaStream_t, Buffer_Type sample_buffer, std::string_view name, std::string_view method,
    int64_t comm_type_flag, int64_t myrank, int64_t numranks, int64_t tp_size, int64_t num_splits,
    int64_t num_max_streams, int64_t cga_size, int64_t num_comm_sm, bool set_sm_margin,
    bool use_ce, bool atomic_gemm, bool aggregate, bool pipeline_rs_overlap_first_gemm);

XLA_FFI_DECLARE_HANDLER_SYMBOL(BootstrapCommGemmOverlapHandler);

void DestroyCommGemmOverlap(const std::string &name);

Error_Type DestroyCommGemmOverlapFFI(cudaStream_t stream, std::string_view name);

XLA_FFI_DECLARE_HANDLER_SYMBOL(DestroyCommGemmOverlapHandler);

Error_Type CopyIntoOverlapBufferFFI(cudaStream_t stream, Buffer_Type input, std::string_view name,
                                    bool sharded);

XLA_FFI_DECLARE_HANDLER_SYMBOL(CopyIntoOverlapBufferHandler);

Error_Type CommGemmOverlapFFI(
    cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
    Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input, Buffer_Type out,
    Buffer_Type out_amax, Buffer_Type out_scale, Buffer_Type extra_out, Result_Type out_updated,
    Result_Type out_amax_updated, Result_Type out_scale_updated, Result_Type pre_gelu_out,
    Result_Type bias_grad, Result_Type extra_out_updated, Result_Type workspace, bool lhs_trans,
    bool rhs_trans, bool fuse_gelu, bool fuse_bias, bool grad, bool accumulate,
    bool use_split_accumulator, int64_t comm_type_flag, std::string_view name);

XLA_FFI_DECLARE_HANDLER_SYMBOL(CommGemmOverlapHandler);

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_EXTENSIONS_H_
