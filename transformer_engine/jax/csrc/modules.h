/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <transformer_engine/fused_attn.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/activation.h>
#include "common/util/logging.h"

namespace transformer_engine {
namespace jax {

constexpr int kMaxNumDim = 8;

size_t get_activation_len(NVTE_Activation_Type activation_enum);

struct Shape {
    int num_dim;
    size_t dims[kMaxNumDim];

    void from_vector(const std::vector<size_t> &shape) {
        num_dim = shape.size();
        assert(num_dim <= kMaxNumDim);
        std::memcpy(dims, shape.data(), num_dim * sizeof(size_t));
    }

    std::vector<size_t> to_vector() const {
        assert(num_dim <= kMaxNumDim);
        std::vector<size_t> shape(num_dim);
        std::memcpy(shape.data(), dims, num_dim * sizeof(size_t));
        return shape;
    }
};

enum class NVTE_Activation_Enum {
  GELU,
  GEGLU,
  SILU,
  SWIGLU,
};

size_t get_activation_len(NVTE_Activation_Enum act_enum);

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
                                                 const std::vector<size_t> &wkshape,
                                                 DType in_dtype, DType out_dtype, DType wk_dtype,
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
    size_t wkspace_size;
    float scaling_factor;
    float dropout_probability;
    NVTE_Bias_Type bias_type;
    NVTE_Mask_Type mask_type;
    NVTE_QKV_Layout qkv_layout;
    DType dtype;
    DType wkspace_dtype;
    bool is_training;
};

pybind11::bytes PackCustomCallFusedAttnDescriptor(
    size_t input_batch, size_t batch_size, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    size_t wkspace_size, float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, DType wkspace_dtype,
    bool is_training);

NVTE_Fused_Attn_Backend GetFusedAttnBackend(DType q_dtype, DType kv_dtype,
                                            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                            NVTE_Mask_Type mask_type, float dropout_probability,
                                            size_t q_num_heads, size_t kv_num_heads,
                                            size_t q_max_seqlen, size_t kv_max_seqlen,
                                            size_t head_dim);

void Transpose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void CastTranspose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void ActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void ActLuFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void DActLu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

pybind11::tuple GetDActDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                         DType in_dtype, DType out_dtype);

void DActLuDBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                             size_t opaque_len);

pybind11::tuple GetDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                         DType in_dtype, DType out_dtype);

void DBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                             size_t opaque_len);

void DGatedActLuCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                             size_t opaque_len);

pybind11::tuple GetLayerNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                  DType in_dtype, DType w_dtype, DType out_dtype,
                                                  bool is_layer_norm, bool zero_centered_gamma,
                                                  float eps);

void LayerNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void LayerNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque,
                         size_t opaque_len);

pybind11::tuple GetLayerNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType w_dtype,
                                                   bool is_layer_norm, bool zero_centered_gamma,
                                                   float eps);

void LayerNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void RMSNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void RMSNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void RMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void Quantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void Dequantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

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

pybind11::tuple GetFusedAttnForwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training);

void FusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training);

void FusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
