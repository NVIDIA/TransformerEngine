/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/util/logging.h"
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/transformer_engine.h>

namespace transformer_engine {
namespace jax {

constexpr int kMaxNumDim = 8;

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

struct CustomCallCommonDescriptor {
    Shape shape;
    DType in_dtype;
    DType out_dtype;
};

pybind11::bytes PackCustomCallCommonDescriptor(const std::vector<size_t> &shape, DType in_dtype,
                                               DType out_dtype);

struct CustomCallGemmDescriptor {
    size_t m;
    size_t n;
    size_t k;
    DType A_dtype;
    DType B_dtype;
    DType D_dtype;
    bool transa;
    bool transb;
    bool use_split_accumulator;
};

pybind11::bytes PackCustomCallGemmDescriptor(size_t m, size_t n, size_t k, DType A_dtype,
                                             DType B_dtype, DType D_dtype, bool transa, bool transb,
                                             bool use_split_accumulator);

struct CustomCallNormDescriptor {
    size_t n;
    size_t hidden;
    DType x_dtype;
    DType w_dtype;
    bool zero_centered_gamma;
    float eps;
};

pybind11::bytes PackCustomCallNormDescriptor(size_t n, size_t hidden, DType x_dtype, DType w_dtype,
                                             bool zero_centered_gamma, float eps);

struct SoftmaxDescriptor {
    size_t batch;
    size_t pad_batch;
    size_t heads;
    size_t q_seqlen;
    size_t k_seqlen;
    DType dtype;
    float scale_factor;
};

pybind11::bytes PackCustomCallSoftmaxDescriptor(size_t batch, size_t pad_batch, size_t heads,
                                                size_t q_seqlen, size_t k_seqlen, DType dtype,
                                                float scale_factor);

struct CustomCallFusedAttnDescriptor {
    size_t batch;
    size_t num_head;
    size_t q_max_seqlen;
    size_t kv_max_seqlen;
    size_t head_dim;
    float scaling_factor;
    float dropout_probability;
    NVTE_Bias_Type bias_type;
    NVTE_Mask_Type mask_type;
    DType dtype;
    bool is_training;
};

pybind11::bytes PackCustomCallFusedAttnDescriptor(
    size_t batch, size_t num_head, size_t q_max_seqlen, size_t kv_max_seqlen, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, DType dtype, bool is_training);

NVTE_Fused_Attn_Backend GetFusedAttnBackend(DType q_dtype, DType kv_dtype,
                                            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                            NVTE_Mask_Type mask_type, float dropout_probability,
                                            size_t q_max_seqlen, size_t kv_max_seqlen,
                                            size_t head_dim);

void Transpose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void CastTranspose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void GatedGelu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void GatedGeluFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void DGatedGelu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void DGatedGeluCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                             size_t opaque_len);

void Gemm(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void LayerNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

void LayerNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque,
                         size_t opaque_len);

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

void SelfFusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque,
                          size_t opaque_len);

void SelfFusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque,
                           size_t opaque_len);

void CrossFusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque,
                           size_t opaque_len);

void CrossFusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque,
                            size_t opaque_len);

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_FP8_MODULES_H_
