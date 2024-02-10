/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "jax/csrc/modules.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/util/logging.h"
#include "transformer_engine/activation.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/layer_norm.h"
#include "transformer_engine/rmsnorm.h"
#include "transformer_engine/softmax.h"
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/transpose.h"
#include "utils.h"

namespace transformer_engine {
namespace jax {

inline bool use_fp8(DType type) { return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2; }

std::vector<size_t> MakeShapeVector(NVTEShape shape) {
    return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

template <typename T>
pybind11::bytes PackOpaque(const T &descriptor) {
    auto str = std::string(reinterpret_cast<const char *>(&descriptor), sizeof(T));
    return pybind11::bytes(str);
}

template <typename T>
const T *UnpackOpaque(const char *opaque, size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size");
    }
    return reinterpret_cast<const T *>(opaque);
}

pybind11::bytes PackCustomCallCommonDescriptor(const std::vector<size_t> &shape, DType in_dtype,
                                               DType out_dtype) {
    CustomCallCommonDescriptor desc;
    desc.shape.from_vector(shape);
    desc.in_dtype = in_dtype;
    desc.out_dtype = out_dtype;
    return PackOpaque(desc);
}

pybind11::bytes PackCustomCallCommonWkDescriptor(const std::vector<size_t> &shape,
                                                 const std::vector<size_t> &wkshape, DType in_dtype,
                                                 DType out_dtype, DType wk_dtype) {
    CustomCallCommonWkDescriptor desc;
    desc.shape.from_vector(shape);
    desc.wkshape.from_vector(wkshape);
    desc.in_dtype = in_dtype;
    desc.out_dtype = out_dtype;
    desc.wk_dtype = wk_dtype;
    return PackOpaque(desc);
}

pybind11::bytes PackCustomCallNormDescriptor(size_t batch_size, size_t hidden_size,
                                             size_t wkspace_size, size_t barrier_size,
                                             size_t *dgamma_part_sizes, size_t *dbeta_part_sizes,
                                             DType x_dtype, DType w_dtype, DType wkspace_dtype,
                                             DType barrier_dtype, DType dgamma_part_dtype,
                                             DType dbeta_part_dtype, bool zero_centered_gamma,
                                             float eps, int sm_margin) {
    return PackOpaque(CustomCallNormDescriptor{
        batch_size, hidden_size, wkspace_size, barrier_size, dgamma_part_sizes, dbeta_part_sizes,
        x_dtype, w_dtype, wkspace_dtype, barrier_dtype, dgamma_part_dtype, dbeta_part_dtype,
        zero_centered_gamma, eps, sm_margin});
}

pybind11::bytes PackCustomCallSoftmaxDescriptor(size_t batch_size, size_t padding_size,
                                                size_t head_dim, size_t q_seqlen, size_t k_seqlen,
                                                DType dtype, float scale_factor) {
    return PackOpaque(SoftmaxDescriptor{batch_size, padding_size, head_dim, q_seqlen, k_seqlen,
                                        dtype, scale_factor});
}

pybind11::bytes PackCustomCallFusedAttnDescriptor(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    size_t wkspace_size, float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, DType wkspace_dtype,
    bool is_training) {
    return PackOpaque(CustomCallFusedAttnDescriptor{
        input_batch, bias_batch, q_max_seqlen, kv_max_seqlen, attn_heads, num_gqa_groups,
        bias_heads, head_dim, wkspace_size, scaling_factor, dropout_probability, bias_type,
        mask_type, qkv_layout, dtype, wkspace_dtype, is_training});
}

void TransposeImpl(void *input, size_t rows, size_t cols, DType dtype, cudaStream_t stream,
                   void *output) {
    auto input_shape = std::vector<size_t>{rows, cols};
    auto output_shape = std::vector<size_t>{cols, rows};

    auto input_tensor = TensorWrapper(input, input_shape, dtype);
    auto transposed_tensor = TensorWrapper(output, output_shape, dtype);

    nvte_transpose(input_tensor.data(), transposed_tensor.data(), stream);
}

void Transpose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    void *input = buffers[0];
    void *output = buffers[1];

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    auto rows = desc.shape.dims[0];
    auto cols = desc.shape.dims[1];
    assert(desc.in_dtype == desc.out_dtype);
    auto dtype = desc.out_dtype;

    TransposeImpl(input, rows, cols, dtype, stream, output);
}

void CastTranspose(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    float *amax = reinterpret_cast<float *>(buffers[1]);
    float *scale = reinterpret_cast<float *>(buffers[2]);
    float *scale_inv = reinterpret_cast<float *>(buffers[3]);
    auto *input_cast = buffers[4];
    auto *input_cast_trans = buffers[5];
    float *amax_out = reinterpret_cast<float *>(buffers[6]);
    assert(amax == amax_out);

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    if (!use_fp8(desc.out_dtype)) {
        scale = nullptr;
        scale_inv = nullptr;
        amax_out = nullptr;
    }
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];
    auto input_shape = std::vector<size_t>{m, n};
    auto input_trans_shape = std::vector<size_t>{n, m};

    auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
    auto input_cast_tensor =
        TensorWrapper(input_cast, input_shape, desc.out_dtype, amax_out, scale, scale_inv);
    auto input_cast_trans_tensor = TensorWrapper(input_cast_trans, input_trans_shape,
                                                 desc.out_dtype, amax_out, scale, scale_inv);

    nvte_cast_transpose(input_tensor.data(), input_cast_tensor.data(),
                        input_cast_trans_tensor.data(), stream);
}

void GeluImpl(void *input, size_t m, size_t n, DType in_dtype, DType out_dtype, float *scale,
              cudaStream_t stream, float *scale_inverse, float *amax, void *output) {
    auto input_shape = std::vector<size_t>{m, n};
    auto output_shape = std::vector<size_t>{m, n};

    auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));

    auto output_tensor = TensorWrapper(output, output_shape, static_cast<DType>(out_dtype), amax,
                                       scale, scale_inverse);

    nvte_gelu(input_tensor.data(), output_tensor.data(), stream);
}

void Gelu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *output = buffers[1];

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];

    GeluImpl(input, m, n, desc.in_dtype, desc.out_dtype, nullptr, stream, nullptr, nullptr, output);
}

void GeluFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    float *amax = reinterpret_cast<float *>(buffers[1]);
    float *scale = reinterpret_cast<float *>(buffers[2]);
    float *scale_inv = reinterpret_cast<float *>(buffers[3]);
    auto *output = buffers[4];
    float *amax_out = reinterpret_cast<float *>(buffers[5]);
    assert(amax == amax_out);

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    if (!use_fp8(desc.out_dtype)) {
        scale = nullptr;
        scale_inv = nullptr;
        amax_out = nullptr;
    }
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];

    GeluImpl(input, m, n, desc.in_dtype, desc.out_dtype, scale, stream, scale_inv, amax_out,
             output);
}

void DGelu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *gelu_input = buffers[1];
    auto *output = buffers[2];

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];
    auto input_shape = std::vector<size_t>{m, n};
    auto gelu_input_shape = std::vector<size_t>{m, n};
    auto output_shape = std::vector<size_t>{m, n};

    auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
    auto gelu_input_tensor = TensorWrapper(gelu_input, gelu_input_shape, desc.in_dtype);
    auto output_tensor = TensorWrapper(output, output_shape, desc.out_dtype);

    nvte_dgelu(input_tensor.data(), gelu_input_tensor.data(), output_tensor.data(), stream);
}

pybind11::tuple GetDGeluDBiasCastTransposeWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                         DType in_dtype, DType out_dtype) {
    auto input_shape = std::vector<size_t>{batch_size, hidden_size};
    auto gelu_input_shape = std::vector<size_t>{batch_size, hidden_size};
    auto output_shape = std::vector<size_t>{batch_size, hidden_size};
    auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
    auto dbias_shape = std::vector<size_t>{hidden_size};

    auto input_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
    auto gelu_input_tensor = TensorWrapper(nullptr, gelu_input_shape, in_dtype);
    auto output_tensor = TensorWrapper(nullptr, output_shape, out_dtype);
    auto output_trans_tensor = TensorWrapper(nullptr, output_trans_shape, out_dtype);
    auto dbias_tensor = TensorWrapper(nullptr, dbias_shape, in_dtype);

    TensorWrapper dummy_workspace;

    nvte_cast_transpose_dbias_dgelu(input_tensor.data(), gelu_input_tensor.data(),
                                    output_tensor.data(), output_trans_tensor.data(),
                                    dbias_tensor.data(), dummy_workspace.data(), nullptr);

    auto work_shape = MakeShapeVector(dummy_workspace.shape());
    return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

void DGeluDBiasCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                             size_t opaque_len) {
    auto *input = buffers[0];
    auto *gelu_input = buffers[1];
    float *amax = reinterpret_cast<float *>(buffers[2]);
    float *scale = reinterpret_cast<float *>(buffers[3]);
    float *scale_inv = reinterpret_cast<float *>(buffers[4]);
    auto *output = buffers[5];
    auto *output_trans = buffers[6];
    auto *dbias = buffers[7];
    float *amax_out = reinterpret_cast<float *>(buffers[8]);
    void *workspace_ptr = buffers[9];

    const auto &desc = *UnpackOpaque<CustomCallCommonWkDescriptor>(opaque, opaque_len);
    assert(amax == amax_out);
    if (!use_fp8(desc.out_dtype)) {
        scale = nullptr;
        scale_inv = nullptr;
        amax_out = nullptr;
    }
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];
    auto input_shape = std::vector<size_t>{m, n};
    auto gelu_input_shape = std::vector<size_t>{m, n};
    auto output_shape = std::vector<size_t>{m, n};
    auto output_trans_shape = std::vector<size_t>{n, m};
    auto dbias_shape = std::vector<size_t>{n};

    auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
    auto gelu_input_tensor = TensorWrapper(gelu_input, gelu_input_shape, desc.in_dtype);
    auto output_tensor =
        TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
    auto output_trans_tensor =
        TensorWrapper(output_trans, output_trans_shape, desc.out_dtype, amax_out, scale, scale_inv);
    auto dbias_tensor = TensorWrapper(dbias, dbias_shape, desc.in_dtype);

    auto workspace = TensorWrapper(workspace_ptr, desc.wkshape.to_vector(), desc.wk_dtype);

    nvte_cast_transpose_dbias_dgelu(input_tensor.data(), gelu_input_tensor.data(),
                                    output_tensor.data(), output_trans_tensor.data(),
                                    dbias_tensor.data(), workspace.data(), stream);
}

void GatedGeluImpl(void *input, size_t m, size_t n, DType in_dtype, DType out_dtype, float *scale,
                   cudaStream_t stream, float *scale_inverse, float *amax, void *output) {
    auto input_shape = std::vector<size_t>{m, n * 2};
    auto output_shape = std::vector<size_t>{m, n};

    auto input_tensor = TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));

    auto output_tensor = TensorWrapper(output, output_shape, static_cast<DType>(out_dtype), amax,
                                       scale, scale_inverse);

    nvte_geglu(input_tensor.data(), output_tensor.data(), stream);
}

void GatedGelu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *output = buffers[1];

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];

    GatedGeluImpl(input, m, n, desc.in_dtype, desc.out_dtype, nullptr, stream, nullptr, nullptr,
                  output);
}

void GatedGeluFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    float *amax = reinterpret_cast<float *>(buffers[1]);
    float *scale = reinterpret_cast<float *>(buffers[2]);
    float *scale_inv = reinterpret_cast<float *>(buffers[3]);
    auto *output = buffers[4];
    float *amax_out = reinterpret_cast<float *>(buffers[5]);
    assert(amax == amax_out);

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    if (!use_fp8(desc.out_dtype)) {
        scale = nullptr;
        scale_inv = nullptr;
        amax_out = nullptr;
    }
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];

    GatedGeluImpl(input, m, n, desc.in_dtype, desc.out_dtype, scale, stream, scale_inv, amax_out,
                  output);
}

void DGatedGelu(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *gelu_input = buffers[1];
    auto *output = buffers[2];

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];
    auto input_shape = std::vector<size_t>{m, n};
    auto gelu_input_shape = std::vector<size_t>{m, n * 2};
    auto output_shape = std::vector<size_t>{m, n * 2};

    auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
    auto gelu_input_tensor = TensorWrapper(gelu_input, gelu_input_shape, desc.in_dtype);
    auto output_tensor = TensorWrapper(output, output_shape, desc.out_dtype);

    nvte_dgeglu(input_tensor.data(), gelu_input_tensor.data(), output_tensor.data(), stream);
}

void DGatedGeluCastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                             size_t opaque_len) {
    auto *input = buffers[0];
    auto *gelu_input = buffers[1];
    float *amax = reinterpret_cast<float *>(buffers[2]);
    float *scale = reinterpret_cast<float *>(buffers[3]);
    float *scale_inv = reinterpret_cast<float *>(buffers[4]);
    auto *output = buffers[5];
    auto *output_trans = buffers[6];
    float *amax_out = reinterpret_cast<float *>(buffers[7]);

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    assert(amax == amax_out);
    if (!use_fp8(desc.out_dtype)) {
        scale = nullptr;
        scale_inv = nullptr;
        amax_out = nullptr;
    }
    auto m = desc.shape.dims[0];
    auto n = desc.shape.dims[1];
    auto input_shape = desc.shape.to_vector();
    auto gelu_input_shape = std::vector<size_t>{m, n * 2};
    auto output_shape = std::vector<size_t>{m, n * 2};
    auto output_trans_shape = std::vector<size_t>{n * 2, m};

    auto input_tensor = TensorWrapper(input, input_shape, desc.in_dtype);
    auto gelu_input_tensor = TensorWrapper(gelu_input, gelu_input_shape, desc.in_dtype);
    auto output_tensor =
        TensorWrapper(output, output_shape, desc.out_dtype, amax_out, scale, scale_inv);
    auto output_trans_tensor =
        TensorWrapper(output_trans, output_trans_shape, desc.out_dtype, amax_out, scale, scale_inv);

    nvte_dgeglu_cast_transpose(input_tensor.data(), gelu_input_tensor.data(), output_tensor.data(),
                               output_trans_tensor.data(), stream);
}

pybind11::tuple GetLayerNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                  DType in_dtype, DType w_dtype, DType out_dtype,
                                                  bool is_layer_norm, bool zero_centered_gamma,
                                                  float eps) {
    auto input_shape = std::vector<size_t>{batch_size, hidden_size};
    auto weight_shape = std::vector<size_t>{hidden_size};
    auto intermediates_shape = std::vector<size_t>{batch_size};

    // empty tensor wrappers are okay just to get workspace size
    auto input_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
    auto gamma_tensor = TensorWrapper(nullptr, weight_shape, in_dtype);
    auto output_tensor = TensorWrapper(nullptr, input_shape, out_dtype);
    auto rsigma_tensor = TensorWrapper(nullptr, intermediates_shape, DType::kFloat32);

    // dummy tensor wrappers that will carry workspace size info later
    TensorWrapper dummy_work_tensor, dummy_barrier_tensor;
    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();
    auto layernorm_fwd_func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    if (is_layer_norm) {
        auto beta_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);
        auto mu_tensor = TensorWrapper(nullptr, intermediates_shape, DType::kFloat32);

        layernorm_fwd_func(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), eps,
                           output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(), nullptr,
                           num_sm, dummy_work_tensor.data(), dummy_barrier_tensor.data());
    } else {
        NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
        nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), eps, output_tensor.data(),
                         rsigma_tensor.data(), nullptr, num_sm, dummy_work_tensor.data(),
                         dummy_barrier_tensor.data());
    }

    auto work_shape = MakeShapeVector(dummy_work_tensor.shape());
    auto barrier_shape = MakeShapeVector(dummy_barrier_tensor.shape());
    return pybind11::make_tuple(std::make_pair(work_shape, dummy_work_tensor.dtype()),
                                std::make_pair(barrier_shape, dummy_barrier_tensor.dtype()));
}

void LayerNormForwardImpl(size_t batch_size, size_t hidden_size, size_t workspace_size,
                          size_t barrier_size, bool zero_centered_gamma, float eps, void *input,
                          DType in_dtype, void *weight, DType w_dtype, void *bias, void *output,
                          DType out_dtype, void *workspace, DType work_dtype, void *barrier,
                          DType barrier_dtype, void *mu, void *rsigma, float *amax, float *scale,
                          float *scale_inv, cudaStream_t stream) {
    auto input_shape = std::vector<size_t>{batch_size, hidden_size};
    auto weight_shape = std::vector<size_t>{hidden_size};
    auto intermediates_shape = std::vector<size_t>{batch_size};
    auto workspace_shape = std::vector<size_t>{workspace_size};
    auto barrier_shape = std::vector<size_t>{barrier_size};
    auto is_layer_norm = (bias) ? true : false;

    auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
    auto gamma_tensor = TensorWrapper(weight, weight_shape, in_dtype);

    // assume output dtype = input dtype
    // If we need mixed I/O precision in the future, we need an additional
    // parameter for output type
    auto output_tensor = TensorWrapper(output, input_shape, out_dtype, amax, scale, scale_inv);
    auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, DType::kFloat32);

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();
    auto layernorm_fwd_func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;

    auto workspace_tensor = TensorWrapper(workspace, workspace_shape, work_dtype);
    auto barrier_tensor = TensorWrapper(barrier, barrier_shape, barrier_dtype);

    if (is_layer_norm) {
        auto beta_tensor = TensorWrapper(bias, weight_shape, w_dtype);
        auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

        layernorm_fwd_func(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), eps,
                           output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(), stream,
                           num_sm, workspace_tensor.data(), barrier_tensor.data());
    } else {
        NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
        nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), eps, output_tensor.data(),
                         rsigma_tensor.data(), stream, num_sm, workspace_tensor.data(),
                         barrier_tensor.data());
    }
}

pybind11::tuple GetLayerNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType w_dtype,
                                                   bool is_layer_norm, bool zero_centered_gamma,
                                                   float eps) {
    auto input_shape = std::vector<size_t>{batch_size, hidden_size};
    auto weight_shape = std::vector<size_t>{hidden_size};
    auto intermediates_shape = std::vector<size_t>{batch_size};
    auto intermediates_dtype = DType::kFloat32;

    // empty tensor wrappers are okay just to get workspace size
    auto dz_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
    auto rsigma_tensor = TensorWrapper(nullptr, intermediates_shape, intermediates_dtype);
    auto x_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
    auto gamma_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);
    auto xgrad_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
    auto wgrad_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);

    // dummy tensor wrappers that will carry workspace size info later
    TensorWrapper dummy_work_tensor, dummy_barrier_tensor;
    TensorWrapper dummy_dgamma_part_tensor, dummy_dbeta_part_tensor;
    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();
    auto layernorm_bwd_func = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;

    // initialize dBeta information here -- layernorm will modify but RMSnorm will not
    std::vector<size_t> dbeta_part_shape;
    if (is_layer_norm) {
        auto mu_tensor = TensorWrapper(nullptr, intermediates_shape, intermediates_dtype);
        auto dbeta_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);

        layernorm_bwd_func(dz_tensor.data(), x_tensor.data(), mu_tensor.data(),
                           rsigma_tensor.data(), gamma_tensor.data(), xgrad_tensor.data(),
                           wgrad_tensor.data(), dbeta_tensor.data(),
                           dummy_dgamma_part_tensor.data(), dummy_dbeta_part_tensor.data(), nullptr,
                           num_sm, dummy_work_tensor.data(), dummy_barrier_tensor.data());

        dbeta_part_shape = MakeShapeVector(dummy_dbeta_part_tensor.shape());
    } else {
        NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
        nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(),
                         gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                         dummy_dgamma_part_tensor.data(), nullptr, num_sm, dummy_work_tensor.data(),
                         dummy_barrier_tensor.data());

        dbeta_part_shape = std::vector<size_t>{0, 0};
    }

    auto work_shape = MakeShapeVector(dummy_work_tensor.shape());
    auto barrier_shape = MakeShapeVector(dummy_barrier_tensor.shape());
    auto dgamma_part_shape = MakeShapeVector(dummy_dgamma_part_tensor.shape());
    return pybind11::make_tuple(std::make_pair(work_shape, dummy_work_tensor.dtype()),
                                std::make_pair(barrier_shape, dummy_barrier_tensor.dtype()),
                                std::make_pair(dgamma_part_shape, dummy_dgamma_part_tensor.dtype()),
                                std::make_pair(dbeta_part_shape, dummy_dbeta_part_tensor.dtype()));
}

void LayerNormBackwardImpl(size_t batch_size, size_t hidden_size, size_t wkspace_size,
                           size_t barrier_size, size_t *dgamma_part_sizes, size_t *dbeta_part_sizes,
                           bool zero_centered_gamma, float eps, void *input, DType in_dtype,
                           void *weight, DType w_dtype, void *ograd, void *workspace,
                           DType wkspace_dtype, void *barrier, DType barrier_dtype, void *mu,
                           void *rsigma, void *xgrad, void *wgrad, void *dbeta, void *dgamma_part,
                           DType dgamma_dtype, void *dbeta_part, DType dbeta_dtype,
                           cudaStream_t stream) {
    auto input_shape = std::vector<size_t>{batch_size, hidden_size};
    auto weight_shape = std::vector<size_t>{hidden_size};
    auto intermediates_shape = std::vector<size_t>{batch_size};
    auto intermediates_dtype = DType::kFloat32;
    auto is_layer_norm = (dbeta) ? true : false;

    // assume input type = output type
    auto *grad_output = ograd;
    auto x_dtype = in_dtype;
    auto dz_tensor = TensorWrapper(grad_output, input_shape, x_dtype);

    auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, intermediates_dtype);

    auto *x = input;
    auto x_tensor = TensorWrapper(x, input_shape, x_dtype);

    auto gamma_tensor = TensorWrapper(weight, weight_shape, w_dtype);
    auto xgrad_tensor = TensorWrapper(xgrad, input_shape, x_dtype);
    auto wgrad_tensor = TensorWrapper(wgrad, weight_shape, w_dtype);

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();
    auto layernorm_bwd_func = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;

    auto workspace_shape = std::vector<size_t>{wkspace_size};
    auto workspace_tensor = TensorWrapper(workspace, workspace_shape, wkspace_dtype);
    auto barrier_shape = std::vector<size_t>{barrier_size};
    auto barrier_tensor = TensorWrapper(barrier, barrier_shape, barrier_dtype);
    auto dgamma_part_shape = std::vector<size_t>{dgamma_part_sizes[0], dgamma_part_sizes[1]};
    auto dgamma_part_tensor = TensorWrapper(dgamma_part, dgamma_part_shape, dgamma_dtype);

    if (is_layer_norm) {
        auto mu_tensor = TensorWrapper(mu, intermediates_shape, intermediates_dtype);
        auto dbeta_tensor = TensorWrapper(dbeta, weight_shape, w_dtype);
        auto dbeta_part_shape = std::vector<size_t>{dbeta_part_sizes[0], dbeta_part_sizes[1]};
        auto dbeta_part_tensor = TensorWrapper(dbeta_part, dbeta_part_shape, dbeta_dtype);

        layernorm_bwd_func(dz_tensor.data(), x_tensor.data(), mu_tensor.data(),
                           rsigma_tensor.data(), gamma_tensor.data(), xgrad_tensor.data(),
                           wgrad_tensor.data(), dbeta_tensor.data(), dgamma_part_tensor.data(),
                           dbeta_part_tensor.data(), stream, num_sm, workspace_tensor.data(),
                           barrier_tensor.data());
    } else {
        NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
        nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(),
                         gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                         dgamma_part_tensor.data(), stream, num_sm, workspace_tensor.data(),
                         barrier_tensor.data());
    }
}

void LayerNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque,
                         size_t opaque_len) {
    auto *input = buffers[0];
    auto *weight = buffers[1];
    auto *bias = buffers[2];
    auto *amax = reinterpret_cast<float *>(buffers[3]);
    auto *scale = reinterpret_cast<float *>(buffers[4]);
    auto *scale_inv = reinterpret_cast<float *>(buffers[5]);
    auto *output = buffers[6];
    auto *mu = buffers[7];
    auto *rsigma = buffers[8];
    auto *amax_out = buffers[9];
    auto *workspace = buffers[10];
    auto *barrier = buffers[11];
    assert(amax_out == amax);

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto batch_size = desc.batch_size;
    auto hidden_size = desc.hidden_size;
    auto wkspace_size = desc.wkspace_size;
    auto barrier_size = desc.barrier_size;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto wkspace_dtype = desc.wkspace_dtype;
    auto barrier_dtype = desc.barrier_dtype;
    auto eps = desc.eps;
    auto zero_centered_gamma = desc.zero_centered_gamma;
    auto sm_margin = desc.sm_margin;

    auto out_dtype = DType::kFloat8E4M3;

    LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                         eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                         wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                         stream);
}

void LayerNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *weight = buffers[1];
    auto *bias = buffers[2];
    auto *output = buffers[3];
    auto *mu = buffers[4];
    auto *rsigma = buffers[5];
    auto *workspace = buffers[6];
    auto *barrier = buffers[7];

    float *amax = nullptr;
    float *scale = nullptr;
    float *scale_inv = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto batch_size = desc.batch_size;
    auto hidden_size = desc.hidden_size;
    auto wkspace_size = desc.wkspace_size;
    auto barrier_size = desc.barrier_size;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto wkspace_dtype = desc.wkspace_dtype;
    auto barrier_dtype = desc.barrier_dtype;
    auto eps = desc.eps;
    auto out_dtype = in_dtype;
    auto zero_centered_gamma = desc.zero_centered_gamma;
    auto sm_margin = desc.sm_margin;

    LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                         eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                         wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                         stream);
}

void LayerNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);

    auto batch_size = desc.batch_size;
    auto hidden_size = desc.hidden_size;
    auto wkspace_size = desc.wkspace_size;
    auto barrier_size = desc.barrier_size;
    auto *dgamma_part_sizes = desc.dgamma_part_sizes;
    auto *dbeta_part_sizes = desc.dbeta_part_sizes;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto wkspace_dtype = desc.wkspace_dtype;
    auto barrier_dtype = desc.barrier_dtype;
    auto dgamma_part_dtype = desc.dgamma_part_dtype;
    auto dbeta_part_dtype = desc.dbeta_part_dtype;
    auto eps = desc.eps;
    auto zero_centered_gamma = desc.zero_centered_gamma;
    auto sm_margin = desc.sm_margin;

    auto *ograd = buffers[0];
    auto *mu = buffers[1];
    auto *rsigma = buffers[2];
    auto *input = buffers[3];
    auto *weight = buffers[4];
    auto *xgrad = buffers[5];
    auto *wgrad = buffers[6];
    auto *dbeta = buffers[7];
    auto *workspace = buffers[8];
    auto *barrier = buffers[9];
    auto *dgamma_part = buffers[10];
    auto *dbeta_part = buffers[11];

    LayerNormBackwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, dgamma_part_sizes,
                          dbeta_part_sizes, zero_centered_gamma, eps, input, in_dtype, weight,
                          w_dtype, ograd, workspace, wkspace_dtype, barrier, barrier_dtype, mu,
                          rsigma, xgrad, wgrad, dbeta, dgamma_part, dgamma_part_dtype, dbeta_part,
                          dbeta_part_dtype, stream);
}

void RMSNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *weight = buffers[1];
    auto *amax = reinterpret_cast<float *>(buffers[2]);
    auto *scale = reinterpret_cast<float *>(buffers[3]);
    auto *scale_inv = reinterpret_cast<float *>(buffers[4]);
    auto *output = buffers[5];
    auto *rsigma = buffers[6];
    auto *amax_out = buffers[7];
    auto *workspace = buffers[8];
    auto *barrier = buffers[9];
    assert(amax_out == amax);

    void *bias = nullptr;
    void *mu = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto batch_size = desc.batch_size;
    auto hidden_size = desc.hidden_size;
    auto wkspace_size = desc.wkspace_size;
    auto barrier_size = desc.barrier_size;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto wkspace_dtype = desc.wkspace_dtype;
    auto barrier_dtype = desc.barrier_dtype;
    auto eps = desc.eps;
    auto zero_centered_gamma = desc.zero_centered_gamma;
    auto sm_margin = desc.sm_margin;
    auto out_dtype = DType::kFloat8E4M3;

    LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                         eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                         wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                         stream);
}

void RMSNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *weight = buffers[1];
    auto *output = buffers[2];
    auto *rsigma = buffers[3];
    auto *workspace = buffers[4];
    auto *barrier = buffers[5];

    void *bias = nullptr;
    void *mu = nullptr;
    float *amax = nullptr;
    float *scale = nullptr;
    float *scale_inv = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto batch_size = desc.batch_size;
    auto hidden_size = desc.hidden_size;
    auto wkspace_size = desc.wkspace_size;
    auto barrier_size = desc.barrier_size;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto wkspace_dtype = desc.wkspace_dtype;
    auto barrier_dtype = desc.barrier_dtype;
    auto eps = desc.eps;
    auto zero_centered_gamma = desc.zero_centered_gamma;
    auto sm_margin = desc.sm_margin;
    auto out_dtype = in_dtype;

    LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                         eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                         wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                         stream);
}

void RMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *ograd = buffers[0];
    auto *rsigma = buffers[1];
    auto *input = buffers[2];
    auto *weight = buffers[3];
    auto *xgrad = buffers[4];
    auto *wgrad = buffers[5];
    auto *workspace = buffers[6];
    auto *barrier = buffers[7];
    auto *dgamma_part = buffers[8];

    void *mu = nullptr;
    void *dbeta = nullptr;
    void *dbeta_part = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto batch_size = desc.batch_size;
    auto hidden_size = desc.hidden_size;
    auto wkspace_size = desc.wkspace_size;
    auto barrier_size = desc.barrier_size;
    auto dgamma_part_sizes = desc.dgamma_part_sizes;
    size_t dbeta_part_sizes[2] = {0, 0};
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto wkspace_dtype = desc.wkspace_dtype;
    auto barrier_dtype = desc.barrier_dtype;
    auto dgamma_part_dtype = desc.dgamma_part_dtype;
    auto dbeta_part_dtype = DType::kByte;
    auto eps = desc.eps;
    auto zero_centered_gamma = desc.zero_centered_gamma;

    LayerNormBackwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, dgamma_part_sizes,
                          dbeta_part_sizes, zero_centered_gamma, eps, input, in_dtype, weight,
                          w_dtype, ograd, workspace, wkspace_dtype, barrier, barrier_dtype, mu,
                          rsigma, xgrad, wgrad, dbeta, dgamma_part, dgamma_part_dtype, dbeta_part,
                          dbeta_part_dtype, stream);
}

void Quantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *amax = reinterpret_cast<float *>(buffers[1]);
    auto *scale = reinterpret_cast<float *>(buffers[2]);
    auto *scale_inv = reinterpret_cast<float *>(buffers[3]);
    auto *output = buffers[4];
    auto *amax_out = reinterpret_cast<float *>(buffers[5]);
    assert(amax == amax_out);

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);
    auto shape = desc.shape.to_vector();
    auto input_tensor = TensorWrapper(input, shape, desc.in_dtype);
    auto output_tensor = TensorWrapper(output, shape, desc.out_dtype, amax_out, scale, scale_inv);

    nvte_fp8_quantize(input_tensor.data(), output_tensor.data(), stream);
}

void Dequantize(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *amax = reinterpret_cast<float *>(buffers[1]);
    auto *scale = reinterpret_cast<float *>(buffers[2]);
    auto *scale_inv = reinterpret_cast<float *>(buffers[3]);
    auto *output = buffers[4];

    const auto &desc = *UnpackOpaque<CustomCallCommonDescriptor>(opaque, opaque_len);

    auto shape = desc.shape.to_vector();
    auto input_tensor = TensorWrapper(input, shape, desc.in_dtype, amax, scale, scale_inv);

    auto output_tensor = TensorWrapper(output, shape, desc.out_dtype);

    nvte_fp8_dequantize(input_tensor.data(), output_tensor.data(), stream);
}

void ScaledSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                          size_t opaque_len) {
    auto *input = buffers[0];
    auto *output = buffers[1];

    const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
    auto shape = std::vector<size_t>{desc.batch_size, desc.head_dim, desc.q_seqlen, desc.k_seqlen};
    auto dtype = desc.dtype;

    auto input_tensor = TensorWrapper(input, shape, dtype);
    auto output_tensor = TensorWrapper(output, shape, dtype);

    nvte_scaled_softmax_forward(input_tensor.data(), output_tensor.data(), desc.scale_factor,
                                stream);
}

void ScaledSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                           size_t opaque_len) {
    auto *grad_output = buffers[0];
    auto *softmax_output = buffers[1];
    auto *dgrad = buffers[2];

    const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
    auto shape = std::vector<size_t>{desc.batch_size, desc.head_dim, desc.q_seqlen, desc.k_seqlen};
    auto dtype = desc.dtype;

    auto grad_output_tensor = TensorWrapper(grad_output, shape, dtype);
    auto softmax_output_tensor = TensorWrapper(softmax_output, shape, dtype);
    auto dgrad_tensor = TensorWrapper(dgrad, shape, dtype);

    nvte_scaled_softmax_backward(grad_output_tensor.data(), softmax_output_tensor.data(),
                                 dgrad_tensor.data(), desc.scale_factor, stream);
}

void ScaledMaskedSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                                size_t opaque_len) {
    auto *input = buffers[0];
    auto *mask = buffers[1];
    auto *output = buffers[2];

    const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
    auto io_shape =
        std::vector<size_t>{desc.batch_size, desc.head_dim, desc.q_seqlen, desc.k_seqlen};
    auto mask_shape = std::vector<size_t>{desc.padding_size, 1, desc.q_seqlen, desc.k_seqlen};
    auto dtype = desc.dtype;

    auto input_tensor = TensorWrapper(input, io_shape, dtype);
    // Mask would be casted to uint8_t
    auto mask_tensor = TensorWrapper(mask, mask_shape, DType::kByte);
    auto output_tensor = TensorWrapper(output, io_shape, dtype);

    nvte_scaled_masked_softmax_forward(input_tensor.data(), mask_tensor.data(),
                                       output_tensor.data(), desc.scale_factor, stream);
}

void ScaledMaskedSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                                 size_t opaque_len) {
    // The backward of ScaledMaskedSoftmax is equivalent to ScaledSoftmax.
    ScaledSoftmaxBackward(stream, buffers, opaque, opaque_len);
}

void ScaledUpperTriangMaskedSoftmaxForward(cudaStream_t stream, void **buffers, const char *opaque,
                                           size_t opaque_len) {
    auto *input = buffers[0];
    auto *output = buffers[1];

    const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
    auto attn_batch = desc.batch_size * desc.head_dim;
    auto shape = std::vector<size_t>{attn_batch, desc.q_seqlen, desc.k_seqlen};
    auto dtype = desc.dtype;

    auto input_tensor = TensorWrapper(input, shape, dtype);

    auto output_tensor = TensorWrapper(output, shape, dtype);

    nvte_scaled_upper_triang_masked_softmax_forward(input_tensor.data(), output_tensor.data(),
                                                    desc.scale_factor, stream);
}

void ScaledUpperTriangMaskedSoftmaxBackward(cudaStream_t stream, void **buffers, const char *opaque,
                                            size_t opaque_len) {
    auto *grad_output = buffers[0];
    auto *softmax_output = buffers[1];
    auto *dgrad = buffers[2];

    const auto &desc = *UnpackOpaque<SoftmaxDescriptor>(opaque, opaque_len);
    auto attn_batch = desc.batch_size * desc.head_dim;
    auto shape = std::vector<size_t>{attn_batch, desc.q_seqlen, desc.k_seqlen};
    auto dtype = desc.dtype;

    auto grad_output_tensor = TensorWrapper(grad_output, shape, dtype);
    auto softmax_output_tensor = TensorWrapper(softmax_output, shape, dtype);
    auto dgrad_tensor = TensorWrapper(dgrad, shape, dtype);

    nvte_scaled_upper_triang_masked_softmax_backward(
        grad_output_tensor.data(), softmax_output_tensor.data(), dgrad_tensor.data(),
        desc.scale_factor, stream);
}

NVTE_Fused_Attn_Backend GetFusedAttnBackend(DType q_dtype, DType kv_dtype,
                                            NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                            NVTE_Mask_Type mask_type, float dropout_probability,
                                            size_t q_attn_heads, size_t kv_attn_heads,
                                            size_t q_max_seqlen, size_t kv_max_seqlen,
                                            size_t head_dim) {
    auto backend = nvte_get_fused_attn_backend(
        static_cast<NVTEDType>(q_dtype), static_cast<NVTEDType>(kv_dtype), qkv_layout, bias_type,
        mask_type, dropout_probability, q_attn_heads, kv_attn_heads, q_max_seqlen, kv_max_seqlen,
        head_dim);
    return backend;
}

/*
    NOTE: PrepareFusedAttnForwardAuxTensors unifies the auxiliary tensor pack logic from the fused
    attention forward kernels in:
        - common/fused_attn/fused_attn_f16_max512_seqlen.cu lines 594-634 and 773-812
        - common/fused_attn/fused_attn_f16_arbitrary_seqlen.cu lines 1270-1281 and 1348-1359
*/
void PrepareFusedAttnForwardAuxTensors(NVTETensorPack *tensor_pack,
                                       const CustomCallFusedAttnDescriptor *desc,
                                       NVTE_Bias_Type bias_type, NVTE_Fused_Attn_Backend backend,
                                       void *softmax_buf, void *rng_state_buf = nullptr,
                                       void *bias_buf = nullptr) {
    auto input_batch = desc->input_batch;
    auto bias_batch = desc->bias_batch;
    auto attn_heads = desc->attn_heads;
    auto bias_heads = desc->bias_heads;
    auto q_max_seqlen = desc->q_max_seqlen;
    auto kv_max_seqlen = desc->kv_max_seqlen;

    // all backends need softmax but expect different shapes/dtypes
    // start with the max512 sequence length softmax shape/dtype and correct later
    tensor_pack->size = 1;
    Tensor *softmax_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[0]);
    softmax_aux->data.dptr = softmax_buf;
    softmax_aux->data.shape =
        std::vector<size_t>{input_batch, attn_heads, q_max_seqlen, kv_max_seqlen};
    softmax_aux->data.dtype = desc->dtype;

    // arbitrary sequence length backend needs the RNG state and a different shape/dtype softmax
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen) {
        tensor_pack->size = 2;
        Tensor *rng_state_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[1]);
        rng_state_aux->data.dptr = rng_state_buf;
        rng_state_aux->data.shape = std::vector<size_t>{2};
        rng_state_aux->data.dtype = DType::kInt64;
        // correct softmax shape/dtype
        softmax_aux->data.shape.at(3) = 1;  // {B,H,Qs,Ks} -> {B,H,Qs,1}
        softmax_aux->data.dtype = DType::kFloat32;

        // include bias if enabled
        if (bias_type != NVTE_Bias_Type::NVTE_NO_BIAS && bias_type != NVTE_Bias_Type::NVTE_ALIBI) {
            tensor_pack->size = 3;
            Tensor *bias_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[2]);
            bias_aux->data.dptr = bias_buf;
            bias_aux->data.shape =
                std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
            bias_aux->data.dtype = desc->dtype;
        }
    }
}

/*
    NOTE: Backward fused attention kernels accept auxiliary tensors as explicit function arguments
    instead of an NVTETensorPack and nvte_fused_attn_bwd() API does all the logic for pulling the
    necessary tensors out of the tensor pack for the active kernel. That means we can just dump
    everything we got into the tensor pack and not worry about its sizing for the backward pass.

    TODO(Alp): Refactor the nvte_fused_attn_fwd() to work like nvte_fused_attn_bwd()?
*/
void PrepareFusedAttnBackwardAuxTensors(NVTETensorPack *tensor_pack,
                                        const CustomCallFusedAttnDescriptor *desc,
                                        NVTE_Fused_Attn_Backend backend, void *softmax_buf,
                                        void *rng_state_buf, void *bias_buf) {
    // Backward calls put everything into the tensor pack for every backend
    // so we set dummy bias_type and backend choices here to follow the correct code path
    auto dummy_bias_type = NVTE_Bias_Type::NVTE_POST_SCALE_BIAS;
    auto dummy_backend = NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen;
    PrepareFusedAttnForwardAuxTensors(tensor_pack, desc, dummy_bias_type, dummy_backend,
                                      softmax_buf, rng_state_buf, bias_buf);

    // correct softmax shape for max512 sequence length kernel
    if (backend == NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen) {
        Tensor *softmax_aux = reinterpret_cast<Tensor *>(tensor_pack->tensors[0]);
        softmax_aux->data.shape.at(3) = desc->kv_max_seqlen;  // {B,H,Qs,1} -> {B,H,Qs,Ks}
        softmax_aux->data.dtype = desc->dtype;
    }
}

pybind11::tuple GetFusedAttnForwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training) {
    // For qkv_packed
    auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
    auto qkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);

    // For kv_packed
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto kv_shape = std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
    auto kv_tensor = TensorWrapper(nullptr, kv_shape, dtype);

    // For separate q, k, v
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
    auto v_shape = k_shape;
    auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);

    auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
    auto bias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

    // F16 doesn't use this tensor
    auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);
    auto o_tensor = TensorWrapper(nullptr, q_shape, dtype);

    auto q_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);

    auto dummy_rng_state_tensor = TensorWrapper(nullptr, std::vector<size_t>{2}, DType::kInt64);

    NVTETensorPack aux_output_tensors;
    nvte_tensor_pack_create(&aux_output_tensors);

    TensorWrapper query_workspace_tensor;
    if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
        assert(q_max_seqlen == kv_max_seqlen);
        nvte_fused_attn_fwd_qkvpacked(
            qkv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
            &aux_output_tensors, q_cu_seqlens_tensor.data(), dummy_rng_state_tensor.data(),
            q_max_seqlen, is_training, scaling_factor, dropout_probability, qkv_layout, bias_type,
            mask_type, query_workspace_tensor.data(), nullptr);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
        nvte_fused_attn_fwd_kvpacked(q_tensor.data(), kv_tensor.data(), bias_tensor.data(),
                                     s_tensor.data(), o_tensor.data(), &aux_output_tensors,
                                     q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                                     dummy_rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen,
                                     is_training, scaling_factor, dropout_probability, qkv_layout,
                                     bias_type, mask_type, query_workspace_tensor.data(), nullptr);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
        nvte_fused_attn_fwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), bias_tensor.data(),
                            s_tensor.data(), o_tensor.data(), &aux_output_tensors,
                            q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                            dummy_rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, is_training,
                            scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                            query_workspace_tensor.data(), nullptr);
    } else {
        NVTE_ERROR("Unsupported QKVLayout.");
    }

    auto workspace_shape = MakeShapeVector(query_workspace_tensor.shape());
    return pybind11::make_tuple(workspace_shape, query_workspace_tensor.dtype());
}

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t batch_size, size_t q_max_seqlen, size_t kv_max_seqlen, size_t attn_heads,
    size_t num_gqa_groups, size_t head_dim, float scaling_factor, float dropout_probability,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype,
    bool is_training) {
    auto output_shape = std::vector<size_t>{batch_size * q_max_seqlen, attn_heads, head_dim};
    auto output_tensor = TensorWrapper(nullptr, output_shape, dtype);
    auto doutput_tensor = TensorWrapper(nullptr, output_shape, dtype);

    auto bias_shape = std::vector<size_t>{1, attn_heads, q_max_seqlen, kv_max_seqlen};
    auto dbias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

    // F16 doesn't use s_tensor
    auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);

    auto q_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{batch_size + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{batch_size + 1}, DType::kInt32);

    NVTETensorPack aux_input_tensors;
    nvte_tensor_pack_create(&aux_input_tensors);

    TensorWrapper query_workspace_tensor;

    if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
        assert(q_max_seqlen == kv_max_seqlen);
        auto qkv_shape = std::vector<size_t>{batch_size * q_max_seqlen, 3, attn_heads, head_dim};
        auto qkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);
        auto dqkv_tensor = TensorWrapper(nullptr, qkv_shape, dtype);
        nvte_fused_attn_bwd_qkvpacked(
            qkv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
            s_tensor.data(),  // not used for F16
            s_tensor.data(),  // not used for F16
            &aux_input_tensors, dqkv_tensor.data(), dbias_tensor.data(), q_cu_seqlens_tensor.data(),
            q_max_seqlen, scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
            query_workspace_tensor.data(), nullptr);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
        auto q_shape = std::vector<size_t>{batch_size * q_max_seqlen, attn_heads, head_dim};
        auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
        auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
        auto kv_shape =
            std::vector<size_t>{batch_size * kv_max_seqlen, 2, num_gqa_groups, head_dim};
        auto kv_tensor = TensorWrapper(nullptr, kv_shape, dtype);
        auto dkv_tensor = TensorWrapper(nullptr, kv_shape, dtype);
        nvte_fused_attn_bwd_kvpacked(
            q_tensor.data(), kv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
            s_tensor.data(),  // not used for F16
            s_tensor.data(),  // not used for F16
            &aux_input_tensors, dq_tensor.data(), dkv_tensor.data(), dbias_tensor.data(),
            q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(), q_max_seqlen, kv_max_seqlen,
            scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
            query_workspace_tensor.data(), nullptr);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
        auto q_shape = std::vector<size_t>{batch_size * q_max_seqlen, attn_heads, head_dim};
        auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
        auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
        auto k_shape = std::vector<size_t>{batch_size * kv_max_seqlen, num_gqa_groups, head_dim};
        auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
        auto dk_tensor = TensorWrapper(nullptr, k_shape, dtype);
        auto v_shape = k_shape;
        auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);
        auto dv_tensor = TensorWrapper(nullptr, v_shape, dtype);
        nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                            doutput_tensor.data(),
                            s_tensor.data(),  // not used for F16
                            s_tensor.data(),  // not used for F16
                            &aux_input_tensors, dq_tensor.data(), dk_tensor.data(),
                            dv_tensor.data(), dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                            kv_cu_seqlens_tensor.data(), q_max_seqlen, kv_max_seqlen,
                            scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                            query_workspace_tensor.data(), nullptr);
    } else {
        NVTE_ERROR("Unsupported QKVLayout.");
    }

    auto workspace_shape = MakeShapeVector(query_workspace_tensor.shape());
    return pybind11::make_tuple(workspace_shape, query_workspace_tensor.dtype());
}

void FusedAttnForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const CustomCallFusedAttnDescriptor &descriptor =
        *UnpackOpaque<CustomCallFusedAttnDescriptor>(opaque, opaque_len);

    /* Input buffers from XLA */
    /* Buffers[0-2] are q, k, v, which are parsed later for different qkv_layout */
    void *bias = buffers[3];
    void *q_cu_seqlens = buffers[4];
    void *kv_cu_seqlens = buffers[5];
    void *seed = buffers[6];

    /* Output buffer from XLA */
    void *output = buffers[7];
    void *softmax_aux = buffers[8];
    void *rng_state = buffers[9];
    void *workspace = buffers[10];

    /* Descriptor */
    auto input_batch = descriptor.input_batch;
    auto bias_batch = descriptor.bias_batch;
    auto q_max_seqlen = descriptor.q_max_seqlen;
    auto kv_max_seqlen = descriptor.kv_max_seqlen;
    auto attn_heads = descriptor.attn_heads;
    auto num_gqa_groups = descriptor.num_gqa_groups;
    auto bias_heads = descriptor.bias_heads;
    auto head_dim = descriptor.head_dim;
    auto scaling_factor = descriptor.scaling_factor;
    auto dropout_probability = descriptor.dropout_probability;
    auto bias_type = descriptor.bias_type;
    auto mask_type = descriptor.mask_type;
    auto qkv_layout = descriptor.qkv_layout;
    auto dtype = descriptor.dtype;

    /* Input tensors */
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto v_shape = k_shape;
    auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
    auto bias_tensor = TensorWrapper(bias, bias_shape, dtype);

    /* Output tensors */
    auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);  // not used in F16
    auto o_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto o_tensor = TensorWrapper(output, o_shape, dtype);
    auto q_cu_seqlens_tensor =
        TensorWrapper(q_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(kv_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);

    /* Prepare RNG state */
    auto rng_state_tensor = TensorWrapper(rng_state, std::vector<size_t>{2}, DType::kInt64);
    auto backend = nvte_get_fused_attn_backend(
        static_cast<NVTEDType>(dtype), static_cast<NVTEDType>(dtype), qkv_layout, bias_type,
        mask_type, dropout_probability, attn_heads, num_gqa_groups, q_max_seqlen, kv_max_seqlen,
        head_dim);
    PopulateRngStateAsync(rng_state, seed, q_max_seqlen, kv_max_seqlen, backend, stream);

    /* Auxiliary tensors (to be propagated to the backward pass later) */
    NVTETensorPack aux_output_tensors;
    nvte_tensor_pack_create(&aux_output_tensors);
    PrepareFusedAttnForwardAuxTensors(&aux_output_tensors, &descriptor, bias_type, backend,
                                      softmax_aux);

    /* cuDNN workspace */
    auto workspace_tensor = TensorWrapper(workspace, std::vector<size_t>{descriptor.wkspace_size},
                                          descriptor.wkspace_dtype);

    /* Call the underly NVTE API */
    if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
        auto qkv = buffers[0];
        auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
        auto qkv_tensor = TensorWrapper(qkv, qkv_shape, dtype);
        nvte_fused_attn_fwd_qkvpacked(
            qkv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
            &aux_output_tensors, q_cu_seqlens_tensor.data(), rng_state_tensor.data(), q_max_seqlen,
            descriptor.is_training, descriptor.scaling_factor, dropout_probability, qkv_layout,
            bias_type, mask_type, workspace_tensor.data(), stream);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
        auto q = buffers[0];
        auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
        auto q_tensor = TensorWrapper(q, q_shape, dtype);
        auto kv = buffers[1];
        auto kv_shape =
            std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
        auto kv_tensor = TensorWrapper(kv, kv_shape, dtype);
        nvte_fused_attn_fwd_kvpacked(
            q_tensor.data(), kv_tensor.data(), bias_tensor.data(), s_tensor.data(), o_tensor.data(),
            &aux_output_tensors, q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
            rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen, descriptor.is_training,
            scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
            workspace_tensor.data(), stream);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
        auto q = buffers[0];
        auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
        auto q_tensor = TensorWrapper(q, q_shape, dtype);
        auto k = buffers[1];
        auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
        auto k_tensor = TensorWrapper(k, k_shape, dtype);
        auto v = buffers[2];
        auto v_shape = k_shape;
        auto v_tensor = TensorWrapper(v, v_shape, dtype);
        nvte_fused_attn_fwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), bias_tensor.data(),
                            s_tensor.data(), o_tensor.data(), &aux_output_tensors,
                            q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(),
                            rng_state_tensor.data(), q_max_seqlen, kv_max_seqlen,
                            descriptor.is_training, scaling_factor, dropout_probability, qkv_layout,
                            bias_type, mask_type, workspace_tensor.data(), stream);
    } else {
        NVTE_ERROR("Unsupported qkv_layout.");
    }

    nvte_tensor_pack_destroy(&aux_output_tensors);
}

pybind11::tuple GetFusedAttnBackwardWorkspaceSizes(
    size_t input_batch, size_t bias_batch, size_t q_max_seqlen, size_t kv_max_seqlen,
    size_t attn_heads, size_t num_gqa_groups, size_t bias_heads, size_t head_dim,
    float scaling_factor, float dropout_probability, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, NVTE_QKV_Layout qkv_layout, DType dtype, bool is_training) {
    auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
    auto v_shape = k_shape;
    auto output_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};

    auto q_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto k_tensor = TensorWrapper(nullptr, k_shape, dtype);
    auto v_tensor = TensorWrapper(nullptr, v_shape, dtype);
    auto doutput_tensor = TensorWrapper(nullptr, output_shape, dtype);
    auto output_tensor = TensorWrapper(nullptr, output_shape, dtype);
    // F16 doesn't use this tensor
    auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);

    auto dq_tensor = TensorWrapper(nullptr, q_shape, dtype);
    auto dk_tensor = TensorWrapper(nullptr, k_shape, dtype);
    auto dv_tensor = TensorWrapper(nullptr, v_shape, dtype);
    auto dbias_tensor = TensorWrapper(nullptr, bias_shape, dtype);

    auto q_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(nullptr, std::vector<size_t>{input_batch + 1}, DType::kInt32);

    NVTETensorPack aux_input_tensors;
    nvte_tensor_pack_create(&aux_input_tensors);

    TensorWrapper query_workspace_tensor;
    nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                        doutput_tensor.data(),
                        s_tensor.data(),  // not used for F16
                        s_tensor.data(),  // not used for F16
                        &aux_input_tensors, dq_tensor.data(), dk_tensor.data(), dv_tensor.data(),
                        dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                        kv_cu_seqlens_tensor.data(), q_max_seqlen, kv_max_seqlen, scaling_factor,
                        dropout_probability, qkv_layout, bias_type, mask_type,
                        query_workspace_tensor.data(), nullptr);

    auto work_shape = MakeShapeVector(query_workspace_tensor.shape());
    return pybind11::make_tuple(work_shape, query_workspace_tensor.dtype());
}

void FusedAttnBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const CustomCallFusedAttnDescriptor &descriptor =
        *UnpackOpaque<CustomCallFusedAttnDescriptor>(opaque, opaque_len);

    /* Input buffers from XLA */
    /* Buffers[0-2] are q, k, v, which are parsed later for different qkv_layout */
    void *bias = buffers[3];
    void *softmax_aux = buffers[4];
    void *rng_state = buffers[5];
    void *output = buffers[6];
    void *doutput = buffers[7];
    void *q_cu_seqlens = buffers[8];
    void *kv_cu_seqlens = buffers[9];

    /* Output buffer from XLA */
    /* Buffers[10-12] are dq, dk, dv, which are parsed later for different qkv_layout */
    void *dbias = buffers[13];
    void *workspace = buffers[14];

    /* Descriptor */
    auto input_batch = descriptor.input_batch;
    auto bias_batch = descriptor.bias_batch;
    auto q_max_seqlen = descriptor.q_max_seqlen;
    auto kv_max_seqlen = descriptor.kv_max_seqlen;
    auto attn_heads = descriptor.attn_heads;
    auto num_gqa_groups = descriptor.num_gqa_groups;
    auto bias_heads = descriptor.bias_heads;
    auto head_dim = descriptor.head_dim;
    auto scaling_factor = descriptor.scaling_factor;
    auto dropout_probability = descriptor.dropout_probability;
    auto bias_type = descriptor.bias_type;
    auto mask_type = descriptor.mask_type;
    auto qkv_layout = descriptor.qkv_layout;
    auto dtype = descriptor.dtype;

    /* Input tensors */
    auto output_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
    auto bias_shape = std::vector<size_t>{bias_batch, bias_heads, q_max_seqlen, kv_max_seqlen};
    auto output_tensor = TensorWrapper(output, output_shape, dtype);
    auto doutput_tensor = TensorWrapper(doutput, output_shape, dtype);

    /* Output tensors */
    auto s_tensor = TensorWrapper(nullptr, std::vector<size_t>{1}, dtype);  // not used in F16
    auto dbias_tensor = TensorWrapper(dbias, bias_shape, dtype);
    auto q_cu_seqlens_tensor =
        TensorWrapper(q_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);
    auto kv_cu_seqlens_tensor =
        TensorWrapper(kv_cu_seqlens, std::vector<size_t>{input_batch + 1}, DType::kInt32);

    /* Auxiliary tensors (propagated from the forward pass) */
    NVTETensorPack aux_input_tensors;
    nvte_tensor_pack_create(&aux_input_tensors);
    auto backend = nvte_get_fused_attn_backend(
        static_cast<NVTEDType>(dtype), static_cast<NVTEDType>(dtype), qkv_layout, bias_type,
        mask_type, dropout_probability, attn_heads, num_gqa_groups, q_max_seqlen, kv_max_seqlen,
        head_dim);
    PrepareFusedAttnBackwardAuxTensors(&aux_input_tensors, &descriptor, backend, softmax_aux,
                                       rng_state, bias);

    /* cuDNN workspace */
    auto wkspace_size = std::vector<size_t>{descriptor.wkspace_size};
    auto wkspace_dtype = descriptor.wkspace_dtype;
    auto workspace_tensor = TensorWrapper(workspace, wkspace_size, wkspace_dtype);

    /* Call the underly NVTE API */
    if (qkv_layout == NVTE_QKV_Layout::NVTE_BS3HD) {
        auto qkv = buffers[0];
        auto qkv_shape = std::vector<size_t>{input_batch * q_max_seqlen, 3, attn_heads, head_dim};
        auto qkv_tensor = TensorWrapper(qkv, qkv_shape, dtype);
        auto dqkv = buffers[10];
        auto dqkv_tensor = TensorWrapper(dqkv, qkv_shape, dtype);
        nvte_fused_attn_bwd_qkvpacked(
            qkv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
            s_tensor.data(),  // not used for F16
            s_tensor.data(),  // not used for F16
            &aux_input_tensors, dqkv_tensor.data(), dbias_tensor.data(), q_cu_seqlens_tensor.data(),
            q_max_seqlen, scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
            workspace_tensor.data(), stream);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BS2HD) {
        auto q = buffers[0];
        auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
        auto q_tensor = TensorWrapper(q, q_shape, dtype);
        auto kv = buffers[1];
        auto kv_shape =
            std::vector<size_t>{input_batch * kv_max_seqlen, 2, num_gqa_groups, head_dim};
        auto kv_tensor = TensorWrapper(kv, kv_shape, dtype);
        auto dq = buffers[10];
        auto dq_tensor = TensorWrapper(dq, q_shape, dtype);
        auto dkv = buffers[11];
        auto dkv_tensor = TensorWrapper(dkv, kv_shape, dtype);
        nvte_fused_attn_bwd_kvpacked(
            q_tensor.data(), kv_tensor.data(), output_tensor.data(), doutput_tensor.data(),
            s_tensor.data(),  // not used for F16
            s_tensor.data(),  // not used for F16
            &aux_input_tensors, dq_tensor.data(), dkv_tensor.data(), dbias_tensor.data(),
            q_cu_seqlens_tensor.data(), kv_cu_seqlens_tensor.data(), q_max_seqlen, kv_max_seqlen,
            scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
            workspace_tensor.data(), stream);
    } else if (qkv_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
        auto q = buffers[0];
        auto q_shape = std::vector<size_t>{input_batch * q_max_seqlen, attn_heads, head_dim};
        auto q_tensor = TensorWrapper(q, q_shape, dtype);
        auto k = buffers[1];
        auto k_shape = std::vector<size_t>{input_batch * kv_max_seqlen, num_gqa_groups, head_dim};
        auto k_tensor = TensorWrapper(k, k_shape, dtype);
        auto v = buffers[2];
        auto v_shape = k_shape;
        auto v_tensor = TensorWrapper(v, v_shape, dtype);
        auto dq = buffers[10];
        auto dq_tensor = TensorWrapper(dq, q_shape, dtype);
        auto dk = buffers[11];
        auto dk_tensor = TensorWrapper(dk, k_shape, dtype);
        auto dv = buffers[12];
        auto dv_tensor = TensorWrapper(dv, v_shape, dtype);
        nvte_fused_attn_bwd(q_tensor.data(), k_tensor.data(), v_tensor.data(), output_tensor.data(),
                            doutput_tensor.data(),
                            s_tensor.data(),  // not used for F16
                            s_tensor.data(),  // not used for F16
                            &aux_input_tensors, dq_tensor.data(), dk_tensor.data(),
                            dv_tensor.data(), dbias_tensor.data(), q_cu_seqlens_tensor.data(),
                            kv_cu_seqlens_tensor.data(), q_max_seqlen, kv_max_seqlen,
                            scaling_factor, dropout_probability, qkv_layout, bias_type, mask_type,
                            workspace_tensor.data(), stream);
    } else {
        NVTE_ERROR("Unsupported qkv_layout.");
    }

    nvte_tensor_pack_destroy(&aux_input_tensors);
}

}  // namespace jax
}  // namespace transformer_engine
