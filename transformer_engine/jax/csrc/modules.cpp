/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reservedesc.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "jax/csrc/modules.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/common.h"
#include "transformer_engine/activation.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/layer_norm.h"
#include "transformer_engine/rmsnorm.h"
#include "transformer_engine/softmax.h"
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/transpose.h"
#include "utils.h"

namespace transformer_engine {
namespace jax {

constexpr size_t kCublasLtForwardWorkspaceSize = 32 * 1024 * 1024;
constexpr size_t kCublasLtBackwardWorkspaceSize = 32 * 1024 * 1024;

inline bool use_fp8(DType type) { return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2; }

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

pybind11::bytes PackCustomCallGemmDescriptor(size_t m, size_t n, size_t k, DType A_dtype,
                                             DType B_dtype, DType D_dtype, bool transa, bool transb,
                                             bool use_split_accumulator) {
    return PackOpaque(CustomCallGemmDescriptor{m, n, k, A_dtype, B_dtype, D_dtype, transa, transb,
                                               use_split_accumulator});
}

pybind11::bytes PackCustomCallNormDescriptor(size_t n, size_t hidden, DType x_dtype, DType w_dtype,
                                             float eps) {
    return PackOpaque(CustomCallNormDescriptor{n, hidden, x_dtype, w_dtype, eps});
}

pybind11::bytes PackCustomCallSoftmaxDescriptor(size_t batch, size_t pad_batch, size_t heads,
                                                size_t q_seqlen, size_t k_seqlen, DType dtype,
                                                float scale_factor) {
    return PackOpaque(
        SoftmaxDescriptor{batch, pad_batch, heads, q_seqlen, k_seqlen, dtype, scale_factor});
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

void Gemm(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *A = buffers[0];
    auto *B = buffers[1];
    auto *A_scale_inverse = reinterpret_cast<float *>(buffers[2]);
    auto *B_scale_inverse = reinterpret_cast<float *>(buffers[3]);
    auto *D = buffers[4];

    // We transposes shape of A, B and D here to correctly invoke
    // cuBlasLt GEMM (col-major) for row-major data.
    const auto &desc = *UnpackOpaque<CustomCallGemmDescriptor>(opaque, opaque_len);

    auto m = desc.m;
    auto n = desc.n;
    auto k = desc.k;
    auto A_shape = std::vector<size_t>{k, m};
    auto A_tensor = TensorWrapper(A, A_shape, desc.A_dtype, nullptr, nullptr, A_scale_inverse);

    auto B_shape = std::vector<size_t>{n, k};
    auto B_tensor = TensorWrapper(B, B_shape, desc.B_dtype, nullptr, nullptr, B_scale_inverse);

    auto D_shape = std::vector<size_t>{n, m};
    auto D_tensor = TensorWrapper(D, D_shape, desc.D_dtype);

    auto null_tensor = TensorWrapper(nullptr, std::vector<size_t>{0}, DType::kFloat32);

    size_t workspace_size = kCublasLtForwardWorkspaceSize;
    auto *workspace = cublasLtMetaManager::Instance().GetWorkspace(workspace_size);
    auto wk_tensor = TensorWrapper(workspace, std::vector<size_t>{workspace_size}, DType::kByte);

    nvte_cublas_gemm(A_tensor.data(), B_tensor.data(), D_tensor.data(), null_tensor.data(),
                     null_tensor.data(), (desc.transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
                     (desc.transb) ? CUBLAS_OP_T : CUBLAS_OP_N, false, wk_tensor.data(), false,
                     desc.use_split_accumulator, stream);
}

void LayerNormForwardImpl(size_t n, size_t hidden, void *input, DType in_dtype, void *weight,
                          DType w_dtype, void *bias, float eps, void *output, DType out_dtype,
                          void *mu, void *rsigma, float *amax, float *scale, float *scale_inv,
                          cudaStream_t stream) {
    auto input_shape = std::vector<size_t>{n, hidden};
    auto weight_shape = std::vector<size_t>{hidden};
    auto intermediates_shape = std::vector<size_t>{n};
    auto is_layer_norm = (bias) ? true : false;

    auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
    auto gamma_tensor = TensorWrapper(weight, weight_shape, in_dtype);

    // assume output dtype = input dtype
    // If we need mixed I/O precision in the future, we need an additional
    // parameter for output type
    auto output_tensor = TensorWrapper(output, input_shape, out_dtype, amax, scale, scale_inv);
    auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, DType::kFloat32);

    // Create uninitialized workspace, barrier and init them on the first
    TensorWrapper dummy_workspace_tensor, dummy_barrier_tensor;
    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // The first call is to query the required workspace
    if (is_layer_norm) {
        auto beta_tensor = TensorWrapper(bias, weight_shape, w_dtype);
        auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

        nvte_layernorm_fwd(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), eps,
                           output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(), stream,
                           num_sm, dummy_workspace_tensor.data(), dummy_barrier_tensor.data());
    } else {
        nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), eps, output_tensor.data(),
                         rsigma_tensor.data(), stream, num_sm, dummy_workspace_tensor.data(),
                         dummy_barrier_tensor.data());
    }

    size_t workspace_size =
        dummy_workspace_tensor.shape().data[0] * typeToSize(dummy_workspace_tensor.dtype()) +
        dummy_barrier_tensor.shape().data[0] * typeToSize(dummy_barrier_tensor.dtype());

    void *workspace = cublasLtMetaManager::Instance().GetWorkspace(workspace_size);

    auto workspace_tensor =
        TensorWrapper(workspace, dummy_workspace_tensor.shape(), dummy_workspace_tensor.dtype());

    auto barrier_tensor =
        TensorWrapper(reinterpret_cast<char *>(workspace) + dummy_workspace_tensor.shape().data[0],
                      dummy_barrier_tensor.shape(), dummy_barrier_tensor.dtype());

    if (is_layer_norm) {
        auto beta_tensor = TensorWrapper(bias, weight_shape, w_dtype);
        auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

        nvte_layernorm_fwd(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), eps,
                           output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(), stream,
                           num_sm, workspace_tensor.data(), barrier_tensor.data());
    } else {
        nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), eps, output_tensor.data(),
                         rsigma_tensor.data(), stream, num_sm, workspace_tensor.data(),
                         barrier_tensor.data());
    }
}

void LayerNormBackwardImpl(size_t n, size_t hidden, void *input, DType in_dtype, void *weight,
                           DType w_dtype, void *ograd, void *mu, void *rsigma, float eps,
                           void *xgrad, void *wgrad, void *dbeta, cudaStream_t stream) {
    auto input_shape = std::vector<size_t>{n, hidden};
    auto weight_shape = std::vector<size_t>{hidden};
    auto intermediates_shape = std::vector<size_t>{n};
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

    TensorWrapper dummy_workspace_tensor, dummy_barrier_tensor;
    TensorWrapper dummy_dgamma_part_tensor, dummy_dbeta_part_tensor;
    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();
    size_t dbeta_part_size{};

    // The first call is to query the workspace
    if (is_layer_norm) {
        auto mu_tensor = TensorWrapper(mu, intermediates_shape, intermediates_dtype);
        auto dbeta_tensor = TensorWrapper(dbeta, weight_shape, w_dtype);

        nvte_layernorm_bwd(dz_tensor.data(), x_tensor.data(), mu_tensor.data(),
                           rsigma_tensor.data(), gamma_tensor.data(), xgrad_tensor.data(),
                           wgrad_tensor.data(), dbeta_tensor.data(),
                           dummy_dgamma_part_tensor.data(), dummy_dbeta_part_tensor.data(), stream,
                           num_sm, dummy_workspace_tensor.data(), dummy_barrier_tensor.data());

        dbeta_part_size = dummy_dbeta_part_tensor.shape().data[0] *
                          dummy_dbeta_part_tensor.shape().data[1] *
                          typeToSize(dummy_dbeta_part_tensor.dtype());
    } else {
        nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(),
                         gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                         dummy_dgamma_part_tensor.data(), stream, num_sm,
                         dummy_workspace_tensor.data(), dummy_barrier_tensor.data());
    }

    size_t workspace_size =
        dummy_workspace_tensor.shape().data[0] * typeToSize(dummy_workspace_tensor.dtype());
    size_t barrier_size =
        dummy_barrier_tensor.shape().data[0] * typeToSize(dummy_barrier_tensor.dtype());
    size_t dgamma_part_size = dummy_dgamma_part_tensor.shape().data[0] *
                              dummy_dgamma_part_tensor.shape().data[1] *
                              typeToSize(dummy_dgamma_part_tensor.dtype());
    size_t total_workspace_size =
        (workspace_size + barrier_size + dgamma_part_size + dbeta_part_size);

    void *workspace = cublasLtMetaManager::Instance().GetWorkspace(total_workspace_size);
    void *barrier = static_cast<char *>(workspace) + workspace_size;
    void *dgamma_part = static_cast<char *>(barrier) + barrier_size;
    void *dbeta_part = static_cast<char *>(dgamma_part) + dgamma_part_size;

    auto workspace_tensor =
        TensorWrapper(workspace, dummy_workspace_tensor.shape(), dummy_workspace_tensor.dtype());

    auto barrier_tensor =
        TensorWrapper(barrier, dummy_barrier_tensor.shape(), dummy_barrier_tensor.dtype());

    auto dgamma_part_tensor = TensorWrapper(dgamma_part, dummy_dgamma_part_tensor.shape(),
                                            dummy_dgamma_part_tensor.dtype());

    if (is_layer_norm) {
        auto mu_tensor = TensorWrapper(mu, intermediates_shape, intermediates_dtype);
        auto dbeta_tensor = TensorWrapper(dbeta, weight_shape, w_dtype);
        auto dbeta_part_tensor = TensorWrapper(dbeta_part, dummy_dbeta_part_tensor.shape(),
                                               dummy_dbeta_part_tensor.dtype());

        nvte_layernorm_bwd(dz_tensor.data(), x_tensor.data(), mu_tensor.data(),
                           rsigma_tensor.data(), gamma_tensor.data(), xgrad_tensor.data(),
                           wgrad_tensor.data(), dbeta_tensor.data(), dgamma_part_tensor.data(),
                           dbeta_part_tensor.data(), stream, num_sm, workspace_tensor.data(),
                           barrier_tensor.data());
    } else {
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
    assert(amax_out == amax);

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto n = desc.n;
    auto hidden = desc.hidden;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto eps = desc.eps;

    auto out_dtype = DType::kFloat8E4M3;

    LayerNormForwardImpl(n, hidden, input, in_dtype, weight, w_dtype, bias, eps, output, out_dtype,
                         mu, rsigma, amax, scale, scale_inv, stream);
}

void LayerNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *weight = buffers[1];
    auto *bias = buffers[2];
    auto *output = buffers[3];
    auto *mu = buffers[4];
    auto *rsigma = buffers[5];

    float *amax = nullptr;
    float *scale = nullptr;
    float *scale_inv = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto n = desc.n;
    auto hidden = desc.hidden;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto eps = desc.eps;
    auto out_dtype = in_dtype;

    LayerNormForwardImpl(n, hidden, input, in_dtype, weight, w_dtype, bias, eps, output, out_dtype,
                         mu, rsigma, amax, scale, scale_inv, stream);
}

void LayerNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);

    auto n = desc.n;
    auto hidden = desc.hidden;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto eps = desc.eps;

    auto *ograd = buffers[0];
    auto *mu = buffers[1];
    auto *rsigma = buffers[2];
    auto *input = buffers[3];
    auto *weight = buffers[4];
    auto *xgrad = buffers[5];
    auto *wgrad = buffers[6];
    auto *dbeta = buffers[7];

    LayerNormBackwardImpl(n, hidden, input, in_dtype, weight, w_dtype, ograd, mu, rsigma, eps,
                          xgrad, wgrad, dbeta, stream);
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
    assert(amax_out == amax);

    void *bias = nullptr;
    void *mu = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto n = desc.n;
    auto hidden = desc.hidden;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto eps = desc.eps;
    auto out_dtype = DType::kFloat8E4M3;

    LayerNormForwardImpl(n, hidden, input, in_dtype, weight, w_dtype, bias, eps, output, out_dtype,
                         mu, rsigma, amax, scale, scale_inv, stream);
}

void RMSNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *input = buffers[0];
    auto *weight = buffers[1];
    auto *output = buffers[2];
    auto *rsigma = buffers[3];

    void *bias = nullptr;
    void *mu = nullptr;
    float *amax = nullptr;
    float *scale = nullptr;
    float *scale_inv = nullptr;

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto n = desc.n;
    auto hidden = desc.hidden;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto eps = desc.eps;
    auto out_dtype = in_dtype;

    LayerNormForwardImpl(n, hidden, input, in_dtype, weight, w_dtype, bias, eps, output, out_dtype,
                         mu, rsigma, amax, scale, scale_inv, stream);
}

void RMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    auto *ograd = buffers[0];
    auto *rsigma = buffers[1];
    auto *input = buffers[2];
    auto *weight = buffers[3];
    auto *xgrad = buffers[4];
    auto *wgrad = buffers[5];

    const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
    auto n = desc.n;
    auto hidden = desc.hidden;
    auto in_dtype = desc.x_dtype;
    auto w_dtype = desc.w_dtype;
    auto eps = desc.eps;

    void *mu = nullptr;
    void *dbeta = nullptr;

    LayerNormBackwardImpl(n, hidden, input, in_dtype, weight, w_dtype, ograd, mu, rsigma, eps,
                          xgrad, wgrad, dbeta, stream);
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
    auto shape = std::vector<size_t>{desc.batch, desc.heads, desc.q_seqlen, desc.k_seqlen};
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
    auto shape = std::vector<size_t>{desc.batch, desc.heads, desc.q_seqlen, desc.k_seqlen};
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
    auto io_shape = std::vector<size_t>{desc.batch, desc.heads, desc.q_seqlen, desc.k_seqlen};
    auto mask_shape = std::vector<size_t>{desc.pad_batch, 1, desc.q_seqlen, desc.k_seqlen};
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
    auto attn_batch = desc.batch * desc.heads;
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
    auto attn_batch = desc.batch * desc.heads;
    auto shape = std::vector<size_t>{attn_batch, desc.q_seqlen, desc.k_seqlen};
    auto dtype = desc.dtype;

    auto grad_output_tensor = TensorWrapper(grad_output, shape, dtype);
    auto softmax_output_tensor = TensorWrapper(softmax_output, shape, dtype);
    auto dgrad_tensor = TensorWrapper(dgrad, shape, dtype);

    nvte_scaled_upper_triang_masked_softmax_backward(
        grad_output_tensor.data(), softmax_output_tensor.data(), dgrad_tensor.data(),
        desc.scale_factor, stream);
}
}  // namespace jax
}  // namespace transformer_engine
