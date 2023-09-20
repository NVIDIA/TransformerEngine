/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <vector>
#include "../common.h"
#include "common.h"

namespace transformer_engine {
namespace paddle_ext {

// MHA utils
// convert QKV layout to enum
NVTE_QKV_Layout get_nvte_qkv_layout(const std::string qkv_layout) {
    if (qkv_layout == "not_interleaved") {
        return NVTE_QKV_Layout::NVTE_NOT_INTERLEAVED;
    } else if (qkv_layout == "qkv_interleaved") {
        return NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED;
    } else if (qkv_layout == "kv_interleaved") {
        return NVTE_QKV_Layout::NVTE_KV_INTERLEAVED;
    } else {
        NVTE_ERROR("Invalid QKV layout. \n");
    }
}

// convert bias type to enum
NVTE_Bias_Type get_nvte_bias_type(const std::string bias_type) {
    if (bias_type == "no_bias") {
        return NVTE_Bias_Type::NVTE_NO_BIAS;
    } else if (bias_type == "pre_scale_bias") {
        return NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS;
    } else if (bias_type == "post_scale_bias") {
        return NVTE_Bias_Type::NVTE_POST_SCALE_BIAS;
    } else {
        NVTE_ERROR("Invalid bias type. \n");
    }
}

// convert attn mask type to enum
NVTE_Mask_Type get_nvte_mask_type(const std::string mask_type) {
    if (mask_type == "padding") {
        return NVTE_Mask_Type::NVTE_PADDING_MASK;
    } else if (mask_type == "causal") {
        return NVTE_Mask_Type::NVTE_CAUSAL_MASK;
    } else if (mask_type == "no_mask") {
        return NVTE_Mask_Type::NVTE_NO_MASK;
    } else {
        NVTE_ERROR("Invalid attention mask type. \n");
    }
}

std::vector<paddle::Tensor> cast_to_fp8(const paddle::Tensor &input, const paddle::Tensor &scale,
                                        paddle::Tensor &amax, paddle::Tensor &scale_inv,  // NOLINT
                                        int64_t index, int64_t otype) {
    auto shape = GetShapeArray(input);

    auto output = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)));

    auto input_cu = MakeNvteTensor(input);
    auto output_cu = MakeNvteTensor(
        output.data(), shape, Int2NvteDType(otype), GetDataPtr<float>(amax, index),
        const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));

    nvte_fp8_quantize(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

std::vector<paddle::Tensor> cast_from_fp8(const paddle::Tensor &input,
                                          const paddle::Tensor &scale_inv, int64_t index,
                                          int64_t itype, int64_t otype) {
    auto shape = GetShapeArray(input);

    auto output = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)));
    auto input_cu =
        MakeNvteTensor(const_cast<void *>(input.data()), shape, Int2NvteDType(itype), nullptr,
                       nullptr, const_cast<void *>(GetDataPtr<float>(scale_inv, index)));
    auto output_cu = MakeNvteTensor(output);

    nvte_fp8_dequantize(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

std::vector<paddle::Tensor> te_transpose(const paddle::Tensor &input, int64_t otype) {
    auto shape = GetShapeArray(input);
    NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");
    size_t M = shape[0];
    size_t N = shape[1];

    auto output = paddle::empty({input.shape()[1], input.shape()[0]}, input.dtype(), input.place());

    auto input_cu = MakeNvteTensor(const_cast<void *>(input.data()), {M, N}, Int2NvteDType(otype));
    auto output_cu = MakeNvteTensor(output.data(), {N, M}, Int2NvteDType(otype));

    nvte_transpose(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

std::vector<paddle::Tensor> te_cast_transpose(const paddle::Tensor &input,
                                              const paddle::Tensor &scale,
                                              paddle::Tensor &amax,       // NOLINT
                                              paddle::Tensor &scale_inv,  // NOLINT
                                              int64_t index, int64_t otype) {
    auto shape = GetShapeArray(input);
    NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");

    size_t M = shape[0];
    size_t N = shape[1];

    auto input_cast =
        paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)), input.place());
    auto input_transpose = paddle::empty({input.shape()[1], input.shape()[0]},
                                         Nvte2PaddleDType(Int2NvteDType(otype)), input.place());

    auto input_cu = MakeNvteTensor(input);
    void *amax_data = GetDataPtr<float>(amax, index);
    void *scale_data = const_cast<void *>(GetDataPtr<float>(scale, index));
    void *scale_inv_data = GetDataPtr<float>(scale_inv, index);
    auto output_cast_cu = MakeNvteTensor(input_cast.data(), {M, N}, Int2NvteDType(otype), amax_data,
                                         scale_data, scale_inv_data);
    auto output_transpose_cu = MakeNvteTensor(input_transpose.data(), {N, M}, Int2NvteDType(otype),
                                              amax_data, scale_data, scale_inv_data);

    nvte_cast_transpose(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                        input.stream());

    return {input_cast, input_transpose};
}

std::vector<paddle::Tensor> te_cast_transpose_bgrad(const paddle::Tensor &grad_output,
                                                    const paddle::Tensor &scale,
                                                    paddle::Tensor &amax,       // NOLINT
                                                    paddle::Tensor &scale_inv,  // NOLINT
                                                    int64_t index, int64_t otype) {
    auto shape = GetShapeArray(grad_output);
    NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");

    size_t M = shape[0];
    size_t N = shape[1];

    auto grad_bias =
        paddle::empty({grad_output.shape()[1]}, grad_output.dtype(), grad_output.place());
    auto grad_output_cast = paddle::empty_like(grad_output, Nvte2PaddleDType(Int2NvteDType(otype)),
                                               grad_output.place());
    auto grad_output_transpose =
        paddle::empty({grad_output.shape()[1], grad_output.shape()[0]},
                      Nvte2PaddleDType(Int2NvteDType(otype)), grad_output.place());

    auto input_cu = MakeNvteTensor(grad_output);
    void *amax_data = GetDataPtr<float>(amax, index);
    void *scale_data = const_cast<void *>(GetDataPtr<float>(scale, index));
    void *scale_inv_data = GetDataPtr<float>(scale_inv, index);
    auto output_cast_cu = MakeNvteTensor(grad_output_cast.data(), {M, N}, Int2NvteDType(otype),
                                         amax_data, scale_data, scale_inv_data);
    auto output_transpose_cu =
        MakeNvteTensor(grad_output_transpose.data(), {N, M}, Int2NvteDType(otype), amax_data,
                       scale_data, scale_inv_data);
    auto dbias_cu = MakeNvteTensor(grad_bias);
    transformer_engine::TensorWrapper workspace;

    nvte_cast_transpose_dbias(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                              dbias_cu.data(), workspace.data(), grad_output.stream());

    // Fill workspace
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), grad_output.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

    nvte_cast_transpose_dbias(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                              dbias_cu.data(), workspace.data(), grad_output.stream());

    return {grad_bias, grad_output_cast, grad_output_transpose};
}

void te_gemm(const paddle::Tensor &A, const paddle::optional<paddle::Tensor> &A_scale_inverse,
             const paddle::Tensor &B, const paddle::optional<paddle::Tensor> &B_scale_inverse,
             const paddle::optional<paddle::Tensor> &bias, paddle::Tensor &D,            // NOLINT
             paddle::optional<paddle::Tensor> &D_scale,                                  // NOLINT
             paddle::optional<paddle::Tensor> &D_amax,                                   // NOLINT
             paddle::optional<paddle::Tensor> &pre_gelu_out, paddle::Tensor &workspace,  // NOLINT
             int64_t A_index, int64_t B_index, int64_t D_index, int64_t A_type, int64_t B_type,
             int64_t D_type, int64_t bias_type, bool transa, bool transb, bool grad,
             int64_t workspace_size, bool accumulate, bool use_split_accumulator,
             int64_t math_sm_count) {
    auto te_A = MakeNvteTensor(
        const_cast<void *>(A.data()), GetShapeArray(A), Int2NvteDType(A_type), nullptr, nullptr,
        const_cast<void *>(GetOptionalDataPtr<float>(A_scale_inverse, A_index)));
    auto te_B = MakeNvteTensor(
        const_cast<void *>(B.data()), GetShapeArray(B), Int2NvteDType(B_type), nullptr, nullptr,
        const_cast<void *>(GetOptionalDataPtr<float>(B_scale_inverse, B_index)));
    auto te_D = MakeNvteTensor(D.data(), GetShapeArray(D), Int2NvteDType(D_type),
                               GetOptionalDataPtr<float>(D_amax, D_index),
                               GetOptionalDataPtr<float>(D_scale, D_index), nullptr);

    auto te_bias = MakeNvteTensor(const_cast<void *>(GetOptionalDataPtr(bias)), GetShapeArray(bias),
                                  Int2NvteDType(bias_type));

    DType gelu_dtype =
        pre_gelu_out ? Paddle2NvteDType(pre_gelu_out->dtype()) : Int2NvteDType(D_type);
    auto te_pre_gelu_out =
        MakeNvteTensor(GetOptionalDataPtr(pre_gelu_out), GetShapeArray(pre_gelu_out), gelu_dtype);
    auto te_workspace =
        MakeNvteTensor(workspace.data(), {static_cast<size_t>(workspace_size)}, DType::kByte);

    nvte_cublas_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(), te_pre_gelu_out.data(),
                     transa, transb, grad, te_workspace.data(), accumulate, use_split_accumulator,
                     math_sm_count, A.stream());
}

std::vector<paddle::Tensor> te_gelu_fp8(const paddle::Tensor &input, const paddle::Tensor &scale,
                                        paddle::Tensor &amax,       // NOLINT
                                        paddle::Tensor &scale_inv,  // NOLINT
                                        int64_t index, int64_t otype) {
    auto output = paddle::empty_like(input, Nvte2PaddleDType(DType::kByte), input.place());

    auto input_cu = MakeNvteTensor(input);
    auto output_cu = MakeNvteTensor(
        output.data(), GetShapeArray(input), Int2NvteDType(otype), GetDataPtr<float>(amax, index),
        const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));

    nvte_gelu(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

std::vector<paddle::Tensor> te_gelu(const paddle::Tensor &input, int64_t otype) {
    auto output = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)), input.place());

    auto input_cu = MakeNvteTensor(input);
    auto output_cu = MakeNvteTensor(output.data(), GetShapeArray(input), Int2NvteDType(otype));

    nvte_gelu(input_cu.data(), output_cu.data(), input.stream());

    return {output};
}

std::vector<paddle::Tensor> te_cast_transpose_bgrad_dgelu(const paddle::Tensor &grad_output,
                                                          const paddle::Tensor &gelu_input,
                                                          const paddle::Tensor &scale,
                                                          paddle::Tensor &amax,       // NOLINT
                                                          paddle::Tensor &scale_inv,  // NOLINT
                                                          int64_t index, int64_t otype) {
    auto shape = GetShapeArray(grad_output);
    NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

    size_t M = shape[0];
    size_t N = shape[1];

    // DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
    auto grad_bias =
        paddle::empty({grad_output.shape()[1]}, grad_output.dtype(), grad_output.place());

    auto dgelu =
        paddle::empty_like(grad_output, Nvte2PaddleDType(DType::kByte), grad_output.place());

    auto dgelu_transpose = paddle::empty({grad_output.shape()[1], grad_output.shape()[0]},
                                         Nvte2PaddleDType(DType::kByte), grad_output.place());

    void *amax_data = GetDataPtr<float>(amax, index);
    void *scale_data = const_cast<void *>(GetDataPtr<float>(scale, index));
    void *scale_inv_data = GetDataPtr<float>(scale_inv, index);

    TensorWrapper workspace;

    auto gelu_input_cu = MakeNvteTensor(gelu_input);
    auto input_cu = MakeNvteTensor(grad_output);
    auto cast_output_cu = MakeNvteTensor(dgelu.data(), {M, N}, Int2NvteDType(otype), amax_data,
                                         scale_data, scale_inv_data);
    auto transposed_output_cu = MakeNvteTensor(dgelu_transpose.data(), {N, M}, Int2NvteDType(otype),
                                               amax_data, scale_data, scale_inv_data);
    auto dbias_cu = MakeNvteTensor(grad_bias);

    nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(), cast_output_cu.data(),
                                    transposed_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    grad_output.stream());

    // Fill workspace
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), grad_output.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

    nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(), cast_output_cu.data(),
                                    transposed_output_cu.data(), dbias_cu.data(), workspace.data(),
                                    grad_output.stream());

    return {dgelu, dgelu_transpose, grad_bias};
}

std::vector<paddle::Tensor> te_layernorm_fwd_fp8(const paddle::Tensor &input,
                                                 const paddle::Tensor &weight,
                                                 const paddle::Tensor &bias,
                                                 const paddle::Tensor &scale,
                                                 paddle::Tensor &amax,       // NOLINT
                                                 paddle::Tensor &scale_inv,  // NOLINT
                                                 float eps, int64_t index, int64_t otype,
                                                 int64_t sm_margin, bool zero_centered_gamma) {
    auto shape = GetShapeArray(input);
    NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

    size_t N = shape[0];
    size_t H = shape[1];

    auto ln_out = paddle::empty_like(input, input.dtype(), input.place());
    auto mu = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
    auto rsigma =
        paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
    auto input_cu = MakeNvteTensor(input);
    auto gamma_cu = MakeNvteTensor(weight);
    auto beta_cu = MakeNvteTensor(bias);
    auto z_cu = MakeNvteTensor(
        ln_out.data(), {N, H}, Int2NvteDType(otype), GetDataPtr<float>(amax, index),
        const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));
    auto mu_cu = MakeNvteTensor(mu);
    auto rsigma_cu = MakeNvteTensor(rsigma);
    TensorWrapper workspace, barrier;

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // This call populates workspace and barrier tensors with the required config
    const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(), mu_cu.data(),
         rsigma_cu.data(), input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), input.place());
    auto barrier_data = AllocateSpace(barrier.shape(), barrier.dtype(), input.place(), true);
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());
    barrier = MakeNvteTensor(barrier_data.data(), barrier.shape(), barrier.dtype());

    // Actual call to fwd kernel
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(), mu_cu.data(),
         rsigma_cu.data(), input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    return {ln_out, mu, rsigma};
}

std::vector<paddle::Tensor> te_layernorm_fwd(const paddle::Tensor &input,
                                             const paddle::Tensor &weight,
                                             const paddle::Tensor &bias, float eps, int64_t otype,
                                             int64_t sm_margin, bool zero_centered_gamma) {
    auto shape = GetShapeArray(input);
    NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

    size_t N = shape[0];
    size_t H = shape[1];

    auto ln_out = paddle::empty_like(input, input.dtype(), input.place());
    auto mu = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
    auto rsigma =
        paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
    auto input_cu = MakeNvteTensor(input);
    auto gamma_cu = MakeNvteTensor(weight);
    auto beta_cu = MakeNvteTensor(bias);
    auto z_cu = MakeNvteTensor(ln_out.data(), {N, H}, Int2NvteDType(otype));
    auto mu_cu = MakeNvteTensor(mu);
    auto rsigma_cu = MakeNvteTensor(rsigma);
    TensorWrapper workspace, barrier;

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // This call populates workspace and barrier tensors with the required config
    const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(), mu_cu.data(),
         rsigma_cu.data(), input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), input.place());
    auto barrier_data = AllocateSpace(barrier.shape(), barrier.dtype(), input.place(), true);
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());
    barrier = MakeNvteTensor(barrier_data.data(), barrier.shape(), barrier.dtype());

    // Actual call to fwd kernel
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(), mu_cu.data(),
         rsigma_cu.data(), input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    return {ln_out, mu, rsigma};
}

std::vector<paddle::Tensor> te_layernorm_bwd(const paddle::Tensor &dz, const paddle::Tensor &x,
                                             const paddle::Tensor &mu, const paddle::Tensor &rsigma,
                                             const paddle::Tensor &gamma, int64_t sm_margin,
                                             bool zero_centered_gamma) {
    auto dx = paddle::empty_like(x, x.dtype(), x.place());
    auto dgamma = paddle::empty_like(gamma, gamma.dtype(), gamma.place());
    auto dbeta = paddle::empty_like(gamma, gamma.dtype(), gamma.place());

    TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

    auto dz_cu = MakeNvteTensor(dz);
    auto x_cu = MakeNvteTensor(x);
    auto mu_cu = MakeNvteTensor(mu);
    auto rsigma_cu = MakeNvteTensor(rsigma);
    auto gamma_cu = MakeNvteTensor(gamma);
    auto dx_cu = MakeNvteTensor(dx);
    auto dgamma_cu = MakeNvteTensor(dgamma);
    auto dbeta_cu = MakeNvteTensor(dbeta);

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // This call populates tensors with the required config.
    const auto bwd_fun = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;
    bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
            dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(),
            dz.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    // Alloc space for Tensors.
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), x.place());
    auto barrier_data = AllocateSpace(barrier.shape(), barrier.dtype(), x.place(), true);
    auto dgamma_part_data = AllocateSpace(dgamma_part.shape(), dgamma_part.dtype(), x.place());
    auto dbeta_part_data = AllocateSpace(dbeta_part.shape(), dbeta_part.dtype(), x.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());
    barrier = MakeNvteTensor(barrier_data.data(), barrier.shape(), barrier.dtype());
    dgamma_part = MakeNvteTensor(dgamma_part_data.data(), dgamma_part.shape(), dgamma_part.dtype());
    dbeta_part = MakeNvteTensor(dbeta_part_data.data(), dbeta_part.shape(), dbeta_part.dtype());

    // Actual call to bwd kernel.
    bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
            dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(),
            dz.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    return {dx, dgamma, dbeta};
}

std::vector<paddle::Tensor> te_rmsnorm_fwd(const paddle::Tensor &input,
                                           const paddle::Tensor &weight, float eps, int64_t otype,
                                           int64_t sm_margin) {
    auto shape = GetShapeArray(input);
    NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

    size_t N = shape[0];
    size_t H = shape[1];

    auto ln_out = paddle::empty_like(input, input.dtype(), input.place());
    auto rsigma =
        paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
    auto input_cu = MakeNvteTensor(input);
    auto gamma_cu = MakeNvteTensor(weight);
    auto z_cu = MakeNvteTensor(ln_out.data(), {N, H}, Int2NvteDType(otype));
    auto rsigma_cu = MakeNvteTensor(rsigma);
    TensorWrapper workspace, barrier;

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // This call populates workspace and barrier tensors with the required config

    nvte_rmsnorm_fwd(input_cu.data(), gamma_cu.data(), eps, z_cu.data(), rsigma_cu.data(),
                     input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), input.place());
    auto barrier_data = AllocateSpace(barrier.shape(), barrier.dtype(), input.place(), true);
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());
    barrier = MakeNvteTensor(barrier_data.data(), barrier.shape(), barrier.dtype());

    // Actual call to fwd kernel
    nvte_rmsnorm_fwd(input_cu.data(), gamma_cu.data(), eps, z_cu.data(), rsigma_cu.data(),
                     input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    return {ln_out, rsigma};
}

std::vector<paddle::Tensor> te_rmsnorm_fwd_fp8(const paddle::Tensor &input,
                                               const paddle::Tensor &weight,
                                               const paddle::Tensor &scale,
                                               paddle::Tensor &amax,       // NOLINT
                                               paddle::Tensor &scale_inv,  // NOLINT
                                               float eps, int64_t index, int64_t otype,
                                               int64_t sm_margin) {
    auto shape = GetShapeArray(input);
    NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

    size_t N = shape[0];
    size_t H = shape[1];

    auto ln_out = paddle::empty_like(input, input.dtype(), input.place());
    auto rsigma =
        paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
    auto input_cu = MakeNvteTensor(input);
    auto gamma_cu = MakeNvteTensor(weight);
    auto z_cu = MakeNvteTensor(
        ln_out.data(), {N, H}, Int2NvteDType(otype), GetDataPtr<float>(amax, index),
        const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));
    auto rsigma_cu = MakeNvteTensor(rsigma);
    TensorWrapper workspace, barrier;

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // This call populates workspace and barrier tensors with the required config
    nvte_rmsnorm_fwd(input_cu.data(), gamma_cu.data(), eps, z_cu.data(), rsigma_cu.data(),
                     input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), input.place());
    auto barrier_data = AllocateSpace(barrier.shape(), barrier.dtype(), input.place(), true);
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());
    barrier = MakeNvteTensor(barrier_data.data(), barrier.shape(), barrier.dtype());

    // Actual call to fwd kernel
    nvte_rmsnorm_fwd(input_cu.data(), gamma_cu.data(), eps, z_cu.data(), rsigma_cu.data(),
                     input.stream(), num_sm - sm_margin, workspace.data(), barrier.data());

    return {ln_out, rsigma};
}

std::vector<paddle::Tensor> te_rmsnorm_bwd(const paddle::Tensor &dz, const paddle::Tensor &x,
                                           const paddle::Tensor &rsigma,
                                           const paddle::Tensor &gamma, int64_t sm_margin) {
    auto dx = paddle::empty_like(x, x.dtype(), x.place());
    auto dgamma = paddle::empty_like(gamma, gamma.dtype(), gamma.place());

    TensorWrapper workspace, barrier, dgamma_part;

    auto dz_cu = MakeNvteTensor(dz);
    auto x_cu = MakeNvteTensor(x);
    auto rsigma_cu = MakeNvteTensor(rsigma);
    auto gamma_cu = MakeNvteTensor(gamma);
    auto dx_cu = MakeNvteTensor(dx);
    auto dgamma_cu = MakeNvteTensor(dgamma);

    auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

    // This call populates tensors with the required config.
    nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                     dgamma_cu.data(), dgamma_part.data(), dz.stream(), num_sm - sm_margin,
                     workspace.data(), barrier.data());

    // Alloc space for Tensors.
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), x.place());
    auto barrier_data = AllocateSpace(barrier.shape(), barrier.dtype(), x.place(), true);
    auto dgamma_part_data = AllocateSpace(dgamma_part.shape(), dgamma_part.dtype(), x.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());
    barrier = MakeNvteTensor(barrier_data.data(), barrier.shape(), barrier.dtype());
    dgamma_part = MakeNvteTensor(dgamma_part_data.data(), dgamma_part.shape(), dgamma_part.dtype());

    // Actual call to bwd kernel.
    nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                     dgamma_cu.data(), dgamma_part.data(), dz.stream(), num_sm - sm_margin,
                     workspace.data(), barrier.data());

    return {dx, dgamma};
}

void te_fused_attn_fwd_qkvpacked(const paddle::Tensor &QKV, const paddle::Tensor &cu_seqlens,
                                 const paddle::optional<paddle::Tensor> &Bias,
                                 paddle::Tensor &O,                              // NOLINT
                                 paddle::optional<paddle::Tensor> &softmax_aux,  // NOLINT
                                 paddle::Tensor &rng_state,                      // NOLINT
                                 int64_t b, int64_t h, int64_t d, int64_t total_seqs,
                                 int64_t max_seqlen, bool is_training, float attn_scale,
                                 float p_dropout, const std::string &qkv_layout,
                                 const std::string &bias_type, const std::string &attn_mask_type,
                                 const int64_t qkv_type) {
    if (is_training && !softmax_aux) {
        NVTE_ERROR("softmax_aux must be provided when training. \n");
    }

    auto qkv_dtype = Int2NvteDType(qkv_type);
    // construct NVTE tensors
    TensorWrapper te_QKV, te_S, te_O, te_Bias, te_cu_seqlens;
    if (qkv_dtype == DType::kBFloat16 || qkv_dtype == DType::kFloat16) {
        // BF16 or FP16
        te_QKV = MakeNvteTensor(QKV);
        te_S = MakeNvteTensor(nullptr, std::vector<size_t>{0}, DType::kFloat32);
        te_O = MakeNvteTensor(O);
    } else {  // TODO: support fp8
        NVTE_ERROR("Fused attention only supports BF16/FP16 data types. \n");
    }
    if ((bias_type != "no_bias") && Bias) {
        auto bias_shape = Bias->shape();
        std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
        te_Bias = MakeNvteTensor(GetOptionalDataPtr(Bias), shape, DType::kFloat32);
    }
    te_cu_seqlens = MakeNvteTensor(cu_seqlens.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);

    // convert strings to enums
    NVTE_QKV_Layout qkv_layout_enum = get_nvte_qkv_layout(qkv_layout);
    NVTE_Bias_Type bias_type_enum = get_nvte_bias_type(bias_type);
    NVTE_Mask_Type attn_mask_type_enum = get_nvte_mask_type(attn_mask_type);

    // extract random number generator seed and offset
    auto te_rng_state = MakeNvteTensor(rng_state);

    // create auxiliary output tensors
    NVTETensorPack nvte_aux_tensor_pack;
    nvte_tensor_pack_create(&nvte_aux_tensor_pack);

    // create workspace
    TensorWrapper workspace;

    // populate tensors with appropriate shapes and dtypes
    nvte_fused_attn_fwd_qkvpacked(
        te_QKV.data(), te_Bias.data(), te_S.data(), te_O.data(), &nvte_aux_tensor_pack,
        te_cu_seqlens.data(), te_rng_state.data(), max_seqlen, is_training, attn_scale, p_dropout,
        qkv_layout_enum, bias_type_enum, attn_mask_type_enum, workspace.data(), QKV.stream());

    // allocate memory for workspace and auxiliary output tensors
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), QKV.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

    auto *output_s =
        reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[0]);
    output_s->data.dptr = GetOptionalDataPtr(softmax_aux);

    // execute the kernel
    nvte_fused_attn_fwd_qkvpacked(
        te_QKV.data(), te_Bias.data(), te_S.data(), te_O.data(), &nvte_aux_tensor_pack,
        te_cu_seqlens.data(), te_rng_state.data(), max_seqlen, is_training, attn_scale, p_dropout,
        qkv_layout_enum, bias_type_enum, attn_mask_type_enum, workspace.data(), QKV.stream());

    // destroy tensor wrappers, but not allocated memory
    nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

// fused attention BWD with packed QKV
void te_fused_attn_bwd_qkvpacked(const paddle::Tensor &QKV, const paddle::Tensor &cu_seqlens,
                                 const paddle::Tensor &O, const paddle::Tensor &dO,
                                 const paddle::Tensor &softmax_aux,
                                 paddle::Tensor &dQKV,                     // NOLINT
                                 paddle::optional<paddle::Tensor> &dBias,  // NOLINT
                                 paddle::Tensor &rng_state,                // NOLINT
                                 int64_t b, int64_t h, int64_t d, int64_t total_seqs,
                                 int64_t max_seqlen, float attn_scale, float p_dropout,
                                 const std::string &qkv_layout, const std::string &bias_type,
                                 const std::string &attn_mask_type, int64_t qkv_type) {
    TensorWrapper te_dBias;
    if (bias_type != "no_bias" && dBias) {
        auto bias_shape = dBias->shape();
        std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
        te_dBias = MakeNvteTensor(GetOptionalDataPtr(dBias), shape, DType::kFloat32);
    }

    auto qkv_dtype = Int2NvteDType(qkv_type);
    // construct NVTE tensors
    TensorWrapper te_QKV, te_O, te_dO, te_S, te_dP, te_dQKV;
    if (qkv_dtype == DType::kBFloat16 || qkv_dtype == DType::kFloat16) {
        // BF16 or FP16
        te_QKV = MakeNvteTensor(QKV);
        te_O = MakeNvteTensor(O);
        te_dO = MakeNvteTensor(dO);
        te_S = MakeNvteTensor(nullptr, std::vector<size_t>(0), DType::kFloat32);
        te_dP = MakeNvteTensor(nullptr, std::vector<size_t>(0), DType::kFloat32);
        te_dQKV = MakeNvteTensor(dQKV);
    } else {
        NVTE_ERROR("Fused attention only supports BF16/FP16 data types. \n");
    }

    // convert strings to enums
    NVTE_QKV_Layout qkv_layout_enum = get_nvte_qkv_layout(qkv_layout);
    NVTE_Bias_Type bias_type_enum = get_nvte_bias_type(bias_type);
    NVTE_Mask_Type attn_mask_type_enum = get_nvte_mask_type(attn_mask_type);

    // convert auxiliary tensors from forward into NVTETensors
    NVTETensorPack nvte_aux_tensor_pack;
    nvte_tensor_pack_create(&nvte_aux_tensor_pack);

    nvte_aux_tensor_pack.size = 2;  // 1. softmax_aux  2. rng_state
    auto *output_s = reinterpret_cast<Tensor *>(nvte_aux_tensor_pack.tensors[0]);
    auto *fwd_rng_state = reinterpret_cast<Tensor *>(nvte_aux_tensor_pack.tensors[1]);
    output_s->data.shape =
        std::vector<size_t>({static_cast<size_t>(b), static_cast<size_t>(h),
                             static_cast<size_t>(max_seqlen), static_cast<size_t>(max_seqlen)});
    output_s->data.dptr = const_cast<void *>(softmax_aux.data());
    fwd_rng_state->data.shape = std::vector<size_t>({2});
    fwd_rng_state->data.dptr = const_cast<void *>(rng_state.data());

    // create cu_seqlens tensorwrappers
    TensorWrapper te_cu_seqlens;
    te_cu_seqlens = MakeNvteTensor(cu_seqlens.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);

    // create workspace
    TensorWrapper workspace;

    // populate tensors with appropriate shapes and dtypes
    nvte_fused_attn_bwd_qkvpacked(
        te_QKV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(), &nvte_aux_tensor_pack,
        te_dQKV.data(), te_dBias.data(), te_cu_seqlens.data(), max_seqlen, attn_scale, p_dropout,
        qkv_layout_enum, bias_type_enum, attn_mask_type_enum, workspace.data(), QKV.stream());

    // allocate memory for workspace
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), QKV.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

    // execute kernel
    nvte_fused_attn_bwd_qkvpacked(
        te_QKV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(), &nvte_aux_tensor_pack,
        te_dQKV.data(), te_dBias.data(), te_cu_seqlens.data(), max_seqlen, attn_scale, p_dropout,
        qkv_layout_enum, bias_type_enum, attn_mask_type_enum, workspace.data(), QKV.stream());

    // destroy tensor wrappers
    nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

void te_fused_attn_fwd_kvpacked(const paddle::Tensor &Q, const paddle::Tensor &KV,
                                const paddle::Tensor &cu_seqlens_q,
                                const paddle::Tensor &cu_seqlens_kv,
                                const paddle::optional<paddle::Tensor> &Bias,
                                paddle::Tensor &O,                              // NOLINT
                                paddle::optional<paddle::Tensor> &softmax_aux,  // NOLINT
                                paddle::Tensor &rng_state,                      // NOLINT
                                int64_t b, int64_t h, int64_t d, int64_t total_seqs_q,
                                int64_t total_seqs_kv, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                                bool is_training, float attn_scale, float p_dropout,
                                const std::string &qkv_layout, const std::string &bias_type,
                                const std::string &attn_mask_type, const int64_t qkv_type) {
    if (is_training && !softmax_aux) {
        NVTE_ERROR("softmax_aux must be provided when training. \n");
    }

    auto qkv_dtype = Int2NvteDType(qkv_type);

    // construct NVTE tensors
    TensorWrapper te_Q, te_KV, te_S, te_O, te_Bias, te_cu_seqlens_q, te_cu_seqlens_kv;
    if (qkv_dtype == DType::kBFloat16 || qkv_dtype == DType::kFloat16) {
        // BF16 or FP16
        te_Q = MakeNvteTensor(
            Q.data(),
            {static_cast<size_t>(total_seqs_q), static_cast<size_t>(h), static_cast<size_t>(d)},
            qkv_dtype);
        te_KV = MakeNvteTensor(
            KV.data(),
            {static_cast<size_t>(total_seqs_kv), 2, static_cast<size_t>(h), static_cast<size_t>(d)},
            qkv_dtype);
        te_S = MakeNvteTensor(nullptr, std::vector<size_t>{0}, DType::kFloat32);
        te_O = MakeNvteTensor(
            O.data(),
            {static_cast<size_t>(total_seqs_q), static_cast<size_t>(h), static_cast<size_t>(d)},
            qkv_dtype);
    } else {
        NVTE_ERROR("Fused attention only supports BF16/FP16 data types. \n");
    }

    if ((bias_type != "no_bias") && Bias) {
        auto bias_shape = Bias->shape();
        std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
        te_Bias = MakeNvteTensor(GetOptionalDataPtr(Bias), shape, DType::kFloat32);
    }

    te_cu_seqlens_q =
        MakeNvteTensor(cu_seqlens_q.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);
    te_cu_seqlens_kv =
        MakeNvteTensor(cu_seqlens_kv.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);

    // convert strings to enums
    NVTE_QKV_Layout qkv_layout_enum = get_nvte_qkv_layout(qkv_layout);
    NVTE_Bias_Type bias_type_enum = get_nvte_bias_type(bias_type);
    NVTE_Mask_Type attn_mask_type_enum = get_nvte_mask_type(attn_mask_type);

    auto te_rng_state = MakeNvteTensor(rng_state);

    // create auxiliary output tensors
    NVTETensorPack nvte_aux_tensor_pack;
    nvte_tensor_pack_create(&nvte_aux_tensor_pack);

    // create workspace
    TensorWrapper workspace;

    // populate tensors with appropriate shapes and dtypes
    nvte_fused_attn_fwd_kvpacked(te_Q.data(), te_KV.data(), te_Bias.data(), te_S.data(),
                                 te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                                 te_cu_seqlens_kv.data(), te_rng_state.data(), max_seqlen_q,
                                 max_seqlen_kv, is_training, attn_scale, p_dropout, qkv_layout_enum,
                                 bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

    // allocate memory for workspace and auxiliary output tensors
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), Q.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

    auto *output_s =
        reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[0]);
    output_s->data.dptr = GetOptionalDataPtr(softmax_aux);

    // execute the kernel
    nvte_fused_attn_fwd_kvpacked(te_Q.data(), te_KV.data(), te_Bias.data(), te_S.data(),
                                 te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                                 te_cu_seqlens_kv.data(), te_rng_state.data(), max_seqlen_q,
                                 max_seqlen_kv, is_training, attn_scale, p_dropout, qkv_layout_enum,
                                 bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

    // destroy tensor wrappers, but not allocated memory
    nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

// fused attention BWD with packed KV
void te_fused_attn_bwd_kvpacked(const paddle::Tensor &Q, const paddle::Tensor &KV,
                                const paddle::Tensor &cu_seqlens_q,
                                const paddle::Tensor &cu_seqlens_kv, const paddle::Tensor &O,
                                const paddle::Tensor &dO, const paddle::Tensor &softmax_aux,
                                paddle::Tensor &dQ,                       // NOLINT
                                paddle::Tensor &dKV,                      // NOLINT
                                paddle::optional<paddle::Tensor> &dBias,  // NOLINT
                                paddle::Tensor &rng_state,                // NOLINT
                                int64_t b, int64_t h, int64_t d, int64_t total_seqs_q,
                                int64_t total_seqs_kv, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                                float attn_scale, float p_dropout, const std::string &qkv_layout,
                                const std::string &bias_type, const std::string &attn_mask_type,
                                int64_t qkv_type) {
    TensorWrapper te_dBias;
    if (bias_type != "no_bias" && dBias) {
        auto bias_shape = dBias->shape();
        std::vector<size_t> shape{bias_shape.begin(), bias_shape.end()};
        te_dBias = MakeNvteTensor(GetOptionalDataPtr(dBias), shape, DType::kFloat32);
    }

    auto qkv_dtype = Int2NvteDType(qkv_type);
    // construct NVTE tensors
    TensorWrapper te_Q, te_KV, te_O, te_dO, te_S, te_dP, te_dQ, te_dKV;
    if (qkv_dtype == DType::kBFloat16 || qkv_dtype == DType::kFloat16) {
        // BF16 or FP16
        te_Q = MakeNvteTensor(Q);
        te_KV = MakeNvteTensor(KV);
        te_O = MakeNvteTensor(O);
        te_dO = MakeNvteTensor(dO);
        te_S = MakeNvteTensor(nullptr, std::vector<size_t>(0), DType::kFloat32);
        te_dP = MakeNvteTensor(nullptr, std::vector<size_t>(0), DType::kFloat32);
        te_dQ = MakeNvteTensor(dQ);
        te_dKV = MakeNvteTensor(dKV);
    } else {
        NVTE_ERROR("Fused attention only supports BF16/FP16 data types. \n");
    }

    // convert strings to enums
    NVTE_QKV_Layout qkv_layout_enum = get_nvte_qkv_layout(qkv_layout);
    NVTE_Bias_Type bias_type_enum = get_nvte_bias_type(bias_type);
    NVTE_Mask_Type attn_mask_type_enum = get_nvte_mask_type(attn_mask_type);

    // convert auxiliary tensors from forward into NVTETensors
    NVTETensorPack nvte_aux_tensor_pack;
    nvte_tensor_pack_create(&nvte_aux_tensor_pack);

    nvte_aux_tensor_pack.size = 2;
    auto *output_s = reinterpret_cast<Tensor *>(nvte_aux_tensor_pack.tensors[0]);
    auto *fwd_rng_state = reinterpret_cast<Tensor *>(nvte_aux_tensor_pack.tensors[1]);
    output_s->data.shape = std::vector<size_t>({static_cast<size_t>(b), static_cast<size_t>(h),
                                                static_cast<size_t>(max_seqlen_q),
                                                static_cast<size_t>(max_seqlen_kv)});
    output_s->data.dptr = const_cast<void *>(softmax_aux.data());
    fwd_rng_state->data.shape = std::vector<size_t>({2});
    fwd_rng_state->data.dptr = const_cast<void *>(rng_state.data());

    // create cu_seqlens tensorwrappers
    TensorWrapper te_cu_seqlens_q, te_cu_seqlens_kv;
    te_cu_seqlens_q =
        MakeNvteTensor(cu_seqlens_q.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);
    te_cu_seqlens_kv =
        MakeNvteTensor(cu_seqlens_kv.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);

    // create workspace
    TensorWrapper workspace;

    // populate tensors with appropriate shapes and dtypes
    nvte_fused_attn_bwd_kvpacked(
        te_Q.data(), te_KV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
        &nvte_aux_tensor_pack, te_dQ.data(), te_dKV.data(), te_dBias.data(), te_cu_seqlens_q.data(),
        te_cu_seqlens_kv.data(), max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout,
        qkv_layout_enum, bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

    // allocate memory for workspace
    auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), Q.place());
    workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

    // execute kernel
    nvte_fused_attn_bwd_kvpacked(
        te_Q.data(), te_KV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
        &nvte_aux_tensor_pack, te_dQ.data(), te_dKV.data(), te_dBias.data(), te_cu_seqlens_q.data(),
        te_cu_seqlens_kv.data(), max_seqlen_q, max_seqlen_kv, attn_scale, p_dropout,
        qkv_layout_enum, bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

    // destroy tensor wrappers
    nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

std::vector<paddle::Tensor> te_scaled_softmax_forward(const paddle::Tensor &input,
                                                      float scale_factor) {
    NVTE_CHECK(input.shape().size() == 4, "expected 4D tensor");
    NVTE_CHECK((input.dtype() == paddle::DataType::FLOAT16) ||
                   (input.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");

    const int batches = input.shape()[0];
    const int attn_heads = input.shape()[1];
    const int query_seq_len = input.shape()[2];
    const int key_seq_len = input.shape()[3];

    NVTE_CHECK(key_seq_len <= 4096);
    NVTE_CHECK(query_seq_len > 1);

    // Output
    auto softmax_results = paddle::empty_like(input, input.dtype(), input.place());

    auto input_cu = MakeNvteTensor(input);
    auto softmax_results_cu = MakeNvteTensor(softmax_results);

    nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(), scale_factor,
                                input.stream());

    return {softmax_results};
}

void te_scaled_softmax_backward(paddle::Tensor &output_grads,  // NOLINT
                                const paddle::Tensor &softmax_results, float scale_factor) {
    NVTE_CHECK(output_grads.shape().size() == 4, "expected 4D tensor");
    NVTE_CHECK(softmax_results.shape().size() == 4, "expected 4D tensor");

    NVTE_CHECK((output_grads.dtype() == paddle::DataType::FLOAT16) ||
                   (output_grads.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");
    NVTE_CHECK((softmax_results.dtype() == paddle::DataType::FLOAT16) ||
                   (softmax_results.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");

    auto output_grads_cu = MakeNvteTensor(output_grads);
    auto softmax_results_cu = MakeNvteTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                                 output_grads_cu.data(), scale_factor, softmax_results.stream());
}

std::vector<paddle::Tensor> te_scaled_masked_softmax_forward(const paddle::Tensor &input,
                                                             const paddle::Tensor &mask,
                                                             float scale_factor) {
    NVTE_CHECK(input.shape().size() == 4, "expected 4D tensor");
    NVTE_CHECK(mask.shape().size() == 4, "expected 4D tensor");
    NVTE_CHECK((input.dtype() == paddle::DataType::FLOAT16) ||
                   (input.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");

    const int batches = input.shape()[0];
    const int pad_batches = mask.shape()[0];
    const int attn_heads = input.shape()[1];
    const int query_seq_len = input.shape()[2];
    const int key_seq_len = input.shape()[3];

    NVTE_CHECK(key_seq_len <= 4096);
    NVTE_CHECK(query_seq_len > 1);
    NVTE_CHECK(pad_batches == 1 || pad_batches == batches);
    NVTE_CHECK(mask.shape()[1] == 1);
    NVTE_CHECK(mask.shape()[2] == query_seq_len);
    NVTE_CHECK(mask.shape()[3] == key_seq_len);

    // Output
    auto softmax_results = paddle::empty_like(input, input.dtype(), input.place());

    auto input_cu = MakeNvteTensor(input);
    auto mask_cu = MakeNvteTensor(mask);
    auto softmax_results_cu = MakeNvteTensor(softmax_results);

    nvte_scaled_masked_softmax_forward(input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
                                       scale_factor, input.stream());

    return {softmax_results};
}

void te_scaled_masked_softmax_backward(paddle::Tensor &output_grads,  // NOLINT
                                       const paddle::Tensor &softmax_results, float scale_factor) {
    NVTE_CHECK(output_grads.shape().size() == 4, "expected 4D tensor");
    NVTE_CHECK(softmax_results.shape().size() == 4, "expected 4D tensor");

    NVTE_CHECK((output_grads.dtype() == paddle::DataType::FLOAT16) ||
                   (output_grads.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");
    NVTE_CHECK((softmax_results.dtype() == paddle::DataType::FLOAT16) ||
                   (softmax_results.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");

    auto output_grads_cu = MakeNvteTensor(output_grads);
    auto softmax_results_cu = MakeNvteTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(output_grads_cu.data(), softmax_results_cu.data(),
                                 output_grads_cu.data(), scale_factor, softmax_results.stream());
}

std::vector<paddle::Tensor> te_scaled_upper_triang_masked_softmax_forward(
    const paddle::Tensor &input, float scale_factor) {
    NVTE_CHECK(input.shape().size() == 3, "expected 3D tensor");
    NVTE_CHECK((input.dtype() == paddle::DataType::FLOAT16) ||
                   (input.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");

    const int attn_batches = input.shape()[0];
    const int seq_len = input.shape()[1];
    NVTE_CHECK(seq_len <= 2048);

    // Output
    auto softmax_results = paddle::empty_like(input, input.dtype(), input.place());

    auto input_cu = MakeNvteTensor(input);
    auto softmax_results_cu = MakeNvteTensor(softmax_results);

    nvte_scaled_upper_triang_masked_softmax_forward(input_cu.data(), softmax_results_cu.data(),
                                                    scale_factor, input.stream());

    return {softmax_results};
}

void te_scaled_upper_triang_masked_softmax_backward(paddle::Tensor &output_grads,  // NOLINT
                                                    const paddle::Tensor &softmax_results,
                                                    float scale_factor) {
    NVTE_CHECK(output_grads.shape().size() == 3, "expected 3D tensor");
    NVTE_CHECK(softmax_results.shape().size() == 3, "expected 3D tensor");

    NVTE_CHECK((output_grads.dtype() == paddle::DataType::FLOAT16) ||
                   (output_grads.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");
    NVTE_CHECK((softmax_results.dtype() == paddle::DataType::FLOAT16) ||
                   (softmax_results.dtype() == paddle::DataType::BFLOAT16),
               "Only fp16 and bf16 are supported");
    NVTE_CHECK(output_grads.shape()[1] == output_grads.shape()[2]);

    auto output_grads_cu = MakeNvteTensor(output_grads);
    auto softmax_results_cu = MakeNvteTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_upper_triang_masked_softmax_backward(
        output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(), scale_factor,
        softmax_results.stream());
}

__global__ void UpdateFP8MetaKernel(const float *amax, const float *rolled_amax_history,
                                    float *amax_history, float *scale, float *scale_inv,
                                    float margin, float fp8_max, size_t history_numel,
                                    size_t amax_numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= history_numel) {
        return;
    }

    amax_history[idx] = rolled_amax_history[idx];

    if (idx < amax_numel) {
        float exp = floor(log2(fp8_max / amax[idx])) - margin;
        float sf = round(powf(2.0f, abs(exp)));
        float scale_reg = scale[idx];
        sf = ((amax[idx] > 0.0f) && isfinite(amax[idx])) ? sf : scale_reg;
        scale_reg = exp < 0.0f ? 1 / sf : sf;
        scale[idx] = scale_reg;
        scale_inv[idx] = 1.0f / scale_reg;
        amax_history[idx] = 0.0f;
    }
}

void amax_and_scale_update_inplace(paddle::Tensor &amax_history,  // NOLINT
                                   paddle::Tensor &scale,         // NOLINT
                                   paddle::Tensor &scale_inv,     // NOLINT
                                   float fp8_max, float margin, const std::string &amax_compute) {
    NVTE_CHECK(amax_compute == "max" || amax_compute == "most_recent");

    paddle::Tensor amax;

    if (amax_compute == "max") {
        amax = amax_history.max({0});
    } else {
        amax = amax_history.slice(0, 1);
    }

    const auto rolled_amax_history = amax_history.roll({-1}, {0});

    auto size = amax_history.numel();
    constexpr int BLOCK_SIZE = 256;
    size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    UpdateFP8MetaKernel<<<num_blocks, BLOCK_SIZE, 0, amax_history.stream()>>>(
        amax.data<float>(), rolled_amax_history.data<float>(), amax_history.data<float>(),
        scale.data<float>(), scale_inv.data<float>(), margin, fp8_max, amax_history.numel(),
        amax.numel());
    NVTE_CHECK_CUDA(cudaGetLastError());
}

void update_latest_amax_history_inplace(paddle::Tensor &history,  // NOLINT
                                        const paddle::Tensor &amax) {
    // Copy amax to history[0]
    NVTE_CHECK_CUDA(cudaMemcpyAsync(history.data(), amax.data(),
                                    amax.numel() * SizeOf(amax.dtype()), cudaMemcpyDeviceToDevice,
                                    amax.stream()));
}

}  // namespace paddle_ext
}  // namespace transformer_engine

PD_BUILD_OP(te_gemm)
    .Inputs({"A", paddle::Optional("A_scale_inverse"), "B", paddle::Optional("B_scale_inverse"),
             paddle::Optional("bias"), "_D", paddle::Optional("_D_scale"),
             paddle::Optional("_D_amax"), paddle::Optional("_pre_gelu_out"), "_workspace"})
    .Outputs({"D", paddle::Optional("D_scale"), paddle::Optional("D_amax"),
              paddle::Optional("pre_gelu_out"), "workspace"})
    .Attrs({"A_index: int64_t", "B_index: int64_t", "D_index: int64_t", "A_type: int64_t",
            "B_type: int64_t", "D_type: int64_t", "bias_type: int64_t", "transa: bool",
            "transb: bool", "grad: bool", "workspace_size: int64_t", "accumulate: bool",
            "use_split_accumulator: bool", "math_sm_count: int64_t"})
    .SetInplaceMap({{"_D", "D"},
                    {paddle::Optional("_D_scale"), paddle::Optional("D_scale")},
                    {paddle::Optional("_D_amax"), paddle::Optional("D_amax")},
                    {paddle::Optional("_pre_gelu_out"), paddle::Optional("pre_gelu_out")},
                    {"_workspace", "workspace"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_gemm));

PD_BUILD_OP(cast_to_fp8)
    .Inputs({"Input", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "Amax", "ScaleInv"})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::cast_to_fp8));

PD_BUILD_OP(cast_from_fp8)
    .Inputs({"Input", "ScaleInv"})
    .Outputs({"Output"})
    .Attrs({"index: int64_t", "itype: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::cast_from_fp8));

PD_BUILD_OP(te_transpose)
    .Inputs({"Input"})
    .Outputs({"Output"})
    .Attrs({"otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_transpose));

PD_BUILD_OP(te_cast_transpose)
    .Inputs({"Input", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"CastedOutput", "TransposedOutput", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_cast_transpose));

PD_BUILD_OP(te_cast_transpose_bgrad)
    .Inputs({"GradOutput", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"dBias", "CastedOutput", "TransposedOutput", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_cast_transpose_bgrad));

PD_BUILD_OP(te_gelu_fp8)
    .Inputs({"Input", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_gelu_fp8));

PD_BUILD_OP(te_gelu)
    .Inputs({"Input"})
    .Outputs({"Output"})
    .Attrs({"otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_gelu));

PD_BUILD_OP(te_cast_transpose_bgrad_dgelu)
    .Inputs({"GradOutput", "GeluInput", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"CastedDgelu", "TransposedDgelu", "Dbias", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_cast_transpose_bgrad_dgelu));

PD_BUILD_OP(te_layernorm_fwd_fp8)
    .Inputs({"Input", "Weight", "Bias", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "Mu", "Rsigma", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"eps: float", "index: int64_t", "otype: int64_t", "sm_margin: int64_t",
            "zero_centered_gamma: bool"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_layernorm_fwd_fp8));

PD_BUILD_OP(te_layernorm_fwd)
    .Inputs({"Input", "Weight", "Bias"})
    .Outputs({"Output", "Mu", "Rsigma"})
    .Attrs({"eps: float", "otype: int64_t", "sm_margin: int64_t", "zero_centered_gamma: bool"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_layernorm_fwd));

PD_BUILD_OP(te_layernorm_bwd)
    .Inputs({"Dz", "X", "Mu", "Rsigma", "Gamma"})
    .Outputs({"Dx", "Dgamma", "Dbeta"})
    .Attrs({"sm_margin: int64_t", "zero_centered_gamma: bool"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_layernorm_bwd));

PD_BUILD_OP(te_rmsnorm_fwd)
    .Inputs({"Input", "Weight"})
    .Outputs({"Output", "InvVariance"})
    .Attrs({"eps: float", "otype: int64_t", "sm_margin: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_rmsnorm_fwd));

PD_BUILD_OP(te_rmsnorm_fwd_fp8)
    .Inputs({"Input", "Weight", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "InvVariance", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"eps: float", "index: int64_t", "otype: int64_t", "sm_margin: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_rmsnorm_fwd_fp8));

PD_BUILD_OP(te_rmsnorm_bwd)
    .Inputs({"Dz", "X", "Rsigma", "Gamma"})
    .Outputs({"Dx", "Dgamma"})
    .Attrs({"sm_margin: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_rmsnorm_bwd));

PD_BUILD_OP(te_fused_attn_fwd_qkvpacked)
    .Inputs({"QKV", "cu_seqlens", paddle::Optional("Bias"), "_O", paddle::Optional("_softmax_aux"),
             "rng_state"})
    .Outputs({"O", paddle::Optional("softmax_aux")})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "total_seqs: int64_t", "max_seqlen: int64_t",
            "is_training: bool", "attn_scale: float", "p_dropout: float", "qkv_layout: std::string",
            "bias_type: std::string", "attn_mask_type: std::string", "qkv_type: int64_t"})
    .SetInplaceMap({{"_O", "O"},
                    {paddle::Optional("_softmax_aux"), paddle::Optional("softmax_aux")}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_fused_attn_fwd_qkvpacked));

PD_BUILD_OP(te_fused_attn_bwd_qkvpacked)
    .Inputs({"QKV", "cu_seqlens", "O", "dO", "softmax_aux", "_dQKV", paddle::Optional("_dBias"),
             "rng_state"})
    .Outputs({"dQKV", paddle::Optional("dBias")})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "total_seqs: int64_t", "max_seqlen: int64_t",
            "attn_scale: float", "p_dropout: float", "qkv_layout: std::string",
            "bias_type: std::string", "attn_mask_type: std::string", "qkv_type: int64_t"})
    .SetInplaceMap({{"_dQKV", "dQKV"}, {paddle::Optional("_dBias"), paddle::Optional("dBias")}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_fused_attn_bwd_qkvpacked));

PD_BUILD_OP(te_fused_attn_fwd_kvpacked)
    .Inputs({"Q", "KV", "cu_seqlens_q", "cu_seqlens_kv", paddle::Optional("Bias"), "_O",
             paddle::Optional("_softmax_aux"), "rng_state"})
    .Outputs({"O", paddle::Optional("softmax_aux")})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "total_seqs_q: int64_t",
            "total_seqs_kv: int64_t", "max_seqlen_q: int64_t", "max_seqlen_kv: int64_t",
            "is_training: bool", "attn_scale: float", "p_dropout: float", "qkv_layout: std::string",
            "bias_type: std::string", "attn_mask_type: std::string", "qkv_type: int64_t"})
    .SetInplaceMap({{"_O", "O"},
                    {paddle::Optional("_softmax_aux"), paddle::Optional("softmax_aux")}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_fused_attn_fwd_kvpacked));

PD_BUILD_OP(te_fused_attn_bwd_kvpacked)
    .Inputs({"Q", "KV", "cu_seqlens_q", "cu_seqlens_kv", "O", "dO", "softmax_aux", "_dQ", "_dKV",
             paddle::Optional("_dBias"), "rng_state"})
    .Outputs({"dQ", "dKV", paddle::Optional("dBias")})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "total_seqs_q: int64_t",
            "total_seqs_kv: int64_t", "max_seqlen_q: int64_t", "max_seqlen_kv: int64_t",
            "attn_scale: float", "p_dropout: float", "qkv_layout: std::string",
            "bias_type: std::string", "attn_mask_type: std::string", "qkv_type: int64_t"})
    .SetInplaceMap({{"_dQ", "dQ"},
                    {"_dKV", "dKV"},
                    {paddle::Optional("_dBias"), paddle::Optional("dBias")}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_fused_attn_bwd_kvpacked));

PD_BUILD_OP(te_scaled_softmax_forward)
    .Inputs({"input"})
    .Outputs({"softmax_results"})
    .Attrs({"scale_factor: float"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_scaled_softmax_forward));

PD_BUILD_OP(te_scaled_softmax_backward)
    .Inputs({"out_grad_", "softmax_results"})
    .Outputs({"out_grad"})
    .Attrs({"scale_factor: float"})
    .SetInplaceMap({{"out_grad_", "out_grad"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_scaled_softmax_backward));

PD_BUILD_OP(te_scaled_masked_softmax_forward)
    .Inputs({"input", "mask"})
    .Outputs({"softmax_results"})
    .Attrs({"scale_factor: float"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_scaled_masked_softmax_forward));

PD_BUILD_OP(te_scaled_masked_softmax_backward)
    .Inputs({"out_grad_", "softmax_results"})
    .Outputs({"out_grad"})
    .Attrs({"scale_factor: float"})
    .SetInplaceMap({{"out_grad_", "out_grad"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_scaled_masked_softmax_backward));

PD_BUILD_OP(te_scaled_upper_triang_masked_softmax_forward)
    .Inputs({"input"})
    .Outputs({"softmax_results"})
    .Attrs({"scale_factor: float"})
    .SetKernelFn(
        PD_KERNEL(transformer_engine::paddle_ext::te_scaled_upper_triang_masked_softmax_forward));

PD_BUILD_OP(te_scaled_upper_triang_masked_softmax_backward)
    .Inputs({"out_grad_", "softmax_results"})
    .Outputs({"out_grad"})
    .Attrs({"scale_factor: float"})
    .SetInplaceMap({{"out_grad_", "out_grad"}})
    .SetKernelFn(
        PD_KERNEL(transformer_engine::paddle_ext::te_scaled_upper_triang_masked_softmax_backward));

PD_BUILD_OP(amax_and_scale_update_inplace)
    .Inputs({"_amax_history", "_scale", "_scale_inv"})
    .Outputs({"amax_history", "scale", "scale_inv"})
    .SetInplaceMap({{"_amax_history", "amax_history"},
                    {"_scale", "scale"},
                    {"_scale_inv", "scale_inv"}})
    .Attrs({"fp8_max: float", "margin: float", "amax_compute: std::string"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::amax_and_scale_update_inplace));

PD_BUILD_OP(update_latest_amax_history_inplace)
    .Inputs({"_history", "amax"})
    .Outputs({"history"})
    .SetInplaceMap({{"_history", "history"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::update_latest_amax_history_inplace));
