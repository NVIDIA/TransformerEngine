/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <vector>
#include "common.h"
namespace transformer_engine {
namespace paddle_ext {

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
