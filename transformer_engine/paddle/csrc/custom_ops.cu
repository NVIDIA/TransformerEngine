/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cub/cub.cuh>
#include <map>
#include <vector>

#include "common.h"
#include "common/common.h"

namespace transformer_engine {
namespace paddle_ext {

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

void cast_to_fp8(const paddle::Tensor &input, const paddle::Tensor &scale,
                 paddle::Tensor &output,     // NOLINT
                 paddle::Tensor &amax,       // NOLINT
                 paddle::Tensor &scale_inv,  // NOLINT
                 int64_t index, int64_t otype) {
  auto shape = GetShapeArray(input);

  auto input_cu = MakeNvteTensor(input);
  auto output_cu = MakeNvteTensor(
      output.data(), shape, Int2NvteDType(otype), GetDataPtr<float>(amax, index),
      const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), input.stream());
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

void te_cast_transpose(const paddle::Tensor &input, const paddle::Tensor &scale,
                       paddle::Tensor &output_cast,       // NOLINT
                       paddle::Tensor &output_transpose,  // NOLINT
                       paddle::Tensor &amax,              // NOLINT
                       paddle::Tensor &scale_inv,         // NOLINT
                       int64_t index, int64_t otype) {
  auto shape = GetShapeArray(input);
  NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");

  size_t M = shape[0];
  size_t N = shape[1];

  auto input_cu = MakeNvteTensor(input);
  void *amax_data = GetDataPtr<float>(amax, index);
  void *scale_data = const_cast<void *>(GetDataPtr<float>(scale, index));
  void *scale_inv_data = GetDataPtr<float>(scale_inv, index);
  auto output_cast_cu = MakeNvteTensor(output_cast.data(), {M, N}, Int2NvteDType(otype), amax_data,
                                       scale_data, scale_inv_data);
  auto output_transpose_cu = MakeNvteTensor(output_transpose.data(), {N, M}, Int2NvteDType(otype),
                                            amax_data, scale_data, scale_inv_data);

  nvte_cast_transpose(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                      input.stream());
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
  auto grad_output_cast =
      paddle::empty_like(grad_output, Nvte2PaddleDType(Int2NvteDType(otype)), grad_output.place());
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

  DType gelu_dtype = pre_gelu_out ? Paddle2NvteDType(pre_gelu_out->dtype()) : Int2NvteDType(D_type);
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

std::vector<paddle::Tensor> te_swiglu(const paddle::Tensor &input, int64_t otype) {
  auto shape = GetShapeArray(input);
  NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");

  size_t M = shape[0];
  size_t N = shape[1];

  auto output = paddle::empty({input.shape()[0], input.shape()[1] / 2},
                              Nvte2PaddleDType(Int2NvteDType(otype)), input.place());

  auto input_cu = MakeNvteTensor(input);
  auto output_cu = MakeNvteTensor(output.data(), GetShapeArray(output), Int2NvteDType(otype));

  nvte_swiglu(input_cu.data(), output_cu.data(), input.stream());

  return {output};
}

std::vector<paddle::Tensor> te_swiglu_fp8(const paddle::Tensor &input, const paddle::Tensor &scale,
                                          paddle::Tensor &amax,       // NOLINT
                                          paddle::Tensor &scale_inv,  // NOLINT
                                          int64_t index, int64_t otype) {
  auto shape = GetShapeArray(input);
  NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");

  size_t M = shape[0];
  size_t N = shape[1];

  auto output = paddle::empty({input.shape()[0], input.shape()[1] / 2},
                              Nvte2PaddleDType(Int2NvteDType(otype)), input.place());

  auto input_cu = MakeNvteTensor(input);
  auto output_cu = MakeNvteTensor(
      output.data(), GetShapeArray(output), Int2NvteDType(otype), GetDataPtr<float>(amax, index),
      const_cast<void *>(GetDataPtr<float>(scale, index)), GetDataPtr<float>(scale_inv, index));

  nvte_swiglu(input_cu.data(), output_cu.data(), input.stream());

  return {output};
}

std::vector<paddle::Tensor> te_dswiglu(const paddle::Tensor &grad, const paddle::Tensor &input,
                                       int64_t otype) {
  auto shape = GetShapeArray(input);
  NVTE_CHECK(shape.size() == 2, "Expect the input to have 2 dimensions.");

  size_t M = shape[0];
  size_t N = shape[1];

  auto output = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)), input.place());

  auto input_cu = MakeNvteTensor(input.data(), {M, N}, Paddle2NvteDType(input.dtype()));
  auto grad_cu = MakeNvteTensor(grad.data(), {M, N / 2}, Paddle2NvteDType(grad.dtype()));
  auto output_cu = MakeNvteTensor(output.data(), {M, N}, Paddle2NvteDType(output.dtype()));

  nvte_dswiglu(grad_cu.data(), input_cu.data(), output_cu.data(), input.stream());

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

  auto dgelu = paddle::empty_like(grad_output, Nvte2PaddleDType(DType::kByte), grad_output.place());

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

  auto ln_out = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)), input.place());
  auto mu = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
  auto rsigma = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
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
  auto rsigma = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
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
  bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(), dz.stream(),
          num_sm - sm_margin, workspace.data(), barrier.data());

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
  bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(), dz.stream(),
          num_sm - sm_margin, workspace.data(), barrier.data());

  return {dx, dgamma, dbeta};
}

std::vector<paddle::Tensor> te_rmsnorm_fwd(const paddle::Tensor &input,
                                           const paddle::Tensor &weight, float eps, int64_t otype,
                                           int64_t sm_margin, bool zero_centered_gamma) {
  NVTE_CHECK(zero_centered_gamma == false, "zero_centered_gamma is not supported yet for RMSNorm.");
  auto shape = GetShapeArray(input);
  NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

  size_t N = shape[0];
  size_t H = shape[1];

  auto ln_out = paddle::empty_like(input, input.dtype(), input.place());
  auto rsigma = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
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
                                               int64_t sm_margin, bool zero_centered_gamma) {
  NVTE_CHECK(zero_centered_gamma == false, "zero_centered_gamma is not supported yet for RMSNorm.");
  auto shape = GetShapeArray(input);
  NVTE_CHECK(shape.size() == 2, "Expect the grad_output to have 2 dimensions.");

  size_t N = shape[0];
  size_t H = shape[1];

  auto ln_out = paddle::empty_like(input, Nvte2PaddleDType(Int2NvteDType(otype)), input.place());
  auto rsigma = paddle::empty({static_cast<int64_t>(N)}, paddle::DataType::FLOAT32, input.place());
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
                                           const paddle::Tensor &gamma, int64_t sm_margin,
                                           bool zero_centered_gamma) {
  NVTE_CHECK(zero_centered_gamma == false, "zero_centered_gamma is not supported yet for RMSNorm.");
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

__global__ void set_rng_state(std::pair<uint64_t, uint64_t> seed_offset, int64_t *rng_state_ptr) {
  rng_state_ptr[0] = static_cast<int64_t>(seed_offset.first);
  rng_state_ptr[1] = static_cast<int64_t>(seed_offset.second);
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
                                 const int64_t qkv_type, int64_t rng_elts_per_thread) {
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
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(QKV.place());
  auto gen_cuda = dev_ctx->GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(rng_elts_per_thread);
  set_rng_state<<<1, 1, 0, QKV.stream()>>>(seed_offset, static_cast<int64_t *>(rng_state.data()));

  auto te_rng_state = MakeNvteTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  auto dummy_seq_offsets = TensorWrapper(nullptr, {static_cast<size_t>(b + 1)}, DType::kInt32);
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd_qkvpacked(te_QKV.data(), te_Bias.data(), te_S.data(), te_O.data(),
                                &nvte_aux_tensor_pack, te_cu_seqlens.data(),
                                dummy_seq_offsets.data(), te_rng_state.data(), max_seqlen,
                                is_training, attn_scale, p_dropout, qkv_layout_enum, bias_type_enum,
                                attn_mask_type_enum, workspace.data(), QKV.stream());

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), QKV.place());
  workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

  auto *output_s = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[0]);
  output_s->data.dptr = GetOptionalDataPtr(softmax_aux);

  // execute the kernel
  nvte_fused_attn_fwd_qkvpacked(te_QKV.data(), te_Bias.data(), te_S.data(), te_O.data(),
                                &nvte_aux_tensor_pack, te_cu_seqlens.data(),
                                dummy_seq_offsets.data(), te_rng_state.data(), max_seqlen,
                                is_training, attn_scale, p_dropout, qkv_layout_enum, bias_type_enum,
                                attn_mask_type_enum, workspace.data(), QKV.stream());

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

  auto dummy_seq_offsets = TensorWrapper(nullptr, {static_cast<size_t>(b + 1)}, DType::kInt32);
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd_qkvpacked(te_QKV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
                                &nvte_aux_tensor_pack, te_dQKV.data(), te_dBias.data(),
                                te_cu_seqlens.data(), dummy_seq_offsets.data(), max_seqlen,
                                attn_scale, p_dropout, qkv_layout_enum, bias_type_enum,
                                attn_mask_type_enum, workspace.data(), QKV.stream());

  // allocate memory for workspace
  auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), QKV.place());
  workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

  // execute kernel
  nvte_fused_attn_bwd_qkvpacked(te_QKV.data(), te_O.data(), te_dO.data(), te_S.data(), te_dP.data(),
                                &nvte_aux_tensor_pack, te_dQKV.data(), te_dBias.data(),
                                te_cu_seqlens.data(), dummy_seq_offsets.data(), max_seqlen,
                                attn_scale, p_dropout, qkv_layout_enum, bias_type_enum,
                                attn_mask_type_enum, workspace.data(), QKV.stream());

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

void te_fused_attn_fwd_kvpacked(
    const paddle::Tensor &Q, const paddle::Tensor &KV, const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &cu_seqlens_kv, const paddle::optional<paddle::Tensor> &Bias,
    paddle::Tensor &O,                              // NOLINT
    paddle::optional<paddle::Tensor> &softmax_aux,  // NOLINT
    paddle::Tensor &rng_state,                      // NOLINT
    int64_t b, int64_t h, int64_t d, int64_t total_seqs_q, int64_t total_seqs_kv,
    int64_t max_seqlen_q, int64_t max_seqlen_kv, bool is_training, float attn_scale,
    float p_dropout, const std::string &qkv_layout, const std::string &bias_type,
    const std::string &attn_mask_type, const int64_t qkv_type, int64_t rng_elts_per_thread) {
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

  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(Q.place());
  auto gen_cuda = dev_ctx->GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(rng_elts_per_thread);
  set_rng_state<<<1, 1, 0, Q.stream()>>>(seed_offset, static_cast<int64_t *>(rng_state.data()));
  auto te_rng_state = MakeNvteTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  auto dummy_seq_offsets = TensorWrapper(nullptr, {static_cast<size_t>(b + 1)}, DType::kInt32);
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd_kvpacked(te_Q.data(), te_KV.data(), te_Bias.data(), te_S.data(), te_O.data(),
                               &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                               te_cu_seqlens_kv.data(), dummy_seq_offsets.data(),
                               dummy_seq_offsets.data(), te_rng_state.data(), max_seqlen_q,
                               max_seqlen_kv, is_training, attn_scale, p_dropout, qkv_layout_enum,
                               bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), Q.place());
  workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

  auto *output_s = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[0]);
  output_s->data.dptr = GetOptionalDataPtr(softmax_aux);

  // execute the kernel
  nvte_fused_attn_fwd_kvpacked(te_Q.data(), te_KV.data(), te_Bias.data(), te_S.data(), te_O.data(),
                               &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                               te_cu_seqlens_kv.data(), dummy_seq_offsets.data(),
                               dummy_seq_offsets.data(), te_rng_state.data(), max_seqlen_q,
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
  output_s->data.shape =
      std::vector<size_t>({static_cast<size_t>(b), static_cast<size_t>(h),
                           static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv)});
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

  auto dummy_seq_offsets = TensorWrapper(nullptr, {static_cast<size_t>(b + 1)}, DType::kInt32);
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd_kvpacked(te_Q.data(), te_KV.data(), te_O.data(), te_dO.data(), te_S.data(),
                               te_dP.data(), &nvte_aux_tensor_pack, te_dQ.data(), te_dKV.data(),
                               te_dBias.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
                               dummy_seq_offsets.data(), dummy_seq_offsets.data(), max_seqlen_q,
                               max_seqlen_kv, attn_scale, p_dropout, qkv_layout_enum,
                               bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

  // allocate memory for workspace
  auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), Q.place());
  workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

  // execute kernel
  nvte_fused_attn_bwd_kvpacked(te_Q.data(), te_KV.data(), te_O.data(), te_dO.data(), te_S.data(),
                               te_dP.data(), &nvte_aux_tensor_pack, te_dQ.data(), te_dKV.data(),
                               te_dBias.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
                               dummy_seq_offsets.data(), dummy_seq_offsets.data(), max_seqlen_q,
                               max_seqlen_kv, attn_scale, p_dropout, qkv_layout_enum,
                               bias_type_enum, attn_mask_type_enum, workspace.data(), Q.stream());

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

void te_fused_attn_fwd(const paddle::Tensor &Q, const paddle::Tensor &K, const paddle::Tensor &V,
                       const paddle::Tensor &cu_seqlens_q, const paddle::Tensor &cu_seqlens_kv,
                       const paddle::optional<paddle::Tensor> &Bias,
                       paddle::Tensor &O,                              // NOLINT
                       paddle::optional<paddle::Tensor> &softmax_aux,  // NOLINT
                       paddle::Tensor &rng_state,                      // NOLINT
                       int64_t b, int64_t h, int64_t d, int64_t max_seqlen_q, int64_t max_seqlen_kv,
                       bool is_training, float attn_scale, float p_dropout,
                       const std::string &qkv_layout, const std::string &bias_type,
                       const std::string &attn_mask_type, const int64_t qkv_type,
                       int64_t rng_elts_per_thread) {
  if (is_training && !softmax_aux) {
    NVTE_ERROR("softmax_aux must be provided when training. \n");
  }

  auto qkv_dtype = Int2NvteDType(qkv_type);
  // construct NVTE tensors
  TensorWrapper te_Q, te_K, te_V, te_S, te_O, te_Bias, te_cu_seqlens_q, te_cu_seqlens_kv;
  if (qkv_dtype == DType::kBFloat16 || qkv_dtype == DType::kFloat16) {
    // BF16 or FP16
    te_Q = MakeNvteTensor(Q);
    te_K = MakeNvteTensor(K);
    te_V = MakeNvteTensor(V);
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
  te_cu_seqlens_q =
      MakeNvteTensor(cu_seqlens_q.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);
  te_cu_seqlens_kv =
      MakeNvteTensor(cu_seqlens_kv.data(), {static_cast<size_t>(b + 1)}, DType::kInt32);

  // convert strings to enums
  NVTE_QKV_Layout qkv_layout_enum = get_nvte_qkv_layout(qkv_layout);
  NVTE_Bias_Type bias_type_enum = get_nvte_bias_type(bias_type);
  NVTE_Mask_Type attn_mask_type_enum = get_nvte_mask_type(attn_mask_type);

  // extract random number generator seed and offset
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(Q.place());
  auto gen_cuda = dev_ctx->GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(rng_elts_per_thread);
  set_rng_state<<<1, 1, 0, Q.stream()>>>(seed_offset, static_cast<int64_t *>(rng_state.data()));

  auto te_rng_state = MakeNvteTensor(rng_state);

  // create auxiliary output tensors
  NVTETensorPack nvte_aux_tensor_pack;
  nvte_tensor_pack_create(&nvte_aux_tensor_pack);

  // create workspace
  TensorWrapper workspace;

  auto dummy_seq_offsets = TensorWrapper(nullptr, {static_cast<size_t>(b + 1)}, DType::kInt32);
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_fwd(te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_S.data(),
                      te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                      te_cu_seqlens_kv.data(), dummy_seq_offsets.data(), dummy_seq_offsets.data(),
                      te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training, attn_scale,
                      p_dropout, qkv_layout_enum, bias_type_enum, attn_mask_type_enum,
                      workspace.data(), Q.stream());

  // allocate memory for workspace and auxiliary output tensors
  auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), Q.place());

  workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

  auto *output_s = reinterpret_cast<transformer_engine::Tensor *>(nvte_aux_tensor_pack.tensors[0]);
  output_s->data.dptr = GetOptionalDataPtr(softmax_aux);

  // execute the kernel
  nvte_fused_attn_fwd(te_Q.data(), te_K.data(), te_V.data(), te_Bias.data(), te_S.data(),
                      te_O.data(), &nvte_aux_tensor_pack, te_cu_seqlens_q.data(),
                      te_cu_seqlens_kv.data(), dummy_seq_offsets.data(), dummy_seq_offsets.data(),
                      te_rng_state.data(), max_seqlen_q, max_seqlen_kv, is_training, attn_scale,
                      p_dropout, qkv_layout_enum, bias_type_enum, attn_mask_type_enum,
                      workspace.data(), Q.stream());

  // destroy tensor wrappers, but not allocated memory
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

void te_fused_attn_bwd(const paddle::Tensor &Q, const paddle::Tensor &K, const paddle::Tensor &V,
                       const paddle::Tensor &cu_seqlens_q, const paddle::Tensor &cu_seqlens_kv,
                       const paddle::Tensor &O, const paddle::Tensor &dO,
                       const paddle::Tensor &softmax_aux,
                       paddle::Tensor &dQ,                       // NOLINT
                       paddle::Tensor &dK,                       // NOLINT
                       paddle::Tensor &dV,                       // NOLINT
                       paddle::optional<paddle::Tensor> &dBias,  // NOLINT
                       paddle::Tensor &rng_state,                // NOLINT
                       int64_t b, int64_t h, int64_t d, int64_t max_seqlen_q, int64_t max_seqlen_kv,
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
  TensorWrapper te_Q, te_K, te_V, te_O, te_dO, te_S, te_dP, te_dQ, te_dK, te_dV;
  if (qkv_dtype == DType::kBFloat16 || qkv_dtype == DType::kFloat16) {
    // BF16 or FP16
    te_Q = MakeNvteTensor(Q);
    te_K = MakeNvteTensor(K);
    te_V = MakeNvteTensor(V);
    te_O = MakeNvteTensor(O);
    te_dO = MakeNvteTensor(dO);
    te_S = MakeNvteTensor(nullptr, std::vector<size_t>(0), DType::kFloat32);
    te_dP = MakeNvteTensor(nullptr, std::vector<size_t>(0), DType::kFloat32);
    te_dQ = MakeNvteTensor(dQ);
    te_dK = MakeNvteTensor(dK);
    te_dV = MakeNvteTensor(dV);
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
  output_s->data.shape =
      std::vector<size_t>({static_cast<size_t>(b), static_cast<size_t>(h),
                           static_cast<size_t>(max_seqlen_q), static_cast<size_t>(max_seqlen_kv)});
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

  auto dummy_seq_offsets = TensorWrapper(nullptr, {static_cast<size_t>(b + 1)}, DType::kInt32);
  // populate tensors with appropriate shapes and dtypes
  nvte_fused_attn_bwd(te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(),
                      te_dP.data(), &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(),
                      te_dBias.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
                      dummy_seq_offsets.data(), dummy_seq_offsets.data(), max_seqlen_q,
                      max_seqlen_kv, attn_scale, p_dropout, qkv_layout_enum, bias_type_enum,
                      attn_mask_type_enum, workspace.data(), Q.stream());

  // allocate memory for workspace
  auto workspace_data = AllocateSpace(workspace.shape(), workspace.dtype(), Q.place());
  workspace = MakeNvteTensor(workspace_data.data(), workspace.shape(), workspace.dtype());

  // execute kernel
  nvte_fused_attn_bwd(te_Q.data(), te_K.data(), te_V.data(), te_O.data(), te_dO.data(), te_S.data(),
                      te_dP.data(), &nvte_aux_tensor_pack, te_dQ.data(), te_dK.data(), te_dV.data(),
                      te_dBias.data(), te_cu_seqlens_q.data(), te_cu_seqlens_kv.data(),
                      dummy_seq_offsets.data(), dummy_seq_offsets.data(), max_seqlen_q,
                      max_seqlen_kv, attn_scale, p_dropout, qkv_layout_enum, bias_type_enum,
                      attn_mask_type_enum, workspace.data(), Q.stream());

  // destroy tensor wrappers
  nvte_tensor_pack_destroy(&nvte_aux_tensor_pack);
}

std::vector<paddle::Tensor> te_scaled_softmax_forward(const paddle::Tensor &input,
                                                      float scale_factor) {
  NVTE_CHECK(input.shape().size() == 4, "expected 4D tensor");
  NVTE_CHECK(
      (input.dtype() == paddle::DataType::FLOAT16) || (input.dtype() == paddle::DataType::BFLOAT16),
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
  NVTE_CHECK(
      (input.dtype() == paddle::DataType::FLOAT16) || (input.dtype() == paddle::DataType::BFLOAT16),
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
  NVTE_CHECK(
      (input.dtype() == paddle::DataType::FLOAT16) || (input.dtype() == paddle::DataType::BFLOAT16),
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

constexpr int BLOCK_SIZE = 512;

void amax_and_scale_update_inplace(paddle::Tensor &amax_history,  // NOLINT
                                   paddle::Tensor &scale,         // NOLINT
                                   paddle::Tensor &scale_inv,     // NOLINT
                                   const paddle::Tensor &non_weight_mask, int64_t fp8_dtype,
                                   float margin, const std::string &amax_compute) {
  auto amax_history_ = MakeNvteTensor(amax_history);
  auto scale_ = MakeNvteTensor(scale);
  auto scale_inv_ = MakeNvteTensor(scale_inv);
  const auto non_weight_mask_ = MakeNvteTensor(non_weight_mask);
  nvte_delayed_scaling_recipe_amax_and_scale_update(
      amax_history_.data(), scale_.data(), scale_inv_.data(), non_weight_mask_.data(),
      amax_history_.data(), scale_.data(), scale_inv_.data(), amax_compute.c_str(),
      static_cast<NVTEDType>(fp8_dtype), margin, amax_history.stream());
}

void update_latest_amax_history_inplace(paddle::Tensor &history,  // NOLINT
                                        const paddle::Tensor &amax) {
  // Copy amax to history[0]
  NVTE_CHECK_CUDA(cudaMemcpyAsync(history.data(), amax.data(), amax.numel() * SizeOf(amax.dtype()),
                                  cudaMemcpyDeviceToDevice, amax.stream()));
}

__global__ __launch_bounds__(BLOCK_SIZE) void mask_to_actual_seqlens_kernel(
    const bool *mask, int32_t *q_actual_seqlen, int32_t *kv_actual_seqlen, int q_seqlen,
    int kv_seqlen, bool need_kv) {
  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage q_smem;
  __shared__ typename BlockReduce::TempStorage kv_smem;
  unsigned int tid = threadIdx.x;
  unsigned int batch_offset = blockIdx.x * q_seqlen * kv_seqlen;

  // load mask, convert to 1/0, do accumulation
  int q = 0, kv = 0;
  for (unsigned int q_idx = tid * kv_seqlen; q_idx < q_seqlen * kv_seqlen;
       q_idx += BLOCK_SIZE * kv_seqlen) {
    q += (mask[q_idx + batch_offset] ? 0 : 1);
  }

  if (need_kv) {
    for (unsigned int kv_idx = tid; kv_idx < kv_seqlen; kv_idx += BLOCK_SIZE) {
      kv += (mask[kv_idx + batch_offset] ? 0 : 1);
    }
  }
  __syncthreads();

  // compute cub::BlockReduce
  int q_sum, kv_sum;
  q_sum = BlockReduce(q_smem).Sum(q);
  if (need_kv) kv_sum = BlockReduce(kv_smem).Sum(kv);

  // write result for this block to global mem
  if (tid == 0) {
    q_actual_seqlen[blockIdx.x + 1] = q_sum;
    if (need_kv) {
      kv_actual_seqlen[blockIdx.x + 1] = kv_sum;
    }
  }
}

__global__ __launch_bounds__(BLOCK_SIZE) void block_prefix_sum_inplace(int32_t *x, int n) {
  typedef cub::BlockScan<int32_t, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage smem;
  // +1 to ignore the first element
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;

  // load data
  int32_t thread_data[1];
  thread_data[0] = i < n ? x[i] : 0;
  __syncthreads();

  // CUB block prefix sum
  BlockScan(smem).InclusiveSum(thread_data, thread_data);
  __syncthreads();

  // write result
  if (i < n) {
    x[i] = thread_data[0];
  }
}

void mask_to_cu_seqlens(const paddle::Tensor &mask,
                        paddle::Tensor &q_cu_seqlen,                     // NOLINT
                        paddle::optional<paddle::Tensor> &kv_cu_seqlen,  // NOLINT
                        int q_seqlen, int kv_seqlen, bool need_kv) {
  if (need_kv) {
    NVTE_CHECK(GetOptionalDataPtr(kv_cu_seqlen) != nullptr,
               "kv_cu_seqlen must be provided when need_kv is true");
  }
  mask_to_actual_seqlens_kernel<<<mask.shape()[0], BLOCK_SIZE, 0, mask.stream()>>>(
      mask.data<bool>(), q_cu_seqlen.data<int32_t>(),
      reinterpret_cast<int32_t *>(GetOptionalDataPtr(kv_cu_seqlen)), q_seqlen, kv_seqlen, need_kv);
  // q_cu_seqlen shape: [bs+1], assume bs is not too large (<=512), so we can use a single block
  // to do prefix sum
  NVTE_CHECK(q_cu_seqlen.numel() - 1 <= BLOCK_SIZE, "batch size too large, kernel may fail");
  block_prefix_sum_inplace<<<1, BLOCK_SIZE, 0, mask.stream()>>>(q_cu_seqlen.data<int32_t>(),
                                                                q_cu_seqlen.numel());
  if (need_kv) {
    block_prefix_sum_inplace<<<1, BLOCK_SIZE, 0, mask.stream()>>>(
        reinterpret_cast<int32_t *>(GetOptionalDataPtr(kv_cu_seqlen)), kv_cu_seqlen->numel());
  }
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
    .Inputs({"Input", "Scale", "_Output", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "Amax", "ScaleInv"})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetInplaceMap({{"_Output", "Output"}, {"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
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
    .Inputs({"Input", "Scale", "_CastedOutput", "_TransposedOutput", "_Amax", "_ScaleInv"})
    .Outputs({"CastedOutput", "TransposedOutput", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_CastedOutput", "CastedOutput"},
                    {"_TransposedOutput", "TransposedOutput"},
                    {"_Amax", "Amax"},
                    {"_ScaleInv", "ScaleInv"}})
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

PD_BUILD_OP(te_swiglu)
    .Inputs({"Input"})
    .Outputs({"Output"})
    .Attrs({"otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_swiglu));

PD_BUILD_OP(te_swiglu_fp8)
    .Inputs({"Input", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"index: int64_t", "otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_swiglu_fp8));

PD_BUILD_OP(te_dswiglu)
    .Inputs({"Grad", "Input"})
    .Outputs({"Output"})
    .Attrs({"otype: int64_t"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_dswiglu));

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
    .Attrs({"eps: float", "otype: int64_t", "sm_margin: int64_t", "zero_centered_gamma: bool"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_rmsnorm_fwd));

PD_BUILD_OP(te_rmsnorm_fwd_fp8)
    .Inputs({"Input", "Weight", "Scale", "_Amax", "_ScaleInv"})
    .Outputs({"Output", "InvVariance", "Amax", "ScaleInv"})
    .SetInplaceMap({{"_Amax", "Amax"}, {"_ScaleInv", "ScaleInv"}})
    .Attrs({"eps: float", "index: int64_t", "otype: int64_t", "sm_margin: int64_t",
            "zero_centered_gamma: bool"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_rmsnorm_fwd_fp8));

PD_BUILD_OP(te_rmsnorm_bwd)
    .Inputs({"Dz", "X", "Rsigma", "Gamma"})
    .Outputs({"Dx", "Dgamma"})
    .Attrs({"sm_margin: int64_t", "zero_centered_gamma: bool"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_rmsnorm_bwd));

PD_BUILD_OP(te_fused_attn_fwd_qkvpacked)
    .Inputs({"QKV", "cu_seqlens", paddle::Optional("Bias"), "_O", paddle::Optional("_softmax_aux"),
             "_rng_state"})
    .Outputs({"O", paddle::Optional("softmax_aux"), "rng_state"})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "total_seqs: int64_t", "max_seqlen: int64_t",
            "is_training: bool", "attn_scale: float", "p_dropout: float", "qkv_layout: std::string",
            "bias_type: std::string", "attn_mask_type: std::string", "qkv_type: int64_t",
            "rng_elts_per_thread: int64_t"})
    .SetInplaceMap({{"_O", "O"},
                    {paddle::Optional("_softmax_aux"), paddle::Optional("softmax_aux")},
                    {"_rng_state", "rng_state"}})
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
             paddle::Optional("_softmax_aux"), "_rng_state"})
    .Outputs({"O", paddle::Optional("softmax_aux"), "rng_state"})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "total_seqs_q: int64_t",
            "total_seqs_kv: int64_t", "max_seqlen_q: int64_t", "max_seqlen_kv: int64_t",
            "is_training: bool", "attn_scale: float", "p_dropout: float", "qkv_layout: std::string",
            "bias_type: std::string", "attn_mask_type: std::string", "qkv_type: int64_t",
            "rng_elts_per_thread: int64_t"})
    .SetInplaceMap({{"_O", "O"},
                    {paddle::Optional("_softmax_aux"), paddle::Optional("softmax_aux")},
                    {"_rng_state", "rng_state"}})
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

PD_BUILD_OP(te_fused_attn_fwd)
    .Inputs({"Q", "K", "V", "cu_seqlens_q", "cu_seqlens_kv", paddle::Optional("Bias"), "_O",
             paddle::Optional("_softmax_aux"), "_rng_state"})
    .Outputs({"O", paddle::Optional("softmax_aux"), "rng_state"})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "max_seqlen_q: int64_t",
            "max_seqlen_kv: int64_t", "is_training: bool", "attn_scale: float", "p_dropout: float",
            "qkv_layout: std::string", "bias_type: std::string", "attn_mask_type: std::string",
            "qkv_type: int64_t", "rng_elts_per_thread: int64_t"})
    .SetInplaceMap({{"_O", "O"},
                    {paddle::Optional("_softmax_aux"), paddle::Optional("softmax_aux")},
                    {"_rng_state", "rng_state"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_fused_attn_fwd));

PD_BUILD_OP(te_fused_attn_bwd)
    .Inputs({"Q", "K", "V", "cu_seqlens_q", "cu_seqlens_kv", "O", "dO", "softmax_aux", "_dQ", "_dK",
             "_dV", paddle::Optional("_dBias"), "rng_state"})
    .Outputs({"dQ", "dK", "dV", paddle::Optional("dBias")})
    .Attrs({"b: int64_t", "h: int64_t", "d: int64_t", "max_seqlen_q: int64_t",
            "max_seqlen_kv: int64_t", "attn_scale: float", "p_dropout: float",
            "qkv_layout: std::string", "bias_type: std::string", "attn_mask_type: std::string",
            "qkv_type: int64_t"})
    .SetInplaceMap({{"_dQ", "dQ"},
                    {"_dK", "dK"},
                    {"_dV", "dV"},
                    {paddle::Optional("_dBias"), paddle::Optional("dBias")}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::te_fused_attn_bwd));

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
    .Inputs({"_amax_history", "_scale", "_scale_inv", "non_weight_mask"})
    .Outputs({"amax_history", "scale", "scale_inv"})
    .SetInplaceMap({{"_amax_history", "amax_history"},
                    {"_scale", "scale"},
                    {"_scale_inv", "scale_inv"}})
    .Attrs({"fp8_dtype: int64_t", "margin: float", "amax_compute: std::string"})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::amax_and_scale_update_inplace));

PD_BUILD_OP(update_latest_amax_history_inplace)
    .Inputs({"_history", "amax"})
    .Outputs({"history"})
    .SetInplaceMap({{"_history", "history"}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::update_latest_amax_history_inplace));

PD_BUILD_OP(mask_to_cu_seqlens)
    .Inputs({"mask", "_q_cu_seqlen", paddle::Optional("_kv_cu_seqlen")})
    .Outputs({"q_cu_seqlen", paddle::Optional("kv_cu_seqlen")})
    .Attrs({"q_seqlen: int", "kv_seqlen: int", "need_kv: bool"})
    .SetInplaceMap({{"_q_cu_seqlen", "q_cu_seqlen"},
                    {paddle::Optional("_kv_cu_seqlen"), paddle::Optional("kv_cu_seqlen")}})
    .SetKernelFn(PD_KERNEL(transformer_engine::paddle_ext::mask_to_cu_seqlens));
