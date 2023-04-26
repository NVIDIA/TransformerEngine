/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"


void te_gemm(at::Tensor A,
             at::Tensor A_scale_inverse,
             transformer_engine::DType A_type,
             bool transa,
             at::Tensor B,
             at::Tensor B_scale_inverse,
             transformer_engine::DType B_type,
             bool transb,
             at::Tensor D,
             at::Tensor D_scale,
             transformer_engine::DType D_type,
             at::Tensor D_amax,
             at::Tensor bias,
             transformer_engine::DType bias_type,
             at::Tensor pre_gelu_out,
             bool grad,
             at::Tensor workspace,
             size_t workspaceSize,
             bool accumulate,
             bool use_split_accumulator
) {
  using namespace transformer_engine;
  auto te_A = makeTransformerEngineTensor(A.data_ptr(),
                                          {static_cast<size_t>(A.size(0)),
                                           static_cast<size_t>(A.size(1))},
                                          A_type, nullptr, nullptr,
                                          A_scale_inverse.data_ptr());
  auto te_B = makeTransformerEngineTensor(B.data_ptr(),
                                          {static_cast<size_t>(B.size(0)),
                                           static_cast<size_t>(B.size(1))},
                                          B_type, nullptr, nullptr,
                                          B_scale_inverse.data_ptr());
  auto te_D = makeTransformerEngineTensor(D.data_ptr(),
                                          {static_cast<size_t>(D.size(0)),
                                           static_cast<size_t>(D.size(1))},
                                          D_type, D_amax.data_ptr(),
                                          D_scale.data_ptr(), nullptr);
  auto te_bias = makeTransformerEngineTensor(bias.data_ptr(), {static_cast<size_t>(bias.size(0))},
                                             bias_type);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                          ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                          : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(pre_gelu_out.data_ptr(),
                                                     gelu_shape,
                                                     GetTransformerEngineDType(
                                                         pre_gelu_out.scalar_type()));
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  {workspaceSize},
                                                  DType::kByte);

  nvte_cublas_gemm(te_A.data(),
                   te_B.data(),
                   te_D.data(),
                   te_bias.data(),
                   te_pre_gelu_out.data(),
                   transa,
                   transb,
                   grad,
                   te_workspace.data(),
                   accumulate,
                   use_split_accumulator,
                   at::cuda::getCurrentCUDAStream());
}


void fused_cast_transpose(at::Tensor input,
                          at::Tensor scale,
                          at::Tensor amax,
                          at::Tensor scale_inv,
                          at::Tensor input_cast,
                          at::Tensor input_transpose,
                          transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto input_cu            = makeTransformerEngineTensor(input);
  auto output_cast_cu      = makeTransformerEngineTensor(input_cast.data_ptr(), {M, N}, otype,
                                                         amax.data_ptr(), scale.data_ptr(),
                                                         scale_inv.data_ptr());
  auto output_transpose_cu = makeTransformerEngineTensor(input_transpose.data_ptr(), {N, M}, otype,
                                                         amax.data_ptr(), scale.data_ptr(),
                                                         scale_inv.data_ptr());

  nvte_cast_transpose(input_cu.data(), output_cast_cu.data(), output_transpose_cu.data(),
                      at::cuda::getCurrentCUDAStream());
}


std::vector<at::Tensor> fused_cast_transpose_bgrad(at::Tensor grad_output,
                                                   at::Tensor scale,
                                                   at::Tensor amax,
                                                   at::Tensor scale_inv,
                                                   transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto grad_output_cast =
            allocateTorchTensor(grad_output.size(0),
                                grad_output.size(1),
                                DType::kByte);
  auto grad_output_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);

  auto input_cu             = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu       = makeTransformerEngineTensor(grad_output_cast.data_ptr(), {M, N},
                                                          otype, amax.data_ptr(), scale.data_ptr(),
                                                          scale_inv.data_ptr());
  auto transposed_output_cu = makeTransformerEngineTensor(grad_output_transpose.data_ptr(),
                                                          {N, M}, otype, amax.data_ptr(),
                                                          scale.data_ptr(), scale_inv.data_ptr());
  auto dbias_cu             = makeTransformerEngineTensor(grad_bias);
  transformer_engine::TensorWrapper workspace;

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_cast_transpose_dbias(input_cu.data(), cast_output_cu.data(),
                            transposed_output_cu.data(), dbias_cu.data(),
                            workspace.data(), at::cuda::getCurrentCUDAStream());

  return {grad_bias, grad_output_cast, grad_output_transpose};
}


std::vector<at::Tensor> fused_cast_transpose_bgrad_dgelu(at::Tensor grad_output,
                                                         at::Tensor gelu_input,
                                                         at::Tensor scale,
                                                         at::Tensor amax,
                                                         at::Tensor scale_inv,
                                                         transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(grad_output.size(0));
  size_t N = static_cast<size_t>(grad_output.size(1));

  DType grad_output_type = GetTransformerEngineDType(grad_output.scalar_type());
  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_output_type);
  auto dgelu =
            allocateTorchTensor(grad_output.size(0),
                                grad_output.size(1),
                                DType::kByte);
  auto dgelu_transpose =
            allocateTorchTensor(grad_output.size(1),
                                grad_output.size(0),
                                DType::kByte);

  transformer_engine::TensorWrapper workspace;
  auto gelu_input_cu        = makeTransformerEngineTensor(gelu_input);
  auto input_cu             = makeTransformerEngineTensor(grad_output);
  auto cast_output_cu       = makeTransformerEngineTensor(dgelu.data_ptr(), {M, N},
                                                          otype, amax.data_ptr(), scale.data_ptr(),
                                                          scale_inv.data_ptr());
  auto transposed_output_cu = makeTransformerEngineTensor(dgelu_transpose.data_ptr(), {N, M},
                                                          otype, amax.data_ptr(), scale.data_ptr(),
                                                          scale_inv.data_ptr());
  auto dbias_cu             = makeTransformerEngineTensor(grad_bias);

  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(),
                                  cast_output_cu.data(), transposed_output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  // Fill workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                          workspace.shape(),
                                          workspace.dtype());

  nvte_cast_transpose_dbias_dgelu(input_cu.data(), gelu_input_cu.data(),
                                  cast_output_cu.data(), transposed_output_cu.data(),
                                  dbias_cu.data(), workspace.data(),
                                  at::cuda::getCurrentCUDAStream());

  return {grad_bias, dgelu, dgelu_transpose};
}


void fused_multi_cast_transpose(std::vector<at::Tensor> input_list,
                                std::vector<at::Tensor> scale_list,
                                std::vector<at::Tensor> cast_output_list,
                                std::vector<at::Tensor> transposed_output_list,
                                std::vector<at::Tensor> amax_list,
                                std::vector<at::Tensor> scale_inv_list,
                                transformer_engine::DType otype
) {
  using namespace transformer_engine;

  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, scale_dptr_list,
    cast_output_dptr_list, transposed_output_dptr_list,
    amax_dptr_list, scale_inv_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, scale_shape_list,
    cast_output_shape_list, transposed_output_shape_list,
    amax_shape_list, scale_inv_shape_list;
  std::vector<transformer_engine::DType> input_type_list, scale_type_list,
    cast_output_type_list, transposed_output_type_list,
    amax_type_list, scale_inv_type_list;
  auto extract_tensor_props_skip_dtype = [](at::Tensor& tensor,
                                            std::vector<void*>& dptr_list,
                                            std::vector<std::vector<size_t>>& shape_list) {
    dptr_list.push_back(tensor.data_ptr());
    shape_list.push_back({});
    for (int d = 0; d < tensor.dim(); ++d) {
      shape_list.back().push_back(tensor.size(d));
    }
  };
  auto extract_tensor_props = [](at::Tensor& tensor,
                                 std::vector<void*>& dptr_list,
                                 std::vector<std::vector<size_t>>& shape_list,
                                 std::vector<transformer_engine::DType>& type_list) {
    dptr_list.push_back(tensor.data_ptr());
    shape_list.push_back({});
    for (int d = 0; d < tensor.dim(); ++d) {
      shape_list.back().push_back(tensor.size(d));
    }
    type_list.push_back(GetTransformerEngineDType(tensor.scalar_type()));
  };
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    extract_tensor_props(input_list[tensor_id],
                         input_dptr_list,
                         input_shape_list,
                         input_type_list);
    extract_tensor_props(scale_list[tensor_id],
                         scale_dptr_list,
                         scale_shape_list,
                         scale_type_list);
    extract_tensor_props_skip_dtype(cast_output_list[tensor_id],
                                    cast_output_dptr_list,
                                    cast_output_shape_list);
    cast_output_type_list.push_back(otype);
    extract_tensor_props_skip_dtype(transposed_output_list[tensor_id],
                                    transposed_output_dptr_list,
                                    transposed_output_shape_list);
    transposed_output_type_list.push_back(otype);
    extract_tensor_props(amax_list[tensor_id],
                         amax_dptr_list,
                         amax_shape_list,
                         amax_type_list);
    extract_tensor_props(scale_inv_list[tensor_id],
                         scale_inv_dptr_list,
                         scale_inv_shape_list,
                         scale_inv_type_list);
  }

  transformer_engine::TensorWrapper workspace;

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list,
    nvte_cast_output_list, nvte_transposed_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr,
                                        const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype,
                                        void* amax_dptr,
                                        void* scale_dptr,
                                        void* scale_inv_dptr)
    -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype, amax_dptr,
                                                             scale_dptr, scale_inv_dptr));
    return tensor_wrappers.back().data();
  };
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    nvte_input_list.emplace_back(make_tensor(input_dptr_list[i],
                                             input_shape_list[i],
                                             input_type_list[i],
                                             nullptr,
                                             nullptr,
                                             nullptr));
    nvte_cast_output_list.emplace_back(make_tensor(cast_output_dptr_list[i],
                                                   cast_output_shape_list[i],
                                                   cast_output_type_list[i],
                                                   amax_dptr_list[i],
                                                   scale_dptr_list[i],
                                                   scale_inv_dptr_list[i]));
    nvte_transposed_output_list.emplace_back(make_tensor(transposed_output_dptr_list[i],
                                                         transposed_output_shape_list[i],
                                                         transposed_output_type_list[i],
                                                         amax_dptr_list[i],
                                                         scale_dptr_list[i],
                                                         scale_inv_dptr_list[i]));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_cast_output_list.size() == nvte_input_list.size(),
             "Number of input and C output tensors must match");
  NVTE_CHECK(nvte_transposed_output_list.size() == nvte_input_list.size(),
             "Number of input and T output tensors must match");

  // Launch TE kernel
  nvte_multi_cast_transpose(nvte_input_list.size(),
                            nvte_input_list.data(),
                            nvte_cast_output_list.data(),
                            nvte_transposed_output_list.data(),
                            at::cuda::getCurrentCUDAStream());
}


at::Tensor fp8_transpose(at::Tensor input,
                         transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto output =
            allocateTorchTensor(input.size(1),
                                input.size(0),
                                DType::kByte);

  auto input_cu  = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}


at::Tensor fp8_gelu(at::Tensor input,
                    at::Tensor scale,
                    at::Tensor amax,
                    at::Tensor scale_inv,
                    transformer_engine::DType otype
) {
  using namespace transformer_engine;

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  auto output =
            allocateTorchTensor(input.size(0),
                                input.size(1),
                                DType::kByte);

  auto input_cu =  makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {M, N}, otype,
                                               amax.data_ptr(), scale.data_ptr(),
                                               scale_inv.data_ptr());

  nvte_gelu(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}


std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz,
                                      const at::Tensor &x,
                                      const at::Tensor &mu,
                                      const at::Tensor &rsigma,
                                      const at::Tensor &gamma,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
) {
    auto dx = at::empty_like(x);
    auto dgamma = at::empty_like(gamma);
    auto dbeta = at::empty_like(gamma);
    transformer_engine::TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

    auto dz_cu      = makeTransformerEngineTensor(dz);
    auto x_cu       = makeTransformerEngineTensor(x);
    auto mu_cu      = makeTransformerEngineTensor(mu);
    auto rsigma_cu  = makeTransformerEngineTensor(rsigma);
    auto gamma_cu   = makeTransformerEngineTensor(gamma);
    auto dx_cu      = makeTransformerEngineTensor(dx);
    auto dgamma_cu  = makeTransformerEngineTensor(dgamma);
    auto dbeta_cu   = makeTransformerEngineTensor(dbeta);

    // This call populates tensors with the required config.
    const auto bwd_fun = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;
    bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
            dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(),
            dbeta_part.data(), at::cuda::getCurrentCUDAStream(),
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
            workspace.data(), barrier.data());

    // Alloc space for Tensors.
    auto workspace_data     = allocateSpace(workspace.shape(), workspace.dtype());
    auto barrier_data       = allocateSpace(barrier.shape(), barrier.dtype(), true);
    auto dgamma_part_data   = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
    auto dbeta_part_data    = allocateSpace(dbeta_part.shape(), dbeta_part.dtype());
    workspace   = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                              workspace.shape(),
                                              workspace.dtype());
    barrier     = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                              barrier.shape(),
                                              barrier.dtype());
    dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(),
                                              dgamma_part.shape(),
                                              dgamma_part.dtype());
    dbeta_part  = makeTransformerEngineTensor(dbeta_part_data.data_ptr(),
                                              dbeta_part.shape(),
                                              dbeta_part.dtype());

    // Actual call to bwd kernel.
    bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
            dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(),
            dbeta_part.data(), at::cuda::getCurrentCUDAStream(),
            at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
            workspace.data(), barrier.data());

    return { dx, dgamma, dbeta };
}


std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input,
                                          const at::Tensor &weight,
                                          const at::Tensor &bias,
                                          float eps,
                                          at::Tensor scale,
                                          at::Tensor amax,
                                          at::Tensor scale_inv,
                                          transformer_engine::DType otype,
                                          const int sm_margin,
                                          const bool zero_centered_gamma
) {
    using namespace transformer_engine;

    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    DType itype = GetTransformerEngineDType(input.scalar_type());

    auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(otype)));
    auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto input_cu     = makeTransformerEngineTensor(input);
    auto gamma_cu     = makeTransformerEngineTensor(weight);
    auto beta_cu      = makeTransformerEngineTensor(bias);
    auto z_cu         = makeTransformerEngineTensor(ln_out.data_ptr(), {N, H}, otype,
                                                    amax.data_ptr(), scale.data_ptr(),
                                                    scale_inv.data_ptr());
    auto mu_cu        = makeTransformerEngineTensor(mu);
    auto rsigma_cu    = makeTransformerEngineTensor(rsigma);
    transformer_engine::TensorWrapper workspace, barrier;

    // This call populates workspace and barrier tensors with the required config
    const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = allocateSpace(workspace.shape(),
                                        workspace.dtype());
    auto barrier_data = allocateSpace(barrier.shape(),
                                      barrier.dtype(),
                                      true);
    workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                            workspace.shape(),
                                            workspace.dtype());
    barrier   = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                            barrier.shape(),
                                            barrier.dtype());

    // Actual call to fwd kernel
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    return {ln_out, mu, rsigma};
}


at::Tensor layernorm_fwd_fp8_inf(const at::Tensor &input,
                                 const at::Tensor &weight,
                                 const at::Tensor &bias,
                                 float eps,
                                 at::Tensor scale,
                                 at::Tensor amax,
                                 at::Tensor scale_inv,
                                 transformer_engine::DType otype,
                                 const bool zero_centered_gamma
) {
    // This is a specialized version of layernorm_fwd_fp8, optimized for inference,
    // which only returns the normalized output.
    std::vector<at::Tensor> out = layernorm_fwd_fp8(
      input, weight, bias, eps, scale, amax, scale_inv, otype, 0, zero_centered_gamma);
    return out[0];
}


std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input,
                                      const at::Tensor &weight,
                                      const at::Tensor &bias,
                                      float eps,
                                      const int sm_margin,
                                      const bool zero_centered_gamma
) {
    using namespace transformer_engine;

    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    DType itype = GetTransformerEngineDType(input.scalar_type());

    auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(itype)));
    auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
    auto input_cu     = makeTransformerEngineTensor(input);
    auto gamma_cu     = makeTransformerEngineTensor(weight);
    auto beta_cu      = makeTransformerEngineTensor(bias);
    auto z_cu         = makeTransformerEngineTensor(ln_out);
    auto mu_cu        = makeTransformerEngineTensor(mu);
    auto rsigma_cu    = makeTransformerEngineTensor(rsigma);
    transformer_engine::TensorWrapper workspace, barrier;

    // This call populates workspace and barrier tensors with the required config
    const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    // Fill workspace and barrier
    auto workspace_data = allocateSpace(workspace.shape(),
                                        workspace.dtype());
    auto barrier_data = allocateSpace(barrier.shape(),
                                      barrier.dtype(),
                                      true);
    workspace = makeTransformerEngineTensor(workspace_data.data_ptr(),
                                            workspace.shape(),
                                            workspace.dtype());
    barrier   = makeTransformerEngineTensor(barrier_data.data_ptr(),
                                            barrier.shape(),
                                            barrier.dtype());

    // Actual call to fwd kernel
    func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(),
         mu_cu.data(), rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
         workspace.data(), barrier.data());

    return {ln_out, mu, rsigma};
}


at::Tensor layernorm_fwd_inf(const at::Tensor &input,
                             const at::Tensor &weight,
                             const at::Tensor &bias,
                             float eps,
                             const bool zero_centered_gamma
) {
    // This is a specialized version of layernorm_fwd, optimized for inference,
    // which only returns the normalized output.
    std::vector<at::Tensor> out = layernorm_fwd(input, weight, bias, eps, 0, zero_centered_gamma);
    return out[0];
}


at::Tensor cast_to_fp8(const at::Tensor &input,
                       const at::Tensor &scale,
                       at::Tensor amax,
                       at::Tensor scale_inv,
                       transformer_engine::DType otype
) {
    using namespace transformer_engine;
    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

    auto input_cu     = makeTransformerEngineTensor(input);
    auto output_cu    = makeTransformerEngineTensor(output.data_ptr(), {N, H}, otype,
                                                    amax.data_ptr(), scale.data_ptr(),
                                                    scale_inv.data_ptr());

    nvte_fp8_quantize(input_cu.data(), output_cu.data(),
                      at::cuda::getCurrentCUDAStream());

    return output;
}


at::Tensor cast_from_fp8(const at::Tensor &input,
                         const at::Tensor &scale_inv,
                         transformer_engine::DType itype,
                         transformer_engine::DType otype
) {
    using namespace transformer_engine;
    size_t N = static_cast<size_t>(input.size(0));
    size_t H = static_cast<size_t>(input.size(1));

    auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

    auto input_cu     = makeTransformerEngineTensor(input.data_ptr(), {N, H}, itype,
                                                    nullptr, nullptr, scale_inv.data_ptr());
    auto output_cu    = makeTransformerEngineTensor(output);

    nvte_fp8_dequantize(input_cu.data(), output_cu.data(),
                        at::cuda::getCurrentCUDAStream());

    return output;
}


at::Tensor scaled_softmax_forward(at::Tensor input,
                                  float scale_factor
) {
    using namespace transformer_engine;
    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int batches = input.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);

    TORCH_CHECK(key_seq_len <= 4096);
    TORCH_CHECK(query_seq_len > 1);

    // Output
  auto act_options = input.options().requires_grad(false);
  auto softmax_results =
      torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);

  auto input_cu = makeTransformerEngineTensor(input);
  auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

  nvte_scaled_softmax_forward(input_cu.data(), softmax_results_cu.data(), scale_factor,
                              at::cuda::getCurrentCUDAStream());

  return softmax_results;
}


at::Tensor scaled_softmax_backward(at::Tensor output_grad_,
                                   at::Tensor softmax_results_,
                                   float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 4D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 4D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(
          output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return output_grads;
}


at::Tensor scaled_masked_softmax_forward(at::Tensor input,
                                         at::Tensor mask,
                                         float scale_factor
) {
    using namespace transformer_engine;

    AT_ASSERTM(input.dim() == 4, "expected 4D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");
    AT_ASSERTM(mask.dim() == 4, "expected 4D tensor");
    if (!input.is_contiguous())
        input = input.contiguous();
    if (!mask.is_contiguous())
        mask = mask.contiguous();

    const int batches = input.size(0);
    const int pad_batches = mask.size(0);
    const int attn_heads = input.size(1);
    const int query_seq_len = input.size(2);
    const int key_seq_len = input.size(3);
    TORCH_CHECK(key_seq_len <= 4096);
    TORCH_CHECK(query_seq_len > 1);
    TORCH_CHECK(pad_batches == 1 || pad_batches == batches);
    TORCH_CHECK(mask.size(1) == 1);
    TORCH_CHECK(mask.size(2) == query_seq_len);
    TORCH_CHECK(mask.size(3) == key_seq_len);

    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({batches, attn_heads, query_seq_len, key_seq_len}, act_options);


    auto input_cu = makeTransformerEngineTensor(input);
    auto mask_cu = makeTransformerEngineTensor(mask);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_masked_softmax_forward(
          input_cu.data(), mask_cu.data(), softmax_results_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_masked_softmax_backward(at::Tensor output_grad_,
                                          at::Tensor softmax_results_,
                                          float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grad_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 4, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 4, "expected 3D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_softmax_backward(
          output_grads_cu.data(), softmax_results_cu.data(), output_grads_cu.data(),
          scale_factor, at::cuda::getCurrentCUDAStream());

    return output_grads;
}


at::Tensor scaled_upper_triang_masked_softmax_forward(at::Tensor input,
                                                      float scale_factor
) {
    using namespace transformer_engine;

    AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
    AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
               (input.scalar_type() == at::ScalarType::BFloat16),
               "Only fp16 and bf16 are supported");

    const int attn_batches = input.size(0);
    const int seq_len = input.size(1);
    TORCH_CHECK(seq_len <= 2048);

    // Output
    auto act_options = input.options().requires_grad(false);
    auto softmax_results =
        torch::empty({attn_batches, seq_len, seq_len}, act_options);

    auto input_cu = makeTransformerEngineTensor(input);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    nvte_scaled_upper_triang_masked_softmax_forward(input_cu.data(),
                                                    softmax_results_cu.data(),
                                                    scale_factor,
                                                    at::cuda::getCurrentCUDAStream());

    return softmax_results;
}


at::Tensor scaled_upper_triang_masked_softmax_backward(at::Tensor output_grads_,
                                                       at::Tensor softmax_results_,
                                                       float scale_factor
) {
    using namespace transformer_engine;

    auto output_grads = output_grads_.contiguous();
    auto softmax_results = softmax_results_.contiguous();

    AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");

    AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
        (output_grads.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");
    AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
        (softmax_results.scalar_type() == at::ScalarType::BFloat16),
        "Only fp16 and bf16 are supported");

    TORCH_CHECK(output_grads.size(1) == output_grads.size(2));

    auto output_grads_cu = makeTransformerEngineTensor(output_grads);
    auto softmax_results_cu = makeTransformerEngineTensor(softmax_results);

    // Produce gradients in place.
    nvte_scaled_upper_triang_masked_softmax_backward(output_grads_cu.data(),
                                                     softmax_results_cu.data(),
                                                     output_grads_cu.data(),
                                                     scale_factor,
                                                     at::cuda::getCurrentCUDAStream());

  return output_grads;
}


size_t get_cublasLt_version() {
    return cublasLtGetVersion();
}


bool userbuf_comm_available() {  // TODO(ksivamani) check on python side
#ifdef NVTE_WITH_USERBUFFERS
    return true;
#else
    return false;
#endif
}

void placeholder() {}  // TODO(ksivamani) clean this up

namespace flash_attention {

constexpr int warp_size = 32;
constexpr int type_size = 2;  // FP16 or BF16
constexpr int nvec = sizeof(uint64_t) / type_size;
constexpr int load_size = warp_size * nvec;
constexpr int block_size = 512;

template <typename T>
__launch_bounds__(block_size)
__global__ void prepare_kernel_fwd(const T *qkvi,
                                   T *qkv,
                                   const size_t B,
                                   const size_t S,
                                   const size_t Z,
                                   const size_t W) {
    const int warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
    const int id_in_warp = threadIdx.x % warp_size;
    const size_t offset_input = blockIdx.y * W + warpid * 3 * W * Z + id_in_warp * nvec;
    const T *my_input = qkvi + offset_input;

    const size_t s = warpid / B;
    if (s >= S) return;

    const size_t b = warpid % B;

    const size_t offset_output = blockIdx.y * B * S * Z * W +
                                 (s + b * S) * W * Z +
                                 id_in_warp * nvec;

    T *my_output = qkv + offset_output;

    for (int i = 0; i < Z; ++i) {
        uint64_t *out = reinterpret_cast<uint64_t*>(my_output + i * load_size);
        *out = *reinterpret_cast<const uint64_t*>(my_input + i * load_size * 3);
    }
}

template <typename T>
__launch_bounds__(block_size)
__global__ void prepare_kernel_bwd(const T *q, const T *k, const T *v,
                                   T *qkv, const size_t B, const size_t S,
                                   const size_t Z, const size_t W) {
    const T *input = blockIdx.y == 0 ? q : (blockIdx.y == 1 ? k : v);

    const int warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
    const int id_in_warp = threadIdx.x % warp_size;
    const size_t offset_input = warpid * W * Z + id_in_warp * nvec;
    const T *my_input = input + offset_input;

    const size_t b = warpid / S;
    if (b >= B) return;

    const size_t s = warpid % S;

    const size_t offset_output = (b + s * B) * 3 * W * Z +
                                 id_in_warp * nvec + blockIdx.y * W;

    T *my_output = qkv + offset_output;

    for (int i = 0; i < Z; ++i) {
        uint64_t *out = reinterpret_cast<uint64_t*>(my_output + i * load_size * 3);
        *out = *reinterpret_cast<const uint64_t*>(my_input + i * load_size);
    }
}

}  // namespace flash_attention

at::Tensor fa_prepare_fwd(at::Tensor qkvi) {
    NVTE_CHECK(qkvi.dim() == 4, "Expected 4-dim tensor.");
    NVTE_CHECK(qkvi.scalar_type() == at::ScalarType::Half ||
               qkvi.scalar_type() == at::ScalarType::BFloat16);
    NVTE_CHECK(qkvi.size(3) % flash_attention::load_size == 0);
    NVTE_CHECK(qkvi.size(3) == flash_attention::load_size);
    NVTE_CHECK(qkvi.stride(3) == 1, "Wrong stride.");
    NVTE_CHECK(qkvi.stride(2) == 3 * qkvi.size(3), "Wrong stride.");
    NVTE_CHECK(qkvi.stride(1) == 3 * qkvi.size(3) * qkvi.size(2), "Wrong stride.");
    NVTE_CHECK(qkvi.stride(0) == 3 * qkvi.size(3) * qkvi.size(2) * qkvi.size(1), "Wrong stride.");

    // [s, b, n, h * 3] -> [3, b, s, n, h]
    std::vector<int64_t> shape = {3, qkvi.size(1), qkvi.size(0), qkvi.size(2), qkvi.size(3)};
    at::Tensor qkv = at::empty(shape, at::CUDA(qkvi.scalar_type()));

    size_t warps = qkvi.size(0) * qkvi.size(1);
    size_t warps_per_block = flash_attention::block_size / flash_attention::warp_size;
    size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
    dim3 grid(blocks, 3);
    int threads = flash_attention::block_size;
    if (qkvi.scalar_type() == at::ScalarType::Half) {
        using dtype = at::Half;
        flash_attention::prepare_kernel_fwd<dtype><<<grid, threads, 0,
                                                     at::cuda::getCurrentCUDAStream()>>>(
            qkvi.data_ptr<dtype>(),
            qkv.data_ptr<dtype>(),
            shape[1],
            shape[2],
            shape[3],
            shape[4]);
    } else {
        using dtype = at::BFloat16;
        flash_attention::prepare_kernel_fwd<dtype><<<grid, threads, 0,
                                                     at::cuda::getCurrentCUDAStream()>>>(
            qkvi.data_ptr<dtype>(),
            qkv.data_ptr<dtype>(),
            shape[1],
            shape[2],
            shape[3],
            shape[4]);
    }

    return qkv;
}

at::Tensor fa_prepare_bwd(at::Tensor q, at::Tensor k, at::Tensor v) {
    NVTE_CHECK(q.is_contiguous());
    NVTE_CHECK(k.is_contiguous());
    NVTE_CHECK(v.is_contiguous());
    NVTE_CHECK(q.dim() == 4, "Expected 4-dim tensor.");
    NVTE_CHECK(k.dim() == 4, "Expected 4-dim tensor.");
    NVTE_CHECK(v.dim() == 4, "Expected 4-dim tensor.");
    NVTE_CHECK(q.scalar_type() == at::ScalarType::Half ||
               q.scalar_type() == at::ScalarType::BFloat16);
    NVTE_CHECK(k.scalar_type() == q.scalar_type());
    NVTE_CHECK(v.scalar_type() == q.scalar_type());
    NVTE_CHECK(q.size(3) % flash_attention::load_size == 0);
    NVTE_CHECK(q.size(3) == flash_attention::load_size);
    NVTE_CHECK(k.size(3) % flash_attention::load_size == 0);
    NVTE_CHECK(k.size(3) == flash_attention::load_size);
    NVTE_CHECK(v.size(3) % flash_attention::load_size == 0);
    NVTE_CHECK(v.size(3) == flash_attention::load_size);

    // 3 x [s, b, n, h] -> [b, s, n, 3 * h]

    std::vector<int64_t> shape = {q.size(1), q.size(0), q.size(2), 3 * q.size(3)};
    at::Tensor qkv = at::empty(shape, at::CUDA(q.scalar_type()));

    size_t warps = q.size(0) * q.size(1);
    size_t warps_per_block = flash_attention::block_size / flash_attention::warp_size;
    size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
    dim3 grid(blocks, 3);
    int threads = flash_attention::block_size;
    if (q.scalar_type() == at::ScalarType::Half) {
        using dtype = at::Half;
        flash_attention::prepare_kernel_bwd<dtype><<<grid, threads, 0,
                                                 at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<dtype>(),
            k.data_ptr<dtype>(),
            v.data_ptr<dtype>(),
            qkv.data_ptr<dtype>(),
            q.size(0),
            q.size(1),
            q.size(2),
            q.size(3));
    } else {
        using dtype = at::BFloat16;
        flash_attention::prepare_kernel_bwd<dtype><<<grid, threads, 0,
                                                 at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<dtype>(),
            k.data_ptr<dtype>(),
            v.data_ptr<dtype>(),
            qkv.data_ptr<dtype>(),
            q.size(0),
            q.size(1),
            q.size(2),
            q.size(3));
    }

    return qkv;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Softmax functions
  m.def("scaled_softmax_forward", &scaled_softmax_forward, "Scaled Softmax FWD");
  m.def("scaled_softmax_backward", &scaled_softmax_backward, "Scaled Softmax BWD");
  m.def("scaled_masked_softmax_forward", &scaled_masked_softmax_forward,
                                                    "Scaled Masked Softmax FWD");
  m.def("scaled_masked_softmax_backward", &scaled_masked_softmax_backward,
                                                    "Scaled Masked Softmax BWD");
  m.def("scaled_upper_triang_masked_softmax_forward",
            &scaled_upper_triang_masked_softmax_forward,
            "Scaled Upper-Triangular Masked Softmax FWD");
  m.def("scaled_upper_triang_masked_softmax_backward",
            &scaled_upper_triang_masked_softmax_backward,
            "Scaled Upper-Triangular Masked Softmax BWD");

  // Other granular functions
  m.def("layernorm_fwd_fp8", &layernorm_fwd_fp8, "LN FWD FP8");
  m.def("layernorm_bwd", &layernorm_bwd, "LN BWD");
  m.def("layernorm_fwd", &layernorm_fwd, "LN FWD");
  m.def("fused_cast_transpose", &fused_cast_transpose, "Fused Cast + Transpose");
  m.def("fused_cast_transpose_bgrad", &fused_cast_transpose_bgrad,
                                              "Fused Cast + Transpose + BGRAD");
  m.def("fused_cast_transpose_bgrad_dgelu", &fused_cast_transpose_bgrad_dgelu,
                                              "Fused Cast + Transpose + BGRAD + DGELU");
  m.def("fused_multi_cast_transpose", &fused_multi_cast_transpose,
                                              "Fused Multi-tensor Cast + Transpose");
  m.def("cast_to_fp8", &cast_to_fp8, "Cast to FP8");
  m.def("cast_from_fp8", &cast_from_fp8, "Cast from FP8");
  m.def("te_gemm", &te_gemm, "CublasLt GEMM");
  m.def("fp8_transpose", &fp8_transpose, "Transpose with FP8 I/O");
  m.def("fp8_gelu", &fp8_gelu, "GeLU with FP8 output");
  m.def("fa_prepare_fwd", &fa_prepare_fwd, "Prepare QKV for Flash Attention");
  m.def("fa_prepare_bwd", &fa_prepare_bwd, "Backward of QKV preparation for Flash Attention");

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");

  // Data structures
  py::class_<transformer_engine::FP8TensorMeta>(m, "FP8TensorMeta")
    .def(py::init<>())
    .def_readwrite("scale", &transformer_engine::FP8TensorMeta::scale)
    .def_readwrite("scale_inv", &transformer_engine::FP8TensorMeta::scale_inv)
    .def_readwrite("amax_history", &transformer_engine::FP8TensorMeta::amax_history);

  py::enum_<transformer_engine::DType>(m, "DType", py::module_local())
    .value("kByte", transformer_engine::DType::kByte)
    .value("kInt32", transformer_engine::DType::kInt32)
    .value("kFloat32", transformer_engine::DType::kFloat32)
    .value("kFloat16", transformer_engine::DType::kFloat16)
    .value("kBFloat16", transformer_engine::DType::kBFloat16)
    .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
    .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors")
    .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
    .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
    .value("GEMM1_OUTPUT", transformer_engine::FP8FwdTensors::GEMM1_OUTPUT)
    .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
    .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT)
    .value("GEMM2_OUTPUT", transformer_engine::FP8FwdTensors::GEMM2_OUTPUT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors")
    .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
    .value("GRAD_INPUT1", transformer_engine::FP8BwdTensors::GRAD_INPUT1)
    .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2)
    .value("GRAD_INPUT2", transformer_engine::FP8BwdTensors::GRAD_INPUT2);
}
