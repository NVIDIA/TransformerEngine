/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

std::vector<at::Tensor> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma) {
  auto dx = at::empty_like(x);
  auto dgamma = at::empty_like(gamma);
  auto dbeta = at::empty_like(gamma);
  transformer_engine::TensorWrapper workspace, barrier, dgamma_part, dbeta_part;

  auto dz_cu = makeTransformerEngineTensor(dz);
  auto x_cu = makeTransformerEngineTensor(x);
  auto mu_cu = makeTransformerEngineTensor(mu);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);
  auto gamma_cu = makeTransformerEngineTensor(gamma);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);
  auto dbeta_cu = makeTransformerEngineTensor(dbeta);

  // This call populates tensors with the required config.
  const auto bwd_fun = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;
  bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(),
          at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
  auto dgamma_part_data = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
  auto dbeta_part_data = allocateSpace(dbeta_part.shape(), dbeta_part.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  barrier = makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());
  dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(), dgamma_part.shape(),
                                            dgamma_part.dtype());
  dbeta_part = makeTransformerEngineTensor(dbeta_part_data.data_ptr(), dbeta_part.shape(),
                                           dbeta_part.dtype());

  // Actual call to bwd kernel.
  bwd_fun(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dbeta_cu.data(), dgamma_part.data(), dbeta_part.data(),
          at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  return {dx, dgamma, dbeta};
}

std::vector<at::Tensor> layernorm_fwd_fp8(const at::Tensor &input, const at::Tensor &weight,
                                          const at::Tensor &bias, float eps, at::Tensor scale,
                                          at::Tensor amax, at::Tensor scale_inv,
                                          transformer_engine::DType otype, const int sm_margin,
                                          const bool zero_centered_gamma, const int scale_offset,
                                          const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine;

  auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  return layernorm_fwd_fp8_noalloc(input, weight, bias, eps, scale, ln_out, amax, scale_inv, otype,
                                   sm_margin, zero_centered_gamma, scale_offset, amax_offset,
                                   scale_inv_offset);
}

std::vector<at::Tensor> layernorm_fwd_fp8_noalloc(
    const at::Tensor &input, const at::Tensor &weight, const at::Tensor &bias, float eps,
    at::Tensor scale, at::Tensor ln_out, at::Tensor amax, at::Tensor scale_inv,
    transformer_engine::DType otype, const int sm_margin, const bool zero_centered_gamma,
    const int scale_offset, const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine;

  // Choose kernel implementation
  const auto func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;

  // Tensor dimensions
  size_t N = static_cast<size_t>(input.size(0));
  size_t H = static_cast<size_t>(input.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void *scale_dptr = getDataPtr(scale, scale_offset);
  void *amax_dptr = getDataPtr(amax, amax_offset);
  void *scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType itype = GetTransformerEngineDType(input.scalar_type());
  auto mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  auto input_cu = makeTransformerEngineTensor(input);
  auto gamma_cu = makeTransformerEngineTensor(weight);
  auto beta_cu = makeTransformerEngineTensor(bias);
  auto z_cu = makeTransformerEngineTensor(ln_out.data_ptr(), {N, H}, otype, amax_dptr, scale_dptr,
                                          scale_inv_dptr);
  auto mu_cu = makeTransformerEngineTensor(mu);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Query workspace sizes
  transformer_engine::TensorWrapper workspace, barrier;
  func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(), mu_cu.data(),
       rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  // Allocate workspaces
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  barrier = makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());

  // Launch kernel
  func(input_cu.data(), gamma_cu.data(), beta_cu.data(), eps, z_cu.data(), mu_cu.data(),
       rsigma_cu.data(), at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  return {ln_out, mu, rsigma};
}

at::Tensor layernorm_fwd_fp8_inf(const at::Tensor &input, const at::Tensor &weight,
                                 const at::Tensor &bias, float eps, at::Tensor scale,
                                 at::Tensor amax, at::Tensor scale_inv,
                                 transformer_engine::DType otype, const int sm_margin,
                                 const bool zero_centered_gamma, const int scale_offset,
                                 const int amax_offset, const int scale_inv_offset

) {
  // This is a specialized version of layernorm_fwd_fp8, optimized for inference,
  // which only returns the normalized output.
  std::vector<at::Tensor> out =
      layernorm_fwd_fp8(input, weight, bias, eps, scale, amax, scale_inv, otype, sm_margin,
                        zero_centered_gamma, scale_offset, amax_offset, scale_inv_offset);
  return out[0];
}

std::vector<at::Tensor> layernorm_fwd(const at::Tensor &input, const at::Tensor &weight,
                                      const at::Tensor &bias, float eps, const int sm_margin,
                                      const bool zero_centered_gamma) {
  using namespace transformer_engine;

  DType itype = GetTransformerEngineDType(input.scalar_type());
  auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(itype)));

  return layernorm_fwd_noalloc(input, weight, bias, ln_out, eps, sm_margin, zero_centered_gamma);
}

std::vector<at::Tensor> layernorm_fwd_noalloc(const at::Tensor &input, const at::Tensor &weight,
                                              const at::Tensor &bias, at::Tensor ln_out, float eps,
                                              const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine;

  DType itype = GetTransformerEngineDType(input.scalar_type());

  return layernorm_fwd_fp8_noalloc(input, weight, bias, eps, at::Tensor(), ln_out, at::Tensor(),
                                   at::Tensor(), itype, sm_margin, zero_centered_gamma);
}

at::Tensor layernorm_fwd_inf(const at::Tensor &input, const at::Tensor &weight,
                             const at::Tensor &bias, float eps, const int sm_margin,
                             const bool zero_centered_gamma) {
  // This is a specialized version of layernorm_fwd, optimized for inference,
  // which only returns the normalized output.
  std::vector<at::Tensor> out =
      layernorm_fwd(input, weight, bias, eps, sm_margin, zero_centered_gamma);
  return out[0];
}

std::vector<at::Tensor> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma) {
  auto dx = at::empty_like(x);
  auto dgamma = at::empty_like(gamma);
  transformer_engine::TensorWrapper workspace, barrier, dgamma_part;

  auto dz_cu = makeTransformerEngineTensor(dz);
  auto x_cu = makeTransformerEngineTensor(x);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);
  auto gamma_cu = makeTransformerEngineTensor(gamma);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  const auto bwd_fun = zero_centered_gamma ? nvte_rmsnorm1p_bwd : nvte_rmsnorm_bwd;
  bwd_fun(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dgamma_part.data(), at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
  auto dgamma_part_data = allocateSpace(dgamma_part.shape(), dgamma_part.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  barrier = makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());
  dgamma_part = makeTransformerEngineTensor(dgamma_part_data.data_ptr(), dgamma_part.shape(),
                                            dgamma_part.dtype());

  // Actual call to bwd kernel.
  bwd_fun(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
          dgamma_cu.data(), dgamma_part.data(), at::cuda::getCurrentCUDAStream(),
          at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
          barrier.data());

  return {dx, dgamma};
}

std::vector<at::Tensor> rmsnorm_fwd_fp8(const at::Tensor &input, const at::Tensor &weight,
                                        float eps, at::Tensor scale, at::Tensor amax,
                                        at::Tensor scale_inv, transformer_engine::DType otype,
                                        const int sm_margin, const bool zero_centered_gamma,
                                        const int scale_offset, const int amax_offset,
                                        const int scale_inv_offset) {
  using namespace transformer_engine;

  auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(otype)));
  return rmsnorm_fwd_fp8_noalloc(input, weight, eps, scale, ln_out, amax, scale_inv, otype,
                                 sm_margin, zero_centered_gamma, scale_offset, amax_offset,
                                 scale_inv_offset);
}

std::vector<at::Tensor> rmsnorm_fwd_fp8_noalloc(const at::Tensor &input, const at::Tensor &weight,
                                                float eps, at::Tensor scale, at::Tensor ln_out,
                                                at::Tensor amax, at::Tensor scale_inv,
                                                transformer_engine::DType otype,
                                                const int sm_margin, const bool zero_centered_gamma,
                                                const int scale_offset, const int amax_offset,
                                                const int scale_inv_offset) {
  using namespace transformer_engine;

  // Choose kernel implementation
  const auto func = zero_centered_gamma ? nvte_rmsnorm1p_fwd : nvte_rmsnorm_fwd;

  // Tensor dimensions
  size_t N = static_cast<size_t>(input.size(0));
  size_t H = static_cast<size_t>(input.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void *scale_dptr = getDataPtr(scale, scale_offset);
  void *amax_dptr = getDataPtr(amax, amax_offset);
  void *scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  // Construct Transformer Engine tensors
  DType itype = GetTransformerEngineDType(input.scalar_type());
  auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  auto input_cu = makeTransformerEngineTensor(input);
  auto gamma_cu = makeTransformerEngineTensor(weight);
  auto z_cu = makeTransformerEngineTensor(ln_out.data_ptr(), {N, H}, otype, amax_dptr, scale_dptr,
                                          scale_inv_dptr);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Query workspace sizes
  transformer_engine::TensorWrapper workspace, barrier;
  func(input_cu.data(), gamma_cu.data(), eps, z_cu.data(), rsigma_cu.data(),
       at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  // Allocate workspaces
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  auto barrier_data = allocateSpace(barrier.shape(), barrier.dtype(), true);
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());
  barrier = makeTransformerEngineTensor(barrier_data.data_ptr(), barrier.shape(), barrier.dtype());

  // Launch kernel
  func(input_cu.data(), gamma_cu.data(), eps, z_cu.data(), rsigma_cu.data(),
       at::cuda::getCurrentCUDAStream(),
       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin, workspace.data(),
       barrier.data());

  return {ln_out, rsigma};
}

at::Tensor rmsnorm_fwd_fp8_inf(const at::Tensor &input, const at::Tensor &weight, float eps,
                               at::Tensor scale, at::Tensor amax, at::Tensor scale_inv,
                               transformer_engine::DType otype, const int sm_margin,
                               const bool zero_centered_gamma, const int scale_offset,
                               const int amax_offset, const int scale_inv_offset) {
  // This is a specialized version of rmsnorm_fwd_fp8, optimized for inference,
  // which only returns the normalized output.
  std::vector<at::Tensor> out =
      rmsnorm_fwd_fp8(input, weight, eps, scale, amax, scale_inv, otype, sm_margin,
                      zero_centered_gamma, scale_offset, amax_offset, scale_inv_offset);
  return out[0];
}

std::vector<at::Tensor> rmsnorm_fwd(const at::Tensor &input, const at::Tensor &weight, float eps,
                                    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine;

  DType itype = GetTransformerEngineDType(input.scalar_type());
  auto ln_out = at::empty_like(input, at::CUDA(GetATenDType(itype)));

  return rmsnorm_fwd_noalloc(input, weight, ln_out, eps, sm_margin, zero_centered_gamma);
}

std::vector<at::Tensor> rmsnorm_fwd_noalloc(const at::Tensor &input, const at::Tensor &weight,
                                            at::Tensor ln_out, float eps, const int sm_margin,
                                            const bool zero_centered_gamma) {
  using namespace transformer_engine;

  DType itype = GetTransformerEngineDType(input.scalar_type());

  return rmsnorm_fwd_fp8_noalloc(input, weight, eps, at::Tensor(), ln_out, at::Tensor(),
                                 at::Tensor(), itype, sm_margin, zero_centered_gamma);
}

at::Tensor rmsnorm_fwd_inf(const at::Tensor &input, const at::Tensor &weight, float eps,
                           const int sm_margin, const bool zero_centered_gamma) {
  // This is a specialized version of rmsnorm_fwd, optimized for inference,
  // which only returns the normalized output.
  std::vector<at::Tensor> out = rmsnorm_fwd(input, weight, eps, sm_margin, zero_centered_gamma);
  return out[0];
}
