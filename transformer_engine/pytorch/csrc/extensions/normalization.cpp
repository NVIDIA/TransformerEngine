/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

namespace transformer_engine::pytorch {
std::pair<TensorWrapper, py::object> createOutputTensor(const NVTEShape &shape, DType dtype,
                                                        py::handle quantizer) {
  std::vector<size_t> shape_vec;
  for (int i = 0; i < shape.ndim; i++) {
    size_t t = shape.data[i];
    shape_vec.push_back(t);
  }
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  return my_quantizer->create_tensor(shape_vec, dtype);
}
std::pair<TensorWrapper, py::object> createOutputTensor(std::vector<size_t> &shape, DType dtype,
                                                        py::handle quantizer) {
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  return my_quantizer->create_tensor(shape, dtype);
}
}  // namespace transformer_engine::pytorch

std::vector<py::object> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &mu_ = mu.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  auto dbeta = at::empty_like(gamma_);
  transformer_engine::TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto mu_cu = makeTransformerEngineTensor(mu_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);
  auto dbeta_cu = makeTransformerEngineTensor(dbeta);

  // This call populates tensors with the required config.
  nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                     dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                     dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  return {py::cast(dx), py::cast(dgamma), py::cast(dbeta)};
}

std::vector<py::object> layernorm_fwd(py::handle input, py::handle weight, MaybeTensor bias,
                                      float eps, py::object ln_out, py::handle quantizer,
                                      DType out_dtype, const int sm_margin,
                                      const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  using namespace transformer_engine;

  auto none = py::none();
  const TensorWrapper &input_tensor = makeTransformerEngineTensor(input, none);
  const TensorWrapper &weight_tensor = makeTransformerEngineTensor(weight, none);

  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
    bias_tensor = makeTransformerEngineTensor(*bias);
  }

  // Tensor dimensions
  size_t N = static_cast<size_t>(input_tensor.size(0));
  size_t H = static_cast<size_t>(input_tensor.size(1));
  std::vector<size_t> size = {N, H};

  // Construct Transformer Engine tensors
  at::Tensor mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  at::Tensor rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));

  TensorWrapper ln_out_tensor;
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  py::object ln_output;

  if (my_quantizer->get_scaling_mode() == NVTE_MXFP8_1D_SCALING) {
    // Use high precision output from normalization
    NoneQuantizer q{none};
    std::tie(ln_out_tensor, ln_output) = q.create_tensor(size, out_dtype);
  } else {
    if (ln_out.is_none()) {
      std::tie(ln_out_tensor, ln_out) = my_quantizer->create_tensor(size, out_dtype);
    } else {
      ln_out_tensor = makeTransformerEngineTensor(ln_out, quantizer);
    }
  }
  TensorWrapper mu_cu = makeTransformerEngineTensor(mu);
  TensorWrapper rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Query workspace sizes
  transformer_engine::TensorWrapper workspace;
  nvte_layernorm_fwd(input_tensor.data(), weight_tensor.data(), bias_tensor.data(), eps,
                     ln_out_tensor.data(), mu_cu.data(), rsigma_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  // Allocate workspaces
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_layernorm_fwd(input_tensor.data(), weight_tensor.data(), bias_tensor.data(), eps,
                     ln_out_tensor.data(), mu_cu.data(), rsigma_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  if (my_quantizer->get_scaling_mode() == NVTE_MXFP8_1D_SCALING) {
    TensorWrapper cast_out_tensor;
    if (ln_out.is_none()) {
      std::tie(cast_out_tensor, ln_out) = my_quantizer->create_tensor(size, out_dtype);
    } else {
      cast_out_tensor = makeTransformerEngineTensor(ln_out, quantizer);
    }

    nvte_quantize_noop(ln_out_tensor.data(), cast_out_tensor.data(), nullptr,
                       at::cuda::getCurrentCUDAStream());
  }

  return {ln_out, py::cast(mu), py::cast(rsigma)};
}

std::vector<py::object> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  transformer_engine::TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                   dgamma_cu.data(), workspace.data(),
                   at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                   zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                   dgamma_cu.data(), workspace.data(),
                   at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                   zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  return {py::cast(dx), py::cast(dgamma)};
}

std::vector<py::object> rmsnorm_fwd(const py::handle &input, const py::handle &weight, float eps,
                                    py::object ln_out, py::handle quantizer,
                                    transformer_engine::DType otype, const int sm_margin,
                                    const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch;
  using namespace transformer_engine;

  auto none = py::none();
  const TensorWrapper &input_tensor = makeTransformerEngineTensor(input, none);
  const TensorWrapper &weight_tensor = makeTransformerEngineTensor(weight, none);

  // Tensor dimensions
  size_t N = static_cast<size_t>(input_tensor.shape().data[0]);
  size_t H = static_cast<size_t>(input_tensor.shape().data[1]);

  // Construct Transformer Engine tensors
  auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  std::vector<size_t> size = {N, H};
  TensorWrapper ln_out_tensor;
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  py::object ln_output;

  if (my_quantizer->get_scaling_mode() == NVTE_MXFP8_1D_SCALING) {
    // Use high precision output from normalization
    NoneQuantizer q{none};
    std::tie(ln_out_tensor, ln_output) = q.create_tensor(size, otype);
  } else {
    if (ln_out.is_none()) {
      std::tie(ln_out_tensor, ln_out) = my_quantizer->create_tensor(size, otype);
    } else {
      ln_out_tensor = makeTransformerEngineTensor(ln_out, quantizer);
    }
  }
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Query workspace sizes
  transformer_engine::TensorWrapper workspace;
  nvte_rmsnorm_fwd(input_tensor.data(), weight_tensor.data(), eps, ln_out_tensor.data(),
                   rsigma_cu.data(), workspace.data(),
                   at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                   zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  // Allocate workspaces
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_rmsnorm_fwd(input_tensor.data(), weight_tensor.data(), eps, ln_out_tensor.data(),
                   rsigma_cu.data(), workspace.data(),
                   at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                   zero_centered_gamma, at::cuda::getCurrentCUDAStream());

  if (my_quantizer->get_scaling_mode() == NVTE_MXFP8_1D_SCALING) {
    TensorWrapper cast_out_tensor;
    if (ln_out.is_none()) {
      std::tie(cast_out_tensor, ln_out) = my_quantizer->create_tensor(size, otype);
    } else {
      cast_out_tensor = makeTransformerEngineTensor(ln_out, quantizer);
    }

    nvte_quantize_noop(ln_out_tensor.data(), cast_out_tensor.data(), nullptr,
                       at::cuda::getCurrentCUDAStream());
  }

  return {ln_out, py::none(), py::cast(rsigma)};
}
