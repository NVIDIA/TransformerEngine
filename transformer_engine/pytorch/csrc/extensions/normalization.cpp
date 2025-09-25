/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common/util/system.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

std::vector<py::object> layernorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                      const at::Tensor &mu, const at::Tensor &rsigma,
                                      const at::Tensor &gamma, const int sm_margin,
                                      const bool zero_centered_gamma) {
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &mu_ = mu.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  auto dbeta = at::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto mu_cu = makeTransformerEngineTensor(mu_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);
  auto dbeta_cu = makeTransformerEngineTensor(dbeta);

  // This call populates tensors with the required config.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma), py::cast(dbeta)};
}

std::vector<py::object> layernorm_fwd(py::handle input, py::handle weight, MaybeTensor bias,
                                      float eps, py::object out, py::handle quantizer,
                                      DType out_dtype, const int sm_margin,
                                      const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch::detail;

  // Input and param tensors
  auto none = py::none();
  const TensorWrapper &input_cu = makeTransformerEngineTensor(input, none);
  const TensorWrapper &weight_cu = makeTransformerEngineTensor(weight, none);
  TensorWrapper bias_cu;
  if (bias.has_value()) {
    bias_cu = makeTransformerEngineTensor(*bias);
  }

  // Tensor dimensions
  const size_t N = static_cast<size_t>(input_cu.size(0));
  const size_t H = static_cast<size_t>(input_cu.size(1));
  const std::vector<size_t> size = {N, H};

  // Tensors to save for backward pass
  at::Tensor mu = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  at::Tensor rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  TensorWrapper mu_cu = makeTransformerEngineTensor(mu);
  TensorWrapper rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Output tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  TensorWrapper out_cu;
  if (out.is_none()) {
    std::tie(out_cu, out) = quantizer_cpp->create_tensor(size, out_dtype);
  } else {
    out_cu = makeTransformerEngineTensor(out, quantizer);
  }

  // Choose implementation
  enum class Impl {
    // Compute norm in high precision, then quantize
    UNFUSED,
    // Compute norm directly
    FULLY_FUSED,
    // Compute norm and amax in high precision, then quantize to FP8
    FUSED_NORM_AMAX_FP8,
    // Compute norm and amax in high precision, then quantize to NVFP4
    FUSED_NORM_AMAX_NVFP4 };
  Impl impl = Impl::UNFUSED;
  if (quantizer.is_none() || IsFloat8Quantizers(quantizer.ptr())) {
    impl = Impl::FULLY_FUSED;
  } else if (IsMXFP8Quantizers(quantizer.ptr())) {
    if (transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")
        && N % 128 == 0 && H % 128 == 0) {
      // cuDNN MXFP8 kernel requires full 128x128 tiles
      impl = Impl::FULLY_FUSED;
    }
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    auto fp8_quantizer_cpp = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    NVTE_CHECK(fp8_quantizer_cpp != nullptr, "Could not cast to FP8 current scaling quantizer");
    impl = Impl::FUSED_NORM_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer*>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax) {
      // Post-RHT amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else {
      impl = Impl::FUSED_NORM_AMAX_NVFP4;
    }
  }

  // Construct unquantized output tensor if needed
  TensorWrapper unquantized_out_cu;
  py::object unquantized_out;
  TensorWrapper *kernel_out_cu = &out_cu;
  switch (impl) {
  case Impl::UNFUSED:
    {
      NoneQuantizer q{none};
      std::tie(unquantized_out_cu, unquantized_out) = q.create_tensor(size, out_dtype);
      kernel_out_cu = &unquantized_out_cu;
    }
    break;
  case Impl::FUSED_NORM_AMAX_FP8:
    {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_cu, unquantized_out) =
          fp8_quantizer_cpp->create_unquantized_tensor_with_amax(size, out_dtype);
      kernel_out_cu = &unquantized_out_cu;
    }
    break;
  case Impl::FUSED_NORM_AMAX_NVFP4:
    {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_cu, unquantized_out) =
          nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(out_cu, out_dtype);
      kernel_out_cu = &unquantized_out_cu;
    }
    break;
  default: {}
  }

  // Query workspace size
  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(), eps, kernel_out_cu->data(),
                       mu_cu.data(), rsigma_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Allocate workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(), eps, kernel_out_cu->data(),
                       mu_cu.data(), rsigma_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Quantize output if needed
  switch (impl) {
  case Impl::UNFUSED:
    {
      quantizer_cpp->quantize(unquantized_out_cu, out_cu);
    }
    break;
  case Impl::FUSED_NORM_AMAX_FP8:
    {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      fp8_quantizer_cpp->quantize_with_amax(unquantized_out_cu, out_cu);
    }
    break;
  case Impl::FUSED_NORM_AMAX_NVFP4:
    {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      nvfp4_quantizer_cpp->quantize_with_amax(unquantized_out_cu, out_cu);
    }
    break;
  default: {}
  }

  return {out, py::cast(mu), py::cast(rsigma)};
}

std::vector<py::object> rmsnorm_bwd(const at::Tensor &dz, const at::Tensor &x,
                                    const at::Tensor &rsigma, const at::Tensor &gamma,
                                    const int sm_margin, const bool zero_centered_gamma) {
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                     dgamma_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                     dgamma_cu.data(), workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma)};
}

std::vector<py::object> rmsnorm_bwd_add(const at::Tensor &dz, const at::Tensor &x,
                                        const at::Tensor &add, const at::Tensor &rsigma,
                                        const at::Tensor &gamma, const int sm_margin,
                                        const bool zero_centered_gamma) {
  const auto &dz_ = dz.contiguous();
  const auto &x_ = x.contiguous();
  const auto &add_ = add.contiguous();
  const auto &rsigma_ = rsigma.contiguous();
  const auto &gamma_ = gamma.contiguous();

  auto dx = at::empty_like(x_);
  auto dgamma = at::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto add_cu = makeTransformerEngineTensor(add_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  // This call populates tensors with the required config.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd_add(dz_cu.data(), x_cu.data(), add_cu.data(), rsigma_cu.data(),
                         gamma_cu.data(), dx_cu.data(), dgamma_cu.data(), workspace.data(),
                         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                         zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Alloc space for Tensors.
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Actual call to bwd kernel.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_bwd_add(dz_cu.data(), x_cu.data(), add_cu.data(), rsigma_cu.data(),
                         gamma_cu.data(), dx_cu.data(), dgamma_cu.data(), workspace.data(),
                         at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                         zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dx), py::cast(dgamma)};
}

std::vector<py::object> rmsnorm_fwd(const py::handle &input, const py::handle &weight, float eps,
                                    py::object out, py::handle quantizer, DType out_dtype,
                                    const int sm_margin, const bool zero_centered_gamma) {
  using namespace transformer_engine::pytorch::detail;

  // Input and param tensors
  auto none = py::none();
  const TensorWrapper &input_cu = makeTransformerEngineTensor(input, none);
  const TensorWrapper &weight_cu = makeTransformerEngineTensor(weight, none);

  // Tensor dimensions
  const size_t N = static_cast<size_t>(input_cu.shape().data[0]);
  const size_t H = static_cast<size_t>(input_cu.shape().data[1]);
  const std::vector<size_t> size = {N, H};

  // Tensors to save for backward pass
  auto rsigma = at::empty({static_cast<int64_t>(N)}, at::CUDA(at::kFloat));
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  // Output tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  TensorWrapper out_cu;
  if (out.is_none()) {
    std::tie(out_cu, out) = quantizer_cpp->create_tensor(size, out_dtype);
  } else {
    out_cu = makeTransformerEngineTensor(out, quantizer);
  }

  // Choose implementation
  enum class Impl {
    // Compute norm in high precision, then quantize
    UNFUSED,
    // Compute norm directly
    FULLY_FUSED,
    // Compute norm and amax in high precision, then quantize to FP8
    FUSED_NORM_AMAX_FP8,
    // Compute norm and amax in high precision, then quantize to NVFP4
    FUSED_NORM_AMAX_NVFP4 };
  Impl impl = Impl::UNFUSED;
  if (quantizer.is_none() || IsFloat8Quantizers(quantizer.ptr())) {
    impl = Impl::FULLY_FUSED;
  } else if (IsMXFP8Quantizers(quantizer.ptr())) {
    if (transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")
        && N % 128 == 0 && H % 128 == 0) {
      // cuDNN MXFP8 kernel requires full 128x128 tiles
      impl = Impl::FULLY_FUSED;
    }
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    auto fp8_quantizer_cpp = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    NVTE_CHECK(fp8_quantizer_cpp != nullptr, "Could not cast to FP8 current scaling quantizer");
    impl = Impl::FUSED_NORM_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer*>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax) {
      // Post-RHT amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else {
      impl = Impl::FUSED_NORM_AMAX_NVFP4;
    }
  }

  // Construct unquantized output tensor if needed
  TensorWrapper unquantized_out_cu;
  py::object unquantized_out;
  TensorWrapper *kernel_out_cu = &out_cu;
  switch (impl) {
  case Impl::UNFUSED:
    {
      NoneQuantizer q{none};
      std::tie(unquantized_out_cu, unquantized_out) = q.create_tensor(size, out_dtype);
      kernel_out_cu = &unquantized_out_cu;
    }
    break;
  case Impl::FUSED_NORM_AMAX_FP8:
    {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_cu, unquantized_out) =
          fp8_quantizer_cpp->create_unquantized_tensor_with_amax(size, out_dtype);
      kernel_out_cu = &unquantized_out_cu;
    }
    break;
  case Impl::FUSED_NORM_AMAX_NVFP4:
    {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      std::tie(unquantized_out_cu, unquantized_out) =
          nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(out_cu, out_dtype);
      kernel_out_cu = &unquantized_out_cu;
    }
    break;
  default: {}
  }

  // Query workspace size
  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), eps, kernel_out_cu->data(), rsigma_cu.data(),
                     workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Allocate workspace
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), eps, kernel_out_cu->data(), rsigma_cu.data(),
                     workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Quantize output if needed
  switch (impl) {
  case Impl::UNFUSED:
    {
      quantizer_cpp->quantize(unquantized_out_cu, out_cu);
    }
    break;
  case Impl::FUSED_NORM_AMAX_FP8:
    {
      auto fp8_quantizer_cpp = static_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
      fp8_quantizer_cpp->quantize_with_amax(unquantized_out_cu, out_cu);
    }
    break;
  case Impl::FUSED_NORM_AMAX_NVFP4:
    {
      auto nvfp4_quantizer_cpp = static_cast<NVFP4Quantizer *>(quantizer_cpp.get());
      nvfp4_quantizer_cpp->quantize_with_amax(unquantized_out_cu, out_cu);
    }
    break;
  default: {}
  }

  return {out, py::none(), py::cast(rsigma)};
}

}  // namespace transformer_engine::pytorch
