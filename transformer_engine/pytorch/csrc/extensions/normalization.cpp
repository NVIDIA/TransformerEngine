/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/util/system.h"
#include "extensions.h"
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
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  TensorWrapper out_cu;
  if (out.is_none()) {
    std::tie(out_cu, out) = my_quantizer->create_tensor(size, out_dtype);
  } else {
    out_cu = makeTransformerEngineTensor(out, quantizer);
  }

  // Determine whether to avoid fused kernel
  bool force_unfused_kernel = true;
  if (quantizer.is_none()) {
    // No need for separate quantization step if output is unquantized
    force_unfused_kernel = false;
  } else if (IsFloat8Quantizers(quantizer.ptr())) {
    // Always used fused kernel for FP8 delayed scaling
    force_unfused_kernel = false;
  } else if (IsMXFP8Quantizers(quantizer.ptr())) {
    if (transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
      // cuDNN MXFP8 kernel requires full tile
      force_unfused_kernel = N % 128 != 0 || H % 128 != 0;
    }
  }
  TensorWrapper unquantized_out_cu;
  if (force_unfused_kernel) {
    NoneQuantizer q{none};
    py::object unquantized_out;
    std::tie(unquantized_out_cu, unquantized_out) = q.create_tensor(size, out_dtype);
  }
  TensorWrapper &kernel_out_cu = force_unfused_kernel ? unquantized_out_cu : out_cu;

  // Query workspace size
  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(), eps, kernel_out_cu.data(),
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
    nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(), eps, kernel_out_cu.data(),
                       mu_cu.data(), rsigma_cu.data(), workspace.data(),
                       at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                       zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Quantize output if using unfused kernel
  if (force_unfused_kernel) {
    QuantizationConfigWrapper quant_config;
    if (IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
      // my_quantizer here has to be a Float8CurrentScalingQuantizer
      auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer *>(my_quantizer.get());
      NVTE_SCOPED_GIL_RELEASE({
        nvte_compute_amax(unquantized_out_cu.data(), out_cu.data(),
                          at::cuda::getCurrentCUDAStream());
      });
      // check if we need to do amax reudction (depending on model parallel configs)
      if (my_quantizer_cs->with_amax_reduction) {
        c10::intrusive_ptr<dist_group_type> process_group_ptr =
            my_quantizer_cs->amax_reduction_group;
        // construct torch tesnor from NVTEBasicTensor without reallocating memory
        at::Tensor &amax_tensor_torch = my_quantizer_cs->amax;
        std::vector<at::Tensor> tensors = {amax_tensor_torch};
        // allreduce amax tensor
        c10d::AllreduceOptions allreduce_opts;
        allreduce_opts.reduceOp = c10d::ReduceOp::MAX;
        process_group_ptr->allreduce(tensors, allreduce_opts)->wait();
      }
      quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
      quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);
      NVTE_SCOPED_GIL_RELEASE({
        nvte_compute_scale_from_amax(out_cu.data(), quant_config, at::cuda::getCurrentCUDAStream());
      });
      // set amax ptr to null in te_output TensorWrapper to avoid atomic amax updates in kernel
      out_cu.set_amax(nullptr, DType::kFloat32, out_cu.defaultShape);
    } else if (IsFloat8BlockwiseQuantizers(quantizer.ptr())) {
      auto my_quantizer_bw = static_cast<Float8BlockQuantizer *>(my_quantizer.get());
      quant_config.set_force_pow_2_scales(my_quantizer_bw->force_pow_2_scales);
      quant_config.set_amax_epsilon(my_quantizer_bw->amax_epsilon);
    }
    NVTE_SCOPED_GIL_RELEASE({
      nvte_quantize_v2(unquantized_out_cu.data(), out_cu.data(), quant_config,
                       at::cuda::getCurrentCUDAStream());
    });
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
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  TensorWrapper out_cu;
  if (out.is_none()) {
    std::tie(out_cu, out) = my_quantizer->create_tensor(size, out_dtype);
  } else {
    out_cu = makeTransformerEngineTensor(out, quantizer);
  }

  // Determine whether to avoid fused kernel
  bool force_unfused_kernel = true;
  if (quantizer.is_none()) {
    // No need for separate quantization step if output is unquantized
    force_unfused_kernel = false;
  } else if (IsFloat8Quantizers(quantizer.ptr())) {
    // Always used fused kernel for FP8 delayed scaling
    force_unfused_kernel = false;
  } else if (IsMXFP8Quantizers(quantizer.ptr())) {
    if (transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN")) {
      // cuDNN MXFP8 kernel requires full tile
      force_unfused_kernel = N % 128 != 0 || H % 128 != 0;
    }
  }
  TensorWrapper unquantized_out_cu;
  if (force_unfused_kernel) {
    NoneQuantizer q{none};
    py::object unquantized_out;
    std::tie(unquantized_out_cu, unquantized_out) = q.create_tensor(size, out_dtype);
  }
  TensorWrapper &kernel_out_cu = force_unfused_kernel ? unquantized_out_cu : out_cu;

  // Query workspace size
  TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), eps, kernel_out_cu.data(), rsigma_cu.data(),
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
    nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), eps, kernel_out_cu.data(), rsigma_cu.data(),
                     workspace.data(),
                     at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin,
                     zero_centered_gamma, at::cuda::getCurrentCUDAStream());
  });

  // Quantize output if using unfused kernel
  if (force_unfused_kernel) {
    QuantizationConfigWrapper quant_config;
    if (IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
      // my_quantizer here has to be a Float8CurrentScalingQuantizer
      auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer *>(my_quantizer.get());
      NVTE_SCOPED_GIL_RELEASE({
        nvte_compute_amax(unquantized_out_cu.data(), out_cu.data(),
                          at::cuda::getCurrentCUDAStream());
      });
      // check if we need to do amax reudction (depending on model parallel configs)
      if (my_quantizer_cs->with_amax_reduction) {
        c10::intrusive_ptr<dist_group_type> process_group_ptr =
            my_quantizer_cs->amax_reduction_group;
        // construct torch tesnor from NVTEBasicTensor without reallocating memory
        at::Tensor &amax_tensor_torch = my_quantizer_cs->amax;
        std::vector<at::Tensor> tensors = {amax_tensor_torch};
        // allreduce amax tensor
        c10d::AllreduceOptions allreduce_opts;
        allreduce_opts.reduceOp = c10d::ReduceOp::MAX;
        process_group_ptr->allreduce(tensors, allreduce_opts)->wait();
      }
      quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
      quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);
      NVTE_SCOPED_GIL_RELEASE({
        nvte_compute_scale_from_amax(out_cu.data(), quant_config, at::cuda::getCurrentCUDAStream());
      });
      // set amax ptr to null in te_output TensorWrapper to avoid atomic amax updates in kernel
      out_cu.set_amax(nullptr, DType::kFloat32, out_cu.defaultShape);
    } else if (IsFloat8BlockwiseQuantizers(quantizer.ptr())) {
      auto my_quantizer_bw = static_cast<Float8BlockQuantizer *>(my_quantizer.get());
      quant_config.set_force_pow_2_scales(my_quantizer_bw->force_pow_2_scales);
      quant_config.set_amax_epsilon(my_quantizer_bw->amax_epsilon);
    }
    NVTE_SCOPED_GIL_RELEASE({
      nvte_quantize_v2(unquantized_out_cu.data(), out_cu.data(), quant_config,
                       at::cuda::getCurrentCUDAStream());
    });
  }

  return {out, py::none(), py::cast(rsigma)};
}

}  // namespace transformer_engine::pytorch
