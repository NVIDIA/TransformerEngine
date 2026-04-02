/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/normalization.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// Layernorm backward
// ============================================================================

std::tuple<Tensor, Tensor, Tensor> layernorm_bwd(Tensor dz, Tensor x, Tensor mu, Tensor rsigma,
                                                 Tensor gamma, int64_t sm_margin,
                                                 bool zero_centered_gamma) {
  auto dz_ = torch::stable::contiguous(dz);
  auto x_ = torch::stable::contiguous(x);
  auto mu_ = torch::stable::contiguous(mu);
  auto rsigma_ = torch::stable::contiguous(rsigma);
  auto gamma_ = torch::stable::contiguous(gamma);

  auto dx = torch::stable::empty_like(x_);
  auto dgamma = torch::stable::empty_like(gamma_);
  auto dbeta = torch::stable::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto mu_cu = makeTransformerEngineTensor(mu_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);
  auto dbeta_cu = makeTransformerEngineTensor(dbeta);

  auto device_idx = dz_.get_device_index();
  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  // First call: query workspace size
  nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                     dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(), sm_count,
                     zero_centered_gamma, stream);

  // Allocate workspace
  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  auto workspace_data = allocateStableTensor(
      std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
  workspace = makeTransformerEngineTensor(
      workspace_data.data_ptr(), std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
      ws_dtype);

  // Second call: actual computation
  nvte_layernorm_bwd(dz_cu.data(), x_cu.data(), mu_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                     dx_cu.data(), dgamma_cu.data(), dbeta_cu.data(), workspace.data(), sm_count,
                     zero_centered_gamma, stream);

  return std::make_tuple(dx, dgamma, dbeta);
}

// ============================================================================
// Layernorm forward (unquantized output)
// ============================================================================

std::tuple<Tensor, Tensor, Tensor> layernorm_fwd(Tensor input, Tensor weight,
                                                 std::optional<Tensor> bias, double eps,
                                                 int64_t sm_margin, bool zero_centered_gamma) {
  auto input_ = torch::stable::contiguous(input);
  auto weight_ = torch::stable::contiguous(weight);

  auto input_cu = makeTransformerEngineTensor(input_);
  auto weight_cu = makeTransformerEngineTensor(weight_);
  // bias_ must outlive the kernel launch — declaring at function scope ensures
  // the contiguous tensor (if created) stays alive until after the kernel.
  Tensor bias_contiguous;
  TensorWrapper bias_cu;
  if (bias.has_value()) {
    bias_contiguous = torch::stable::contiguous(bias.value());
    bias_cu = makeTransformerEngineTensor(bias_contiguous);
  }

  auto shape = getStableTensorShape(input_);
  size_t outer_size = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) outer_size *= shape[i];

  auto device_idx = input_.get_device_index();
  auto output = torch::stable::empty_like(input_);
  auto mu = allocateStableTensor({static_cast<int64_t>(outer_size)}, ScalarType::Float, device_idx);
  auto rsigma =
      allocateStableTensor({static_cast<int64_t>(outer_size)}, ScalarType::Float, device_idx);

  auto output_cu = makeTransformerEngineTensor(output);
  auto mu_cu = makeTransformerEngineTensor(mu);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);
  TensorWrapper workspace;

  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  // First call: query workspace
  nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(), static_cast<float>(eps),
                     output_cu.data(), mu_cu.data(), rsigma_cu.data(), workspace.data(), sm_count,
                     zero_centered_gamma, stream);

  // workspace_data must outlive the second kernel call — hoist out of if block.
  Tensor workspace_data;
  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  if (ws_shape.ndim > 0) {
    workspace_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
    workspace = makeTransformerEngineTensor(
        workspace_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype);
  }

  // Second call: actual computation
  nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(), static_cast<float>(eps),
                     output_cu.data(), mu_cu.data(), rsigma_cu.data(), workspace.data(), sm_count,
                     zero_centered_gamma, stream);

  return std::make_tuple(output, mu, rsigma);
}

// ============================================================================
// RMSnorm backward
// ============================================================================

std::tuple<Tensor, Tensor> rmsnorm_bwd(Tensor dz, Tensor x, Tensor rsigma, Tensor gamma,
                                       int64_t sm_margin, bool zero_centered_gamma) {
  auto dz_ = torch::stable::contiguous(dz);
  auto x_ = torch::stable::contiguous(x);
  auto rsigma_ = torch::stable::contiguous(rsigma);
  auto gamma_ = torch::stable::contiguous(gamma);

  auto dx = torch::stable::empty_like(x_);
  auto dgamma = torch::stable::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  auto device_idx = dz_.get_device_index();
  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                   dgamma_cu.data(), workspace.data(), sm_count, zero_centered_gamma, stream);

  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  auto workspace_data = allocateStableTensor(
      std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
  workspace = makeTransformerEngineTensor(
      workspace_data.data_ptr(), std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
      ws_dtype);

  nvte_rmsnorm_bwd(dz_cu.data(), x_cu.data(), rsigma_cu.data(), gamma_cu.data(), dx_cu.data(),
                   dgamma_cu.data(), workspace.data(), sm_count, zero_centered_gamma, stream);

  return std::make_tuple(dx, dgamma);
}

// ============================================================================
// RMSnorm forward (unquantized output)
// ============================================================================

std::tuple<Tensor, Tensor> rmsnorm_fwd(Tensor input, Tensor weight, double eps, int64_t sm_margin,
                                       bool zero_centered_gamma) {
  auto input_ = torch::stable::contiguous(input);
  auto weight_ = torch::stable::contiguous(weight);

  auto input_cu = makeTransformerEngineTensor(input_);
  auto weight_cu = makeTransformerEngineTensor(weight_);

  auto shape = getStableTensorShape(input_);
  size_t outer_size = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i) outer_size *= shape[i];

  auto device_idx = input_.get_device_index();
  auto output = torch::stable::empty_like(input_);
  auto rsigma =
      allocateStableTensor({static_cast<int64_t>(outer_size)}, ScalarType::Float, device_idx);

  auto output_cu = makeTransformerEngineTensor(output);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);
  TensorWrapper workspace;

  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), static_cast<float>(eps), output_cu.data(),
                   rsigma_cu.data(), workspace.data(), sm_count, zero_centered_gamma, stream);

  // workspace_data must outlive the second kernel call — hoist out of if block.
  Tensor workspace_data;
  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  if (ws_shape.ndim > 0) {
    workspace_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
    workspace = makeTransformerEngineTensor(
        workspace_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype);
  }

  nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), static_cast<float>(eps), output_cu.data(),
                   rsigma_cu.data(), workspace.data(), sm_count, zero_centered_gamma, stream);

  return std::make_tuple(output, rsigma);
}

// ============================================================================
// RMSnorm backward with add
// ============================================================================

std::tuple<Tensor, Tensor> rmsnorm_bwd_add(Tensor dz, Tensor x, Tensor add, Tensor rsigma,
                                           Tensor gamma, int64_t sm_margin,
                                           bool zero_centered_gamma) {
  auto dz_ = torch::stable::contiguous(dz);
  auto x_ = torch::stable::contiguous(x);
  auto add_ = torch::stable::contiguous(add);
  auto rsigma_ = torch::stable::contiguous(rsigma);
  auto gamma_ = torch::stable::contiguous(gamma);

  auto dx = torch::stable::empty_like(x_);
  auto dgamma = torch::stable::empty_like(gamma_);
  TensorWrapper workspace;

  auto dz_cu = makeTransformerEngineTensor(dz_);
  auto x_cu = makeTransformerEngineTensor(x_);
  auto add_cu = makeTransformerEngineTensor(add_);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma_);
  auto gamma_cu = makeTransformerEngineTensor(gamma_);
  auto dx_cu = makeTransformerEngineTensor(dx);
  auto dgamma_cu = makeTransformerEngineTensor(dgamma);

  auto device_idx = dz_.get_device_index();
  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  nvte_rmsnorm_bwd_add(dz_cu.data(), x_cu.data(), add_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), workspace.data(), sm_count,
                       zero_centered_gamma, stream);

  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  auto workspace_data = allocateStableTensor(
      std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
  workspace = makeTransformerEngineTensor(
      workspace_data.data_ptr(), std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim),
      ws_dtype);

  nvte_rmsnorm_bwd_add(dz_cu.data(), x_cu.data(), add_cu.data(), rsigma_cu.data(), gamma_cu.data(),
                       dx_cu.data(), dgamma_cu.data(), workspace.data(), sm_count,
                       zero_centered_gamma, stream);

  return std::make_tuple(dx, dgamma);
}

// ============================================================================
// Layernorm forward — no-alloc variant for quantized output
// ============================================================================
//
// The caller pre-allocates output_data and all quantization buffers.
// The NVTE kernel writes to the output TensorWrapper, which is configured
// from the raw buffer arguments. This preserves all kernel fusion:
//
//   FULLY_FUSED:     pass quantized output_data + amax + scale + scale_inv
//   NORM+AMAX fused: pass hp output_data + amax (no scale/scale_inv)
//   UNFUSED:         pass hp output_data only (no quantization buffers)
//
// The Python shim decides which buffers to provide based on quantizer type.

std::tuple<Tensor, Tensor> layernorm_fwd_noalloc(
    Tensor input, Tensor weight, std::optional<Tensor> bias, double eps,
    // Pre-allocated output buffer
    Tensor output_data,
    int64_t output_te_dtype,  // transformer_engine::DType as int
    // Optional quantization metadata (pass empty tensors if unused)
    std::optional<Tensor> output_amax, std::optional<Tensor> output_scale,
    std::optional<Tensor> output_scale_inv,
    int64_t scaling_mode,  // NVTEScalingMode as int
    // mu/rsigma pre-allocated by caller
    Tensor mu, Tensor rsigma, int64_t sm_margin, bool zero_centered_gamma) {
  auto input_ = torch::stable::contiguous(input);
  auto weight_ = torch::stable::contiguous(weight);

  auto input_cu = makeTransformerEngineTensor(input_);
  auto weight_cu = makeTransformerEngineTensor(weight_);
  // bias_contiguous must outlive the kernel — hoist out of if block.
  Tensor bias_contiguous;
  TensorWrapper bias_cu;
  if (bias.has_value()) {
    bias_contiguous = torch::stable::contiguous(bias.value());
    bias_cu = makeTransformerEngineTensor(bias_contiguous);
  }

  auto shape = getStableTensorShape(input_);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);
  auto mu_cu = makeTransformerEngineTensor(mu);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  auto device_idx = input_.get_device_index();
  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  runWithWorkspace(
      [&](NVTETensor ws) {
        nvte_layernorm_fwd(input_cu.data(), weight_cu.data(), bias_cu.data(),
                           static_cast<float>(eps), output_cu.data(), mu_cu.data(),
                           rsigma_cu.data(), ws, sm_count, zero_centered_gamma, stream);
      },
      device_idx);

  return std::make_tuple(mu, rsigma);
}

// ============================================================================
// RMSnorm forward — no-alloc variant for quantized output
// ============================================================================

Tensor rmsnorm_fwd_noalloc(Tensor input, Tensor weight, double eps, Tensor output_data,
                           int64_t output_te_dtype, std::optional<Tensor> output_amax,
                           std::optional<Tensor> output_scale,
                           std::optional<Tensor> output_scale_inv, int64_t scaling_mode,
                           Tensor rsigma, int64_t sm_margin, bool zero_centered_gamma) {
  auto input_ = torch::stable::contiguous(input);
  auto weight_ = torch::stable::contiguous(weight);

  auto input_cu = makeTransformerEngineTensor(input_);
  auto weight_cu = makeTransformerEngineTensor(weight_);

  auto shape = getStableTensorShape(input_);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);
  auto rsigma_cu = makeTransformerEngineTensor(rsigma);

  auto device_idx = input_.get_device_index();
  int sm_count = getSMCount(device_idx) - static_cast<int>(sm_margin);
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  runWithWorkspace(
      [&](NVTETensor ws) {
        nvte_rmsnorm_fwd(input_cu.data(), weight_cu.data(), static_cast<float>(eps),
                         output_cu.data(), rsigma_cu.data(), ws, sm_count, zero_centered_gamma,
                         stream);
      },
      device_idx);

  return rsigma;
}

}  // namespace transformer_engine::pytorch::stable

// Schema definitions (added to the transformer_engine_stable library)
STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "layernorm_bwd(Tensor dz, Tensor x, Tensor mu, Tensor rsigma, Tensor gamma, int sm_margin, "
      "bool zero_centered_gamma) -> (Tensor, Tensor, Tensor)");
  m.def(
      "layernorm_fwd(Tensor input, Tensor weight, Tensor? bias, float eps, int sm_margin, bool "
      "zero_centered_gamma) -> (Tensor, Tensor, Tensor)");
  m.def(
      "layernorm_fwd_noalloc(Tensor input, Tensor weight, Tensor? bias, float eps, Tensor "
      "output_data, int output_te_dtype, Tensor? output_amax, Tensor? output_scale, Tensor? "
      "output_scale_inv, int scaling_mode, Tensor mu, Tensor rsigma, int sm_margin, bool "
      "zero_centered_gamma) -> (Tensor, Tensor)");
  m.def(
      "rmsnorm_bwd(Tensor dz, Tensor x, Tensor rsigma, Tensor gamma, int sm_margin, bool "
      "zero_centered_gamma) -> (Tensor, Tensor)");
  m.def(
      "rmsnorm_fwd(Tensor input, Tensor weight, float eps, int sm_margin, bool "
      "zero_centered_gamma) -> (Tensor, Tensor)");
  m.def(
      "rmsnorm_fwd_noalloc(Tensor input, Tensor weight, float eps, Tensor output_data, int "
      "output_te_dtype, Tensor? output_amax, Tensor? output_scale, Tensor? output_scale_inv, int "
      "scaling_mode, Tensor rsigma, int sm_margin, bool zero_centered_gamma) -> Tensor");
  m.def(
      "rmsnorm_bwd_add(Tensor dz, Tensor x, Tensor add, Tensor rsigma, Tensor gamma, int "
      "sm_margin, bool zero_centered_gamma) -> (Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("layernorm_bwd", TORCH_BOX(layernorm_bwd));
  m.impl("layernorm_fwd", TORCH_BOX(layernorm_fwd));
  m.impl("layernorm_fwd_noalloc", TORCH_BOX(layernorm_fwd_noalloc));
  m.impl("rmsnorm_bwd", TORCH_BOX(rmsnorm_bwd));
  m.impl("rmsnorm_fwd", TORCH_BOX(rmsnorm_fwd));
  m.impl("rmsnorm_fwd_noalloc", TORCH_BOX(rmsnorm_fwd_noalloc));
  m.impl("rmsnorm_bwd_add", TORCH_BOX(rmsnorm_bwd_add));
}
