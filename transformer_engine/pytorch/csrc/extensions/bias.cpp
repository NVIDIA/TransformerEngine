/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// bgrad_quantize: compute grad_bias and optionally quantize grad_output
//
// Fused kernel: nvte_quantize_dbias computes both dbias and quantized grad_input
// Unfused: at::sum for dbias + separate quantize
// ============================================================================

void bgrad_quantize_noalloc(Tensor grad_output,
                            Tensor grad_bias,  // pre-allocated [hidden_size]
                            Tensor grad_input_data, int64_t grad_input_te_dtype,
                            std::optional<Tensor> grad_input_amax,
                            std::optional<Tensor> grad_input_scale,
                            std::optional<Tensor> grad_input_scale_inv, int64_t scaling_mode) {
  auto grad_output_ = torch::stable::contiguous(grad_output);
  auto grad_output_cu = makeTransformerEngineTensor(grad_output_);
  auto grad_bias_cu = makeTransformerEngineTensor(grad_bias);

  auto shape = getStableTensorShape(grad_output_);
  auto te_dtype = static_cast<DType>(grad_input_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto grad_input_cu =
      makeQuantizedTensorWrapper(grad_input_data, te_dtype, shape, grad_input_amax,
                                 grad_input_scale, grad_input_scale_inv, nvte_scaling);

  auto device_idx = grad_output_.get_device_index();
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  TensorWrapper workspace;

  // First call: query workspace
  nvte_quantize_dbias(grad_output_cu.data(), grad_input_cu.data(), grad_bias_cu.data(),
                      workspace.data(), stream);

  // workspace_data must outlive the second kernel call — hoist out of if block.
  Tensor workspace_data;
  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  if (ws_shape.ndim > 0 && workspace.numel() > 0) {
    workspace_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
    workspace = makeTransformerEngineTensor(
        workspace_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype);
  }

  // Second call: compute
  nvte_quantize_dbias(grad_output_cu.data(), grad_input_cu.data(), grad_bias_cu.data(),
                      workspace.data(), stream);
}

// ============================================================================
// Fused dact + dbias + quantize
//
// activation_type: 0=dgelu, 1=dsilu, 2=drelu, 3=dqgelu, 4=dsrelu
// Fused kernel computes dact(grad_output, act_input), dbias, and quantize in one pass
// ============================================================================

void dact_dbias_noalloc(Tensor grad_output, Tensor act_input,
                        Tensor grad_bias,  // pre-allocated [hidden_size]
                        Tensor grad_input_data, int64_t grad_input_te_dtype,
                        std::optional<Tensor> grad_input_amax,
                        std::optional<Tensor> grad_input_scale,
                        std::optional<Tensor> grad_input_scale_inv, int64_t scaling_mode,
                        int64_t activation_type) {
  auto grad_output_ = torch::stable::contiguous(grad_output);
  auto act_input_ = torch::stable::contiguous(act_input);

  auto grad_output_cu = makeTransformerEngineTensor(grad_output_);
  auto act_input_cu = makeTransformerEngineTensor(act_input_);
  auto grad_bias_cu = makeTransformerEngineTensor(grad_bias);

  auto shape = getStableTensorShape(act_input_);
  auto te_dtype = static_cast<DType>(grad_input_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto grad_input_cu =
      makeQuantizedTensorWrapper(grad_input_data, te_dtype, shape, grad_input_amax,
                                 grad_input_scale, grad_input_scale_inv, nvte_scaling);

  auto device_idx = grad_output_.get_device_index();
  auto stream = getCurrentCUDAStreamRaw(device_idx);

  // Fused dact + dbias + quantize kernel table
  using FusedFn = void (*)(const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                           cudaStream_t);
  static constexpr FusedFn fused_table[] = {
      nvte_quantize_dbias_dgelu,  nvte_quantize_dbias_dsilu,  nvte_quantize_dbias_drelu,
      nvte_quantize_dbias_dqgelu, nvte_quantize_dbias_dsrelu,
  };
  constexpr int num_fns = sizeof(fused_table) / sizeof(fused_table[0]);
  STD_TORCH_CHECK(activation_type >= 0 && activation_type < num_fns,
                  "Invalid activation_type for dact_dbias: ", activation_type);

  auto fn = fused_table[activation_type];
  TensorWrapper workspace;

  // First call: query workspace
  fn(grad_output_cu.data(), act_input_cu.data(), grad_input_cu.data(), grad_bias_cu.data(),
     workspace.data(), stream);

  // workspace_data must outlive the second kernel call — hoist out of if block.
  Tensor workspace_data;
  auto ws_shape = workspace.shape();
  auto ws_dtype = workspace.dtype();
  if (ws_shape.ndim > 0 && workspace.numel() > 0) {
    workspace_data = allocateStableTensor(
        std::vector<int64_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype, device_idx);
    workspace = makeTransformerEngineTensor(
        workspace_data.data_ptr(),
        std::vector<size_t>(ws_shape.data, ws_shape.data + ws_shape.ndim), ws_dtype);
  }

  // Second call: compute
  fn(grad_output_cu.data(), act_input_cu.data(), grad_input_cu.data(), grad_bias_cu.data(),
     workspace.data(), stream);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "bgrad_quantize_noalloc(Tensor grad_output, Tensor grad_bias, Tensor grad_input_data, int "
      "grad_input_te_dtype, Tensor? grad_input_amax, Tensor? grad_input_scale, Tensor? "
      "grad_input_scale_inv, int scaling_mode) -> ()");
  // activation_type: 0=dgelu, 1=dsilu, 2=drelu, 3=dqgelu, 4=dsrelu
  m.def(
      "dact_dbias_noalloc(Tensor grad_output, Tensor act_input, Tensor grad_bias, Tensor "
      "grad_input_data, int grad_input_te_dtype, Tensor? grad_input_amax, Tensor? "
      "grad_input_scale, Tensor? grad_input_scale_inv, int scaling_mode, int activation_type) -> "
      "()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("bgrad_quantize_noalloc", TORCH_BOX(bgrad_quantize_noalloc));
  m.impl("dact_dbias_noalloc", TORCH_BOX(dact_dbias_noalloc));
}
