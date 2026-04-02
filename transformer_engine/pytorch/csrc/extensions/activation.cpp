/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// Generic activation forward — no-alloc variant
//
// Handles all fusion paths:
//   FULLY_FUSED:  output is quantized, kernel writes directly
//   NORM+AMAX:    output is hp + amax attached, kernel computes act + amax
//   UNFUSED:      output is hp, Python calls quantize_from_amax separately
//
// The Python shim selects the path by choosing output buffer configuration.
// ============================================================================

// Forward activation with pre-allocated output buffers (no quantizer dispatch)
// shape_divisor: 2 for GLU variants (output last dim = input last dim / 2), 1 otherwise
void activation_fwd_noalloc(Tensor input, Tensor output_data, int64_t output_te_dtype,
                            std::optional<Tensor> output_amax, std::optional<Tensor> output_scale,
                            std::optional<Tensor> output_scale_inv, int64_t scaling_mode,
                            int64_t activation_type) {
  auto input_ = torch::stable::contiguous(input);
  auto input_cu = makeTransformerEngineTensor(input_);
  auto shape = getStableTensorShape(input_);
  // Output shape may differ (GLU halves last dim) — use output_data's shape
  auto out_shape = getStableTensorShape(output_data);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, out_shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);

  auto stream = getCurrentCUDAStreamRaw(input_.get_device_index());

  // Dispatch activation type
  using ActFn = void (*)(const NVTETensor, NVTETensor, cudaStream_t);
  // Activation type enum matches the order in registration
  static constexpr ActFn act_table[] = {
      nvte_gelu,  nvte_glu,   nvte_geglu,  nvte_qgelu, nvte_qgeglu, nvte_relu,
      nvte_reglu, nvte_srelu, nvte_sreglu, nvte_silu,  nvte_swiglu,
  };
  constexpr int num_acts = sizeof(act_table) / sizeof(act_table[0]);
  STD_TORCH_CHECK(activation_type >= 0 && activation_type < num_acts,
                  "Invalid activation_type: ", activation_type);
  act_table[activation_type](input_cu.data(), output_cu.data(), stream);
}

// Backward activation (grad_output, input → grad_input)
// Same noalloc pattern — output buffer may be quantized
void dactivation_noalloc(Tensor grad_output, Tensor input, Tensor grad_input_data,
                         int64_t grad_input_te_dtype, std::optional<Tensor> grad_input_amax,
                         std::optional<Tensor> grad_input_scale,
                         std::optional<Tensor> grad_input_scale_inv, int64_t scaling_mode,
                         int64_t activation_type) {
  auto grad_output_ = torch::stable::contiguous(grad_output);
  auto input_ = torch::stable::contiguous(input);

  auto grad_output_cu = makeTransformerEngineTensor(grad_output_);
  auto input_cu = makeTransformerEngineTensor(input_);
  auto grad_shape = getStableTensorShape(input_);
  auto te_dtype = static_cast<DType>(grad_input_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto grad_input_cu =
      makeQuantizedTensorWrapper(grad_input_data, te_dtype, grad_shape, grad_input_amax,
                                 grad_input_scale, grad_input_scale_inv, nvte_scaling);

  auto stream = getCurrentCUDAStreamRaw(input_.get_device_index());

  using DActFn = void (*)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t);
  static constexpr DActFn dact_table[] = {
      nvte_dgelu,  nvte_dglu,   nvte_dgeglu,  nvte_dqgelu, nvte_dqgeglu, nvte_drelu,
      nvte_dreglu, nvte_dsrelu, nvte_dsreglu, nvte_dsilu,  nvte_dswiglu,
  };
  constexpr int num_acts = sizeof(dact_table) / sizeof(dact_table[0]);
  STD_TORCH_CHECK(activation_type >= 0 && activation_type < num_acts,
                  "Invalid activation_type: ", activation_type);
  dact_table[activation_type](grad_output_cu.data(), input_cu.data(), grad_input_cu.data(), stream);
}

// Clamped activations (with extra float params)
void clamped_activation_fwd_noalloc(Tensor input, Tensor output_data, int64_t output_te_dtype,
                                    std::optional<Tensor> output_amax,
                                    std::optional<Tensor> output_scale,
                                    std::optional<Tensor> output_scale_inv, int64_t scaling_mode,
                                    double limit, double alpha, int64_t activation_type) {
  auto input_ = torch::stable::contiguous(input);
  auto input_cu = makeTransformerEngineTensor(input_);
  auto out_shape = getStableTensorShape(output_data);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);
  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, out_shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);
  auto stream = getCurrentCUDAStreamRaw(input_.get_device_index());

  // 0 = clamped_swiglu
  if (activation_type == 0) {
    nvte_clamped_swiglu(input_cu.data(), output_cu.data(), static_cast<float>(limit),
                        static_cast<float>(alpha), stream);
  } else {
    NVTE_ERROR("Invalid clamped activation_type: ", activation_type);
  }
}

void clamped_dactivation_noalloc(Tensor grad_output, Tensor input, Tensor grad_input_data,
                                 int64_t grad_input_te_dtype, std::optional<Tensor> grad_input_amax,
                                 std::optional<Tensor> grad_input_scale,
                                 std::optional<Tensor> grad_input_scale_inv, int64_t scaling_mode,
                                 double limit, double alpha, int64_t activation_type) {
  auto grad_output_ = torch::stable::contiguous(grad_output);
  auto input_ = torch::stable::contiguous(input);
  auto grad_output_cu = makeTransformerEngineTensor(grad_output_);
  auto input_cu = makeTransformerEngineTensor(input_);
  auto grad_shape = getStableTensorShape(input_);
  auto te_dtype = static_cast<DType>(grad_input_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);
  auto grad_input_cu =
      makeQuantizedTensorWrapper(grad_input_data, te_dtype, grad_shape, grad_input_amax,
                                 grad_input_scale, grad_input_scale_inv, nvte_scaling);
  auto stream = getCurrentCUDAStreamRaw(input_.get_device_index());

  if (activation_type == 0) {
    nvte_clamped_dswiglu(grad_output_cu.data(), input_cu.data(), grad_input_cu.data(),
                         static_cast<float>(limit), static_cast<float>(alpha), stream);
  } else {
    NVTE_ERROR("Invalid clamped activation_type: ", activation_type);
  }
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  // activation_type: 0=gelu, 1=glu, 2=geglu, 3=qgelu, 4=qgeglu,
  //   5=relu, 6=reglu, 7=srelu, 8=sreglu, 9=silu, 10=swiglu
  m.def(
      "activation_fwd_noalloc(Tensor input, Tensor output_data, int output_te_dtype, Tensor? "
      "output_amax, Tensor? output_scale, Tensor? output_scale_inv, int scaling_mode, int "
      "activation_type) -> ()");
  m.def(
      "dactivation_noalloc(Tensor grad_output, Tensor input, Tensor grad_input_data, int "
      "grad_input_te_dtype, Tensor? grad_input_amax, Tensor? grad_input_scale, Tensor? "
      "grad_input_scale_inv, int scaling_mode, int activation_type) -> ()");
  m.def(
      "clamped_activation_fwd_noalloc(Tensor input, Tensor output_data, int output_te_dtype, "
      "Tensor? output_amax, Tensor? output_scale, Tensor? output_scale_inv, int scaling_mode, "
      "float limit, float alpha, int activation_type) -> ()");
  m.def(
      "clamped_dactivation_noalloc(Tensor grad_output, Tensor input, Tensor grad_input_data, int "
      "grad_input_te_dtype, Tensor? grad_input_amax, Tensor? grad_input_scale, Tensor? "
      "grad_input_scale_inv, int scaling_mode, float limit, float alpha, int activation_type) -> "
      "()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("activation_fwd_noalloc", TORCH_BOX(activation_fwd_noalloc));
  m.impl("dactivation_noalloc", TORCH_BOX(dactivation_noalloc));
  m.impl("clamped_activation_fwd_noalloc", TORCH_BOX(clamped_activation_fwd_noalloc));
  m.impl("clamped_dactivation_noalloc", TORCH_BOX(clamped_dactivation_noalloc));
}
