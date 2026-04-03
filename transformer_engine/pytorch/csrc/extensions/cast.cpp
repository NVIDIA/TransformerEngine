/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

// ============================================================================
// Quantize: input (hp) → output (fp8/fp4)
// Covers delayed scaling (pre-computed scale) and the quantize-only step
// of the FUSED_NORM_AMAX path (scale computed from amax).
// ============================================================================

void quantize(Tensor input, Tensor output_data, int64_t output_te_dtype,
              std::optional<Tensor> output_amax, std::optional<Tensor> output_scale,
              std::optional<Tensor> output_scale_inv, int64_t scaling_mode, bool force_pow_2_scales,
              double amax_epsilon, std::optional<Tensor> noop_flag, bool nvfp4_2d_quantization) {
  auto shape = getStableTensorShape(input);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);

  QuantizationConfigWrapper quant_config;
  std::optional<TensorWrapper> noop_cu;
  if (noop_flag.has_value()) {
    noop_cu.emplace(makeTransformerEngineTensor(noop_flag.value()));
    quant_config.set_noop_tensor(noop_cu->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(static_cast<float>(amax_epsilon));
  quant_config.set_nvfp4_2d_quantization(nvfp4_2d_quantization);

  auto stream = getCurrentCUDAStreamRaw(input.get_device_index());
  nvte_quantize_v2(input_cu.data(), output_cu.data(), quant_config, stream);
}

// ============================================================================
// Quantize with both rowwise and columnwise output in one fused kernel call.
// This is needed for MXFP8 (and potentially other formats) where
// nvte_quantize_v2 can fill both rowwise and columnwise buffers simultaneously.
// ============================================================================

void quantize_bidirectional(Tensor input, Tensor output_rowwise_data, int64_t output_te_dtype,
                            std::optional<Tensor> output_amax, std::optional<Tensor> output_scale,
                            Tensor output_rowwise_scale_inv, Tensor output_columnwise_data,
                            Tensor output_columnwise_scale_inv, int64_t scaling_mode,
                            bool force_pow_2_scales, double amax_epsilon,
                            std::optional<Tensor> noop_flag, bool nvfp4_2d_quantization) {
  auto shape = getStableTensorShape(input);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto input_cu = makeTransformerEngineTensor(input);

  // Build output TensorWrapper with both rowwise and columnwise data
  TensorWrapper output_cu(nvte_scaling);
  output_cu.set_rowwise_data(output_rowwise_data.data_ptr(), te_dtype, shape);

  // Determine scale_inv dtype from scaling mode
  DType si_dtype = DType::kFloat32;
  if (nvte_scaling == NVTE_MXFP8_1D_SCALING) {
    si_dtype = DType::kFloat8E8M0;
  } else if (nvte_scaling == NVTE_NVFP4_1D_SCALING) {
    si_dtype = DType::kFloat8E4M3;
  }

  auto rw_si_shape = getStableTensorShape(output_rowwise_scale_inv);
  output_cu.set_rowwise_scale_inv(output_rowwise_scale_inv.data_ptr(), si_dtype, rw_si_shape);

  // For MXFP8, columnwise data has the same logical shape as the input [M, K].
  // For NVFP4/block-scaling, columnwise data is the transpose [K, M].
  // Use the actual tensor shape to let TensorWrapper::shape() compute correctly.
  auto cw_data_shape = getStableTensorShape(output_columnwise_data);
  // FP4 data is packed (2 elements per byte). Double last dim for logical shape.
  if (is_fp4_dtype(te_dtype) && !cw_data_shape.empty()) {
    cw_data_shape.back() *= 2;
  }
  output_cu.set_columnwise_data(output_columnwise_data.data_ptr(), te_dtype, cw_data_shape);

  auto cw_si_shape = getStableTensorShape(output_columnwise_scale_inv);
  output_cu.set_columnwise_scale_inv(output_columnwise_scale_inv.data_ptr(), si_dtype, cw_si_shape);

  const std::vector<size_t> scalar_shape{1};
  if (output_amax.has_value() && output_amax->numel() > 0) {
    output_cu.set_amax(output_amax->data_ptr(), DType::kFloat32, scalar_shape);
  }
  if (output_scale.has_value() && output_scale->numel() > 0) {
    output_cu.set_scale(output_scale->data_ptr(), DType::kFloat32, scalar_shape);
  }

  QuantizationConfigWrapper quant_config;
  std::optional<TensorWrapper> noop_cu;
  if (noop_flag.has_value()) {
    noop_cu.emplace(makeTransformerEngineTensor(noop_flag.value()));
    quant_config.set_noop_tensor(noop_cu->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(static_cast<float>(amax_epsilon));
  quant_config.set_nvfp4_2d_quantization(nvfp4_2d_quantization);

  auto stream = getCurrentCUDAStreamRaw(input.get_device_index());
  nvte_quantize_v2(input_cu.data(), output_cu.data(), quant_config, stream);
}

// ============================================================================
// Quantize with amax: compute amax → compute scale → quantize
// This is the full current-scaling quantization pipeline.
// ============================================================================

void quantize_with_amax(Tensor input, Tensor output_data, int64_t output_te_dtype,
                        Tensor output_amax, Tensor output_scale,
                        std::optional<Tensor> output_scale_inv, int64_t scaling_mode,
                        bool force_pow_2_scales, double amax_epsilon,
                        std::optional<Tensor> noop_flag) {
  auto shape = getStableTensorShape(input);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);

  QuantizationConfigWrapper quant_config;
  std::optional<TensorWrapper> noop_cu;
  if (noop_flag.has_value()) {
    noop_cu.emplace(makeTransformerEngineTensor(noop_flag.value()));
    quant_config.set_noop_tensor(noop_cu->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(static_cast<float>(amax_epsilon));

  auto stream = getCurrentCUDAStreamRaw(input.get_device_index());

  // Step 1: Compute amax from input, store in output's amax buffer
  nvte_compute_amax_with_config(input_cu.data(), output_cu.data(), quant_config, stream);

  // Step 2: Compute scale from amax
  nvte_compute_scale_from_amax(output_cu.data(), quant_config, stream);

  // Step 3: Quantize using computed scale
  // Clear amax before quantize to avoid atomic conflicts
  output_cu.set_amax(nullptr, DType::kFloat32, std::vector<size_t>{1});
  nvte_quantize_v2(input_cu.data(), output_cu.data(), quant_config, stream);
}

// ============================================================================
// Quantize from pre-computed amax (skip amax computation)
// Used after FUSED_NORM_AMAX path where norm kernel already wrote amax.
// ============================================================================

void quantize_from_amax(Tensor input, Tensor output_data, int64_t output_te_dtype,
                        Tensor output_amax, Tensor output_scale,
                        std::optional<Tensor> output_scale_inv, int64_t scaling_mode,
                        bool force_pow_2_scales, double amax_epsilon,
                        std::optional<Tensor> noop_flag) {
  auto shape = getStableTensorShape(input);
  auto te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeQuantizedTensorWrapper(output_data, te_dtype, shape, output_amax,
                                              output_scale, output_scale_inv, nvte_scaling);

  QuantizationConfigWrapper quant_config;
  std::optional<TensorWrapper> noop_cu;
  if (noop_flag.has_value()) {
    noop_cu.emplace(makeTransformerEngineTensor(noop_flag.value()));
    quant_config.set_noop_tensor(noop_cu->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(static_cast<float>(amax_epsilon));

  auto stream = getCurrentCUDAStreamRaw(input.get_device_index());

  // Amax is already computed (by fused norm kernel) — just compute scale + quantize
  nvte_compute_scale_from_amax(output_cu.data(), quant_config, stream);
  output_cu.set_amax(nullptr, DType::kFloat32, std::vector<size_t>{1});
  nvte_quantize_v2(input_cu.data(), output_cu.data(), quant_config, stream);
}

// ============================================================================
// Dequantize: input (fp8) → output (hp)
// ============================================================================

Tensor dequantize(Tensor input_data, int64_t input_te_dtype, std::optional<Tensor> input_scale_inv,
                  std::optional<Tensor> input_amax, int64_t scaling_mode, int64_t output_te_dtype) {
  auto shape = getStableTensorShape(input_data);
  auto in_te_dtype = static_cast<DType>(input_te_dtype);
  auto out_te_dtype = static_cast<DType>(output_te_dtype);
  auto nvte_scaling = static_cast<NVTEScalingMode>(scaling_mode);

  // FP4 data is packed (2 elements per byte). Report logical element count.
  if (is_fp4_dtype(in_te_dtype) && !shape.empty()) {
    shape.back() *= 2;
  }

  auto input_cu = makeQuantizedTensorWrapper(input_data, in_te_dtype, shape, input_amax,
                                             std::nullopt, input_scale_inv, nvte_scaling);

  auto output = allocateStableTensor(std::vector<int64_t>(shape.begin(), shape.end()), out_te_dtype,
                                     input_data.get_device_index());
  auto output_cu = makeTransformerEngineTensor(output);

  nvte_dequantize(input_cu.data(), output_cu.data(),
                  getCurrentCUDAStreamRaw(input_data.get_device_index()));

  return output;
}

// ============================================================================
// Group Quantize: input grouped tensor (hp) → output grouped tensor (fp8)
// Wraps nvte_group_quantize for fused per-group quantization.
// ============================================================================

void group_quantize(
    // Input grouped tensor (high-precision)
    Tensor input_rowwise_data, Tensor input_first_dims, Tensor input_tensor_offsets,
    int64_t input_te_dtype, int64_t input_logical_0, int64_t input_logical_1, int64_t num_tensors,
    // Output grouped tensor (quantized)
    std::optional<Tensor> output_rowwise_data, std::optional<Tensor> output_colwise_data,
    std::optional<Tensor> output_scale_inv, std::optional<Tensor> output_colwise_scale_inv,
    Tensor output_first_dims, Tensor output_tensor_offsets, int64_t output_te_dtype,
    int64_t output_scaling_mode, int64_t output_logical_0, int64_t output_logical_1,
    bool output_swizzled) {
  auto in_dtype = static_cast<DType>(input_te_dtype);
  auto out_dtype = static_cast<DType>(output_te_dtype);
  auto out_sm = static_cast<NVTEScalingMode>(output_scaling_mode);
  DType si_dtype = DType::kFloat32;
  if (out_sm == NVTE_MXFP8_1D_SCALING) {
    si_dtype = DType::kFloat8E8M0;
  } else if (out_sm == NVTE_NVFP4_1D_SCALING) {
    si_dtype = DType::kFloat8E4M3;
  }

  auto n = static_cast<size_t>(num_tensors);

  // Build input GroupedTensorWrapper (unquantized, no scaling)
  std::vector<size_t> in_logical = {static_cast<size_t>(input_logical_0),
                                    static_cast<size_t>(input_logical_1)};
  GroupedTensorWrapper input_gt(n, in_logical, NVTE_DELAYED_TENSOR_SCALING);
  {
    auto shape = getStableTensorShape(input_rowwise_data);
    input_gt.set_rowwise_data(input_rowwise_data.data_ptr(), in_dtype, shape);
  }
  {
    auto shape = getStableTensorShape(input_first_dims);
    input_gt.set_first_dims(input_first_dims.data_ptr(), DType::kInt64, shape);
  }
  {
    auto shape = getStableTensorShape(input_tensor_offsets);
    input_gt.set_tensor_offsets(input_tensor_offsets.data_ptr(), DType::kInt64, shape);
  }

  // Build output GroupedTensorWrapper (quantized)
  std::vector<size_t> out_logical = {static_cast<size_t>(output_logical_0),
                                     static_cast<size_t>(output_logical_1)};
  GroupedTensorWrapper output_gt(n, out_logical, out_sm);
  if (output_rowwise_data.has_value() && output_rowwise_data->numel() > 0) {
    auto shape = getStableTensorShape(*output_rowwise_data);
    output_gt.set_rowwise_data(output_rowwise_data->data_ptr(), out_dtype, shape);
  }
  if (output_colwise_data.has_value() && output_colwise_data->numel() > 0) {
    auto shape = getStableTensorShape(*output_colwise_data);
    output_gt.set_columnwise_data(output_colwise_data->data_ptr(), out_dtype, shape);
  }
  if (output_scale_inv.has_value() && output_scale_inv->numel() > 0) {
    auto shape = getStableTensorShape(*output_scale_inv);
    output_gt.set_rowwise_scale_inv(output_scale_inv->data_ptr(), si_dtype, shape);
  }
  if (output_colwise_scale_inv.has_value() && output_colwise_scale_inv->numel() > 0) {
    auto shape = getStableTensorShape(*output_colwise_scale_inv);
    output_gt.set_columnwise_scale_inv(output_colwise_scale_inv->data_ptr(), si_dtype, shape);
  }
  {
    auto shape = getStableTensorShape(output_first_dims);
    output_gt.set_first_dims(output_first_dims.data_ptr(), DType::kInt64, shape);
  }
  {
    auto shape = getStableTensorShape(output_tensor_offsets);
    output_gt.set_tensor_offsets(output_tensor_offsets.data_ptr(), DType::kInt64, shape);
  }
  output_gt.set_with_gemm_swizzled_scales(output_swizzled);

  auto stream = getCurrentCUDAStreamRaw(input_rowwise_data.get_device_index());
  nvte_group_quantize(input_gt.data(), output_gt.data(), stream);
}

}  // namespace transformer_engine::pytorch::stable

STABLE_TORCH_LIBRARY_FRAGMENT(transformer_engine_stable, m) {
  m.def(
      "quantize(Tensor input, Tensor output_data, int output_te_dtype, Tensor? output_amax, "
      "Tensor? output_scale, Tensor? output_scale_inv, int scaling_mode, bool force_pow_2_scales, "
      "float amax_epsilon, Tensor? noop_flag, bool nvfp4_2d_quantization=False) -> ()");
  m.def(
      "quantize_with_amax(Tensor input, Tensor output_data, int output_te_dtype, Tensor "
      "output_amax, Tensor output_scale, Tensor? output_scale_inv, int scaling_mode, bool "
      "force_pow_2_scales, float amax_epsilon, Tensor? noop_flag) -> ()");
  m.def(
      "quantize_from_amax(Tensor input, Tensor output_data, int output_te_dtype, Tensor "
      "output_amax, Tensor output_scale, Tensor? output_scale_inv, int scaling_mode, bool "
      "force_pow_2_scales, float amax_epsilon, Tensor? noop_flag) -> ()");
  m.def(
      "quantize_bidirectional(Tensor input, Tensor output_rowwise_data, int output_te_dtype, "
      "Tensor? output_amax, Tensor? output_scale, Tensor output_rowwise_scale_inv, "
      "Tensor output_columnwise_data, Tensor output_columnwise_scale_inv, "
      "int scaling_mode, bool force_pow_2_scales, float amax_epsilon, Tensor? noop_flag, "
      "bool nvfp4_2d_quantization=False) -> ()");
  m.def(
      "dequantize(Tensor input_data, int input_te_dtype, Tensor? input_scale_inv, Tensor? "
      "input_amax, int scaling_mode, int output_te_dtype) -> Tensor");
  // group_quantize: input(7) + output(11) = 18 args
  m.def(
      "group_quantize("
      "Tensor input_rowwise_data, "
      "Tensor input_first_dims, "
      "Tensor input_tensor_offsets, "
      "int input_te_dtype, "
      "int input_logical_0, "
      "int input_logical_1, "
      "int num_tensors, "
      "Tensor? output_rowwise_data, "
      "Tensor? output_colwise_data, "
      "Tensor? output_scale_inv, "
      "Tensor? output_colwise_scale_inv, "
      "Tensor output_first_dims, "
      "Tensor output_tensor_offsets, "
      "int output_te_dtype, "
      "int output_scaling_mode, "
      "int output_logical_0, "
      "int output_logical_1, "
      "bool output_swizzled"
      ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  using namespace transformer_engine::pytorch::stable;
  m.impl("quantize", TORCH_BOX(quantize));
  m.impl("quantize_with_amax", TORCH_BOX(quantize_with_amax));
  m.impl("quantize_from_amax", TORCH_BOX(quantize_from_amax));
  m.impl("quantize_bidirectional", TORCH_BOX(quantize_bidirectional));
  m.impl("dequantize", TORCH_BOX(dequantize));
  m.impl("group_quantize", TORCH_BOX(group_quantize));
}
