/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "transformer_engine/transformer_engine.h"

void swizzle_scaling_factors(transformer_engine::TensorWrapper& input, bool rowwise) {
  using namespace transformer_engine::pytorch;

  if (input.scaling_mode() == NVTE_INVALID_SCALING) {
    NVTE_ERROR("Invalid scaling mode for swizzle.");
  } else if (input.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
    return;
  }

  NVTE_CHECK(input.element_size() == 1, "8-bit input required for swizzling scaling factors.");

  NVTEBasicTensor scale_inv;
  if (rowwise) {
    scale_inv = input.get_rowwise_scale_inv();
  } else {
    scale_inv = input.get_columnwise_scale_inv();
  }

  auto input_shape = nvte_shape_to_vector(input.shape());
  auto scale_inv_shape = nvte_shape_to_vector(scale_inv.shape);

  // Allocate memory for swizzled output.
  auto options = at::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
  std::vector<int64_t> scale_inv_shape_int;
  for (size_t i = 0; i < scale_inv_shape.size(); ++i) {
    scale_inv_shape_int.push_back(static_cast<int64_t>(scale_inv_shape[i]));
  }
  auto swizzled_scale_inv = at::empty(scale_inv_shape_int, options);
  void* scale_inv_dptr = scale_inv.data_ptr;
  void* swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);

  // Reconstruct input only to avoid swizzling both directions if not needed.
  // Use any 8 bit type, it's irrelevant.
  transformer_engine::TensorWrapper input_cu(NVTE_MXFP8_1D_SCALING);
  transformer_engine::TensorWrapper output_cu(NVTE_MXFP8_1D_SCALING);
  if (rowwise) {
    input_cu.set_rowwise_data(input.dptr(), DType::kFloat8E4M3, input_shape);
    input_cu.set_rowwise_scale_inv(scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
    output_cu.set_rowwise_data(input.dptr(), DType::kFloat8E4M3, input_shape);
    output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  } else {
    input_cu.set_columnwise_data(input.dptr(), DType::kFloat8E4M3, input_shape);
    input_cu.set_columnwise_scale_inv(scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
    output_cu.set_columnwise_data(input.dptr(), DType::kFloat8E4M3, input_shape);
    output_cu.set_columnwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0,
                                       scale_inv_shape);
  }

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  if (rowwise) {
    input.set_rowwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  } else {
    input.set_columnwise_scale_inv(swizzled_scale_inv_dptr, DType::kFloat8E8M0, scale_inv_shape);
  }
}

at::Tensor rowwise_swizzle(at::Tensor input, at::Tensor scale_inv) {
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input.element_size() == 1, "8-bit input required for swizzling scaling factors.");

  auto options = at::TensorOptions().dtype(scale_inv.dtype()).device(torch::kCUDA);
  auto swizzled_scale_inv = at::empty_like(scale_inv, options);

  void* scale_inv_dptr = getDataPtr(scale_inv, 0);
  void* swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), getTensorShape(input),
                                              DType::kFloat8E4M3, nullptr, nullptr, scale_inv_dptr,
                                              getTensorShape(scale_inv), NVTE_MXFP8_1D_SCALING);
  auto output_cu = makeTransformerEngineTensor(
      input.data_ptr(), getTensorShape(input), DType::kFloat8E4M3, nullptr, nullptr,
      swizzled_scale_inv_dptr, getTensorShape(swizzled_scale_inv), NVTE_MXFP8_1D_SCALING);

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return swizzled_scale_inv;
}

at::Tensor columnwise_swizzle(at::Tensor input, at::Tensor scale_inv) {
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input.element_size() == 1, "8-bit input required for swizzling scaling factors.");

  auto options = at::TensorOptions().dtype(scale_inv.dtype()).device(torch::kCUDA);
  auto swizzled_scale_inv = at::empty_like(scale_inv, options);

  // Return immediately if tensor is empty
  if (scale_inv.numel() == 0) {
    return swizzled_scale_inv;
  }

  void* scale_inv_dptr = getDataPtr(scale_inv, 0);
  void* swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);

  auto input_cu = makeTransformerEngineTensor(
      nullptr, input.data_ptr(), {1}, getTensorShape(input), DType::kFloat8E4M3, nullptr, nullptr,
      nullptr, scale_inv_dptr, {1}, getTensorShape(scale_inv), NVTE_MXFP8_1D_SCALING);
  auto output_cu = makeTransformerEngineTensor(
      nullptr, input.data_ptr(), {1}, getTensorShape(input), DType::kFloat8E4M3, nullptr, nullptr,
      nullptr, swizzled_scale_inv_dptr, {1}, getTensorShape(swizzled_scale_inv),
      NVTE_MXFP8_1D_SCALING);

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return swizzled_scale_inv;
}
