/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util.h"

#include "common.h"

std::optional<at::Tensor> swizzle_scaling_factors(transformer_engine::TensorWrapper& input,
                                                  bool rowwise) {
  using namespace transformer_engine::pytorch;

  if (input.scaling_mode() == NVTE_INVALID_SCALING) {
    NVTE_ERROR("Invalid scaling mode for swizzle.");
  } else if (input.scaling_mode() != NVTE_MXFP8_1D_SCALING &&
             input.scaling_mode() != NVTE_NVFP4_1D_SCALING) {
    return std::nullopt;
  }

  NVTE_CHECK(input.element_size_bits() == 4 || input.element_size_bits() == 8,
             "4-bit or 8-bit input required for swizzling scaling factors.");

  const auto nvfp4 = input.scaling_mode() == NVTE_NVFP4_1D_SCALING;

  NVTEBasicTensor scale_inv;
  NVTEShape nvte_input_shape;
  if (rowwise) {
    nvte_input_shape = input.shape();
    scale_inv = input.get_rowwise_scale_inv();
  } else {
    nvte_input_shape = input.get_columnwise_data().shape;
    scale_inv = input.get_columnwise_scale_inv();
  }

  auto input_shape = nvte_shape_to_vector(nvte_input_shape);
  auto scale_inv_shape = nvte_shape_to_vector(scale_inv.shape);

  NVTE_CHECK(input_shape.size() >= 2, "Wrong ndims for swizzle input shape.");

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
  // The specific dtype used is irrelevant, just needs to be correct bits.
  transformer_engine::TensorWrapper input_cu(input.scaling_mode());
  transformer_engine::TensorWrapper output_cu(input.scaling_mode());

  const auto input_dtype =
      (nvfp4) ? transformer_engine::DType::kFloat4E2M1 : transformer_engine::DType::kFloat8E4M3;
  const auto scale_inv_dtype =
      (nvfp4) ? transformer_engine::DType::kFloat8E4M3 : transformer_engine::DType::kFloat8E8M0;

  if (rowwise) {
    input_cu.set_rowwise_data(input.dptr(), input_dtype, input_shape);
    input_cu.set_rowwise_scale_inv(scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
    output_cu.set_rowwise_data(input.dptr(), input_dtype, input_shape);
    output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
  } else {
    input_cu.set_columnwise_data(input.columnwise_dptr(), input_dtype, input_shape);
    input_cu.set_columnwise_scale_inv(scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
    output_cu.set_columnwise_data(input.columnwise_dptr(), input_dtype, input_shape);
    output_cu.set_columnwise_scale_inv(swizzled_scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
  }

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  if (rowwise) {
    input.set_rowwise_scale_inv(swizzled_scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
  } else {
    input.set_columnwise_scale_inv(swizzled_scale_inv_dptr, scale_inv_dtype, scale_inv_shape);
  }

  return swizzled_scale_inv;
}

std::optional<at::Tensor> multi_tensor_swizzle_scaling_factors(
    std::vector<transformer_engine::TensorWrapper>& tensors, bool rowwise) {
  using namespace transformer_engine::pytorch;

  if (tensors.empty()) {
    return std::nullopt;
  }

  bool all_same_scaling_mode = std::all_of(
      tensors.cbegin(), tensors.cend(), [&tensors](const transformer_engine::TensorWrapper& val) {
        return val.scaling_mode() == tensors.front().scaling_mode();
      });
  NVTE_CHECK(all_same_scaling_mode, "Scaling mode of the input tensors must be the same.");

  if (tensors.front().scaling_mode() == NVTE_INVALID_SCALING) {
    NVTE_ERROR("Invalid scaling mode for swizzle.");
  } else if (tensors.front().scaling_mode() != NVTE_MXFP8_1D_SCALING) {
    return std::nullopt;
  }

  std::vector<transformer_engine::TensorWrapper> wrappers;
  std::vector<NVTETensor> input_tensors, output_tensors;

  // Collect scale_inv shapes and calculate buffer size and offsets for scale_invs
  std::vector<std::vector<size_t>> scale_inv_shapes;
  std::vector<void*> scale_inv_dptrs;
  size_t buffer_size = 0;
  std::vector<size_t> scale_inv_offsets;
  constexpr size_t scale_elem_size = 1;
  for (auto& tensor : tensors) {
    NVTEBasicTensor scale_inv;
    if (rowwise) {
      scale_inv = tensor.get_rowwise_scale_inv();
    } else {
      scale_inv = tensor.get_columnwise_scale_inv();
    }
    auto scale_inv_shape = nvte_shape_to_vector(scale_inv.shape);
    buffer_size = roundup(buffer_size, 16);  // align to 16B
    scale_inv_offsets.push_back(buffer_size);
    buffer_size += product(scale_inv_shape) * scale_elem_size;
    scale_inv_shapes.emplace_back(scale_inv_shape);
    scale_inv_dptrs.push_back(scale_inv.data_ptr);
  }

  // Allocate full buffer
  auto buffer = at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8));

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& tensor = tensors[i];
    void* scale_inv_dptr = scale_inv_dptrs[i];
    void* swizzled_scale_inv_dptr = getDataPtr(buffer, scale_inv_offsets[i]);
    auto input_shape = nvte_shape_to_vector(tensor.shape());

    // Reconstruct input only to avoid swizzling both directions if not needed.
    // Use any 8 bit type, it's irrelevant.
    transformer_engine::TensorWrapper input_cu(NVTE_MXFP8_1D_SCALING);
    transformer_engine::TensorWrapper output_cu(NVTE_MXFP8_1D_SCALING);
    if (rowwise) {
      input_cu.set_rowwise_data(tensor.dptr(), transformer_engine::DType::kFloat8E4M3, input_shape);
      input_cu.set_rowwise_scale_inv(scale_inv_dptr, transformer_engine::DType::kFloat8E8M0,
                                     scale_inv_shapes[i]);
      output_cu.set_rowwise_data(tensor.dptr(), transformer_engine::DType::kFloat8E4M3,
                                 input_shape);
      output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr,
                                      transformer_engine::DType::kFloat8E8M0, scale_inv_shapes[i]);
      // Set the swizzled scaling factor to the original tensor.
      tensor.set_rowwise_scale_inv(swizzled_scale_inv_dptr, transformer_engine::DType::kFloat8E8M0,
                                   scale_inv_shapes[i]);
    } else {
      input_cu.set_columnwise_data(tensor.columnwise_dptr(), transformer_engine::DType::kFloat8E4M3,
                                   input_shape);
      input_cu.set_columnwise_scale_inv(scale_inv_dptr, transformer_engine::DType::kFloat8E8M0,
                                        scale_inv_shapes[i]);
      output_cu.set_columnwise_data(tensor.columnwise_dptr(),
                                    transformer_engine::DType::kFloat8E4M3, input_shape);
      output_cu.set_columnwise_scale_inv(
          swizzled_scale_inv_dptr, transformer_engine::DType::kFloat8E8M0, scale_inv_shapes[i]);
      // Set the swizzled scaling factor to the original tensor.
      tensor.set_columnwise_scale_inv(swizzled_scale_inv_dptr,
                                      transformer_engine::DType::kFloat8E8M0, scale_inv_shapes[i]);
    }

    input_tensors.emplace_back(input_cu.data());
    output_tensors.emplace_back(output_cu.data());
    wrappers.emplace_back(std::move(input_cu));
    wrappers.emplace_back(std::move(output_cu));
  }

  // Launch kernel
  nvte_multi_tensor_swizzle_scaling_factors(input_tensors.data(), output_tensors.data(),
                                            input_tensors.size(), at::cuda::getCurrentCUDAStream());

  return buffer;
}
