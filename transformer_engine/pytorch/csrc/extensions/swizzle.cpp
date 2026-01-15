/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "common.h"
#include "common/common.h"
#include "extensions.h"
#include "pybind.h"
#include "util.h"

namespace transformer_engine {
namespace pytorch {

namespace {

void reset_tensor_data(transformer_engine::TensorWrapper &tensor, bool rowwise, bool columnwise) {
  NVTEShape shape;
  shape.ndim = 1;
  shape.data[0] = 0;
  const transformer_engine::DType dtype = transformer_engine::DType::kFloat32;
  if (rowwise) {
    tensor.set_rowwise_data(nullptr, dtype, shape);
    tensor.set_rowwise_scale_inv(nullptr, dtype, shape);
  }
  if (columnwise) {
    tensor.set_columnwise_data(nullptr, dtype, shape);
    tensor.set_columnwise_scale_inv(nullptr, dtype, shape);
  }
}

}  // namespace

std::tuple<std::optional<at::Tensor>, std::optional<at::Tensor>> swizzle_scales_for_gemm(
    transformer_engine::TensorWrapper &tensor, bool rowwise_usage, bool columnwise_usage) {
  // Return early if scale swizzling is not required
  const auto scaling_mode = tensor.scaling_mode();
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
    case NVTE_NVFP4_1D_SCALING:
      // Tensor format requires scale swizzling
      break;
    case NVTE_INVALID_SCALING:
      NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
    default:
      // Tensor format does not require scale swizzling for GEMM
      return {std::nullopt, std::nullopt};
  }

  // Return early if scales are already swizzled
  if (tensor.get_with_gemm_swizzled_scales()) {
    return {std::nullopt, std::nullopt};
  }

  // CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Swizzle row-wise scales if needed
  std::optional<at::Tensor> rowwise_scales_pyt;
  if (rowwise_usage) {
    // Buffer for unswizzled scales
    const auto input_scales_nvte = tensor.get_rowwise_scale_inv();
    void *input_scales_dptr = input_scales_nvte.data_ptr;
    const NVTEShape input_scales_shape = input_scales_nvte.shape;
    const auto scales_dtype = static_cast<DType>(input_scales_nvte.dtype);

    // Allocate buffer for swizzled scales
    const NVTEShape output_scales_shape = input_scales_shape;
    rowwise_scales_pyt = allocateSpace(input_scales_shape, scales_dtype, false);
    void *output_scales_dptr = getDataPtr(*rowwise_scales_pyt);

    // Initialize TE tensors with scales
    const auto data_nvte = tensor.get_rowwise_data();
    const auto data_dtype = static_cast<DType>(data_nvte.dtype);
    TensorWrapper input_nvte(scaling_mode);
    input_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
    input_nvte.set_rowwise_scale_inv(input_scales_dptr, scales_dtype, input_scales_shape);
    TensorWrapper output_nvte(scaling_mode);
    output_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
    output_nvte.set_rowwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
    output_nvte.set_with_gemm_swizzled_scales(true);

    // Launch kernel
    NVTE_SCOPED_GIL_RELEASE(
        { nvte_swizzle_scaling_factors(input_nvte.data(), output_nvte.data(), stream); });

    // Update tensor with swizzled scales
    tensor.set_rowwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
  }

  // Swizzle column-wise scales if needed
  std::optional<at::Tensor> columnwise_scales_pyt;
  if (columnwise_usage) {
    // Buffer for unswizzled scales
    const auto input_scales_nvte = tensor.get_columnwise_scale_inv();
    void *input_scales_dptr = input_scales_nvte.data_ptr;
    const NVTEShape input_scales_shape = input_scales_nvte.shape;
    const auto scales_dtype = static_cast<DType>(input_scales_nvte.dtype);

    // Allocate buffer for swizzled scales
    const NVTEShape output_scales_shape = input_scales_shape;
    columnwise_scales_pyt = allocateSpace(input_scales_shape, scales_dtype, false);
    void *output_scales_dptr = getDataPtr(*columnwise_scales_pyt);

    // Initialize TE tensors with scales
    const auto data_nvte = tensor.get_columnwise_data();
    const auto data_dtype = static_cast<DType>(data_nvte.dtype);
    TensorWrapper input_nvte(scaling_mode);
    input_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
    input_nvte.set_columnwise_scale_inv(input_scales_dptr, scales_dtype, input_scales_shape);
    TensorWrapper output_nvte(scaling_mode);
    output_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
    output_nvte.set_columnwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
    output_nvte.set_with_gemm_swizzled_scales(true);

    // Launch kernel
    NVTE_SCOPED_GIL_RELEASE(
        { nvte_swizzle_scaling_factors(input_nvte.data(), output_nvte.data(), stream); });

    // Update tensor with swizzled scales
    tensor.set_columnwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
  }

  // Update tensor
  reset_tensor_data(tensor, !rowwise_usage, !columnwise_usage);
  tensor.set_with_gemm_swizzled_scales(true);

  return {std::move(rowwise_scales_pyt), std::move(columnwise_scales_pyt)};
}

std::optional<at::Tensor> multi_tensor_swizzle_scales_for_gemm(
    std::vector<transformer_engine::TensorWrapper> &tensors, bool rowwise_usage,
    bool columnwise_usage) {
  // Checks and trivial cases
  NVTE_CHECK(rowwise_usage != columnwise_usage,
             "Expect exactly one of rowwise_usage=", rowwise_usage,
             " and columnwise_usage=", columnwise_usage, ".");
  if (tensors.empty()) {
    return std::nullopt;
  }
  const auto scaling_mode = tensors.front().scaling_mode();
  for (const auto &tensor : tensors) {
    NVTE_CHECK(tensor.scaling_mode() == scaling_mode, "Tensors have different scaling modes");
  }

  // Return early if scale swizzling is not required
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
    case NVTE_NVFP4_1D_SCALING:
      // Tensor format requires scale swizzling
      break;
    case NVTE_INVALID_SCALING:
      NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
    default:
      // Tensor format does not require scale swizzling for GEMM
      return std::nullopt;
  }

  // Filter out tensors that already have swizzled scales
  std::vector<TensorWrapper *> tensors_needing_swizzle;
  for (auto &tensor : tensors) {
    if (!tensor.get_with_gemm_swizzled_scales()) {
      tensors_needing_swizzle.push_back(&tensor);
    }
  }
  if (tensors_needing_swizzle.empty()) {
    return std::nullopt;
  }

  // Determine buffer size needed for swizzled scales
  std::vector<size_t> output_scales_offsets;
  size_t output_scales_bytes = 0;
  for (auto &tensor : tensors_needing_swizzle) {
    const auto scales_nvte =
        (rowwise_usage ? tensor->get_rowwise_scale_inv() : tensor->get_columnwise_scale_inv());
    const auto &shape = scales_nvte.shape;
    const auto dtype = static_cast<DType>(scales_nvte.dtype);
    const auto dtype_bits = transformer_engine::pytorch::typeToNumBits(dtype);
    const auto size = product(shape, 0, shape.ndim);
    output_scales_bytes = roundup(output_scales_bytes, 16);  // align to 16B
    output_scales_offsets.push_back(output_scales_bytes);
    output_scales_bytes += ceildiv(size * dtype_bits, 8);
  }

  // Allocate buffer for swizzled scales
  auto output_scales_pyt = allocateSpace(std::vector<size_t>{output_scales_bytes},
                                         transformer_engine::DType::kByte, false);
  uint8_t *output_scales_dptr = reinterpret_cast<uint8_t *>(getDataPtr(output_scales_pyt));

  // Construct TE tensors with only scales
  std::vector<transformer_engine::TensorWrapper> inputs_nvte, outputs_nvte;
  for (size_t i = 0; i < tensors_needing_swizzle.size(); ++i) {
    auto &tensor = *tensors_needing_swizzle[i];
    inputs_nvte.emplace_back(scaling_mode);
    outputs_nvte.emplace_back(scaling_mode);
    auto &input_nvte = inputs_nvte.back();
    auto &output_nvte = outputs_nvte.back();
    output_nvte.set_with_gemm_swizzled_scales(true);
    if (rowwise_usage) {
      const auto data_nvte = tensor.get_rowwise_data();
      const auto scales_nvte = tensor.get_rowwise_scale_inv();
      const auto data_dtype = static_cast<transformer_engine::DType>(data_nvte.dtype);
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      input_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
      input_nvte.set_rowwise_scale_inv(scales_nvte.data_ptr, scales_dtype, scales_nvte.shape);
      output_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
      output_nvte.set_rowwise_scale_inv(output_scales_dptr + output_scales_offsets[i], scales_dtype,
                                        scales_nvte.shape);
    } else {
      const auto data_nvte = tensor.get_columnwise_data();
      const auto scales_nvte = tensor.get_columnwise_scale_inv();
      const auto data_dtype = static_cast<transformer_engine::DType>(data_nvte.dtype);
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      input_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
      input_nvte.set_columnwise_scale_inv(scales_nvte.data_ptr, scales_dtype, scales_nvte.shape);
      output_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
      output_nvte.set_columnwise_scale_inv(output_scales_dptr + output_scales_offsets[i],
                                           scales_dtype, scales_nvte.shape);
    }
  }

  // Pack raw NVTETensors into vectors
  std::vector<NVTETensor> inputs_nvte_raw, outputs_nvte_raw;
  for (auto &tensor : inputs_nvte) {
    inputs_nvte_raw.emplace_back(tensor.data());
  }
  for (auto &tensor : outputs_nvte) {
    outputs_nvte_raw.emplace_back(tensor.data());
  }

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_tensor_swizzle_scaling_factors(inputs_nvte_raw.data(), outputs_nvte_raw.data(),
                                              inputs_nvte_raw.size(),
                                              at::cuda::getCurrentCUDAStream());
  });

  // Update tensors with swizzled scales
  for (size_t i = 0; i < tensors_needing_swizzle.size(); ++i) {
    auto &tensor = *tensors_needing_swizzle[i];
    reset_tensor_data(tensor, !rowwise_usage, !columnwise_usage);
    tensor.set_with_gemm_swizzled_scales(true);
    if (rowwise_usage) {
      auto scales_nvte = outputs_nvte[i].get_rowwise_scale_inv();
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      tensor.set_rowwise_scale_inv(output_scales_dptr + output_scales_offsets[i], scales_dtype,
                                   scales_nvte.shape);
    } else {
      auto scales_nvte = outputs_nvte[i].get_columnwise_scale_inv();
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      tensor.set_columnwise_scale_inv(output_scales_dptr + output_scales_offsets[i], scales_dtype,
                                      scales_nvte.shape);
    }
  }

  return std::move(output_scales_pyt);
}

at::Tensor convert_block_scaling_to_mxfp8_tensor(transformer_engine::TensorWrapper &input,
                                                 bool rowwise) {
  // Check input tensor
  const NVTEScalingMode scaling_mode = input.scaling_mode();
  NVTE_CHECK(scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D,
             "Input tensor must be a block scaling tensor");

  // Get tensor data
  NVTEBasicTensor data;
  size_t data_flat_first_dim = 1;
  size_t data_flat_last_dim = 1;
  if (rowwise) {
    data = input.get_rowwise_data();
    for (size_t i = 0; i < data.shape.ndim - 1; ++i) {
      data_flat_first_dim *= data.shape.data[i];
    }
    data_flat_last_dim = data.shape.data[data.shape.ndim - 1];
  } else {
    data = input.get_columnwise_data();
    data_flat_first_dim = data.shape.data[0];
    for (size_t i = 1; i < data.shape.ndim; ++i) {
      data_flat_last_dim *= data.shape.data[i];
    }
  }
  NVTEShape data_shape{};
  data_shape.data[0] = data_flat_first_dim;
  data_shape.data[1] = data_flat_last_dim;
  data_shape.ndim = 2;

  // Recreate input tensor with rowwise usage
  transformer_engine::TensorWrapper input_cu(scaling_mode);
  input_cu.set_rowwise_data(data.data_ptr, input.dtype(), data_shape);
  const NVTEBasicTensor scale_inv =
      rowwise ? input.get_rowwise_scale_inv() : input.get_columnwise_scale_inv();
  input_cu.set_rowwise_scale_inv(
      scale_inv.data_ptr, static_cast<transformer_engine::DType>(scale_inv.dtype), scale_inv.shape);

  // Create output tensor
  transformer_engine::TensorWrapper output_cu(NVTE_MXFP8_1D_SCALING);
  output_cu.set_rowwise_data(data.data_ptr, input.dtype(), data_shape);
  // Output swizzled mxfp8 scaling factor dimensions
  const size_t swizzled_scale_inv_first_dim = ceildiv(data_flat_first_dim, 128) * 128;
  const size_t swizzled_scale_inv_last_dim = ceildiv(data_flat_last_dim, 128) * 4;
  // Allocate memory for swizzled mxfp8 scaling factors
  at::Tensor swizzled_scale_inv =
      allocateSpace(std::vector<size_t>{swizzled_scale_inv_first_dim, swizzled_scale_inv_last_dim},
                    transformer_engine::DType::kByte, false);
  // Set rowwise scaling factors on output
  void *const swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);
  NVTEShape swizzled_scale_inv_shape{};
  swizzled_scale_inv_shape.data[0] = swizzled_scale_inv_first_dim;
  swizzled_scale_inv_shape.data[1] = swizzled_scale_inv_last_dim;
  swizzled_scale_inv_shape.ndim = 2;
  output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr, transformer_engine::DType::kFloat8E8M0,
                                  swizzled_scale_inv_shape);
  output_cu.set_with_gemm_swizzled_scales(true);

  // Convert scaling factors from FP8 block scaling GEMM_READY format to mxfp8 swizzled format
  NVTE_SCOPED_GIL_RELEASE({
    nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(input_cu.data(), output_cu.data(),
                                                        at::cuda::getCurrentCUDAStream());
  });

  // Set the input tensor to be the converted mxfp8 tensor and return the swizzled scaling factor
  // for it to be kept alive during the GEMM
  input = std::move(output_cu);
  return swizzled_scale_inv;
}

void inplace_swizzle_scale_for_gemm(py::handle &tensor) {
  // Convert Python tensor to C++ tensor
  auto tensor_nvte = makeTransformerEngineTensor(tensor, py::none());

  // Return early if scale swizzling is not required
  const auto scaling_mode = tensor_nvte.scaling_mode();
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
    case NVTE_NVFP4_1D_SCALING:
      // Tensor format requires scale swizzling
      break;
    case NVTE_INVALID_SCALING:
      NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
    default:
      // Tensor format does not require scale swizzling for GEMM
      return;
  }

  // Return early if scales are already swizzled
  if (tensor_nvte.get_with_gemm_swizzled_scales()) {
    return;
  }

  // Check what scaling factors the tensor contains
  auto is_empty = [](const NVTEBasicTensor &t) -> bool {
    return t.shape.ndim == 1 && t.shape.data[0] == 0;
  };
  const bool has_rowwise_scales = !is_empty(tensor_nvte.get_rowwise_scale_inv());
  const bool has_columnwise_scales = !is_empty(tensor_nvte.get_columnwise_scale_inv());

  // Swizzle scaling factors
  auto [rowwise_scales, columnwise_scales] =
      swizzle_scales_for_gemm(tensor_nvte, has_rowwise_scales, has_columnwise_scales);

  // Update Python tensor with swizzled scales
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
      if (has_rowwise_scales) {
        tensor.attr("_rowwise_scale_inv") = rowwise_scales;
      }
      if (has_columnwise_scales) {
        tensor.attr("_columnwise_scale_inv") = columnwise_scales;
      }
      tensor.attr("_with_gemm_swizzled_scales") = true;
      break;
    case NVTE_NVFP4_1D_SCALING:
      if (has_rowwise_scales) {
        tensor.attr("_rowwise_scale_inv") = rowwise_scales;
      }
      if (has_columnwise_scales) {
        tensor.attr("_columnwise_scale_inv") = columnwise_scales;
      }
      tensor.attr("_with_gemm_swizzled_scales") = true;
      break;
    default:
      NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
  }
}

}  // namespace pytorch
}  // namespace transformer_engine
