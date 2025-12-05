/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util.h"

#include "common.h"
#include "common/common.h"

namespace {

/*! Buffer size in bytes. */
size_t buffer_size_bytes(size_t size, transformer_engine::DType dtype) {
  return size * transformer_engine::pytorch::typeToNumBits(dtype) / 8;
}

}  // namespace

std::optional<at::Tensor> swizzle_scaling_factors(transformer_engine::TensorWrapper& tensor,
                                                  bool rowwise) {
  using namespace transformer_engine::pytorch;
  const bool true_bool = true;

  // Return early if scale swizzling is not required
  const auto scaling_mode = tensor.scaling_mode();
  switch (scaling_mode) {
  case NVTE_MXFP8_1D_SCALING:
  case NVTE_NVFP4_1D_SCALING:
    // Tensor format requires scale swizzling
    break;
  case NVTE_BLOCK_SCALING_1D:
  case NVTE_BLOCK_SCALING_2D:
    NVTE_ERROR("FP8 block scaling assumes scales are manually swizzled externally.");
  case NVTE_INVALID_SCALING:
    NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
  default:
    // Tensor format does not require scale swizzling for GEMM
    return std::nullopt;
  }

  // Return early if scales are already swizzled
  bool is_already_swizzled = false;
  nvte_get_tensor_param_v2(tensor.data(), kNVTEWithGEMMSwizzledScales,
                           &is_already_swizzled, sizeof(is_already_swizzled),
                           nullptr);
  if (is_already_swizzled) {
    return std::nullopt;
  }

  // Buffer for unswizzled scales
  const auto input_scales_nvte = (rowwise
                                  ? tensor.get_rowwise_scale_inv()
                                  : tensor.get_columnwise_scale_inv());
  void *input_scales_dptr = input_scales_nvte.data_ptr;
  const NVTEShape input_scales_shape = input_scales_nvte.shape;
  const auto scales_dtype = static_cast<transformer_engine::DType>(input_scales_nvte.dtype);

  // Allocate buffer for swizzled scales
  const NVTEShape output_scales_shape = input_scales_shape;
  const size_t output_scales_size = product(output_scales_shape,
                                            0, output_scales_shape.ndim);
  const size_t output_scales_bytes = buffer_size_bytes(output_scales_size,
                                                       scales_dtype);
  auto output_scales_pyt = allocateSpace(std::vector<size_t>{output_scales_bytes},
                                         transformer_engine::DType::kByte,
                                         false);
  void *output_scales_dptr = getDataPtr(output_scales_pyt);

  // Construct TE tensors with only scales
  transformer_engine::TensorWrapper input_nvte(scaling_mode);
  transformer_engine::TensorWrapper output_nvte(scaling_mode);
  if (rowwise) {
    const auto data_nvte = tensor.get_rowwise_data();
    const auto data_dtype = static_cast<transformer_engine::DType>(data_nvte.dtype);
    input_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
    input_nvte.set_rowwise_scale_inv(input_scales_dptr, scales_dtype, input_scales_shape);
    output_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
    output_nvte.set_rowwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
  } else {
    const auto data_nvte = tensor.get_columnwise_data();
    const auto data_dtype = static_cast<transformer_engine::DType>(data_nvte.dtype);
    input_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
    input_nvte.set_columnwise_scale_inv(input_scales_dptr, scales_dtype, input_scales_shape);
    output_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
    output_nvte.set_columnwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
  }
  nvte_set_tensor_param_v2(output_nvte.data(), kNVTEWithGEMMSwizzledScales,
                           &true_bool, sizeof(true_bool));

  // Launch kernel
  nvte_swizzle_scaling_factors(input_nvte.data(), output_nvte.data(),
                               at::cuda::getCurrentCUDAStream());

  // Update tensor with swizzled scales
  if (rowwise) {
    tensor.set_rowwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
  } else {
    tensor.set_columnwise_scale_inv(output_scales_dptr, scales_dtype, output_scales_shape);
  }
  nvte_set_tensor_param_v2(tensor.data(), kNVTEWithGEMMSwizzledScales,
                           &true_bool, sizeof(true_bool));

  return output_scales_pyt;
}

std::optional<at::Tensor> multi_tensor_swizzle_scaling_factors(
    std::vector<transformer_engine::TensorWrapper>& tensors, bool rowwise) {
  using namespace transformer_engine::pytorch;
  const bool true_bool = true;

  // Return early if there are no tensors
  if (tensors.empty()) {
    return std::nullopt;
  }

  // Check that all tensors have same scaling mode
  const auto scaling_mode = tensors.front().scaling_mode();
  for (const auto &tensor: tensors) {
    NVTE_CHECK(tensor.scaling_mode() == scaling_mode, "Tensors have different scaling modes");
  }

  // Return early if scale swizzling is not required
  switch (scaling_mode) {
  case NVTE_MXFP8_1D_SCALING:
  case NVTE_NVFP4_1D_SCALING:
    // Tensor format requires scale swizzling
    break;
  case NVTE_BLOCK_SCALING_1D:
  case NVTE_BLOCK_SCALING_2D:
    NVTE_ERROR("FP8 block scaling assumes scales are manually swizzled externally.");
  case NVTE_INVALID_SCALING:
    NVTE_ERROR("Invalid scaling mode for swizzling scaling factors.");
  default:
    // Tensor format does not require scale swizzling for GEMM
    return std::nullopt;
  }

  // Filter out tensors that already have swizzled scales
  std::vector<transformer_engine::TensorWrapper *> tensors_needing_swizzle;
  for (auto &tensor: tensors) {
    bool is_already_swizzled = false;
    nvte_get_tensor_param_v2(tensor.data(), kNVTEWithGEMMSwizzledScales,
                             &is_already_swizzled, sizeof(is_already_swizzled),
                             nullptr);
    if (!is_already_swizzled) {
      tensors_needing_swizzle.push_back(&tensor);
    }
  }
  if (tensors_needing_swizzle.empty()) {
    return std::nullopt;
  }

  // Determine buffer size needed for swizzled scales
  std::vector<size_t> output_scales_offsets;
  size_t output_scales_bytes = 0;
  for (auto &tensor: tensors_needing_swizzle) {
    const auto scales_nvte = (rowwise
                              ? tensor->get_rowwise_scale_inv()
                              : tensor->get_columnwise_scale_inv());
    const auto &scales_shape = scales_nvte.shape;
    const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
    const auto scales_size = product(scales_shape, 0, scales_shape.ndim);
    output_scales_bytes = roundup(output_scales_bytes, 16);  // align to 16B
    output_scales_offsets.push_back(output_scales_bytes);
    output_scales_bytes += buffer_size_bytes(scales_size, scales_dtype);
  }

  // Allocate buffer for swizzled scales
  auto output_scales_pyt = allocateSpace(std::vector<size_t>{output_scales_bytes},
                                         transformer_engine::DType::kByte,
                                         false);
  uint8_t *output_scales_dptr = reinterpret_cast<uint8_t *>(getDataPtr(output_scales_pyt));

  // Construct TE tensors with only scales
  std::vector<transformer_engine::TensorWrapper> inputs_nvte, outputs_nvte;
  for (size_t i = 0; i < tensors_needing_swizzle.size(); ++i) {
    auto &tensor = *tensors_needing_swizzle[i];
    inputs_nvte.emplace_back(scaling_mode);
    outputs_nvte.emplace_back(scaling_mode);
    auto &input_nvte = inputs_nvte.back();
    auto &output_nvte = outputs_nvte.back();
    if (rowwise) {
      const auto data_nvte = tensor.get_rowwise_data();
      const auto scales_nvte = tensor.get_rowwise_scale_inv();
      const auto data_dtype = static_cast<transformer_engine::DType>(data_nvte.dtype);
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      input_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
      input_nvte.set_rowwise_scale_inv(scales_nvte.data_ptr, scales_dtype,
                                       scales_nvte.shape);
      output_nvte.set_rowwise_data(nullptr, data_dtype, data_nvte.shape);
      output_nvte.set_rowwise_scale_inv(output_scales_dptr + output_scales_offsets[i],
                                        scales_dtype, scales_nvte.shape);
    } else {
      const auto data_nvte = tensor.get_columnwise_data();
      const auto scales_nvte = tensor.get_columnwise_scale_inv();
      const auto data_dtype = static_cast<transformer_engine::DType>(data_nvte.dtype);
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      input_nvte.set_columnwise_data(nullptr, data_dtype, data_nvte.shape);
      input_nvte.set_columnwise_scale_inv(scales_nvte.data_ptr, scales_dtype,
                                          scales_nvte.shape);
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
    inputs_nvte_raw.emplace_back(tensor.data());
  }

  // Launch kernel
  nvte_multi_tensor_swizzle_scaling_factors(inputs_nvte_raw.data(), outputs_nvte_raw.data(),
                                            inputs_nvte_raw.size(),
                                            at::cuda::getCurrentCUDAStream());

  // Update tensors with swizzled scales
  for (size_t i = 0; i < tensors_needing_swizzle.size(); ++i) {
    auto &tensor = *tensors_needing_swizzle[i];
    if (rowwise) {
      auto scales_nvte = outputs_nvte[i].get_rowwise_scale_inv();
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      tensor.set_rowwise_scale_inv(output_scales_dptr + output_scales_offsets[i],
                                   scales_dtype, scales_nvte.shape);
    } else {
      auto scales_nvte = outputs_nvte[i].get_columnwise_scale_inv();
      const auto scales_dtype = static_cast<transformer_engine::DType>(scales_nvte.dtype);
      tensor.set_columnwise_scale_inv(output_scales_dptr + output_scales_offsets[i],
                                      scales_dtype, scales_nvte.shape);
    }
    nvte_set_tensor_param_v2(tensor.data(), kNVTEWithGEMMSwizzledScales,
                             &true_bool, sizeof(true_bool));
  }

  return output_scales_pyt;
}

at::Tensor convert_block_scaling_to_mxfp8_tensor(transformer_engine::TensorWrapper& input,
                                                 bool rowwise) {
  using namespace transformer_engine::pytorch;
  using transformer_engine::DIVUP;
  const bool true_bool = true;

  // Check input tensor
  const NVTEScalingMode scaling_mode = input.scaling_mode();
  NVTE_CHECK(scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D,
             "Input tensor must be a block scaling tensor");
  bool input_is_swizzled = false;
  nvte_get_tensor_param_v2(input.data(), kNVTEWithGEMMSwizzledScales,
                           &input_is_swizzled, sizeof(input_is_swizzled),
                           nullptr);
  NVTE_CHECK(input_is_swizzled, "Input tensor must have swizzled scales");

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
  const size_t swizzled_scale_inv_first_dim = DIVUP<size_t>(data_flat_first_dim, 128) * 128;
  const size_t swizzled_scale_inv_last_dim = DIVUP<size_t>(data_flat_last_dim, 128) * 4;
  // Allocate memory for swizzled mxfp8 scaling factors
  at::Tensor swizzled_scale_inv
    = allocateSpace(std::vector<size_t>{swizzled_scale_inv_first_dim, swizzled_scale_inv_last_dim},
                    transformer_engine::DType::kByte,
                    false);
  // Set rowwise scaling factors on output
  void* const swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);
  NVTEShape swizzled_scale_inv_shape{};
  swizzled_scale_inv_shape.data[0] = swizzled_scale_inv_first_dim;
  swizzled_scale_inv_shape.data[1] = swizzled_scale_inv_last_dim;
  swizzled_scale_inv_shape.ndim = 2;
  output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr, transformer_engine::DType::kFloat8E8M0,
                                  swizzled_scale_inv_shape);
  nvte_set_tensor_param_v2(output_cu.data(), kNVTEWithGEMMSwizzledScales,
                           &true_bool, sizeof(true_bool));

  // Convert scaling factors from FP8 block scaling GEMM_READY format to mxfp8 swizzled format
  nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(input_cu.data(), output_cu.data(),
                                                      at::cuda::getCurrentCUDAStream());

  // Set the input tensor to be the converted mxfp8 tensor and return the swizzled scaling factor
  // for it to be kept alive during the GEMM
  input = std::move(output_cu);
  return swizzled_scale_inv;
}
