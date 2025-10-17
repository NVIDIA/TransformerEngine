/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util.h"

#include "common.h"
#include "extensions.h"
#include "pybind.h"
#include "common/common.h"

namespace transformer_engine::pytorch {

std::optional<at::Tensor> swizzle_scaling_factors(transformer_engine::TensorWrapper& input,
                                                  bool rowwise) {
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

std::vector<py::object> split_quantized_tensor(py::handle tensor, std::vector<size_t>& m_splits) {
  init_extension();

  NVTE_CHECK(detail::IsMXFP8Tensor(tensor.ptr()), "Input must be MXFP8Tensor.");

  bool rowwise_usage = !(tensor.attr("_rowwise_data").is_none());
  bool columnwise_usage = !(tensor.attr("_columnwise_data").is_none());
  NVTE_CHECK(rowwise_usage || columnwise_usage, "No data found for MXFP8 Tensor.");

  const transformer_engine::DType fp8_dtype =
      tensor.attr("_fp8_dtype").cast<transformer_engine::DType>();

  std::vector<at::Tensor> rowwise_data_list, rowwise_scale_inv_list;
  if (rowwise_usage) {
    auto rowwise_data = tensor.attr("_rowwise_data").cast<at::Tensor>();
    auto rowwise_scale_inv = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();

    split_quantized_tensor_impl(rowwise_data, rowwise_scale_inv, m_splits, rowwise_data_list,
                                rowwise_scale_inv_list, true);
  }

  std::vector<at::Tensor> columnwise_data_list, columnwise_scale_inv_list;
  if (columnwise_usage) {
    auto columnwise_data = tensor.attr("_columnwise_data").cast<at::Tensor>();
    auto columnwise_scale_inv = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();

    split_quantized_tensor_impl(columnwise_data, columnwise_scale_inv, m_splits,
                                columnwise_data_list, columnwise_scale_inv_list, false);
  }

  // Construct mxfp8 tensors
  auto quantizer = tensor.attr("_quantizer");
  std::vector<py::object> tensor_py_list;
  py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorBasePythonClass));
  for (size_t i = 0; i < m_splits.size(); ++i) {
    // Create tensor objects with proper reference counting
    py::object rowwise_data = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
    py::object rowwise_scale = rowwise_usage ? py::cast(rowwise_scale_inv_list[i]) : py::none();
    py::object columnwise_data =
        (columnwise_usage ? py::cast(columnwise_data_list[i]) : py::none());
    py::object columnwise_scale =
        (columnwise_usage ? py::cast(columnwise_scale_inv_list[i]) : py::none());
    // Construct Python tensor
    tensor_py_list.emplace_back(MXFP8TensorClass(rowwise_data, rowwise_scale, columnwise_data,
                                                 columnwise_scale, fp8_dtype, quantizer));
  }

  return tensor_py_list;
}

namespace {

constexpr size_t fp8_elem_size = 1;
constexpr size_t scale_elem_size = 1;

void split_quantized_tensor_impl(at::Tensor& data, at::Tensor& scale_inv,
                                 std::vector<size_t>& m_splits, std::vector<at::Tensor>& data_list,
                                 std::vector<at::Tensor>& scale_inv_list, bool rowwise) {
  auto hidden_dim = data.size(-1);
  data = data.reshape({-1, hidden_dim}).contiguous();
  NVTE_CHECK(std::accumulate(m_splits.begin(), m_splits.end(), 0) == data.size(0),
             "Rowwise data size does not match m_splits.");
  NVTE_CHECK(scale_inv.dim() == 2, "Rowwise scale_inv must be 2D.");
  scale_inv = scale_inv.contiguous();

  std::vector<std::vector<size_t>> data_shapes, scale_shapes;
  std::vector<size_t> no_pad_m_splits, padded_m_splits, data_offsets, scale_inv_offsets;
  size_t padded_total_m = 0, data_offset = 0, scale_inv_offset = 0;
  for (size_t i = 0; i < m_splits.size(); ++i) {
    if (rowwise) {
      // Rowwise scaling factors are required to be 128 multiple in the m dimension
      padded_m_splits.push_back(roundup(m_splits[i], 128));
      padded_total_m += padded_m_splits.back();
      no_pad_m_splits.push_back(m_splits[i]);
    } else {
      // Columnwise scaling factors are required to be 4 multiple in the m dimension
      padded_m_splits.push_back(roundup(m_splits[i] / 32, 4));
      padded_total_m += padded_m_splits.back();
      no_pad_m_splits.push_back(m_splits[i] / 32);
    }

    data_shapes.push_back({m_splits[i], static_cast<size_t>(hidden_dim)});
    data_offsets.push_back(data_offset);
    data_offset += m_splits[i] * hidden_dim * fp8_elem_size;

    scale_shapes.push_back({padded_m_splits[i], static_cast<size_t>(scale_inv.size(1))});
    scale_inv_offsets.push_back(scale_inv_offset);
    scale_inv_offset += padded_m_splits[i] * scale_inv.size(1) * scale_elem_size;
  }

  // Optionally pad scale_inv
  at::Tensor scale_inv_padded = scale_inv;
  if (padded_total_m != scale_inv.size(0)) {
    scale_inv_padded =
        at::empty({static_cast<int64_t>(padded_total_m), scale_inv.size(1)}, scale_inv.options());
    fused_multi_row_padding(scale_inv, scale_inv_padded, no_pad_m_splits, padded_m_splits);
  }

  // Split data and scale_inv
  auto data_buffer = std::make_shared<at::Tensor>(data);
  auto scale_inv_buffer = std::make_shared<at::Tensor>(scale_inv_padded);
  for (size_t i = 0; i < m_splits.size(); ++i) {
    data_list.emplace_back(
        make_torch_view(data_buffer, data_shapes[i], data_offsets[i], torch::kUInt8));
    scale_inv_list.emplace_back(
        make_torch_view(scale_inv_buffer, scale_shapes[i], scale_inv_offsets[i], torch::kUInt8));
  }
}

}  // namespace

at::Tensor convert_block_scaling_to_mxfp8_tensor(transformer_engine::TensorWrapper& input,
                                                 bool rowwise) {
  using transformer_engine::DIVUP;

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
    for (int i = 0; i < data.shape.ndim - 1; ++i) {
      data_flat_first_dim *= data.shape.data[i];
    }
    data_flat_last_dim = data.shape.data[data.shape.ndim - 1];
  } else {
    data = input.get_columnwise_data();
    data_flat_first_dim = data.shape.data[0];
    for (int i = 1; i < data.shape.ndim; ++i) {
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
  const auto options = at::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
  at::Tensor swizzled_scale_inv = at::empty(
      std::vector<int64_t>{swizzled_scale_inv_first_dim, swizzled_scale_inv_last_dim}, options);
  // Set rowwise scaling factors on output
  void* const swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);
  NVTEShape swizzled_scale_inv_shape{};
  swizzled_scale_inv_shape.data[0] = swizzled_scale_inv_first_dim;
  swizzled_scale_inv_shape.data[1] = swizzled_scale_inv_last_dim;
  swizzled_scale_inv_shape.ndim = 2;
  output_cu.set_rowwise_scale_inv(swizzled_scale_inv_dptr, transformer_engine::DType::kFloat8E8M0,
                                  swizzled_scale_inv_shape);

  // Convert scaling factors from FP8 block scaling GEMM_READY format to mxfp8 swizzled format
  nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(input_cu.data(), output_cu.data(),
                                                      at::cuda::getCurrentCUDAStream());

  // Set the input tensor to be the converted mxfp8 tensor and return the swizzled scaling factor
  // for it to be kept alive during the GEMM
  input = std::move(output_cu);
  return swizzled_scale_inv;
}

}  // namespace transformer_engine::pytorch
