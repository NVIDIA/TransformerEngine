/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/cast.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "../extensions.h"
#include "common.h"
#include "common/util/system.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace pytorch {

namespace {

std::vector<size_t> get_tensor_shape(const TensorWrapper &tensor) {
  const auto &shape = tensor.shape();
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

}  // namespace

py::object quantize(const at::Tensor &tensor, py::handle quantizer, const py::object &output,
                    std::optional<at::Tensor> noop_flag) {
  // Convert quantizer to C++ object
  auto quantizer_cpp = convert_quantizer(quantizer);

  // Convert input tensor to C++ object
  auto input_contiguous = tensor.contiguous();
  auto input_cpp = makeTransformerEngineTensor(input_contiguous);

  // Set amax if use_existing_amax = true (only valid for CS)
  bool use_existing_amax = false;
  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    use_existing_amax = quantizer.attr("use_existing_amax").cast<bool>();
    if (use_existing_amax) {
      const at::Tensor &amax = quantizer.attr("amax").cast<at::Tensor>();
      input_cpp.set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                         getTensorShape(amax));
    }
  }

  // Initialize output tensor
  TensorWrapper output_cpp;
  py::object output_py;
  if (output.is_none()) {
    const auto shape = get_tensor_shape(input_cpp);
    const auto fake_dtype = input_cpp.dtype();
    std::tie(output_cpp, output_py) = quantizer_cpp->create_tensor(shape, fake_dtype);
  } else {
    std::tie(output_cpp, output_py) = quantizer_cpp->convert_and_update_tensor(output);
  }

  // Initialize no-op flag
  std::optional<TensorWrapper> noop_flag_cpp;
  if (noop_flag.has_value()) {
    noop_flag_cpp = makeTransformerEngineTensor(*noop_flag);
  }

  // Perform quantization
  if (use_existing_amax) {
    auto *quantizer_cs = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
    quantizer_cs->quantize_with_amax(input_cpp, output_cpp, noop_flag_cpp);
  } else {
    quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);
  }

  return output_py;
}

py::object dequantize(const py::handle &input, transformer_engine::DType otype) {
  init_extension();

  const auto none = py::none();

  const auto &input_tensor = makeTransformerEngineTensor(input, none);

  NoneQuantizer q(none);

  const auto &shape = convertShape(input_tensor.shape());

  auto [out_tensor, out] = q.create_tensor(shape, otype);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_dequantize(input_tensor.data(), out_tensor.data(), at::cuda::getCurrentCUDAStream());
  });

  return out;
}

namespace {

void multi_tensor_quantize_impl(const std::vector<TensorWrapper> &input_list,
                                std::vector<py::handle> &quantizer_py_list,
                                std::vector<std::unique_ptr<Quantizer>> &quantizer_cpp_list,
                                std::vector<TensorWrapper> &output_list) {
  // Check number of tensors
  const size_t num_tensors = input_list.size();
  NVTE_CHECK(quantizer_py_list.size() == num_tensors, "Expected ", num_tensors,
             " Python quantizers, but got ", quantizer_py_list.size());
  NVTE_CHECK(quantizer_cpp_list.size() == num_tensors, "Expected ", num_tensors,
             " C++ quantizers, but got ", quantizer_cpp_list.size());
  NVTE_CHECK(output_list.size() == num_tensors, "Expected ", num_tensors,
             " output tensors, but got ", output_list.size());

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 delayed scaling
  bool with_fused_kernel = true;
  for (size_t i = 0; i < num_tensors; i++) {
    if (!detail::IsFloat8Quantizers(quantizer_py_list[i].ptr())) {
      with_fused_kernel = false;
      break;
    }
    if (nvte_tensor_data(output_list[i].data()) == nullptr ||
        nvte_tensor_columnwise_data(output_list[i].data()) == nullptr) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    // Fused kernel for multi-tensor quantize
    std::vector<NVTETensor> nvte_tensor_input_list;
    std::vector<NVTETensor> nvte_tensor_output_list;
    for (size_t i = 0; i < num_tensors; ++i) {
      nvte_tensor_input_list.push_back(input_list[i].data());
      nvte_tensor_output_list.push_back(output_list[i].data());
    }
    NVTE_SCOPED_GIL_RELEASE({
      nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                                nvte_tensor_output_list.data(), at::cuda::getCurrentCUDAStream());
    });
  } else {
    // Quantize kernels individually
    for (size_t i = 0; i < num_tensors; ++i) {
      quantizer_cpp_list[i]->quantize(input_list[i], output_list[i]);
    }
  }
}

}  // namespace

std::vector<py::object> multi_tensor_quantize(const std::vector<at::Tensor> &tensor_list,
                                              std::vector<py::handle> quantizer_list) {
  // Check number of tensors
  const size_t num_tensors = tensor_list.size();
  NVTE_CHECK(quantizer_list.size() == num_tensors, "Expected ", num_tensors,
             " quantizers, but got ", quantizer_list.size());

  // Convert quantizers to C++ objects
  std::vector<std::unique_ptr<Quantizer>> quantizer_cpp_list;
  for (size_t i = 0; i < num_tensors; i++) {
    quantizer_cpp_list.push_back(convert_quantizer(quantizer_list[i]));
  }

  // Initialize input and output tensors
  std::vector<TensorWrapper> input_cpp_list;
  std::vector<TensorWrapper> output_cpp_list;
  std::vector<py::object> output_py_list;
  for (size_t i = 0; i < num_tensors; ++i) {
    // Convert input tensor to C++ object
    const auto &input_py = tensor_list[i];
    NVTE_CHECK(input_py.is_contiguous(), "Input tensor ", i, " is not contiguous");
    input_cpp_list.emplace_back(makeTransformerEngineTensor(input_py));
    const auto &input_cpp = input_cpp_list.back();
    const auto input_shape = input_cpp.shape();
    const auto input_dtype = GetTransformerEngineDType(input_py.scalar_type());

    // Construct output tensor
    std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
    auto [output_cpp, output_py] = quantizer_cpp_list[i]->create_tensor(output_shape, input_dtype);
    output_cpp_list.emplace_back(std::move(output_cpp));
    output_py_list.emplace_back(std::move(output_py));
  }

  // Perform multi-tensor quantization
  multi_tensor_quantize_impl(input_cpp_list, quantizer_list, quantizer_cpp_list, output_cpp_list);

  return output_py_list;
}

namespace {

std::tuple<std::vector<py::object>, std::vector<TensorWrapper>> bulk_allocate_fp8_blockwise_tensors(
    std::vector<std::vector<size_t>> &shape_list, std::vector<py::handle> &quantizer_py_list,
    std::vector<Float8BlockQuantizer *> &quantizer_cpp_list) {
  init_extension();
  std::tuple<std::vector<py::object>, std::vector<TensorWrapper>> retval;
  auto &tensor_py_list = std::get<0>(retval);
  auto &tensor_cpp_list = std::get<1>(retval);

  // Number of tensors
  const size_t num_tensors = shape_list.size();
  if (num_tensors == 0) {
    return retval;
  }

  // Quantization parameters
  const auto rowwise_usage = quantizer_cpp_list[0]->rowwise_usage;
  const auto columnwise_usage = quantizer_cpp_list[0]->columnwise_usage;
  const auto scaling_mode = quantizer_cpp_list[0]->get_scaling_mode();
  const auto is_2D_scaled = scaling_mode == NVTE_BLOCK_SCALING_2D;
  const auto fp8_dtype = quantizer_cpp_list[0]->dtype;
  constexpr size_t fp8_elem_size = 1;
  constexpr size_t scale_elem_size = 4;

  // Helper function to construct tensor view
  // Note: Deleter holds a shared_ptr for the buffer, so the buffer
  // will survive until all views are deleted.
  auto make_torch_view = [](std::shared_ptr<at::Tensor> &buffer, const std::vector<size_t> &shape,
                            size_t offset, at::ScalarType dtype) -> at::Tensor {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    bool is_empty_shape = product(shape) == 0;
    if (buffer->data_ptr<uint8_t>() == nullptr || is_empty_shape) {
      return at::empty(shape_int64, at::device(at::kCUDA).dtype(dtype));
    }
    return at::from_blob(
        buffer->data_ptr<uint8_t>() + offset, shape_int64,
        [buffer](void *) {},  // deleter holds shared_ptr
        at::device(at::kCUDA).dtype(dtype));
  };

  // Allocate row-wise data
  std::vector<at::Tensor> rowwise_data_list, rowwise_scale_list;
  std::vector<std::vector<size_t>> rowwise_data_shapes, rowwise_scale_shapes;
  if (rowwise_usage) {
    // Tensor sizes
    for (size_t i = 0; i < num_tensors; ++i) {
      rowwise_data_shapes.emplace_back(shape_list[i]);
      rowwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], false));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets;
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 256);  // align to 256B
      data_offsets.push_back(buffer_size);
      buffer_size += product(rowwise_data_shapes[i]) * fp8_elem_size;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 16);  // align to 16B
      scale_offsets.push_back(buffer_size);
      buffer_size += product(rowwise_scale_shapes[i]) * scale_elem_size;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(
        at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i = 0; i < num_tensors; ++i) {
      rowwise_data_list.emplace_back(
          make_torch_view(buffer, rowwise_data_shapes[i], data_offsets[i], torch::kUInt8));
      rowwise_scale_list.emplace_back(
          make_torch_view(buffer, rowwise_scale_shapes[i], scale_offsets[i], torch::kFloat32));
    }
  }

  // Allocate column-wise data
  std::vector<at::Tensor> columnwise_data_list, columnwise_scale_list;
  std::vector<std::vector<size_t>> columnwise_data_shapes, columnwise_scale_shapes;
  if (columnwise_usage) {
    // Tensor sizes
    for (size_t i = 0; i < num_tensors; ++i) {
      columnwise_data_shapes.emplace_back();
      auto &shape = columnwise_data_shapes.back();
      shape.push_back(shape_list[i].back());
      for (size_t j = 0; j < shape_list[i].size() - 1; ++j) {
        shape.push_back(shape_list[i][j]);
      }
      columnwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], true));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets;
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 256);  // align to 256B
      data_offsets.push_back(buffer_size);
      buffer_size += product(columnwise_data_shapes[i]) * fp8_elem_size;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 16);  // align to 16B
      scale_offsets.push_back(buffer_size);
      buffer_size += product(columnwise_scale_shapes[i]) * scale_elem_size;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(
        at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i = 0; i < num_tensors; ++i) {
      columnwise_data_list.emplace_back(
          make_torch_view(buffer, columnwise_data_shapes[i], data_offsets[i], torch::kUInt8));
      columnwise_scale_list.emplace_back(
          make_torch_view(buffer, columnwise_scale_shapes[i], scale_offsets[i], torch::kFloat32));
    }
  }

  // Construct FP8 block-wise tensors
  py::handle Float8BlockwiseQTensorClass(
      reinterpret_cast<PyObject *>(Float8BlockwiseQTensorStoragePythonClass));
  for (size_t i = 0; i < num_tensors; ++i) {
    // Create tensor objects with proper reference counting
    py::object rowwise_data = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
    py::object rowwise_scale = rowwise_usage ? py::cast(rowwise_scale_list[i]) : py::none();
    py::object columnwise_data =
        (columnwise_usage ? py::cast(columnwise_data_list[i]) : py::none());
    py::object columnwise_scale =
        (columnwise_usage ? py::cast(columnwise_scale_list[i]) : py::none());

    // Construct Python tensor
    tensor_py_list.emplace_back(Float8BlockwiseQTensorClass(
        rowwise_data, rowwise_scale, columnwise_data, columnwise_scale, fp8_dtype,
        quantizer_py_list[i], is_2D_scaled, Float8BlockScaleTensorFormat::GEMM_READY));

    // Construct C++ tensor
    tensor_cpp_list.emplace_back(makeTransformerEngineTensor(
        rowwise_usage ? rowwise_data_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_data_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_data_shapes[i] : std::vector<size_t>{0},
        columnwise_usage ? columnwise_data_shapes[i] : std::vector<size_t>{0}, fp8_dtype, nullptr,
        nullptr, rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_scale_shapes[i] : std::vector<size_t>{0},
        columnwise_usage ? columnwise_scale_shapes[i] : std::vector<size_t>{0}, scaling_mode));
  }

  return retval;
}

std::tuple<std::vector<py::object>, std::vector<TensorWrapper>> bulk_allocate_mxfp8_tensors(
    std::vector<std::vector<size_t>> &shape_list, std::vector<py::handle> &quantizer_py_list,
    std::vector<MXFP8Quantizer *> &quantizer_cpp_list) {
  init_extension();
  std::tuple<std::vector<py::object>, std::vector<TensorWrapper>> retval;
  auto &tensor_py_list = std::get<0>(retval);
  auto &tensor_cpp_list = std::get<1>(retval);

  // Number of tensors
  const size_t num_tensors = shape_list.size();
  if (num_tensors == 0) {
    return retval;
  }

  // Quantization parameters
  const auto rowwise_usage = quantizer_cpp_list[0]->rowwise_usage;
  const auto columnwise_usage = quantizer_cpp_list[0]->columnwise_usage;
  const auto scaling_mode = quantizer_cpp_list[0]->get_scaling_mode();
  const auto fp8_dtype = quantizer_cpp_list[0]->dtype;
  constexpr size_t fp8_elem_size = 1;
  constexpr size_t scale_elem_size = 1;

  // Helper function to construct tensor view
  // Note: Deleter holds a shared_ptr for the buffer, so the buffer
  // will survive until all views are deleted.
  auto make_torch_view = [](std::shared_ptr<at::Tensor> &buffer, const std::vector<size_t> &shape,
                            size_t offset, at::ScalarType dtype) -> at::Tensor {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    bool is_empty_shape = product(shape) == 0;
    if (buffer->data_ptr<uint8_t>() == nullptr || is_empty_shape) {
      return at::empty(shape_int64, at::device(at::kCUDA).dtype(dtype));
    }
    return at::from_blob(
        buffer->data_ptr<uint8_t>() + offset, shape_int64,
        [buffer](void *) {},  // deleter holds shared_ptr
        at::device(at::kCUDA).dtype(dtype));
  };

  // Allocate row-wise data
  std::vector<at::Tensor> rowwise_data_list, rowwise_scale_list;
  std::vector<std::vector<size_t>> rowwise_data_shapes, rowwise_scale_shapes;
  if (rowwise_usage) {
    // Tensor sizes
    for (size_t i = 0; i < num_tensors; ++i) {
      rowwise_data_shapes.emplace_back(shape_list[i]);
      rowwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], false));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets;
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 256);  // align to 256B
      data_offsets.push_back(buffer_size);
      buffer_size += product(rowwise_data_shapes[i]) * fp8_elem_size;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 16);  // align to 16B
      scale_offsets.push_back(buffer_size);
      buffer_size += product(rowwise_scale_shapes[i]) * scale_elem_size;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(
        at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i = 0; i < num_tensors; ++i) {
      rowwise_data_list.emplace_back(
          make_torch_view(buffer, rowwise_data_shapes[i], data_offsets[i], torch::kUInt8));
      rowwise_scale_list.emplace_back(
          make_torch_view(buffer, rowwise_scale_shapes[i], scale_offsets[i], torch::kUInt8));
    }
  }

  // Allocate column-wise data
  std::vector<at::Tensor> columnwise_data_list, columnwise_scale_list;
  std::vector<std::vector<size_t>> columnwise_data_shapes, columnwise_scale_shapes;
  if (columnwise_usage) {
    // Tensor sizes
    for (size_t i = 0; i < num_tensors; ++i) {
      // For MXFP8, the columnwise data doesn't need transpose
      // because of TN, NT, NN layout support in SM100
      columnwise_data_shapes.emplace_back(shape_list[i]);
      columnwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], true));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets;
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 256);  // align to 256B
      data_offsets.push_back(buffer_size);
      buffer_size += product(columnwise_data_shapes[i]) * fp8_elem_size;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 16);  // align to 16B
      scale_offsets.push_back(buffer_size);
      buffer_size += product(columnwise_scale_shapes[i]) * scale_elem_size;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(
        at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i = 0; i < num_tensors; ++i) {
      columnwise_data_list.emplace_back(
          make_torch_view(buffer, columnwise_data_shapes[i], data_offsets[i], torch::kUInt8));
      columnwise_scale_list.emplace_back(
          make_torch_view(buffer, columnwise_scale_shapes[i], scale_offsets[i], torch::kUInt8));
    }
  }

  // Construct mxfp8 tensors
  py::handle MXFP8TensorClass(reinterpret_cast<PyObject *>(MXFP8TensorStoragePythonClass));
  for (size_t i = 0; i < num_tensors; ++i) {
    // Create tensor objects with proper reference counting
    py::object rowwise_data = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
    py::object rowwise_scale = rowwise_usage ? py::cast(rowwise_scale_list[i]) : py::none();
    py::object columnwise_data =
        (columnwise_usage ? py::cast(columnwise_data_list[i]) : py::none());
    py::object columnwise_scale =
        (columnwise_usage ? py::cast(columnwise_scale_list[i]) : py::none());

    // Construct Python tensor
    tensor_py_list.emplace_back(MXFP8TensorClass(rowwise_data, rowwise_scale, columnwise_data,
                                                 columnwise_scale, fp8_dtype,
                                                 quantizer_py_list[i]));

    // Construct C++ tensor
    tensor_cpp_list.emplace_back(makeTransformerEngineTensor(
        rowwise_usage ? rowwise_data_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_data_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_data_shapes[i] : std::vector<size_t>{0},
        columnwise_usage ? columnwise_data_shapes[i] : std::vector<size_t>{0}, fp8_dtype, nullptr,
        nullptr, rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_scale_shapes[i] : std::vector<size_t>{0},
        columnwise_usage ? columnwise_scale_shapes[i] : std::vector<size_t>{0}, scaling_mode));
  }

  return retval;
}

// allocate fp4 data, fp8 scalings, and amax values
// layout: [fp4_data0, ..., fp4_dataN, fp8_scaling0, ..., fp8_scalingN, amax0, ..., amaxN]
// amax buffer will be zeroed out by later amax kernels, so we can use empty to allocate
std::tuple<std::vector<py::object>, std::vector<TensorWrapper>, bool> bulk_allocate_nvfp4_tensors(
    std::vector<std::vector<size_t>> &shape_list, std::vector<py::handle> &quantizer_py_list,
    std::vector<NVFP4Quantizer *> &quantizer_cpp_list) {
  init_extension();
  std::tuple<std::vector<py::object>, std::vector<TensorWrapper>, bool> retval;
  auto &tensor_py_list = std::get<0>(retval);
  auto &tensor_cpp_list = std::get<1>(retval);
  auto &contiguous_data_and_scale = std::get<2>(retval);
  contiguous_data_and_scale = true;

  // Number of tensors
  const size_t num_tensors = shape_list.size();
  if (num_tensors == 0) {
    return retval;
  }

  // Quantization parameters
  const auto rowwise_usage = quantizer_cpp_list[0]->rowwise_usage;
  const auto columnwise_usage = quantizer_cpp_list[0]->columnwise_usage;
  const auto scaling_mode = quantizer_cpp_list[0]->get_scaling_mode();
  const auto fp4_dtype = quantizer_cpp_list[0]->dtype;
  constexpr size_t scale_elem_size = 1;

  // Helper function to construct tensor view
  // Note: Deleter holds a shared_ptr for the buffer, so the buffer
  // will survive until all views are deleted.
  auto make_torch_view = [](std::shared_ptr<at::Tensor> &buffer, const std::vector<size_t> &shape,
                            size_t offset, at::ScalarType dtype) -> at::Tensor {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    bool is_empty_shape = product(shape) == 0;
    if (buffer->data_ptr<uint8_t>() == nullptr || is_empty_shape) {
      return at::empty(shape_int64, at::device(at::kCUDA).dtype(dtype));
    }
    return at::from_blob(
        buffer->data_ptr<uint8_t>() + offset, shape_int64,
        [buffer](void *) {},  // deleter holds shared_ptr
        at::device(at::kCUDA).dtype(dtype));
  };

  // Lambda function for converting std::vector<size_t> shape to NVFP4 shape (last dim divided by 2)
  auto to_fp4_shape = [](const std::vector<size_t> &shape) {
    std::vector<size_t> fp4_shape(shape.begin(), shape.end());
    if (!fp4_shape.empty()) {
      fp4_shape.back() /= 2;
    }
    return fp4_shape;
  };

  // Allocate row-wise data
  std::vector<at::Tensor> rowwise_data_list, rowwise_scale_list, amax_rowwise_list;
  std::vector<std::vector<size_t>> rowwise_data_shapes, rowwise_scale_shapes;
  if (rowwise_usage) {
    // Tensor sizes
    for (size_t i = 0; i < num_tensors; ++i) {
      rowwise_data_shapes.emplace_back(shape_list[i]);
      rowwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], false));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets, amax_offsets;
    for (size_t i = 0; i < num_tensors; ++i) {
      // FP4 data is aligned to 256B
      const auto offset = roundup(buffer_size, 256);
      if (offset != buffer_size) {
        contiguous_data_and_scale = false;
      }
      data_offsets.push_back(offset);
      buffer_size = offset + (product(rowwise_data_shapes[i]) + 1) / 2;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      // Scales are aligned to 16B
      const auto offset = roundup(buffer_size, 16);
      if (offset != buffer_size) {
        contiguous_data_and_scale = false;
      }
      scale_offsets.push_back(offset);
      buffer_size = offset + product(rowwise_scale_shapes[i]) * scale_elem_size;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      // Amaxes (FP32) are aligned to 16B
      // Note: Multi-quantize kernel does not require contiguous amaxes.
      const auto offset = roundup(buffer_size, 16);
      amax_offsets.push_back(offset);
      buffer_size = offset + 4;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(
        at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i = 0; i < num_tensors; ++i) {
      rowwise_data_list.emplace_back(make_torch_view(buffer, to_fp4_shape(rowwise_data_shapes[i]),
                                                     data_offsets[i], torch::kUInt8));
      rowwise_scale_list.emplace_back(
          make_torch_view(buffer, rowwise_scale_shapes[i], scale_offsets[i], torch::kUInt8));
      amax_rowwise_list.emplace_back(
          make_torch_view(buffer, std::vector<size_t>{1}, amax_offsets[i], torch::kFloat32));
    }
  }

  // Allocate column-wise data
  std::vector<at::Tensor> columnwise_data_list, columnwise_scale_list, amax_columnwise_list;
  std::vector<std::vector<size_t>> columnwise_data_shapes, columnwise_scale_shapes;
  if (columnwise_usage) {
    // Tensor sizes
    for (size_t i = 0; i < num_tensors; ++i) {
      // push the transposed shape into NVFP4 columnwise shape
      // NVFP4 on SM100 is TN only
      columnwise_data_shapes.emplace_back();
      auto &shape = columnwise_data_shapes.back();
      shape.push_back(shape_list[i].back());
      for (size_t j = 0; j < shape_list[i].size() - 1; ++j) {
        shape.push_back(shape_list[i][j]);
      }
      columnwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], true));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets, amax_offsets;
    for (size_t i = 0; i < num_tensors; ++i) {
      // FP4 data is aligned to 256B
      const auto offset = roundup(buffer_size, 256);
      if (offset != buffer_size) {
        contiguous_data_and_scale = false;
      }
      data_offsets.push_back(offset);
      buffer_size = offset + (product(columnwise_data_shapes[i]) + 1) / 2;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      // Scales are aligned to 16B
      const auto offset = roundup(buffer_size, 16);
      if (offset != buffer_size) {
        contiguous_data_and_scale = false;
      }
      scale_offsets.push_back(offset);
      buffer_size = offset + product(columnwise_scale_shapes[i]) * scale_elem_size;
    }
    for (size_t i = 0; i < num_tensors; ++i) {
      // Amaxes (FP32) are aligned to 16B
      // Note: Multi-quantize kernel does not require contiguous amaxes.
      const auto offset = roundup(buffer_size, 16);
      amax_offsets.push_back(offset);
      buffer_size = offset + 4;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(
        at::empty({(int64_t)buffer_size}, at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i = 0; i < num_tensors; ++i) {
      columnwise_data_list.emplace_back(make_torch_view(
          buffer, to_fp4_shape(columnwise_data_shapes[i]), data_offsets[i], torch::kUInt8));
      columnwise_scale_list.emplace_back(
          make_torch_view(buffer, columnwise_scale_shapes[i], scale_offsets[i], torch::kUInt8));
      amax_columnwise_list.emplace_back(
          make_torch_view(buffer, std::vector<size_t>{1}, amax_offsets[i], torch::kFloat32));
    }
  }

  // Construct nvfp4 tensors
  py::handle NVFP4TensorClass(reinterpret_cast<PyObject *>(NVFP4TensorStoragePythonClass));
  for (size_t i = 0; i < num_tensors; ++i) {
    // Create tensor objects with proper reference counting
    py::object rowwise_data = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
    py::object rowwise_scale = rowwise_usage ? py::cast(rowwise_scale_list[i]) : py::none();
    py::object columnwise_data =
        (columnwise_usage ? py::cast(columnwise_data_list[i]) : py::none());
    py::object columnwise_scale =
        (columnwise_usage ? py::cast(columnwise_scale_list[i]) : py::none());
    py::object amax_rowwise = rowwise_usage ? py::cast(amax_rowwise_list[i]) : py::none();
    py::object amax_columnwise = columnwise_usage ? py::cast(amax_columnwise_list[i]) : py::none();

    // Construct Python tensor
    tensor_py_list.emplace_back(NVFP4TensorClass(rowwise_data, rowwise_scale, columnwise_data,
                                                 columnwise_scale, amax_rowwise, amax_columnwise,
                                                 fp4_dtype, quantizer_py_list[i]));

    // Construct C++ tensor
    // Use a TensorWrapper variable to hold the output of makeTransformerEngineTensor,
    // then set the amax and amax_columnwise values.
    {
      auto tensor_wrapper = makeTransformerEngineTensor(
          rowwise_usage ? rowwise_data_list[i].data_ptr() : nullptr,
          columnwise_usage ? columnwise_data_list[i].data_ptr() : nullptr,
          rowwise_usage ? rowwise_data_shapes[i] : std::vector<size_t>{0},
          columnwise_usage ? columnwise_data_shapes[i] : std::vector<size_t>{0}, fp4_dtype,
          /*amax_ptr=*/nullptr,
          /*scale_ptr=*/nullptr, rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
          columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
          rowwise_usage ? rowwise_scale_shapes[i] : std::vector<size_t>{0},
          columnwise_usage ? columnwise_scale_shapes[i] : std::vector<size_t>{0}, scaling_mode);

      // Set the amax rowwise and amax columnwise if available
      if (rowwise_usage) {
        tensor_wrapper.set_amax(amax_rowwise_list[i].data_ptr(), DType::kFloat32,
                                std::vector<size_t>{1});
      }
      if (columnwise_usage) {
        tensor_wrapper.set_columnwise_amax(amax_columnwise_list[i].data_ptr(), DType::kFloat32,
                                           std::vector<size_t>{1});
      }
      tensor_cpp_list.emplace_back(std::move(tensor_wrapper));
    }
  }

  return retval;
}

// Owns all allocations/wrappers backing quant_config_list[*].set_rng_state(...).
struct StochasticRngStateResources {
  at::Tensor rng_states_tensor;          // [2 * num_tensors], int64, CUDA
  at::Tensor rng_states_tensor_colwise;  // optional, same shape/dtype/device
  std::vector<TensorWrapper> te_rng_state_list;
  std::vector<TensorWrapper> te_rng_state_list_colwise;

  bool enabled{false};
  bool need_separate_rng_states{false};
  bool with_bulk_generate_rng_states{false};
};

// Populates quant_config_list (+ optional colwise list) with rng_state pointers and stochastic flag.
static StochasticRngStateResources setup_stochastic_rounding_rng_states_helper(
    size_t num_tensors, bool stochastic_rounding, bool with_bulk_generate_rng_states,
    bool need_separate_rng_states,
    std::vector<QuantizationConfigWrapper> &quant_config_list_rowwise,
    std::vector<QuantizationConfigWrapper> &quant_config_list_colwise) {
  // the return object will be used to keep rng states alive
  StochasticRngStateResources res;
  res.enabled = stochastic_rounding;
  res.need_separate_rng_states = need_separate_rng_states;
  res.with_bulk_generate_rng_states = with_bulk_generate_rng_states;

  if (!stochastic_rounding) return res;

  // Basic sanity: caller usually pre-sizes these to num_tensors.
  TORCH_CHECK(quant_config_list_rowwise.size() == num_tensors,
              "quant_config_list_rowwise must be sized to num_tensors");
  if (need_separate_rng_states) {
    TORCH_CHECK(quant_config_list_colwise.size() == num_tensors,
                "quant_config_list_colwise must be sized to num_tensors when "
                "need_separate_rng_states=true");
  }

  const size_t rng_elts_per_thread =
      res.with_bulk_generate_rng_states ? (1024 * num_tensors) : 1024;

  auto opts = at::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  res.rng_states_tensor = torch::empty({static_cast<int64_t>(2 * num_tensors)}, opts);
  if (need_separate_rng_states) {
    res.rng_states_tensor_colwise = torch::empty({static_cast<int64_t>(2 * num_tensors)}, opts);
  }

  res.te_rng_state_list.reserve(num_tensors);
  if (need_separate_rng_states) res.te_rng_state_list_colwise.reserve(num_tensors);

  for (size_t i = 0; i < num_tensors; ++i) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    // Rowwise RNG state
    at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
    int64_t *rng_state_ptr = static_cast<int64_t *>(res.rng_states_tensor.data_ptr()) + i * 2;
    philox_unpack(philox_args, rng_state_ptr);

    res.te_rng_state_list.push_back(makeTransformerEngineTensor(
        static_cast<void *>(rng_state_ptr), std::vector<size_t>{2}, DType::kInt64));
    quant_config_list_rowwise[i].set_rng_state(res.te_rng_state_list[i].data());
    quant_config_list_rowwise[i].set_stochastic_rounding(true);

    // Colwise RNG state (only if you truly need a different sequence)
    if (need_separate_rng_states) {
      // re-initialize philox_args for colwise RNG state
      at::PhiloxCudaState philox_args_col = init_philox_state(gen, rng_elts_per_thread);
      int64_t *rng_state_ptr_colwise =
          static_cast<int64_t *>(res.rng_states_tensor_colwise.data_ptr()) + i * 2;

      philox_unpack(philox_args_col, rng_state_ptr_colwise);

      res.te_rng_state_list_colwise.push_back(makeTransformerEngineTensor(
          static_cast<void *>(rng_state_ptr_colwise), std::vector<size_t>{2}, DType::kInt64));
      quant_config_list_colwise[i].set_rng_state(res.te_rng_state_list_colwise[i].data());
      quant_config_list_colwise[i].set_stochastic_rounding(true);
    }

    // break the loop if we are using bulk generate rng states
    if (res.with_bulk_generate_rng_states) break;
  }

  return res;
}

// Implements split-quantize NVFP4 with Row/Column-wise Hadamard Transform (RHT)
void split_quantize_nvfp4_impl_with_rht_helper(const TensorWrapper &input,
                                               const std::vector<TensorWrapper> &input_list,
                                               std::vector<TensorWrapper> &output_list,
                                               const std::vector<size_t> &split_sections,
                                               const std::vector<NVFP4Quantizer *> &quantizers,
                                               cudaStream_t stream) {
  const size_t num_tensors = split_sections.size();
  const auto &quantizer = *quantizers.front();

  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  for (size_t i = 0; i < num_tensors; ++i) {
    nvte_tensor_input_list.push_back(input_list[i].data());
    nvte_tensor_output_list.push_back(output_list[i].data());
  }

  // trigger the row-col fusion when the split-sections shapes are all 128 aligned for max performance
  bool all_aligned_token_dim =
      std::all_of(split_sections.begin(), split_sections.end(),
                  [](size_t split_section) { return split_section % 128 == 0; });

  // in the case when rowwise and colwise cannot be fused, we have to generate the RNG states twice
  // so that rowwise and colwise will have different random numbers
  bool need_separate_rng_states =
      (!all_aligned_token_dim) && quantizer.rowwise_usage && quantizer.columnwise_usage;

  // Objects for TE C API
  std::vector<QuantizationConfigWrapper> quant_config_list;
  std::vector<QuantizationConfigWrapper> quant_config_list_colwise;
  for (size_t i = 0; i < num_tensors; ++i) {
    quant_config_list.emplace_back(QuantizationConfigWrapper());
    quant_config_list_colwise.emplace_back(QuantizationConfigWrapper());
  }

  // this is true because we have already built grouped kernels for rowwise and colwise quantization with RHT
  bool with_bulk_generate_rng_states = true;

  // Stochastic rounding
  bool need_stochastic_rounding = quantizer.stochastic_rounding;
  auto stochastic_rng_state_resources = setup_stochastic_rounding_rng_states_helper(
      num_tensors, need_stochastic_rounding, with_bulk_generate_rng_states,
      need_separate_rng_states, quant_config_list, quant_config_list_colwise);

  // Enable NVFP4 kernels to use math operations that sacrifice
  // accuracy for performance. These optimizations are experimental
  // and inconsistently implemented.
  const auto use_fast_math = transformer_engine::getenv<bool>("NVTE_USE_FAST_MATH");
  if (use_fast_math) {
    for (auto &config : quant_config_list) {
      config.set_use_fast_math(true);
    }
    for (auto &config : quant_config_list_colwise) {
      config.set_use_fast_math(true);
    }
  }

  auto &quant_config_list_colwise_to_use =
      need_separate_rng_states ? quant_config_list_colwise : quant_config_list;

  // Compute amaxes
  if (quantizer.with_post_rht_amax) {
    // We need:
    // 1. Rowwise amax = amax for input
    // 2. Columnwise amax = amax for RHT(input.t)
    nvte_group_hadamard_transform_amax(
        input.data(), reinterpret_cast<NVTETensor *>(nvte_tensor_output_list.data()),
        split_sections.data(), num_tensors, 0, quantizer.rht_matrix_random_sign_mask_t, stream);
  } else {
    // RHT is enabled, but amax is pre-RHT amax
    NVTE_ERROR("NVFP4 split-quantize does not yet support pre-RHT amax");
  }

  // Check that RHT matrix is available
  NVTE_CHECK(quantizer.rht_matrix.defined() && quantizer.rht_matrix.numel() > 0,
             "RHT matrix is not available.");
  auto rht_matrix_nvte = makeTransformerEngineTensor(quantizer.rht_matrix);

  if (all_aligned_token_dim) {
    // allocate a tile scheduler workspace
    auto tile_scheduler_workspace_torch =
        at::empty({1}, at::device(at::kCUDA).dtype(torch::kInt32));
    auto nvte_tile_scheduler_workspace =
        makeTransformerEngineTensor(tile_scheduler_workspace_torch);
    // call the fully-fused grouped kernel for rowwise quantization & colwise RHT quantization transpose
    nvte_group_hadamard_transform_cast_fusion(
        input.data(), reinterpret_cast<NVTETensor *>(nvte_tensor_output_list.data()),
        rht_matrix_nvte.data(), split_sections.data(), num_tensors, quant_config_list[0],
        nvte_tile_scheduler_workspace.data(), stream);
  } else {
    // Separate quantization for rowwise usage and columnwise usage
    // Rowwise quantization fusion with grouped version
    if (quantizer.rowwise_usage) {
      std::vector<TensorWrapper> out_identity_list;
      std::vector<NVTETensor> nvte_tensor_out_identity_list;
      for (size_t i = 0; i < num_tensors; i++) {
        bool is_empty_split = input_list[i].numel() == 0;
        TensorWrapper out_identity(output_list[i].scaling_mode());
        auto out_identity_data = output_list[i].get_rowwise_data();
        auto out_identity_scale_inv = output_list[i].get_rowwise_scale_inv();
        auto out_identity_amax = output_list[i].get_amax();
        if (!is_empty_split) {
          out_identity.set_rowwise_data(out_identity_data.data_ptr,
                                        static_cast<DType>(out_identity_data.dtype),
                                        out_identity_data.shape);
          out_identity.set_rowwise_scale_inv(out_identity_scale_inv.data_ptr,
                                             static_cast<DType>(out_identity_scale_inv.dtype),
                                             out_identity_scale_inv.shape);
          out_identity.set_amax(out_identity_amax.data_ptr,
                                static_cast<DType>(out_identity_amax.dtype),
                                out_identity_amax.shape);
        }
        out_identity_list.emplace_back(std::move(out_identity));
        nvte_tensor_out_identity_list.push_back(out_identity_list.back().data());
      }
      nvte_group_nvfp4_quantize_with_amax(input.data(), nvte_tensor_out_identity_list.data(),
                                          split_sections.data(), num_tensors, quant_config_list[0],
                                          stream);
    }

    // Columnwise RHT quantization fusion with grouped version
    if (quantizer.columnwise_usage) {
      std::vector<TensorWrapper> out_transpose_list;
      std::vector<NVTETensor> nvte_tensor_out_transpose_list;
      for (size_t i = 0; i < num_tensors; i++) {
        bool is_empty_split = input_list[i].numel() == 0;
        auto out_columnwise_data = output_list[i].get_columnwise_data();
        auto out_columnwise_scale_inv = output_list[i].get_columnwise_scale_inv();
        auto out_columnwise_amax = output_list[i].get_columnwise_amax();

        // Create a wrapper for the columnwise output, as the rowwise output. Input is in transposed layout.
        TensorWrapper out_transpose(output_list[i].scaling_mode());
        if (!is_empty_split) {
          auto colwise_data_shape = out_columnwise_data.shape;
          std::vector<size_t> colwise_data_shape_2d;
          colwise_data_shape_2d.push_back(colwise_data_shape.data[0]);
          size_t last_dim = 1;
          for (size_t j = 1; j < colwise_data_shape.ndim; ++j) {
            last_dim *= colwise_data_shape.data[j];
          }
          colwise_data_shape_2d.push_back(last_dim);

          out_transpose.set_rowwise_data(out_columnwise_data.data_ptr,
                                         static_cast<DType>(out_columnwise_data.dtype),
                                         colwise_data_shape_2d);
          out_transpose.set_rowwise_scale_inv(out_columnwise_scale_inv.data_ptr,
                                              static_cast<DType>(out_columnwise_scale_inv.dtype),
                                              out_columnwise_scale_inv.shape);
          out_transpose.set_amax(out_columnwise_amax.data_ptr,
                                 static_cast<DType>(out_columnwise_amax.dtype),
                                 out_columnwise_amax.shape);
        }
        out_transpose_list.emplace_back(std::move(out_transpose));
        nvte_tensor_out_transpose_list.push_back(out_transpose_list.back().data());
      }
      nvte_group_hadamard_transform_cast_fusion_columnwise(
          input.data(), reinterpret_cast<NVTETensor *>(nvte_tensor_out_transpose_list.data()),
          rht_matrix_nvte.data(), split_sections.data(), num_tensors,
          quant_config_list_colwise_to_use[0], stream);
    }
  }
}

void split_quantize_nvfp4_impl_helper(const TensorWrapper &input,
                                      const std::vector<TensorWrapper> &input_list,
                                      std::vector<TensorWrapper> &output_list,
                                      const std::vector<size_t> &split_sections,
                                      const std::vector<NVFP4Quantizer *> &quantizers,
                                      cudaStream_t stream) {
  const size_t num_tensors = input_list.size();
  const auto &quantizer = *quantizers.front();

  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  for (size_t i = 0; i < num_tensors; ++i) {
    nvte_tensor_input_list.push_back(input_list[i].data());
    nvte_tensor_output_list.push_back(output_list[i].data());
  }

  // In this case without RHT, the rowwise and colwise quantization are fused
  // we don't need separate rng states for rowwise and colwise
  bool need_separate_rng_states = false;

  // Objects for TE C API
  std::vector<QuantizationConfigWrapper> quant_config_list;
  for (size_t i = 0; i < num_tensors; ++i) {
    quant_config_list.emplace_back(QuantizationConfigWrapper());
  }

  // TODO: this is only true because the non-RHT path doesn't have grouped kernels yet, which we can be optimized
  // so that we can generate all rng states at once
  bool with_bulk_generate_rng_states = false;

  bool need_stochastic_rounding = quantizer.stochastic_rounding;

  // place holder for colwise rng states, which are not needed in this case
  std::vector<QuantizationConfigWrapper> dummy_quant_config_list_colwise;

  auto stochastic_rng_state_resources = setup_stochastic_rounding_rng_states_helper(
      num_tensors, need_stochastic_rounding, with_bulk_generate_rng_states,
      need_separate_rng_states, quant_config_list,
      dummy_quant_config_list_colwise);  // colwise rng states are not needed in this case

  // We need:
  // 1. Rowwise amax = amax for input
  // 2. Columnwise amax = amax for input too
  // Columnwise amax will be filled with a fused D2D copy from rowwise amax
  // Note that the multi compute amax API expects rowwise amax pointer to be not null
  // So we need to set the pointer accordingly to make colwise-only quantization work
  std::vector<void *> orig_amax_ptr_list;
  for (size_t i = 0; i < num_tensors; i++) {
    auto rowwise_amax_ptr = output_list[i].get_amax().data_ptr;
    orig_amax_ptr_list.push_back(rowwise_amax_ptr);
    auto columnwise_amax_ptr = output_list[i].get_columnwise_amax().data_ptr;
    void *amax_ptr = rowwise_amax_ptr != nullptr ? rowwise_amax_ptr : columnwise_amax_ptr;
    NVTE_CHECK(amax_ptr != nullptr, "Could not find amax pointer");
    output_list[i].set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  }
  nvte_group_amax(input.data(), reinterpret_cast<NVTETensor *>(nvte_tensor_output_list.data()),
                  split_sections.data(), num_tensors, stream);
  for (size_t i = 0; i < num_tensors; i++) {
    output_list[i].set_amax(orig_amax_ptr_list[i], DType::kFloat32, std::vector<size_t>{1});
  }

  // Quantize tensors individually
  for (size_t i = 0; i < num_tensors; i++) {
    // skip this round if input is empty
    if (input_list[i].numel() == 0) {
      continue;
    }
    nvte_quantize_v2(input_list[i].data(), output_list[i].data(), quant_config_list[i], stream);
  }
}

void split_quantize_nvfp4_impl(const TensorWrapper &input,
                               const std::vector<TensorWrapper> &input_list,
                               std::vector<TensorWrapper> &output_list,
                               const std::vector<size_t> &split_sections,
                               const std::vector<NVFP4Quantizer *> &quantizers) {
  // Check tensor lists
  const size_t num_tensors = split_sections.size();
  NVTE_CHECK(input_list.size() == num_tensors, "Expected ", num_tensors, " input tensors, but got ",
             input_list.size(), ".");
  NVTE_CHECK(output_list.size() == num_tensors, "Expected ", num_tensors,
             " output tensors, but got ", output_list.size(), ".");
  NVTE_CHECK(quantizers.size() == num_tensors, "Expected ", num_tensors,
             " NVFP4 quantizers, but got ", quantizers.size(), ".");

  // sanity check all the quantizers have the same scaling mode
  bool all_same_scaling_mode =
      std::all_of(quantizers.begin(), quantizers.end(), [&](const NVFP4Quantizer *quantizer) {
        return quantizer->get_scaling_mode() == quantizers.front()->get_scaling_mode();
      });
  NVTE_CHECK(all_same_scaling_mode, "All quantizers must have the same scaling mode");

  // Trivial cases
  if (num_tensors == 0) {
    return;
  }
  if (input.numel() == 0) {
    for (const auto &tensor : input_list) {
      NVTE_CHECK(tensor.numel() == 0,
                 "Input tensor has zero elements but got split with non-zero elements");
    }
    return;
  }

  // Assume all quantizers have identical config
  const auto &quantizer = *quantizers.front();
  NVTE_CHECK(!quantizer.with_2d_quantization,
             "NVFP4 split-quantize does not support 2D quantization");
  NVTE_CHECK(!quantizer.with_amax_reduction,
             "NVFP4 split-quantize does not support amax reduction");

  // Check input tensor shape
  const size_t input_last_dim = input.ndim() > 0 ? input.size(input.ndim() - 1) : 1;
  NVTE_CHECK(input_last_dim % 128 == 0,
             "NVFP4 multi-quantize requires inner dim to be multiple of 128.");

  // CUDA stream
  auto stream = at::cuda::getCurrentCUDAStream();

  // Perform multi-tensor quantization
  NVTE_SCOPED_GIL_RELEASE({
    if (quantizer.with_rht) {  // Quantize row-wise data, RHT+quantize column-wise data
      // Check that config is supported
      NVTE_CHECK(input.dtype() == DType::kBFloat16, "RHT is only supported for bfloat16 input");
      // Fuse the rowwise and colwise into one when the kernel is ready
      split_quantize_nvfp4_impl_with_rht_helper(input, input_list, output_list, split_sections,
                                                quantizers, stream);
    } else {  // NVFP4 quantize
      // Fuse the rowwise and colwise into one when the kernel is ready
      split_quantize_nvfp4_impl_helper(input, input_list, output_list, split_sections, quantizers,
                                       stream);
    }
  });
}

}  // namespace

std::vector<py::object> split_quantize(const at::Tensor &tensor,
                                       const std::vector<size_t> &split_sections,
                                       std::vector<py::handle> quantizer_list,
                                       bool disable_bulk_allocation) {
  init_extension();

  // Check number of tensors
  const size_t num_splits = split_sections.size();
  NVTE_CHECK(quantizer_list.size() == num_splits, "Expected ", num_splits, " quantizers, but got ",
             quantizer_list.size());
  if (num_splits == 0) {
    return {};
  }

  // Input tensor properties
  auto input_py = tensor.contiguous();
  uint8_t *input_dptr = reinterpret_cast<uint8_t *>(input_py.data_ptr());
  auto input_dtype = GetTransformerEngineDType(input_py.scalar_type());
  std::vector<size_t> input_shape;
  size_t input_size = 1;
  for (const auto &d : input_py.sizes()) {
    input_shape.push_back(d);
    input_size *= d;
  }
  NVTE_CHECK(input_shape.size() > 0, "Input tensor has 0 dims");

  // Split input tensor along dim 0
  std::vector<TensorWrapper> input_list;
  std::vector<std::vector<size_t>> split_shapes;
  size_t dim0_offset = 0;
  const size_t dim0_stride =
      input_shape[0] == 0 ? 0 : input_py.element_size() * input_size / input_shape[0];
  for (size_t i = 0; i < num_splits; ++i) {
    NVTE_CHECK(dim0_offset + split_sections[i] <= input_shape[0],
               "Attempted to split tensor with shape=", input_shape,
               " along dim 0 with split_sections=", split_sections);
    split_shapes.push_back(input_shape);
    auto &split_shape = split_shapes.back();
    split_shape[0] = split_sections[i];
    void *split_dptr = static_cast<void *>(input_dptr + dim0_offset * dim0_stride);
    input_list.emplace_back(makeTransformerEngineTensor(split_dptr, split_shape, input_dtype));
    dim0_offset += split_sections[i];
  }

  // Convert quantizers to C++ objects
  std::vector<std::unique_ptr<Quantizer>> quantizer_cpp_list;
  for (size_t i = 0; i < num_splits; i++) {
    quantizer_cpp_list.push_back(convert_quantizer(quantizer_list[i]));
  }

  // Choose implementation for allocating and populating tensors
  enum class AllocationMethod { UNFUSED, BULK_FP8_BLOCKWISE, BULK_MXFP8, BULK_NVFP4 };
  enum class QuantizationMethod { UNFUSED, FUSED_NVFP4 };
  AllocationMethod allocation_method = AllocationMethod::UNFUSED;
  QuantizationMethod quantization_method = QuantizationMethod::UNFUSED;
  if (!disable_bulk_allocation) {
    if (std::all_of(quantizer_list.begin(), quantizer_list.end(),
                    [](const py::handle &quantizer) -> bool {
                      return detail::IsFloat8BlockwiseQuantizers(quantizer.ptr());
                    })) {
      allocation_method = AllocationMethod::BULK_FP8_BLOCKWISE;
    } else if (std::all_of(quantizer_list.begin(), quantizer_list.end(),
                           [](const py::handle &quantizer) -> bool {
                             return detail::IsMXFP8Quantizers(quantizer.ptr());
                           })) {
      allocation_method = AllocationMethod::BULK_MXFP8;
    } else if (std::all_of(quantizer_list.begin(), quantizer_list.end(),
                           [](const py::handle &quantizer) -> bool {
                             return detail::IsNVFP4Quantizers(quantizer.ptr());
                           })) {
      allocation_method = AllocationMethod::BULK_NVFP4;
      quantization_method = QuantizationMethod::FUSED_NVFP4;
    }
  }

  // Allocate output tensors
  std::vector<TensorWrapper> output_cpp_list;
  std::vector<py::object> output_py_list;
  switch (allocation_method) {
    case AllocationMethod::BULK_FP8_BLOCKWISE: {
      // Bulk allocation for FP8 block-scaling tensors
      std::vector<Float8BlockQuantizer *> blockwise_quantizers;
      for (auto &quantizer : quantizer_cpp_list) {
        blockwise_quantizers.push_back(static_cast<Float8BlockQuantizer *>(quantizer.get()));
      }
      std::tie(output_py_list, output_cpp_list) =
          bulk_allocate_fp8_blockwise_tensors(split_shapes, quantizer_list, blockwise_quantizers);
      break;
    }
    case AllocationMethod::BULK_MXFP8: {
      // Bulk allocation for MXFP8 tensors
      std::vector<MXFP8Quantizer *> mxfp8_quantizers;
      for (auto &quantizer : quantizer_cpp_list) {
        mxfp8_quantizers.push_back(static_cast<MXFP8Quantizer *>(quantizer.get()));
      }
      std::tie(output_py_list, output_cpp_list) =
          bulk_allocate_mxfp8_tensors(split_shapes, quantizer_list, mxfp8_quantizers);
      break;
    }
    case AllocationMethod::BULK_NVFP4: {
      // Bulk allocation for NVFP4 tensors
      std::vector<NVFP4Quantizer *> nvfp4_quantizers;
      for (auto &quantizer : quantizer_cpp_list) {
        nvfp4_quantizers.push_back(static_cast<NVFP4Quantizer *>(quantizer.get()));
      }
      bool contiguous_data_and_scale;
      std::tie(output_py_list, output_cpp_list, contiguous_data_and_scale) =
          bulk_allocate_nvfp4_tensors(split_shapes, quantizer_list, nvfp4_quantizers);
      if (!contiguous_data_and_scale) {
        // Avoid fused quantize kernel if data is not contiguous
        quantization_method = QuantizationMethod::UNFUSED;
      }
      break;
    }
    default: {
      // Allocate output tensors individually
      for (size_t i = 0; i < num_splits; ++i) {
        auto [output_cpp, output_py] =
            quantizer_cpp_list[i]->create_tensor(split_shapes[i], input_dtype);
        output_cpp_list.emplace_back(std::move(output_cpp));
        output_py_list.emplace_back(std::move(output_py));
      }
    }
  }

  // Quantize into output tensors
  switch (quantization_method) {
    case QuantizationMethod::FUSED_NVFP4: {
      // Fused NVFP4 quantize kernel
      auto input_nvte = makeTransformerEngineTensor(input_dptr, input_shape, input_dtype);
      std::vector<NVFP4Quantizer *> nvfp4_quantizers;
      for (auto &quantizer : quantizer_cpp_list) {
        nvfp4_quantizers.push_back(static_cast<NVFP4Quantizer *>(quantizer.get()));
      }
      split_quantize_nvfp4_impl(input_nvte, input_list, output_cpp_list, split_sections,
                                nvfp4_quantizers);
      break;
    }
    default:
      // General multi-tensor quantization
      multi_tensor_quantize_impl(input_list, quantizer_list, quantizer_cpp_list, output_cpp_list);
  }

  return output_py_list;
}

}  // namespace pytorch
}  // namespace transformer_engine
