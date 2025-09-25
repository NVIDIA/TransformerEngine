/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/cast.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "../extensions.h"
#include "common.h"
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
  const auto input_cpp = makeTransformerEngineTensor(input_contiguous);

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
  quantizer_cpp->quantize(input_cpp, output_cpp, noop_flag_cpp);

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
      // Add PDL sync for the first tensor to make sure the previous kernel has flushed results to
      // the global memory. The following kernels are launched to the same stream, and there's no
      // data dependency between them.
      // Add PDL trigger for all but the last tensor, so it won't trigger the next unknown kernel.
      quantizer_cpp_list[i]->quantize(input_list[i], output_list[i], std::nullopt,
                                      /*pdl_sync=*/i == 0, /*pdl_trigger=*/i < (num_tensors - 1));
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
      reinterpret_cast<PyObject *>(Float8BlockwiseQTensorBasePythonClass));
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
        rowwise_usage ? rowwise_data_shapes[i] : std::vector<size_t>{},
        columnwise_usage ? columnwise_data_shapes[i] : std::vector<size_t>{}, fp8_dtype, nullptr,
        nullptr, rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_scale_shapes[i] : std::vector<size_t>{},
        columnwise_usage ? columnwise_scale_shapes[i] : std::vector<size_t>{}, scaling_mode));
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
  py::handle MXFP8TensorClass(reinterpret_cast<PyObject *>(MXFP8TensorBasePythonClass));
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
        rowwise_usage ? rowwise_data_shapes[i] : std::vector<size_t>{},
        columnwise_usage ? columnwise_data_shapes[i] : std::vector<size_t>{}, fp8_dtype, nullptr,
        nullptr, rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_scale_shapes[i] : std::vector<size_t>{},
        columnwise_usage ? columnwise_scale_shapes[i] : std::vector<size_t>{}, scaling_mode));
  }

  return retval;
}

}  // namespace

std::vector<py::object> split_quantize(const at::Tensor &tensor,
                                       const std::vector<int> &split_sections,
                                       std::vector<py::handle> quantizer_list) {
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
    NVTE_CHECK(split_sections[i] >= 0, "Attempted to split tensor with shape=", input_shape,
               " along dim 0 with split_sections=", split_sections);
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

  // For FP8 block-scaling, we construct output tensors with bulk allocations
  // For MXFP8, we also use bulk allocations
  bool use_fused_bulk_alloc = true;
  for (size_t i = 0; i < quantizer_list.size(); i++) {
    if (!detail::IsFloat8BlockwiseQuantizers(quantizer_list[i].ptr()) &&
        !detail::IsMXFP8Quantizers(quantizer_list[i].ptr())) {
      use_fused_bulk_alloc = false;
      break;
    }
  }

  // Allocate output tensors
  std::vector<TensorWrapper> output_cpp_list;
  std::vector<py::object> output_py_list;
  if (!use_fused_bulk_alloc) {
    // Allocate output tensors individually
    for (size_t i = 0; i < num_splits; ++i) {
      auto [output_cpp, output_py] =
          quantizer_cpp_list[i]->create_tensor(split_shapes[i], input_dtype);
      output_cpp_list.emplace_back(std::move(output_cpp));
      output_py_list.emplace_back(std::move(output_py));
    }
  } else {
    // TODO(zhongbo): make a better api to make this part less hacky
    bool is_fp8_blockwise = detail::IsFloat8BlockwiseQuantizers(quantizer_list[0].ptr());
    bool is_mxfp8 = detail::IsMXFP8Quantizers(quantizer_list[0].ptr());
    if (is_fp8_blockwise) {
      // FP8 block-scaling: construct output tensors with bulk allocations
      std::vector<Float8BlockQuantizer *> blockwise_quantizers;
      for (auto &quantizer : quantizer_cpp_list) {
        blockwise_quantizers.push_back(static_cast<Float8BlockQuantizer *>(quantizer.get()));
      }
      std::tie(output_py_list, output_cpp_list) =
          bulk_allocate_fp8_blockwise_tensors(split_shapes, quantizer_list, blockwise_quantizers);
    } else if (is_mxfp8) {
      // MXFP8: construct output tensors with bulk allocations
      std::vector<MXFP8Quantizer *> mxfp8_quantizers;
      for (auto &quantizer : quantizer_cpp_list) {
        mxfp8_quantizers.push_back(static_cast<MXFP8Quantizer *>(quantizer.get()));
      }
      std::tie(output_py_list, output_cpp_list) =
          bulk_allocate_mxfp8_tensors(split_shapes, quantizer_list, mxfp8_quantizers);
    } else {
      NVTE_CHECK(false, "Expected either FP8 block-scaling or MXFP8 quantizer");
    }
  }

  // Perform multi-tensor quantization
  multi_tensor_quantize_impl(input_list, quantizer_list, quantizer_cpp_list, output_cpp_list);

  return output_py_list;
}

}  // namespace pytorch
}  // namespace transformer_engine
