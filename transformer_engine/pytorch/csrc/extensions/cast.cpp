/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

#include <cstdint>
#include <optional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "transformer_engine/cast.h"
#include "transformer_engine/transformer_engine.h"

#include "common.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

py::object quantize(const at::Tensor& tensor, py::handle quantizer, const py::object& output,
                    std::optional<at::Tensor> noop) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);
  auto input_tensor = tensor.contiguous();

  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);
  const auto& te_input_shape = te_input.shape();
  std::vector<size_t> input_shape(te_input_shape.data, te_input_shape.data + te_input_shape.ndim);
  auto fake_tensor_type = tensor.scalar_type();
  if (!detail::IsFloatingPointType(fake_tensor_type)) {
    fake_tensor_type = at::kFloat;
  }

  TensorWrapper te_output;
  py::object out;
  if (output.is_none()) {
    DType fake_te_type = GetTransformerEngineDType(fake_tensor_type);
    std::tie(te_output, out) = my_quantizer->create_tensor(input_shape, fake_te_type);
  } else {
    out = output;
    te_output = makeTransformerEngineTensor(output, quantizer);
  }

  TensorWrapper te_noop;
  if (noop.has_value()) {
    te_noop = makeTransformerEngineTensor(*noop);
  } else {
    te_noop = TensorWrapper();
  }

  if (te_output.numel() == 0) return out;

  quantize_cpp(te_input, quantizer, my_quantizer, te_output, te_noop);

  return out;
}

// C++ backend of quantize with tensor converted
// quantizer_py is still required for runtime type check
void quantize_cpp(const TensorWrapper& te_input, py::handle quantizer_py,
                  std::unique_ptr<Quantizer>& quantizer_cpp, TensorWrapper& te_output,
                  TensorWrapper& te_noop) {
  if (te_input.numel() == 0) {
    return;
  }
  if (te_output.numel() == 0) {
    NVTE_CHECK(te_input.numel() == 0, "Input tensor is not 0 when output tensor is 0");
  }

  QuantizationConfigWrapper quant_config;
  quant_config.set_noop_tensor(te_noop.data());

  if (detail::IsFloat8CurrentScalingQuantizers(quantizer_py.ptr())) {
    auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_amax(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());
    });
    // check if we need to do amax reudction (depending on model parallel configs)
    if (my_quantizer_cs->with_amax_reduction) {
      c10::intrusive_ptr<dist_group_type> process_group_ptr = my_quantizer_cs->amax_reduction_group;
      // construct torch tesnor from NVTEBasicTensor without reallocating memory
      at::Tensor& amax_tensor_torch = my_quantizer_cs->amax;
      std::vector<at::Tensor> tensors = {amax_tensor_torch};
      // allreduce amax tensor
      c10d::AllreduceOptions allreduce_opts;
      allreduce_opts.reduceOp = c10d::ReduceOp::MAX;
      process_group_ptr->allreduce(tensors, allreduce_opts)->wait();
    }
    // this config is used for cs scaling factor computation
    // because compute scale is cannot be fused with quantize kernel
    // so in nvte_quantize_v2 with current scaling, the quant config is not used again
    quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_scale_from_amax(te_output.data(), quant_config,
                                   at::cuda::getCurrentCUDAStream());
    });
    // set amax ptr to null in te_output TensorWrapper to avoid atomic amax updates in kernel
    te_output.set_amax(nullptr, DType::kFloat32, te_output.defaultShape);
  } else if (detail::IsFloat8BlockwiseQuantizers(quantizer_py.ptr())) {
    auto my_quantizer_bw = static_cast<Float8BlockQuantizer*>(quantizer_cpp.get());
    quant_config.set_force_pow_2_scales(my_quantizer_bw->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_bw->amax_epsilon);
    if (my_quantizer_bw->all_gather_usage) {
      quant_config.set_float8_block_scale_tensor_format(Float8BlockScaleTensorFormat::COMPACT);
    }
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(te_input.data(), te_output.data(), quant_config,
                     at::cuda::getCurrentCUDAStream());
  });
}

py::object dequantize(const py::handle& input, transformer_engine::DType otype) {
  init_extension();

  const auto none = py::none();

  const auto& input_tensor = makeTransformerEngineTensor(input, none);

  NoneQuantizer q(none);

  const auto& shape = convertShape(input_tensor.shape());

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
  init_extension();

  // Check number of tensors
  const size_t num_tensors = input_list.size();
  NVTE_CHECK(quantizer_py_list.size() == num_tensors,
             "Expected ", num_tensors, " Python quantizers, but got ", quantizer_py_list.size());
  NVTE_CHECK(quantizer_cpp_list.size() == num_tensors,
             "Expected ", num_tensors, " C++ quantizers, but got ", quantizer_cpp_list.size());
  NVTE_CHECK(output_list.size() == num_tensors,
             "Expected ", num_tensors, " output tensors, but got ", output_list.size());

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 delayed scaling
  bool with_fused_kernel = true;
  for (size_t i = 0; i < num_tensors; i++) {
    if (!detail::IsFloat8Quantizers(quantizer_py_list[i].ptr())) {
      with_fused_kernel = false;
      break;
    }
    if (nvte_tensor_columnwise_data(output_list[i].data()) == nullptr) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    // Fused kernel for multi-tensor quantize
    std::vector<NVTETensor> nvte_tensor_input_list;
    std::vector<NVTETensor> nvte_tensor_output_list;
    for (size_t i=0; i<num_tensors; ++i) {
      nvte_tensor_input_list.push_back(input_list[i].data());
      nvte_tensor_output_list.push_back(output_list[i].data());
    }
    NVTE_SCOPED_GIL_RELEASE({
      nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                                nvte_tensor_output_list.data(), at::cuda::getCurrentCUDAStream());
    });
  } else {
    // Quantize kernels individually
    TensorWrapper dummy_noop_flag;
    for (size_t i=0; i<num_tensors; ++i) {
      quantize_cpp(input_list[i], quantizer_py_list[i], quantizer_cpp_list[i], output_list[i],
                   dummy_noop_flag);
    }
  }
}

}  // namespace

std::vector<py::object> multi_tensor_quantize(std::vector<at::Tensor> input_list,
                                              std::vector<py::handle> quantizer_list) {
  // Check number of tensors
  const size_t num_tensors = input_list.size();
  NVTE_CHECK(quantizer_list.size() == num_tensors,
             "Expected ", num_tensors, " quantizers, but got ", quantizer_list.size());

  // Convert quantizers to C++ objects
  std::vector<std::unique_ptr<Quantizer>> quantizer_cpp_list;
  for (size_t i = 0; i < num_tensors; i++) {
    quantizer_cpp_list.push_back(convert_quantizer(quantizer_list[i]));
  }

  // Initialize input and output tensors
  std::vector<TensorWrapper> input_cpp_list;
  std::vector<TensorWrapper> output_cpp_list;
  std::vector<py::object> output_py_list;
  for (size_t i=0; i<num_tensors; ++i) {
    // Convert input tensor to C++ object
    const auto &input_py = input_list[i];
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

std::tuple<std::vector<py::object>, std::vector<TensorWrapper>>
bulk_allocate_fp8_blockwise_tensors(std::vector<std::vector<size_t>> &shape_list,
                                    std::vector<py::handle> &quantizer_py_list,
                                    std::vector<Float8BlockQuantizer*> &quantizer_cpp_list) {
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
  auto make_torch_view = [](std::shared_ptr<at::Tensor> &buffer,
                            const std::vector<size_t> &shape,
                            size_t offset,
                            at::ScalarType dtype) -> at::Tensor {
    std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    return at::from_blob(buffer->data_ptr<uint8_t>() + offset,
                         shape_int64,
                         [buffer](void*) {},  // deleter holds shared_ptr
                         at::device(at::kCUDA).dtype(dtype));
  };

  // Allocate row-wise data
  std::vector<at::Tensor> rowwise_data_list, rowwise_scale_list;
  std::vector<std::vector<size_t>> rowwise_data_shapes, rowwise_scale_shapes;
  if (rowwise_usage) {
    // Tensor sizes
    for (size_t i=0; i<num_tensors; ++i) {
      rowwise_data_shapes.emplace_back(shape_list[i]);
      rowwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], false));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets;
    for (size_t i=0; i<num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 256);  // align to 256B
      data_offsets.push_back(buffer_size);
      buffer_size += product(rowwise_data_shapes[i]) * fp8_elem_size;
    }
    for (size_t i=0; i<num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 16);  // align to 16B
      scale_offsets.push_back(buffer_size);
      buffer_size += product(rowwise_scale_shapes[i]) * scale_elem_size;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(at::empty({(int64_t)buffer_size},
                                                         at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i=0; i<num_tensors; ++i) {
      rowwise_data_list.emplace_back(make_torch_view(buffer,
                                                     rowwise_data_shapes[i],
                                                     data_offsets[i],
                                                     torch::kUInt8));
      rowwise_scale_list.emplace_back(make_torch_view(buffer,
                                                      rowwise_scale_shapes[i],
                                                      scale_offsets[i],
                                                      torch::kFloat32));
    }
  }

  // Allocate column-wise data
  std::vector<at::Tensor> columnwise_data_list, columnwise_scale_list;
  std::vector<std::vector<size_t>> columnwise_data_shapes, columnwise_scale_shapes;
  if (columnwise_usage) {
    // Tensor sizes
    for (size_t i=0; i<num_tensors; ++i) {
      columnwise_data_shapes.emplace_back();
      auto &shape = columnwise_data_shapes.back();
      shape.push_back(shape_list[i].back());
      for (size_t j=0; j<shape_list[i].size()-1; ++j) {
        shape.push_back(shape_list[i][j]);
      }
      columnwise_scale_shapes.emplace_back(
          quantizer_cpp_list[i]->get_scale_shape(shape_list[i], true));
    }

    // Offsets in full buffer
    size_t buffer_size = 0;
    std::vector<size_t> data_offsets, scale_offsets;
    for (size_t i=0; i<num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 256);  // align to 256B
      data_offsets.push_back(buffer_size);
      buffer_size += product(columnwise_data_shapes[i]) * fp8_elem_size;
    }
    for (size_t i=0; i<num_tensors; ++i) {
      buffer_size = roundup(buffer_size, 16);  // align to 16B
      scale_offsets.push_back(buffer_size);
      buffer_size += product(columnwise_scale_shapes[i]) * scale_elem_size;
    }

    // Allocate full buffer
    auto buffer = std::make_shared<at::Tensor>(at::empty({(int64_t)buffer_size},
                                                         at::device(at::kCUDA).dtype(torch::kUInt8)));

    // Construct tensor views
    for (size_t i=0; i<num_tensors; ++i) {
      columnwise_data_list.emplace_back(make_torch_view(buffer,
                                                        columnwise_data_shapes[i],
                                                        data_offsets[i],
                                                        torch::kUInt8));
      columnwise_scale_list.emplace_back(make_torch_view(buffer,
                                                         columnwise_scale_shapes[i],
                                                         scale_offsets[i],
                                                         torch::kFloat32));
    }
  }

  // Construct FP8 block-wise tensors
  py::handle Float8BlockwiseQTensorClass(
    reinterpret_cast<PyObject*>(Float8BlockwiseQTensorBasePythonClass));
  for (size_t i=0; i<num_tensors; ++i) {

    // Create tensor objects with proper reference counting
    py::object rowwise_data = rowwise_usage ? py::cast(rowwise_data_list[i]) : py::none();
    py::object rowwise_scale = rowwise_usage ? py::cast(rowwise_scale_list[i]) : py::none();
    py::object columnwise_data = (columnwise_usage
                                  ? py::cast(columnwise_data_list[i]) : py::none());
    py::object columnwise_scale = (columnwise_usage
                                   ? py::cast(columnwise_scale_list[i]) : py::none());

    // Construct Python tensor
    tensor_py_list.emplace_back(Float8BlockwiseQTensorClass(
        rowwise_data, rowwise_scale, columnwise_data, columnwise_scale, fp8_dtype,
        quantizer_py_list[i], is_2D_scaled, Float8BlockScaleTensorFormat::GEMM_READY));

    // Construct C++ tensor
    tensor_cpp_list.emplace_back(makeTransformerEngineTensor(
        rowwise_usage ? rowwise_data_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_data_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_data_shapes[i] : std::vector<size_t>{},
        columnwise_usage ? columnwise_data_shapes[i] : std::vector<size_t>{},
        fp8_dtype, nullptr, nullptr,
        rowwise_usage ? rowwise_scale_list[i].data_ptr() : nullptr,
        columnwise_usage ? columnwise_scale_list[i].data_ptr() : nullptr,
        rowwise_usage ? rowwise_scale_shapes[i] : std::vector<size_t>{},
        columnwise_usage ? columnwise_scale_shapes[i] : std::vector<size_t>{},
        scaling_mode));
  }

  return retval;
}

}  // namespace

std::vector<py::object> split_quantize(at::Tensor input,
                                       std::vector<int> split_sections,
                                       std::vector<py::handle> quantizer_list) {
  init_extension();

  // Check number of tensors
  const size_t num_tensors = split_sections.size();
  NVTE_CHECK(quantizer_list.size() == num_tensors,
             "Expected ", num_tensors, " quantizers, but got ", quantizer_list.size());
  if (num_tensors == 0) {
    return {};
  }

  // Input tensor properties
  NVTE_CHECK(input.is_contiguous(), "Input tensor is not contiguous");
  uint8_t *input_dptr = reinterpret_cast<uint8_t*>(input.data_ptr());
  auto input_dtype = GetTransformerEngineDType(input.scalar_type());
  std::vector<size_t> input_shape;
  size_t input_size = 1;
  for (const auto &d : input.sizes()) {
    input_shape.push_back(d);
    input_size *= d;
  }
  NVTE_CHECK(input_shape.size() > 0, "Input tensor has 0 dims");

  // Split input tensor along dim 0
  std::vector<TensorWrapper> input_list;
  std::vector<std::vector<size_t>> split_shapes;
  size_t dim0_offset = 0;
  const size_t dim0_stride = input.element_size() * input_size / input_shape[0];
  for (size_t i=0; i<num_tensors; ++i) {
    NVTE_CHECK(split_sections[i] >= 0,
               "Attempted to split tensor with shape=", input_shape,
               " along dim 0 with split_sections=", split_sections);
    NVTE_CHECK(dim0_offset + split_sections[i] <= input_shape[0],
               "Attempted to split tensor with shape=", input_shape,
               " along dim 0 with split_sections=", split_sections);
    split_shapes.push_back(input_shape);
    auto &split_shape = split_shapes.back();
    split_shape[0] = split_sections[i];
    void *split_dptr = static_cast<void*>(input_dptr + dim0_offset * dim0_stride);
    input_list.emplace_back(makeTransformerEngineTensor(split_dptr, split_shape, input_dtype));
    dim0_offset += split_sections[i];
  }

  // Convert quantizers to C++ objects
  std::vector<std::unique_ptr<Quantizer>> quantizer_cpp_list;
  for (size_t i = 0; i < num_tensors; i++) {
    quantizer_cpp_list.push_back(convert_quantizer(quantizer_list[i]));
  }

  // For FP8 block-scaling, we construct output tensors with bulk allocations
  bool use_fused_bulk_alloc = true;
  for (size_t i = 0; i < quantizer_list.size(); i++) {
    if (!detail::IsFloat8BlockwiseQuantizers(quantizer_list[i].ptr())) {
      use_fused_bulk_alloc = false;
      break;
    }
  }

  // Allocate output tensors
  std::vector<TensorWrapper> output_cpp_list;
  std::vector<py::object> output_py_list;
  if (!use_fused_bulk_alloc) {
    // Allocate output tensors individually
    for (size_t i=0; i<num_tensors; ++i) {
      auto [output_cpp, output_py] = quantizer_cpp_list[i]->create_tensor(split_shapes[i], input_dtype);
      output_cpp_list.emplace_back(std::move(output_cpp));
      output_py_list.emplace_back(std::move(output_py));
    }
  } else {
    // FP8 block-scaling: construct output tensors with bulk allocations
    std::vector<Float8BlockQuantizer*> blockwise_quantizers;
    for (auto &quantizer: quantizer_cpp_list) {
      blockwise_quantizers.push_back(static_cast<Float8BlockQuantizer*>(quantizer.get()));
    }
    std::tie(output_py_list, output_cpp_list)
      = bulk_allocate_fp8_blockwise_tensors(split_shapes, quantizer_list, blockwise_quantizers);
  }

  // Perform multi-tensor quantization
  multi_tensor_quantize_impl(input_list, quantizer_list, quantizer_cpp_list, output_cpp_list);

  return output_py_list;
}

template <void (*func)(const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                       cudaStream_t)>
std::vector<py::object> dbias_dact(const at::Tensor& grad_output, const at::Tensor& act_input,
                                   py::handle quantizer) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);

  auto grad_tensor = makeTransformerEngineTensor(grad_output);

  auto grad_bias = allocateTorchTensor(grad_output.size(-1), grad_tensor.dtype());
  auto act_input_tensor = makeTransformerEngineTensor(act_input);

  const auto& shape = convertShape(grad_tensor.shape());
  auto [dact_tensor, dact] = my_quantizer->create_tensor(shape, act_input_tensor.dtype());

  auto dbias_tensor = makeTransformerEngineTensor(grad_bias);

  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  NVTE_SCOPED_GIL_RELEASE({
    func(grad_tensor.data(), act_input_tensor.data(), dact_tensor.data(), dbias_tensor.data(),
         workspace.data(), at::cuda::getCurrentCUDAStream());
  });
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    func(grad_tensor.data(), act_input_tensor.data(), dact_tensor.data(), dbias_tensor.data(),
         workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(grad_bias), dact};
}

std::vector<py::object> dbias_dgelu(const at::Tensor& grad_output, const at::Tensor& act_input,
                                    py::handle quantizer) {
  return dbias_dact<nvte_quantize_dbias_dgelu>(grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsilu(const at::Tensor& grad_output, const at::Tensor& act_input,
                                    py::handle quantizer) {
  return dbias_dact<nvte_quantize_dbias_dsilu>(grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_drelu(const at::Tensor& grad_output, const at::Tensor& act_input,
                                    py::handle quantizer) {
  return dbias_dact<nvte_quantize_dbias_drelu>(grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dqgelu(const at::Tensor& grad_output, const at::Tensor& act_input,
                                     py::handle quantizer) {
  return dbias_dact<nvte_quantize_dbias_dqgelu>(grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsrelu(const at::Tensor& grad_output, const at::Tensor& act_input,
                                     py::handle quantizer) {
  return dbias_dact<nvte_quantize_dbias_dsrelu>(grad_output, act_input, quantizer);
}

}  // namespace pytorch
}  // namespace transformer_engine
