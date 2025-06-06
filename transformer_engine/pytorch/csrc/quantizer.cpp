/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

constexpr size_t MXFP8_BLOCK_SIZE = 32;

bool tensor_is_reusable(const at::Tensor& tensor, const std::vector<int64_t>& shape,
                        const at::TensorOptions& opts) {
  const auto& tensor_shape = tensor.sizes();
  if (!tensor_shape.equals(shape)) return false;
  const at::TensorOptions& tensor_opts = tensor.options();
  if (opts.dtype() != tensor_opts.dtype()) return false;
  if (opts.device().type() != tensor_opts.device().type()) return false;
  if (opts.device_index() != tensor_opts.device_index() && opts.device_index() != -1) return false;
  return true;
}

Quantizer::Quantizer(const py::handle& quantizer) {
  if (quantizer.is_none()) {
    this->rowwise_usage = true;
    this->columnwise_usage = true;
    this->internal = false;
  } else {
    this->rowwise_usage = quantizer.attr("rowwise_usage").cast<bool>();
    this->columnwise_usage = quantizer.attr("columnwise_usage").cast<bool>();
    this->internal = quantizer.attr("internal").cast<bool>();
    this->quantizer = quantizer;
  }
}

Float8Quantizer::Float8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  const at::Tensor& scale = quantizer.attr("scale").cast<at::Tensor>();
  const at::Tensor& amax = quantizer.attr("amax").cast<at::Tensor>();
  const DType type = quantizer.attr("dtype").cast<DType>();

  this->amax = amax;
  this->scale = scale;
  this->dtype = type;
}

// Create torch tensor reusing existing data if possible
at::Tensor _create_torch_tensor(const std::vector<int64_t>& shape, const at::TensorOptions& opts,
                                const py::object& tensor_to_reuse,
                                bool zero_out) {
  if (!tensor_to_reuse.is_none()) {
    // Reuse output
    const at::Tensor temp = tensor_to_reuse.cast<at::Tensor>();
    if (tensor_is_reusable(temp, shape, opts)) {
      if (zero_out) {
        temp.zero_();
      }
      return temp;
    }
  }
  if (zero_out) {
    return at::zeros(shape, opts);
  }
  return at::empty(shape, opts);
}

// Create torch tensor reusing existing data is possible
// The reused tensor is tensor_to_reuse.attr_name
at::Tensor create_torch_tensor(const std::vector<int64_t>& shape, const at::TensorOptions& opts,
                               const py::object& tensor_to_reuse,
                               const std::string_view& attr_name,
                               bool zero_out = false) {
  py::object tensor{py::none()};
  if (!tensor_to_reuse.is_none()) {
    tensor = tensor_to_reuse.attr(attr_name.data());
  }
  return _create_torch_tensor(shape, opts, tensor, zero_out);
}

std::pair<TensorWrapper, py::object> NoneQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, const py::object& output,
    std::optional<at::Tensor> rowwise_data) const {
  at::TensorOptions opts;
  opts = opts.dtype(GetATenDType(dtype)).device(torch::kCUDA);
  std::vector<int64_t> torch_shape(shape.begin(), shape.end());
  at::Tensor ret;
  if (rowwise_data.has_value()) {
    ret = std::move(*rowwise_data);
  } else {
    ret = _create_torch_tensor(torch_shape, opts, output, false);
  }

  TensorWrapper tensor;
  tensor.set_rowwise_data(ret.data_ptr(), dtype, shape);
  return {std::move(tensor), py::cast(ret)};
}

void Float8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  tensor->set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> Float8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, const py::object& output,
    std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;

  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  std::vector<int64_t> rowwise_torch_shape(shape.begin(), shape.end());

  std::optional<at::Tensor> data = std::nullopt;
  std::optional<at::Tensor> columnwise_data = std::nullopt;
  // TODO: Replace with an empty tensor.
  at::Tensor scale_inv = at::reciprocal(scale);

  bool create_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  TensorWrapper tensor(this->get_scaling_mode());
  if (!output.is_none()) {
    NVTE_CHECK(detail::IsFloat8Tensor(output.ptr()), "Wrong Tensor type provided for reuse. ",
               "Expected Float8Tensor or Float8TensorBase, but got ",
               py::repr(output).cast<std::string>());
  }

  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = create_torch_tensor(rowwise_torch_shape, opts, output, "_data");
    }

    tensor.set_rowwise_data(data->data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }

  if (create_transpose) {
    std::vector<int64_t> columnwise_torch_shape;
    columnwise_torch_shape.reserve(shape.size());
    if (!shape.empty()) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape.back()));
    }
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
    }
    std::vector<size_t> transposed_shape(columnwise_torch_shape.begin(),
                                         columnwise_torch_shape.end());

    columnwise_data = create_torch_tensor(columnwise_torch_shape, opts, output, "_transpose");
    tensor.set_columnwise_data(columnwise_data->data_ptr(), this->dtype, transposed_shape);
    tensor.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }

  opts = opts.dtype(torch::kFloat32);
  py::object ret;
  if (output.is_none()) {
    if (internal) {
      py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
      ret = Float8TensorClass("data"_a = data, "fp8_scale_inv"_a = scale_inv,
                              "fp8_dtype"_a = this->dtype, "data_transpose"_a = columnwise_data,
                              "quantizer"_a = this->quantizer);
    } else {
      py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
      ret = Float8TensorClass("shape"_a = rowwise_torch_shape, "dtype"_a = GetATenDType(dtype),
                              "data"_a = data, "fp8_scale_inv"_a = scale_inv,
                              "fp8_dtype"_a = this->dtype, "data_transpose"_a = columnwise_data,
                              "quantizer"_a = this->quantizer);
    }
  } else {
    output.attr("_data") = data;
    output.attr("_scale_inv") = scale_inv;
    output.attr("_fp8_dtype") = this->dtype;
    output.attr("_transpose") = columnwise_data;
    output.attr("_quantizer") = this->quantizer;
    output.attr("_transpose_invalid") = (columnwise_data == std::nullopt);
    ret = output;
  }

  this->set_quantization_params(&tensor);
  return {std::move(tensor), std::move(ret)};
}

Float8CurrentScalingQuantizer::Float8CurrentScalingQuantizer(const py::handle& quantizer)
    : Quantizer(quantizer) {
  const at::Tensor& scale = quantizer.attr("scale").cast<at::Tensor>();
  const at::Tensor& amax = quantizer.attr("amax").cast<at::Tensor>();
  const DType type = quantizer.attr("dtype").cast<DType>();
  this->amax = amax;
  this->scale = scale;
  this->dtype = type;

  // Get amax reduction group if needed
  const bool with_amax_reduction = quantizer.attr("with_amax_reduction").cast<bool>();
  c10::intrusive_ptr<dist_group_type> amax_reduction_group;
  if (with_amax_reduction) {
    auto group = quantizer.attr("_canonicalized_amax_reduction_group")();
    NVTE_CHECK(!group.is_none(),
               "Float8CurrentScalingQuantizer could not canonicalize amax reduction group");
    amax_reduction_group = group.cast<c10::intrusive_ptr<dist_group_type>>();
  }
  this->with_amax_reduction = with_amax_reduction;
  this->amax_reduction_group = amax_reduction_group;

  // fp8 current scaling specific quantization params
  this->force_pow_2_scales = quantizer.attr("force_pow_2_scales").cast<bool>();
  this->amax_epsilon = quantizer.attr("amax_epsilon").cast<float>();
}

void Float8CurrentScalingQuantizer::set_quantization_params(TensorWrapper* tensor) const {
  // transfer amax and scale pointer from quantizer to output tensor (only as gpu buffer, no meaningful data in them)
  tensor->set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  tensor->set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
  // quantize output and its transpose
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> Float8CurrentScalingQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, const py::object& output,
    std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> scale_inv_torch_shape = {1};  // Shape of 1 element for scale_inv

  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  std::vector<int64_t> rowwise_torch_shape(shape.begin(), shape.end());

  std::optional<at::Tensor> data = std::nullopt;
  std::optional<at::Tensor> columnwise_data = std::nullopt;
  // In current scaling, scale is not known but we initialize it with 1 to avoid division by zero. If scale is already calculated, it can be correctly set.
  at::Tensor scale_inv = at::reciprocal(scale);

  TensorWrapper tensor(this->get_scaling_mode());

  bool create_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  if (!output.is_none()) {
    NVTE_CHECK(detail::IsFloat8Tensor(output.ptr()), "Wrong Tensor type provided for reuse. ",
               "Expected Float8Tensor or Float8TensorBase, but got ",
               py::repr(output).cast<std::string>());
  }
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = create_torch_tensor(rowwise_torch_shape, opts, output, "_data");
    }

    tensor.set_rowwise_data(data->data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }

  if (create_transpose) {
    std::vector<int64_t> columnwise_torch_shape;
    columnwise_torch_shape.reserve(shape.size());
    if (!shape.empty()) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape.back()));
    }
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
    }
    columnwise_data = create_torch_tensor(columnwise_torch_shape, opts, output, "_transpose");
    std::vector<size_t> transposed_shape(columnwise_torch_shape.begin(),
                                         columnwise_torch_shape.end());
    tensor.set_columnwise_data(columnwise_data->data_ptr(), this->dtype, transposed_shape);
    tensor.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }

  py::object ret;
  if (output.is_none()) {
    if (internal) {
      py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
      ret = Float8TensorClass("data"_a = data, "fp8_scale_inv"_a = scale_inv,
                              "fp8_dtype"_a = this->dtype, "data_transpose"_a = columnwise_data,
                              "quantizer"_a = this->quantizer);
    } else {
      py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
      ret = Float8TensorClass("shape"_a = rowwise_torch_shape, "dtype"_a = GetATenDType(dtype),
                              "data"_a = data, "fp8_scale_inv"_a = scale_inv,
                              "fp8_dtype"_a = this->dtype, "data_transpose"_a = columnwise_data,
                              "quantizer"_a = this->quantizer);
    }
  } else {
    output.attr("_data") = data;
    output.attr("_scale_inv") = scale_inv;
    output.attr("_fp8_dtype") = this->dtype;
    output.attr("_transpose") = columnwise_data;
    output.attr("_quantizer") = this->quantizer;
    output.attr("_transpose_invalid") = (columnwise_data == std::nullopt);
    ret = output;
  }

  this->set_quantization_params(&tensor);

  return {std::move(tensor), std::move(ret)};
}

Float8BlockQuantizer::Float8BlockQuantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
  this->block_scaling_dim = quantizer.attr("block_scaling_dim").cast<int>();
  this->force_pow_2_scales = quantizer.attr("force_pow_2_scales").cast<bool>();
  this->amax_epsilon = quantizer.attr("amax_epsilon").cast<float>();
  NVTE_CHECK(this->block_scaling_dim == 1 || this->block_scaling_dim == 2,
             "Unsupported block scaling dim.");
}

void Float8BlockQuantizer::set_quantization_params(TensorWrapper* tensor) const {
  // Change the rowwise and columnwise_data to the configured dtype.
  // May be a switch between E5M2 and E4M3.
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> Float8BlockQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, const py::object& output,
    std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape(shape.begin(), shape.end());
  size_t numel = product(shape);

  TensorWrapper tensor(this->get_scaling_mode());
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  at::TensorOptions scale_opts;
  scale_opts = scale_opts.dtype(torch::kFloat32).device(torch::kCUDA);

  std::optional<at::Tensor> data_rowwise, data_colwise, scale_inv_rowwise, scale_inv_colwise;

  size_t k_dim = shape.size() == 0 ? 1u : shape.back();
  size_t m_dim = numel / k_dim;
  constexpr size_t kBlockLen = 128;

  if (!output.is_none()) {
    NVTE_CHECK(detail::IsFloat8BlockwiseQTensor(output.ptr()),
               "Wrong Tensor type provided for reuse. ",
               "Expected Float8BlockwiseQTensor or Float8BlockwiseQTensorBase, but got ",
               py::repr(output).cast<std::string>());
  }

  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data_rowwise = std::move(*rowwise_data);
    } else {
      data_rowwise = create_torch_tensor(torch_shape, opts, output, "_rowwise_data");
    }
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    switch (block_scaling_dim) {
      case 1: {
        sinv0 = divup(k_dim, kBlockLen);
        sinv1 = roundup(m_dim, 4lu);
      } break;
      case 2: {
        sinv0 = divup(m_dim, kBlockLen);
        sinv1 = roundup(divup(k_dim, kBlockLen), 4lu);
      } break;
      default: {
        NVTE_ERROR(
            "Unsupported block_scaling_dim in create_tensor rowwise."
            "Expected 1 or 2. Got ",
            block_scaling_dim);
      } break;
    }
    scale_inv_rowwise =
        create_torch_tensor({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, scale_opts,
                            output, "_rowwise_scale_inv");
    tensor.set_rowwise_data(data_rowwise->data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv_rowwise->data_ptr(), DType::kFloat32,
                                 std::vector<size_t>{sinv0, sinv1});
  }

  if (columnwise_usage) {
    std::vector<int64_t> torch_columnwise_shape;
    if (torch_shape.size() > 0) {
      torch_columnwise_shape.reserve(torch_shape.size());
      torch_columnwise_shape.push_back(torch_shape[torch_shape.size() - 1]);
      for (size_t i = 0; i < torch_shape.size() - 1; ++i) {
        torch_columnwise_shape.push_back(torch_shape[i]);
      }
    }
    std::vector<size_t> columnwise_shape(torch_columnwise_shape.begin(),
                                         torch_columnwise_shape.end());
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    switch (block_scaling_dim) {
      case 1: {
        sinv0 = divup(m_dim, kBlockLen);
        sinv1 = roundup(k_dim, 4lu);
      } break;
      case 2: {
        sinv0 = divup(k_dim, kBlockLen);
        sinv1 = roundup(divup(m_dim, kBlockLen), 4lu);
      } break;
      default: {
        NVTE_ERROR(
            "Unsupported block_scaling_dim in create_tensor columnwise."
            "Expected 1 or 2. Got ",
            block_scaling_dim);
      } break;
    }
    data_colwise = create_torch_tensor(torch_columnwise_shape, opts, output, "_columnwise_data");
    scale_inv_colwise =
        create_torch_tensor({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, scale_opts,
                            output, "_columnwise_scale_inv");

    tensor.set_columnwise_data(data_colwise->data_ptr(), this->dtype, columnwise_shape);
    tensor.set_columnwise_scale_inv(scale_inv_colwise->data_ptr(), DType::kFloat32,
                                    std::vector<size_t>{sinv0, sinv1});
  }
  this->set_quantization_params(&tensor);

  py::object ret;
  if (output.is_none()) {
    if (internal) {
      py::handle Float8BlockwiseQTensorClass(
          reinterpret_cast<PyObject*>(Float8BlockwiseQTensorBasePythonClass));
      ret = Float8BlockwiseQTensorClass(
          "rowwise_data"_a = data_rowwise, "columnwise_data"_a = data_colwise,
          "rowwise_scale_inv"_a = scale_inv_rowwise, "columnwise_scale_inv"_a = scale_inv_colwise,
          "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer,
          "is_2D_scaled"_a = (block_scaling_dim == 2));
    } else {
      py::handle Float8BlockwiseQTensorClass(
          reinterpret_cast<PyObject*>(Float8BlockwiseQTensorPythonClass));
      ret = Float8BlockwiseQTensorClass(
          "shape"_a = torch_shape, "dtype"_a = GetATenDType(dtype), "rowwise_data"_a = data_rowwise,
          "columnwise_data"_a = data_colwise, "rowwise_scale_inv"_a = scale_inv_rowwise,
          "columnwise_scale_inv"_a = scale_inv_colwise, "fp8_dtype"_a = this->dtype,
          "quantizer"_a = this->quantizer, "is_2D_scaled"_a = (block_scaling_dim == 2));
    }
  } else {
    output.attr("_rowwise_data") = data_rowwise;
    output.attr("_columnwise_data") = data_colwise;
    output.attr("_quantizer") = this->quantizer;
    output.attr("_fp8_dtype") = this->dtype;
    output.attr("_rowwise_scale_inv") = scale_inv_rowwise;
    output.attr("_columnwise_scale_inv") = scale_inv_colwise;
    output.attr("_is_2D_scaled") = (block_scaling_dim == 2);
    ret = output;
  }

  return {std::move(tensor), std::move(ret)};
}

MXFP8Quantizer::MXFP8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
}

void MXFP8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> MXFP8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, const py::object& output,
    std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape(shape.begin(), shape.end());
  size_t numel = product(shape);

  TensorWrapper tensor(NVTE_MXFP8_1D_SCALING);
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);

  std::optional<at::Tensor> data_rowwise, data_colwise, rowwise_scale_inv, columnwise_scale_inv;

  auto last_dim = static_cast<size_t>(torch_shape.back());

  NVTE_CHECK(last_dim % MXFP8_BLOCK_SIZE == 0 && (numel / last_dim) % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisble by ", MXFP8_BLOCK_SIZE,
             " (got shape=", torch_shape, ")");

  if (!output.is_none()) {
    NVTE_CHECK(detail::IsMXFP8Tensor(output.ptr()), "Wrong Tensor type provided for reuse. ",
               "Expected MXFP8Tensor or MXFP8TensorBase, but got ",
               py::repr(output).cast<std::string>());
  }

  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data_rowwise = std::move(*rowwise_data);
    } else {
      data_rowwise = create_torch_tensor(torch_shape, opts, output, "_rowwise_data");
    }
    auto sinv0 = roundup(numel / last_dim, 128lu);
    auto sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4lu);
    rowwise_scale_inv =
        create_torch_tensor({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, opts,
                            output, "_rowwise_scale_inv", true);
    tensor.set_rowwise_data(data_rowwise->data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(
        rowwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
        std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }

  if (columnwise_usage) {
    auto sinv0 = roundup(numel / (last_dim * MXFP8_BLOCK_SIZE), 4lu);
    auto sinv1 = roundup(last_dim, 128lu);
    data_colwise = create_torch_tensor(torch_shape, opts, output, "_columnwise_data");
    columnwise_scale_inv =
        create_torch_tensor({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, opts,
                            output, "_columnwise_scale_inv", true);

    tensor.set_columnwise_data(data_colwise->data_ptr(), this->dtype, shape);
    tensor.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                    std::vector<size_t>{sinv0, sinv1});
  }
  this->set_quantization_params(&tensor);

  py::object ret;
  if (output.is_none()) {
    if (internal) {
      py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorBasePythonClass));
      ret = MXFP8TensorClass("rowwise_data"_a = data_rowwise, "columnwise_data"_a = data_colwise,
                             "rowwise_scale_inv"_a = rowwise_scale_inv,
                             "columnwise_scale_inv"_a = columnwise_scale_inv,
                             "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
    } else {
      py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorPythonClass));
      ret = MXFP8TensorClass("shape"_a = torch_shape, "dtype"_a = GetATenDType(dtype),
                             "rowwise_data"_a = data_rowwise, "columnwise_data"_a = data_colwise,
                             "rowwise_scale_inv"_a = rowwise_scale_inv,
                             "columnwise_scale_inv"_a = columnwise_scale_inv,
                             "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
    }
  } else {
    output.attr("_rowwise_data") = data_rowwise;
    output.attr("_columnwise_data") = data_colwise;
    output.attr("_quantizer") = this->quantizer;
    output.attr("_fp8_dtype") = this->dtype;
    output.attr("_rowwise_scale_inv") = rowwise_scale_inv;
    output.attr("_columnwise_scale_inv") = columnwise_scale_inv;
    ret = output;
  }

  return {std::move(tensor), std::move(ret)};
}

}  // namespace transformer_engine::pytorch
