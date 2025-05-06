/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind.h>

#include "common.h"
#include "pybind.h"
#include "torch/torch.h"

namespace transformer_engine::pytorch {

constexpr size_t MXFP8_BLOCK_SIZE = 32;

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

std::pair<TensorWrapper, py::object> NoneQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  at::TensorOptions opts;
  opts = opts.dtype(GetATenDType(dtype)).device(torch::kCUDA);
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }
  at::Tensor ret;
  if (rowwise_data.has_value()) {
    ret = std::move(*rowwise_data);
  } else {
    ret = at::empty(torch_shape, opts);
  }

  TensorWrapper tensor;
  tensor.set_rowwise_data(ret.data_ptr(), dtype, shape);
  return {std::move(tensor), py::cast(ret)};
}

void Float8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  at::TensorOptions opts = opts.dtype(torch::kFloat32).device(torch::kCUDA);
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
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> rowwise_torch_shape;
  std::vector<int64_t> columnwise_torch_shape;

  if (!shape.empty()) {
    columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape.back()));
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i < shape.size() - 1) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
    }
    rowwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
  }
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  at::Tensor data;
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = at::empty(rowwise_torch_shape, opts);
    }
  }
  const py::object py_data = rowwise_usage ? py::cast(data) : py::none();
  at::Tensor columnwise_data;
  bool create_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  if (create_transpose) {
    columnwise_data = at::empty(columnwise_torch_shape, opts);
  }
  const py::object py_columnwise_data = create_transpose ? py::cast(columnwise_data) : py::none();
  opts = opts.dtype(torch::kFloat32);
  // TODO: Replace with an empty tensor.
  at::Tensor scale_inv = at::reciprocal(scale);
  py::object ret;
  if (internal) {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
    ret = Float8TensorClass("data"_a = py_data, "fp8_scale_inv"_a = scale_inv,
                            "fp8_dtype"_a = this->dtype, "data_transpose"_a = py_columnwise_data,
                            "quantizer"_a = this->quantizer);
  } else {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
    ret = Float8TensorClass("shape"_a = rowwise_torch_shape, "dtype"_a = GetATenDType(dtype),
                            "data"_a = py_data, "fp8_scale_inv"_a = scale_inv,
                            "fp8_dtype"_a = this->dtype, "data_transpose"_a = py_columnwise_data,
                            "quantizer"_a = this->quantizer);
  }
  TensorWrapper tensor(this->get_scaling_mode());
  if (rowwise_usage) {
    tensor.set_rowwise_data(data.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (create_transpose) {
    std::vector<size_t> transposed_shape;
    for (auto s : columnwise_torch_shape) {
      transposed_shape.emplace_back(static_cast<size_t>(s));
    }
    tensor.set_columnwise_data(columnwise_data.data_ptr(), this->dtype, transposed_shape);
    tensor.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
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
  at::TensorOptions opts = opts.dtype(torch::kFloat32).device(torch::kCUDA);
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
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> rowwise_torch_shape;
  std::vector<int64_t> columnwise_torch_shape;
  std::vector<int64_t> scale_inv_torch_shape = {1};  // Shape of 1 element for scale_inv

  if (!shape.empty()) {
    columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape.back()));
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i < shape.size() - 1) {
      columnwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
    }
    rowwise_torch_shape.emplace_back(static_cast<int64_t>(shape[i]));
  }
  at::TensorOptions opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  at::Tensor data;
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = at::empty(rowwise_torch_shape, opts);
    }
  }
  const py::object py_data = rowwise_usage ? py::cast(data) : py::none();
  at::Tensor columnwise_data;
  bool create_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  if (create_transpose) {
    columnwise_data = at::empty(columnwise_torch_shape, opts);
  }
  const py::object py_columnwise_data = create_transpose ? py::cast(columnwise_data) : py::none();

  // In current scaling, scale is not known but we initialize it with 1 to avoid division by zero. If scale is already calculated, it can be correctly set.
  at::Tensor scale_inv = at::reciprocal(scale);

  py::object ret;
  if (internal) {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
    ret = Float8TensorClass("data"_a = py_data, "fp8_scale_inv"_a = scale_inv,
                            "fp8_dtype"_a = this->dtype, "data_transpose"_a = py_columnwise_data,
                            "quantizer"_a = this->quantizer);
  } else {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
    ret = Float8TensorClass("shape"_a = rowwise_torch_shape, "dtype"_a = GetATenDType(dtype),
                            "data"_a = py_data, "fp8_scale_inv"_a = scale_inv,
                            "fp8_dtype"_a = this->dtype, "data_transpose"_a = py_columnwise_data,
                            "quantizer"_a = this->quantizer);
  }
  TensorWrapper tensor(this->get_scaling_mode());
  if (rowwise_usage) {
    tensor.set_rowwise_data(data.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (create_transpose) {
    std::vector<size_t> transposed_shape;
    for (auto s : columnwise_torch_shape) {
      transposed_shape.emplace_back(static_cast<size_t>(s));
    }
    tensor.set_columnwise_data(columnwise_data.data_ptr(), this->dtype, transposed_shape);
    tensor.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
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
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  size_t numel = 1;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
    numel *= s;
  }

  TensorWrapper tensor(this->get_scaling_mode());
  at::TensorOptions opts;
  at::TensorOptions scale_opts;
  at::Tensor data_rowwise, data_colwise, scale_inv_rowwise, scale_inv_colwise;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  scale_opts = scale_opts.dtype(torch::kFloat32).device(torch::kCUDA);

  size_t k_dim = torch_shape.size() == 0 ? 1u : torch_shape.back();
  size_t m_dim = numel / k_dim;
  constexpr size_t kBlockLen = 128;

  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data_rowwise = std::move(*rowwise_data);
    } else {
      data_rowwise = at::empty(torch_shape, opts);
    }
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    if (block_scaling_dim == 2) {
      sinv0 = (m_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = roundup((k_dim + kBlockLen - 1) / kBlockLen, 4);
    } else if (block_scaling_dim == 1) {
      sinv0 = (k_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = roundup(m_dim, 4);
    } else {
      NVTE_CHECK(false,
                 "Unsupported block_scaling_dim in create_tensor rowwise."
                 "Expected 1 or 2. Got ",
                 block_scaling_dim);
    }
    scale_inv_rowwise =
        at::empty({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, scale_opts);
    tensor.set_rowwise_data(data_rowwise.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(scale_inv_rowwise.data_ptr(), DType::kFloat32,
                                 std::vector<size_t>{sinv0, sinv1});
  }

  if (columnwise_usage) {
    std::vector<int64_t> torch_columnwise_shape;
    std::vector<size_t> columnwise_shape;
    NVTE_CHECK(torch_shape.size() == shape.size(), "Shape expected to match torch shape. Shape ",
               columnwise_shape, " torch shape: ", torch_columnwise_shape);
    if (torch_shape.size() > 0) {
      torch_columnwise_shape.reserve(torch_shape.size());
      columnwise_shape.reserve(shape.size());
      torch_columnwise_shape.push_back(torch_shape[torch_shape.size() - 1]);
      columnwise_shape.push_back(shape[shape.size() - 1]);
      for (size_t i = 0; i < torch_shape.size() - 1; ++i) {
        torch_columnwise_shape.push_back(torch_shape[i]);
        columnwise_shape.push_back(shape[i]);
      }
    }
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    if (block_scaling_dim == 2) {
      sinv0 = (k_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = roundup((m_dim + kBlockLen - 1) / kBlockLen, 4);
    } else if (block_scaling_dim == 1) {
      sinv0 = (m_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = roundup(k_dim, 4);
    } else {
      NVTE_CHECK(false,
                 "Unsupported block_scaling_dim in create_tensor columnwise."
                 "Expected 1 or 2. Got ",
                 block_scaling_dim);
    }
    data_colwise = at::empty(torch_columnwise_shape, opts);
    scale_inv_colwise =
        at::empty({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, scale_opts);

    tensor.set_columnwise_data(data_colwise.data_ptr(), this->dtype, columnwise_shape);
    tensor.set_columnwise_scale_inv(scale_inv_colwise.data_ptr(), DType::kFloat32,
                                    std::vector<size_t>{sinv0, sinv1});
  }
  this->set_quantization_params(&tensor);

  py::object ret;
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
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> rowwise_data) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  size_t numel = 1;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
    numel *= s;
  }

  TensorWrapper tensor(NVTE_MXFP8_1D_SCALING);
  at::TensorOptions opts;
  at::Tensor rowwise_data1, columnwise_data, rowwise_scale_inv,
      columnwise_scale_inv;  // TODO(pgadzinski) - change
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  auto last_dim = static_cast<size_t>(torch_shape.back());

  NVTE_CHECK(last_dim % MXFP8_BLOCK_SIZE == 0 && (numel / last_dim) % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisble by ", MXFP8_BLOCK_SIZE,
             " (got shape=", torch_shape, ")");

  at::Tensor data;
  if (rowwise_usage) {
    if (rowwise_data.has_value()) {
      data = std::move(*rowwise_data);
    } else {
      data = at::empty(torch_shape, opts);
    }
    auto sinv0 = roundup(numel / last_dim, 128);
    auto sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4);
    rowwise_scale_inv = at::zeros({sinv0, sinv1}, opts);
    tensor.set_rowwise_data(data.data_ptr(), this->dtype, shape);
    tensor.set_rowwise_scale_inv(
        rowwise_scale_inv.data_ptr(), DType::kFloat8E8M0,
        std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }

  if (columnwise_usage) {
    auto sinv0 = roundup(numel / (last_dim * MXFP8_BLOCK_SIZE), 4);
    auto sinv1 = roundup(last_dim, 128);
    columnwise_data = at::empty(torch_shape, opts);
    columnwise_scale_inv = at::zeros({sinv0, sinv1}, opts);

    tensor.set_columnwise_data(columnwise_data.data_ptr(), this->dtype, shape);
    tensor.set_columnwise_scale_inv(
        columnwise_scale_inv.data_ptr(), DType::kFloat8E8M0,
        std::vector<size_t>{static_cast<size_t>(sinv0), static_cast<size_t>(sinv1)});
  }
  this->set_quantization_params(&tensor);

  py::object ret;
  if (internal) {
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorBasePythonClass));
    ret = MXFP8TensorClass("rowwise_data"_a = data, "columnwise_data"_a = columnwise_data,
                           "rowwise_scale_inv"_a = rowwise_scale_inv,
                           "columnwise_scale_inv"_a = columnwise_scale_inv,
                           "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
  } else {
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorPythonClass));
    ret = MXFP8TensorClass("shape"_a = torch_shape, "dtype"_a = GetATenDType(dtype),
                           "rowwise_data"_a = data, "columnwise_data"_a = columnwise_data,
                           "rowwise_scale_inv"_a = rowwise_scale_inv,
                           "columnwise_scale_inv"_a = columnwise_scale_inv,
                           "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
  }

  return {std::move(tensor), std::move(ret)};
}

}  // namespace transformer_engine::pytorch
