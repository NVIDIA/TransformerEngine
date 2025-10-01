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

namespace {

/*! @brief Transposed tensor shape
 *
 * The tensor is interpreted as a 2D matrix by flattening all but the
 * last dimension, and then transposed.
 */
template <typename T = size_t, typename S = T>
std::vector<T> make_transpose_shape(const std::vector<S>& shape) {
  std::vector<T> ret;
  if (shape.size() > 0) {
    ret.push_back(shape.back());
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      ret.push_back(shape[i]);
    }
  }
  return ret;
}

/*! @brief Convert shape for FP4 data by dividing the last dimension by 2 */
template <typename T = size_t>
std::vector<T> convert_shape_for_fp4(const std::vector<T>& shape) {
  std::vector<T> ret;
  for (size_t i = 0; i < shape.size() - 1; ++i) {
    ret.push_back(shape[i]);
  }
  ret.push_back(shape.back() / 2);
  return ret;
}

}  // namespace

constexpr size_t NVFP4_BLOCK_SIZE = 16;
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

std::pair<TensorWrapper, py::object> NoneQuantizer::create_tensor(const std::vector<size_t>& shape,
                                                                  DType dtype) const {
  const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  const auto opts = at::TensorOptions().dtype(GetATenDType(dtype)).device(torch::kCUDA);
  return create_tensor(shape, dtype, at::empty(shape_int64, opts));
}

std::pair<TensorWrapper, py::object> NoneQuantizer::create_tensor(const std::vector<size_t>& shape,
                                                                  DType dtype,
                                                                  at::Tensor data) const {
  TensorWrapper out_cpp;
  out_cpp.set_rowwise_data(data.data_ptr(), dtype, shape);
  set_quantization_params(&out_cpp);
  return {std::move(out_cpp), py::cast(data)};
}

std::pair<TensorWrapper, py::object> NoneQuantizer::convert_and_update_tensor(
    py::object tensor) const {
  auto tensor_pyt = tensor.cast<at::Tensor>();
  TensorWrapper out_cpp;
  out_cpp.set_rowwise_data(tensor_pyt.data_ptr(),
                           GetTransformerEngineDType(tensor_pyt.scalar_type()),
                           getTensorShape(tensor_pyt));
  set_quantization_params(&out_cpp);
  return {std::move(out_cpp), std::move(tensor)};
}

void NoneQuantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                             const std::optional<TensorWrapper>& noop_flag) {
  NVTE_ERROR("NoneQuantizer does not support quantization");
}

void Float8Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  tensor->set_scale(scale.data_ptr(), GetTransformerEngineDType(scale.scalar_type()),
                    getTensorShape(scale));
  at::TensorOptions opts = opts.dtype(torch::kFloat32).device(torch::kCUDA);
  tensor->set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
}

std::pair<TensorWrapper, py::object> Float8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype) const {
  const auto opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  at::Tensor scale_inv = at::empty(std::vector<int64_t>{1}, opts);
  return create_tensor(shape, dtype, std::nullopt, std::nullopt, std::move(scale_inv));
}

std::pair<TensorWrapper, py::object> Float8Quantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype, std::optional<at::Tensor> data,
    std::optional<at::Tensor> transpose, std::optional<at::Tensor> scale_inv) const {
  using namespace pybind11::literals;

  // Initialize data tensor
  const bool with_data = rowwise_usage || nvte_is_non_tn_fp8_gemm_supported();
  if (with_data && !data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    data = at::empty(shape_int64, opts);
  } else if (!with_data && data) {
    data.reset();
  }
  py::object data_py = with_data ? py::cast(*data) : py::none();

  // Initialize transpose tensor
  const bool with_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  if (with_transpose && !transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    transpose = at::empty(transpose_shape, opts);
  } else if (!with_transpose && transpose) {
    transpose.reset();
  }
  py::object transpose_py = with_transpose ? py::cast(*transpose) : py::none();

  // Initialize scale-inverse tensor
  if (!scale_inv) {
    scale_inv = at::reciprocal(scale);
  }

  // Construct Python FP8 tensor
  py::object out_py;
  if (internal) {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorStoragePythonClass));
    out_py = Float8TensorClass("data"_a = data_py, "fp8_scale_inv"_a = *scale_inv,
                               "fp8_dtype"_a = this->dtype, "data_transpose"_a = transpose_py,
                               "quantizer"_a = this->quantizer);
  } else {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    out_py = Float8TensorClass("shape"_a = shape_int64, "dtype"_a = GetATenDType(dtype),
                               "data"_a = data_py, "fp8_scale_inv"_a = *scale_inv,
                               "fp8_dtype"_a = this->dtype, "data_transpose"_a = transpose_py,
                               "quantizer"_a = this->quantizer);
  }

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp(this->get_scaling_mode());
  if (with_data) {
    out_cpp.set_rowwise_data(data->data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv->data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (with_transpose) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose->data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv->data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, py::object> Float8Quantizer::convert_and_update_tensor(
    py::object tensor) const {
  NVTE_CHECK(detail::IsFloat8Tensor(tensor.ptr()), "Float8Quantizer must output to Float8Tensor.");

  // Expected buffers
  const bool need_data = rowwise_usage || nvte_is_non_tn_fp8_gemm_supported();
  const bool need_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  NVTE_CHECK(need_data || need_transpose, "Invalid usages for Float8Quantizer.");

  // Extract buffers from Python tensor
  auto data_py = tensor.attr("_data");
  auto transpose_py = tensor.attr("_transpose");
  const bool has_data = !data_py.is_none();
  const bool has_transpose = !transpose_py.is_none();
  NVTE_CHECK(has_data || has_transpose, "Float8Tensor has no data.");
  std::optional<at::Tensor> data_tensor, transpose_tensor;
  if (has_data) {
    data_tensor = data_py.cast<at::Tensor>();
  }
  if (has_transpose) {
    transpose_tensor = transpose_py.cast<at::Tensor>();
  }
  at::Tensor scale_inv_tensor = tensor.attr("_scale_inv").cast<at::Tensor>();

  // Tensor dimensions
  std::vector<size_t> shape;
  if (has_transpose) {
    const auto transpose_shape = getTensorShape(*transpose_tensor);
    if (transpose_shape.size() > 0) {
      for (size_t i = 1; i < transpose_shape.size(); ++i) {
        shape.push_back(transpose_shape[i]);
      }
      shape.push_back(transpose_shape.front());
    }
    if (has_data) {
      auto expected_shape = getTensorShape(*data_tensor);
      NVTE_CHECK(shape == expected_shape, "FP8 data (shape=", expected_shape,
                 ") and transpose (shape=", transpose_shape, ") do not match");
    }
  } else {  // Already checked has_data == true
    shape = getTensorShape(*data_tensor);
  }

  // Coerce data tensor
  if (has_data && !need_data) {
    data_tensor.reset();
    data_py = py::none();
    tensor.attr("_data") = data_py;
  } else if (!has_data && need_data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    data_tensor = at::empty(shape_int64, opts);
    data_py = py::cast(data_tensor);
    tensor.attr("_data") = data_py;
  }

  // Coerce transpose tensor
  if (has_transpose && !need_transpose) {
    transpose_tensor.reset();
    transpose_py = py::none();
    tensor.attr("_transpose") = transpose_py;
  } else if (!has_transpose && need_transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    transpose_tensor = at::empty(transpose_shape, opts);
    transpose_py = py::cast(transpose_tensor);
    tensor.attr("_transpose") = transpose_py;
  }
  tensor.attr("_transpose_invalid") = !need_transpose;

  // Coerce other attrs
  tensor.attr("_fp8_dtype") = dtype;

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp;
  if (data_tensor) {
    out_cpp.set_rowwise_data(data_tensor->data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                  std::vector<size_t>{1});
  }
  if (transpose_tensor) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose_tensor->data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void Float8Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                               const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, at::cuda::getCurrentCUDAStream());
  });
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
}

std::pair<TensorWrapper, py::object> Float8CurrentScalingQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype) const {
  using namespace pybind11::literals;

  // Initialize data tensor
  at::Tensor data_tensor;
  const bool with_data = rowwise_usage || nvte_is_non_tn_fp8_gemm_supported();
  if (with_data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    data_tensor = at::empty(shape_int64, opts);
  }

  // Initialize transpose tensor
  at::Tensor transpose_tensor;
  const bool with_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  if (with_transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    transpose_tensor = at::empty(transpose_shape, opts);
  }

  // Initialize scale-inverse tensor
  at::Tensor scale_inv_tensor;
  {
    const std::vector<int64_t> scale_inv_shape = {1};
    const auto opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    scale_inv_tensor = at::empty(scale_inv_shape, opts);
  }

  // Construct Python FP8 tensor
  py::object out_py;
  py::object data_py = with_data ? py::cast(data_tensor) : py::none();
  py::object transpose_py = with_transpose ? py::cast(transpose_tensor) : py::none();
  if (internal) {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorStoragePythonClass));
    out_py = Float8TensorClass("data"_a = data_py, "fp8_scale_inv"_a = scale_inv_tensor,
                               "fp8_dtype"_a = this->dtype, "data_transpose"_a = transpose_py,
                               "quantizer"_a = this->quantizer);
  } else {
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorPythonClass));
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    out_py = Float8TensorClass("shape"_a = shape_int64, "dtype"_a = GetATenDType(dtype),
                               "data"_a = data_py, "fp8_scale_inv"_a = scale_inv_tensor,
                               "fp8_dtype"_a = this->dtype, "data_transpose"_a = transpose_py,
                               "quantizer"_a = this->quantizer);
  }

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp(this->get_scaling_mode());
  if (with_data) {
    out_cpp.set_rowwise_data(data_tensor.data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                  std::vector<size_t>{1});
  }
  if (with_transpose) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose_tensor.data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, py::object>
Float8CurrentScalingQuantizer::create_unquantized_tensor_with_amax(const std::vector<size_t>& shape,
                                                                   DType dtype,
                                                                   std::optional<at::Tensor> data) {
  amax.zero_();
  auto out = data.has_value() ? NoneQuantizer(py::none()).create_tensor(shape, dtype, data.value())
                              : NoneQuantizer(py::none()).create_tensor(shape, dtype);
  TensorWrapper out_cpp = std::move(out.first);
  py::object out_py = std::move(out.second);
  out_cpp.set_amax(amax.data_ptr(), GetTransformerEngineDType(amax.scalar_type()),
                   getTensorShape(amax));
  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, py::object> Float8CurrentScalingQuantizer::convert_and_update_tensor(
    py::object tensor) const {
  NVTE_CHECK(detail::IsFloat8Tensor(tensor.ptr()),
             "Float8CurrentScalingQuantizer must output to Float8Tensor.");

  // Expected buffers
  const bool need_data = rowwise_usage || nvte_is_non_tn_fp8_gemm_supported();
  const bool need_transpose = columnwise_usage && !nvte_is_non_tn_fp8_gemm_supported();
  NVTE_CHECK(need_data || need_transpose, "Invalid quantizer usages.");

  // Extract buffers from Python tensor
  auto data_py = tensor.attr("_data");
  auto transpose_py = tensor.attr("_transpose");
  const bool has_data = !data_py.is_none();
  const bool has_transpose = !transpose_py.is_none();
  NVTE_CHECK(has_data || has_transpose, "Tensor has no data.");
  std::optional<at::Tensor> data_tensor, transpose_tensor;
  if (has_data) {
    data_tensor = data_py.cast<at::Tensor>();
  }
  if (has_transpose) {
    transpose_tensor = transpose_py.cast<at::Tensor>();
  }
  at::Tensor scale_inv_tensor = tensor.attr("_scale_inv").cast<at::Tensor>();

  // Tensor dimensions
  std::vector<size_t> shape;
  if (has_transpose) {
    const auto transpose_shape = getTensorShape(*transpose_tensor);
    if (transpose_shape.size() > 0) {
      for (size_t i = 1; i < transpose_shape.size(); ++i) {
        shape.push_back(transpose_shape[i]);
      }
      shape.push_back(transpose_shape.front());
    }
    if (has_data) {
      auto expected_shape = getTensorShape(*data_tensor);
      NVTE_CHECK(shape == expected_shape, "FP8 data (shape=", expected_shape,
                 ") and transpose (shape=", transpose_shape, ") do not match");
    }
  } else {  // Already checked has_data == true
    shape = getTensorShape(*data_tensor);
  }

  // Coerce data tensor in Python tensor
  if (has_data && !need_data) {
    data_tensor.reset();
    data_py = py::none();
    tensor.attr("_data") = data_py;
  } else if (!has_data && need_data) {
    const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    data_tensor = at::empty(shape_int64, opts);
    data_py = py::cast(data_tensor);
    tensor.attr("_data") = data_py;
  }

  // Coerce transpose tensor
  if (has_transpose && !need_transpose) {
    transpose_tensor.reset();
    transpose_py = py::none();
    tensor.attr("_transpose") = transpose_py;
  } else if (!has_transpose && need_transpose) {
    const auto transpose_shape = make_transpose_shape<int64_t>(shape);
    const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    transpose_tensor = at::empty(transpose_shape, opts);
    transpose_py = py::cast(transpose_tensor);
    tensor.attr("_transpose") = transpose_py;
  }
  tensor.attr("_transpose_invalid") = !need_transpose;

  // Coerce other attrs
  tensor.attr("_fp8_dtype") = dtype;

  // Construct C++ FP8 tensor
  TensorWrapper out_cpp;
  if (data_tensor) {
    out_cpp.set_rowwise_data(data_tensor->data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                  std::vector<size_t>{1});
  }
  if (transpose_tensor) {
    const auto transpose_shape = make_transpose_shape(shape);
    out_cpp.set_columnwise_data(transpose_tensor->data_ptr(), this->dtype, transpose_shape);
    out_cpp.set_columnwise_scale_inv(scale_inv_tensor.data_ptr(), DType::kFloat32,
                                     std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void Float8CurrentScalingQuantizer::quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                                                  const std::optional<TensorWrapper>& noop_flag,
                                                  bool compute_amax) {
  auto stream = at::cuda::getCurrentCUDAStream();

  // Nothing to be done if input is empty
  if (input.numel() == 0) {
    return;
  }

  // Quantization configs
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(amax_epsilon);

  // Compute amax
  if (compute_amax) {
    NVTE_SCOPED_GIL_RELEASE(
        { nvte_compute_amax_with_config(input.data(), out.data(), quant_config, stream); });
  }

  // Perform amax reduction if needed
  if (with_amax_reduction) {
    // allreduce amax tensor
    c10d::AllreduceOptions opts;
    opts.reduceOp = c10d::ReduceOp::MAX;
    std::vector<at::Tensor> tensors = {amax};
    NVTE_SCOPED_GIL_RELEASE({ amax_reduction_group->allreduce(tensors, opts)->wait(); });
  }

  // Compute scaling factor
  NVTE_SCOPED_GIL_RELEASE({ nvte_compute_scale_from_amax(out.data(), quant_config, stream); });

  // Cast to FP8
  out.set_amax(nullptr, DType::kFloat32, out.defaultShape);  // Avoid atomic amax updates
  NVTE_SCOPED_GIL_RELEASE({ nvte_quantize_v2(input.data(), out.data(), quant_config, stream); });
}

void Float8CurrentScalingQuantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                                             const std::optional<TensorWrapper>& noop_flag) {
  this->quantize_impl(input, out, noop_flag, true);
}

void Float8CurrentScalingQuantizer::quantize_with_amax(
    TensorWrapper& input, TensorWrapper& out, const std::optional<TensorWrapper>& noop_flag) {
  NVTE_CHECK(input.get_amax().data_ptr == amax.data_ptr(),
             "Input does not use the appropriate amax tensor");
  input.set_amax(nullptr, DType::kFloat32, input.defaultShape);
  this->quantize_impl(input, out, noop_flag, false);
}

Float8BlockQuantizer::Float8BlockQuantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
  this->block_scaling_dim = quantizer.attr("block_scaling_dim").cast<int>();
  this->force_pow_2_scales = quantizer.attr("force_pow_2_scales").cast<bool>();
  this->amax_epsilon = quantizer.attr("amax_epsilon").cast<float>();
  NVTE_CHECK(this->block_scaling_dim == 1 || this->block_scaling_dim == 2,
             "Unsupported block scaling dim.");
  this->all_gather_usage = quantizer.attr("all_gather_usage").cast<bool>();
}

void Float8BlockQuantizer::set_quantization_params(TensorWrapper* tensor) const {}

std::pair<TensorWrapper, py::object> Float8BlockQuantizer::create_tensor(
    const std::vector<size_t>& shape, DType dtype) const {
  using namespace pybind11::literals;
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }

  TensorWrapper tensor(this->get_scaling_mode());
  at::TensorOptions opts;
  at::TensorOptions scale_opts;
  at::Tensor data_rowwise, data_colwise, scale_inv_rowwise, scale_inv_colwise;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  scale_opts = scale_opts.dtype(torch::kFloat32).device(torch::kCUDA);

  Float8BlockScaleTensorFormat data_format =
      (all_gather_usage ? Float8BlockScaleTensorFormat::COMPACT
                        : Float8BlockScaleTensorFormat::GEMM_READY);

  if (rowwise_usage) {
    data_rowwise = at::empty(torch_shape, opts);
    auto scale_shape = get_scale_shape(shape, false);
    size_t sinv0 = scale_shape[0];
    size_t sinv1 = scale_shape[1];
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
      if (!all_gather_usage) {
        torch_columnwise_shape.reserve(torch_shape.size());
        columnwise_shape.reserve(shape.size());
        torch_columnwise_shape.push_back(torch_shape[torch_shape.size() - 1]);
        columnwise_shape.push_back(shape[shape.size() - 1]);
        for (size_t i = 0; i < torch_shape.size() - 1; ++i) {
          torch_columnwise_shape.push_back(torch_shape[i]);
          columnwise_shape.push_back(shape[i]);
        }
      } else {
        // assert we are doing 1D scaling
        NVTE_CHECK(block_scaling_dim == 1,
                   "Compact columnwise format is not supported for 128x128 2D block scaling.");
        torch_columnwise_shape = torch_shape;
        columnwise_shape = shape;
      }
    }
    auto scale_shape = get_scale_shape(shape, true);
    size_t sinv0 = scale_shape[0];
    size_t sinv1 = scale_shape[1];
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
        reinterpret_cast<PyObject*>(Float8BlockwiseQTensorStoragePythonClass));
    ret = Float8BlockwiseQTensorClass(
        "rowwise_data"_a = data_rowwise, "columnwise_data"_a = data_colwise,
        "rowwise_scale_inv"_a = scale_inv_rowwise, "columnwise_scale_inv"_a = scale_inv_colwise,
        "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer,
        "is_2D_scaled"_a = (block_scaling_dim == 2), "data_format"_a = data_format);
  } else {
    py::handle Float8BlockwiseQTensorClass(
        reinterpret_cast<PyObject*>(Float8BlockwiseQTensorPythonClass));
    ret = Float8BlockwiseQTensorClass(
        "shape"_a = torch_shape, "dtype"_a = GetATenDType(dtype), "rowwise_data"_a = data_rowwise,
        "columnwise_data"_a = data_colwise, "rowwise_scale_inv"_a = scale_inv_rowwise,
        "columnwise_scale_inv"_a = scale_inv_colwise, "fp8_dtype"_a = this->dtype,
        "quantizer"_a = this->quantizer, "is_2D_scaled"_a = (block_scaling_dim == 2),
        "data_format"_a = data_format);
  }

  return {std::move(tensor), std::move(ret)};
}

std::pair<TensorWrapper, py::object> Float8BlockQuantizer::convert_and_update_tensor(
    py::object tensor) const {
  const DType dtype = tensor.attr("_fp8_dtype").cast<DType>();
  bool is_2D_scaled = tensor.attr("_is_2D_scaled").cast<bool>();

  // Extract buffers from Python tensor
  auto get_tensor = [&tensor](const char* name) -> std::optional<at::Tensor> {
    auto attr_py = tensor.attr(name);
    if (attr_py.is_none()) {
      return std::nullopt;
    }
    return attr_py.cast<at::Tensor>();
  };
  auto rowwise_data = get_tensor("_rowwise_data");
  auto rowwise_scale_inv = get_tensor("_rowwise_scale_inv");
  auto columnwise_data = get_tensor("_columnwise_data");
  auto columnwise_scale_inv = get_tensor("_columnwise_scale_inv");
  NVTE_CHECK(rowwise_data || columnwise_data, "FP8BlockwiseTensor has no data.");

  // Tensor options and dimensions
  at::TensorOptions opts;
  at::TensorOptions scale_opts;
  opts = opts.dtype(torch::kUInt8).device(torch::kCUDA);
  scale_opts = scale_opts.dtype(torch::kFloat32).device(torch::kCUDA);

  auto get_columnwise_shape = [&columnwise_data](bool all_gather_usage) -> std::vector<size_t> {
    if (!columnwise_data) {
      return std::vector<size_t>();
    }
    if (all_gather_usage) {
      return getTensorShape(*columnwise_data);
    }
    std::vector<size_t> shape = getTensorShape(*columnwise_data);
    std::vector<size_t> shape_transposed(shape.size());
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      shape_transposed[i] = shape[i + 1];
    }
    if (shape.size() > 0) {
      shape_transposed[shape.size() - 1] = shape[0];
    }
    return shape_transposed;
  };
  std::vector<size_t> shape;
  if (rowwise_data) {
    shape = getTensorShape(*rowwise_data);
    if (columnwise_data) {
      auto expected_shape = get_columnwise_shape(all_gather_usage);
      NVTE_CHECK(shape == expected_shape, "BlockwiseFP8 row-wise data (shape=", shape,
                 ") and column-wise data (shape=", expected_shape, ") do not match");
    }
  } else {
    shape = get_columnwise_shape(all_gather_usage);
  }
  std::vector<int64_t> torch_shape;
  for (auto s : shape) {
    torch_shape.emplace_back(static_cast<int64_t>(s));
  }

  // Coerce row-wise data
  if (rowwise_usage) {
    if (!rowwise_data) {
      rowwise_data = at::empty(torch_shape, opts);
      tensor.attr("_rowwise_data") = *rowwise_data;
    }
    if (!rowwise_scale_inv) {
      auto scale_shape = get_scale_shape(shape, false);
      size_t sinv0 = scale_shape[0];
      size_t sinv1 = scale_shape[1];
      rowwise_scale_inv =
          at::empty({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, scale_opts);
      tensor.attr("_rowwise_scale_inv") = *rowwise_scale_inv;
    }
  } else {  // rowwise_usage == false
    if (rowwise_data) {
      rowwise_data.reset();
      tensor.attr("_rowwise_data") = py::none();
    }
    if (rowwise_scale_inv) {
      rowwise_scale_inv.reset();
      tensor.attr("_rowwise_scale_inv") = py::none();
    }
  }

  // Coerce column-wise data
  if (columnwise_usage) {
    std::vector<size_t> columnwise_shape;
    std::vector<int64_t> torch_columnwise_shape;
    if (torch_shape.size() > 0) {
      if (!all_gather_usage) {
        torch_columnwise_shape.reserve(torch_shape.size());
        columnwise_shape.reserve(shape.size());
        torch_columnwise_shape.push_back(torch_shape[torch_shape.size() - 1]);
        columnwise_shape.push_back(shape[shape.size() - 1]);
        for (size_t i = 0; i < torch_shape.size() - 1; ++i) {
          torch_columnwise_shape.push_back(torch_shape[i]);
          columnwise_shape.push_back(shape[i]);
        }
      } else {
        // assert we are doing 1D scaling
        NVTE_CHECK(block_scaling_dim == 1,
                   "Compact columnwise format is not supported for 128x128 2D block scaling.");
        torch_columnwise_shape = torch_shape;
        columnwise_shape = shape;
      }
    }
    if (!columnwise_data) {
      columnwise_data = at::empty(torch_columnwise_shape, opts);
      tensor.attr("_columnwise_data") = *columnwise_data;
    }
    if (!columnwise_scale_inv) {
      auto scale_shape = get_scale_shape(shape, true);
      size_t sinv0 = scale_shape[0];
      size_t sinv1 = scale_shape[1];
      columnwise_scale_inv =
          at::empty({static_cast<int64_t>(sinv0), static_cast<int64_t>(sinv1)}, scale_opts);
      tensor.attr("_columnwise_scale_inv") = *columnwise_scale_inv;
    }
  } else {  // columnwise_usage == false
    if (columnwise_data) {
      columnwise_data.reset();
      tensor.attr("_columnwise_data") = py::none();
    }
    if (columnwise_scale_inv) {
      columnwise_scale_inv.reset();
      tensor.attr("_columnwise_scale_inv") = py::none();
    }
  }

  auto ret = TensorWrapper(is_2D_scaled ? NVTE_BLOCK_SCALING_2D : NVTE_BLOCK_SCALING_1D);

  if (rowwise_usage) {
    const at::Tensor& data_rowwise = tensor.attr("_rowwise_data").cast<at::Tensor>();
    const at::Tensor& scale_inv_rowwise = tensor.attr("_rowwise_scale_inv").cast<at::Tensor>();
    void* scale_inv_rowwise_dptr = scale_inv_rowwise.data_ptr();
    const auto& rowwise_shape = getTensorShape(data_rowwise);
    ret.set_rowwise_data(data_rowwise.data_ptr(), dtype, rowwise_shape);
    const auto scale_inv_rowwise_shape = getTensorShape(scale_inv_rowwise);
    ret.set_rowwise_scale_inv(scale_inv_rowwise_dptr, DType::kFloat32, scale_inv_rowwise_shape);
  }
  if (columnwise_usage) {
    const at::Tensor& data_colwise = tensor.attr("_columnwise_data").cast<at::Tensor>();
    const at::Tensor& scale_inv_colwise = tensor.attr("_columnwise_scale_inv").cast<at::Tensor>();
    void* scale_inv_colwise_dptr = scale_inv_colwise.data_ptr();
    const auto& shape = getTensorShape(data_colwise);
    ret.set_columnwise_data(data_colwise.data_ptr(), dtype, shape);
    const auto scale_inv_colwise_shape = getTensorShape(scale_inv_colwise);
    ret.set_columnwise_scale_inv(scale_inv_colwise_dptr, DType::kFloat32, scale_inv_colwise_shape);
  }
  set_quantization_params(&ret);
  return {std::move(ret), std::move(tensor)};
}

void Float8BlockQuantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                                    const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_force_pow_2_scales(force_pow_2_scales);
  quant_config.set_amax_epsilon(amax_epsilon);
  if (all_gather_usage) {
    quant_config.set_float8_block_scale_tensor_format(Float8BlockScaleTensorFormat::COMPACT);
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, at::cuda::getCurrentCUDAStream());
  });
}

std::vector<size_t> Float8BlockQuantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                          bool columnwise) const {
  size_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }

  size_t k_dim = shape.size() == 0 ? 1u : shape.back();
  size_t m_dim = numel / k_dim;
  constexpr size_t kBlockLen = 128;

  Float8BlockScaleTensorFormat data_format =
      (all_gather_usage ? Float8BlockScaleTensorFormat::COMPACT
                        : Float8BlockScaleTensorFormat::GEMM_READY);

  std::vector<size_t> scale_shape;

  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    if (block_scaling_dim == 2) {
      // 2D scaling is always GEMM_READY for now
      NVTE_CHECK(data_format == Float8BlockScaleTensorFormat::GEMM_READY,
                 "2D scaling is always GEMM_READY for now.");
      sinv0 = (m_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = roundup((k_dim + kBlockLen - 1) / kBlockLen, 4);
    } else if (block_scaling_dim == 1) {
      // 1D scaling can be GEMM_READY or COMPACT
      bool rowwise_compact = data_format == Float8BlockScaleTensorFormat::COMPACT;
      // default rowwise scaling factor shape already transpose the scaling factor so it's GEMM_READY
      sinv0 = (k_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = rowwise_compact ? m_dim : roundup(m_dim, 4);
      // if the rowwise format is compact, the scaling factor is not be transposed
      if (rowwise_compact) {
        std::swap(sinv0, sinv1);
      }
    } else {
      NVTE_CHECK(false,
                 "Unsupported block_scaling_dim in create_tensor rowwise."
                 "Expected 1 or 2. Got ",
                 block_scaling_dim);
    }
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = 0;
    size_t sinv1 = 0;
    if (block_scaling_dim == 2) {
      // 2D scaling is always GEMM_READY for now
      NVTE_CHECK(data_format == Float8BlockScaleTensorFormat::GEMM_READY,
                 "2D scaling is always GEMM_READY for now.");
      sinv0 = (k_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = roundup((m_dim + kBlockLen - 1) / kBlockLen, 4);
    } else if (block_scaling_dim == 1) {
      // 1D scaling can be GEMM_READY or COMPACT
      bool columnwise_compact = data_format == Float8BlockScaleTensorFormat::COMPACT;
      sinv0 = (m_dim + kBlockLen - 1) / kBlockLen;
      sinv1 = columnwise_compact ? k_dim : roundup(k_dim, 4);
      // GEMM READY case: scaling factor is [sinv0, sinv1], already transposed here for CuBLAS
      // for COMPACT case, since we apply 128x1 scaling here without transposing columnwise data, scaling factor is also [sinv0, sinv1]
      // so no need to swap sinv0 and sinv1 here
    } else {
      NVTE_CHECK(false,
                 "Unsupported block_scaling_dim in create_tensor columnwise."
                 "Expected 1 or 2. Got ",
                 block_scaling_dim);
    }
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}

MXFP8Quantizer::MXFP8Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
}

void MXFP8Quantizer::set_quantization_params(TensorWrapper* tensor) const {}

std::pair<TensorWrapper, py::object> MXFP8Quantizer::create_tensor(const std::vector<size_t>& shape,
                                                                   DType dtype) const {
  using namespace pybind11::literals;

  // Tensor dimensions
  const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  size_t flat_first_dim = 1;
  if (shape.size() > 0) {
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      flat_first_dim *= shape[i];
    }
  }
  const size_t flat_last_dim = shape.size() > 0 ? shape.back() : 1;
  NVTE_CHECK(flat_first_dim % MXFP8_BLOCK_SIZE == 0 && flat_last_dim % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisible by ", MXFP8_BLOCK_SIZE,
             " (got shape=", shape, ")");
  const auto rowwise_scale_inv_shape = get_scale_shape(shape, false);
  const auto columnwise_scale_inv_shape = get_scale_shape(shape, true);

  // Allocate tensors
  at::Tensor rowwise_data_tensor, rowwise_scale_inv_tensor;
  at::Tensor columnwise_data_tensor, columnwise_scale_inv_tensor;
  const auto uint8_tensor_opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  if (rowwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(rowwise_scale_inv_shape.begin(),
                                                     rowwise_scale_inv_shape.end());
    rowwise_data_tensor = at::empty(shape_int64, uint8_tensor_opts);
    rowwise_scale_inv_tensor = at::empty(scale_inv_shape_int64, uint8_tensor_opts);
  }
  if (columnwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(columnwise_scale_inv_shape.begin(),
                                                     columnwise_scale_inv_shape.end());
    columnwise_data_tensor = at::empty(shape_int64, uint8_tensor_opts);
    columnwise_scale_inv_tensor = at::empty(scale_inv_shape_int64, uint8_tensor_opts);
  }

  // Convert tensors to Python
  auto py_cast = [](at::Tensor& tensor, bool need_cast) -> py::object {
    return need_cast ? py::cast(tensor) : py::none();
  };
  auto rowwise_data_py = py_cast(rowwise_data_tensor, rowwise_usage);
  auto rowwise_scale_inv_py = py_cast(rowwise_scale_inv_tensor, rowwise_usage);
  auto columnwise_data_py = py_cast(columnwise_data_tensor, columnwise_usage);
  auto columnwise_scale_inv_py = py_cast(columnwise_scale_inv_tensor, columnwise_usage);

  // Construct Python MXFP8 tensor
  py::object out_py;
  if (internal) {
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorStoragePythonClass));
    out_py = MXFP8TensorClass("rowwise_data"_a = rowwise_data_py,
                              "columnwise_data"_a = columnwise_data_py,
                              "rowwise_scale_inv"_a = rowwise_scale_inv_py,
                              "columnwise_scale_inv"_a = columnwise_scale_inv_py,
                              "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
  } else {
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorPythonClass));
    out_py = MXFP8TensorClass("shape"_a = shape_int64, "dtype"_a = GetATenDType(dtype),
                              "rowwise_data"_a = rowwise_data_py,
                              "columnwise_data"_a = columnwise_data_py,
                              "rowwise_scale_inv"_a = rowwise_scale_inv_py,
                              "columnwise_scale_inv"_a = columnwise_scale_inv_py,
                              "fp8_dtype"_a = this->dtype, "quantizer"_a = this->quantizer);
  }

  // Construct C++ MXFP8 tensor
  TensorWrapper out_cpp(NVTE_MXFP8_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data_tensor.data_ptr(), this->dtype, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv_tensor.data_ptr(), DType::kFloat8E8M0,
                                  rowwise_scale_inv_shape);
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data_tensor.data_ptr(), this->dtype, shape);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv_tensor.data_ptr(), DType::kFloat8E8M0,
                                     columnwise_scale_inv_shape);
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, py::object> MXFP8Quantizer::convert_and_update_tensor(
    py::object tensor) const {
  NVTE_CHECK(detail::IsMXFP8Tensor(tensor.ptr()), "MXFP8Quantizer must output to MXFP8Tensor.");

  // Extract buffers from Python tensor
  auto get_tensor = [&tensor](const char* name) -> std::optional<at::Tensor> {
    auto attr_py = tensor.attr(name);
    if (attr_py.is_none()) {
      return std::nullopt;
    }
    return attr_py.cast<at::Tensor>();
  };
  auto rowwise_data = get_tensor("_rowwise_data");
  auto rowwise_scale_inv = get_tensor("_rowwise_scale_inv");
  auto columnwise_data = get_tensor("_columnwise_data");
  auto columnwise_scale_inv = get_tensor("_columnwise_scale_inv");
  NVTE_CHECK(rowwise_data || columnwise_data, "MXFP8Tensor has no data.");

  // Tensor dimensions
  std::vector<size_t> shape;
  if (columnwise_data) {
    shape = getTensorShape(*columnwise_data);
    if (rowwise_data) {
      auto expected_shape = getTensorShape(*rowwise_data);
      NVTE_CHECK(shape == expected_shape, "MXFP8 row-wise data (shape=", expected_shape,
                 ") and column-wise data (shape=", shape, ") do not match");
    }
  } else {  // Already checked columnwise_data_tensor == true
    shape = getTensorShape(*rowwise_data);
  }

  // Coerce row-wise data
  if (rowwise_usage) {
    if (!rowwise_data) {
      const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      rowwise_data = at::empty(shape_int64, opts);
      tensor.attr("_rowwise_data") = *rowwise_data;
    }
    if (!rowwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, false);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      rowwise_scale_inv = at::empty(scale_inv_shape_int64, opts);
      tensor.attr("_rowwise_scale_inv") = *rowwise_scale_inv;
    }
  } else {  // rowwise_usage == false
    if (rowwise_data) {
      rowwise_data.reset();
      tensor.attr("_rowwise_data") = py::none();
    }
    if (rowwise_scale_inv) {
      rowwise_scale_inv.reset();
      tensor.attr("_rowwise_scale_inv") = py::none();
    }
  }

  // Coerce column-wise data
  if (columnwise_usage) {
    if (!columnwise_data) {
      const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      columnwise_data = at::empty(shape_int64, opts);
      tensor.attr("_columnwise_data") = *columnwise_data;
    }
    if (!columnwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, true);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      columnwise_scale_inv = at::empty(scale_inv_shape_int64, opts);
      tensor.attr("_columnwise_scale_inv") = *columnwise_scale_inv;
    }
  } else {  // columnwise_usage == false
    if (columnwise_data) {
      columnwise_data.reset();
      tensor.attr("_columnwise_data") = py::none();
    }
    if (columnwise_scale_inv) {
      columnwise_scale_inv.reset();
      tensor.attr("_columnwise_scale_inv") = py::none();
    }
  }

  // Coerce other attrs
  tensor.attr("_fp8_dtype") = dtype;

  // Construct C++ MXFP8 tensor
  TensorWrapper out_cpp(NVTE_MXFP8_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), dtype, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                  getTensorShape(*rowwise_scale_inv));
  }
  if (columnwise_usage) {
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), dtype, shape);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E8M0,
                                     getTensorShape(*columnwise_scale_inv));
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void MXFP8Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop_flag) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(input.data(), out.data(), quant_config, at::cuda::getCurrentCUDAStream());
  });
}

std::vector<size_t> MXFP8Quantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                    bool columnwise) const {
  size_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }

  auto last_dim = shape.back();

  NVTE_CHECK(last_dim % MXFP8_BLOCK_SIZE == 0 && (numel / last_dim) % MXFP8_BLOCK_SIZE == 0,
             "MXFP8 requires tensor dims that are divisible by ", MXFP8_BLOCK_SIZE,
             " (got shape=", shape, ")");

  std::vector<size_t> scale_shape;

  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = roundup(numel / last_dim, 128);
    size_t sinv1 = roundup(last_dim / MXFP8_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = roundup(numel / (last_dim * MXFP8_BLOCK_SIZE), 4);
    size_t sinv1 = roundup(last_dim, 128);
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}

NVFP4Quantizer::NVFP4Quantizer(const py::handle& quantizer) : Quantizer(quantizer) {
  this->dtype = quantizer.attr("dtype").cast<DType>();
  this->with_rht = quantizer.attr("with_rht").cast<bool>();
  this->with_post_rht_amax = quantizer.attr("with_post_rht_amax").cast<bool>();
  this->with_2d_quantization = quantizer.attr("with_2d_quantization").cast<bool>();
  this->stochastic_rounding = quantizer.attr("stochastic_rounding").cast<bool>();

  // Get amax reduction group if needed for NVFP4 AG
  const bool with_amax_reduction = quantizer.attr("with_amax_reduction").cast<bool>();
  c10::intrusive_ptr<dist_group_type> amax_reduction_group;
  if (with_amax_reduction) {
    auto group = quantizer.attr("_canonicalized_amax_reduction_group")();
    NVTE_CHECK(!group.is_none(), "NVFP4Quantizer could not canonicalize amax reduction group");
    amax_reduction_group = group.cast<c10::intrusive_ptr<dist_group_type>>();
  }
  this->with_amax_reduction = with_amax_reduction;
  this->amax_reduction_group = amax_reduction_group;

  this->rht_matrix_random_sign_mask_t = quantizer.attr("rht_matrix_random_sign_mask_t").cast<int>();
  this->rht_matrix = quantizer.attr("rht_matrix").cast<at::Tensor>();
}

void NVFP4Quantizer::set_quantization_params(TensorWrapper* tensor) const {
  // set dtype for rowwise and columnwise data in tensor wrapper
  auto rowwise_data = tensor->get_rowwise_data();
  rowwise_data.dtype = static_cast<NVTEDType>(this->dtype);

  auto columnwise_data = tensor->get_columnwise_data();
  columnwise_data.dtype = static_cast<NVTEDType>(this->dtype);

  tensor->set_rowwise_data(rowwise_data.data_ptr, static_cast<DType>(rowwise_data.dtype),
                           rowwise_data.shape);
  tensor->set_columnwise_data(columnwise_data.data_ptr, static_cast<DType>(columnwise_data.dtype),
                              columnwise_data.shape);
}

std::pair<TensorWrapper, py::object> NVFP4Quantizer::create_tensor(const std::vector<size_t>& shape,
                                                                   DType dtype) const {
  using namespace pybind11::literals;

  // Tensor dimensions
  const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
  size_t flat_first_dim = 1;
  if (shape.size() > 0) {
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      flat_first_dim *= shape[i];
    }
  }
  const size_t flat_last_dim = shape.size() > 0 ? shape.back() : 1;
  NVTE_CHECK(flat_first_dim % NVFP4_BLOCK_SIZE == 0, "First dim for NVFP4 must be divisible by ",
             NVFP4_BLOCK_SIZE, " (got shape=", shape, ")");
  NVTE_CHECK(flat_last_dim % NVFP4_BLOCK_SIZE == 0,
             "NVFP4 requires tensor dims that are divisible by ", NVFP4_BLOCK_SIZE,
             " (got shape=", shape, ")");
  const auto rowwise_scale_inv_shape = get_scale_shape(shape, false);
  const auto columnwise_scale_inv_shape = get_scale_shape(shape, true);

  // Allocate tensors
  at::Tensor rowwise_data_tensor, rowwise_scale_inv_tensor, amax_rowwise;
  at::Tensor columnwise_data_tensor, columnwise_scale_inv_tensor, amax_columnwise;
  const auto bit8_tensor_opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  const auto bit32_tensor_opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  if (rowwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(rowwise_scale_inv_shape.begin(),
                                                     rowwise_scale_inv_shape.end());
    rowwise_data_tensor = at::empty(convert_shape_for_fp4(shape_int64), bit8_tensor_opts);
    rowwise_scale_inv_tensor = at::empty(scale_inv_shape_int64, bit8_tensor_opts);
    amax_rowwise = at::empty({1}, bit32_tensor_opts);
  }
  if (columnwise_usage) {
    const std::vector<int64_t> scale_inv_shape_int64(columnwise_scale_inv_shape.begin(),
                                                     columnwise_scale_inv_shape.end());
    // enforce 2D shape to avoid [S, B, H] shape and B and be 1
    // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
    std::vector<int64_t> shape_int64_2d = {static_cast<int64_t>(flat_first_dim),
                                           static_cast<int64_t>(flat_last_dim)};
    const auto transpose_shape_int64 = make_transpose_shape<int64_t>(shape_int64_2d);
    columnwise_data_tensor =
        at::empty(convert_shape_for_fp4(transpose_shape_int64), bit8_tensor_opts);
    columnwise_scale_inv_tensor = at::empty(scale_inv_shape_int64, bit8_tensor_opts);
    amax_columnwise = at::empty({1}, bit32_tensor_opts);
  }

  // Convert tensors to Python
  auto py_cast = [](at::Tensor& tensor, bool need_cast) -> py::object {
    return need_cast ? py::cast(tensor) : py::none();
  };
  auto rowwise_data_py = py_cast(rowwise_data_tensor, rowwise_usage);
  auto rowwise_scale_inv_py = py_cast(rowwise_scale_inv_tensor, rowwise_usage);
  auto columnwise_data_py = py_cast(columnwise_data_tensor, columnwise_usage);
  auto columnwise_scale_inv_py = py_cast(columnwise_scale_inv_tensor, columnwise_usage);
  auto amax_rowwise_py = py_cast(amax_rowwise, rowwise_usage);
  auto amax_columnwise_py = py_cast(amax_columnwise, columnwise_usage);

  // Construct Python NVFP4 tensor
  py::object out_py;
  if (internal) {
    py::handle NVFP4TensorClass(reinterpret_cast<PyObject*>(NVFP4TensorStoragePythonClass));
    out_py = NVFP4TensorClass(
        "rowwise_data"_a = rowwise_data_py, "columnwise_data"_a = columnwise_data_py,
        "rowwise_scale_inv"_a = rowwise_scale_inv_py,
        "columnwise_scale_inv"_a = columnwise_scale_inv_py, "amax_rowwise"_a = amax_rowwise_py,
        "amax_columnwise"_a = amax_columnwise_py, "fp4_dtype"_a = this->dtype,
        "quantizer"_a = this->quantizer);
  } else {
    py::handle NVFP4TensorClass(reinterpret_cast<PyObject*>(NVFP4TensorPythonClass));
    out_py = NVFP4TensorClass(
        "shape"_a = shape_int64, "dtype"_a = GetATenDType(dtype),
        "rowwise_data"_a = rowwise_data_py, "columnwise_data"_a = columnwise_data_py,
        "rowwise_scale_inv"_a = rowwise_scale_inv_py,
        "columnwise_scale_inv"_a = columnwise_scale_inv_py, "amax_rowwise"_a = amax_rowwise_py,
        "amax_columnwise"_a = amax_columnwise_py, "fp4_dtype"_a = this->dtype,
        "quantizer"_a = this->quantizer);
  }

  // Construct C++ tensor
  TensorWrapper out_cpp(NVTE_NVFP4_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data_tensor.data_ptr(), DType::kFloat4E2M1, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv_tensor.data_ptr(), DType::kFloat8E4M3,
                                  rowwise_scale_inv_shape);
    out_cpp.set_amax(amax_rowwise.data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (columnwise_usage) {
    // enforce 2D shape to avoid [S, B, H] shape and B and be 1
    // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
    std::vector<size_t> shape_2d = {flat_first_dim, flat_last_dim};
    auto col_data_shape_fp4 = make_transpose_shape<size_t>(shape_2d);
    out_cpp.set_columnwise_data(columnwise_data_tensor.data_ptr(), DType::kFloat4E2M1,
                                col_data_shape_fp4);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv_tensor.data_ptr(), DType::kFloat8E4M3,
                                     columnwise_scale_inv_shape);
    out_cpp.set_columnwise_amax(amax_columnwise.data_ptr(), DType::kFloat32,
                                std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, py::object> NVFP4Quantizer::create_unquantized_tensor_with_amax(
    TensorWrapper& quantized_tensor, DType dtype) {
  // Construct tensor
  auto shape = convertShape(quantized_tensor.shape());
  auto [out_cpp, out_py] = NoneQuantizer(py::none()).create_tensor(shape, dtype);

  // Register amax pointer from quantized tensor
  void* amax_ptr = quantized_tensor.amax();
  if (amax_ptr == nullptr) {
    amax_ptr = quantized_tensor.get_columnwise_amax().data_ptr;
  }
  NVTE_CHECK(amax_ptr != nullptr, "Could not extract amax pointer from NVFP4 tensor.");
  out_cpp.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});

  // Zero out amax
  NVTE_CHECK_CUDA(cudaMemsetAsync(amax_ptr, 0, sizeof(float), at::cuda::getCurrentCUDAStream()));

  return {std::move(out_cpp), std::move(out_py)};
}

std::pair<TensorWrapper, py::object> NVFP4Quantizer::convert_and_update_tensor(
    py::object tensor) const {
  NVTE_CHECK(detail::IsNVFP4Tensor(tensor.ptr()), "NVFP4Quantizer must output to IsNVFP4Tensor.");

  // Extract buffers from Python tensor
  auto get_tensor = [&tensor](const char* name) -> std::optional<at::Tensor> {
    auto attr_py = tensor.attr(name);
    if (attr_py.is_none()) {
      return std::nullopt;
    }
    return attr_py.cast<at::Tensor>();
  };
  auto rowwise_data = get_tensor("_rowwise_data");
  auto rowwise_scale_inv = get_tensor("_rowwise_scale_inv");
  auto columnwise_data = get_tensor("_columnwise_data");
  auto columnwise_scale_inv = get_tensor("_columnwise_scale_inv");
  auto amax_rowwise = get_tensor("_amax_rowwise");
  auto amax_columnwise = get_tensor("_amax_columnwise");
  NVTE_CHECK(rowwise_data || columnwise_data, "NVFP4Tensor has no data.");

  // Tensor dimensions, shape means original shape
  std::vector<size_t> shape;
  if (columnwise_data) {
    shape = convert_shape_back_from_fp4(getTensorShape(*columnwise_data), true);
    if (rowwise_data) {
      auto expected_shape = convert_shape_back_from_fp4(getTensorShape(*rowwise_data), false);
      NVTE_CHECK(shape == expected_shape, "NVFP4 row-wise data (shape=", expected_shape,
                 ") and column-wise data (shape=", shape, ") do not match");
    }
  } else {  // Already checked columnwise_data_tensor == true
    shape = convert_shape_back_from_fp4(getTensorShape(*rowwise_data), false);
  }

  size_t flat_first_dim = 1;
  if (shape.size() > 0) {
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      flat_first_dim *= shape[i];
    }
  }
  const size_t flat_last_dim = shape.size() > 0 ? shape.back() : 1;

  // Coerce row-wise data
  if (rowwise_usage) {
    if (!rowwise_data) {
      const std::vector<int64_t> shape_int64(shape.begin(), shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      rowwise_data = at::empty(convert_shape_for_fp4(shape_int64), opts);
      tensor.attr("_rowwise_data") = *rowwise_data;
    }
    if (!rowwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, false);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      rowwise_scale_inv = at::empty(scale_inv_shape_int64, opts);
      tensor.attr("_rowwise_scale_inv") = *rowwise_scale_inv;
    }
    if (!amax_rowwise) {
      const auto opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
      amax_rowwise = at::empty({1}, opts);
      tensor.attr("_amax_rowwise") = *amax_rowwise;
    }
  } else {  // rowwise_usage == false
    if (rowwise_data) {
      rowwise_data.reset();
      tensor.attr("_rowwise_data") = py::none();
    }
    if (rowwise_scale_inv) {
      rowwise_scale_inv.reset();
      tensor.attr("_rowwise_scale_inv") = py::none();
    }
    if (amax_rowwise) {
      amax_rowwise.reset();
      tensor.attr("_amax_rowwise") = py::none();
    }
  }

  // Coerce column-wise data
  if (columnwise_usage) {
    if (!columnwise_data) {
      // enforce 2D shape to avoid [S, B, H] shape and B and be 1
      // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
      std::vector<int64_t> shape_int64_2d = {static_cast<int64_t>(flat_first_dim),
                                             static_cast<int64_t>(flat_last_dim)};
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      const auto transpose_shape_int64 = make_transpose_shape<int64_t>(shape_int64_2d);
      columnwise_data = at::empty(convert_shape_for_fp4(transpose_shape_int64), opts);
      tensor.attr("_columnwise_data") = *columnwise_data;
    }
    if (!columnwise_scale_inv) {
      const auto scale_inv_shape = get_scale_shape(shape, true);
      const std::vector<int64_t> scale_inv_shape_int64(scale_inv_shape.begin(),
                                                       scale_inv_shape.end());
      const auto opts = at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
      columnwise_scale_inv = at::empty(scale_inv_shape_int64, opts);
      tensor.attr("_columnwise_scale_inv") = *columnwise_scale_inv;
    }
    if (!amax_columnwise) {
      const auto opts = at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
      amax_columnwise = at::zeros({1}, opts);
      tensor.attr("_amax_columnwise") = *amax_columnwise;
    }
  } else {  // columnwise_usage == false
    if (columnwise_data) {
      columnwise_data.reset();
      tensor.attr("_columnwise_data") = py::none();
    }
    if (columnwise_scale_inv) {
      columnwise_scale_inv.reset();
      tensor.attr("_columnwise_scale_inv") = py::none();
    }
    if (amax_columnwise) {
      amax_columnwise.reset();
      tensor.attr("_amax_columnwise") = py::none();
    }
  }

  // Construct C++ tensor
  TensorWrapper out_cpp(NVTE_NVFP4_1D_SCALING);
  if (rowwise_usage) {
    out_cpp.set_rowwise_data(rowwise_data->data_ptr(), DType::kFloat4E2M1, shape);
    out_cpp.set_rowwise_scale_inv(rowwise_scale_inv->data_ptr(), DType::kFloat8E4M3,
                                  getTensorShape(*rowwise_scale_inv));
    out_cpp.set_amax(amax_rowwise->data_ptr(), DType::kFloat32, std::vector<size_t>{1});
  }
  if (columnwise_usage) {
    // enforce 2D shape to avoid [S, B, H] shape and B and be 1
    // and the transposed shape is [H, S, B], so divide last dim by 2 gives zero
    std::vector<size_t> shape_2d = {flat_first_dim, flat_last_dim};
    auto col_data_shape_fp4 = make_transpose_shape<size_t>(shape_2d);
    out_cpp.set_columnwise_data(columnwise_data->data_ptr(), DType::kFloat4E2M1,
                                col_data_shape_fp4);
    out_cpp.set_columnwise_scale_inv(columnwise_scale_inv->data_ptr(), DType::kFloat8E4M3,
                                     getTensorShape(*columnwise_scale_inv));
    out_cpp.set_columnwise_amax(amax_columnwise->data_ptr(), DType::kFloat32,
                                std::vector<size_t>{1});
  }
  this->set_quantization_params(&out_cpp);

  return {std::move(out_cpp), std::move(tensor)};
}

void NVFP4Quantizer::quantize_impl(const TensorWrapper& input, TensorWrapper& out,
                                   const std::optional<TensorWrapper>& noop_flag,
                                   bool compute_amax) {
  // Nothing to be done if input is empty
  if (input.numel() == 0) {
    return;
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_nvfp4_2d_quantization(this->with_2d_quantization);
  quant_config.set_stochastic_rounding(this->stochastic_rounding);

  // We only need RHT for columnwise usage.
  // flat first dim and last dim for multi dimensional input
  size_t rows = 1;
  for (size_t i = 0; i < input.ndim() - 1; ++i) {
    rows *= input.size(i);
  }
  size_t cols = input.size(input.ndim() - 1);

  TensorWrapper te_rng_state;
  if (this->stochastic_rounding) {
    const size_t rng_elts_per_thread = 1024;  // Wild guess, probably can be tightened
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    at::PhiloxCudaState philox_args = init_philox_state(gen, rng_elts_per_thread);
    auto opts = at::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, opts);
    philox_unpack(philox_args, static_cast<int64_t*>(rng_state.data_ptr()));
    te_rng_state = makeTransformerEngineTensor(rng_state);
    quant_config.set_rng_state(te_rng_state.data());
  }

  // Restriction for the RHT cast fusion kernel.
  bool eligible_for_rht_cast_fusion =
      input.dtype() == DType::kBFloat16 && rows % 64 == 0 && cols % 128 == 0;

  // Compute amax.
  if (this->with_rht) {
    if (input.dtype() != DType::kBFloat16) {
      NVTE_CHECK(false, "RHT is only supported for bfloat16 input");
    }
    if (this->with_post_rht_amax) {
      // We need:
      // 1. Rowwise amax = amax for input
      // 2. Columnwise amax = amax for RHT(input.t)
      NVTE_SCOPED_GIL_RELEASE({
        nvte_hadamard_transform_amax(input.data(), out.data(), 0,
                                     this->rht_matrix_random_sign_mask_t, stream);
      });
    } else {
      // raise error since it's not supported yet
      NVTE_CHECK(false, "Pre-RHT amax is not supported yet");
    }
  } else {  // Without RHT
    if (compute_amax) {
      // Amax pointers
      auto rowwise_amax_ptr = out.get_amax().data_ptr;
      auto columnwise_amax_ptr = out.get_columnwise_amax().data_ptr;
      void* amax_ptr = rowwise_amax_ptr != nullptr ? rowwise_amax_ptr : columnwise_amax_ptr;
      NVTE_CHECK(amax_ptr != nullptr, "Could not find amax pointer");

      // Compute amax of input tensor
      out.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
      NVTE_SCOPED_GIL_RELEASE(
          { nvte_compute_amax_with_config(input.data(), out.data(), quant_config, stream); });
      out.set_amax(rowwise_amax_ptr, DType::kFloat32, std::vector<size_t>{1});

      // Make sure row-wise and column-wise amaxes match
      if (rowwise_amax_ptr != amax_ptr && rowwise_amax_ptr != nullptr) {
        NVTE_CHECK_CUDA(cudaMemcpyAsync(rowwise_amax_ptr, amax_ptr, sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
      }
      if (columnwise_amax_ptr != amax_ptr && columnwise_amax_ptr != nullptr) {
        NVTE_CHECK_CUDA(cudaMemcpyAsync(columnwise_amax_ptr, amax_ptr, sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream));
      }
    }
  }

  // amax reduction
  if (this->with_amax_reduction) {
    std::vector<at::Tensor> amax_tensors;
    // push amax tensors inside if they need to be reduced
    auto make_amax_tensor = [](void* data_ptr) {
      return at::from_blob(
          data_ptr, std::vector<int64_t>{1},
          [](void*) {},  // deleter doing nothing since it doesn't own the data
          at::device(at::kCUDA).dtype(torch::kFloat32));
    };
    if (rowwise_usage) {
      amax_tensors.push_back(make_amax_tensor(out.get_amax().data_ptr));
    }
    if (columnwise_usage) {
      amax_tensors.push_back(make_amax_tensor(out.get_columnwise_amax().data_ptr));
    }
    c10d::AllreduceCoalescedOptions opts;
    opts.reduceOp = c10d::ReduceOp::MAX;
    NVTE_SCOPED_GIL_RELEASE(
        { this->amax_reduction_group->allreduce_coalesced(amax_tensors, opts)->wait(); });
  }

  if (this->with_rht) {
    if (rowwise_usage) {
      // For rowwise usage, we need to quantize the input directly, but we need to avoid quantizing columnwise
      TensorWrapper out_identity(out.scaling_mode());
      auto out_identity_data = out.get_rowwise_data();
      auto out_identity_scale_inv = out.get_rowwise_scale_inv();
      auto out_identity_amax = out.get_amax();
      out_identity.set_rowwise_data(out_identity_data.data_ptr,
                                    static_cast<DType>(out_identity_data.dtype),
                                    out_identity_data.shape);
      out_identity.set_rowwise_scale_inv(out_identity_scale_inv.data_ptr,
                                         static_cast<DType>(out_identity_scale_inv.dtype),
                                         out_identity_scale_inv.shape);
      out_identity.set_amax(out_identity_amax.data_ptr, static_cast<DType>(out_identity_amax.dtype),
                            out_identity_amax.shape);

      NVTE_SCOPED_GIL_RELEASE(
          { nvte_quantize_v2(input.data(), out_identity.data(), quant_config, stream); });
    }

    if (columnwise_usage) {
      // Get the output columnwise data, scale_inv, and amax
      auto out_columnwise_data = out.get_columnwise_data();
      auto out_columnwise_scale_inv = out.get_columnwise_scale_inv();
      // NOTE: should already be populated.
      auto out_columnwise_amax = out.get_columnwise_amax();

      // Create a wrapper for the columnwise output, as the rowwise output.
      // The reason is due to the input `rht_output_t` is already in the transposed layout.
      // Thus, we only need a rowwise quantization to generate the columnwise output.
      TensorWrapper out_transpose(out.scaling_mode());
      // Note: since we are faking columnwise tensor into rowwise, the flat first dim check will fail
      // need to convert the shape to 2D here
      auto colwise_data_shape = out_columnwise_data.shape;
      std::vector<size_t> colwise_data_shape_2d;
      // shape could be [512, 32, 64], that's actually 512, 32, 128 because 2 FP4 take 1 byte
      // the 2D shape should be [512, 32*128], but columnwise data shape expect last dim to be halved again
      // so the multiple 2 get cancelled out
      colwise_data_shape_2d.push_back(colwise_data_shape.data[0]);
      size_t last_dim = 1;
      for (size_t i = 1; i < colwise_data_shape.ndim; ++i) {
        last_dim *= colwise_data_shape.data[i];
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

      if (!eligible_for_rht_cast_fusion) {
        // Invoking fallback RHT kernel.

        // If using RHT, then amax will be computed in the RHT step
        // If not using RHT, then amax will be computed based on input x
        at::Tensor rht_output_t;  // The RHT(x_t) output, in columnwise layout
        // This wrapper is going to be passed as input to the quantization kernel.
        TensorWrapper rht_output_t_cpp;  // Wrapper to contain the RHT(x) and RHT(x_t) outputs
        rht_output_t =
            allocateTorchTensor(static_cast<int>(cols), static_cast<int>(rows), input.dtype());
        // NOTE (frsun): This is non-intuitive, we are writing the
        // result of transposed RHT to the output of rowwise.
        rht_output_t_cpp.set_rowwise_data(rht_output_t.data_ptr(), input.dtype(),
                                          std::vector<size_t>{cols, rows});

        NVTE_SCOPED_GIL_RELEASE({
          // Perform the RHT(input.t), and write to rht_output_cpp.columnwise.
          nvte_hadamard_transform(input.data(), rht_output_t_cpp.data(), 0,
                                  this->rht_matrix_random_sign_mask_t, stream);
        });

        // Quantize kernel will treat everything as rowwise input/output, which is
        // intended.
        NVTE_SCOPED_GIL_RELEASE({
          nvte_quantize_v2(rht_output_t_cpp.data(), out_transpose.data(), quant_config, stream);
        });
      } else {
        // RHT cast fusion kernel.
        NVTE_CHECK(this->rht_matrix.defined() && this->rht_matrix.numel() > 0,
                   "RHT matrix is not set");
        auto rht_matrix_nvte = makeTransformerEngineTensor(this->rht_matrix);
        NVTE_SCOPED_GIL_RELEASE({
          nvte_hadamard_transform_cast_fusion_columnwise(
              input.data(), out_transpose.data(), rht_matrix_nvte.data(), quant_config, stream);
        });
      }
    }
  } else {
    NVTE_SCOPED_GIL_RELEASE({ nvte_quantize_v2(input.data(), out.data(), quant_config, stream); });
  }
}

void NVFP4Quantizer::quantize(const TensorWrapper& input, TensorWrapper& out,
                              const std::optional<TensorWrapper>& noop_flag) {
  this->quantize_impl(input, out, noop_flag, true);
}

void NVFP4Quantizer::quantize_with_amax(TensorWrapper& input, TensorWrapper& out) {
  // Update output tensor amaxes with input tensor amax
  auto input_amax_ptr = input.amax();
  auto output_rowwise_amax_ptr = out.get_amax().data_ptr;
  auto output_columnwise_amax_ptr = out.get_columnwise_amax().data_ptr;
  NVTE_CHECK(input_amax_ptr != nullptr ||
                 (output_rowwise_amax_ptr == nullptr && output_columnwise_amax_ptr == nullptr),
             "Input tensor does not have pre-computed amax");
  if (input_amax_ptr != output_rowwise_amax_ptr && input_amax_ptr != nullptr &&
      output_rowwise_amax_ptr != nullptr) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(output_rowwise_amax_ptr, input_amax_ptr, sizeof(float),
                                    cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream()));
  }
  if (input_amax_ptr != output_columnwise_amax_ptr && input_amax_ptr != nullptr &&
      output_columnwise_amax_ptr != nullptr) {
    NVTE_CHECK_CUDA(cudaMemcpyAsync(output_columnwise_amax_ptr, input_amax_ptr, sizeof(float),
                                    cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream()));
  }
  input.set_amax(nullptr, DType::kFloat32, input.defaultShape);

  // Perform quantization
  this->quantize_impl(input, out, std::nullopt, false);
}

std::vector<size_t> NVFP4Quantizer::get_scale_shape(const std::vector<size_t>& shape,
                                                    bool columnwise) const {
  size_t numel = 1;
  for (auto s : shape) {
    numel *= s;
  }

  auto last_dim = shape.back();
  auto flat_first_dim = numel / last_dim;

  NVTE_CHECK(last_dim % NVFP4_BLOCK_SIZE == 0, "Last dim for NVFP4 must be divisible by ",
             NVFP4_BLOCK_SIZE, " (got dim=", last_dim, ")");
  NVTE_CHECK(flat_first_dim % NVFP4_BLOCK_SIZE == 0,
             "NVFP4 requires tensor dims that are divisible by ", NVFP4_BLOCK_SIZE,
             " (got shape=", shape, ")");

  std::vector<size_t> scale_shape;

  bool rowwise_usage = !columnwise;

  if (rowwise_usage) {
    // rowwise scaling factor shape
    size_t sinv0 = roundup(flat_first_dim, 128);
    size_t sinv1 = roundup(last_dim / NVFP4_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  } else {
    // columnwise scaling factor shape
    size_t sinv0 = roundup(last_dim, 128);
    size_t sinv1 = roundup(flat_first_dim / NVFP4_BLOCK_SIZE, 4);
    scale_shape = {sinv0, sinv1};
  }
  return scale_shape;
}

}  // namespace transformer_engine::pytorch
