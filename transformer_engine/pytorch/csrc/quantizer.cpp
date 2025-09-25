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

}  // namespace

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
                             const std::optional<TensorWrapper>& noop_flag, const bool pdl_sync,
                             const bool pdl_trigger) {
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
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
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
                               const std::optional<TensorWrapper>& noop_flag, const bool pdl_sync,
                               const bool pdl_trigger) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  // Ignore pdl_sync/pdl_trigger for float8 delay scaling
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
    py::handle Float8TensorClass(reinterpret_cast<PyObject*>(Float8TensorBasePythonClass));
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

std::pair<TensorWrapper, py::object> Float8CurrentScalingQuantizer::create_hp_tensor_with_amax(
    const std::vector<size_t>& shape, DType dtype) {
  amax.zero_();
  auto [out_cpp, out_py] = NoneQuantizer(py::none()).create_tensor(shape, dtype);
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
                                             const std::optional<TensorWrapper>& noop_flag,
                                             const bool pdl_sync, const bool pdl_trigger) {
  // Ignore pdl_sync/pdl_trigger for float8 current scaling
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
        reinterpret_cast<PyObject*>(Float8BlockwiseQTensorBasePythonClass));
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
                                    const std::optional<TensorWrapper>& noop_flag,
                                    const bool pdl_sync, const bool pdl_trigger) {
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
  quant_config.set_pdl_sync(pdl_sync);
  quant_config.set_pdl_trigger(pdl_trigger);
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
             "MXFP8 requires tensor dims that are divisble by ", MXFP8_BLOCK_SIZE,
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
    py::handle MXFP8TensorClass(reinterpret_cast<PyObject*>(MXFP8TensorBasePythonClass));
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
                              const std::optional<TensorWrapper>& noop_flag, const bool pdl_sync,
                              const bool pdl_trigger) {
  if (input.numel() == 0) {
    return;
  }
  QuantizationConfigWrapper quant_config;
  if (noop_flag) {
    quant_config.set_noop_tensor(noop_flag->data());
  }
  quant_config.set_pdl_sync(pdl_sync);
  quant_config.set_pdl_trigger(pdl_trigger);
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
             "MXFP8 requires tensor dims that are divisble by ", MXFP8_BLOCK_SIZE,
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

}  // namespace transformer_engine::pytorch
