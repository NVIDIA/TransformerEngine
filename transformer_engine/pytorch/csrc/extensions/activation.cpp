/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

template <void (*act_func)(const NVTETensor, NVTETensor, cudaStream_t)>
py::object activation_helper(const at::Tensor& input, py::handle quantizer, int shape_divisor = 1) {
  init_extension();

  // Input tensor
  auto input_tensor = input.contiguous();
  const TensorWrapper& input_cpp = makeTransformerEngineTensor(input_tensor);

  // Construct output tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  const auto input_shape = input_cpp.shape();
  std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
  output_shape.back() /= shape_divisor;
  auto fake_dtype = GetTransformerEngineDType(input_tensor.scalar_type());
  auto [out_cpp, out_py] = quantizer_cpp->create_tensor(output_shape, fake_dtype);

  // Compute activation
  if (quantizer.is_none() || detail::IsFloat8Quantizers(quantizer.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer.ptr())) {
    // Compute activation directly
    NVTE_SCOPED_GIL_RELEASE(
        { act_func(input_cpp.data(), out_cpp.data(), at::cuda::getCurrentCUDAStream()); });
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // Compute activation in high-precision fused together with amax, then quantize.

    auto quantizer_cpp_cs = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    auto [temp_cpp, _] = quantizer_cpp_cs->create_hp_tensor_with_amax(output_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE(
        { act_func(input_cpp.data(), temp_cpp.data(), at::cuda::getCurrentCUDAStream()); });
    quantizer_cpp_cs->quantize_with_amax(temp_cpp, out_cpp);
  } else {
    // Compute activation in high-precision, then quantize

    auto [temp_cpp, _] = NoneQuantizer(py::none()).create_tensor(output_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE(
        { act_func(input_cpp.data(), temp_cpp.data(), at::cuda::getCurrentCUDAStream()); });
    quantizer_cpp->quantize(temp_cpp, out_cpp);
  }

  return out_py;
}

template <void (*dact_func)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t)>
py::object dactivation_helper(const at::Tensor& grad_output, const at::Tensor& input,
                              py::handle quantizer) {
  init_extension();

  // Grad output and input tensors
  auto grad_output_tensor = grad_output.contiguous();
  auto input_tensor = input.contiguous();
  const TensorWrapper& grad_output_cpp = makeTransformerEngineTensor(grad_output_tensor);
  const TensorWrapper& input_cpp = makeTransformerEngineTensor(input_tensor);

  // Construct grad input tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  const auto input_shape_te = input_cpp.shape();
  const std::vector<size_t> input_shape(input_shape_te.data,
                                        input_shape_te.data + input_shape_te.ndim);
  auto fake_dtype = GetTransformerEngineDType(input_tensor.scalar_type());
  auto [grad_input_cpp, grad_input_py] = quantizer_cpp->create_tensor(input_shape, fake_dtype);

  // Compute activation backward
  if (quantizer.is_none() || detail::IsFloat8Quantizers(quantizer.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer.ptr())) {
    // Compute activation backward directly
    NVTE_SCOPED_GIL_RELEASE({
      dact_func(grad_output_cpp.data(), input_cpp.data(), grad_input_cpp.data(),
                at::cuda::getCurrentCUDAStream());
    });
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // Compute activation backward in high-precision fused together with amax, then quantize.
    auto quantizer_cpp_cs = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    auto [temp_cpp, _] = quantizer_cpp_cs->create_hp_tensor_with_amax(input_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE({
      dact_func(grad_output_cpp.data(), input_cpp.data(), temp_cpp.data(),
                at::cuda::getCurrentCUDAStream());
    });
    quantizer_cpp_cs->quantize_with_amax(temp_cpp, grad_input_cpp);
  } else {
    // Compute activation backward in high-precision, then quantize
    auto [temp_cpp, _] = NoneQuantizer(py::none()).create_tensor(input_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE({
      dact_func(grad_output_cpp.data(), input_cpp.data(), temp_cpp.data(),
                at::cuda::getCurrentCUDAStream());
    });
    quantizer_cpp->quantize(temp_cpp, grad_input_cpp);
  }

  return grad_input_py;
}

/* GELU and variants*/
py::object gelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_gelu>(input, quantizer);
}

py::object dgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgelu>(grad, input, quantizer);
}

py::object geglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_geglu>(input, quantizer, 2);
}

py::object dgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgeglu>(grad, input, quantizer);
}

py::object qgelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgelu>(input, quantizer);
}

py::object dqgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgelu>(grad, input, quantizer);
}

py::object qgeglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgeglu>(input, quantizer, 2);
}

py::object dqgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgeglu>(grad, input, quantizer);
}

/* ReLU and variants*/
py::object relu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_relu>(input, quantizer);
}

py::object drelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_drelu>(grad, input, quantizer);
}

py::object reglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_reglu>(input, quantizer, 2);
}

py::object dreglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dreglu>(grad, input, quantizer);
}

py::object srelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_srelu>(input, quantizer);
}

py::object dsrelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsrelu>(grad, input, quantizer);
}

py::object sreglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_sreglu>(input, quantizer, 2);
}

py::object dsreglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsreglu>(grad, input, quantizer);
}

/* Silu and variants*/
py::object silu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_silu>(input, quantizer);
}

py::object dsilu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsilu>(grad, input, quantizer);
}

py::object swiglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_swiglu>(input, quantizer, 2);
}

py::object dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dswiglu>(grad, input, quantizer);
}
}  // namespace transformer_engine::pytorch
