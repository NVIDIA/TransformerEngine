/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

template <void (*act_func)(const NVTETensor, NVTETensor, cudaStream_t)>
py::object activation_helper(const at::Tensor& input, py::handle quantizer, int shape_divisor = 1) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);
  auto input_tensor = input.contiguous();

  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);
  const auto& te_input_shape = te_input.shape();
  std::vector<size_t> input_shape(te_input_shape.data, te_input_shape.data + te_input_shape.ndim);
  input_shape[input_shape.size() - 1] /= shape_divisor;
  auto fake_tensor_type = input.scalar_type();

  auto [te_output, out] =
      my_quantizer->create_tensor(input_shape, GetTransformerEngineDType(fake_tensor_type));

  act_func(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

template <void (*act_func)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t)>
py::object dactivation_helper(const at::Tensor& grad, const at::Tensor& input,
                              py::handle quantizer) {
  init_extension();
  auto my_quantizer = convert_quantizer(quantizer);
  auto input_tensor = input.contiguous();
  auto grad_tensor = grad.contiguous();

  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);
  const TensorWrapper& te_grad = makeTransformerEngineTensor(grad_tensor);
  const auto& te_input_shape = te_input.shape();
  std::vector<size_t> input_shape(te_input_shape.data, te_input_shape.data + te_input_shape.ndim);
  auto fake_tensor_type = input.scalar_type();

  auto [te_output, out] =
      my_quantizer->create_tensor(input_shape, GetTransformerEngineDType(fake_tensor_type));

  act_func(te_grad.data(), te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());

  return out;
}

py::object gelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_gelu>(input, quantizer);
}

py::object dgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgelu>(grad, input, quantizer);
}

py::object relu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_relu>(input, quantizer);
}

py::object drelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_drelu>(grad, input, quantizer);
}

py::object geglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_geglu>(input, quantizer, 2);
}

py::object qgeglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgeglu>(input, quantizer, 2);
}

py::object dgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgeglu>(grad, input, quantizer);
}

py::object dqgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgeglu>(grad, input, quantizer);
}

py::object reglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_reglu>(input, quantizer, 2);
}

py::object dreglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dreglu>(grad, input, quantizer);
}

py::object swiglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_swiglu>(input, quantizer, 2);
}

py::object dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dswiglu>(grad, input, quantizer);
}

py::object qgelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgelu>(input, quantizer);
}

py::object dqgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgelu>(grad, input, quantizer);
}

py::object srelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_srelu>(input, quantizer);
}

py::object dsrelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsrelu>(grad, input, quantizer);
}

}  // namespace transformer_engine::pytorch
