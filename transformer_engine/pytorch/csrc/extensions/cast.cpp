/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/cast.h"

#include "common.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

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
  nvte_quantize_noop(te_input.data(), te_output.data(), te_noop.data(),
                     at::cuda::getCurrentCUDAStream());

  return out;
}

py::object dequantize(const py::handle& input, transformer_engine::DType otype) {
  init_extension();

  const auto none = py::none();

  const auto& input_tensor = makeTransformerEngineTensor(input, none);

  NoneQuantizer q(none);

  const auto& shape = convertShape(input_tensor.shape());

  auto [out_tensor, out] = q.create_tensor(shape, otype);

  nvte_dequantize(input_tensor.data(), out_tensor.data(), at::cuda::getCurrentCUDAStream());

  return out;
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
  func(grad_tensor.data(), act_input_tensor.data(), dact_tensor.data(), dbias_tensor.data(),
       workspace.data(), at::cuda::getCurrentCUDAStream());
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  func(grad_tensor.data(), act_input_tensor.data(), dact_tensor.data(), dbias_tensor.data(),
       workspace.data(), at::cuda::getCurrentCUDAStream());

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

}  // namespace transformer_engine::pytorch
