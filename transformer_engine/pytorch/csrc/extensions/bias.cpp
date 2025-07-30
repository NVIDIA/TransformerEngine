/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include "transformer_engine/cast.h"
#include "transformer_engine/transformer_engine.h"

#include "common.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

std::vector<py::object> bgrad_quantize(const at::Tensor &grad_output, py::handle quantizer) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Check quantizer
  NVTE_CHECK(IsFloat8Quantizers(quantizer.ptr()) || IsMXFP8Quantizers(quantizer.ptr()),
             "Unsupported quantizer for fused bgrad-quantize kernel");
  auto quantizer_cpp = convert_quantizer(quantizer);

  // Grad output tensor
  auto grad_output_torch = grad_output.contiguous();
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());

  // Construct tensors
  auto [grad_input_nvte, grad_input_py] = quantizer_cpp->create_tensor(shape, grad_output_dtype);
  auto grad_bias_torch = allocateTorchTensor(shape.back(), grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Return immediately if tensors are empty
  if (product(shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Query workspace size and allocate workspace
  TensorWrapper workspace_nvte;
  at::Tensor workspace_torch;
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });
  if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
    workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
    workspace_nvte = makeTransformerEngineTensor(workspace_torch.data_ptr(),
                                                 workspace_nvte.shape(),
                                                 workspace_nvte.dtype());
  }

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

namespace {

std::vector<py::object> dbias_dact(void (*nvte_func)(const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                                                     cudaStream_t),
                                   at::Tensor grad_output_torch,
                                   at::Tensor act_input_torch,
                                   py::handle quantizer_py) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Check quantizer
  NVTE_CHECK(IsFloat8Quantizers(quantizer_py.ptr()) || IsMXFP8Quantizers(quantizer_py.ptr()),
             "Unsupported quantizer for fused bgrad-quantize kernel");
  auto quantizer_cpp = convert_quantizer(quantizer_py);

  // Grad output and activation input tensors
  grad_output_torch = grad_output_torch.contiguous();
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());
  act_input_torch = act_input_torch.contiguous();
  const TensorWrapper &act_input_nvte = makeTransformerEngineTensor(act_input_torch);

  // Construct tensors
  auto [grad_input_nvte, grad_input_py] = quantizer_cpp->create_tensor(shape, grad_output_dtype);
  auto grad_bias_torch = allocateTorchTensor(shape.back(), grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Return immediately if tensors are empty
  if (product(shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Query workspace size and allocate workspace
  TensorWrapper workspace_nvte;
  at::Tensor workspace_torch;
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    nvte_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
              grad_bias_nvte.data(), workspace_nvte.data(), stream);
  });
  if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
    workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
    workspace_nvte = makeTransformerEngineTensor(workspace_torch.data_ptr(),
                                               workspace_nvte.shape(),
                                               workspace_nvte.dtype());
  }

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
              grad_bias_nvte.data(), workspace_nvte.data(), stream);
  });

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

}  // namespace

std::vector<py::object> dbias_dgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dbias_dact(nvte_quantize_dbias_dgelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsilu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dbias_dact(nvte_quantize_dbias_dsilu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_drelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dbias_dact(nvte_quantize_dbias_drelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dqgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer) {
  return dbias_dact(nvte_quantize_dbias_dqgelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsrelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer) {
  return dbias_dact(nvte_quantize_dbias_dsrelu, grad_output, act_input, quantizer);
}

}  // namespace pytorch
}  // namespace transformer_engine
