/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"
#include "transformer_engine/cast.h"

namespace transformer_engine::pytorch {

std::vector<py::object> bgrad_quantize(const at::Tensor& input, py::handle py_quantizer) {
  auto quantizer = convert_quantizer(py_quantizer);

  auto input_tensor = makeTransformerEngineTensor(input);

  auto dbias = allocateTorchTensor(input.size(-1), input_tensor.dtype());

  std::vector<size_t> output_shape;
  for (auto s : input.sizes()) {
    output_shape.emplace_back(static_cast<size_t>(s));
  }
  auto [out_tensor, out] = quantizer->create_tensor(output_shape, input_tensor.dtype());

  // Return immediately if tensors are empty
  if (product(output_shape) == 0) {
    return {py::cast(dbias.zero_()), out};
  }

  auto dbias_tensor = makeTransformerEngineTensor(dbias);
  // Query workspace size and allocate workspace
  transformer_engine::TensorWrapper workspace;
  nvte_quantize_dbias(input_tensor.data(), out_tensor.data(), dbias_tensor.data(), workspace.data(),
                      at::cuda::getCurrentCUDAStream());

  void* workspace_data_ptr = nullptr;
  if (workspace.shape().ndim > 0) {
    auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
    workspace_data_ptr = workspace_data.data_ptr();
  }
  workspace = makeTransformerEngineTensor(workspace_data_ptr, workspace.shape(), workspace.dtype());

  // Launch kernel
  nvte_quantize_dbias(input_tensor.data(), out_tensor.data(), dbias_tensor.data(), workspace.data(),
                      at::cuda::getCurrentCUDAStream());

  return {py::cast(dbias), out};
}

}  // namespace transformer_engine::pytorch
