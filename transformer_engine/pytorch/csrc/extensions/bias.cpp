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

namespace transformer_engine::pytorch {

std::vector<py::object> bgrad_quantize(const at::Tensor &grad_output, py::handle quantizer) {
  using namespace transformer_engine::pytorch::detail;

  // Grad output tensor
  auto grad_output_torch = grad_output.contiguous();
  const TensorWrapper &grad_output_te = makeTransformerEngineTensor(grad_output_torch);

  // Construct grad input and grad bias tensors
  auto quantizer_cpp = convert_quantizer(quantizer);
  const auto shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());
  auto [grad_input_te, grad_input_py] = quantizer_cpp->create_tensor(shape, grad_output_dtype);
  auto grad_bias_torch = allocateTorchTensor(shape.back(), grad_output_dtype);
  auto grad_bias_te = makeTransformerEngineTensor(grad_bias_torch);

  // Return immediately if tensors are empty
  if (product(shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Determine whether to use fused kernel
  bool with_fused_kernel = false;
  if (quantizer.is_none()) {
    // No need for separate quantization step if output is unquantized
    with_fused_kernel = true;
  } else if (IsFloat8Quantizers(quantizer.ptr()) || IsMXFP8Quantizers(quantizer.ptr())) {
    // FP8 delayed scaling and MXFP8 support fused kernel
    with_fused_kernel = true;
  }

  // Allocate temporary buffer for grad input if needed for unfused impl
  TensorWrapper temp_grad_input_te;
  py::object temp_grad_input_py;
  if (!with_fused_kernel) {
    NoneQuantizer q{py::none()};
    std::tie(temp_grad_input_te, temp_grad_input_py) = q.create_tensor(shape, grad_output_dtype);
  }
  TensorWrapper &kernel_grad_input_te = with_fused_kernel ? grad_input_te : temp_grad_input_te;

  // Query workspace size and allocate workspace
  TensorWrapper workspace_te;
  at::Tensor workspace_torch;
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_te.data(), kernel_grad_input_te.data(), grad_bias_te.data(),
                        workspace_te.data(), at::cuda::getCurrentCUDAStream());
  });
  if (workspace_te.ndim() > 0 && workspace_te.numel() > 0) {
    workspace_torch = allocateSpace(workspace_te.shape(), workspace_te.dtype());
    workspace_te = makeTransformerEngineTensor(workspace_torch.data_ptr(),
                                               workspace_te.shape(),
                                               workspace_te.dtype());
  }

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_te.data(), kernel_grad_input_te.data(), grad_bias_te.data(),
                        workspace_te.data(), at::cuda::getCurrentCUDAStream());
  });

  // Quantize grad input if needed
  if (!with_fused_kernel) {
    quantizer_cpp->quantize(temp_grad_input_te, grad_input_te);
  }

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

}  // namespace transformer_engine::pytorch
