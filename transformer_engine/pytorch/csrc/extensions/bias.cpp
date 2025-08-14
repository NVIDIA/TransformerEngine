/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>

#include <utility>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "pybind.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace pytorch {

std::vector<py::object> bgrad_quantize(const at::Tensor &grad_output, py::handle quantizer) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Grad output tensor
  auto grad_output_torch = grad_output.contiguous();
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());

  // Construct grad bias tensor
  const int64_t bias_size = static_cast<int64_t>(shape.back());
  auto grad_bias_torch = allocateTorchTensor(bias_size, grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Unquantized impl only requires computing grad bias
  if (quantizer.is_none()) {
    if (product(shape) == 0) {
      grad_bias_torch.zero_();
    } else {
      at::sum_out(grad_bias_torch, grad_output_torch.reshape({-1, bias_size}), {0});
    }
    return {py::cast(std::move(grad_bias_torch)), py::cast(std::move(grad_output_torch))};
  }

  // Construct grad input tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  auto [grad_input_nvte, grad_input_py] = quantizer_cpp->create_tensor(shape, grad_output_dtype);

  // Trivial impl if tensors are empty
  if (product(shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Unfused impl if quantizer is not supported
  const bool with_fused_dbias_quantize_kernel =
      detail::IsFloat8Quantizers(quantizer.ptr()) || detail::IsMXFP8Quantizers(quantizer.ptr());
  if (!with_fused_dbias_quantize_kernel) {
    at::sum_out(grad_bias_torch, grad_output_torch.reshape({-1, bias_size}), {0});
    quantizer_cpp->quantize(grad_output_nvte, grad_input_nvte);
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Query workspace size
  TensorWrapper workspace_nvte;
  at::Tensor workspace_torch;
  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  // Allocate workspace
  if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
    workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
    workspace_nvte = makeTransformerEngineTensor(workspace_torch.data_ptr(), workspace_nvte.shape(),
                                                 workspace_nvte.dtype());
  }

  // Launch fused kernel
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(grad_output_nvte.data(), grad_input_nvte.data(), grad_bias_nvte.data(),
                        workspace_nvte.data(), stream);
  });

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

namespace {

std::vector<py::object> dact_dbias(
    void (*dact_dbias_func)(const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                            cudaStream_t),
    void (*dact_func)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
    at::Tensor grad_output_torch, at::Tensor act_input_torch, py::handle quantizer_py) {
  using namespace transformer_engine::pytorch::detail;
  init_extension();

  // Grad output and activation input tensors
  grad_output_torch = grad_output_torch.contiguous();
  const TensorWrapper &grad_output_nvte = makeTransformerEngineTensor(grad_output_torch);
  const auto output_shape = getTensorShape(grad_output_torch);
  auto grad_output_dtype = GetTransformerEngineDType(grad_output_torch.scalar_type());
  act_input_torch = act_input_torch.contiguous();
  const TensorWrapper &act_input_nvte = makeTransformerEngineTensor(act_input_torch);
  const auto input_shape = getTensorShape(act_input_torch);

  // Construct tensors
  auto quantizer_cpp = convert_quantizer(quantizer_py);
  auto [grad_input_nvte, grad_input_py] =
      quantizer_cpp->create_tensor(input_shape, grad_output_dtype);
  const int64_t bias_size = static_cast<int64_t>(input_shape.back());
  auto grad_bias_torch = allocateTorchTensor(bias_size, grad_output_dtype);
  auto grad_bias_nvte = makeTransformerEngineTensor(grad_bias_torch);

  // Return immediately if tensors are empty
  if (product(output_shape) == 0) {
    grad_bias_torch.zero_();
    return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
  }

  // Choose implementation
  enum class Impl { UNFUSED, FUSED_DACT_DBIAS_QUANTIZE, FUSED_DACT_AMAX };
  Impl impl = Impl::UNFUSED;
  if (detail::IsFloat8Quantizers(quantizer_py.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer_py.ptr())) {
    impl = Impl::FUSED_DACT_DBIAS_QUANTIZE;
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer_py.ptr())) {
    impl = Impl::FUSED_DACT_AMAX;
  }

  // Perform compute
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (impl) {
    case Impl::UNFUSED:
      // Unfused dact, dbias, quantize
      {
        auto [temp_nvte, temp_py] =
            NoneQuantizer(py::none()).create_tensor(input_shape, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = temp_py.cast<at::Tensor>();
        at::sum_out(grad_bias_torch, temp_torch.reshape({-1, bias_size}), {0});
        quantizer_cpp->quantize(temp_nvte, grad_input_nvte);
        break;
      }
    case Impl::FUSED_DACT_DBIAS_QUANTIZE:
      // Fused dact-dbias-quantize kernel
      {
        // Query workspace size
        TensorWrapper workspace_nvte;
        NVTE_SCOPED_GIL_RELEASE({
          dact_dbias_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
                          grad_bias_nvte.data(), workspace_nvte.data(), stream);
        });

        // Allocate workspace
        at::Tensor workspace_torch;
        if (workspace_nvte.ndim() > 0 && workspace_nvte.numel() > 0) {
          workspace_torch = allocateSpace(workspace_nvte.shape(), workspace_nvte.dtype());
          workspace_nvte = makeTransformerEngineTensor(
              workspace_torch.data_ptr(), workspace_nvte.shape(), workspace_nvte.dtype());
        }

        // Launch kernel
        NVTE_SCOPED_GIL_RELEASE({
          dact_dbias_func(grad_output_nvte.data(), act_input_nvte.data(), grad_input_nvte.data(),
                          grad_bias_nvte.data(), workspace_nvte.data(), stream);
        });
        break;
      }
    case Impl::FUSED_DACT_AMAX:
      // Fused dact-amax kernel, unfused dbias and quantize
      {
        auto *quantizer_cpp_cs = dynamic_cast<Float8CurrentScalingQuantizer *>(quantizer_cpp.get());
        NVTE_CHECK(quantizer_cpp_cs != nullptr,
                   "Invalid quantizer for fused dact-amax kernel impl");
        auto [temp_nvte, temp_py] =
            quantizer_cpp_cs->create_hp_tensor_with_amax(input_shape, grad_output_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          dact_func(grad_output_nvte.data(), act_input_nvte.data(), temp_nvte.data(), stream);
        });
        const auto temp_torch = temp_py.cast<at::Tensor>();
        at::sum_out(grad_bias_torch, temp_torch.reshape({-1, bias_size}), {0});
        quantizer_cpp_cs->quantize_with_amax(temp_nvte, grad_input_nvte);
        break;
      }
    default:
      NVTE_ERROR("Invalid implementation");
  }

  return {py::cast(std::move(grad_bias_torch)), std::move(grad_input_py)};
}

}  // namespace

std::vector<py::object> dbias_dgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dgelu, nvte_dgelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsilu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dsilu, nvte_dsilu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_drelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                    py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_drelu, nvte_drelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dqgelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dqgelu, nvte_dqgelu, grad_output, act_input, quantizer);
}

std::vector<py::object> dbias_dsrelu(const at::Tensor &grad_output, const at::Tensor &act_input,
                                     py::handle quantizer) {
  return dact_dbias(nvte_quantize_dbias_dsrelu, nvte_dsrelu, grad_output, act_input, quantizer);
}

}  // namespace pytorch
}  // namespace transformer_engine
