/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

/* Type aliases for readability */
using FuncType = void(const NVTETensor, NVTETensor, cudaStream_t);
using DFuncType = void(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t);

template <FuncType* act_func, auto act_func_with_args, typename... Args>
py::object activation_helper(const at::Tensor& input, py::handle quantizer, int shape_divisor = 1,
                             Args&&... args) {
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
    NVTE_SCOPED_GIL_RELEASE({
      if constexpr (act_func == nullptr) {
        act_func_with_args(input_cpp.data(), out_cpp.data(), std::forward<Args>(args)...,
                           at::cuda::getCurrentCUDAStream());
      } else {
        act_func(input_cpp.data(), out_cpp.data(), at::cuda::getCurrentCUDAStream());
      }
    });
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // Compute activation in high-precision fused together with amax, then quantize.
    auto quantizer_cpp_cs = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    auto [temp_cpp, _] = quantizer_cpp_cs->create_hp_tensor_with_amax(output_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE({
      if constexpr (act_func == nullptr) {
        act_func_with_args(input_cpp.data(), temp_cpp.data(), std::forward<Args>(args)...,
                           at::cuda::getCurrentCUDAStream());
      } else {
        act_func(input_cpp.data(), temp_cpp.data(), at::cuda::getCurrentCUDAStream());
      }
    });
    quantizer_cpp_cs->quantize_with_amax(temp_cpp, out_cpp);
  } else {
    // Compute activation in high-precision, then quantize
    auto [temp_cpp, _] = NoneQuantizer(py::none()).create_tensor(output_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE({
      if constexpr (act_func == nullptr) {
        act_func_with_args(input_cpp.data(), temp_cpp.data(), std::forward<Args>(args)...,
                           at::cuda::getCurrentCUDAStream());
      } else {
        act_func(input_cpp.data(), temp_cpp.data(), at::cuda::getCurrentCUDAStream());
      }
    });
    quantizer_cpp->quantize(temp_cpp, out_cpp);
  }

  return out_py;
}

template <DFuncType* dact_func, auto dact_func_with_args, typename... Args>
py::object dactivation_helper(const at::Tensor& grad_output, const at::Tensor& input,
                              py::handle quantizer, Args&&... args) {
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
      if constexpr (dact_func == nullptr) {
        dact_func_with_args(grad_output_cpp.data(), input_cpp.data(), grad_input_cpp.data(),
                            std::forward<Args>(args)..., at::cuda::getCurrentCUDAStream());
      } else {
        dact_func(grad_output_cpp.data(), input_cpp.data(), grad_input_cpp.data(),
                  at::cuda::getCurrentCUDAStream());
      }
    });
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // Compute activation backward in high-precision fused together with amax, then quantize.
    auto quantizer_cpp_cs = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
    auto [temp_cpp, _] = quantizer_cpp_cs->create_hp_tensor_with_amax(input_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE({
      if constexpr (dact_func == nullptr) {
        dact_func_with_args(grad_output_cpp.data(), input_cpp.data(), temp_cpp.data(),
                            std::forward<Args>(args)..., at::cuda::getCurrentCUDAStream());
      } else {
        dact_func(grad_output_cpp.data(), input_cpp.data(), temp_cpp.data(),
                  at::cuda::getCurrentCUDAStream());
      }
    });
    quantizer_cpp_cs->quantize_with_amax(temp_cpp, grad_input_cpp);
  } else {
    // Compute activation backward in high-precision, then quantize
    auto [temp_cpp, _] = NoneQuantizer(py::none()).create_tensor(input_shape, fake_dtype);
    NVTE_SCOPED_GIL_RELEASE({
      if constexpr (dact_func == nullptr) {
        dact_func_with_args(grad_output_cpp.data(), input_cpp.data(), temp_cpp.data(),
                            std::forward<Args>(args)..., at::cuda::getCurrentCUDAStream());
      } else {
        dact_func(grad_output_cpp.data(), input_cpp.data(), temp_cpp.data(),
                  at::cuda::getCurrentCUDAStream());
      }
    });
    quantizer_cpp->quantize(temp_cpp, grad_input_cpp);
  }

  return grad_input_py;
}

/* GELU and variants */
py::object gelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_gelu, nullptr>(input, quantizer);
}

py::object dgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgelu, nullptr>(grad, input, quantizer);
}

py::object geglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_geglu, nullptr>(input, quantizer, 2);
}

py::object dgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dgeglu, nullptr>(grad, input, quantizer);
}

py::object qgelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgelu, nullptr>(input, quantizer);
}

py::object dqgelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgelu, nullptr>(grad, input, quantizer);
}

py::object qgeglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_qgeglu, nullptr>(input, quantizer, 2);
}

py::object dqgeglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dqgeglu, nullptr>(grad, input, quantizer);
}

/* ReLU and variants*/
py::object relu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_relu, nullptr>(input, quantizer);
}

py::object drelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_drelu, nullptr>(grad, input, quantizer);
}

py::object reglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_reglu, nullptr>(input, quantizer, 2);
}

py::object dreglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dreglu, nullptr>(grad, input, quantizer);
}

py::object srelu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_srelu, nullptr>(input, quantizer);
}

py::object dsrelu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsrelu, nullptr>(grad, input, quantizer);
}

py::object sreglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_sreglu, nullptr>(input, quantizer, 2);
}

py::object dsreglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsreglu, nullptr>(grad, input, quantizer);
}
/* Silu and variants */
py::object silu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_silu, nullptr>(input, quantizer);
}

py::object dsilu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dsilu, nullptr>(grad, input, quantizer);
}

py::object swiglu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_swiglu, nullptr>(input, quantizer, 2);
}

py::object dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dswiglu, nullptr>(grad, input, quantizer);
}

/* gpt_oss functions */
py::object gpt_oss_swiglu(const at::Tensor& input, py::handle quantizer, float limit, float alpha) {
  return activation_helper<nullptr, nvte_gptoss_swiglu>(input, quantizer, 2, limit, alpha);
}

py::object gpt_oss_dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer,
                           float limit, float alpha) {
  return dactivation_helper<nullptr, nvte_gptoss_dswiglu>(grad, input, quantizer, limit, alpha);
}

}  // namespace transformer_engine::pytorch
