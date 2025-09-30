/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "../extensions.h"
#include "common.h"
#include "pybind.h"

namespace transformer_engine {
namespace pytorch {

namespace {
using FuncType = void(const NVTETensor, NVTETensor, cudaStream_t);
using DFuncType = void(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t);

template <FuncType* act_func, auto act_func_with_args, typename... Args>
py::object activation_helper(const at::Tensor& input, py::handle quantizer, int shape_divisor = 1,
                             Args&&... args) {
  init_extension();

  // Input tensor
  auto input_tensor = input.contiguous();
  const TensorWrapper& input_nvte = makeTransformerEngineTensor(input_tensor);

  // Construct output tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  const auto input_shape = input_nvte.shape();
  std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
  output_shape.back() /= shape_divisor;
  auto fake_dtype = GetTransformerEngineDType(input_tensor.scalar_type());
  auto [out_nvte, out_py] = quantizer_cpp->create_tensor(output_shape, fake_dtype);

  // Choose implementation
  enum class Impl { UNFUSED, FULLY_FUSED, FUSED_ACTIVATION_AMAX_FP8, FUSED_ACTIVATION_AMAX_NVFP4 };
  Impl impl = Impl::UNFUSED;
  if (quantizer.is_none() || detail::IsFloat8Quantizers(quantizer.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer.ptr())) {
    impl = Impl::FULLY_FUSED;
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    impl = Impl::FUSED_ACTIVATION_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer*>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax) {
      // Post-RHT amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else {
      impl = Impl::FUSED_ACTIVATION_AMAX_NVFP4;
    }
  }

  // Perform compute
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (impl) {
    case Impl::UNFUSED:
      // Compute activation in high precision, then quantize
      {
        auto [temp_nvte, _] = NoneQuantizer(py::none()).create_tensor(output_shape, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (act_func == nullptr) {
            act_func_with_args(input_nvte.data(), temp_nvte.data(), std::forward<Args>(args)...,
                               stream);
          } else {
            act_func(input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        quantizer_cpp->quantize(temp_nvte, out_nvte);
      }
      break;
    case Impl::FULLY_FUSED:
      // Compute activation directly
      {
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (act_func == nullptr) {
            act_func_with_args(input_nvte.data(), out_nvte.data(), std::forward<Args>(args)...,
                               stream);
          } else {
            act_func(input_nvte.data(), out_nvte.data(), stream);
          }
        });
      }
      break;
    case Impl::FUSED_ACTIVATION_AMAX_FP8:
      // Compute activation and amax in high precision, then quantize to FP8
      {
        auto fp8_quantizer_cpp = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
        NVTE_CHECK(fp8_quantizer_cpp != nullptr, "Could not cast to FP8 current scaling quantizer");
        auto [temp_nvte, _] =
            fp8_quantizer_cpp->create_unquantized_tensor_with_amax(output_shape, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (act_func == nullptr) {
            act_func_with_args(input_nvte.data(), temp_nvte.data(), std::forward<Args>(args)...,
                               stream);
          } else {
            act_func(input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        fp8_quantizer_cpp->quantize_with_amax(temp_nvte, out_nvte);
      }
      break;
    case Impl::FUSED_ACTIVATION_AMAX_NVFP4:
      // Compute activation and amax in high precision, then quantize to NVFP4
      {
        auto nvfp4_quantizer_cpp =
            static_cast<NVFP4Quantizer*>(quantizer_cpp.get());  // Already checked cast is valid
        auto [temp_nvte, _] =
            nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(out_nvte, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (act_func == nullptr) {
            act_func_with_args(input_nvte.data(), temp_nvte.data(), std::forward<Args>(args)...,
                               stream);
          } else {
            act_func(input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        nvfp4_quantizer_cpp->quantize_with_amax(temp_nvte, out_nvte);
      }
      break;
    default:
      NVTE_ERROR("Invalid activation implementation (", static_cast<int>(impl), ")");
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
  const TensorWrapper& grad_output_nvte = makeTransformerEngineTensor(grad_output_tensor);
  const TensorWrapper& input_nvte = makeTransformerEngineTensor(input_tensor);

  // Construct grad input tensor
  auto quantizer_cpp = convert_quantizer(quantizer);
  const auto input_shape_te = input_nvte.shape();
  const std::vector<size_t> input_shape(input_shape_te.data,
                                        input_shape_te.data + input_shape_te.ndim);
  auto fake_dtype = GetTransformerEngineDType(input_tensor.scalar_type());
  auto [grad_input_nvte, grad_input_py] = quantizer_cpp->create_tensor(input_shape, fake_dtype);

  // Choose implementation
  enum class Impl { UNFUSED, FULLY_FUSED, FUSED_ACTIVATION_AMAX_FP8, FUSED_ACTIVATION_AMAX_NVFP4 };
  Impl impl = Impl::UNFUSED;
  if (quantizer.is_none() || detail::IsFloat8Quantizers(quantizer.ptr()) ||
      detail::IsMXFP8Quantizers(quantizer.ptr())) {
    impl = Impl::FULLY_FUSED;
  } else if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    impl = Impl::FUSED_ACTIVATION_AMAX_FP8;
  } else if (detail::IsNVFP4Quantizers(quantizer.ptr())) {
    auto nvfp4_quantizer_cpp = dynamic_cast<NVFP4Quantizer*>(quantizer_cpp.get());
    NVTE_CHECK(nvfp4_quantizer_cpp != nullptr, "Could not cast to NVFP4 quantizer");
    if (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax) {
      // Post-RHT amax is handled within NVFP4 quantizer
      impl = Impl::UNFUSED;
    } else {
      impl = Impl::FUSED_ACTIVATION_AMAX_NVFP4;
    }
  }

  // Perform compute
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (impl) {
    case Impl::UNFUSED:
      // Compute activation backward in high precision, then quantize
      {
        auto [temp_nvte, _] = NoneQuantizer(py::none()).create_tensor(input_shape, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (dact_func == nullptr) {
            dact_func_with_args(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(),
                                std::forward<Args>(args)..., stream);
          } else {
            dact_func(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        quantizer_cpp->quantize(temp_nvte, grad_input_nvte);
      }
      break;
    case Impl::FULLY_FUSED:
      // Compute activation backward directly
      {
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (dact_func == nullptr) {
            dact_func_with_args(grad_output_nvte.data(), input_nvte.data(), grad_input_nvte.data(),
                                std::forward<Args>(args)..., stream);
          } else {
            dact_func(grad_output_nvte.data(), input_nvte.data(), grad_input_nvte.data(), stream);
          }
        });
      }
      break;
    case Impl::FUSED_ACTIVATION_AMAX_FP8:
      // Compute activation and amax in high precision, then quantize to FP8
      {
        auto fp8_quantizer_cpp = dynamic_cast<Float8CurrentScalingQuantizer*>(quantizer_cpp.get());
        NVTE_CHECK(fp8_quantizer_cpp != nullptr, "Could not cast to FP8 current scaling quantizer");
        auto [temp_nvte, _] =
            fp8_quantizer_cpp->create_unquantized_tensor_with_amax(input_shape, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (dact_func == nullptr) {
            dact_func_with_args(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(),
                                std::forward<Args>(args)..., stream);
          } else {
            dact_func(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        fp8_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte);
      }
      break;
    case Impl::FUSED_ACTIVATION_AMAX_NVFP4:
      // Compute activation and amax in high precision, then quantize to NVFP4
      {
        auto nvfp4_quantizer_cpp =
            static_cast<NVFP4Quantizer*>(quantizer_cpp.get());  // Already checked cast is valid
        auto [temp_nvte, _] =
            nvfp4_quantizer_cpp->create_unquantized_tensor_with_amax(grad_input_nvte, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (dact_func == nullptr) {
            dact_func_with_args(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(),
                                std::forward<Args>(args)..., stream);
          } else {
            dact_func(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        nvfp4_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte);
      }
      break;
    default:
      NVTE_ERROR("Invalid activation implementation (", static_cast<int>(impl), ")");
  }

  return grad_input_py;
}
}  // namespace

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

/* ReLU and variants */
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

/* clamped functions */
py::object clamped_swiglu(const at::Tensor& input, py::handle quantizer, float limit, float alpha) {
  return activation_helper<nullptr, nvte_clamped_swiglu>(input, quantizer, 2, limit, alpha);
}

py::object clamped_dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer,
                           float limit, float alpha) {
  return dactivation_helper<nullptr, nvte_clamped_dswiglu>(grad, input, quantizer, limit, alpha);
}

}  // namespace pytorch
}  // namespace transformer_engine
