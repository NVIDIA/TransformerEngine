/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    if (nvfp4_quantizer_cpp->row_scaled_nvfp4 ||
        (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax)) {
      // Amax is handled within NVFP4 quantizer
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
        auto [temp_nvte, _, amax_buf] =
            fp8_quantizer_cpp->create_unquantized_tensor_with_amax(output_shape, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (act_func == nullptr) {
            act_func_with_args(input_nvte.data(), temp_nvte.data(), std::forward<Args>(args)...,
                               stream);
          } else {
            act_func(input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        fp8_quantizer_cpp->quantize_with_amax(temp_nvte, out_nvte, amax_buf);
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
    if (nvfp4_quantizer_cpp->row_scaled_nvfp4 ||
        (nvfp4_quantizer_cpp->with_rht && nvfp4_quantizer_cpp->with_post_rht_amax)) {
      // Amax is handled within NVFP4 quantizer
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
        auto [temp_nvte, _, amax_buf] =
            fp8_quantizer_cpp->create_unquantized_tensor_with_amax(input_shape, fake_dtype);
        NVTE_SCOPED_GIL_RELEASE({
          if constexpr (dact_func == nullptr) {
            dact_func_with_args(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(),
                                std::forward<Args>(args)..., stream);
          } else {
            dact_func(grad_output_nvte.data(), input_nvte.data(), temp_nvte.data(), stream);
          }
        });
        fp8_quantizer_cpp->quantize_with_amax(temp_nvte, grad_input_nvte, amax_buf);
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

py::object glu(const at::Tensor& input, py::handle quantizer) {
  return activation_helper<nvte_glu, nullptr>(input, quantizer, 2);
}

py::object dglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer) {
  return dactivation_helper<nvte_dglu, nullptr>(grad, input, quantizer);
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
py::object clamped_swiglu(const at::Tensor& input, py::handle quantizer, float limit, float alpha,
                          float glu_linear_offset) {
  return activation_helper<nullptr, nvte_clamped_swiglu_v2>(input, quantizer, 2, limit, alpha,
                                                            glu_linear_offset);
}

py::object clamped_dswiglu(const at::Tensor& grad, const at::Tensor& input, py::handle quantizer,
                           float limit, float alpha, float glu_linear_offset) {
  return dactivation_helper<nullptr, nvte_clamped_dswiglu_v2>(grad, input, quantizer, limit, alpha,
                                                              glu_linear_offset);
}

/* Scaled activation helpers (activation + per-row scale via nvte_scaled_*).
 *
 * Grouped variants reuse the dense compute helpers, then optionally apply
 * group_quantize. Keep the nvte_scaled_* launch path in one place.
 */

template <auto act_func, typename... Args>
at::Tensor scaled_activation_compute(const at::Tensor& input, const at::Tensor& act_scales,
                                     int shape_divisor, Args&&... args) {
  init_extension();
  NVTE_CHECK(input.dim() >= 1, "scaled activation input must have at least 1 dimension");
  NVTE_CHECK(shape_divisor > 0 && input.size(-1) % shape_divisor == 0,
             "scaled activation input width is not compatible with activation");

  auto input_tensor = input.contiguous();
  auto scales_tensor = act_scales.contiguous().reshape({-1});
  const int64_t rows = input_tensor.numel() / input_tensor.size(-1);
  NVTE_CHECK(scales_tensor.numel() == rows, "scaled activation expects one scale per input row");

  std::vector<int64_t> output_sizes(input_tensor.sizes().begin(), input_tensor.sizes().end());
  output_sizes.back() /= shape_divisor;
  auto output = at::empty(output_sizes, input_tensor.options());

  const TensorWrapper& input_nvte = makeTransformerEngineTensor(input_tensor);
  const TensorWrapper& scales_nvte = makeTransformerEngineTensor(scales_tensor);
  const TensorWrapper& output_nvte = makeTransformerEngineTensor(output);

  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    act_func(input_nvte.data(), scales_nvte.data(), output_nvte.data(), std::forward<Args>(args)...,
             stream);
  });
  return output;
}

template <auto dact_func, typename... Args>
std::tuple<at::Tensor, at::Tensor> scaled_dactivation_compute(const at::Tensor& grad,
                                                              const at::Tensor& input,
                                                              const at::Tensor& act_scales,
                                                              bool compute_scale_grad,
                                                              Args&&... args) {
  init_extension();
  NVTE_CHECK(input.dim() >= 1 && grad.dim() >= 1,
             "scaled dactivation input and grad must have at least 1 dimension");

  auto grad_tensor = grad.contiguous();
  auto input_tensor = input.contiguous();
  auto scales_tensor = act_scales.contiguous();
  const int64_t rows = input_tensor.numel() / input_tensor.size(-1);
  NVTE_CHECK(scales_tensor.numel() == rows, "scaled dactivation expects one scale per input row");

  auto scales_flat = scales_tensor.reshape({-1});
  auto grad_input = at::empty_like(input_tensor);
  auto grad_scales = compute_scale_grad ? at::empty_like(scales_tensor) : at::Tensor();
  auto grad_scales_flat = compute_scale_grad ? grad_scales.reshape({-1}) : at::Tensor();

  const TensorWrapper& grad_nvte = makeTransformerEngineTensor(grad_tensor);
  const TensorWrapper& input_nvte = makeTransformerEngineTensor(input_tensor);
  const TensorWrapper& scales_nvte = makeTransformerEngineTensor(scales_flat);
  const TensorWrapper& grad_input_nvte = makeTransformerEngineTensor(grad_input);
  std::optional<TensorWrapper> grad_scales_nvte;
  if (compute_scale_grad) {
    grad_scales_nvte.emplace(makeTransformerEngineTensor(grad_scales_flat));
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  NVTE_SCOPED_GIL_RELEASE({
    dact_func(grad_nvte.data(), input_nvte.data(), scales_nvte.data(), grad_input_nvte.data(),
              compute_scale_grad ? grad_scales_nvte->data() : nullptr, std::forward<Args>(args)...,
              stream);
  });
  return {grad_input, grad_scales};
}

py::object maybe_quantize(const at::Tensor& tensor, py::handle quantizer) {
  if (quantizer.is_none()) {
    return py::cast(tensor);
  }
  auto quantizer_cpp = convert_quantizer(quantizer);
  const TensorWrapper& tensor_nvte = makeTransformerEngineTensor(tensor);
  const auto shape_te = tensor_nvte.shape();
  const std::vector<size_t> shape(shape_te.data, shape_te.data + shape_te.ndim);
  auto fake_dtype = GetTransformerEngineDType(tensor.scalar_type());
  auto [out_nvte, out_py] = quantizer_cpp->create_tensor(shape, fake_dtype);
  quantizer_cpp->quantize(tensor_nvte, out_nvte);
  return out_py;
}

py::object maybe_group_quantize(const at::Tensor& tensor, py::handle quantizer,
                                const size_t num_tensors, std::optional<at::Tensor> first_dims,
                                std::optional<at::Tensor> tensor_offsets) {
  if (quantizer.is_none()) {
    return py::cast(tensor);
  }
  return group_quantize(tensor, quantizer, num_tensors, first_dims, std::nullopt, tensor_offsets,
                        std::nullopt);
}

template <auto act_func, typename... Args>
py::object scaled_activation_helper(const at::Tensor& input, const at::Tensor& act_scales,
                                    py::handle quantizer, int shape_divisor, Args&&... args) {
  auto output = scaled_activation_compute<act_func>(input, act_scales, shape_divisor,
                                                    std::forward<Args>(args)...);
  return maybe_quantize(output, quantizer);
}

template <auto dact_func, typename... Args>
py::tuple scaled_dactivation_helper(const at::Tensor& grad, const at::Tensor& input,
                                    const at::Tensor& act_scales, py::handle quantizer,
                                    bool compute_scale_grad, Args&&... args) {
  auto [grad_input, grad_scales] = scaled_dactivation_compute<dact_func>(
      grad, input, act_scales, compute_scale_grad, std::forward<Args>(args)...);
  return py::make_tuple(maybe_quantize(grad_input, quantizer),
                        compute_scale_grad ? py::cast(grad_scales) : py::none());
}

template <auto act_func, typename... Args>
py::object grouped_scaled_activation_helper(const at::Tensor& input, const at::Tensor& act_scales,
                                            py::handle quantizer, const size_t num_tensors,
                                            std::optional<at::Tensor> first_dims,
                                            std::optional<at::Tensor> tensor_offsets,
                                            int shape_divisor, Args&&... args) {
  NVTE_CHECK(input.dim() == 2, "grouped scaled activation input must be 2D");
  auto output = scaled_activation_compute<act_func>(input, act_scales, shape_divisor,
                                                    std::forward<Args>(args)...);
  return maybe_group_quantize(output, quantizer, num_tensors, first_dims, tensor_offsets);
}

template <auto dact_func, typename... Args>
py::tuple grouped_scaled_dactivation_helper(const at::Tensor& grad, const at::Tensor& input,
                                            const at::Tensor& act_scales, py::handle quantizer,
                                            const size_t num_tensors,
                                            std::optional<at::Tensor> first_dims,
                                            std::optional<at::Tensor> tensor_offsets,
                                            bool compute_scale_grad, Args&&... args) {
  NVTE_CHECK(input.dim() == 2 && grad.dim() == 2,
             "grouped scaled dactivation input and grad must be 2D");
  auto [grad_input, grad_scales] = scaled_dactivation_compute<dact_func>(
      grad, input, act_scales, compute_scale_grad, std::forward<Args>(args)...);
  // Return both the (optionally) grouped-quantized grad input for the next
  // grouped GEMM and the dense high-precision grad input so callers can reuse
  // it (e.g. bias gradient) without a lossy dequantize.
  return py::make_tuple(
      maybe_group_quantize(grad_input, quantizer, num_tensors, first_dims, tensor_offsets),
      py::cast(grad_input), compute_scale_grad ? py::cast(grad_scales) : py::none());
}

py::object scaled_swiglu(const at::Tensor& input, const at::Tensor& act_scales,
                         py::handle quantizer, int64_t glu_interleave_size) {
  return scaled_activation_helper<nvte_scaled_swiglu>(input, act_scales, quantizer,
                                                      /*shape_divisor=*/2, glu_interleave_size);
}

py::object scaled_clamped_swiglu(const at::Tensor& input, const at::Tensor& act_scales,
                                 py::handle quantizer, float limit, float alpha,
                                 float glu_linear_offset, int64_t glu_interleave_size) {
  return scaled_activation_helper<nvte_scaled_clamped_swiglu>(
      input, act_scales, quantizer, /*shape_divisor=*/2, limit, alpha, glu_linear_offset,
      glu_interleave_size);
}

py::object scaled_srelu(const at::Tensor& input, const at::Tensor& act_scales,
                        py::handle quantizer) {
  return scaled_activation_helper<nvte_scaled_srelu>(input, act_scales, quantizer,
                                                     /*shape_divisor=*/1);
}

py::tuple scaled_dswiglu(const at::Tensor& grad, const at::Tensor& input,
                         const at::Tensor& act_scales, py::handle quantizer,
                         int64_t glu_interleave_size, bool compute_scale_grad) {
  return scaled_dactivation_helper<nvte_scaled_dswiglu>(grad, input, act_scales, quantizer,
                                                        compute_scale_grad, glu_interleave_size);
}

py::tuple scaled_clamped_dswiglu(const at::Tensor& grad, const at::Tensor& input,
                                 const at::Tensor& act_scales, py::handle quantizer, float limit,
                                 float alpha, float glu_linear_offset, int64_t glu_interleave_size,
                                 bool compute_scale_grad) {
  return scaled_dactivation_helper<nvte_scaled_clamped_dswiglu>(
      grad, input, act_scales, quantizer, compute_scale_grad, limit, alpha, glu_linear_offset,
      glu_interleave_size);
}

py::tuple scaled_dsrelu(const at::Tensor& grad, const at::Tensor& input,
                        const at::Tensor& act_scales, py::handle quantizer,
                        bool compute_scale_grad) {
  return scaled_dactivation_helper<nvte_scaled_dsrelu>(grad, input, act_scales, quantizer,
                                                       compute_scale_grad);
}

py::object grouped_scaled_swiglu(const at::Tensor& input, const at::Tensor& act_scales,
                                 py::handle quantizer, const size_t num_tensors,
                                 std::optional<at::Tensor> first_dims,
                                 std::optional<at::Tensor> tensor_offsets,
                                 int64_t glu_interleave_size) {
  return grouped_scaled_activation_helper<nvte_scaled_swiglu>(
      input, act_scales, quantizer, num_tensors, first_dims, tensor_offsets, /*shape_divisor=*/2,
      glu_interleave_size);
}

py::object grouped_scaled_clamped_swiglu(const at::Tensor& input, const at::Tensor& act_scales,
                                         py::handle quantizer, const size_t num_tensors,
                                         std::optional<at::Tensor> first_dims,
                                         std::optional<at::Tensor> tensor_offsets, float limit,
                                         float alpha, float glu_linear_offset,
                                         int64_t glu_interleave_size) {
  return grouped_scaled_activation_helper<nvte_scaled_clamped_swiglu>(
      input, act_scales, quantizer, num_tensors, first_dims, tensor_offsets, /*shape_divisor=*/2,
      limit, alpha, glu_linear_offset, glu_interleave_size);
}

py::object grouped_scaled_srelu(const at::Tensor& input, const at::Tensor& act_scales,
                                py::handle quantizer, const size_t num_tensors,
                                std::optional<at::Tensor> first_dims,
                                std::optional<at::Tensor> tensor_offsets) {
  return grouped_scaled_activation_helper<nvte_scaled_srelu>(
      input, act_scales, quantizer, num_tensors, first_dims, tensor_offsets,
      /*shape_divisor=*/1);
}

py::tuple grouped_scaled_dswiglu(const at::Tensor& grad, const at::Tensor& input,
                                 const at::Tensor& act_scales, py::handle quantizer,
                                 const size_t num_tensors, std::optional<at::Tensor> first_dims,
                                 std::optional<at::Tensor> tensor_offsets,
                                 int64_t glu_interleave_size, bool compute_scale_grad) {
  return grouped_scaled_dactivation_helper<nvte_scaled_dswiglu>(
      grad, input, act_scales, quantizer, num_tensors, first_dims, tensor_offsets,
      compute_scale_grad, glu_interleave_size);
}

py::tuple grouped_scaled_clamped_dswiglu(const at::Tensor& grad, const at::Tensor& input,
                                         const at::Tensor& act_scales, py::handle quantizer,
                                         const size_t num_tensors,
                                         std::optional<at::Tensor> first_dims,
                                         std::optional<at::Tensor> tensor_offsets, float limit,
                                         float alpha, float glu_linear_offset,
                                         int64_t glu_interleave_size, bool compute_scale_grad) {
  return grouped_scaled_dactivation_helper<nvte_scaled_clamped_dswiglu>(
      grad, input, act_scales, quantizer, num_tensors, first_dims, tensor_offsets,
      compute_scale_grad, limit, alpha, glu_linear_offset, glu_interleave_size);
}

py::tuple grouped_scaled_dsrelu(const at::Tensor& grad, const at::Tensor& input,
                                const at::Tensor& act_scales, py::handle quantizer,
                                const size_t num_tensors, std::optional<at::Tensor> first_dims,
                                std::optional<at::Tensor> tensor_offsets, bool compute_scale_grad) {
  return grouped_scaled_dactivation_helper<nvte_scaled_dsrelu>(grad, input, act_scales, quantizer,
                                                               num_tensors, first_dims,
                                                               tensor_offsets, compute_scale_grad);
}

}  // namespace pytorch
}  // namespace transformer_engine
