/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
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

  // for current scaling, we need to compute amax first and then quantize
  // because cache cannot fit in the entire tensor to compute amax and quantize
  // the quantizer should not need amax reduction, no process group needed here
  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // activation function might change the input data range, we need to first call the activation function
    // and then find the amax and scale of that and then do the quantization
    // get a NoneQuantizer to calculate amax of activation output
    auto my_quantizer_none = std::make_unique<NoneQuantizer>(py::none());
    auto [te_output_act, out_act] =
        my_quantizer_none->create_tensor(input_shape, GetTransformerEngineDType(fake_tensor_type));

    NVTE_SCOPED_GIL_RELEASE({
      act_func(te_input.data(), te_output_act.data(), at::cuda::getCurrentCUDAStream());
      // use te_output_act as input to the compute amax and find the amax of activated tensor
      nvte_compute_amax(te_output_act.data(), te_output.data(), at::cuda::getCurrentCUDAStream());
    });

    // my_quantizer here has to be a Float8CurrentScalingQuantizer
    auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(my_quantizer.get());
    if (my_quantizer_cs->with_amax_reduction) {
      NVTE_ERROR(
          "per-tensor current scaling amax reduction is not supported in activation functions.");
    }
    QuantizationConfigWrapper quant_config;
    quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);

    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_scale_from_amax(te_output.data(), quant_config,
                                   at::cuda::getCurrentCUDAStream());
      // set amax ptr to null in te_output TensorWrapper to avoid atomic amax updates in kernel
      te_output.set_amax(nullptr, DType::kFloat32, te_output.defaultShape);
      nvte_quantize_v2(te_output_act.data(), te_output.data(), quant_config,
                       at::cuda::getCurrentCUDAStream());
    });
  } else if (detail::IsFloat8BlockwiseQuantizers(quantizer.ptr())) {
    // sanity check, since activation fusion is not supported for blockwise quantization yet
    // need to raise an error here instead of silently going into act_func with wrong numerics
    NVTE_ERROR("Activation fusion is not supported for blockwise quantization yet.");
  } else {
    NVTE_SCOPED_GIL_RELEASE(
        { act_func(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream()); });
  }

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

  NVTE_SCOPED_GIL_RELEASE({
    act_func(te_grad.data(), te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());
  });

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
