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

  QuantizationConfigWrapper quant_config;
  quant_config.set_noop_tensor(te_noop.data());

  if (detail::IsFloat8CurrentScalingQuantizers(quantizer.ptr())) {
    // my_quantizer here has to be a Float8CurrentScalingQuantizer
    auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(my_quantizer.get());
    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_amax(te_input.data(), te_output.data(), at::cuda::getCurrentCUDAStream());
    });
    // check if we need to do amax reudction (depending on model parallel configs)
    if (my_quantizer_cs->with_amax_reduction) {
      c10::intrusive_ptr<dist_group_type> process_group_ptr = my_quantizer_cs->amax_reduction_group;
      // construct torch tesnor from NVTEBasicTensor without reallocating memory
      at::Tensor& amax_tensor_torch = my_quantizer_cs->amax;
      std::vector<at::Tensor> tensors = {amax_tensor_torch};
      // allreduce amax tensor
      c10d::AllreduceOptions allreduce_opts;
      allreduce_opts.reduceOp = c10d::ReduceOp::MAX;
      process_group_ptr->allreduce(tensors, allreduce_opts)->wait();
    }
    // this config is used for cs scaling factor computation
    // because compute scale is cannot be fused with quantize kernel
    // so in nvte_quantize_v2 with current scaling, the quant config is not used again
    quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_scale_from_amax(te_output.data(), quant_config,
                                   at::cuda::getCurrentCUDAStream());
    });
    // set amax ptr to null in te_output TensorWrapper to avoid atomic amax updates in kernel
    te_output.set_amax(nullptr, DType::kFloat32, te_output.defaultShape);
  } else if (detail::IsFloat8BlockwiseQuantizers(quantizer.ptr())) {
    auto my_quantizer_bw = static_cast<Float8BlockQuantizer*>(my_quantizer.get());
    quant_config.set_force_pow_2_scales(my_quantizer_bw->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_bw->amax_epsilon);
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_v2(te_input.data(), te_output.data(), quant_config,
                     at::cuda::getCurrentCUDAStream());
  });

  return out;
}

py::object dequantize(const py::handle& input, transformer_engine::DType otype) {
  init_extension();

  const auto none = py::none();

  const auto& input_tensor = makeTransformerEngineTensor(input, none);

  NoneQuantizer q(none);

  const auto& shape = convertShape(input_tensor.shape());

  auto [out_tensor, out] = q.create_tensor(shape, otype);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_dequantize(input_tensor.data(), out_tensor.data(), at::cuda::getCurrentCUDAStream());
  });

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
  NVTE_SCOPED_GIL_RELEASE({
    func(grad_tensor.data(), act_input_tensor.data(), dact_tensor.data(), dbias_tensor.data(),
         workspace.data(), at::cuda::getCurrentCUDAStream());
  });
  auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
  workspace =
      makeTransformerEngineTensor(workspace_data.data_ptr(), workspace.shape(), workspace.dtype());

  // Launch kernel
  NVTE_SCOPED_GIL_RELEASE({
    func(grad_tensor.data(), act_input_tensor.data(), dact_tensor.data(), dbias_tensor.data(),
         workspace.data(), at::cuda::getCurrentCUDAStream());
  });

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
