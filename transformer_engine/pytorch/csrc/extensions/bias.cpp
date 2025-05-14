/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"
#include "pybind.h"
#include "transformer_engine/cast.h"
#include "transformer_engine/transformer_engine.h"

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
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(input_tensor.data(), out_tensor.data(), dbias_tensor.data(),
                        workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  void* workspace_data_ptr = nullptr;
  if (workspace.shape().ndim > 0) {
    auto workspace_data = allocateSpace(workspace.shape(), workspace.dtype());
    workspace_data_ptr = workspace_data.data_ptr();
  }
  workspace = makeTransformerEngineTensor(workspace_data_ptr, workspace.shape(), workspace.dtype());

  // Launch kernel
  if (detail::IsFloat8CurrentScalingQuantizers(py_quantizer.ptr())) {
    // my_quantizer here has to be a Float8CurrentScalingQuantizer
    auto my_quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(quantizer.get());
    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_amax(input_tensor.data(), out_tensor.data(), at::cuda::getCurrentCUDAStream());
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
    QuantizationConfigWrapper quant_config;
    quant_config.set_force_pow_2_scales(my_quantizer_cs->force_pow_2_scales);
    quant_config.set_amax_epsilon(my_quantizer_cs->amax_epsilon);
    NVTE_SCOPED_GIL_RELEASE({
      nvte_compute_scale_from_amax(out_tensor.data(), quant_config,
                                   at::cuda::getCurrentCUDAStream());
    });
    // set amax ptr to null in te_output TensorWrapper to avoid atomic amax updates in kernel
    out_tensor.set_amax(nullptr, DType::kFloat32, out_tensor.defaultShape);
  }
  NVTE_SCOPED_GIL_RELEASE({
    nvte_quantize_dbias(input_tensor.data(), out_tensor.data(), dbias_tensor.data(),
                        workspace.data(), at::cuda::getCurrentCUDAStream());
  });

  return {py::cast(dbias), out};
}

}  // namespace transformer_engine::pytorch
