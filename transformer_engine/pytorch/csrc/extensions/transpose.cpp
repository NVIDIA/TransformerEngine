/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "ATen/core/TensorBody.h"
#include "extensions.h"

std::vector<py::object> fused_multi_quantize(std::vector<py::handle> input_list,
                                             std::optional<std::vector<py::handle>> output_list,
                                             std::vector<py::handle> quantizer_list,
                                             transformer_engine::DType otype) {
  using namespace transformer_engine::pytorch;
  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrapper_input_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrapper_output_list;
  auto none = py::none();

  // create TE tensors from input
  for (int i = 0; i < input_list.size(); i++) {
    auto input_tensor = makeTransformerEngineTensor(input_list[i], none);
    const NVTEShape input_shape = input_tensor.shape();

    transformer_engine::TensorWrapper output_tensor;

    if (output_list == std::nullopt) {
      std::unique_ptr<Quantizer> quantizer = convert_quantizer(quantizer_list[i]);
      std::vector<size_t> output_shape(input_shape.data, input_shape.data + input_shape.ndim);
      py::object o;
      std::tie(output_tensor, o) = quantizer->create_tensor(output_shape, otype);
      py_output_objects_list.push_back(o);
    } else {
      output_tensor = makeTransformerEngineTensor((*output_list)[i], quantizer_list[i]);
    }
    if (input_tensor.numel() == 0) continue;

    nvte_tensor_output_list.emplace_back(output_tensor.data());
    nvte_tensor_input_list.emplace_back(input_tensor.data());
    tensor_wrapper_input_list.emplace_back(std::move(input_tensor));
    tensor_wrapper_output_list.emplace_back(std::move(output_tensor));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_tensor_output_list.size() == nvte_tensor_input_list.size(),
             "Number of input and output tensors must match");

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 per-tensor scaling
  bool with_fused_kernel = true;
  for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
    const auto& tensor = nvte_tensor_output_list[i];
    if (nvte_tensor_scaling_mode(tensor) != NVTE_DELAYED_TENSOR_SCALING) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    nvte_multi_quantize(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                        nvte_tensor_output_list.data(), at::cuda::getCurrentCUDAStream());
  } else {
    for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
      // For per-tensor current scaling mode, we need to compute amax and scale before quantization
      if (nvte_tensor_scaling_mode(nvte_tensor_output_list[i]) == NVTE_CURRENT_TENSOR_SCALING) {
        auto quantizer = convert_quantizer(quantizer_list[i]);
        // quantizer here has to be a Float8CurrentScalingQuantizer
        auto quantizer_cs = static_cast<Float8CurrentScalingQuantizer*>(quantizer.get());
        nvte_compute_amax(nvte_tensor_input_list[i], nvte_tensor_output_list[i],
                          at::cuda::getCurrentCUDAStream());
        // check if we need to do amax reudction (depending on model parallel configs)
        if (quantizer_cs->with_amax_reduction) {
          c10::intrusive_ptr<dist_group_type> process_group_ptr =
              quantizer_cs->amax_reduction_group;
          // construct torch tensor from NVTEBasicTensor without reallocating memory
          at::Tensor amax_tensor_torch = makeATenTensor(tensor_wrapper_output_list[i].get_amax());
          std::vector<at::Tensor> tensors = {amax_tensor_torch};
          // allreduce amax tensor
          c10d::AllreduceOptions allreduce_opts;
          allreduce_opts.reduceOp = c10d::ReduceOp::MAX;
          process_group_ptr->allreduce(tensors, allreduce_opts)->wait();
        }
        nvte_compute_scale_from_amax(nvte_tensor_output_list[i], at::cuda::getCurrentCUDAStream());
      }
      nvte_quantize(nvte_tensor_input_list[i], nvte_tensor_output_list[i],
                    at::cuda::getCurrentCUDAStream());
    }
  }
  return py_output_objects_list;
}

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype,
                         std::optional<at::Tensor> output) {
  using namespace transformer_engine::pytorch;

  const auto dim = input.dim();
  NVTE_CHECK(dim >= 2, "Need at least 2D tensor to transpose.");

  if (input.dim() > 2) {
    input = input.view({-1, input.size(dim - 1)});
  }

  size_t M = static_cast<size_t>(input.size(0));
  size_t N = static_cast<size_t>(input.size(1));

  at::Tensor out;
  if (output.has_value()) {
    out = *output;
  } else {
    out = allocateTorchTensor(input.size(1), input.size(0), DType::kByte);
  }
  if (M == 0 || N == 0) return out;

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), {M, N}, otype);
  auto output_cu = makeTransformerEngineTensor(out.data_ptr(), {N, M}, otype);

  nvte_transpose(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return out;
}
