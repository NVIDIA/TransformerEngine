/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

std::vector<py::object> fused_multi_quantize(std::vector<at::Tensor> input_list,
                                             std::optional<std::vector<py::object>> output_list,
                                             std::vector<py::handle> quantizer_list,
                                             transformer_engine::DType otype) {
  init_extension();
  std::vector<NVTETensor> nvte_tensor_input_list;
  std::vector<NVTETensor> nvte_tensor_output_list;
  std::vector<py::object> py_output_objects_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  if (output_list.has_value()) {
    py_output_objects_list = output_list.value();
  }

  // Choose implementation
  // Note: Currently only have fused kernel for FP8 cast-transpose
  bool with_fused_kernel = true;

  // create TE tensors from input
  for (size_t i = 0; i < input_list.size(); i++) {
    auto input_tensor = makeTransformerEngineTensor(input_list[i]);
    const NVTEShape input_shape = input_tensor.shape();

    transformer_engine::TensorWrapper output_tensor;

    if (!detail::IsFloat8Quantizers(quantizer_list[i].ptr())) {
      with_fused_kernel = false;
    }
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
    tensor_wrappers.emplace_back(std::move(input_tensor));
    tensor_wrappers.emplace_back(std::move(output_tensor));
  }

  // Check tensor lists
  NVTE_CHECK(nvte_tensor_output_list.size() == nvte_tensor_input_list.size(),
             "Number of input and output tensors must match");

  for (size_t i = 0; i < nvte_tensor_output_list.size(); i++) {
    if (nvte_tensor_columnwise_data(nvte_tensor_output_list[i]) == nullptr) {
      with_fused_kernel = false;
      break;
    }
  }

  // Launch TE kernel
  if (with_fused_kernel) {
    nvte_multi_cast_transpose(nvte_tensor_input_list.size(), nvte_tensor_input_list.data(),
                              nvte_tensor_output_list.data(), at::cuda::getCurrentCUDAStream());
  } else {
    for (size_t i = 0; i < py_output_objects_list.size(); i++) {
      quantize(input_list[i], quantizer_list[i], py_output_objects_list[i], std::nullopt);
    }
  }
  return py_output_objects_list;
}

at::Tensor fp8_transpose(at::Tensor input, transformer_engine::DType otype,
                         std::optional<at::Tensor> output) {
  init_extension();

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

std::vector<py::object> fp8_blockwise_transpose(std::vector<py::object> tensor_list) {
  init_extension();
  auto none = py::none();
  for (auto tensor : tensor_list) {
    NVTE_CHECK(!tensor.is_none(), "Tensor has not been provided");
    TensorWrapper te_tensor = makeTransformerEngineTensor(tensor, none);
    nvte_transpose_blockwise(te_tensor.data(), at::cuda::getCurrentCUDAStream());
  }

  return tensor_list;
}

}  // namespace transformer_engine::pytorch
