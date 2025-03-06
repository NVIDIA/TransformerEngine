/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

void fused_multi_row_padding(at::Tensor input, at::Tensor output,
                             std::vector<size_t> input_row_list,
                             std::vector<size_t> padded_input_row_list) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  NVTE_CHECK(input_row_list.size() == padded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal  2.");

  const int num_tensors = input_row_list.size();
  // Extract properties from PyTorch tensors
  std::vector<void*> input_dptr_list, output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, output_shape_list;
  std::vector<transformer_engine::DType> input_type_list;
  void* d_input_ptr = reinterpret_cast<void*>(input.data_ptr());
  void* d_output_ptr = reinterpret_cast<void*>(output.data_ptr());
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_dptr_list.push_back(d_input_ptr);
    output_dptr_list.push_back(d_output_ptr);

    // Move the input pointer to the next split.
    char* input_char_ptr = reinterpret_cast<char*>(d_input_ptr);
    const size_t input_dptr_offset =
        input_row_list[tensor_id] * input.size(1) * input.element_size();
    input_char_ptr += input_dptr_offset;
    d_input_ptr = reinterpret_cast<void*>(input_char_ptr);

    input_shape_list.push_back({input_row_list[tensor_id], static_cast<size_t>(input.size(1))});
    input_type_list.push_back(GetTransformerEngineDType(input.scalar_type()));

    // Move the output pointer to the next split.
    char* output_char_ptr = reinterpret_cast<char*>(d_output_ptr);
    const size_t output_dptr_offset =
        padded_input_row_list[tensor_id] * output.size(1) * output.element_size();
    output_char_ptr += output_dptr_offset;
    d_output_ptr = reinterpret_cast<void*>(output_char_ptr);

    output_shape_list.push_back(
        {padded_input_row_list[tensor_id], static_cast<size_t>(output.size(1))});
  }

  // Construct TE tensors
  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<transformer_engine::TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        transformer_engine::DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> padded_num_rows_list;
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    if (input_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_list.emplace_back(
        make_tensor(input_dptr_list[i], input_shape_list[i], input_type_list[i]));
    nvte_output_list.emplace_back(
        make_tensor(output_dptr_list[i], output_shape_list[i], input_type_list[i]));
    padded_num_rows_list.emplace_back(padded_input_row_list[i]);
  }

  // Check tensor lists
  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");
  NVTE_CHECK(padded_num_rows_list.size() == nvte_input_list.size() &&
             "Number of input and padded row list must match");

  // Launch TE kernel
  nvte_multi_padding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                     padded_num_rows_list.data(), at::cuda::getCurrentCUDAStream());
}
