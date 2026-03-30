/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/padding.h>

#include "../stable_common.h"

namespace transformer_engine::pytorch::stable {

using Tensor = torch::stable::Tensor;

void fused_multi_row_padding(Tensor input, Tensor output, std::vector<int64_t> input_row_list,
                             std::vector<int64_t> padded_input_row_list) {
  NVTE_CHECK(input_row_list.size() == padded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal 2.");

  const auto num_tensors = input_row_list.size();
  std::vector<void*> input_dptr_list, output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, output_shape_list;
  std::vector<DType> input_type_list;
  void* d_input_ptr = input.data_ptr();
  void* d_output_ptr = output.data_ptr();

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_dptr_list.push_back(d_input_ptr);
    output_dptr_list.push_back(d_output_ptr);

    char* input_char_ptr = reinterpret_cast<char*>(d_input_ptr);
    const size_t input_dptr_offset = static_cast<size_t>(input_row_list[tensor_id]) *
                                     static_cast<size_t>(input.size(1)) * input.element_size();
    input_char_ptr += input_dptr_offset;
    d_input_ptr = reinterpret_cast<void*>(input_char_ptr);

    input_shape_list.push_back(
        {static_cast<size_t>(input_row_list[tensor_id]), static_cast<size_t>(input.size(1))});
    input_type_list.push_back(GetTransformerEngineDType(input.scalar_type()));

    char* output_char_ptr = reinterpret_cast<char*>(d_output_ptr);
    const size_t output_dptr_offset = static_cast<size_t>(padded_input_row_list[tensor_id]) *
                                      static_cast<size_t>(output.size(1)) * output.element_size();
    output_char_ptr += output_dptr_offset;
    d_output_ptr = reinterpret_cast<void*>(output_char_ptr);

    output_shape_list.push_back({static_cast<size_t>(padded_input_row_list[tensor_id]),
                                 static_cast<size_t>(output.size(1))});
  }

  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        DType dtype) -> NVTETensor {
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
    padded_num_rows_list.emplace_back(static_cast<int>(padded_input_row_list[i]));
  }

  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");

  nvte_multi_padding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                     padded_num_rows_list.data(),
                     getCurrentCUDAStreamRaw(input.get_device_index()));
}

void fused_multi_row_unpadding(Tensor input, Tensor output, std::vector<int64_t> input_row_list,
                               std::vector<int64_t> unpadded_input_row_list) {
  NVTE_CHECK(input_row_list.size() == unpadded_input_row_list.size(),
             "Number of input row list and padded row list must match.");
  NVTE_CHECK(input.dim() == 2, "Dimension of input must equal 2.");
  NVTE_CHECK(output.dim() == 2, "Dimension of output must equal 2.");

  const auto num_tensors = input_row_list.size();
  std::vector<void*> input_dptr_list, output_dptr_list;
  std::vector<std::vector<size_t>> input_shape_list, output_shape_list;
  std::vector<DType> input_type_list;
  void* d_input_ptr = input.data_ptr();
  void* d_output_ptr = output.data_ptr();

  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    input_dptr_list.push_back(d_input_ptr);
    output_dptr_list.push_back(d_output_ptr);

    char* input_char_ptr = reinterpret_cast<char*>(d_input_ptr);
    const size_t input_dptr_offset = static_cast<size_t>(input_row_list[tensor_id]) *
                                     static_cast<size_t>(input.size(1)) * input.element_size();
    input_char_ptr += input_dptr_offset;
    d_input_ptr = reinterpret_cast<void*>(input_char_ptr);

    input_shape_list.push_back(
        {static_cast<size_t>(input_row_list[tensor_id]), static_cast<size_t>(input.size(1))});
    input_type_list.push_back(GetTransformerEngineDType(input.scalar_type()));

    char* output_char_ptr = reinterpret_cast<char*>(d_output_ptr);
    const size_t output_dptr_offset = static_cast<size_t>(unpadded_input_row_list[tensor_id]) *
                                      static_cast<size_t>(output.size(1)) * output.element_size();
    output_char_ptr += output_dptr_offset;
    d_output_ptr = reinterpret_cast<void*>(output_char_ptr);

    output_shape_list.push_back({static_cast<size_t>(unpadded_input_row_list[tensor_id]),
                                 static_cast<size_t>(output.size(1))});
  }

  std::vector<NVTETensor> nvte_input_list, nvte_output_list;
  std::vector<TensorWrapper> tensor_wrappers;
  auto make_tensor = [&tensor_wrappers](void* dptr, const std::vector<size_t>& shape,
                                        DType dtype) -> NVTETensor {
    tensor_wrappers.emplace_back(makeTransformerEngineTensor(dptr, shape, dtype));
    return tensor_wrappers.back().data();
  };

  std::vector<int> unpadded_num_rows_list;
  for (size_t i = 0; i < input_dptr_list.size(); ++i) {
    if (input_dptr_list[i] == nullptr || input_row_list[i] == 0) continue;
    nvte_input_list.emplace_back(
        make_tensor(input_dptr_list[i], input_shape_list[i], input_type_list[i]));
    nvte_output_list.emplace_back(
        make_tensor(output_dptr_list[i], output_shape_list[i], input_type_list[i]));
    unpadded_num_rows_list.emplace_back(static_cast<int>(unpadded_input_row_list[i]));
  }

  NVTE_CHECK(nvte_output_list.size() == nvte_input_list.size(),
             "Number of input and output tensors must match");

  nvte_multi_unpadding(nvte_input_list.size(), nvte_input_list.data(), nvte_output_list.data(),
                       unpadded_num_rows_list.data(),
                       getCurrentCUDAStreamRaw(input.get_device_index()));
}

STABLE_TORCH_LIBRARY_IMPL(transformer_engine_stable, CUDA, m) {
  m.impl("fused_multi_row_padding", TORCH_BOX(fused_multi_row_padding));
  m.impl("fused_multi_row_unpadding", TORCH_BOX(fused_multi_row_unpadding));
}

}  // namespace transformer_engine::pytorch::stable
