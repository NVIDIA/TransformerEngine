/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <cstdio>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/padding.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_unpadding_ref(const std::vector<std::vector<InputType>>& input_list,
                         std::vector<std::vector<OutputType>>& output_list,
                         const std::vector<size_t>& height_list,
                         const std::vector<size_t>& width_list,
                         const std::vector<int>& padded_height_list) {
  using compute_t = float;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = input_list[tensor_id];
    auto& output = output_list[tensor_id];
    const size_t height = height_list[tensor_id];
    const size_t width = width_list[tensor_id];
    const size_t padded_height = padded_height_list[tensor_id];

    // Only copy the valid (unpadded) portion
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        const compute_t x = static_cast<compute_t>(input[i * width + j]);
        const OutputType y = static_cast<OutputType>(x);
        output[i * width + j] = y;
      }
    }
  }
}

template <typename InputType, typename OutputType>
void performUnpaddingTest() {
  using namespace test;

  const DType itype = TypeInfo<InputType>::dtype;
  const DType otype = TypeInfo<OutputType>::dtype;
  const std::vector<std::pair<size_t, size_t>> tensor_dims = {{1,1},
                                                            {1,768},
                                                            {768,1},
                                                            {768,768},
                                                            {43,43},
                                                            {43,256},
                                                            {256,43},
                                                            {256,256}};
  const size_t num_tensors = tensor_dims.size();
  constexpr int align = 16;

  // Buffers for Transformer Engine implementation
  std::vector<Tensor> padded_input_list, unpadded_output_list;

  // Buffers for reference implementation
  std::vector<std::vector<InputType>> ref_padded_input_list;
  std::vector<std::vector<OutputType>> ref_unpadded_output_list;
  std::vector<size_t> ref_height_list(num_tensors), ref_width_list(num_tensors);
  std::vector<int> ref_padded_height_list(num_tensors);

  // Initialize buffers
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t original_height = tensor_dims[tensor_id].first;
    const size_t width = tensor_dims[tensor_id].second;
    const size_t padded_height = (original_height + align - 1) / align * align;

    // Input is padded tensor (padded_height x width)
    padded_input_list.emplace_back(
        Tensor("padded_input_" + std::to_string(tensor_id),
               std::vector<size_t>{padded_height, width}, itype));

    // Output is unpadded tensor (original_height x width)
    unpadded_output_list.emplace_back(
        Tensor("unpadded_output_" + std::to_string(tensor_id),
               std::vector<size_t>{original_height, width}, otype));

    auto& padded_input = padded_input_list.back();
    auto& unpadded_output = unpadded_output_list.back();

    // Fill padded input with random data (including padding area)
    fillUniform(&padded_input);
    setRandomScale(&unpadded_output);

    // Initialize reference buffers
    ref_padded_input_list.emplace_back(padded_height * width);
    ref_unpadded_output_list.emplace_back(original_height * width);

    // Copy data to reference buffers
    std::copy(padded_input.rowwise_cpu_dptr<InputType>(),
              padded_input.rowwise_cpu_dptr<InputType>() + padded_height * width,
              ref_padded_input_list.back().begin());

    ref_height_list[tensor_id] = original_height;
    ref_width_list[tensor_id] = width;
    ref_padded_height_list[tensor_id] = padded_height;
  }

  // Transformer Engine implementation
  auto make_nvte_vector = [](std::vector<Tensor>& tensor_list)
    -> std::vector<NVTETensor> {
    std::vector<NVTETensor> nvte_tensor_list;
    for (auto& tensor : tensor_list) {
      nvte_tensor_list.emplace_back(tensor.data());
    }
    return nvte_tensor_list;
  };

  // Convert height_list to int for the API
  std::vector<int> original_height_list_int(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    original_height_list_int[i] = static_cast<int>(ref_height_list[i]);
  }

  // Call unpadding API
  nvte_multi_unpadding(num_tensors,
                      make_nvte_vector(padded_input_list).data(),
                      make_nvte_vector(unpadded_output_list).data(),
                      original_height_list_int.data(),
                      0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Reference implementation
  compute_unpadding_ref<InputType, OutputType>(ref_padded_input_list,
                                             ref_unpadded_output_list,
                                             ref_height_list,
                                             ref_width_list,
                                             ref_padded_height_list);

  // Check correctness
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("unpadded_output",
                  unpadded_output_list[tensor_id],
                  ref_unpadded_output_list[tensor_id].data(),
                  true,
                  atol, rtol);
  }
}

}  // namespace

class MultiUnpaddingTestSuite
  : public ::testing::TestWithParam<transformer_engine::DType> {};

TEST_P(MultiUnpaddingTestSuite, TestMultiUnpadding) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = GetParam();
  const DType output_type = input_type;

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performUnpaddingTest<InputType, OutputType>();
    );
  );
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MultiUnpaddingTestSuite,
  ::testing::ValuesIn(test::all_fp_types),
  [](const testing::TestParamInfo<MultiUnpaddingTestSuite::ParamType>& info) {
    std::string name = test::typeName(info.param);
    return name;
  });
