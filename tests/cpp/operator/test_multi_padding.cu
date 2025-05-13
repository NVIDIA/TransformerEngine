/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
void compute_ref(const std::vector<std::vector<InputType>>& input_list,
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

    for (size_t i = 0; i < padded_height; ++i) {
      if (i < height) {
        for (size_t j = 0; j < width; ++j) {
          const compute_t x = static_cast<compute_t>(input[i * width + j]);
          const OutputType y = static_cast<OutputType>(x);
          output[i * width + j] = y;
        }
      } else {
        for (size_t j = 0; j < width; ++j) {
          output[i * width + j] = static_cast<OutputType>(0.f);
        }
      }
    }
  }
}

template <typename InputType, typename OutputType>
void performTest() {
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
  std::vector<Tensor> input_list, output_list, output_t_list;

  // Buffers for reference implementation
  std::vector<std::vector<InputType>> ref_input_list;
  std::vector<std::vector<OutputType>> ref_output_list;
  std::vector<size_t> ref_height_list(num_tensors), ref_width_list(num_tensors);
  std::vector<int> ref_padded_height_list(num_tensors);

  // Initialize buffers
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t height = tensor_dims[tensor_id].first;
    const size_t width = tensor_dims[tensor_id].second;
    const size_t padded_height = (height + align - 1) / align * align;
    input_list.emplace_back(Tensor("input_" + std::to_string(tensor_id), std::vector<size_t>{ height, width }, itype));
    output_list.emplace_back(Tensor("output_" + std::to_string(tensor_id), std::vector<size_t>{ padded_height, width }, otype));

    auto& input = input_list.back();
    auto& output = output_list.back();
    fillUniform(&input);
    setRandomScale(&output);

    ref_input_list.emplace_back(height*width);
    ref_output_list.emplace_back(padded_height*width);

    std::copy(input.rowwise_cpu_dptr<InputType>(),
              input.rowwise_cpu_dptr<InputType>() + height * width,
              ref_input_list.back().begin());
    ref_height_list[tensor_id] = height;
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
  nvte_multi_padding(num_tensors,
                                make_nvte_vector(input_list).data(),
                                make_nvte_vector(output_list).data(),
                                ref_padded_height_list.data(),
                                0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Reference implementation
  compute_ref<InputType, OutputType>(ref_input_list,
                                     ref_output_list,
                                     ref_height_list,
                                     ref_width_list,
                                     ref_padded_height_list);

  // Check correctness
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output",
                   output_list[tensor_id],
                   ref_output_list[tensor_id].data(),
                   true,
                   atol, rtol);
  }
}

}  // namespace

class MultiPaddingTestSuite
  : public ::testing::TestWithParam<
                                               transformer_engine::DType> {};

TEST_P(MultiPaddingTestSuite, TestMultiPaddingTranspose) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = GetParam();
  const DType output_type = input_type;

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>();
    );
  );
}


INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MultiPaddingTestSuite,
  ::testing::ValuesIn(test::all_fp_types),
  [](const testing::TestParamInfo<MultiPaddingTestSuite::ParamType>& info) {
    std::string name = test::typeName(info.param);
    return name;
  });
