/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_ref(const std::vector<std::vector<InputType>>& input_list,
                 std::vector<std::vector<OutputType>>& output_c_list,
                 std::vector<std::vector<OutputType>>& output_t_list,
                 const std::vector<float>& scale_list,
                 std::vector<float>& amax_list,
                 const std::vector<size_t>& height_list,
                 const std::vector<size_t>& width_list) {
  using compute_t = float;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = input_list[tensor_id];
    auto& output_c = output_c_list[tensor_id];
    auto& output_t = output_t_list[tensor_id];
    const compute_t scale = scale_list[tensor_id];
    compute_t& amax = amax_list[tensor_id];
    const size_t height = height_list[tensor_id];
    const size_t width = width_list[tensor_id];
    amax = -1e100;
    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        const compute_t x = static_cast<compute_t>(input[i * width + j]);
        const OutputType y = static_cast<OutputType>(scale * x);
        amax = fmaxf(amax, fabsf(x));
        output_c[i * width + j] = y;
        output_t[j * height + i] = y;
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

  // Buffers for Transformer Engine implementation
  std::vector<Tensor> input_list, output_c_list, output_t_list;

  // Buffers for reference implementation
  std::vector<std::vector<InputType>> ref_input_list;
  std::vector<std::vector<OutputType>> ref_output_c_list, ref_output_t_list;
  std::vector<float> ref_scale_list(num_tensors), ref_amax_list(num_tensors);
  std::vector<size_t> ref_height_list(num_tensors), ref_width_list(num_tensors);

  // Initialize buffers
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    const size_t height = tensor_dims[tensor_id].first;
    const size_t width = tensor_dims[tensor_id].second;
    input_list.emplace_back(Tensor({ height, width }, itype));
    output_c_list.emplace_back(Tensor({ height, width }, otype));
    output_t_list.emplace_back(Tensor({ width, height }, otype));

    auto& input = input_list.back();
    auto& output_c = output_c_list.back();
    auto& output_t = output_t_list.back();
    fillUniform(&input);
    setRandomScale(&output_c);
    output_t.shareFP8Meta(output_c);

    ref_input_list.emplace_back(height*width);
    ref_output_c_list.emplace_back(height*width);
    ref_output_t_list.emplace_back(width*height);

    std::copy(input.cpu_dptr<InputType>(),
              input.cpu_dptr<InputType>() + height * width,
              ref_input_list.back().begin());
    ref_scale_list[tensor_id] = output_c.scale();
    ref_height_list[tensor_id] = height;
    ref_width_list[tensor_id] = width;
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
  nvte_multi_cast_transpose(num_tensors,
                            make_nvte_vector(input_list).data(),
                            make_nvte_vector(output_c_list).data(),
                            make_nvte_vector(output_t_list).data(),
                            0);

  // Reference implementation
  compute_ref<InputType, OutputType>(ref_input_list,
                                     ref_output_c_list,
                                     ref_output_t_list,
                                     ref_scale_list,
                                     ref_amax_list,
                                     ref_height_list,
                                     ref_width_list);

  // Check correctness
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  for (size_t tensor_id = 0; tensor_id < num_tensors; ++tensor_id) {
    if (isFp8Type(otype)) {
      auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
      compareResults("amax",
                     output_c_list[tensor_id].amax(),
                     ref_amax_list[tensor_id],
                     atol_amax, rtol_amax);
    }
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_c",
                   output_c_list[tensor_id],
                   ref_output_c_list[tensor_id].data(),
                   atol, rtol);
    compareResults("output_t",
                   output_t_list[tensor_id],
                   ref_output_t_list[tensor_id].data(),
                   atol, rtol);
  }
}

}  // namespace

class MultiCastTransposeTestSuite
  : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                               transformer_engine::DType>> {};

TEST_P(MultiCastTransposeTestSuite, TestMultiCastTranspose) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>();
    );
  );
}


INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  MultiCastTransposeTestSuite,
  ::testing::Combine(
      ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
      ::testing::ValuesIn(test::all_fp_types)),
  [](const testing::TestParamInfo<MultiCastTransposeTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       test::typeName(std::get<1>(info.param));
    return name;
  });
