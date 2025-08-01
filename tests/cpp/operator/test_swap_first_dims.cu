/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename Type>
void compute_ref(const Type *input,  Type *output,
                 const std::vector<size_t> &shape) {
  const size_t dim0 = shape[0];
  const size_t dim1 = shape[1];
  size_t dim2 = 1;
  for (size_t i = 2; i < shape.size(); ++i) {
    dim2 *= shape[i];
  }
  for (size_t i = 0; i < dim0; ++i) {
    for (size_t j = 0; j < dim1; ++j) {
      for (size_t k = 0; k < dim2; ++k) {
        const size_t in_offset = i * dim1 * dim2 + j * dim2 + k;
        const size_t out_offset = j * dim0 * dim2 + i * dim2 + k;
        output[out_offset] = input[in_offset];
      }
    }
  }
}

template <typename Type>
void performTest(const std::vector<size_t> &in_shape) {
  using namespace test;

  DType dtype = TypeInfo<Type>::dtype;

  // Tensor dimensions
  std::vector<size_t> out_shape = in_shape;
  out_shape[0] = in_shape[1];
  out_shape[1] = in_shape[0];
  size_t numel = 1;
  for (const auto& dim : in_shape) {
    numel *= dim;
  }

  // Transformer engine implementation
  Tensor input("input", in_shape, dtype);
  Tensor output("output", out_shape, dtype);
  fillUniform(&input);
  nvte_swap_first_dims(input.data(), output.data(), 0);

  // Reference implementation
  std::unique_ptr<Type[]> ref_output = std::make_unique<Type[]>(numel);
  compute_ref<Type>(input.rowwise_cpu_dptr<Type>(), ref_output.get(), in_shape);

  // Check for CUDA failure
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  // Check for exact numerics
  compareResults("output", output, ref_output.get(), true, 0, 0);
}

std::vector<std::vector<size_t>> test_cases = {{4, 64, 1280},
                                               {48, 8, 128, 16},
                                               {229, 173},  // Primes 50, 40
                                               {113, 71, 1, 1, 1, 29, 1, 1}};  // Primes 30, 20, 10
}  // namespace

class SwapFirstDimsTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                          std::vector<size_t>>> {};

TEST_P(SwapFirstDimsTestSuite, TestSwapFirstDims) {
  using namespace transformer_engine;
  using namespace test;

  const DType type = std::get<0>(GetParam());
  const auto shape = std::get<1>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
    performTest<T>(shape);
  );
}



INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwapFirstDimsTestSuite,
  ::testing::Combine(
      ::testing::ValuesIn(test::all_fp_types),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<SwapFirstDimsTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param));
    for (const auto& dim : std::get<1>(info.param)) {
      name += "X";
      name += std::to_string(dim);
    }
    return name;
  });
