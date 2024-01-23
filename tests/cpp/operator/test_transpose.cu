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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename Type>
void compute_ref(const Type *data,  Type *output,
                 const size_t N, const size_t H) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      output[j * N + i] = data[i * H + j];
    }
  }
}

template <typename Type>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType dtype = TypeInfo<Type>::dtype;

  Tensor input({ N, H }, dtype);
  Tensor output({ H, N }, dtype);

  std::unique_ptr<Type[]> ref_output = std::make_unique<Type[]>(N * H);

  fillUniform(&input);

  nvte_transpose(input.data(), output.data(), 0);

  compute_ref<Type>(input.cpu_dptr<Type>(), ref_output.get(), N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  auto [atol, rtol] = getTolerances(dtype);
  compareResults("output", output, ref_output.get(), atol, rtol);
}

std::vector<std::pair<size_t, size_t>> test_cases = {{2048, 12288},
                                                     {768, 1024},
                                                     {256, 65536},
                                                     {65536, 128},
                                                     {256, 256},
                                                     {120, 2080},
                                                     {8, 8},
                                                     {1223, 1583}, // Primes 200, 250
                                                     {1, 541},     // Prime 100
                                                     {1987, 1}};   // Prime 300
}  // namespace

class TTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                              std::pair<size_t, size_t>>> {};

TEST_P(TTestSuite, TestTranspose) {
  using namespace transformer_engine;
  using namespace test;

  const DType type = std::get<0>(GetParam());
  const auto size = std::get<1>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
    performTest<T>(size.first, size.second);
  );
}



INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  TTestSuite,
  ::testing::Combine(
      ::testing::ValuesIn(test::all_fp_types),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<TTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       std::to_string(std::get<1>(info.param).first) + "X" +
                       std::to_string(std::get<1>(info.param).second);
    return name;
  });
