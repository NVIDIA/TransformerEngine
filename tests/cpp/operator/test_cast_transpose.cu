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

template <typename InputType, typename OutputType>
void compute_ref(const InputType *data, OutputType *output_c, OutputType *output_t,
                 const size_t N, const size_t H,
                 float *amax, float scale) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < H; ++j) {
      compute_t current = static_cast<compute_t>(data[i * H + j]);
      current_max = fmaxf(current_max, fabsf(current));
      output_c[i * H + j] = OutputType(scale * current);
      output_t[j * N + i] = OutputType(scale * current);
    }
  }
  *amax = current_max;
}

template <typename InputType, typename OutputType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  Tensor input({ N, H }, itype);
  Tensor output_c({ N, H }, otype);
  Tensor output_t({ H, N }, otype);

  std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(N * H);
  std::unique_ptr<OutputType[]> ref_output_t = std::make_unique<OutputType[]>(N * H);

  fillUniform(&input);
  setRandomScale(&output_c);
  output_t.shareFP8Meta(output_c);

  nvte_cast_transpose(input.data(), output_c.data(), output_t.data(), 0);

  float ref_amax;
  compute_ref<InputType, OutputType>(input.cpu_dptr<InputType>(), ref_output_c.get(),
                                     ref_output_t.get(), N, H, &ref_amax,
                                     output_c.scale());

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output_c.amax(), ref_amax, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output_c, ref_output_c.get(), atol, rtol);
  compareResults("output_t", output_t, ref_output_t.get(), atol, rtol);
}

std::vector<std::pair<size_t, size_t>> test_cases = {{2048, 12288},
                                                     {768, 1024},
                                                     {256, 65536},
                                                     {65536, 128},
                                                     {256, 256},
                                                     {120, 2080},
                                                     {8, 8},
                                                     {1, 3221},       // Prime 456
                                                     {2333, 1},       // Prime 345
                                                     {1481, 677}};    // Primes 234, 123
}  // namespace

class CTTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                               transformer_engine::DType,
                                                               std::pair<size_t, size_t>>> {};

TEST_P(CTTestSuite, TestCastTranspose) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>(size.first, size.second);
    );
  );
}



INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  CTTestSuite,
  ::testing::Combine(
      ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
      ::testing::ValuesIn(test::all_fp_types),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<CTTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       test::typeName(std::get<1>(info.param)) + "X" +
                       std::to_string(std::get<2>(info.param).first) + "X" +
                       std::to_string(std::get<2>(info.param).second);
    return name;
  });
