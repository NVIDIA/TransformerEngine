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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_ref(const InputType *data, OutputType *output_c,
                 const size_t size,
                 float *amax, float scale) {
  using compute_t = float;
  compute_t current_max = -1e100;
  for (size_t i = 0; i < size; ++i) {
      compute_t current = static_cast<compute_t>(data[i]);
      current_max = fmaxf(current_max, fabsf(current));
      output_c[i] = OutputType(scale * current);
  }
  *amax = current_max;
}

template <typename InputType, typename OutputType>
void performTest(const std::vector<size_t>& shape) {
  using namespace test;

  const size_t full_size = product(shape);

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  Tensor input(shape, itype);
  Tensor output_c(shape, otype);

  std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(full_size);

  fillUniform(&input);
  setRandomScale(&output_c);

  nvte_quantize(input.data(), output_c.data(), 0);

  float ref_amax;
  compute_ref<InputType, OutputType>(input.rowwise_cpu_dptr<InputType>(), ref_output_c.get(),
                                     full_size, &ref_amax, output_c.scale());

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output_c.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / output_c.scale();
    compareResults("scale_inv", output_c.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output_c, ref_output_c.get(), true, atol, rtol);
}

std::vector<std::vector<size_t>> test_cases = {
  {16},
  {16000},
  {128, 128},
  {256, 256},
  {768, 1024},
  {256, 65536},
  {2048, 12288},
  {65536, 128},
  {65536, 160},
  {16384, 1616},
  {1, 128},
  {1, 1296},
  {1, 16},
  {5, 160},
  {5, 4, 3, 160},
  {217, 256},
};
}  // namespace

class CastTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                 transformer_engine::DType,
                                                                 std::vector<size_t>>> {};

TEST_P(CastTestSuite, TestCast) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
      performTest<InputType, OutputType>(size);
    );
  );
}



INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  CastTestSuite,
  ::testing::Combine(
      ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
      ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<CastTestSuite::ParamType>& info) {
    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       test::typeName(std::get<1>(info.param));
    const auto& shape = std::get<2>(info.param);
    for ( const auto& s: shape) {
      name += "X" + std::to_string(s);
    }
    return name;
  });
