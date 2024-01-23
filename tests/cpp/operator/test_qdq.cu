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

#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename InputType, typename OutputType>
void compute_ref_q(const InputType *data, OutputType *output,
                   const size_t N,
                   float *amax, float scale) {
    using compute_t = float;
    compute_t current_max = -1e100;
    for (size_t i = 0; i < N; ++i) {
      compute_t current = static_cast<compute_t>(data[i]);
      current_max = fmaxf(current_max, fabsf(current));
      if (std::is_same<OutputType, test::fp8e4m3>::value ||
          std::is_same<OutputType, test::fp8e5m2>::value) {
        output[i] = OutputType(scale * current);
      } else {
        output[i] = OutputType(current);
      }
    }
    *amax = current_max;
}

template <typename InputType, typename OutputType>
void compute_ref_dq(const InputType *data, OutputType *output,
                    const size_t N, float scale_inv) {
    using compute_t = float;
    for (size_t i = 0; i < N; ++i) {
      compute_t current = static_cast<compute_t>(data[i]);
      output[i] = OutputType(scale_inv * current);
    }
}

template <typename InputType, typename OutputType>
void performTestQ(const size_t N) {
  using namespace test;

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  Tensor input({ N }, itype);
  Tensor output({ N }, otype);

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(N);

  fillUniform(&input);
  setRandomScale(&output);

  nvte_fp8_quantize(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_q<InputType, OutputType>(input.cpu_dptr<InputType>(), ref_output.get(),
                                       N, &ref_amax, output.scale());

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
  compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_q", output, ref_output.get(), atol, rtol);
}

template <typename InputType, typename OutputType>
void performTestDQ(const size_t N) {
  using namespace test;

  DType itype = TypeInfo<InputType>::dtype;
  DType otype = TypeInfo<OutputType>::dtype;

  Tensor input({ N }, itype);
  Tensor output({ N }, otype);

  std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(N);

  fillUniform(&input);

  nvte_fp8_dequantize(input.data(), output.data(), 0);

  compute_ref_dq<InputType, OutputType>(input.cpu_dptr<InputType>(), ref_output.get(),
                                        N, input.scale_inv());

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_dq", output, ref_output.get(), atol, rtol);
}

std::vector<size_t> qdq_test_cases = {2048* 12288,
                                      768 * 1024,
                                      256 * 65536,
                                      65536 * 128,
                                      257 * 259,
                                      128*128+1};

} //namespace

class QDQTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                transformer_engine::DType,
                                                                size_t>> {};

TEST_P(QDQTestSuite, TestQ) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const size_t N = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTestQ<InputType, OutputType>(N);
      );
    );
}

TEST_P(QDQTestSuite, TestDQ) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const size_t N = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTestDQ<OutputType, InputType>(N);
      );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    QDQTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(qdq_test_cases)),
    [](const testing::TestParamInfo<QDQTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param));
      return name;
    });
