/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <omp.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename IType, typename OType>
void compute_ref_cast_dgated_swiglu(const IType * const grad,
                                    const IType * const input,
                                    const float scale,
                                    OType * const output,
                                    float * const amax_ptr,
                                    const size_t rows,
                                    const size_t cols) {
  float amax = 0;
  const size_t stride = cols * 2;

  #pragma omp parallel for reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      float grad_elt = static_cast<float>(grad[i * cols + j]);
      float silu_elt = static_cast<float>(input[i * stride + j]);
      float gate_elt = static_cast<float>(input[i * stride + cols + j]);

      float after_dsilu = dsilu(silu_elt) * grad_elt * gate_elt;
      float after_dgate = grad_elt * silu(silu_elt);

      if (abs(after_dsilu) > amax) { amax = abs(after_dsilu); }
      if (abs(after_dgate) > amax) { amax = abs(after_dgate); }

      output[i * stride + j] = static_cast<OType>(scale * after_dsilu);
      output[i * stride + cols + j] = static_cast<OType>(scale * after_dgate);
    }
  }

  *amax_ptr = amax;
}

template <typename IType, typename OType>
void performTest(const size_t rows, const size_t cols) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor grad({rows, cols}, itype);
  Tensor input({rows, cols * 2}, itype);
  Tensor output_c({rows, cols * 2}, otype);

  fillUniform(&grad);
  fillUniform(&input);
  setRandomScale(&output_c);

  std::unique_ptr<OType[]> ref_output_c = std::make_unique<OType[]>(rows * cols * 2);

  nvte_fp8_quantize_swiglu(grad.data(), input.data(), output_c.data(), 0);
  cudaDeviceSynchronize();

  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  float ref_amax;
  compute_ref_cast_dgated_swiglu(grad.cpu_dptr<IType>(),
                                 input.cpu_dptr<IType>(),
                                 output_c.scale(),
                                 ref_output_c.get(),
                                 &ref_amax,
                                 rows,
                                 cols);

  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output_c.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / output_c.scale();
    compareResults("scale_inv", output_c.scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }

  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output_c, ref_output_c.get(), atol, rtol);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
  {128, 128},
  {256, 256},
  {768, 1024},
  // {256, 65536},
  // {2048, 12288},
  // {65536, 128},
  // {16384, 6144},
};

}  // namespace

class CastSwiGLUTestSuite
    : public ::testing::TestWithParam<std::tuple<
          transformer_engine::DType, transformer_engine::DType, std::pair<size_t, size_t>>> {};

TEST_P(CastSwiGLUTestSuite, TestCastSwiGLU) {
  using namespace transformer_engine;
  using namespace test;
  // Skip tests for pre-Blackwell architectures
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
      GTEST_SKIP();
  }

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType, performTest<InputType, OutputType>(size.first, size.second);););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CastSwiGLUTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<CastSwiGLUTestSuite::ParamType> &info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
