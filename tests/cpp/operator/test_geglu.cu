/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/logging.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>
#include "../test_common.h"

using namespace transformer_engine;

template <typename IT, typename OT, typename CT>
void compute_ref_geglu_cast(const IT *input_h, OT *output_h, const CT scale, CT *amax_h,
                            const size_t N, const size_t H) {
  CT amax = 0.;

  const int col = H * 2;

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT gelu_elt = CT(input_h[i * col + j]);
      gelu_elt = 0.5f * gelu_elt *
                 (1.0f + tanhf(0.79788456F * gelu_elt * (1.0f + 0.044715f * gelu_elt * gelu_elt)));
      CT gate_elt = CT(input_h[i * col + H + j]);
      CT elt = gelu_elt * gate_elt;
      output_h[i * H + j] = OT(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <typename IType, typename OType>
void performTestGEGLU(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input({N, H * 2}, itype);
  Tensor output({N, H}, otype);

  fillUniform(&input);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N * H);

  nvte_geglu(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_geglu_cast(input.cpu_dptr<IType>(), ref_output.get(), output.scale(), &ref_amax, N,
                         H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_gelu", output, ref_output.get(), atol, rtol);
}

class GeGLUTestSuite
    : public ::testing::TestWithParam<std::tuple<
          transformer_engine::DType, transformer_engine::DType, std::pair<size_t, size_t>>> {};

TEST_P(GeGLUTestSuite, TestGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGEGLU<InputType, OutputType>(size.first, size.second);););
}

namespace {

std::vector<std::pair<size_t, size_t>> test_cases = {
    {4096, 2048}, {768, 2816}, {256, 5120}, {128, 10240}, {256, 256}, {257, 259}, {128, 128 + 1}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, GeGLUTestSuite,
    ::testing::Combine(::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::ValuesIn(test::all_fp_types), ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<GeGLUTestSuite::ParamType> &info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
