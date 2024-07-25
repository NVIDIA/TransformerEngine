/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

float gelu(const float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456F * x * (1.0f + 0.044715f * x * x)));
}

float silu(const float x) {
  return x / (1 + expf(-x));
}

float relu(const float x) {
  return x > 0 ? x : 0;
}

}  // namespace

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_act_cast(const IT *input_h,
                           OT *output_h,
                           const CT scale,
                           CT *amax_h,
                           const size_t N,
                           const size_t H) {
  CT amax  = 0.;

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = CT(input_h[i * H + j]);
      elt = act(elt);
      output_h[i * H + j] = OT(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_glu_act_cast(const IT *input_h, OT *output_h, const CT scale, CT *amax_h,
                              const size_t N, const size_t H) {
  CT amax = 0.;

  const int col = H * 2;

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT gelu_elt = CT(input_h[i * col + j]);
      gelu_elt = act(gelu_elt);
      CT gate_elt = CT(input_h[i * col + H + j]);
      CT elt = gelu_elt * gate_elt;
      output_h[i * H + j] = OT(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}


template <float (*ref_act)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input({ N, H }, itype);
  Tensor output({ N, H }, otype);

  fillUniform(&input);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N*H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_act_cast(input.cpu_dptr<IType>(), ref_output.get(),
                        output.scale(), &ref_amax, N, H);

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

template <float (*ref_act)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTestGLU(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input({N, H * 2}, itype);
  Tensor output({N, H}, otype);

  fillUniform(&input);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N * H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_glu_act_cast<ref_act>(input.cpu_dptr<IType>(), ref_output.get(),
                                    output.scale(), &ref_amax, N, H);

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


class ActTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                transformer_engine::DType,
                                                                std::pair<size_t, size_t>>> {};

TEST_P(ActTestSuite, TestGELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<gelu, nvte_gelu, InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestSILU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<silu, nvte_silu, InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestRELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<relu, nvte_relu, InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<gelu, nvte_geglu, InputType,
                      OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestReGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<relu, nvte_reglu, InputType,
                      OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestSwiGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<silu, nvte_swiglu, InputType,
                      OutputType>(size.first, size.second);););
}

namespace {

std::vector<std::pair<size_t, size_t>> gelu_test_cases = {{2048, 12288},
                                                          {4096, 2048}, 
                                                          {768, 2816}, 
                                                          {128, 10240},
                                                          {768, 1024},
                                                          {256, 65536},
                                                          {65536, 128},
                                                          {256, 256},
                                                          {257, 259},
                                                          {128, 128+1}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    GELUTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::ValuesIn(gelu_test_cases)),
    [](const testing::TestParamInfo<SimpleActTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
