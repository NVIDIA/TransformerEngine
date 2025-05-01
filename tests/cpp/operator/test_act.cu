/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_act_cast(const IT *input_h,
                          OT *output_h,
                          const CT scale,
                          CT *amax_h,
                          const size_t N,
                          const size_t H) {
  CT amax  = 0.;

  #pragma omp parallel for schedule(static) reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);
      elt = act(elt);
      output_h[i * H + j] = static_cast<OT>(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*dact)(const float), typename IT, typename OT>
void compute_ref_dact_cast(const IT *input_h,
                           const IT *grad_h,
                           OT *output_h,
                           const size_t N,
                           const size_t H) {
  using CT = float;
  #pragma omp parallel for schedule(static) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);
      elt = dact(elt);
      CT grad = static_cast<CT>(grad_h[i * H + j]);
      output_h[i * H + j] = static_cast<OT>(grad * elt);
    }
  }
}

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_glu_act_cast(const IT *input_h, OT *output_h, const CT scale, CT *amax_h,
                              const size_t N, const size_t H) {
  CT amax = 0.;

  const int col = H * 2;

  #pragma omp parallel for schedule(static) reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT gelu_elt = static_cast<CT>(input_h[i * col + j]);
      gelu_elt = act(gelu_elt);
      CT gate_elt = static_cast<CT>(input_h[i * col + H + j]);
      CT elt = gelu_elt * gate_elt;
      output_h[i * H + j] = static_cast<OT>(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*dact)(const float), float (*act)(const float),
          typename IT, typename OT>
void compute_ref_dglu_act_cast(const IT *input_h, const IT *grad_h, OT *output_h,
                               const size_t N, const size_t H) {
  const int col = H * 2;
  using CT = float;

  #pragma omp parallel for schedule(static) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT grad = static_cast<CT>(grad_h[i * H + j]);
      CT gelu_elt = static_cast<CT>(input_h[i * col + j]);
      CT gate_elt = static_cast<CT>(input_h[i * col + H + j]);
      output_h[i * col + H + j] = static_cast<OT>(grad * act(gelu_elt));
      gelu_elt = dact(gelu_elt);
      CT elt = gelu_elt * gate_elt;
      output_h[i * col + j] = static_cast<OT>(grad * elt);
    }
  }
}


template <float (*ref_act)(const float),
          float (*ref_dact)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
          void (*nvte_dact)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input("input", std::vector<size_t>{ N, H }, itype);
  Tensor output("output", std::vector<size_t>{ N, H }, otype);
  Tensor igrad("igrad", std::vector<size_t>{ N, H }, itype);
  Tensor ograd("ograd", std::vector<size_t>{ N, H }, itype);

  fillUniform(&input);
  fillUniform(&ograd);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_igrad = std::make_unique<IType[]>(N*H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_act_cast<ref_act>(input.rowwise_cpu_dptr<IType>(), ref_output.get(),
                                output.scale(), &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_act", output, ref_output.get(), atol, rtol);

  nvte_dact(ograd.data(), input.data(), igrad.data(), 0);

  compute_ref_dact_cast<ref_dact>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                  ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);
  }
}

template <float (*ref_act)(const float),
          float (*ref_dact)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
          void (*nvte_dact)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTestGLU(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input("input", std::vector<size_t>{N, H * 2}, itype);
  Tensor output("output", std::vector<size_t>{N, H}, otype);
  Tensor igrad("igrad", std::vector<size_t>{ N, H * 2 }, itype);
  Tensor ograd("ograd", std::vector<size_t>{ N, H }, itype);

  fillUniform(&input);
  fillUniform(&ograd);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N * H);
  std::unique_ptr<IType[]> ref_igrad = std::make_unique<IType[]>(2 * N * H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_glu_act_cast<ref_act>(input.rowwise_cpu_dptr<IType>(), ref_output.get(),
                                    output.scale(), &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol, rtol] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol, rtol);
    if (output.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
      const float ref_scale = 1.f / output.scale();
      compareResults("scale_inv", *output.rowwise_cpu_scale_inv_ptr<float>(), ref_scale, atol, rtol);
    }
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_gelu", output, ref_output.get(), atol, rtol);

  nvte_dact(ograd.data(), input.data(), igrad.data(), 0);

  compute_ref_dglu_act_cast<ref_dact, ref_act>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                               ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);
  }
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
        performTest<gelu, dgelu, nvte_gelu, nvte_dgelu,
                    InputType, OutputType>(size.first, size.second);
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
        performTest<silu, dsilu, nvte_silu, nvte_dsilu,
                    InputType, OutputType>(size.first, size.second);
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
        performTest<relu, drelu, nvte_relu, nvte_drelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestQGELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<qgelu, dqgelu, nvte_qgelu, nvte_dqgelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestSRELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<srelu, dsrelu, nvte_srelu, nvte_dsrelu,
                    InputType, OutputType>(size.first, size.second);
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
          performTestGLU<gelu, dgelu, nvte_geglu, nvte_dgeglu, InputType,
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
          performTestGLU<relu, drelu, nvte_reglu, nvte_dreglu, InputType,
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
          performTestGLU<silu, dsilu, nvte_swiglu, nvte_dswiglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestQGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<qgelu, dqgelu, nvte_qgeglu, nvte_dqgeglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestSReGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<srelu, dsrelu, nvte_sreglu, nvte_dsreglu, InputType,
                         OutputType>(size.first, size.second);););
}

namespace {

std::vector<std::pair<size_t, size_t>> act_test_cases = {{2048, 12288},
                                                         {768, 2816},
                                                         {256, 65536},
                                                         {65536, 128},
                                                         {256, 256},
                                                         {257, 259},
                                                         {128, 128+1}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    ActTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::ValuesIn(act_test_cases)),
    [](const testing::TestParamInfo<ActTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
