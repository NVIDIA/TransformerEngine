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

#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename CType, typename IType>
inline CType gelu(const IType val) {
  CType cval = val;
  return cval * (0.5f + 0.5f * tanhf(cval * (0.79788456f + 0.03567741f * cval * cval)));
}

template <typename CType, typename IType>
inline CType dgelu(const IType val) {
  CType cval = val;
  const CType tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
  return 0.5f * cval * ((1.f - tanh_out * tanh_out) * (0.79788456f + 0.1070322243f * cval * cval)) +
         0.5f * (1.f + tanh_out);
}

template <typename IT, typename OT, typename CT>
void compute_ref_cast_transpose_dgated_gelu(const IT *grad_h, const IT *input_h, const CT scale,
                                            OT *output_c_h, OT *output_t_h, CT *amax_h,
                                            const size_t N, const size_t H) {
  CT amax = 0.;

  const size_t col = H * 2;
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT grad_elt = CT(grad_h[i * H + j]);
      CT gelu_elt = CT(input_h[i * col + j]);
      CT gate_elt = CT(input_h[i * col + H + j]);

      CT after_dgelu = dgelu<CT, CT>(gelu_elt) * grad_elt * gate_elt;
      CT after_dgate = grad_elt * gelu<CT, CT>(gelu_elt);

      amax = std::abs(after_dgelu) > amax ? std::abs(after_dgelu) : amax;
      amax = std::abs(after_dgate) > amax ? std::abs(after_dgate) : amax;

      output_c_h[i * col + j] = static_cast<OT>(scale * after_dgelu);
      output_c_h[i * col + H + j] = static_cast<OT>(scale * after_dgate);

      output_t_h[j * N + i] = static_cast<OT>(scale * after_dgelu);
      output_t_h[(j + H) * N + i] = static_cast<OT>(scale * after_dgate);
    }
  }

  *amax_h = amax;
}

template <typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;
  using CType = fp32;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor grad("grad", std::vector<size_t>{N, H}, itype);
  Tensor input("input", std::vector<size_t>{N, H * 2}, itype);
  Tensor output("output", std::vector<size_t>{N, H * 2}, otype, true, true);

  fillUniform(&grad);
  fillUniform(&input);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output_c = std::make_unique<OType[]>(N * H * 2);
  std::unique_ptr<OType[]> ref_output_t = std::make_unique<OType[]>(N * H * 2);

  nvte_dgeglu_cast_transpose(grad.data(), input.data(), output.data(), 0);

  CType ref_amax;
  compute_ref_cast_transpose_dgated_gelu(grad.rowwise_cpu_dptr<IType>(), input.rowwise_cpu_dptr<IType>(),
                                         output.scale(), ref_output_c.get(), ref_output_t.get(),
                                         &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / output.scale();
    compareResults("scale_inv", output.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }

  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output, ref_output_c.get(), true, atol, rtol);
  compareResults("output_t", output, ref_output_t.get(), false, atol, rtol);
}

std::vector<std::pair<size_t, size_t>> test_cases = {{64, 400},   {4096, 2048}, {768, 2816},
                                                     {256, 5120}, {128, 10240}, {256, 256}};

}  // namespace

class DGeGLUCTTestSuite
    : public ::testing::TestWithParam<std::tuple<
          transformer_engine::DType, transformer_engine::DType, std::pair<size_t, size_t>>> {};

TEST_P(DGeGLUCTTestSuite, TestDGeGLUCT) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType, performTest<InputType, OutputType>(size.first, size.second);););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, DGeGLUCTTestSuite,
    ::testing::Combine(::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(DType::kFloat8E5M2, DType::kFloat8E4M3),
                       ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<DGeGLUCTTestSuite::ParamType> &info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
