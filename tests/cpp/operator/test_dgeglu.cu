/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
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
void compute_ref_dgeglu(const IT *grad_h, const IT *input_h, OT *output_h, const size_t N,
                        const size_t H) {
  const size_t col = H * 2;

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT grad_elt = CT(grad_h[i * H + j]);
      CT gelu_elt = CT(input_h[i * col + j]);
      CT gate_elt = CT(input_h[i * col + H + j]);

      CT after_dgelu = dgelu<CT, CT>(gelu_elt) * grad_elt * gate_elt;
      CT after_dgate = grad_elt * gelu<CT, CT>(gelu_elt);

      output_h[i * col + j] = OT(after_dgelu);
      output_h[i * col + H + j] = OT(after_dgate);
    }
  }
}

template <typename IType, typename OType>
void performTestDGeGLU(const size_t N, const size_t H) {
  using namespace test;

  using CType = fp32;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor grad({N, H}, itype);
  Tensor input({N, H * 2}, itype);
  Tensor output({N, H * 2}, otype);

  fillUniform(&grad);
  fillUniform(&input);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N * H * 2);

  nvte_dgeglu(grad.data(), input.data(), output.data(), 0);

  compute_ref_dgeglu<IType, OType, CType>(grad.cpu_dptr<IType>(), input.cpu_dptr<IType>(),
                                          ref_output.get(), N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_dgelu", output, ref_output.get(), atol, rtol);
}

std::vector<std::pair<size_t, size_t>> test_cases = {
    {4096, 2048}, {768, 2816}, {256, 5120}, {128, 10240}, {256, 256}, {257, 259}, {128, 128 + 1}};

}  // namespace

class DGeGLUTestSuite
    : public ::testing::TestWithParam<std::tuple<
          transformer_engine::DType, transformer_engine::DType, std::pair<size_t, size_t>>> {};

TEST_P(DGeGLUTestSuite, TestDGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestDGeGLU<InputType, OutputType>(size.first, size.second);););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, DGeGLUTestSuite,
    ::testing::Combine(::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<DGeGLUTestSuite::ParamType> &info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
