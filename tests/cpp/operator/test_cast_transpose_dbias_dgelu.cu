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

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/transpose.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename CType>
CType dgelu(const CType cval) {
    const CType tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
    return 0.5f * cval * ((1.f - tanh_out * tanh_out) *
                          (0.79788456f + 0.1070322243f * cval * cval)) +
           0.5f * (1.f + tanh_out);
}

template <typename IT, typename OT, typename CT>
void compute_ref_cast_transpose_dbias_dgelu(const IT *input,
                                            const IT *gelu_input,
                                            const CT scale,
                                            OT *output_c,
                                            OT *output_t,
                                            CT *amax_h,
                                            IT *dbias,
                                            const size_t N,
                                            const size_t H) {
  CT amax  = 0.;

  std::vector<CT> acc_dbias(H, 0.);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input[i * H + j]);
      const CT gelu_in = static_cast<CT>(gelu_input[i * H + j]);
      elt = dgelu(gelu_in) * elt;

      // update amax
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;

      output_c[i * H + j] = static_cast<OT>(scale * elt);
      output_t[j * N + i] = static_cast<OT>(scale * elt);

      // dbias
      acc_dbias[j] += elt;
    }
  }

  *amax_h = amax;

  for (size_t i = 0; i < H; i++) {
    dbias[i] = static_cast<IT>(acc_dbias[i]);
  }
}

template <typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;
  using CType = fp32;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;
  DType ctype = TypeInfo<CType>::dtype;

  Tensor input({N, H}, itype);
  Tensor gelu_input({N, H}, itype);

  Tensor output_c({N, H}, otype);
  Tensor output_t({ H, N}, otype);
  // dbias has the same data type with "output grad"
  Tensor dbias({H}, itype);

  fillUniform(&input);
  fillUniform(&gelu_input);
  setRandomScale(&output_c);
  output_t.shareFP8Meta(output_c);

  std::unique_ptr<OType[]> ref_output_c = std::make_unique<OType[]>(N*H);
  std::unique_ptr<OType[]> ref_output_t = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_output_dbias = std::make_unique<IType[]>(H);

  CType ref_amax;
  compute_ref_cast_transpose_dbias_dgelu(input.cpu_dptr<IType>(),
                                         gelu_input.cpu_dptr<IType>(),
                                         output_c.scale(),
                                         ref_output_c.get(),
                                         ref_output_t.get(),
                                         &ref_amax,
                                         ref_output_dbias.get(),
                                         N, H);

  Tensor workspace;

  nvte_cast_transpose_dbias_dgelu(input.data(),
                                  gelu_input.data(),
                                  output_c.data(),
                                  output_t.data(),
                                  dbias.data(),
                                  workspace.data(),
                                  0);

  workspace = Tensor(workspace.shape(), workspace.dtype());


  nvte_cast_transpose_dbias_dgelu(input.data(),
                                  gelu_input.data(),
                                  output_c.data(),
                                  output_t.data(),
                                  dbias.data(),
                                  workspace.data(),
                                  0);

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

  auto [atol_dbias, rtol_dbias] = getTolerances(itype);
  rtol_dbias *= 4;
  compareResults("output_dbias", dbias, ref_output_dbias.get(), atol_dbias, rtol_dbias);
}

std::vector<std::pair<size_t, size_t>> test_cases = {{64, 400},
                                                     {2048, 12288},
                                                     {768, 1024},
                                                     {256, 65536},
                                                     {65536, 128},
                                                     {256, 256}};

}  // namespace;


class CTDBiasDGeluTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                         transformer_engine::DType,
                                                                         std::pair<size_t,
                                                                                   size_t>>> {};

TEST_P(CTDBiasDGeluTestSuite, TestCTDBiasDgelu) {
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
    CTDBiasDGeluTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<CTDBiasDGeluTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
