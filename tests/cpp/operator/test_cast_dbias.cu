/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename IT, typename OT, typename CT>
void compute_ref_cast_dbias(const IT *input_h,
                            const CT scale,
                            OT *output_c_h,
                            CT *amax_h,
                            IT *dbias_h,
                            const size_t N,
                            const size_t H) {
  CT amax  = 0.;

  std::vector<CT> acc_dbias(H, 0.);

  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);

      // update amax
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;

      output_c_h[i * H + j] = static_cast<OT>(scale * elt);

      // dbias
      acc_dbias[j] += elt;
    }
  }

  *amax_h = amax;

  for (size_t i = 0; i < H; i++) {
    dbias_h[i] = static_cast<IT>(acc_dbias[i]);
  }
}

template <typename IType, typename OType>
void performTest(const std::vector<size_t>& shape) {
  using namespace test;
  using CType = fp32;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  const size_t N = first_dimension(shape);
  const size_t H = last_dimension(shape);

  Tensor input("input", shape, itype);

  Tensor output_c("output_c", shape, otype);
  // dbias has the same data type with "output grad"
  Tensor dbias("dbias", std::vector<size_t>{H}, itype);

  fillUniform(&input);
  setRandomScale(&output_c);
  const float ref_scale = isFp8Type(otype) ? output_c.scale() : 1.0f;

  std::unique_ptr<OType[]> ref_output_c = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_output_dbias = std::make_unique<IType[]>(H);

  CType ref_amax;
  compute_ref_cast_dbias(input.rowwise_cpu_dptr<IType>(),
                         ref_scale,
                         ref_output_c.get(),
                         &ref_amax,
                         ref_output_dbias.get(),
                         N, H);

  Tensor workspace;

  nvte_quantize_dbias(input.data(),
                      output_c.data(),
                      dbias.data(),
                      workspace.data(),
                      0);

  workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

  nvte_quantize_dbias(input.data(),
                      output_c.data(),
                      dbias.data(),
                      workspace.data(),
                      0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (isFp8Type(otype)) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output_c.amax(), ref_amax, atol_amax, rtol_amax);
    float ref_scale_inv = 1.f / ref_scale;
    compareResults("scale_inv", output_c.rowwise_scale_inv(), ref_scale_inv, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_c", output_c, ref_output_c.get(), true, atol, rtol);

  auto [atol_dbias, rtol_dbias] = getTolerances(itype);
  rtol_dbias *= 4;
  compareResults("output_dbias", dbias, ref_output_dbias.get(), true, atol_dbias, rtol_dbias);
}

std::vector<std::vector<size_t>> test_cases = {
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

// The FP8 quantize sums each 128-row tile of a column into an FP32 workspace row, then adds those
// rows in row order, starting from 0, and rounds once to the input type. With one nonzero row per
// tile, each tile sum is exact, so this checks dbias bit for bit against the sum of those rows in
// order. Their magnitudes span many binades with mixed signs, so for FP32 any other order or
// grouping of the adds changes some sums; BF16 and FP16 mostly check the rounding.
template <typename IType>
void performTestDBiasOrder(const size_t tiles, const size_t cols, const uint32_t seed,
                           const int min_exponent, const int max_exponent) {
  using namespace test;
  constexpr size_t tile_rows = 128;                 // FP8_CHUNK_DIM_Y
  const size_t rows = (tiles - 1) * tile_rows + 1;  // The last tile holds one row.
  const DType itype = TypeInfo<IType>::dtype;

  Tensor input("input", std::vector<size_t>{rows, cols}, itype);
  Tensor output("output", std::vector<size_t>{rows, cols}, DType::kFloat8E4M3);
  Tensor dbias("dbias", std::vector<size_t>{cols}, itype);
  setRandomScale(&output);

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> mantissa(1.0f, 2.0f);
  std::uniform_int_distribution<int> exponent(min_exponent, max_exponent);
  std::bernoulli_distribution negative(0.5);
  IType *const input_h = input.rowwise_cpu_dptr<IType>();
  std::vector<float> ref_dbias(cols, 0.0f);
  for (size_t r = 0; r < rows; r += tile_rows) {
    for (size_t c = 0; c < cols; ++c) {
      const float m = mantissa(gen);
      const int e = exponent(gen);
      const IType x = static_cast<IType>(std::ldexp(negative(gen) ? -m : m, e));
      input_h[r * cols + c] = x;
      ref_dbias[c] += static_cast<float>(x);
    }
  }
  input.from_cpu();

  Tensor workspace;
  nvte_quantize_dbias(input.data(), output.data(), dbias.data(), workspace.data(), 0);
  workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
  nvte_quantize_dbias(input.data(), output.data(), dbias.data(), workspace.data(), 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  dbias.to_cpu();
  const IType *const dbias_h = dbias.rowwise_cpu_dptr<IType>();
  size_t mismatches = 0;
  for (size_t c = 0; c < cols; ++c) {
    const IType expected = static_cast<IType>(ref_dbias[c]);
    if (std::memcmp(&dbias_h[c], &expected, sizeof(IType)) != 0 && mismatches++ < 4) {
      ADD_FAILURE() << typeName(itype) << " column " << c << " of " << rows << "x" << cols
                    << ": got " << static_cast<float>(dbias_h[c]) << ", expected "
                    << static_cast<float>(expected);
    }
  }
  EXPECT_EQ(mismatches, 0u);
}

}  // namespace;


class CastDBiasTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                      transformer_engine::DType,
                                                                      std::vector<size_t>>> {};

TEST_P(CastDBiasTestSuite, TestCastDBias) {
    using namespace transformer_engine;
    using namespace test;
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

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
    CastDBiasTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(test_cases)),
    [](const testing::TestParamInfo<CastDBiasTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
      test::typeName(std::get<1>(info.param));
      const auto& shape = std::get<2>(info.param);
      for ( const auto& s: shape) {
        name += "X" + std::to_string(s);
      }
      return name;
    });

class CastDBiasOrderTestSuite : public ::testing::TestWithParam<std::tuple<size_t, size_t>> {};

TEST_P(CastDBiasOrderTestSuite, TestCastDBiasOrder) {
  using namespace test;
  // Skip tests for pre-Blackwell architectures
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  const auto [tiles, cols] = GetParam();
  performTestDBiasOrder<fp32>(tiles, cols, 1, -30, 30);
  performTestDBiasOrder<bf16>(tiles, cols, 2, -30, 30);
  // Most FP16 sums stay finite; some overflow.
  performTestDBiasOrder<fp16>(tiles, cols, 3, -24, 12);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CastDBiasOrderTestSuite,
    // Tiles: around the 32-row stages of the dbias reduction and its 4-stage pipeline, up to 128K
    // rows. Columns: multiples of 16, as TMA needs, around its 64-thread CTA.
    ::testing::Combine(::testing::Values(1, 2, 31, 32, 33, 63, 64, 65, 97, 128, 129, 512, 1000),
                       ::testing::Values(16, 48, 64, 80, 1040)),
    [](const testing::TestParamInfo<CastDBiasOrderTestSuite::ParamType> &info) {
      return std::to_string(std::get<0>(info.param)) + "X" +
             std::to_string(std::get<1>(info.param));
    });
