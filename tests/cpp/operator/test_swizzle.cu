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
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/swizzle.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;

constexpr int MAT_TILE_DIM_M = 128;
constexpr int MAT_TILE_DIM_K = 128;

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_swizzle(const uint8_t *h_input, uint8_t *h_output,
                         const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_output_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int out_index = tile_output_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[out_index] = h_input[k + m * K];
      else
        h_output[out_index] = h_input[k * M + m];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_unswizzle(const uint8_t *h_input, uint8_t *h_output,
                           const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_input_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int in_index = tile_input_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[k + m * K] = h_input[in_index];
      else
        h_output[k * M + m] = h_input[in_index];
    }
  }
}

void performTestSwizzle1D(const int num_tiles_M, const int num_tiles_K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if ((rowwise && columnwise) || !(rowwise || columnwise)){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + "is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const size_t M = num_tiles_M * MAT_TILE_DIM_M;
  const size_t K = num_tiles_K * MAT_TILE_DIM_K;
  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  const auto scale_shape = std::vector<size_t>{data_shape[0] / SF_MODE_X, data_shape[1] /SF_MODE_Y};

  std::vector<int> scaling_mode = {SF_MODE_X, SF_MODE_Y, 0};
  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  output.set_with_gemm_swizzled_scales(true);

  fillUniform(&input);

  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(scale_shape[0] * scale_shape[1]);

  nvte_swizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_swizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0], scale_shape[1]);
  else
    compute_ref_swizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[1], scale_shape[0]);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_swizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  } else {
    compareResults("output_swizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  }
}

void performTestUnswizzle1D(const size_t M, const size_t K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if (!rowwise && !columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, Either rowwise or columnwise scaling mode must be true.";
  }
  if (rowwise && columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + " is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  input.set_with_gemm_swizzled_scales(true);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  // Use the actual padded compact scale shape from the tensor for both the reference
  // and the comparison. This correctly covers padded cases where M is not a multiple
  // of 128 or K/32 is not a multiple of 4.
  const auto padded_scale_shape = rowwise
    ? input.rowwise_scale_inv_shape()
    : input.columnwise_scale_inv_shape();
  const size_t padded_dim0 = padded_scale_shape.data[0];
  const size_t padded_dim1 = padded_scale_shape.data[1];
  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(padded_dim0 * padded_dim1);

  nvte_unswizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_unswizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0, padded_dim1);
  else
    compute_ref_unswizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim1, padded_dim0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_unswizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0 * padded_dim1);
  } else {
    compareResults("output_unswizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0 * padded_dim1);
  }
}

class SwizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<int, int>, std::pair<bool, bool>, bool>> {};


TEST_P(SwizzleTestSuite, TestSwizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto num_tiles = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzle1D(num_tiles.first, num_tiles.second,
                       scaling_mode.first, scaling_mode.second,
                       transa);
}

class UnswizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, std::pair<bool, bool>, bool>> {};

TEST_P(UnswizzleTestSuite, TestUnswizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto data_shape = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestUnswizzle1D(data_shape.first, data_shape.second,
                         scaling_mode.first, scaling_mode.second,
                         transa);
}

namespace {

std::vector<std::pair<int, int>> num_tiles = {
  {1, 1},
  {1, 132},
  {132, 1},
  {65, 256},
  {65, 257},
  {65, 258},
  {65, 259},
};

// Raw {M, K} data shapes for unswizzle tests. Includes aligned cases (scale dims
// already multiples of 128 and 4) and padded cases where M or K/32 are not yet
// aligned, forcing the compact scale_inv to carry a padded tail.
// All K values must be multiples of 32 (MXFP8 block size).
std::vector<std::pair<size_t, size_t>> unswizzle_data_shapes = {
  // Aligned: scale dims are already multiples of 128 and 4
  {128, 128},
  {128, 16896},   // K = 132 * 128, large K
  {16896, 128},   // M = 132 * 128, large M
  // M-padding only: M not a multiple of 128 (scale-M needs padding to 256)
  {160, 128},
  // scale-K padding only: K/32 = 3, padded to 4
  {128, 96},
  // Both M and scale-K need padding
  {160, 96},
};

std::vector<std::pair<bool, bool>> scaling_mode = {
  {true, false},
  {false, true}
};

std::vector<bool> transa = {true, false};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(num_tiles),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleTestSuite::ParamType>& info) {
    std::string name = "ntiles" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  UnswizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(unswizzle_data_shapes),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<UnswizzleTestSuite::ParamType>& info) {
    std::string name = "MK" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

void performTestSwizzleUnswizzleRoundtrip(const int num_tiles_M, const int num_tiles_K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if (!rowwise && !columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, Either rowwise or columnwise scaling mode must be true.";
  }
  if (rowwise && columnwise){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + " is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const size_t M = num_tiles_M * MAT_TILE_DIM_M;
  const size_t K = num_tiles_K * MAT_TILE_DIM_K;
  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  const auto scale_shape = std::vector<size_t>{data_shape[0] / SF_MODE_X, data_shape[1] /SF_MODE_Y};

  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor swizzled("swizzled", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  swizzled.set_with_gemm_swizzled_scales(true);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  nvte_swizzle_scaling_factors(input.data(), swizzled.data(), 0);
  nvte_unswizzle_scaling_factors(swizzled.data(), output.data(), 0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  input.to_cpu();
  output.to_cpu();
  if (rowwise) {
    compareResults("roundtrip_rowwise", output.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                   input.rowwise_cpu_scale_inv_ptr<uint8_t>(), scale_shape[0] * scale_shape[1]);
  } else {
    compareResults("roundtrip_columnwise", output.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                   input.columnwise_cpu_scale_inv_ptr<uint8_t>(), scale_shape[0] * scale_shape[1]);
  }
}

class SwizzleUnswizzleRoundtripTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<int, int>, std::pair<bool, bool>, bool>> {};

TEST_P(SwizzleUnswizzleRoundtripTestSuite, TestSwizzleUnswizzleRoundtrip) {
  using namespace transformer_engine;
  using namespace test;

  const auto num_tiles = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzleUnswizzleRoundtrip(num_tiles.first, num_tiles.second,
                                       scaling_mode.first, scaling_mode.second,
                                       transa);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleUnswizzleRoundtripTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(num_tiles),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleUnswizzleRoundtripTestSuite::ParamType>& info) {
    std::string name = "roundtrip_ntiles" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });
