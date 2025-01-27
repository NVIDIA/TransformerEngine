/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  Tensor input(data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor output(data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

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
