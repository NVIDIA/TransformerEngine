/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

std::vector<int64_t> reference_cumsum_with_leading_zero(const std::vector<int64_t> &input) {
  std::vector<int64_t> output(input.size() + 1, 0);
  for (size_t i = 0; i < input.size(); ++i) {
    output[i + 1] = output[i] + input[i];
  }
  return output;
}

void run_cumsum_test(const std::vector<int64_t> &h_input) {
  const size_t n = h_input.size();
  auto h_expected = reference_cumsum_with_leading_zero(h_input);
  std::vector<int64_t> h_output(n + 1, 0);

  int64_t *d_input = nullptr;
  int64_t *d_output = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(int64_t)));
  NVTE_CHECK_CUDA(cudaMalloc(&d_output, (n + 1) * sizeof(int64_t)));

  NVTE_CHECK_CUDA(
      cudaMemcpy(d_input, h_input.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
  nvte_cumsum(d_input, d_output, n, 0 /* stream */);
  NVTE_CHECK_CUDA(
      cudaMemcpy(h_output.data(), d_output, (n + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  NVTE_CHECK_CUDA(cudaFree(d_input));
  NVTE_CHECK_CUDA(cudaFree(d_output));

  ASSERT_EQ(h_output.size(), h_expected.size());
  for (size_t i = 0; i < h_output.size(); ++i) {
    EXPECT_EQ(h_output[i], h_expected[i]) << "Mismatch at output index " << i;
  }
}

std::vector<int64_t> make_input(size_t n) {
  std::vector<int64_t> input(n);
  for (size_t i = 0; i < n; ++i) {
    // Deterministic signed values in [-3, 3].
    input[i] = static_cast<int64_t>(i % 7) - 3;
  }
  return input;
}

std::vector<size_t> cumsum_test_sizes = {
    1,
    2,
    17,
    256,
    257,
    513,
    1024,
};

}  // namespace

TEST(CumsumTest, KnownValues) {
  const std::vector<int64_t> input = {3, -1, 4, 0, -5};
  run_cumsum_test(input);
}

class CumsumSizeTestSuite : public ::testing::TestWithParam<size_t> {};

TEST_P(CumsumSizeTestSuite, TestCumsumBySize) {
  run_cumsum_test(make_input(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CumsumSizeTestSuite, ::testing::ValuesIn(cumsum_test_sizes),
    [](const testing::TestParamInfo<CumsumSizeTestSuite::ParamType> &info) {
      return "N" + std::to_string(info.param);
    });
