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

class SplitsToOffsetsTestSuite : public ::testing::TestWithParam<std::tuple<size_t, int64_t>> {};

TEST_P(SplitsToOffsetsTestSuite, TestSplitsToOffsets) {
  const size_t num_tensors = std::get<0>(GetParam());
  const int64_t logical_last_dim = std::get<1>(GetParam());

  std::vector<int64_t> h_first_dims(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    h_first_dims[i] = static_cast<int64_t>((i % 17) + 1);
  }

  std::vector<int64_t> h_expected(num_tensors + 1, 0);
  for (size_t i = 0; i < num_tensors; ++i) {
    h_expected[i + 1] = h_expected[i] + h_first_dims[i] * logical_last_dim;
  }

  std::vector<int64_t> h_output(num_tensors + 1, -1);

  int64_t *d_first_dims = nullptr;
  int64_t *d_output = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&d_first_dims, sizeof(int64_t) * num_tensors));
  NVTE_CHECK_CUDA(cudaMalloc(&d_output, sizeof(int64_t) * (num_tensors + 1)));
  NVTE_CHECK_CUDA(cudaMemcpy(d_first_dims, h_first_dims.data(), sizeof(int64_t) * num_tensors,
                             cudaMemcpyHostToDevice));

  nvte_splits_to_offsets(d_first_dims, d_output, num_tensors, logical_last_dim, 0 /* stream */);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  NVTE_CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, sizeof(int64_t) * (num_tensors + 1),
                             cudaMemcpyDeviceToHost));

  NVTE_CHECK_CUDA(cudaFree(d_first_dims));
  NVTE_CHECK_CUDA(cudaFree(d_output));

  for (size_t i = 0; i < h_output.size(); ++i) {
    EXPECT_EQ(h_output[i], h_expected[i])
        << "Mismatch at index " << i << ": expected " << h_expected[i] << ", got " << h_output[i];
  }
}

namespace {

std::vector<size_t> splits_to_offsets_num_tensors = {
    1,
    4,
    255,
    256,
    257,
    1024,
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, SplitsToOffsetsTestSuite,
    ::testing::Combine(::testing::ValuesIn(splits_to_offsets_num_tensors),
                       ::testing::Values(static_cast<int64_t>(1), static_cast<int64_t>(7),
                                         static_cast<int64_t>(128))),
    [](const testing::TestParamInfo<SplitsToOffsetsTestSuite::ParamType> &info) {
      std::string name = std::to_string(std::get<0>(info.param)) + "X" +
                         std::to_string(std::get<1>(info.param));
      return name;
    });
