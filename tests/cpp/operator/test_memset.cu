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

#include <transformer_engine/transformer_engine.h>
#include "../test_common.h"

using namespace transformer_engine;


class MemsetTestSuite : public ::testing::TestWithParam<std::tuple<int,
                                                                size_t>> {};

TEST_P(MemsetTestSuite, TestMemset) {
    using namespace transformer_engine;
    using namespace test;

    int value = std::get<0>(GetParam());
    size_t size_in_bytes = std::get<1>(GetParam());

    std::vector<uint8_t> h_buffer{};
    h_buffer.resize(size_in_bytes);
    for (size_t i = 0; i < size_in_bytes; ++i) {
        h_buffer[i] = value + 1;  // Initialize host buffer to a different value than memset value to verify memset is working correctly
    }

    char* d_ptr;
    NVTE_CHECK_CUDA(cudaMalloc(&d_ptr, size_in_bytes));

    NVTE_CHECK_CUDA(cudaMemcpy(d_ptr, h_buffer.data(), size_in_bytes, cudaMemcpyHostToDevice));

    nvte_memset(d_ptr, value, size_in_bytes, 0 /* stream */);

    NVTE_CHECK_CUDA(cudaMemcpy(
        h_buffer.data(), d_ptr, size_in_bytes, cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaFree(d_ptr));

    NVTE_CHECK_CUDA(cudaDeviceSynchronize());

    for (size_t i = 0; i < size_in_bytes; ++i) {
        EXPECT_EQ(h_buffer[i], static_cast<uint8_t>(value))
            << "Mismatch at index " << i << ": expected " << static_cast<int>(value)
            << ", got " << static_cast<int>(h_buffer[i]);
    }
}

namespace {

std::vector<size_t> memset_test_sizes = {
  1,
  4,
  9,
  16,
  128,
  4096,
  4097,
  8192,
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    MemsetTestSuite,
    ::testing::Combine(
        ::testing::Values(0, 6),
        ::testing::ValuesIn(memset_test_sizes)),
    [](const testing::TestParamInfo<MemsetTestSuite::ParamType>& info) {
      std::string name = std::to_string(std::get<0>(info.param)) + "X" +
                         std::to_string(std::get<1>(info.param));
      return name;
    });
