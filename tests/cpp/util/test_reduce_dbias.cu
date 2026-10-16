/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The dbias reduction of the MXFP8 quantize must add each column of the FP32 workspace in row
// order, starting from 0, and round once to the output type. This checks it bit for bit against
// that sum on the host, on data whose magnitudes span 2^-30..2^30 with mixed signs, so that any
// other order or grouping of the adds would change the result.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include "cast/core/grouped_layout.cuh"

namespace {

using transformer_engine::DType;

template <typename T>
T round_to(float x);
template <>
float round_to<float>(float x) {
  return x;
}
template <>
__nv_bfloat16 round_to<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}
template <>
__half round_to<__half>(float x) {
  return __float2half_rn(x);
}

template <typename T>
DType dtype_of();
template <>
DType dtype_of<float>() {
  return DType::kFloat32;
}
template <>
DType dtype_of<__nv_bfloat16>() {
  return DType::kBFloat16;
}
template <>
DType dtype_of<__half>() {
  return DType::kFloat16;
}

template <typename T>
void check(const size_t rows, const size_t cols, const uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> mantissa(1.0f, 2.0f);
  std::uniform_int_distribution<int> exponent(-30, 30);
  std::bernoulli_distribution negative(0.5);
  std::vector<float> workspace(rows * cols);
  for (float &w : workspace) {
    w = std::ldexp(mantissa(gen), exponent(gen)) * (negative(gen) ? -1.0f : 1.0f);
  }
  std::vector<T> expected(cols);
  for (size_t c = 0; c < cols; ++c) {
    float acc = 0.0f;
    for (size_t r = 0; r < rows; ++r) acc += workspace[r * cols + c];
    expected[c] = round_to<T>(acc);
  }

  float *workspace_dev = nullptr;
  T *dbias_dev = nullptr;
  ASSERT_EQ(cudaMalloc(&workspace_dev, workspace.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dbias_dev, cols * sizeof(T)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(workspace_dev, workspace.data(), workspace.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(dbias_dev, 0xA5, cols * sizeof(T)), cudaSuccess);
  transformer_engine::Tensor dbias;
  dbias.data = transformer_engine::SimpleTensor(dbias_dev, {cols}, dtype_of<T>());
  transformer_engine::dispatch::common::reduce_dbias<T>(workspace_dev, &dbias, rows, cols, 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<T> got(cols);
  ASSERT_EQ(cudaMemcpy(got.data(), dbias_dev, cols * sizeof(T), cudaMemcpyDeviceToHost),
            cudaSuccess);
  cudaFree(workspace_dev);
  cudaFree(dbias_dev);

  size_t mismatches = 0;
  for (size_t c = 0; c < cols; ++c) {
    if (std::memcmp(&got[c], &expected[c], sizeof(T)) != 0 && mismatches++ < 4) {
      ADD_FAILURE() << "column " << c << " of " << rows << "x" << cols << " differs";
    }
  }
  EXPECT_EQ(mismatches, 0u);
}

using Params = std::tuple<size_t /*rows*/, size_t /*cols*/>;

class ReduceDbiasTest : public ::testing::TestWithParam<Params> {};

TEST_P(ReduceDbiasTest, AddsEachColumnInRowOrder) {
  const auto [rows, cols] = GetParam();
  check<float>(rows, cols, 1);
  check<__nv_bfloat16>(rows, cols, 2);
  check<__half>(rows, cols, 3);
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, ReduceDbiasTest,
    // Rows: around the 32-row load batch and up to the workspace of 128K-row tensors. Columns:
    // multiples of 4 (the output vector of the BF16/FP16 store), around the 64-thread CTA.
    ::testing::Combine(::testing::Values(1, 2, 31, 32, 33, 63, 64, 65, 512, 1000, 2000),
                       ::testing::Values(4, 28, 60, 64, 68, 1028, 2052, 7168)),
    [](const testing::TestParamInfo<Params> &info) {
      return std::to_string(std::get<0>(info.param)) + "X" +
             std::to_string(std::get<1>(info.param));
    });

}  // namespace
