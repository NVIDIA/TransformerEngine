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

namespace {

// Allocate a device buffer holding `host` stored as `dtype` (int32 or int64).
void *copy_to_device(const std::vector<int64_t> &host, transformer_engine::DType dtype) {
  using namespace transformer_engine;
  NVTE_CHECK(dtype == DType::kInt32 || dtype == DType::kInt64,
             "splits_to_offsets test only supports int32/int64.");
  void *dptr = nullptr;
  if (dtype == DType::kInt32) {
    std::vector<int32_t> tmp(host.begin(), host.end());
    NVTE_CHECK_CUDA(cudaMalloc(&dptr, sizeof(int32_t) * tmp.size()));
    NVTE_CHECK_CUDA(
        cudaMemcpy(dptr, tmp.data(), sizeof(int32_t) * tmp.size(), cudaMemcpyHostToDevice));
  } else {
    NVTE_CHECK_CUDA(cudaMalloc(&dptr, sizeof(int64_t) * host.size()));
    NVTE_CHECK_CUDA(
        cudaMemcpy(dptr, host.data(), sizeof(int64_t) * host.size(), cudaMemcpyHostToDevice));
  }
  return dptr;
}

// Copy a device buffer of `n` `dtype` (int32 or int64) elements back to host as int64.
std::vector<int64_t> copy_to_host(const void *dptr, size_t n, transformer_engine::DType dtype) {
  using namespace transformer_engine;
  NVTE_CHECK(dtype == DType::kInt32 || dtype == DType::kInt64,
             "splits_to_offsets test only supports int32/int64.");
  std::vector<int64_t> out(n);
  if (dtype == DType::kInt32) {
    std::vector<int32_t> tmp(n);
    NVTE_CHECK_CUDA(cudaMemcpy(tmp.data(), dptr, sizeof(int32_t) * n, cudaMemcpyDeviceToHost));
    out.assign(tmp.begin(), tmp.end());
  } else {
    NVTE_CHECK_CUDA(cudaMemcpy(out.data(), dptr, sizeof(int64_t) * n, cudaMemcpyDeviceToHost));
  }
  return out;
}

}  // namespace

class SplitsToOffsets2DTestSuite
    : public ::testing::TestWithParam<std::tuple<size_t, transformer_engine::DType>> {};

TEST_P(SplitsToOffsets2DTestSuite, TestSplitsToOffsets2D) {
  using namespace transformer_engine;

  const size_t num_tensors = std::get<0>(GetParam());
  const DType dtype = std::get<1>(GetParam());

  // Generate per-tensor first/last dims. Vary both dimensions so the test
  // exercises the 2D prefix sum (offset[i+1] = sum_{j<=i} first_dims[j] * last_dims[j]).
  std::vector<int64_t> h_first_dims(num_tensors);
  std::vector<int64_t> h_last_dims(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    h_first_dims[i] = static_cast<int64_t>((i % 17) + 1);
    h_last_dims[i] = static_cast<int64_t>((i % 5) + 1) * 16;
  }

  std::vector<int64_t> h_expected(num_tensors + 1, 0);
  for (size_t i = 0; i < num_tensors; ++i) {
    h_expected[i + 1] = h_expected[i] + h_first_dims[i] * h_last_dims[i];
  }

  void *d_first_dims = copy_to_device(h_first_dims, dtype);
  void *d_last_dims = copy_to_device(h_last_dims, dtype);

  std::vector<int64_t> h_output_init(num_tensors + 1, -1);
  void *d_output = copy_to_device(h_output_init, dtype);

  TensorWrapper first_dims_w(d_first_dims, std::vector<size_t>{num_tensors}, dtype);
  TensorWrapper last_dims_w(d_last_dims, std::vector<size_t>{num_tensors}, dtype);
  TensorWrapper output_w(d_output, std::vector<size_t>{num_tensors + 1}, dtype);

  nvte_splits_to_offsets_2d(first_dims_w.data(), last_dims_w.data(), output_w.data(),
                            0 /* stream */);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int64_t> h_output = copy_to_host(d_output, num_tensors + 1, dtype);

  NVTE_CHECK_CUDA(cudaFree(d_first_dims));
  NVTE_CHECK_CUDA(cudaFree(d_last_dims));
  NVTE_CHECK_CUDA(cudaFree(d_output));

  for (size_t i = 0; i < h_output.size(); ++i) {
    EXPECT_EQ(h_output[i], h_expected[i])
        << "Mismatch at index " << i << ": expected " << h_expected[i] << ", got " << h_output[i];
  }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, SplitsToOffsets2DTestSuite,
    ::testing::Combine(::testing::ValuesIn(splits_to_offsets_num_tensors),
                       ::testing::Values(transformer_engine::DType::kInt32,
                                         transformer_engine::DType::kInt64)),
    [](const testing::TestParamInfo<SplitsToOffsets2DTestSuite::ParamType> &info) {
      std::string name =
          std::to_string(std::get<0>(info.param)) + "X" + test::typeName(std::get<1>(info.param));
      return name;
    });
