/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/swizzle.h>

#include <array>
#include <memory>
#include <vector>

#include "util/logging.h"

namespace {
using transformer_engine::DType;
using transformer_engine::TensorWrapper;

struct CudaDeleter {
  void operator()(uint8_t* ptr) const { cudaFree(ptr); }
};

struct SwizzleCase {
  std::unique_ptr<uint8_t, CudaDeleter> input_storage, output_storage;
  TensorWrapper input{NVTE_MXFP8_1D_SCALING}, output{NVTE_MXFP8_1D_SCALING};
  std::vector<uint8_t> before, expected;

  SwizzleCase(size_t rows, size_t cols, bool rowwise, bool compact) {
    const size_t m = rowwise ? rows : cols;
    const size_t k = ((rowwise ? cols : rows) + 31) / 32;
    const size_t padded_m = (m + 127) / 128 * 128;
    const size_t padded_k = (k + 3) / 4 * 4;
    const size_t input_m = compact ? m : padded_m;
    const size_t input_k = compact ? k : padded_k;
    uint8_t* allocation = nullptr;
    NVTE_CHECK_CUDA(cudaMalloc(&allocation, input_m * input_k));
    input_storage.reset(allocation);
    NVTE_CHECK_CUDA(cudaMalloc(&allocation, padded_m * padded_k));
    output_storage.reset(allocation);
    before.resize(input_m * input_k);
    expected.resize(padded_m * padded_k, 0);
    for (size_t i = 0; i < before.size(); ++i) before[i] = (i * 17 + i / 7 + 1) % 256;
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        // Independent forward mapping from logical scales to the cuBLAS layout.
        const size_t dst = (i / 128) * 128 * padded_k + (j / 4) * 512 + (i % 32) * 16 +
                           ((i % 128) / 32) * 4 + j % 4;
        expected[dst] = before[rowwise ? i * input_k + j : j * input_m + i];
      }
    }
    NVTE_CHECK_CUDA(
        cudaMemcpy(input_storage.get(), before.data(), before.size(), cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemset(output_storage.get(), 0xff, expected.size()));
    const std::vector<size_t> data_shape{rows, cols};
    const std::vector<size_t> input_shape =
        rowwise ? std::vector<size_t>{input_m, input_k} : std::vector<size_t>{input_k, input_m};
    const std::vector<size_t> output_shape =
        rowwise ? std::vector<size_t>{padded_m, padded_k} : std::vector<size_t>{padded_k, padded_m};
    if (rowwise) {
      input.set_rowwise_data(nullptr, DType::kFloat8E4M3, data_shape);
      output.set_rowwise_data(nullptr, DType::kFloat8E4M3, data_shape);
      input.set_rowwise_scale_inv(input_storage.get(), DType::kFloat8E8M0, input_shape);
      output.set_rowwise_scale_inv(output_storage.get(), DType::kFloat8E8M0, output_shape);
    } else {
      input.set_columnwise_data(nullptr, DType::kFloat8E4M3, data_shape);
      output.set_columnwise_data(nullptr, DType::kFloat8E4M3, data_shape);
      input.set_columnwise_scale_inv(input_storage.get(), DType::kFloat8E8M0, input_shape);
      output.set_columnwise_scale_inv(output_storage.get(), DType::kFloat8E8M0, output_shape);
    }
    output.set_with_gemm_swizzled_scales(true);
  }

  void check() {
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<uint8_t> actual(expected.size()), after(before.size());
    NVTE_CHECK_CUDA(
        cudaMemcpy(actual.data(), output_storage.get(), actual.size(), cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(
        cudaMemcpy(after.data(), input_storage.get(), after.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(after, before);
    EXPECT_EQ(actual, expected);
    EXPECT_FALSE(input.get_with_gemm_swizzled_scales());
  }
};

class CompactSwizzleTest : public ::testing::TestWithParam<bool> {};

TEST_P(CompactSwizzleTest, SingleTensor) {
  for (const auto shape : std::vector<std::array<size_t, 2>>{{1, 1},
                                                             {31, 33},
                                                             {32, 32},
                                                             {64, 64},
                                                             {96, 160},
                                                             {160, 96},
                                                             {128, 32},
                                                             {32, 128},
                                                             {256, 128},
                                                             {129, 4097},
                                                             {4096, 4128}}) {
    SCOPED_TRACE(::testing::Message() << shape[0] << "x" << shape[1]);
    SwizzleCase test(shape[0], shape[1], GetParam(), true);
    nvte_swizzle_scaling_factors(test.input.data(), test.output.data(), nullptr);
    test.check();
  }
}

TEST_P(CompactSwizzleTest, MixedBatch) {
  // More compact tensors than fit in two launch argument buffers, interleaved
  // with padded tensors. Each compact allocation ends at its last scale byte.
  std::vector<std::unique_ptr<SwizzleCase>> cases;
  std::vector<NVTETensor> inputs, outputs;
  for (int i = 0; i < 100; ++i) {
    cases.emplace_back(
        std::make_unique<SwizzleCase>(32 * (i % 3 + 1), 32 * (i % 5 + 1), GetParam(), i % 4 != 0));
    inputs.push_back(cases.back()->input.data());
    outputs.push_back(cases.back()->output.data());
  }
  nvte_multi_tensor_swizzle_scaling_factors(inputs.data(), outputs.data(), inputs.size(), nullptr);
  for (auto& test : cases) test->check();
}

TEST_P(CompactSwizzleTest, TallTensor) {
  // A 2D launch with one tile per blockIdx.y would exceed CUDA's 65535 limit.
  constexpr size_t m = 128 * 65536 + 32;
  SwizzleCase test(GetParam() ? m : 32, GetParam() ? 32 : m, GetParam(), true);
  nvte_swizzle_scaling_factors(test.input.data(), test.output.data(), nullptr);
  test.check();
}

TEST_P(CompactSwizzleTest, RejectInvalidScaleDtype) {
  SwizzleCase test(64, 64, GetParam(), true);
  const auto shape = GetParam() ? std::vector<size_t>{64, 2} : std::vector<size_t>{2, 64};
  if (GetParam())
    test.input.set_rowwise_scale_inv(test.input_storage.get(), DType::kFloat32, shape);
  else
    test.input.set_columnwise_scale_inv(test.input_storage.get(), DType::kFloat32, shape);
  EXPECT_THROW(nvte_swizzle_scaling_factors(test.input.data(), test.output.data(), nullptr),
               std::runtime_error);
}

TEST_P(CompactSwizzleTest, RejectInvalidShapes) {
  for (bool grouped : {false, true}) {
    for (bool invalid_output : {false, true}) {
      SwizzleCase test(64, 64, GetParam(), true);
      auto& tensor = invalid_output ? test.output : test.input;
      const auto shape = GetParam() ? std::vector<size_t>{64, 3} : std::vector<size_t>{3, 64};
      auto* ptr = invalid_output ? test.output_storage.get() : test.input_storage.get();
      if (GetParam())
        tensor.set_rowwise_scale_inv(ptr, DType::kFloat8E8M0, shape);
      else
        tensor.set_columnwise_scale_inv(ptr, DType::kFloat8E8M0, shape);
      if (grouped) {
        NVTETensor input = test.input.data(), output = test.output.data();
        EXPECT_THROW(nvte_multi_tensor_swizzle_scaling_factors(&input, &output, 1, nullptr),
                     std::runtime_error);
      } else {
        EXPECT_THROW(nvte_swizzle_scaling_factors(test.input.data(), test.output.data(), nullptr),
                     std::runtime_error);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, CompactSwizzleTest, ::testing::Bool());
}  // namespace
