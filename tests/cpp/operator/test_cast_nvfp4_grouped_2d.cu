/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#include <gtest/gtest.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

class NVFP4Grouped2D : public ::testing::TestWithParam<int> {};

TEST_P(NVFP4Grouped2D, MatchesSingleTensorWithEmptyExpertAndCapacityTail) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) GTEST_SKIP();
  constexpr size_t capacity = 512, k = 128, count = 3;
  const bool rowwise = GetParam() != 2;
  const bool columnwise = GetParam() != 1;
  const std::vector<int64_t> rows{128, 0, 256};
  const std::vector<int64_t> offsets{0, 128 * k, 128 * k, 384 * k};
  Tensor source("source", std::vector<size_t>{capacity, k}, DType::kBFloat16);
  auto* host = source.rowwise_cpu_dptr<bf16>();
  for (size_t i = 0; i < capacity * k; ++i) {
    host[i] = bf16(i < 384 * k ? (static_cast<int>(i % 97) - 48) * 0.125f : 10000.0f);
  }
  source.from_cpu();
  auto device_rows = cuda_alloc<int64_t>(rows.size() * sizeof(int64_t));
  auto device_offsets = cuda_alloc<int64_t>(offsets.size() * sizeof(int64_t));
  NVTE_CHECK_CUDA(cudaMemcpy(device_rows.get(), rows.data(), rows.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
  NVTE_CHECK_CUDA(cudaMemcpy(device_offsets.get(), offsets.data(), offsets.size() * sizeof(int64_t),
                             cudaMemcpyHostToDevice));
  auto data_r = cuda_alloc<uint8_t>(capacity * k / 2);
  auto data_c = cuda_alloc<uint8_t>(capacity * k / 2);
  auto scale_r = cuda_alloc<uint8_t>(capacity * k / 16);
  auto scale_c = cuda_alloc<uint8_t>(capacity * k / 16);
  auto amax_r = cuda_alloc<float>(count * sizeof(float));
  auto amax_c = cuda_alloc<float>(count * sizeof(float));
  GroupedTensorWrapper input(count, std::vector<size_t>{capacity, k});
  input.set_rowwise_data(source.rowwise_dptr(), DType::kBFloat16, std::vector<size_t>{capacity, k});
  GroupedTensorWrapper output(count, std::vector<size_t>{capacity, k}, NVTE_NVFP4_1D_SCALING);
  output.set_first_dims(device_rows.get(), DType::kInt64, std::vector<size_t>{count});
  output.set_tensor_offsets(device_offsets.get(), DType::kInt64, std::vector<size_t>{count + 1});
  if (rowwise) {
    output.set_rowwise_data(data_r.get(), DType::kFloat4E2M1, std::vector<size_t>{capacity, k});
    output.set_rowwise_scale_inv(scale_r.get(), DType::kFloat8E4M3,
                                 std::vector<size_t>{capacity, k / 16});
    output.set_amax(amax_r.get(), DType::kFloat32, std::vector<size_t>{count});
  }
  if (columnwise) {
    output.set_columnwise_data(data_c.get(), DType::kFloat4E2M1, std::vector<size_t>{k, capacity});
    output.set_columnwise_scale_inv(scale_c.get(), DType::kFloat8E4M3,
                                    std::vector<size_t>{k, capacity / 16});
    output.set_columnwise_amax(amax_c.get(), DType::kFloat32, std::vector<size_t>{count});
  }
  QuantizationConfigWrapper config;
  config.set_nvfp4_2d_quantization(true);
  nvte_group_quantize(input.data(), output.data(), config, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  for (size_t i = 0; i < count; ++i) {
    if (rows[i] == 0) {
      float amax;
      if (rowwise) {
        NVTE_CHECK_CUDA(cudaMemcpy(&amax, amax_r.get() + i, sizeof(float), cudaMemcpyDeviceToHost));
        EXPECT_EQ(amax, 0.0f);
      }
      if (columnwise) {
        NVTE_CHECK_CUDA(cudaMemcpy(&amax, amax_c.get() + i, sizeof(float), cudaMemcpyDeviceToHost));
        EXPECT_EQ(amax, 0.0f);
      }
      continue;
    }
    const size_t m = rows[i];
    Tensor expert("expert", std::vector<size_t>{m, k}, DType::kBFloat16);
    std::copy_n(host + offsets[i], m * k, expert.rowwise_cpu_dptr<bf16>());
    expert.from_cpu();
    Tensor reference("reference", std::vector<size_t>{m, k}, DType::kFloat4E2M1, true, true,
                     NVTE_NVFP4_1D_SCALING);
    nvte_compute_amax(expert.data(), reference.data(), 0);
    nvte_quantize_v2(expert.data(), reference.data(), config, 0);
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    reference.to_cpu();
    for (bool col : {false, true}) {
      if (col ? !columnwise : !rowwise) continue;
      std::vector<uint8_t> actual(m * k / 2), scales(m * k / 16);
      NVTE_CHECK_CUDA(cudaMemcpy(actual.data(),
                                 (col ? data_c.get() : data_r.get()) + offsets[i] / 2,
                                 actual.size(), cudaMemcpyDeviceToHost));
      NVTE_CHECK_CUDA(cudaMemcpy(scales.data(),
                                 (col ? scale_c.get() : scale_r.get()) + offsets[i] / 16,
                                 scales.size(), cudaMemcpyDeviceToHost));
      const auto* expected =
          col ? reference.columnwise_cpu_dptr<fp4e2m1>() : reference.rowwise_cpu_dptr<fp4e2m1>();
      const auto* expected_scales = col ? reference.columnwise_cpu_scale_inv_ptr<fp8e4m3>()
                                        : reference.rowwise_cpu_scale_inv_ptr<fp8e4m3>();
      EXPECT_EQ(std::memcmp(actual.data(), expected, actual.size()), 0);
      EXPECT_EQ(std::memcmp(scales.data(), expected_scales, scales.size()), 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Directions, NVFP4Grouped2D, ::testing::Values(0, 1, 2));
