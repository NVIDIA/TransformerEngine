/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/cast.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {
template <typename T>
void check_qdq(size_t rows, size_t cols) {
  cudaDeviceProp prop;
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major != 10 || prop.minor != 0) GTEST_SKIP() << "QDQ targets SM100";
  const auto dtype = TypeInfo<T>::dtype;
  const std::vector<size_t> shape{rows, cols};
  Tensor input("input", shape, dtype);
  Tensor packed("packed", shape, DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING,
                DType::kFloat8E4M3);
  Tensor reference("reference", shape, dtype);
  Tensor output("output", shape, dtype);
  Tensor amax("amax", std::vector<size_t>{1}, DType::kFloat32);
  Tensor noop("noop", std::vector<size_t>{1}, DType::kFloat32);
  fillCase<fp32>(&input, InputsFillCase::uniform);
  input.to_cpu();
  float maximum = 0;
  for (size_t i = 0; i < rows * cols; ++i) {
    maximum = std::max(maximum, std::abs(static_cast<float>(input.rowwise_cpu_dptr<T>()[i])));
  }
  packed.set_amax(maximum);
  ASSERT_EQ(cudaMemcpy(amax.rowwise_dptr(), &maximum, sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  QuantizationConfigWrapper config;
  nvte_quantize_v2(input.data(), packed.data(), config, 0);
  nvte_dequantize(packed.data(), reference.data(), 0);
  nvte_nvfp4_qdq(input.data(), output.data(), amax.data(), nullptr, 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  reference.to_cpu();
  output.to_cpu();
  EXPECT_EQ(std::memcmp(reference.rowwise_cpu_dptr<T>(), output.rowwise_cpu_dptr<T>(),
                        rows * cols * sizeof(T)), 0);
  const float one = 1;
  ASSERT_EQ(cudaMemcpy(noop.rowwise_dptr(), &one, sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(input.rowwise_dptr(), 0, rows * cols * sizeof(T)), cudaSuccess);
  nvte_nvfp4_qdq(input.data(), output.data(), amax.data(), noop.data(), 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  output.to_cpu();
  EXPECT_EQ(std::memcmp(reference.rowwise_cpu_dptr<T>(), output.rowwise_cpu_dptr<T>(),
                        rows * cols * sizeof(T)), 0);
}
}  // namespace

TEST(NVFP4QDQ, BF16) {
  check_qdq<bf16>(32, 32);
  check_qdq<bf16>(96, 160);
  check_qdq<bf16>(512, 1024);
}

TEST(NVFP4QDQ, FP16) {
  check_qdq<fp16>(32, 32);
  check_qdq<fp16>(96, 160);
}
