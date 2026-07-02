/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <transformer_engine/softmax.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

template <typename Type>
void ref_softmax_row(Type *out, const Type *in, const uint8_t *mask, int cols, float scale) {
  float max_value = -10000.0f;
  bool has_unmasked = false;
  for (int j = 0; j < cols; ++j) {
    if (mask != nullptr && mask[j] == 1) continue;
    max_value = std::max(max_value, static_cast<float>(in[j]) * scale);
    has_unmasked = true;
  }
  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    if (mask != nullptr && mask[j] == 1) {
      out[j] = static_cast<Type>(0.0f);
      continue;
    }
    const float val = has_unmasked ? std::exp(static_cast<float>(in[j]) * scale - max_value) : 0.0f;
    sum += val;
    out[j] = static_cast<Type>(val);
  }
  for (int j = 0; j < cols; ++j) {
    out[j] = static_cast<Type>(static_cast<float>(out[j]) / sum);
  }
}

template <typename Type>
void ref_softmax_bwd(Type *grad_in, const Type *grad, const Type *softmax, int cols, float scale) {
  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    sum += static_cast<float>(grad[j]) * static_cast<float>(softmax[j]);
  }
  for (int j = 0; j < cols; ++j) {
    grad_in[j] =
        static_cast<Type>(scale * (static_cast<float>(grad[j]) - sum) *
                          static_cast<float>(softmax[j]));
  }
}

template <typename Type>
void ref_upper_row(Type *out, const Type *in, int row, int cols, float scale) {
  float max_value = -10000.0f;
  for (int j = 0; j <= row; ++j) {
    max_value = std::max(max_value, static_cast<float>(in[j]) * scale);
  }
  float sum = 0.0f;
  for (int j = 0; j < cols; ++j) {
    if (j <= row) {
      const float val = std::exp(static_cast<float>(in[j]) * scale - max_value);
      sum += val;
      out[j] = static_cast<Type>(val);
    } else {
      out[j] = static_cast<Type>(0.0f);
    }
  }
  for (int j = 0; j <= row; ++j) {
    out[j] = static_cast<Type>(static_cast<float>(out[j]) / sum);
  }
}

template <typename Type>
void test_scaled_softmax(DType dtype) {
  using namespace test;
  constexpr int batches = 2;
  constexpr int heads = 2;
  constexpr int rows = 8;
  constexpr int cols = 32;
  constexpr float scale = 0.7f;
  constexpr size_t elements_total = batches * heads * rows * cols;
  Tensor input("input", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  Tensor softmax("softmax", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  Tensor grad("grad", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  Tensor grad_out("grad_out", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  fillUniform(&input);
  fillUniform(&grad);
  nvte_scaled_softmax_forward(input.data(), softmax.data(), scale, 0);
  nvte_scaled_softmax_backward(grad.data(), softmax.data(), grad_out.data(), scale, 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::unique_ptr<Type[]> ref = std::make_unique<Type[]>(elements_total);
  std::unique_ptr<Type[]> ref_grad = std::make_unique<Type[]>(elements_total);
  const Type *input_cpu = input.rowwise_cpu_dptr<Type>();
  const Type *grad_cpu = grad.rowwise_cpu_dptr<Type>();
  for (size_t row = 0; row < elements_total / cols; ++row) {
    ref_softmax_row(ref.get() + row * cols, input_cpu + row * cols, nullptr, cols, scale);
    ref_softmax_bwd(ref_grad.get() + row * cols, grad_cpu + row * cols, ref.get() + row * cols,
                    cols, scale);
  }
  auto [atol, rtol] = getTolerances(dtype);
  if (dtype == DType::kBFloat16) atol = 1e-3;
  compareResults("scaled_softmax_fwd", softmax, ref.get(), true, atol, rtol);
  compareResults("scaled_softmax_bwd", grad_out, ref_grad.get(), true, atol, rtol);
}

template <typename Type>
void test_masked_softmax(DType dtype) {
  using namespace test;
  constexpr int batches = 2;
  constexpr int heads = 2;
  constexpr int rows = 8;
  constexpr int cols = 32;
  constexpr float scale = -0.3f;
  constexpr size_t elements_total = batches * heads * rows * cols;
  Tensor input("input", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  Tensor mask("mask", std::vector<size_t>{1, 1, rows, cols}, DType::kByte);
  Tensor softmax("softmax", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  Tensor grad("grad", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  Tensor grad_out("grad_out", std::vector<size_t>{batches, heads, rows, cols}, dtype);
  fillUniform(&input);
  fillUniform(&grad);
  uint8_t *mask_cpu = mask.rowwise_cpu_dptr<uint8_t>();
  for (size_t i = 0; i < rows * cols; ++i) {
    mask_cpu[i] = (i % 7 == 0) ? 1 : 0;
  }
  mask.from_cpu();
  nvte_scaled_masked_softmax_forward(input.data(), mask.data(), softmax.data(), scale, 0);
  nvte_scaled_masked_softmax_backward(grad.data(), softmax.data(), grad_out.data(), scale, 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::unique_ptr<Type[]> ref = std::make_unique<Type[]>(elements_total);
  std::unique_ptr<Type[]> ref_grad = std::make_unique<Type[]>(elements_total);
  const Type *input_cpu = input.rowwise_cpu_dptr<Type>();
  const Type *grad_cpu = grad.rowwise_cpu_dptr<Type>();
  for (int row = 0; row < batches * heads * rows; ++row) {
    const int mask_row = row % rows;
    ref_softmax_row(ref.get() + row * cols, input_cpu + row * cols, mask_cpu + mask_row * cols,
                    cols, scale);
    ref_softmax_bwd(ref_grad.get() + row * cols, grad_cpu + row * cols, ref.get() + row * cols,
                    cols, scale);
  }
  auto [atol, rtol] = getTolerances(dtype);
  if (dtype == DType::kBFloat16) atol = 1e-3;
  compareResults("masked_softmax_fwd", softmax, ref.get(), true, atol, rtol);
  compareResults("masked_softmax_bwd", grad_out, ref_grad.get(), true, atol, rtol);
}

template <typename Type>
void test_upper_softmax(DType dtype) {
  using namespace test;
  constexpr int attn_batches = 4;
  constexpr int seq = 32;
  constexpr float scale = 1.2f;
  constexpr size_t elements_total = attn_batches * seq * seq;
  Tensor input("input", std::vector<size_t>{attn_batches, seq, seq}, dtype);
  Tensor softmax("softmax", std::vector<size_t>{attn_batches, seq, seq}, dtype);
  Tensor grad("grad", std::vector<size_t>{attn_batches, seq, seq}, dtype);
  Tensor grad_out("grad_out", std::vector<size_t>{attn_batches, seq, seq}, dtype);
  fillUniform(&input);
  fillUniform(&grad);
  nvte_scaled_upper_triang_masked_softmax_forward(input.data(), softmax.data(), scale, 0);
  nvte_scaled_upper_triang_masked_softmax_backward(grad.data(), softmax.data(), grad_out.data(),
                                                   scale, 0);
  cudaDeviceSynchronize();
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  std::unique_ptr<Type[]> ref = std::make_unique<Type[]>(elements_total);
  std::unique_ptr<Type[]> ref_grad = std::make_unique<Type[]>(elements_total);
  const Type *input_cpu = input.rowwise_cpu_dptr<Type>();
  const Type *grad_cpu = grad.rowwise_cpu_dptr<Type>();
  for (int batch = 0; batch < attn_batches; ++batch) {
    for (int row = 0; row < seq; ++row) {
      const size_t offset = (batch * seq + row) * seq;
      ref_upper_row(ref.get() + offset, input_cpu + offset, row, seq, scale);
      ref_softmax_bwd(ref_grad.get() + offset, grad_cpu + offset, ref.get() + offset, seq, scale);
      for (int col = row + 1; col < seq; ++col) {
        ref_grad[offset + col] = static_cast<Type>(0.0f);
      }
    }
  }
  auto [atol, rtol] = getTolerances(dtype);
  if (dtype == DType::kBFloat16) atol = 1e-3;
  compareResults("upper_softmax_fwd", softmax, ref.get(), true, atol, rtol);
  compareResults("upper_softmax_bwd", grad_out, ref_grad.get(), true, atol, rtol);
}

}  // namespace

// Dispatch a 16-bit float dtype to a templated test body. Mirrors
// TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT but uses the test harness's own fp16/bf16
// aliases so we don't have to include common.h here -- doing so would make the
// test's Tensor type ambiguous with transformer_engine::Tensor.
#define SOFTMAX_TEST_DISPATCH_16BIT(dtype, fn)              \
  switch (dtype) {                                          \
    case DType::kFloat16:                                   \
      fn<test::fp16>(dtype);                                \
      break;                                                \
    case DType::kBFloat16:                                  \
      fn<test::bf16>(dtype);                                \
      break;                                                \
    default:                                                \
      GTEST_FAIL() << "Unsupported 16-bit dtype for test";  \
  }

class SoftmaxApiTestSuite : public ::testing::TestWithParam<DType> {};

TEST_P(SoftmaxApiTestSuite, ScaledSoftmax) {
  const DType dtype = GetParam();
  SOFTMAX_TEST_DISPATCH_16BIT(dtype, test_scaled_softmax);
}

TEST_P(SoftmaxApiTestSuite, MaskedSoftmax) {
  const DType dtype = GetParam();
  SOFTMAX_TEST_DISPATCH_16BIT(dtype, test_masked_softmax);
}

TEST_P(SoftmaxApiTestSuite, UpperTriangularSoftmax) {
  const DType dtype = GetParam();
  SOFTMAX_TEST_DISPATCH_16BIT(dtype, test_upper_softmax);
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, SoftmaxApiTestSuite,
                         ::testing::Values(DType::kFloat16, DType::kBFloat16),
                         [](const testing::TestParamInfo<DType> &info) {
                           return test::typeName(info.param);
                         });
