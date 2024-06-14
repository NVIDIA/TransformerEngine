/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/softmax.h>
#include "../test_common.h"

using namespace transformer_engine;

namespace {

using compute_t = float;

template <typename Type>
void compute_single_head_fwd(
  Type *softmax_out,
  const Type *data_in,
  compute_t *buff,
  const float scaling_factor,
  const int rows,
  const int cols)
{
  for (int i = 0; i < rows; ++i) {
    size_t offset = i * cols;

    const int masked_elements = i + cols - rows + 1;
    compute_t max_value = static_cast<compute_t>(-10'000.f);
    for (int j = 0; j < masked_elements; ++j) {
      compute_t tmp = scaling_factor * static_cast<compute_t>(data_in[offset + j]);
      buff[offset + j] = tmp;
      max_value = std::max(max_value, tmp);
    }

    compute_t accumulator = static_cast<compute_t>(0.f);
    for (int j = 0; j < masked_elements; ++j) {
      compute_t tmp = std::exp(buff[offset + j] - max_value);
      buff[offset + j] = tmp;
      accumulator += tmp;
    }

    for (int j = 0; j < cols; ++j) {
      if (j < masked_elements) {
        compute_t tmp = buff[offset + j] / accumulator;
        softmax_out[offset + j] = static_cast<Type>(tmp);
      } else {
        softmax_out[offset + j] = static_cast<Type>(0.f);
      }
    }
  }
}

template <typename Type>
void compute_single_head_bwd(
  Type *grad_out,
  const Type *grad_in,
  const Type *softmax_in,
  compute_t *buff,
  const float scaling_factor,
  const int batches,
  const int heads,
  const int rows,
  const int cols)
{
  for (int i = 0; i < rows; ++i) {
    size_t offset = i * cols;

    const int masked_elements = i + cols - rows + 1;
    compute_t accumulator = static_cast<compute_t>(0.f);
    for (int j = 0; j < masked_elements; ++j) {
      compute_t tmp = static_cast<compute_t>(softmax_in[offset + j])
                      * static_cast<compute_t>(grad_in[offset + j]);
      buff[offset + j] = tmp;
      accumulator += tmp;
    }

    for (int j = 0; j < cols; ++j) {
      if (j < masked_elements) {
        compute_t tmp = buff[offset + j]
                        - static_cast<compute_t>(softmax_in[offset + j]) * accumulator;
        grad_out[offset + j] = static_cast<Type>(scaling_factor * tmp);
      } else {
        grad_out[offset + j] = static_cast<Type>(0.f);
      }
    }
  }
}

template <typename Type>
void compute_fwd_ref(
  Type *softmax_out,
  const Type *data_in,
  compute_t *buff,
  const float scaling_factor,
  const int batches,
  const int heads,
  const int rows,
  const int cols)
{
  size_t head_size = rows * cols;
  size_t batch_size = heads * head_size;

  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < heads; ++h) {
      size_t offset = b * batch_size + h * head_size;
      compute_single_head_fwd(softmax_out + offset, data_in + offset,
                              buff + offset, scaling_factor, rows, cols);
    }
  }
}

template <typename Type>
void compute_bwd_ref(
  Type *grad_out,
  const Type *grad_in,
  const Type *softmax_in,
  compute_t *buff,
  const float scaling_factor,
  const int batches,
  const int heads,
  const int rows,
  const int cols)
{
  size_t head_size = rows * cols;
  size_t batch_size = heads * head_size;

  for (int b = 0; b < batches; ++b) {
    for (int h = 0; h < heads; ++h) {
      size_t offset = b * batch_size + h * head_size;
      compute_single_head_bwd(grad_out + offset, grad_in + offset, softmax_in + offset,
                              buff + offset, scaling_factor, batches, heads, rows, cols);
    }
  }
}


// Query Sequence Length = rows
// Key Sequence Length = cols
template <typename Type>
void performTest(
  const size_t batches,
  const size_t heads,
  const size_t rows,
  const size_t cols,
  float scaling_factor)
{
  using namespace test;

  DType itype = TypeInfo<Type>::dtype;

  Tensor data_in({ batches, heads, rows, cols }, itype);
  Tensor softmax_out({ batches, heads, rows, cols }, itype);
  Tensor softmax_in({ batches, heads, rows, cols }, itype);
  Tensor grads_in({ batches, heads, rows, cols }, itype);
  Tensor grads_out({ batches, heads, rows, cols }, itype);

  const size_t elements_total = batches * heads * rows * cols;
  std::unique_ptr<Type[]> softmax_out_ref = std::make_unique<Type[]>(elements_total);
  std::unique_ptr<Type[]> grads_out_ref = std::make_unique<Type[]>(elements_total);
  std::unique_ptr<compute_t[]> compute_buffer = std::make_unique<compute_t[]>(elements_total);

  fillUniform(&data_in);
  fillUniform(&softmax_in);
  fillUniform(&grads_in);

  nvte_scaled_aligned_causal_masked_softmax_forward(
      data_in.data(), softmax_out.data(), scaling_factor, 0);
  nvte_scaled_aligned_causal_masked_softmax_backward(
      grads_in.data(), softmax_in.data(), grads_out.data(), scaling_factor, 0);


  // Reference implementations
  compute_fwd_ref(softmax_out_ref.get(), data_in.cpu_dptr<Type>(),
                  compute_buffer.get(), scaling_factor, batches, heads, rows, cols);
  compute_bwd_ref(grads_out_ref.get(), grads_in.cpu_dptr<Type>(), softmax_in.cpu_dptr<Type>(),
                  compute_buffer.get(), scaling_factor, batches, heads, rows, cols);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  auto [atol, rtol] = getTolerances(itype);
  if(itype == DType::kBFloat16) {
    atol = 1e-3;
  }
  compareResults("softmax_fwd", softmax_out, softmax_out_ref.get(), atol, rtol);
  compareResults("softmax_bwd", grads_out, grads_out_ref.get(), atol, rtol);
}

// [Batches, Attention Heads, Query Sequence Length, Key Sequence Length, Scaling Factor]
std::vector<std::tuple<size_t, size_t, size_t, size_t, float>> test_cases = {
    {   1,    1,     1,    16,  -1.0f},
    {   1,    2,    17,    32,   0.8f},
    {   2,    1,    37,   112,   1.0f},
    {   2,    4,   127,   128,  -0.2f},
    {   8,    6,   128,   256,   1.3f},
    {   1,    4,   270,   256,   0.8f},
    {   2,    2,   512,   512,  -1.5f},
    {   1,    2,   819,  1024,   2.1f},
    {   1,    2,   281,  1024,   0.2f},
    {   1,    2,   277,  1024,  -2.1f},
    {   1,    2,   127,  1024,   1.1f},
    {   2,    2,   107,  2048,   0.4f},
    {   2,    1,   103,  2048,  -3.0f},
    {   2,    2,   101,  2048,   2.6f},
    {   1,    1,  1024,  4096,   0.6f},
    {   1,    2,    61,  4096,   0.6f},
    {   1,    2,    59,  4096,  -4.9f},
    {   1,    2,    53,  4096,   3.5f},
    {   1,    1,    37,  8192,   0.7f},
    {   1,    1,    31,  8192,  -5.8f},
    {   1,    1,    29,  8192,   4.4f},
    {   1,    1,    23, 12288,   0.8f},
    {   1,    1,    19, 12288,  -6.7f},
    {   1,    1,    17, 12288,   3.3f},
    {   1,    1,    13, 16384,   0.9f},
    {   1,    1,    11, 16384,  -7.6f},
    {   1,    1,     7, 16384,   6.2f}};

}  // namespace

class CausalSoftmaxTestSuite
    : public ::testing::TestWithParam<std::tuple<
        transformer_engine::DType,
        std::tuple<size_t, size_t, size_t, size_t, float>>> {};

TEST_P(CausalSoftmaxTestSuite, TestCausalSoftmax) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const auto size = std::get<1>(GetParam());

  const size_t batches = std::get<0>(size);
  const size_t heads = std::get<1>(size);
  const size_t query_seq_len = std::get<2>(size);
  const size_t key_seq_len = std::get<3>(size);
  const float scaling_factor = std::get<4>(size);

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
    performTest<InputType>(batches, heads, query_seq_len, key_seq_len, scaling_factor);
  );
}


INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  CausalSoftmaxTestSuite,
  ::testing::Combine(
      ::testing::Values(DType::kFloat16, DType::kBFloat16),
      ::testing::ValuesIn(test_cases)),
  [](const testing::TestParamInfo<CausalSoftmaxTestSuite::ParamType>& info) {
    const auto size = std::get<1>(info.param);
    const size_t batches = std::get<0>(size);
    const size_t heads = std::get<1>(size);
    const size_t query_seq_len = std::get<2>(size);
    const size_t key_seq_len = std::get<3>(size);

    std::string scaling_factor = std::to_string(std::get<4>(size));
    for (char& c : scaling_factor) {
      if (c == '-') { c = 'N'; }
      if (c == '.') { c = 'p'; }
    }

    std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                       std::to_string(batches) + "X" +
                       std::to_string(heads) + "X" +
                       std::to_string(query_seq_len) + "X" +
                       std::to_string(key_seq_len) + "X" +
                       scaling_factor;
    return name;
  });
