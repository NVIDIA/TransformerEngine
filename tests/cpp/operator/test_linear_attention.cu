/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <transformer_engine/linear_attention.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "../test_common.h"

namespace {

constexpr size_t kDescriptorWidth = 4;
using AffineDescriptor = std::array<int, kDescriptorWidth>;

void RunAffinePrefixTest(size_t key_dim, size_t heads, size_t value_dim, size_t num_summaries,
                         const std::vector<AffineDescriptor> &descriptor_rows) {
  using test::Tensor;
  using transformer_engine::DType;
  constexpr size_t kNumInitialStates = 3;
  constexpr unsigned kSeed = 2026;
  constexpr double kTolerance = 2e-5;
  const size_t num_outputs = descriptor_rows.size();
  const size_t state_size = heads * value_dim * key_dim;
  Tensor initial_states("initial_states",
                        std::vector<size_t>{kNumInitialStates, heads, value_dim, key_dim},
                        DType::kFloat32);
  Tensor additive("additive", std::vector<size_t>{num_summaries, heads, value_dim, key_dim},
                  DType::kFloat32);
  Tensor transitions("transitions", std::vector<size_t>{num_summaries, heads, key_dim, key_dim},
                     DType::kFloat32);
  Tensor descriptors("descriptors", std::vector<size_t>{num_outputs, kDescriptorWidth},
                     DType::kInt32);
  Tensor output("output", std::vector<size_t>{num_outputs, heads, value_dim, key_dim},
                DType::kFloat32);
  std::mt19937 generator(kSeed);
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);
  auto *initial_states_cpu = initial_states.rowwise_cpu_dptr<float>();
  auto *additive_cpu = additive.rowwise_cpu_dptr<float>();
  auto *transitions_cpu = transitions.rowwise_cpu_dptr<float>();
  auto *descriptors_cpu = descriptors.rowwise_cpu_dptr<int>();
  for (size_t i = 0; i < kNumInitialStates * state_size; ++i) {
    initial_states_cpu[i] = distribution(generator);
  }
  for (size_t i = 0; i < num_summaries * state_size; ++i) {
    additive_cpu[i] = distribution(generator);
  }
  for (size_t i = 0; i < num_summaries * heads * key_dim * key_dim; ++i) {
    transitions_cpu[i] = distribution(generator) / std::sqrt(key_dim);
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    std::copy(descriptor_rows[i].begin(), descriptor_rows[i].end(),
              descriptors_cpu + i * kDescriptorWidth);
  }
  // Detect unwritten output elements, including outputs that should be zero.
  std::fill_n(output.rowwise_cpu_dptr<float>(), num_outputs * state_size,
              std::numeric_limits<float>::quiet_NaN());
  initial_states.from_cpu();
  additive.from_cpu();
  transitions.from_cpu();
  descriptors.from_cpu();
  output.from_cpu();
  nvte_linear_attention_affine_prefix(initial_states.data(), additive.data(), transitions.data(),
                                      descriptors.data(), output.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  output.to_cpu();
  const auto *actual = output.rowwise_cpu_dptr<float>();
  for (size_t output_index = 0; output_index < num_outputs; ++output_index) {
    const auto [initial_memory_index, first_summary, count, stride] = descriptor_rows[output_index];
    for (size_t head = 0; head < heads; ++head) {
      for (size_t value = 0; value < value_dim; ++value) {
        const size_t row_offset = (head * value_dim + value) * key_dim;
        std::vector<double> state(key_dim, 0.), next(key_dim);
        if (initial_memory_index >= 0) {
          const float *start = initial_states_cpu + initial_memory_index * state_size + row_offset;
          std::copy(start, start + key_dim, state.begin());
        }
        for (int step = 0; step < count; ++step) {
          const int summary_index = first_summary + step * stride;
          const float *transition_matrix =
              transitions_cpu + (summary_index * heads + head) * key_dim * key_dim;
          for (size_t col = 0; col < key_dim; ++col) {
            next[col] = additive_cpu[summary_index * state_size + row_offset + col];
            for (size_t inner = 0; inner < key_dim; ++inner) {
              next[col] += state[inner] * transition_matrix[inner * key_dim + col];
            }
          }
          state.swap(next);
        }
        for (size_t col = 0; col < key_dim; ++col) {
          ASSERT_NEAR(actual[output_index * state_size + row_offset + col], state[col], kTolerance)
              << "output=" << output_index << ", head=" << head << ", value=" << value
              << ", column=" << col;
        }
      }
    }
  }
}

class LinearAttentionPrefixTest
    : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(LinearAttentionPrefixTest, OrderedAffineRecipes) {
  const auto [key_dim, heads, value_dim] = GetParam();
  constexpr size_t kNumSummaries = 7;
  // Rows are (initial_memory_index, first_summary, count, stride).
  const std::vector<AffineDescriptor> descriptor_rows = {
      {0, 0, 0, 1},    // Copy initial memory 0.
      {1, 0, 3, 1},    // Apply summaries 0, 1, 2 to initial memory 1.
      {2, 5, 3, -2},   // Reverse composition using summaries 5, 3, 1.
      {-1, 0, 0, 1},   // Zero initial memory, with no summaries.
      {-1, 0, 3, 2},   // Zero initial memory, then summaries 0, 2, 4.
      {-1, 6, 3, -3},  // Zero initial memory, then summaries 6, 3, 0.
      {1, 4, 1, 1},    // One summary with an explicit initial memory.
      {-1, 4, 1, 1},   // One summary with zero initial memory: copy H[4].
  };
  RunAffinePrefixTest(key_dim, heads, value_dim, kNumSummaries, descriptor_rows);
}

TEST(LinearAttentionPrefixLongChainTest, ForwardAndReverse) {
  constexpr size_t kKeyDim = 64;
  constexpr size_t kHeads = 2;
  constexpr size_t kValueDim = 5;
  constexpr size_t kNumSummaries = 32;
  constexpr int kLastSummary = static_cast<int>(kNumSummaries) - 1;
  // CP=32 can require composing 31 preceding or following fragment summaries.
  constexpr int kChainLength = 31;
  const std::vector<AffineDescriptor> descriptor_rows = {
      {1, 0, kChainLength, 1},
      {2, kLastSummary, kChainLength, -1},
      {-1, 0, kChainLength, 1},
      {-1, kLastSummary, kChainLength, -1},
  };
  RunAffinePrefixTest(kKeyDim, kHeads, kValueDim, kNumSummaries, descriptor_rows);
}

INSTANTIATE_TEST_SUITE_P(
    Common, LinearAttentionPrefixTest,
    ::testing::Combine(::testing::Values(3, 64, 128, 257), ::testing::Values(1, 8),
                       ::testing::Values(1, 5, 16)),
    [](const testing::TestParamInfo<LinearAttentionPrefixTest::ParamType> &info) {
      return "K" + std::to_string(std::get<0>(info.param)) + "H" +
             std::to_string(std::get<1>(info.param)) + "V" +
             std::to_string(std::get<2>(info.param));
    });

}  // namespace
