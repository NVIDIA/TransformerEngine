/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <transformer_engine/linear_attention.h>

#include <cmath>
#include <random>
#include <vector>

#include "../test_common.h"

namespace {

class LinearAttentionPrefixTest : public ::testing::TestWithParam<size_t> {};

TEST_P(LinearAttentionPrefixTest, OrderedAffineRecipes) {
  using test::Tensor;
  using transformer_engine::DType;
  const size_t k = GetParam(), heads = 2, values = 5, row_size = heads * values * k;
  Tensor seeds("seeds", std::vector<size_t>{3, heads, values, k}, DType::kFloat32);
  Tensor offsets("offsets", std::vector<size_t>{6, heads, values, k}, DType::kFloat32);
  Tensor matrices("matrices", std::vector<size_t>{6, heads, k, k}, DType::kFloat32);
  Tensor descriptors("descriptors", std::vector<size_t>{6, 4}, DType::kInt32);
  Tensor output("output", std::vector<size_t>{6, heads, values, k}, DType::kFloat32);
  std::mt19937 generator(317);
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);
  auto *s = seeds.rowwise_cpu_dptr<float>();
  auto *h = offsets.rowwise_cpu_dptr<float>();
  auto *t = matrices.rowwise_cpu_dptr<float>();
  auto *d = descriptors.rowwise_cpu_dptr<int>();
  for (size_t i = 0; i < 3 * row_size; ++i) s[i] = distribution(generator);
  for (size_t i = 0; i < 6 * row_size; ++i) h[i] = distribution(generator);
  for (size_t i = 0; i < 6 * heads * k * k; ++i) t[i] = distribution(generator) / sqrt(k);
  const int rows[] = {0, 0, 0, 1, 1, 0, 3, 1, 2, 5, 3, -2,
                      -1, 0, 0, 1, -1, 0, 3, 1, -1, 5, 3, -2};
  std::copy(rows, rows + 24, d);
  seeds.from_cpu();
  offsets.from_cpu();
  matrices.from_cpu();
  descriptors.from_cpu();
  nvte_linear_attention_affine_prefix(seeds.data(), offsets.data(), matrices.data(),
                                      descriptors.data(), output.data(), nullptr);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  output.to_cpu();
  const auto *actual = output.rowwise_cpu_dptr<float>();
  for (size_t span = 0; span < 6; ++span) {
    for (size_t head = 0; head < heads; ++head) {
      for (size_t value = 0; value < values; ++value) {
        const size_t row = head * values + value;
        std::vector<double> state(k, 0.), next(k);
        if (rows[span * 4] >= 0) {
          const float *start = s + rows[span * 4] * row_size + row * k;
          std::copy(start, start + k, state.begin());
        }
        for (int step = 0; step < rows[span * 4 + 2]; ++step) {
          const int record = rows[span * 4 + 1] + step * rows[span * 4 + 3];
          for (size_t col = 0; col < k; ++col) {
            next[col] = h[record * row_size + row * k + col];
            for (size_t inner = 0; inner < k; ++inner) {
              next[col] += state[inner] * t[(record * heads + head) * k * k + inner * k + col];
            }
          }
          state.swap(next);
        }
        for (size_t col = 0; col < k; ++col) {
          ASSERT_NEAR(actual[span * row_size + row * k + col], state[col], 2e-5);
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Common, LinearAttentionPrefixTest, ::testing::Values(3, 64, 128, 257));

}  // namespace
