/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Random streams of NVFP4 1D stochastic rounding.
//
// The 1D quantize kernel draws Philox once per output block (nvfp4_block_philox.cuh) with a
// subsequence built from the block's coordinates and direction.  These tests check that
//  - each draw is exactly what curanddx's Philox returns for the same key and counter, and
//  - no two output blocks, and not the rowwise and columnwise draws of one element, share a
//    stream: for inputs where every element sits halfway between two E2M1 values, a shared
//    stream would show up as identical codes far more often than chance.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>

#include "../test_common.h"
#include "cast/nvfp4/nvfp4_block_philox.cuh"
#include "util/curanddx.hpp"

using namespace transformer_engine;

namespace {

namespace bp = transformer_engine::dispatch::nvfp4::block_philox;

#define CHECK_CUDA(expr) ASSERT_EQ((expr), cudaSuccess)

// ---------------------------------------------------------------------------------------------
// The folded generator against curanddx.
// ---------------------------------------------------------------------------------------------

__global__ void compare_with_curanddx_kernel(const uint64_t *seeds, const uint64_t *offsets,
                                             const uint32_t *sub_lo, const uint32_t *logical_hi,
                                             int n, uint4 *folded, uint4 *reference) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const bp::PhiloxState p = bp::make_philox(seeds[i], offsets[i]);
  const bp::PhiloxFold f = bp::make_fold(p, sub_lo[i]);
  const uint32_t hi[2] = {logical_hi[i], logical_hi[i] + 1u};
  uint4 out[2];
  bp::philox_fold_n<2>(hi, f, out);
  for (int k = 0; k < 2; ++k) {
    // The fold's subsequence high word is the logical coordinate XORed with p.b.
    const uint64_t subsequence =
        (static_cast<uint64_t>(hi[k] ^ p.b) << 32) | static_cast<uint64_t>(sub_lo[i]);
    curanddx::detail::philox4x32_native_state<bp::kRounds> state;
    state.init(seeds[i], subsequence, offsets[i]);
    reference[2 * i + k] = state.generate4();
    folded[2 * i + k] = out[k];
  }
}

TEST(NVFP4StochasticRoundingStreams, BlockPhiloxMatchesCuranddx) {
  constexpr int n = 1 << 16;
  std::mt19937_64 rng(2026);
  std::vector<uint64_t> seeds(n), offsets(n);
  std::vector<uint32_t> sub_lo(n), logical_hi(n);
  for (int i = 0; i < n; ++i) {
    seeds[i] = rng();
    // Small offsets as the framework produces them, and full 64-bit ones.
    offsets[i] = (i % 2 == 0) ? (rng() & 0xFFFFF) : rng();
    sub_lo[i] = static_cast<uint32_t>(rng());
    logical_hi[i] = static_cast<uint32_t>(rng());
  }
  uint64_t *d_seeds, *d_offsets;
  uint32_t *d_lo, *d_hi;
  uint4 *d_folded, *d_reference;
  CHECK_CUDA(cudaMalloc(&d_seeds, n * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_offsets, n * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_lo, n * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_hi, n * sizeof(uint32_t)));
  CHECK_CUDA(cudaMalloc(&d_folded, 2 * n * sizeof(uint4)));
  CHECK_CUDA(cudaMalloc(&d_reference, 2 * n * sizeof(uint4)));
  CHECK_CUDA(cudaMemcpy(d_seeds, seeds.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_offsets, offsets.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_lo, sub_lo.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_hi, logical_hi.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  compare_with_curanddx_kernel<<<n / 256, 256>>>(d_seeds, d_offsets, d_lo, d_hi, n, d_folded,
                                                 d_reference);
  CHECK_CUDA(cudaGetLastError());
  std::vector<uint4> folded(2 * n), reference(2 * n);
  CHECK_CUDA(cudaMemcpy(folded.data(), d_folded, 2 * n * sizeof(uint4), cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(reference.data(), d_reference, 2 * n * sizeof(uint4), cudaMemcpyDeviceToHost));
  int mismatches = 0;
  for (int i = 0; i < 2 * n; ++i) {
    const uint4 a = folded[i], b = reference[i];
    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) ++mismatches;
  }
  EXPECT_EQ(mismatches, 0) << "folded Philox differs from curanddx (" << bp::kRounds
                           << " rounds) in " << mismatches << " of " << 2 * n << " draws";
  for (void *p : {static_cast<void *>(d_seeds), static_cast<void *>(d_offsets),
                  static_cast<void *>(d_lo), static_cast<void *>(d_hi),
                  static_cast<void *>(d_folded), static_cast<void *>(d_reference)}) {
    cudaFree(p);
  }
}

// ---------------------------------------------------------------------------------------------
// The block -> subsequence map is one-to-one.
// ---------------------------------------------------------------------------------------------

uint64_t subsequence_key(uint32_t lo, uint32_t hi) { return (static_cast<uint64_t>(hi) << 32) | lo; }

TEST(NVFP4StochasticRoundingStreams, EveryBlockHasItsOwnSubsequence) {
  // Every rowwise and columnwise block of one tensor.
  constexpr uint32_t M = 2048, K = 4096;
  std::vector<uint64_t> keys;
  keys.reserve(2 * M * K / 16);
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t block_col = 0; block_col < K / 16; ++block_col) {
      keys.push_back(subsequence_key(bp::rowwise_sub_lo(block_col), bp::rowwise_sub_hi(row)));
    }
  }
  for (uint32_t row_block = 0; row_block < M / 16; ++row_block) {
    for (uint32_t col = 0; col < K; ++col) {
      keys.push_back(subsequence_key(bp::columnwise_sub_lo(col & ~3u),
                                     bp::columnwise_sub_hi(row_block, col & 3u)));
    }
  }
  std::sort(keys.begin(), keys.end());
  EXPECT_EQ(std::adjacent_find(keys.begin(), keys.end()), keys.end())
      << "two output blocks share a Philox subsequence";

  // Across the supported coordinate range the map decodes back to the block, so it cannot
  // collide anywhere: rows < 2^30, block columns and columns < 2^31.
  std::mt19937_64 rng(7);
  for (int i = 0; i < (1 << 20); ++i) {
    const uint32_t row = static_cast<uint32_t>(rng()) >> 2, block_col = static_cast<uint32_t>(rng()) >> 1;
    const uint32_t lo = bp::rowwise_sub_lo(block_col), hi = bp::rowwise_sub_hi(row);
    ASSERT_EQ(lo & 1u, 0u);
    ASSERT_EQ(lo >> 1, block_col);
    ASSERT_EQ(hi, row);
    const uint32_t row_block = static_cast<uint32_t>(rng()) >> 2;
    const uint32_t col = static_cast<uint32_t>(rng()) >> 1;
    const uint32_t clo = bp::columnwise_sub_lo(col & ~3u);
    const uint32_t chi = bp::columnwise_sub_hi(row_block, col & 3u);
    ASSERT_EQ(clo & 1u, 1u);
    ASSERT_EQ(((clo >> 1) & ~3u) | (chi & 3u), col);
    ASSERT_EQ(chi >> 2, row_block);
  }
}

// ---------------------------------------------------------------------------------------------
// Stream separation, observed through nvte_quantize_v2.
//
// With the global amax set to 448 * 6, S_enc = 1; every 16-element block contains one 6.0, so
// every block scale is exactly 1.0 and elements are quantized unscaled.  All other elements are
// midpoints between adjacent E2M1 magnitudes, so stochastic rounding sends each one up or down
// with probability 1/2.
// ---------------------------------------------------------------------------------------------

constexpr float kMidpoints[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
// E2M1 magnitude code of the grid value just below kMidpoints[i] is i.
constexpr uint8_t kE4M3One = 0x38;

// Symmetric pseudo-random choice of midpoint and sign for (r, c).
uint32_t pair_hash(size_t r, size_t c) {
  const uint64_t a = std::min(r, c), b = std::max(r, c);
  uint64_t h = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull);
  h ^= h >> 31;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 29;
  return static_cast<uint32_t>(h);
}

enum class Pattern { Symmetric, RowsEqual, ColumnsEqual };

float element(Pattern pattern, size_t r, size_t c) {
  size_t a = r, b = c;
  if (pattern == Pattern::RowsEqual) a = 0;     // x[r][c] depends on c only
  if (pattern == Pattern::ColumnsEqual) b = 0;  // x[r][c] depends on r only
  if (pattern == Pattern::Symmetric && r % 16 == c % 16) return 6.0f;
  if (pattern == Pattern::RowsEqual && c % 16 == 0) return 6.0f;
  if (pattern == Pattern::ColumnsEqual && r % 16 == 0) return 6.0f;
  const uint32_t h = pattern == Pattern::Symmetric ? pair_hash(r, c) : pair_hash(a, b);
  const float v = kMidpoints[h % 7];
  return (h & 0x100) ? -v : v;
}

struct QuantizedOutput {
  std::vector<uint8_t> data, scales, data_t, scales_t;
};

// Quantizes the M x K pattern with stochastic rounding; columnwise output if `colwise`.
void quantize_sr(Pattern pattern, size_t M, size_t K, bool colwise, QuantizedOutput *out) {
  std::vector<__nv_bfloat16> h_x(M * K);
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < K; ++c) h_x[r * K + c] = __float2bfloat16(element(pattern, r, c));
  }
  const size_t s_rows = (M + 127) / 128 * 128, s_cols = (K / 16 + 3) / 4 * 4;
  const size_t st_rows = (K + 127) / 128 * 128, st_cols = (M / 16 + 3) / 4 * 4;
  void *x, *q, *s, *qt, *st, *amax, *rng;
  ASSERT_EQ(cudaMalloc(&x, M * K * 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&q, M * K / 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&s, s_rows * s_cols), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&qt, M * K / 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&st, st_rows * st_cols), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&amax, 2 * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&rng, 2 * sizeof(int64_t)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(x, h_x.data(), M * K * 2, cudaMemcpyHostToDevice), cudaSuccess);
  const float h_amax[2] = {448.0f * 6.0f, 448.0f * 6.0f};
  ASSERT_EQ(cudaMemcpy(amax, h_amax, sizeof(h_amax), cudaMemcpyHostToDevice), cudaSuccess);
  const int64_t h_rng[2] = {12345, 678};
  ASSERT_EQ(cudaMemcpy(rng, h_rng, sizeof(h_rng), cudaMemcpyHostToDevice), cudaSuccess);

  TensorWrapper input(x, std::vector<size_t>{M, K}, DType::kBFloat16);
  TensorWrapper output(NVTE_NVFP4_1D_SCALING);
  output.set_rowwise_data(q, DType::kFloat4E2M1, std::vector<size_t>{M, K});
  output.set_rowwise_scale_inv(s, DType::kFloat8E4M3, std::vector<size_t>{s_rows, s_cols});
  output.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
  if (colwise) {
    output.set_columnwise_data(qt, DType::kFloat4E2M1, std::vector<size_t>{K, M});
    output.set_columnwise_scale_inv(st, DType::kFloat8E4M3, std::vector<size_t>{st_rows, st_cols});
    output.set_columnwise_amax(static_cast<float *>(amax) + 1, DType::kFloat32,
                               std::vector<size_t>{1});
  }
  TensorWrapper rng_state(rng, std::vector<size_t>{2}, DType::kInt64);
  QuantizationConfigWrapper config;
  config.set_stochastic_rounding(true);
  config.set_rng_state(rng_state.data());
  nvte_quantize_v2(input.data(), output.data(), config, 0);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  out->data.resize(M * K / 2);
  out->scales.resize(s_rows * s_cols);
  ASSERT_EQ(cudaMemcpy(out->data.data(), q, out->data.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(out->scales.data(), s, out->scales.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
  if (colwise) {
    out->data_t.resize(M * K / 2);
    out->scales_t.resize(st_rows * st_cols);
    ASSERT_EQ(cudaMemcpy(out->data_t.data(), qt, out->data_t.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(out->scales_t.data(), st, out->scales_t.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
  }
  for (void *p : {x, q, s, qt, st, amax, rng}) cudaFree(p);
}

uint8_t code_at(const std::vector<uint8_t> &data, size_t row, size_t cols, size_t col) {
  const uint8_t byte = data[(row * cols + col) / 2];
  return (col % 2 == 0) ? (byte & 0xF) : (byte >> 4);
}

// Checks every code of an (rows x cols) output whose element (i, j) was input value v(i, j):
// 6.0 stays exact, midpoints land on one of their two neighbours.  Returns the fraction of
// midpoints rounded up.
template <typename ValueFn>
double check_codes(const std::vector<uint8_t> &data, size_t rows, size_t cols, ValueFn v) {
  size_t up = 0, total = 0, bad = 0;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      const float x = v(i, j);
      const uint8_t code = code_at(data, i, cols, j);
      const bool neg = (code & 8) != 0;
      const uint8_t mag = code & 7;
      if (x == 6.0f) {
        bad += (code != 7);
        continue;
      }
      int lo = 0;
      while (kMidpoints[lo] != std::fabs(x)) ++lo;
      const bool sign_ok = (neg == (x < 0));
      if (!sign_ok || (mag != lo && mag != lo + 1)) {
        ++bad;
      } else {
        up += (mag == lo + 1);
      }
      ++total;
    }
  }
  EXPECT_EQ(bad, 0u) << "codes that are not a neighbour of their input";
  return total ? static_cast<double>(up) / total : 0.0;
}

class NVFP4StochasticRoundingStreamsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) GTEST_SKIP();
  }
};

constexpr size_t kN = 1024;  // square, so the symmetric pattern has a transpose to compare

TEST_F(NVFP4StochasticRoundingStreamsTest, RowwiseAndColumnwiseDrawsDiffer) {
  QuantizedOutput out;
  quantize_sr(Pattern::Symmetric, kN, kN, /*colwise=*/true, &out);
  for (size_t r = 0; r < kN; ++r) {
    for (size_t b = 0; b < kN / 16; ++b) {
      ASSERT_EQ(out.scales[r * ((kN / 16 + 3) / 4 * 4) + b], kE4M3One);
    }
  }
  auto v = [](size_t i, size_t j) { return element(Pattern::Symmetric, i, j); };
  const double up_row = check_codes(out.data, kN, kN, v);
  // Row i of the columnwise output is column i of the input, which by symmetry is row i.
  const double up_col = check_codes(out.data_t, kN, kN, v);
  EXPECT_NEAR(up_row, 0.5, 0.02);
  EXPECT_NEAR(up_col, 0.5, 0.02);

  // Rowwise block (r, b) and columnwise block (r, b) quantize the same 16 values.  With 15
  // fair coin flips each, independent streams match with probability 2^-15.
  size_t same = 0;
  for (size_t r = 0; r < kN; ++r) {
    for (size_t b = 0; b < kN / 16; ++b) {
      bool eq = true;
      for (size_t k = 0; k < 16 && eq; ++k) {
        eq = code_at(out.data, r, kN, 16 * b + k) == code_at(out.data_t, r, kN, 16 * b + k);
      }
      same += eq;
    }
  }
  EXPECT_LT(same, 10u) << same << " of " << kN * kN / 16
                       << " rowwise blocks drew the same bits as their columnwise twin";
}

// Every row holds the same values, so blocks in one block column differ only in their stream.
void expect_rows_draw_independently(bool colwise) {
  QuantizedOutput out;
  quantize_sr(Pattern::RowsEqual, kN, kN, colwise, &out);
  auto v = [](size_t i, size_t j) { return element(Pattern::RowsEqual, i, j); };
  EXPECT_NEAR(check_codes(out.data, kN, kN, v), 0.5, 0.02);
  size_t same = 0;
  for (size_t r = 1; r < kN; ++r) {
    for (size_t b = 0; b < kN / 16; ++b) {
      bool eq = true;
      for (size_t k = 0; k < 16 && eq; ++k) {
        eq = code_at(out.data, r, kN, 16 * b + k) == code_at(out.data, r - 1, kN, 16 * b + k);
      }
      same += eq;
    }
  }
  EXPECT_LT(same, 10u) << same << " rowwise blocks drew the same bits as the block above";
}

TEST_F(NVFP4StochasticRoundingStreamsTest, RowwiseOnlyBlocksDrawIndependently) {
  expect_rows_draw_independently(/*colwise=*/false);
}

TEST_F(NVFP4StochasticRoundingStreamsTest, FusedRowwiseBlocksDrawIndependently) {
  expect_rows_draw_independently(/*colwise=*/true);
}

// Every column holds the same values, so columnwise blocks in one row block differ only in
// their stream.
TEST_F(NVFP4StochasticRoundingStreamsTest, ColumnwiseBlocksDrawIndependently) {
  QuantizedOutput out;
  quantize_sr(Pattern::ColumnsEqual, kN, kN, /*colwise=*/true, &out);
  // Row c of the columnwise output is column c of the input: value at (c, r) is x[r][c].
  auto vt = [](size_t c, size_t r) { return element(Pattern::ColumnsEqual, r, c); };
  EXPECT_NEAR(check_codes(out.data_t, kN, kN, vt), 0.5, 0.02);
  size_t same = 0;
  for (size_t c = 1; c < kN; ++c) {
    for (size_t b = 0; b < kN / 16; ++b) {
      bool eq = true;
      for (size_t k = 0; k < 16 && eq; ++k) {
        eq = code_at(out.data_t, c, kN, 16 * b + k) == code_at(out.data_t, c - 1, kN, 16 * b + k);
      }
      same += eq;
    }
  }
  EXPECT_LT(same, 10u) << same << " columnwise blocks drew the same bits as the previous column";
}

}  // namespace
