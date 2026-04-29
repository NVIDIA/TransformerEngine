/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/swizzle.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_swizzle(const uint8_t *h_input, uint8_t *h_output,
                         const size_t M, const size_t K) {
  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (size_t m = 0; m < M; m++) {
    for (size_t k = 0; k < K; k++) {
      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_output_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int out_index = tile_output_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr (row_scaling)
        h_output[out_index] = h_input[k + m * K];
      else
        h_output[out_index] = h_input[k * M + m];
    }
  }
}

static void zero_scale_inv_padding(uint8_t *buf,
                                   size_t padded_rows, size_t padded_cols,
                                   size_t orig_rows, size_t orig_cols) {
  for (size_t r = 0; r < padded_rows; ++r) {
    for (size_t c = 0; c < padded_cols; ++c) {
      if (r >= orig_rows || c >= orig_cols) {
        buf[r * padded_cols + c] = 0;
      }
    }
  }
}

// ===================================================================
// Multi-tensor swizzle test
// ===================================================================

void performTestMultiTensorSwizzle(const int num_tensors, const size_t M, const size_t K,
                                   bool rowwise) {
  using namespace test;
  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};

  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<NVTETensor> input_handles;
  std::vector<NVTETensor> output_handles;

  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, rowwise, !rowwise,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, rowwise, !rowwise,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    output->set_with_gemm_swizzled_scales(true);

    input->to_cpu();
    if (rowwise) {
      const NVTEShape rs = input->rowwise_scale_inv_shape();
      zero_scale_inv_padding(input->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                             rs.data[0], rs.data[1],
                             M, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    } else {
      const NVTEShape cs = input->columnwise_scale_inv_shape();
      zero_scale_inv_padding(input->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                             cs.data[0], cs.data[1],
                             (M + BLOCK_SIZE - 1) / BLOCK_SIZE, K);
    }
    input->from_cpu();

    input_handles.push_back(input->data());
    output_handles.push_back(output->data());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  nvte_multi_tensor_swizzle_scaling_factors(input_handles.data(), output_handles.data(),
                                            num_tensors, 0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  for (int i = 0; i < num_tensors; ++i) {
    output_tensors[i]->to_cpu();
    if (rowwise) {
      const NVTEShape rs = input_tensors[i]->rowwise_scale_inv_shape();
      const size_t numel = rs.data[0] * rs.data[1];
      std::unique_ptr<uint8_t[]> ref = std::make_unique<uint8_t[]>(numel);
      compute_ref_swizzle<128, 4, true>(
          input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
          ref.get(), rs.data[0], rs.data[1]);
      compareResults("multi_tensor_swizzle_row_" + std::to_string(i),
                     output_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                     ref.get(), numel);
    } else {
      const NVTEShape cs = input_tensors[i]->columnwise_scale_inv_shape();
      const size_t numel = cs.data[0] * cs.data[1];
      std::unique_ptr<uint8_t[]> ref = std::make_unique<uint8_t[]>(numel);
      compute_ref_swizzle<128, 4, false>(
          input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
          ref.get(), cs.data[1], cs.data[0]);
      compareResults("multi_tensor_swizzle_col_" + std::to_string(i),
                     output_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                     ref.get(), numel);
    }
  }
}

// ===================================================================
// Multi-tensor unswizzle test (uses single-tensor swizzle to prepare)
// ===================================================================

void performTestMultiTensorUnswizzle(const int num_tensors, const size_t M, const size_t K,
                                     bool rowwise) {
  using namespace test;
  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};

  std::vector<std::unique_ptr<Tensor>> orig_tensors, swizzled_tensors, output_tensors;
  std::vector<NVTETensor> swizzled_handles, output_handles;

  for (int i = 0; i < num_tensors; ++i) {
    auto orig = std::make_unique<Tensor>("orig_" + std::to_string(i), shape,
                                         DType::kFloat8E4M3, rowwise, !rowwise,
                                         NVTE_MXFP8_1D_SCALING);
    auto swizzled = std::make_unique<Tensor>("swizzled_" + std::to_string(i), shape,
                                             DType::kFloat8E4M3, rowwise, !rowwise,
                                             NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, rowwise, !rowwise,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(orig.get());
    swizzled->set_with_gemm_swizzled_scales(true);

    orig->to_cpu();
    if (rowwise) {
      const NVTEShape rs = orig->rowwise_scale_inv_shape();
      zero_scale_inv_padding(orig->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                             rs.data[0], rs.data[1],
                             M, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    } else {
      const NVTEShape cs = orig->columnwise_scale_inv_shape();
      zero_scale_inv_padding(orig->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                             cs.data[0], cs.data[1],
                             (M + BLOCK_SIZE - 1) / BLOCK_SIZE, K);
    }
    orig->from_cpu();

    nvte_swizzle_scaling_factors(orig->data(), swizzled->data(), 0);

    swizzled_handles.push_back(swizzled->data());
    output_handles.push_back(output->data());
    orig_tensors.emplace_back(std::move(orig));
    swizzled_tensors.emplace_back(std::move(swizzled));
    output_tensors.emplace_back(std::move(output));
  }

  nvte_multi_tensor_unswizzle_scaling_factors(swizzled_handles.data(), output_handles.data(),
                                              num_tensors, 0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  for (int i = 0; i < num_tensors; ++i) {
    orig_tensors[i]->to_cpu();
    output_tensors[i]->to_cpu();
    if (rowwise) {
      const NVTEShape rs = orig_tensors[i]->rowwise_scale_inv_shape();
      const size_t numel = rs.data[0] * rs.data[1];
      compareResults("multi_unswizzle_row_" + std::to_string(i),
                     output_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                     orig_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                     numel);
    } else {
      const NVTEShape cs = orig_tensors[i]->columnwise_scale_inv_shape();
      const size_t numel = cs.data[0] * cs.data[1];
      compareResults("multi_unswizzle_col_" + std::to_string(i),
                     output_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                     orig_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                     numel);
    }
  }
}

// ===================================================================
// Multi-tensor swizzle -> unswizzle roundtrip test
// ===================================================================

void performTestMultiTensorRoundtrip(const int num_tensors, const size_t M, const size_t K,
                                     bool rowwise) {
  using namespace test;
  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};

  std::vector<std::unique_ptr<Tensor>> orig_tensors, mid_tensors, final_tensors;
  std::vector<NVTETensor> orig_handles, mid_handles, final_handles;

  for (int i = 0; i < num_tensors; ++i) {
    auto orig = std::make_unique<Tensor>("orig_" + std::to_string(i), shape,
                                         DType::kFloat8E4M3, rowwise, !rowwise,
                                         NVTE_MXFP8_1D_SCALING);
    auto mid = std::make_unique<Tensor>("mid_" + std::to_string(i), shape,
                                        DType::kFloat8E4M3, rowwise, !rowwise,
                                        NVTE_MXFP8_1D_SCALING);
    auto fin = std::make_unique<Tensor>("fin_" + std::to_string(i), shape,
                                        DType::kFloat8E4M3, rowwise, !rowwise,
                                        NVTE_MXFP8_1D_SCALING);
    fillUniform(orig.get());
    mid->set_with_gemm_swizzled_scales(true);

    orig->to_cpu();
    if (rowwise) {
      const NVTEShape rs = orig->rowwise_scale_inv_shape();
      zero_scale_inv_padding(orig->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                             rs.data[0], rs.data[1],
                             M, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    } else {
      const NVTEShape cs = orig->columnwise_scale_inv_shape();
      zero_scale_inv_padding(orig->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                             cs.data[0], cs.data[1],
                             (M + BLOCK_SIZE - 1) / BLOCK_SIZE, K);
    }
    orig->from_cpu();

    orig_handles.push_back(orig->data());
    mid_handles.push_back(mid->data());
    final_handles.push_back(fin->data());
    orig_tensors.emplace_back(std::move(orig));
    mid_tensors.emplace_back(std::move(mid));
    final_tensors.emplace_back(std::move(fin));
  }

  nvte_multi_tensor_swizzle_scaling_factors(orig_handles.data(), mid_handles.data(),
                                            num_tensors, 0);
  nvte_multi_tensor_unswizzle_scaling_factors(mid_handles.data(), final_handles.data(),
                                              num_tensors, 0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  for (int i = 0; i < num_tensors; ++i) {
    orig_tensors[i]->to_cpu();
    final_tensors[i]->to_cpu();
    if (rowwise) {
      const NVTEShape rs = orig_tensors[i]->rowwise_scale_inv_shape();
      const size_t numel = rs.data[0] * rs.data[1];
      compareResults("multi_roundtrip_row_" + std::to_string(i),
                     final_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                     orig_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                     numel);
    } else {
      const NVTEShape cs = orig_tensors[i]->columnwise_scale_inv_shape();
      const size_t numel = cs.data[0] * cs.data[1];
      compareResults("multi_roundtrip_col_" + std::to_string(i),
                     final_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                     orig_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                     numel);
    }
  }
}

// ===================================================================
// Test suites
// ===================================================================

class MultiTensorSwizzleTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t, bool>> {};

TEST_P(MultiTensorSwizzleTestSuite, TestMultiTensorSwizzle) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  const auto rowwise = std::get<3>(GetParam());
  performTestMultiTensorSwizzle(num_tensors, M, K, rowwise);
}

class MultiTensorUnswizzleTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t, bool>> {};

TEST_P(MultiTensorUnswizzleTestSuite, TestMultiTensorUnswizzle) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  const auto rowwise = std::get<3>(GetParam());
  performTestMultiTensorUnswizzle(num_tensors, M, K, rowwise);
}

class MultiTensorRoundtripTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t, bool>> {};

TEST_P(MultiTensorRoundtripTestSuite, TestMultiTensorRoundtrip) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  const auto rowwise = std::get<3>(GetParam());
  performTestMultiTensorRoundtrip(num_tensors, M, K, rowwise);
}

namespace {

// Shapes that exercise the narrow_k kernel (rowwise) / narrow_m kernel (colwise):
//   Narrow-K fires when ALL tensors have scale num_tiles_k < TB_DIM (32),
//   i.e. padded ceil(K/32) < 128.
//   Narrow-M fires analogously for colwise when padded K < 4096
//   (since colwise m = K padded to 128, num_tiles_m = m / 128 < 32).
//
// Shapes that bypass narrow and use the regular multi_tensor kernel:
//   K >= 4096 makes num_tiles_k >= 32 (rowwise) and num_tiles_m >= 32 (colwise).

std::vector<std::tuple<int, size_t, size_t, bool>> multi_tensor_test_cases = {
    // --- Narrow path cases (K small → narrow_k for row, narrow_m for col) ---
    // M and K both aligned to 128
    {3, 256, 256, true},
    {3, 256, 256, false},
    {4, 128, 128, true},
    {4, 128, 128, false},
    // M not divisible by 128 (but must be divisible by 32 for colwise —
    // the kernel computes original_K = M / BLOCK_SIZE using floor division)
    {3, 192, 256, true},
    {3, 192, 256, false},
    {2, 64, 256, true},
    {2, 64, 256, false},
    // Larger narrow K (num_tiles_k = 8, shared mem = 128 KB)
    {2, 128, 1024, true},
    {2, 128, 1024, false},
    // K not divisible by 128
    {3, 256, 160, true},
    {3, 256, 160, false},
    // Neither M nor K divisible by 128
    {3, 192, 160, true},
    {3, 192, 160, false},
    // Minimum sizes (M=32 is the MXFP8 block size minimum for colwise)
    {2, 32, 32, true},
    {2, 32, 32, false},
    {4, 32, 64, true},
    {4, 32, 64, false},

    // --- Non-narrow path cases (K >= 4096 → regular multi_tensor kernel) ---
    {3, 256, 4096, true},
    {3, 256, 4096, false},
    {2, 128, 8192, true},
    {2, 128, 8192, false},
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    MultiTensorSwizzleTestSuite,
    ::testing::ValuesIn(multi_tensor_test_cases),
    [](const testing::TestParamInfo<MultiTensorSwizzleTestSuite::ParamType>& info) {
      return "n" + std::to_string(std::get<0>(info.param)) +
             "_M" + std::to_string(std::get<1>(info.param)) +
             "_K" + std::to_string(std::get<2>(info.param)) +
             (std::get<3>(info.param) ? "_row" : "_col");
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    MultiTensorUnswizzleTestSuite,
    ::testing::ValuesIn(multi_tensor_test_cases),
    [](const testing::TestParamInfo<MultiTensorUnswizzleTestSuite::ParamType>& info) {
      return "n" + std::to_string(std::get<0>(info.param)) +
             "_M" + std::to_string(std::get<1>(info.param)) +
             "_K" + std::to_string(std::get<2>(info.param)) +
             (std::get<3>(info.param) ? "_row" : "_col");
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    MultiTensorRoundtripTestSuite,
    ::testing::ValuesIn(multi_tensor_test_cases),
    [](const testing::TestParamInfo<MultiTensorRoundtripTestSuite::ParamType>& info) {
      return "n" + std::to_string(std::get<0>(info.param)) +
             "_M" + std::to_string(std::get<1>(info.param)) +
             "_K" + std::to_string(std::get<2>(info.param)) +
             (std::get<3>(info.param) ? "_row" : "_col");
    });
