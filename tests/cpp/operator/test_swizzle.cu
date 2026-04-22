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

constexpr int MAT_TILE_DIM_M = 128;
constexpr int MAT_TILE_DIM_K = 128;

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_swizzle(const uint8_t *h_input, uint8_t *h_output,
                         const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_output_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int out_index = tile_output_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[out_index] = h_input[k + m * K];
      else
        h_output[out_index] = h_input[k * M + m];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K, bool row_scaling>
void compute_ref_unswizzle(const uint8_t *h_input, uint8_t *h_output,
                           const size_t M, const size_t K) {

  constexpr int NEW_SF_TILE_DIM_M = SF_TILE_DIM_M / 4;
  constexpr int NEW_SF_TILE_DIM_K = SF_TILE_DIM_K * 4;
  constexpr int SF_TILE_SIZE = SF_TILE_DIM_M * SF_TILE_DIM_K;

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {

      int tile_id_m = m / SF_TILE_DIM_M;
      int tile_id_k = k / SF_TILE_DIM_K;
      int m_in_tile = m % SF_TILE_DIM_M;
      int k_in_tile = k % SF_TILE_DIM_K;

      int row_in_new_tile = m_in_tile % NEW_SF_TILE_DIM_M;
      int col_in_new_tile = m_in_tile / NEW_SF_TILE_DIM_M * SF_TILE_DIM_K + k_in_tile;

      int tile_input_ptr = tile_id_m * SF_TILE_DIM_M * K + tile_id_k * SF_TILE_SIZE;
      int in_index = tile_input_ptr + row_in_new_tile * NEW_SF_TILE_DIM_K + col_in_new_tile;
      if constexpr(row_scaling)
        h_output[k + m * K] = h_input[in_index];
      else
        h_output[k * M + m] = h_input[in_index];
    }
  }
}

void performTestSwizzle1D(const int num_tiles_M, const int num_tiles_K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if ((rowwise && columnwise) || !(rowwise || columnwise)){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + "is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const size_t M = num_tiles_M * MAT_TILE_DIM_M;
  const size_t K = num_tiles_K * MAT_TILE_DIM_K;
  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  const auto scale_shape = std::vector<size_t>{data_shape[0] / SF_MODE_X, data_shape[1] /SF_MODE_Y};

  std::vector<int> scaling_mode = {SF_MODE_X, SF_MODE_Y, 0};
  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  output.set_with_gemm_swizzled_scales(true);

  fillUniform(&input);

  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(scale_shape[0] * scale_shape[1]);

  nvte_swizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_swizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0], scale_shape[1]);
  else
    compute_ref_swizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[1], scale_shape[0]);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_swizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  } else {
    compareResults("output_swizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), scale_shape[0] * scale_shape[1]);
  }
}

void performTestUnswizzle1D(const size_t M, const size_t K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if (!rowwise && !columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, Either rowwise or columnwise scaling mode must be true.";
  }
  if (rowwise && columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + " is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};

  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  input.set_with_gemm_swizzled_scales(true);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  // Use the actual padded compact scale shape from the tensor for both the reference
  // and the comparison. This correctly covers padded cases where M is not a multiple
  // of 128 or K/32 is not a multiple of 4.
  const auto padded_scale_shape = rowwise
    ? input.rowwise_scale_inv_shape()
    : input.columnwise_scale_inv_shape();
  const size_t padded_dim0 = padded_scale_shape.data[0];
  const size_t padded_dim1 = padded_scale_shape.data[1];
  std::unique_ptr<uint8_t[]> ref_output = std::make_unique<uint8_t[]>(padded_dim0 * padded_dim1);

  nvte_unswizzle_scaling_factors(input.data(), output.data(), 0);

  if (rowwise)
    compute_ref_unswizzle<128, 4, true>(input.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0, padded_dim1);
  else
    compute_ref_unswizzle<128, 4, false>(input.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim1, padded_dim0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  output.to_cpu();
  if (rowwise) {
    compareResults("output_unswizzle", output.rowwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0 * padded_dim1);
  } else {
    compareResults("output_unswizzle", output.columnwise_cpu_scale_inv_ptr<uint8_t>(), ref_output.get(), padded_dim0 * padded_dim1);
  }
}

// Zero out padding in a scale_inv CPU buffer so that the CPU reference
// matches the kernel, which zeroes elements outside the original dims.
// The buffer is stored in leading-dim-major order (row-major for rowwise,
// column-major for colwise).  `padded_rows x padded_cols` is the full
// (padded) shape; `orig_rows` / `orig_cols` are the unpadded extents.
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

void performTestGroupedSwizzleMXFP8(const int num_tensors, const size_t M, const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<Tensor*> input_ptrs;
  std::vector<Tensor*> output_ptrs;
  input_tensors.reserve(num_tensors);
  output_tensors.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  output_ptrs.reserve(num_tensors);

  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};
  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, true, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, true, true,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    fillUniform(output.get());

    // The grouped swizzle kernel zeroes scale_inv elements that fall
    // outside the original (unpadded) dimensions.  Mirror that in the
    // per-tensor CPU buffers so the CPU reference produces identical output.
    input->to_cpu();
    const NVTEShape rs = input->rowwise_scale_inv_shape();
    zero_scale_inv_padding(input->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                           rs.data[0], rs.data[1],
                           M, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const NVTEShape cs = input->columnwise_scale_inv_shape();
    zero_scale_inv_padding(input->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                           cs.data[0], cs.data[1],
                           (M + BLOCK_SIZE - 1) / BLOCK_SIZE, K);
    input->from_cpu();

    input_ptrs.push_back(input.get());
    output_ptrs.push_back(output.get());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(output_ptrs, NVTE_MXFP8_1D_SCALING);
  const uint8_t input_swizzled = 0;
  nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &input_swizzled, sizeof(input_swizzled));
  const uint8_t output_swizzled = 1;
  nvte_set_grouped_tensor_param(grouped_output.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &output_swizzled, sizeof(output_swizzled));

  const NVTEShape row_shape = input_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = input_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  NVTE_CHECK_CUDA(cudaMemset(grouped_output.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_output.columnwise_scale_inv.get(), 0, num_tensors * col_numel));

  nvte_swizzle_grouped_scaling_factors(grouped_input.get_handle(),
                                       grouped_output.get_handle(), 0);

  std::vector<uint8_t> output_row(num_tensors * row_numel);
  std::vector<uint8_t> output_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(output_row.data(), grouped_output.scale_inv.get(),
                             output_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(output_col.data(), grouped_output.columnwise_scale_inv.get(),
                             output_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    compute_ref_swizzle<128, 4, true>(input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                      ref_row.data() + i * row_numel,
                                      row_shape.data[0], row_shape.data[1]);
    compute_ref_swizzle<128, 4, false>(
        input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_col.data() + i * col_numel,
        col_shape.data[1], col_shape.data[0]);
  }

  compareResults("grouped_swizzle_rowwise", output_row.data(), ref_row.data(),
                 num_tensors * row_numel);
  compareResults("grouped_swizzle_colwise", output_col.data(), ref_col.data(),
                 num_tensors * col_numel);
}

class SwizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<int, int>, std::pair<bool, bool>, bool>> {};


TEST_P(SwizzleTestSuite, TestSwizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto num_tiles = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzle1D(num_tiles.first, num_tiles.second,
                       scaling_mode.first, scaling_mode.second,
                       transa);
}

class UnswizzleTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, std::pair<bool, bool>, bool>> {};

TEST_P(UnswizzleTestSuite, TestUnswizzle) {
    using namespace transformer_engine;
    using namespace test;

  const auto data_shape = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestUnswizzle1D(data_shape.first, data_shape.second,
                         scaling_mode.first, scaling_mode.second,
                         transa);
}

void performTestGroupedUnswizzleMXFP8(const int num_tensors, const size_t M, const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  std::vector<std::unique_ptr<Tensor>> input_tensors;
  std::vector<std::unique_ptr<Tensor>> output_tensors;
  std::vector<Tensor*> input_ptrs;
  std::vector<Tensor*> output_ptrs;
  input_tensors.reserve(num_tensors);
  output_tensors.reserve(num_tensors);
  input_ptrs.reserve(num_tensors);
  output_ptrs.reserve(num_tensors);

  const std::vector<size_t> shape{M, K};
  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, true, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, true, true,
                                           NVTE_MXFP8_1D_SCALING);
    fillUniform(input.get());
    fillUniform(output.get());

    input_ptrs.push_back(input.get());
    output_ptrs.push_back(output.get());
    input_tensors.emplace_back(std::move(input));
    output_tensors.emplace_back(std::move(output));
  }

  GroupedBuffers grouped_input = build_grouped_tensor(input_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_output = build_grouped_tensor(output_ptrs, NVTE_MXFP8_1D_SCALING);
  const uint8_t input_swizzled = 1;
  nvte_set_grouped_tensor_param(grouped_input.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &input_swizzled, sizeof(input_swizzled));
  const uint8_t output_swizzled = 0;
  nvte_set_grouped_tensor_param(grouped_output.get_handle(),
                                kNVTEGroupedWithGEMMSwizzledScales,
                                &output_swizzled, sizeof(output_swizzled));

  const NVTEShape row_shape = input_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = input_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  NVTE_CHECK_CUDA(cudaMemset(grouped_output.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_output.columnwise_scale_inv.get(), 0, num_tensors * col_numel));

  nvte_unswizzle_grouped_scaling_factors(grouped_input.get_handle(),
                                         grouped_output.get_handle(), 0);

  std::vector<uint8_t> output_row(num_tensors * row_numel);
  std::vector<uint8_t> output_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(output_row.data(), grouped_output.scale_inv.get(),
                             output_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(output_col.data(), grouped_output.columnwise_scale_inv.get(),
                             output_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    compute_ref_unswizzle<128, 4, true>(input_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                        ref_row.data() + i * row_numel,
                                        row_shape.data[0], row_shape.data[1]);
    compute_ref_unswizzle<128, 4, false>(
        input_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(),
        ref_col.data() + i * col_numel,
        col_shape.data[1], col_shape.data[0]);
  }

  compareResults("grouped_unswizzle_rowwise", output_row.data(), ref_row.data(),
                 num_tensors * row_numel);
  compareResults("grouped_unswizzle_colwise", output_col.data(), ref_col.data(),
                 num_tensors * col_numel);
}

void performTestGroupedSwizzleUnswizzleRoundtrip(const int num_tensors, const size_t M,
                                                  const size_t K) {
  using namespace transformer_engine;
  using namespace test;

  constexpr size_t BLOCK_SIZE = 32;
  const std::vector<size_t> shape{M, K};

  std::vector<std::unique_ptr<Tensor>> orig_tensors, mid_tensors, final_tensors;
  std::vector<Tensor*> orig_ptrs, mid_ptrs, final_ptrs;
  orig_tensors.reserve(num_tensors);
  mid_tensors.reserve(num_tensors);
  final_tensors.reserve(num_tensors);

  for (int i = 0; i < num_tensors; ++i) {
    auto orig = std::make_unique<Tensor>("orig_" + std::to_string(i), shape,
                                         DType::kFloat8E4M3, true, true, NVTE_MXFP8_1D_SCALING);
    auto mid = std::make_unique<Tensor>("mid_" + std::to_string(i), shape,
                                        DType::kFloat8E4M3, true, true, NVTE_MXFP8_1D_SCALING);
    auto fin = std::make_unique<Tensor>("fin_" + std::to_string(i), shape,
                                        DType::kFloat8E4M3, true, true, NVTE_MXFP8_1D_SCALING);
    fillUniform(orig.get());

    // Zero padding so the round-trip comparison is exact.
    orig->to_cpu();
    const NVTEShape rs = orig->rowwise_scale_inv_shape();
    zero_scale_inv_padding(orig->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                           rs.data[0], rs.data[1],
                           M, (K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const NVTEShape cs = orig->columnwise_scale_inv_shape();
    zero_scale_inv_padding(orig->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                           cs.data[0], cs.data[1],
                           (M + BLOCK_SIZE - 1) / BLOCK_SIZE, K);
    orig->from_cpu();

    orig_ptrs.push_back(orig.get());
    mid_ptrs.push_back(mid.get());
    final_ptrs.push_back(fin.get());
    orig_tensors.emplace_back(std::move(orig));
    mid_tensors.emplace_back(std::move(mid));
    final_tensors.emplace_back(std::move(fin));
  }

  GroupedBuffers grouped_orig = build_grouped_tensor(orig_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_mid = build_grouped_tensor(mid_ptrs, NVTE_MXFP8_1D_SCALING);
  GroupedBuffers grouped_fin = build_grouped_tensor(final_ptrs, NVTE_MXFP8_1D_SCALING);

  const NVTEShape row_shape = orig_tensors[0]->rowwise_scale_inv_shape();
  const NVTEShape col_shape = orig_tensors[0]->columnwise_scale_inv_shape();
  const size_t row_numel = row_shape.data[0] * row_shape.data[1];
  const size_t col_numel = col_shape.data[0] * col_shape.data[1];

  const uint8_t no_swizzle = 0, has_swizzle = 1;
  nvte_set_grouped_tensor_param(grouped_orig.get_handle(), kNVTEGroupedWithGEMMSwizzledScales,
                                &no_swizzle, sizeof(no_swizzle));
  nvte_set_grouped_tensor_param(grouped_mid.get_handle(), kNVTEGroupedWithGEMMSwizzledScales,
                                &has_swizzle, sizeof(has_swizzle));
  nvte_set_grouped_tensor_param(grouped_fin.get_handle(), kNVTEGroupedWithGEMMSwizzledScales,
                                &no_swizzle, sizeof(no_swizzle));

  NVTE_CHECK_CUDA(cudaMemset(grouped_mid.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_mid.columnwise_scale_inv.get(), 0, num_tensors * col_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_fin.scale_inv.get(), 0, num_tensors * row_numel));
  NVTE_CHECK_CUDA(cudaMemset(grouped_fin.columnwise_scale_inv.get(), 0, num_tensors * col_numel));

  nvte_swizzle_grouped_scaling_factors(grouped_orig.get_handle(), grouped_mid.get_handle(), 0);
  nvte_unswizzle_grouped_scaling_factors(grouped_mid.get_handle(), grouped_fin.get_handle(), 0);

  std::vector<uint8_t> result_row(num_tensors * row_numel);
  std::vector<uint8_t> result_col(num_tensors * col_numel);
  NVTE_CHECK_CUDA(cudaMemcpy(result_row.data(), grouped_fin.scale_inv.get(),
                             result_row.size(), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(result_col.data(), grouped_fin.columnwise_scale_inv.get(),
                             result_col.size(), cudaMemcpyDeviceToHost));

  std::vector<uint8_t> ref_row(num_tensors * row_numel);
  std::vector<uint8_t> ref_col(num_tensors * col_numel);
  for (int i = 0; i < num_tensors; ++i) {
    memcpy(ref_row.data() + i * row_numel,
           orig_tensors[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(), row_numel);
    memcpy(ref_col.data() + i * col_numel,
           orig_tensors[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(), col_numel);
  }

  compareResults("grouped_roundtrip_rowwise", result_row.data(), ref_row.data(),
                 num_tensors * row_numel);
  compareResults("grouped_roundtrip_colwise", result_col.data(), ref_col.data(),
                 num_tensors * col_numel);
}

class SwizzleGroupedTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(SwizzleGroupedTestSuite, TestGroupedSwizzleMXFP8) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedSwizzleMXFP8(num_tensors, M, K);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleGroupedTestSuite,
  ::testing::Values(
    // M and K both divisible by 128
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    // M not divisible by 128
    std::make_tuple(3, 200, 256),
    std::make_tuple(2, 65, 256),
    // K not divisible by 128
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    // Neither M nor K divisible by 128
    std::make_tuple(3, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32)
  ),
  [](const testing::TestParamInfo<SwizzleGroupedTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

class UnswizzleGroupedTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(UnswizzleGroupedTestSuite, TestGroupedUnswizzleMXFP8) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedUnswizzleMXFP8(num_tensors, M, K);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  UnswizzleGroupedTestSuite,
  ::testing::Values(
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    std::make_tuple(3, 200, 256),
    std::make_tuple(2, 65, 256),
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    std::make_tuple(3, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32)
  ),
  [](const testing::TestParamInfo<UnswizzleGroupedTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

class SwizzleUnswizzleGroupedRoundtripTestSuite
    : public ::testing::TestWithParam<std::tuple<int, size_t, size_t>> {};

TEST_P(SwizzleUnswizzleGroupedRoundtripTestSuite, TestGroupedSwizzleUnswizzleRoundtrip) {
  const auto num_tensors = std::get<0>(GetParam());
  const auto M = std::get<1>(GetParam());
  const auto K = std::get<2>(GetParam());
  performTestGroupedSwizzleUnswizzleRoundtrip(num_tensors, M, K);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleUnswizzleGroupedRoundtripTestSuite,
  ::testing::Values(
    std::make_tuple(3, 256, 256),
    std::make_tuple(4, 128, 128),
    std::make_tuple(3, 200, 256),
    std::make_tuple(2, 65, 256),
    std::make_tuple(3, 256, 160),
    std::make_tuple(2, 256, 96),
    std::make_tuple(3, 200, 160),
    std::make_tuple(4, 33, 64),
    std::make_tuple(2, 1, 32)
  ),
  [](const testing::TestParamInfo<SwizzleUnswizzleGroupedRoundtripTestSuite::ParamType>& info) {
    return "n" + std::to_string(std::get<0>(info.param)) +
           "_M" + std::to_string(std::get<1>(info.param)) +
           "_K" + std::to_string(std::get<2>(info.param));
  }
);

namespace {

std::vector<std::pair<int, int>> num_tiles = {
  {1, 1},
  {1, 132},
  {132, 1},
  {65, 256},
  {65, 257},
  {65, 258},
  {65, 259},
  // Additional narrow-path coverage: narrow_k (row) when num_tiles_K < 32,
  // narrow_m (col) when num_tiles_M < 32.
  {1, 4},     // narrow_k with 4 K-tiles
  {1, 8},     // narrow_k with 8 K-tiles
  {4, 1},     // narrow_m with 4 M-tiles
  {8, 1},     // narrow_m with 8 M-tiles
  {31, 1},    // narrow_m at boundary (31 < TB_DIM=32)
  {1, 31},    // narrow_k at boundary (31 < TB_DIM=32)
};

// Raw {M, K} data shapes for unswizzle tests. Includes aligned cases (scale dims
// already multiples of 128 and 4) and padded cases where M or K/32 are not yet
// aligned, forcing the compact scale_inv to carry a padded tail.
// All K values must be multiples of 32 (MXFP8 block size).
std::vector<std::pair<size_t, size_t>> unswizzle_data_shapes = {
  // Aligned: scale dims are already multiples of 128 and 4
  {128, 128},
  {128, 16896},   // K = 132 * 128, large K
  {16896, 128},   // M = 132 * 128, large M
  // M-padding only: M not a multiple of 128 (scale-M needs padding to 256)
  {160, 128},
  // scale-K padding only: K/32 = 3, padded to 4
  {128, 96},
  // Both M and scale-K need padding
  {160, 96},
  {16896, 16896},
};

std::vector<std::pair<bool, bool>> scaling_mode = {
  {true, false},
  {false, true}
};

std::vector<bool> transa = {true, false};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(num_tiles),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleTestSuite::ParamType>& info) {
    std::string name = "ntiles" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  UnswizzleTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(unswizzle_data_shapes),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<UnswizzleTestSuite::ParamType>& info) {
    std::string name = "MK" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });

void performTestSwizzleUnswizzleRoundtrip(const size_t M, const size_t K, bool rowwise, bool columnwise, const bool transa) {
  using namespace test;

  int SF_MODE_X, SF_MODE_Y;
  if (rowwise) {
    SF_MODE_X = 1;
    SF_MODE_Y = 32;
  }
  if (columnwise) {
    SF_MODE_X = 32;
    SF_MODE_Y = 1;
  }

  if (!rowwise && !columnwise) {
    GTEST_SKIP() << "TEST SKIPPED, Either rowwise or columnwise scaling mode must be true.";
  }
  if (rowwise && columnwise){
    GTEST_SKIP() << "TEST SKIPPED, The scaling mode " + std::to_string(SF_MODE_X) + "x" +
      std::to_string(SF_MODE_Y) + " is not implemented.";
  }

  DType dtype = DType::kFloat8E4M3;

  const auto data_shape = transa ? std::vector<size_t>{M, K} : std::vector<size_t>{K, M};
  const size_t logical_dim0 = data_shape[0] / SF_MODE_X;
  const size_t logical_dim1 = data_shape[1] / SF_MODE_Y;

  Tensor input("input", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  Tensor swizzled("swizzled", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);
  swizzled.set_with_gemm_swizzled_scales(true);
  Tensor output("output", data_shape, dtype, rowwise, columnwise, NVTE_MXFP8_1D_SCALING);

  fillUniform(&input);

  // fillUniform fills all scale_inv entries including the padded region with random bytes.
  // After swizzle, the swizzle kernel zeroes padded positions in the swizzled output, so
  // after unswizzle those positions come back as zero in the compact output. Zero them in
  // the input now so the full-buffer comparison is valid.
  const auto padded_scale_shape = rowwise
    ? input.rowwise_scale_inv_shape()
    : input.columnwise_scale_inv_shape();
  const size_t padded_dim0 = padded_scale_shape.data[0];
  const size_t padded_dim1 = padded_scale_shape.data[1];

  if (padded_dim0 != logical_dim0 || padded_dim1 != logical_dim1) {
    auto* scale_ptr = rowwise
      ? input.rowwise_cpu_scale_inv_ptr<uint8_t>()
      : input.columnwise_cpu_scale_inv_ptr<uint8_t>();
    for (size_t r = 0; r < padded_dim0; r++) {
      for (size_t c = 0; c < padded_dim1; c++) {
        if (r >= logical_dim0 || c >= logical_dim1) {
          scale_ptr[r * padded_dim1 + c] = 0;
        }
      }
    }
    input.from_cpu();
  }

  nvte_swizzle_scaling_factors(input.data(), swizzled.data(), 0);
  nvte_unswizzle_scaling_factors(swizzled.data(), output.data(), 0);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  input.to_cpu();
  output.to_cpu();
  if (rowwise) {
    compareResults("roundtrip_rowwise", output.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                   input.rowwise_cpu_scale_inv_ptr<uint8_t>(), padded_dim0 * padded_dim1);
  } else {
    compareResults("roundtrip_columnwise", output.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                   input.columnwise_cpu_scale_inv_ptr<uint8_t>(), padded_dim0 * padded_dim1);
  }
}

class SwizzleUnswizzleRoundtripTestSuite : public ::testing::TestWithParam<std::tuple<std::pair<size_t, size_t>, std::pair<bool, bool>, bool>> {};

TEST_P(SwizzleUnswizzleRoundtripTestSuite, TestSwizzleUnswizzleRoundtrip) {
  using namespace transformer_engine;
  using namespace test;

  const auto data_shape = std::get<0>(GetParam());
  const auto scaling_mode = std::get<1>(GetParam());
  const auto transa = std::get<2>(GetParam());

  performTestSwizzleUnswizzleRoundtrip(data_shape.first, data_shape.second,
                                       scaling_mode.first, scaling_mode.second,
                                       transa);
}

INSTANTIATE_TEST_SUITE_P(
  OperatorTest,
  SwizzleUnswizzleRoundtripTestSuite,
  ::testing::Combine(
    ::testing::ValuesIn(unswizzle_data_shapes),
    ::testing::ValuesIn(scaling_mode),
    ::testing::ValuesIn(transa)
  ),
  [](const testing::TestParamInfo<SwizzleUnswizzleRoundtripTestSuite::ParamType>& info) {
    std::string name = "roundtrip_MK" +
      std::to_string(std::get<0>(info.param).first) + "X" +
      std::to_string(std::get<0>(info.param).second) + "smode" +
      std::to_string(std::get<1>(info.param).first) + "X"+
      std::to_string(std::get<1>(info.param).second) + "trans" +
      std::to_string(std::get<2>(info.param));
    return name;
    });
