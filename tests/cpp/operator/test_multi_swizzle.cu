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

static void set_scale_inv_padding(uint8_t *buf,
                                  size_t padded_rows, size_t padded_cols,
                                  size_t orig_rows, size_t orig_cols,
                                  uint8_t value) {
  for (size_t r = 0; r < padded_rows; ++r) {
    for (size_t c = 0; c < padded_cols; ++c) {
      if (r >= orig_rows || c >= orig_cols) {
        buf[r * padded_cols + c] = value;
      }
    }
  }
}

static void zero_scale_inv_padding(uint8_t *buf,
                                   size_t padded_rows, size_t padded_cols,
                                   size_t orig_rows, size_t orig_cols) {
  set_scale_inv_padding(buf, padded_rows, padded_cols, orig_rows, orig_cols, 0);
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
                             M, divide_round_up(K, BLOCK_SIZE));
    } else {
      const NVTEShape cs = input->columnwise_scale_inv_shape();
      zero_scale_inv_padding(input->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                             cs.data[0], cs.data[1],
                             divide_round_up(M, BLOCK_SIZE), K);
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

void performTestMultiTensorSwizzleNVFP4(const int num_tensors, const size_t M, const size_t K,
                                        bool rowwise) {
  using namespace test;
  constexpr size_t BLOCK_SIZE = 16;
  std::vector<std::unique_ptr<Tensor>> inputs;
  std::vector<std::unique_ptr<Tensor>> outputs;
  std::vector<NVTETensor> input_handles;
  std::vector<NVTETensor> output_handles;

  for (int i = 0; i < num_tensors; ++i) {
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i),
                                          std::vector<size_t>{M, K}, DType::kFloat4E2M1, rowwise,
                                          !rowwise, NVTE_NVFP4_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i),
                                           std::vector<size_t>{M, K}, DType::kFloat4E2M1, rowwise,
                                           !rowwise, NVTE_NVFP4_1D_SCALING);
    output->set_with_gemm_swizzled_scales(true);
    if (rowwise) {
      input->fill_uniform_rowwise_scale_inv();
    } else {
      input->fill_uniform_columnwise_scale_inv();
    }
    const NVTEShape scale_shape =
        rowwise ? input->rowwise_scale_inv_shape() : input->columnwise_scale_inv_shape();
    uint8_t *input_scales = rowwise ? input->rowwise_cpu_scale_inv_ptr<uint8_t>()
                                    : input->columnwise_cpu_scale_inv_ptr<uint8_t>();
    const size_t original_rows = rowwise ? M : K;
    const size_t original_cols =
        rowwise ? divide_round_up(K, BLOCK_SIZE) : divide_round_up(M, BLOCK_SIZE);
    set_scale_inv_padding(input_scales, scale_shape.data[0], scale_shape.data[1], original_rows,
                          original_cols, 0xFF);
    input->from_cpu();
    input_handles.push_back(input->data());
    output_handles.push_back(output->data());
    inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(output));
  }

  nvte_multi_tensor_swizzle_scaling_factors(input_handles.data(), output_handles.data(),
                                            num_tensors, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < num_tensors; ++i) {
    outputs[i]->to_cpu();
    const NVTEShape scale_shape =
        rowwise ? inputs[i]->rowwise_scale_inv_shape() : inputs[i]->columnwise_scale_inv_shape();
    const uint8_t *input_scales = rowwise ? inputs[i]->rowwise_cpu_scale_inv_ptr<uint8_t>()
                                          : inputs[i]->columnwise_cpu_scale_inv_ptr<uint8_t>();
    const uint8_t *actual = rowwise ? outputs[i]->rowwise_cpu_scale_inv_ptr<uint8_t>()
                                    : outputs[i]->columnwise_cpu_scale_inv_ptr<uint8_t>();
    std::vector<uint8_t> reference_input(
        input_scales, input_scales + scale_shape.data[0] * scale_shape.data[1]);
    const size_t original_rows = rowwise ? M : K;
    const size_t original_cols =
        rowwise ? divide_round_up(K, BLOCK_SIZE) : divide_round_up(M, BLOCK_SIZE);
    zero_scale_inv_padding(reference_input.data(), scale_shape.data[0], scale_shape.data[1],
                           original_rows, original_cols);
    std::vector<uint8_t> reference(scale_shape.data[0] * scale_shape.data[1]);
    compute_ref_swizzle<128, 4, true>(reference_input.data(), reference.data(), scale_shape.data[0],
                                      scale_shape.data[1]);
    compareResults("multi_nvfp4_swizzle", actual, reference.data(), reference.size());
  }
}

TEST(MultiTensorSwizzleTest, MXFP8ColumnwiseBatchIgnoresUnusedRowwiseBuffer) {
  using namespace test;
  constexpr int num_tensors = 2;
  const std::vector<size_t> shape{256, 128};
  std::vector<std::unique_ptr<Tensor>> inputs;
  std::vector<std::unique_ptr<Tensor>> outputs;
  std::vector<NVTETensor> input_handles;
  std::vector<NVTETensor> output_handles;

  for (int i = 0; i < num_tensors; ++i) {
    const bool has_rowwise_scales = i == 0;
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i), shape,
                                          DType::kFloat8E4M3, has_rowwise_scales, true,
                                          NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i), shape,
                                           DType::kFloat8E4M3, has_rowwise_scales, true,
                                           NVTE_MXFP8_1D_SCALING);
    if (has_rowwise_scales) input->fill_uniform_rowwise_scale_inv();
    input->fill_uniform_columnwise_scale_inv();
    output->set_with_gemm_swizzled_scales(true);
    input->from_cpu();
    input_handles.push_back(input->data());
    output_handles.push_back(output->data());
    inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(output));
  }

  nvte_multi_tensor_swizzle_scaling_factors(input_handles.data(), output_handles.data(),
                                            num_tensors, 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < num_tensors; ++i) {
    outputs[i]->to_cpu();
    const NVTEShape scale_shape = inputs[i]->columnwise_scale_inv_shape();
    std::vector<uint8_t> reference(scale_shape.data[0] * scale_shape.data[1]);
    compute_ref_swizzle<128, 4, false>(
        inputs[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(), reference.data(), scale_shape.data[1],
        scale_shape.data[0]);
    compareResults("multi_mixed_buffers_columnwise_" + std::to_string(i),
                   outputs[i]->columnwise_cpu_scale_inv_ptr<uint8_t>(), reference.data(),
                   reference.size());
  }
}

TEST(MultiTensorSwizzleTest, NVFP4FullWidthChunkLaunchSplitting) {
  for (const bool rowwise : {false, true}) {
    performTestMultiTensorSwizzleNVFP4(4, 160, 96, rowwise);
    performTestMultiTensorSwizzleNVFP4(4, 128, 128, rowwise);
    performTestMultiTensorSwizzleNVFP4(65, 128, 128, rowwise);
  }
}

TEST(MultiTensorSwizzleTest, RowwiseMixedSharedMemoryFallback) {
  using namespace test;
  // The first tensor fits the full-width shared-memory limit, but the second does not.
  const std::vector<std::pair<size_t, size_t>> shapes{{128, 7168}, {128, 8064}};
  std::vector<std::unique_ptr<Tensor>> inputs;
  std::vector<std::unique_ptr<Tensor>> outputs;
  std::vector<NVTETensor> input_handles;
  std::vector<NVTETensor> output_handles;
  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto [M, K] = shapes[i];
    auto input = std::make_unique<Tensor>("input_" + std::to_string(i),
                                          std::vector<size_t>{M, K}, DType::kFloat8E4M3, true,
                                          false, NVTE_MXFP8_1D_SCALING);
    auto output = std::make_unique<Tensor>("output_" + std::to_string(i),
                                           std::vector<size_t>{M, K}, DType::kFloat8E4M3, true,
                                           false, NVTE_MXFP8_1D_SCALING);
    output->set_with_gemm_swizzled_scales(true);
    input->fill_uniform_rowwise_scale_inv();
    input->from_cpu();
    input_handles.push_back(input->data());
    output_handles.push_back(output->data());
    inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(output));
  }

  nvte_multi_tensor_swizzle_scaling_factors(
      input_handles.data(), output_handles.data(), static_cast<int>(inputs.size()), 0);
  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  for (size_t i = 0; i < inputs.size(); ++i) {
    outputs[i]->to_cpu();
    const NVTEShape scale_shape = inputs[i]->rowwise_scale_inv_shape();
    std::vector<uint8_t> reference(scale_shape.data[0] * scale_shape.data[1]);
    compute_ref_swizzle<128, 4, true>(inputs[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                                      reference.data(), scale_shape.data[0], scale_shape.data[1]);
    compareResults("multi_rowwise_fallback", outputs[i]->rowwise_cpu_scale_inv_ptr<uint8_t>(),
                   reference.data(), reference.size());
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
                             M, divide_round_up(K, BLOCK_SIZE));
    } else {
      const NVTEShape cs = orig->columnwise_scale_inv_shape();
      zero_scale_inv_padding(orig->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                             cs.data[0], cs.data[1],
                             divide_round_up(M, BLOCK_SIZE), K);
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
                             M, divide_round_up(K, BLOCK_SIZE));
    } else {
      const NVTEShape cs = orig->columnwise_scale_inv_shape();
      zero_scale_inv_padding(orig->columnwise_cpu_scale_inv_ptr<uint8_t>(),
                             cs.data[0], cs.data[1],
                             divide_round_up(M, BLOCK_SIZE), K);
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
  const auto& [num_tensors, M, K, rowwise] = GetParam();
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

std::vector<std::tuple<int, size_t, size_t, bool>> multi_tensor_test_cases = {
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
    // Larger K
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

    {3, 256, 4096, true},
    {3, 256, 4096, false},
    {2, 128, 8192, true},
    {2, 128, 8192, false},
    // Rowwise num_tiles_k = 34 selects vec_load_size = 2 (int2 kernel)
    {2, 128, 4352, true},
    // Wide row-wise scales fall back for an underfilled grid and use the full-width kernel once
    // the aggregate grid is large enough.
    {2, 2048, 7168, true},   // 32 aggregate row tiles: regular fallback.
    {16, 2048, 7168, true},  // 256 aggregate row tiles: rowwise full-width kernel.
    // Colwise num_tiles_k = 512 / 32 / 4 = 4 selects vec_load_size = 4 (int4 kernel)
    {2, 512, 4096, false},
    // MXFP8 swizzle: more tensors than one launch takes, several row-wise tiles per CTA,
    // and 128-row column-wise chunks
    {80, 128, 128, true},
    {80, 128, 128, false},
    {20, 16384, 128, true},
    {20, 128, 8192, false},
    // A full 128-row column-wise chunk and a clipped last one (132 scale rows)
    {20, 4224, 2048, false},
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
