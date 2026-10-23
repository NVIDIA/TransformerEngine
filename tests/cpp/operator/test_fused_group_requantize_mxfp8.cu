/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/swizzle.h>

#include "../test_common.h"

namespace {

using namespace transformer_engine;
using namespace test;

using transformer_engine::DType;
using transformer_engine::QuantizationConfigWrapper;
using test::Tensor;
using test::bf16;
using test::fp32;
using test::fp8e4m3;
using test::fp8e8m0;
using test::int64;

constexpr size_t MXFP8_SCALE_DIM = 32;
constexpr uint8_t kSentinel = 0xAB;

struct FusedGroupRequantizeCase {
  size_t hidden_size;
  std::vector<size_t> splits;  // per-group LIVE row counts, each a multiple of 128
  size_t capacity_rows;        // allocated rows; 0 means sum(splits) (no tail)
};

size_t product(const NVTEShape &shape) {
  size_t result = 1;
  for (size_t i = 0; i < shape.ndim; ++i) {
    result *= shape.data[i];
  }
  return result;
}

class FusedGroupRequantizeTestSuite
    : public ::testing::TestWithParam<
          std::tuple<FusedGroupRequantizeCase, DType, bool, bool>> {};

TEST_P(FusedGroupRequantizeTestSuite, MatchesUnfusedChainReference) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP() << "Fused grouped MXFP8 requantization requires Blackwell or newer";
  }

  const auto test_case = std::get<0>(GetParam());
  const DType input_type = std::get<1>(GetParam());
  const bool use_fast_math = std::get<2>(GetParam());
  const bool return_dequantized = std::get<3>(GetParam());

  const size_t hidden_size = test_case.hidden_size;
  const std::vector<size_t> &splits = test_case.splits;
  const size_t num_groups = splits.size();
  const size_t num_live_rows =
      std::accumulate(splits.begin(), splits.end(), static_cast<size_t>(0));
  const size_t num_rows =
      test_case.capacity_rows != 0 ? test_case.capacity_rows : num_live_rows;

  ASSERT_EQ(hidden_size % 128, 0);
  for (const size_t split : splits) {
    ASSERT_EQ(split % 128, 0);
  }
  ASSERT_GT(num_live_rows, 0);
  ASSERT_GE(num_rows, num_live_rows);
  ASSERT_EQ(num_rows % 128, 0);

  const std::vector<size_t> shape{num_rows, hidden_size};

  // Deterministic, position-dependent data over the FULL capacity (including any
  // tail): a missing tail guard would then read plausible data and corrupt live
  // scales, which the sentinel checks below catch.
  Tensor input_fp32("input_fp32", shape, DType::kFloat32);
  auto *input_fp32_cpu = input_fp32.rowwise_cpu_dptr<fp32>();
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t col = 0; col < hidden_size; ++col) {
      const int value = static_cast<int>((131 * row + 17 * col + (row * col) % 29) % 257) - 128;
      input_fp32_cpu[row * hidden_size + col] = static_cast<float>(value) / 32.0f;
    }
  }
  input_fp32.from_cpu();

  // The wire tensor: rowwise MXFP8 with compact (unswizzled) scales. Under the
  // /128 contract TE's padded scale allocation degenerates to the compact layout.
  Tensor input_mxfp8("input_mxfp8", shape, input_type, /*rowwise=*/true,
                     /*columnwise=*/false, NVTE_MXFP8_1D_SCALING);
  nvte_quantize(input_fp32.data(), input_mxfp8.data(), 0);

  // Element-based exclusive-cumsum offsets (the grouped tensor's tensor_offsets
  // convention): offsets[g] = row offset x hidden, num_groups + 1 entries, on
  // the device. offsets[num_groups] covers only the LIVE rows.
  Tensor tensor_offsets("tensor_offsets", std::vector<size_t>{num_groups + 1}, DType::kInt64);
  auto *tensor_offsets_cpu = tensor_offsets.rowwise_cpu_dptr<int64>();
  tensor_offsets_cpu[0] = 0;
  for (size_t g = 0; g < num_groups; ++g) {
    tensor_offsets_cpu[g + 1] =
        tensor_offsets_cpu[g] + static_cast<int64>(splits[g] * hidden_size);
  }
  tensor_offsets.from_cpu();

  // Destination: columnwise E4M3 + per-group swizzled columnwise scales, plus the
  // rowwise scale buffer receiving the swizzled copy of the input's scales.
  Tensor actual("actual", shape, DType::kFloat8E4M3, /*rowwise=*/true,
                /*columnwise=*/true, NVTE_MXFP8_1D_SCALING);
  actual.set_with_gemm_swizzled_scales(true);

  Tensor dequantized_out("dequantized_out",
                         return_dequantized ? shape : std::vector<size_t>{128, 128},
                         DType::kBFloat16);

  // Sentinel-fill every output the kernel writes, so both unwritten-tail and
  // stray-write behavior are observable.
  const NVTEBasicTensor actual_rowwise_scales_param =
      nvte_get_tensor_param(actual.data(), kNVTERowwiseScaleInv);
  const NVTEBasicTensor actual_colwise_scales_param =
      nvte_get_tensor_param(actual.data(), kNVTEColumnwiseScaleInv);
  const size_t rowwise_scales_alloc = product(actual_rowwise_scales_param.shape);
  const size_t colwise_scales_alloc = product(actual_colwise_scales_param.shape);
  ASSERT_EQ(cudaMemset(actual.columnwise_dptr(), kSentinel, num_rows * hidden_size),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(actual_rowwise_scales_param.data_ptr, kSentinel, rowwise_scales_alloc),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(actual_colwise_scales_param.data_ptr, kSentinel, colwise_scales_alloc),
            cudaSuccess);
  if (return_dequantized) {
    ASSERT_EQ(cudaMemset(dequantized_out.rowwise_dptr(), kSentinel,
                         num_rows * hidden_size * sizeof(bf16)),
              cudaSuccess);
  }

  QuantizationConfigWrapper fused_config;
  fused_config.set_use_fast_math(use_fast_math);
  nvte_group_requantize(input_mxfp8.data(), actual.data(), tensor_offsets.data(),
                                    return_dequantized ? dequantized_out.data() : nullptr,
                                    fused_config, 0);

  // Reference intermediate: the unfused chain materializes the dequantized tensor;
  // fast math rounds it to BF16, the default path keeps FP32.
  const DType intermediate_type = use_fast_math ? DType::kBFloat16 : DType::kFloat32;
  Tensor dequantized_ref("dequantized_ref", shape, intermediate_type);
  nvte_dequantize(input_mxfp8.data(), dequantized_ref.data(), 0);
  dequantized_ref.to_cpu();

  // Rowwise-scale reference: the production dense swizzle over the full capacity.
  // For 128-aligned groups its live prefix is byte-identical to the per-group
  // layout (tiles are row-major over 128-row tiles, and the live bound is
  // 128-aligned).
  Tensor rowwise_swizzled_ref("rowwise_swizzled_ref", shape, input_type, /*rowwise=*/true,
                              /*columnwise=*/false, NVTE_MXFP8_1D_SCALING);
  rowwise_swizzled_ref.set_with_gemm_swizzled_scales(true);
  ASSERT_EQ(cudaMemcpy(rowwise_swizzled_ref.rowwise_dptr(), input_mxfp8.rowwise_dptr(),
                       num_rows * hidden_size, cudaMemcpyDeviceToDevice),
            cudaSuccess);
  nvte_swizzle_scaling_factors(input_mxfp8.data(), rowwise_swizzled_ref.data(), 0);

  // Columnwise reference, one group at a time through the production single-tensor
  // kernels: slice the intermediate, quantize columnwise, swizzle. Group blocks
  // concatenate exactly (no padding) because every count is a multiple of 128.
  std::vector<uint8_t> reference_colwise_data(num_live_rows * hidden_size);
  std::vector<uint8_t> reference_colwise_scales(num_live_rows / MXFP8_SCALE_DIM * hidden_size);
  size_t row_offset = 0;
  for (size_t g = 0; g < num_groups; ++g) {
    const size_t group_rows = splits[g];
    if (group_rows == 0) {
      continue;
    }
    const std::vector<size_t> group_shape{group_rows, hidden_size};

    Tensor group_intermediate("group_intermediate", group_shape, intermediate_type);
    const size_t element_size = use_fast_math ? sizeof(bf16) : sizeof(fp32);
    const uint8_t *intermediate_cpu = nullptr;
    uint8_t *group_intermediate_cpu = nullptr;
    if (use_fast_math) {
      intermediate_cpu =
          reinterpret_cast<const uint8_t *>(dequantized_ref.rowwise_cpu_dptr<bf16>());
      group_intermediate_cpu =
          reinterpret_cast<uint8_t *>(group_intermediate.rowwise_cpu_dptr<bf16>());
    } else {
      intermediate_cpu =
          reinterpret_cast<const uint8_t *>(dequantized_ref.rowwise_cpu_dptr<fp32>());
      group_intermediate_cpu =
          reinterpret_cast<uint8_t *>(group_intermediate.rowwise_cpu_dptr<fp32>());
    }
    memcpy(group_intermediate_cpu, intermediate_cpu + row_offset * hidden_size * element_size,
           group_rows * hidden_size * element_size);
    group_intermediate.from_cpu();

    Tensor group_quantized("group_quantized", group_shape, DType::kFloat8E4M3,
                           /*rowwise=*/false, /*columnwise=*/true, NVTE_MXFP8_1D_SCALING);
    nvte_quantize(group_intermediate.data(), group_quantized.data(), 0);

    Tensor group_swizzled("group_swizzled", group_shape, DType::kFloat8E4M3,
                          /*rowwise=*/false, /*columnwise=*/true, NVTE_MXFP8_1D_SCALING);
    group_swizzled.set_with_gemm_swizzled_scales(true);
    ASSERT_EQ(cudaMemcpy(group_swizzled.columnwise_dptr(), group_quantized.columnwise_dptr(),
                         group_rows * hidden_size, cudaMemcpyDeviceToDevice),
              cudaSuccess);
    nvte_swizzle_scaling_factors(group_quantized.data(), group_swizzled.data(), 0);

    group_swizzled.to_cpu();
    const auto *group_data =
        reinterpret_cast<const uint8_t *>(group_swizzled.columnwise_cpu_dptr<fp8e4m3>());
    memcpy(reference_colwise_data.data() + row_offset * hidden_size, group_data,
           group_rows * hidden_size);
    const auto *group_scales = reinterpret_cast<const uint8_t *>(
        group_swizzled.columnwise_cpu_scale_inv_ptr<fp8e8m0>());
    memcpy(reference_colwise_scales.data() + row_offset / MXFP8_SCALE_DIM * hidden_size,
           group_scales, group_rows / MXFP8_SCALE_DIM * hidden_size);

    row_offset += group_rows;
  }

  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);

  actual.to_cpu();

  // Columnwise data: live rows must match the per-group production chain bit for
  // bit; capacity-tail rows must be untouched.
  const auto *actual_data =
      reinterpret_cast<const uint8_t *>(actual.columnwise_cpu_dptr<fp8e4m3>());
  for (size_t row = 0; row < num_live_rows; ++row) {
    for (size_t col = 0; col < hidden_size; ++col) {
      const size_t i = row * hidden_size + col;
      ASSERT_EQ(actual_data[i], reference_colwise_data[i])
          << "FP8 columnwise data mismatch at row " << row << ", column " << col;
    }
  }
  for (size_t i = num_live_rows * hidden_size; i < num_rows * hidden_size; ++i) {
    ASSERT_EQ(actual_data[i], kSentinel)
        << "Capacity-tail columnwise data byte was written at flat index " << i;
  }

  // Columnwise scales: the per-group swizzled blocks fill exactly the live
  // prefix; everything past it (including any allocation padding) stays sentinel.
  const auto *actual_colwise_scales = reinterpret_cast<const uint8_t *>(
      actual.columnwise_cpu_scale_inv_ptr<fp8e8m0>());
  for (size_t i = 0; i < reference_colwise_scales.size(); ++i) {
    ASSERT_EQ(actual_colwise_scales[i], reference_colwise_scales[i])
        << "E8M0 columnwise scale mismatch at physical index " << i;
  }
  for (size_t i = reference_colwise_scales.size(); i < colwise_scales_alloc; ++i) {
    ASSERT_EQ(actual_colwise_scales[i], kSentinel)
        << "Capacity-tail columnwise scale byte was written at physical index " << i;
  }

  // Rowwise scales: the swizzled copy must match the production dense swizzle on
  // the live prefix and stay sentinel past it.
  const size_t num_live_rowwise_scales = num_live_rows * hidden_size / MXFP8_SCALE_DIM;
  const auto *actual_rowwise_scales =
      reinterpret_cast<const uint8_t *>(actual.rowwise_cpu_scale_inv_ptr<fp8e8m0>());
  const auto *reference_rowwise_scales = reinterpret_cast<const uint8_t *>(
      rowwise_swizzled_ref.rowwise_cpu_scale_inv_ptr<fp8e8m0>());
  for (size_t i = 0; i < num_live_rowwise_scales; ++i) {
    ASSERT_EQ(actual_rowwise_scales[i], reference_rowwise_scales[i])
        << "E8M0 rowwise scale mismatch at physical index " << i;
  }
  for (size_t i = num_live_rowwise_scales; i < rowwise_scales_alloc; ++i) {
    ASSERT_EQ(actual_rowwise_scales[i], kSentinel)
        << "Capacity-tail rowwise scale byte was written at physical index " << i;
  }

  // Optional BF16 output: live rows dequantize exactly (FP8 values scaled by a
  // power of two are exactly representable in BF16 after one rounding), so the
  // comparison is bitwise; tail rows stay sentinel.
  if (return_dequantized) {
    Tensor dequantized_bf16_ref("dequantized_bf16_ref", shape, DType::kBFloat16);
    nvte_dequantize(input_mxfp8.data(), dequantized_bf16_ref.data(), 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    dequantized_out.to_cpu();
    dequantized_bf16_ref.to_cpu();
    const auto *actual_dq =
        reinterpret_cast<const uint16_t *>(dequantized_out.rowwise_cpu_dptr<bf16>());
    const auto *reference_dq =
        reinterpret_cast<const uint16_t *>(dequantized_bf16_ref.rowwise_cpu_dptr<bf16>());
    for (size_t row = 0; row < num_live_rows; ++row) {
      for (size_t col = 0; col < hidden_size; ++col) {
        const size_t i = row * hidden_size + col;
        ASSERT_EQ(actual_dq[i], reference_dq[i])
            << "BF16 dequantized mismatch at row " << row << ", column " << col;
      }
    }
    constexpr uint16_t kSentinelBf16 =
        static_cast<uint16_t>(kSentinel) | (static_cast<uint16_t>(kSentinel) << 8);
    for (size_t i = num_live_rows * hidden_size; i < num_rows * hidden_size; ++i) {
      ASSERT_EQ(actual_dq[i], kSentinelBf16)
          << "Capacity-tail dequantized element was written at flat index " << i;
    }
  }
}

std::string splitsToString(const std::vector<size_t> &splits) {
  std::string result;
  for (const size_t split : splits) {
    if (!result.empty()) {
      result += "_";
    }
    result += std::to_string(split);
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, FusedGroupRequantizeTestSuite,
    ::testing::Combine(
        ::testing::Values(
            FusedGroupRequantizeCase{512, {1024}, 0},                     // single group
            FusedGroupRequantizeCase{512, {512, 512, 512, 512}, 0},       // uniform
            FusedGroupRequantizeCase{256, {256, 1024, 128, 640}, 0},      // variable
            FusedGroupRequantizeCase{512, {0, 512, 256}, 0},              // zero-token front
            FusedGroupRequantizeCase{384, {256, 0, 0, 512}, 0},           // adjacent zero-token
            FusedGroupRequantizeCase{512, {512, 256, 0}, 0},              // zero-token end
            FusedGroupRequantizeCase{8192, {512, 256, 512}, 2048},        // capacity tail
            FusedGroupRequantizeCase{512, {0, 512, 0, 256}, 1024},        // zero groups + tail
            FusedGroupRequantizeCase{8192, {2048, 4096, 1024, 1024}, 0}), // production-like
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::Values(false, true),    // use_fast_math
        ::testing::Values(false, true)),   // return_dequantized
    [](const testing::TestParamInfo<FusedGroupRequantizeTestSuite::ParamType> &info) {
      const auto test_case = std::get<0>(info.param);
      return "H" + std::to_string(test_case.hidden_size) + "xS" +
             splitsToString(test_case.splits) + "xCap" +
             std::to_string(test_case.capacity_rows) + "x" +
             test::typeName(std::get<1>(info.param)) + "xFastMath" +
             std::to_string(std::get<2>(info.param)) + "xDequant" +
             std::to_string(std::get<3>(info.param));
    });

}  // namespace
