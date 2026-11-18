/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#include <transformer_engine/cast.h>
#include <transformer_engine/swizzle.h>

#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

#if FP4_TYPE_SUPPORTED

namespace {

constexpr size_t kBlockSize = 32;

uint8_t to_fp4x2(float x, float y) {
  const fp4e2m1x2 pair(float2{x, y});
  uint8_t byte;
  std::memcpy(&byte, &pair, 1);
  return byte;
}

float2 from_fp4x2(uint8_t byte) {
  const __half2 h2(__nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1));
  return {static_cast<float>(h2.x), static_cast<float>(h2.y)};
}

/* Reference MXFP4 quantization
 *
 * Row-wise blocks are 1x32 and column-wise blocks are 32x1. Both keep the
 * data in the logical layout, so a byte packs two adjacent columns.
 */
template <typename InputType>
void ref_quantize(const InputType *input, uint8_t *data, fp8e8m0 *scales, size_t rows, size_t cols,
                  size_t scales_stride, bool rowwise) {
  const size_t block_rows = rowwise ? 1 : kBlockSize;
  const size_t block_cols = rowwise ? kBlockSize : 1;
  std::vector<float> scale_rcp(rows * cols);
  for (size_t i0 = 0; i0 < rows; i0 += block_rows) {
    for (size_t j0 = 0; j0 < cols; j0 += block_cols) {
      float amax = 0.f;
      for (size_t i = i0; i < i0 + block_rows; ++i) {
        for (size_t j = j0; j < j0 + block_cols; ++j) {
          amax = std::fmax(amax, std::fabs(static_cast<float>(input[i * cols + j])));
        }
      }
      const fp8e8m0 scale = float_to_e8m0(amax * (1.f / 6.f));
      scales[(i0 / block_rows) * scales_stride + j0 / block_cols] = scale;
      for (size_t i = i0; i < i0 + block_rows; ++i) {
        for (size_t j = j0; j < j0 + block_cols; ++j) {
          scale_rcp[i * cols + j] = exp2f_rcp(scale);
        }
      }
    }
  }
  for (size_t idx = 0; idx < rows * cols; idx += 2) {
    data[idx / 2] = to_fp4x2(static_cast<float>(input[idx]) * scale_rcp[idx],
                             static_cast<float>(input[idx + 1]) * scale_rcp[idx + 1]);
  }
}

template <typename OutputType>
void ref_dequantize(const uint8_t *data, const fp8e8m0 *scales, OutputType *output, size_t rows,
                    size_t cols, size_t scales_stride, bool rowwise) {
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      const float2 pair = from_fp4x2(data[(i * cols + j) / 2]);
      const fp8e8m0 scale = rowwise ? scales[i * scales_stride + j / kBlockSize]
                                    : scales[(i / kBlockSize) * scales_stride + j];
      output[i * cols + j] =
          static_cast<OutputType>(std::ldexp(j % 2 == 0 ? pair.x : pair.y, int(scale) - 127));
    }
  }
}

template <typename InputType>
void check_compact(Tensor &input, Tensor &quantized, Tensor &output,
                   const std::vector<size_t> &shape, bool rowwise) {
  const size_t rows = first_dimension(shape);
  const size_t cols = last_dimension(shape);
  const NVTEShape scale_shape =
      rowwise ? quantized.rowwise_scale_inv_shape() : quantized.columnwise_scale_inv_shape();
  const size_t scales_stride = scale_shape.data[scale_shape.ndim - 1];

  std::vector<uint8_t> ref_data(rows * cols / 2);
  std::vector<fp8e8m0> ref_scales(product(scale_shape));
  ref_quantize(input.rowwise_cpu_dptr<InputType>(), ref_data.data(), ref_scales.data(), rows, cols,
               scales_stride, rowwise);

  const std::string name = rowwise ? "rowwise" : "columnwise";
  const auto *data = reinterpret_cast<const uint8_t *>(
      rowwise ? quantized.rowwise_cpu_dptr<fp4e2m1>() : quantized.columnwise_cpu_dptr<fp4e2m1>());
  compareResults(name + "_data", data, ref_data.data(), ref_data.size());
  const fp8e8m0 *scales = rowwise ? quantized.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                  : quantized.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
  size_t mismatches = 0;
  compare_scaling_factors<fp8e8m0>(name + "_scales", scales, ref_scales.data(),
                                   rowwise ? rows : rows / kBlockSize,
                                   rowwise ? cols / kBlockSize : cols, scales_stride, mismatches);

  // Dequantization reads row-wise data when it is available
  if (rowwise || !quantized.rowwise()) {
    std::vector<InputType> ref_output(rows * cols);
    ref_dequantize(ref_data.data(), ref_scales.data(), ref_output.data(), rows, cols, scales_stride,
                   rowwise);
    compareResults("output", output, ref_output.data(), true, 0, 0);
  }
}

/* Compare GEMM-swizzled output with compact output followed by
 * nvte_swizzle_scaling_factors, which swizzles one direction at a time
 */
template <typename InputType>
void check_swizzled(Tensor &input, Tensor &swizzled, const std::vector<size_t> &shape,
                    bool rowwise) {
  const size_t data_bytes = product(shape) / 2;
  const DType itype = input.dtype();

  Tensor compact("compact", shape, DType::kFloat4E2M1, rowwise, !rowwise, NVTE_MXFP4_1D_SCALING);
  Tensor ref("ref", shape, DType::kFloat4E2M1, rowwise, !rowwise, NVTE_MXFP4_1D_SCALING);
  ref.set_with_gemm_swizzled_scales(true);
  nvte_quantize(input.data(), compact.data(), 0);
  cudaMemcpy(rowwise ? ref.rowwise_dptr() : ref.columnwise_dptr(),
             rowwise ? compact.rowwise_dptr() : compact.columnwise_dptr(), data_bytes,
             cudaMemcpyDeviceToDevice);
  nvte_swizzle_scaling_factors(compact.data(), ref.data(), 0);

  Tensor output("output", shape, itype);
  Tensor ref_output("ref_output", shape, itype);
  nvte_dequantize(ref.data(), output.data(), 0);
  nvte_dequantize(compact.data(), ref_output.data(), 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  swizzled.to_cpu();
  compact.to_cpu();
  ref.to_cpu();
  ref_output.to_cpu();
  const std::string name = rowwise ? "rowwise" : "columnwise";
  const auto *data = reinterpret_cast<const uint8_t *>(
      rowwise ? swizzled.rowwise_cpu_dptr<fp4e2m1>() : swizzled.columnwise_cpu_dptr<fp4e2m1>());
  const auto *ref_data = reinterpret_cast<const uint8_t *>(
      rowwise ? compact.rowwise_cpu_dptr<fp4e2m1>() : compact.columnwise_cpu_dptr<fp4e2m1>());
  compareResults(name + "_data", data, ref_data, data_bytes);
  const fp8e8m0 *scales = rowwise ? swizzled.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                  : swizzled.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
  const fp8e8m0 *ref_scales = rowwise ? ref.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                      : ref.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
  const NVTEShape scale_shape =
      rowwise ? ref.rowwise_scale_inv_shape() : ref.columnwise_scale_inv_shape();
  compareResults(name + "_scales", scales, ref_scales, product(scale_shape));
  compareResults(name + "_output", output, ref_output.rowwise_cpu_dptr<InputType>(), true, 0, 0);
}

template <typename InputType>
void performTest(const std::vector<size_t> &shape, bool rowwise, bool columnwise, bool swizzled) {
  const DType itype = TypeInfo<InputType>::dtype;
  Tensor input("input", shape, itype);
  Tensor quantized("quantized", shape, DType::kFloat4E2M1, rowwise, columnwise,
                   NVTE_MXFP4_1D_SCALING);
  Tensor output("output", shape, itype);
  fillUniform(&input);
  quantized.set_with_gemm_swizzled_scales(swizzled);

  nvte_quantize(input.data(), quantized.data(), 0);
  nvte_dequantize(quantized.data(), output.data(), 0);
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
  input.to_cpu();
  quantized.to_cpu();

  if (swizzled) {
    if (rowwise) check_swizzled<InputType>(input, quantized, shape, true);
    if (columnwise) check_swizzled<InputType>(input, quantized, shape, false);
  } else {
    if (rowwise) check_compact<InputType>(input, quantized, output, shape, true);
    if (columnwise) check_compact<InputType>(input, quantized, output, shape, false);
  }
}

// Columns must be multiples of 32, and rows too for column-wise blocks
const std::vector<std::vector<size_t>> mxfp4_shapes = {
    {32, 32}, {64, 96}, {96, 160}, {128, 128}, {256, 512}, {992, 512}, {4, 64, 256},
};

}  // namespace

using CastMXFP4TestParams = std::tuple<std::vector<size_t>,    // Shape
                                       DType,                  // Input dtype
                                       std::pair<bool, bool>,  // Row-wise, column-wise
                                       bool>;                  // GEMM-swizzled scales

class CastMXFP4TestSuite : public ::testing::TestWithParam<CastMXFP4TestParams> {};

TEST_P(CastMXFP4TestSuite, TestCastMXFP4) {
  if (getDeviceComputeCapability() < blackwellComputeCapability) {
    GTEST_SKIP();
  }
  const auto [shape, itype, usage, swizzled] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(
      itype, InputType, performTest<InputType>(shape, usage.first, usage.second, swizzled););
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest, CastMXFP4TestSuite,
    ::testing::Combine(::testing::ValuesIn(mxfp4_shapes),
                       ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
                       ::testing::Values(std::make_pair(true, false), std::make_pair(false, true),
                                         std::make_pair(true, true)),
                       ::testing::Bool()),
    [](const testing::TestParamInfo<CastMXFP4TestSuite::ParamType> &info) {
      std::string name;
      for (const size_t dim : std::get<0>(info.param)) {
        name += std::to_string(dim) + "X";
      }
      const auto usage = std::get<2>(info.param);
      return name + test::typeName(std::get<1>(info.param)) + "X" + (usage.first ? "R" : "") +
             (usage.second ? "C" : "") + "X" + (std::get<3>(info.param) ? "Swizzled" : "Compact");
    });

#endif  // FP4_TYPE_SUPPORTED
