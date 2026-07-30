/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>
#include <transformer_engine/swizzle.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

#if FP4_TYPE_SUPPORTED

namespace {

float2 cvt_fp4x2_to_float2(fp4e2m1x2 fp4_pair) {
    const __half2_raw raw =
        __nv_cvt_fp4x2_to_halfraw2(
            *reinterpret_cast<__nv_fp4x2_storage_t *>(&fp4_pair), __NV_E2M1);
    const __half2 h2(raw);
    return {static_cast<float>(h2.x), static_cast<float>(h2.y)};
}

template <typename OType, typename ScaleType>
void compute_ref_dequantize_nvfp4(const uint8_t *packed_data,
                                  const ScaleType *scales,
                                  const std::vector<float> &amax,
                                  OType *output,
                                  size_t rows,
                                  size_t cols,
                                  size_t scale_stride,
                                  float scale_max) {
    const float factor_inv = 1.0f / (6.0f * scale_max);
    constexpr size_t BLOCK_SIZE = 16;
    const size_t Mread = cols / BLOCK_SIZE;
    const size_t bytes_per_block = BLOCK_SIZE / 2;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < Mread; ++block) {
            const ScaleType scale = scales[row * scale_stride + block];
            const float final_scale =
                static_cast<float>(scale) * (amax.size() == 1 ? amax[0] : amax[row]) * factor_inv;

            for (size_t pair_idx = 0; pair_idx < bytes_per_block; ++pair_idx) {
                const size_t byte_idx =
                    (row * Mread + block) * bytes_per_block + pair_idx;
                fp4e2m1x2 fp4_pair;
                std::memcpy(&fp4_pair, &packed_data[byte_idx], 1);
                const float2 values = cvt_fp4x2_to_float2(fp4_pair);

                const size_t col0 = block * BLOCK_SIZE + pair_idx * 2;
                output[row * cols + col0] =
                    static_cast<OType>(values.x * final_scale);
                output[row * cols + col0 + 1] =
                    static_cast<OType>(values.y * final_scale);
            }
        }
    }
}

template <typename OutputType>
float compute_amax(test::Tensor &t, size_t rows, size_t cols) {
    t.to_cpu();
    const auto *data = t.rowwise_cpu_dptr<OutputType>();
    float amax = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        amax = std::max(amax, std::abs(static_cast<float>(data[i])));
    }
    return amax;
}

struct NVFP4DequantizeTestConfig {
  NVTENVFP44Over6Mode mode = kNVTENVFP44Over6Disabled;
  int e4m3_max = 448;
};

// Quantize a high-precision input to NVFP4, then dequantize and compare
// against a CPU reference computed from the quantized data.
template <typename OutputType, typename ScaleType = fp8e4m3>
void performTest_dequantize_nvfp4(const size_t rows, const size_t cols,
                                  const bool row_scaled_nvfp4,
                                  const NVTENVFP44Over6Mode mode,
                                  const int e4m3_max) {
    using namespace test;
    DType otype = TypeInfo<OutputType>::dtype;

    // Tensors
    Tensor input("input", std::vector<size_t>{rows, cols}, otype);
    Tensor quantized("quantized", std::vector<size_t>{rows, cols},
                     DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING,
                     TypeInfo<ScaleType>::dtype);
    Tensor output("output", std::vector<size_t>{rows, cols}, otype, true, false);

    // Fill input with random data
    fillCase<fp32>(&input, InputsFillCase::uniform);

    // Configure quantized tensor amax
    size_t amax_size = 1;
    quantized.set_nvfp4_e4m3_max(e4m3_max);
    ASSERT_EQ(quantized.nvfp4_e4m3_max(), e4m3_max);
    if (row_scaled_nvfp4) {
      quantized.set_row_scaled_nvfp4(true);
      amax_size = rows;
    } else if (rows > 0 && cols > 0) {
      quantized.set_amax(compute_amax<OutputType>(input, rows, cols));
    } else {
      quantized.set_amax(0.0f);
    }

    // Quantize
    if (rows > 0 && cols > 0) {
        QuantizationConfigWrapper quant_config;
        quant_config.set_nvfp4_4over6_mode(mode);
        nvte_quantize_v2(input.data(), quantized.data(), quant_config, 0);
        cudaDeviceSynchronize();
        auto err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    }

    // Dequantize
    nvte_dequantize(quantized.data(), output.data(), 0);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Nothing to be done if tensor is empty
    if (rows == 0 && cols == 0) {
      return;
    }

    // Dequantize reference implementation
    quantized.to_cpu();
    const uint8_t *fp4_data =
      reinterpret_cast<const uint8_t *>(quantized.rowwise_cpu_dptr<fp4e2m1>());
    const ScaleType *scales = quantized.rowwise_cpu_scale_inv_ptr<ScaleType>();
    const auto *amax = quantized.cpu_rowwise_amax_ptr<float>();
    const std::vector<float> amax_vals(amax, amax + amax_size);
    const NVTEShape scale_shape = quantized.rowwise_scale_inv_shape();
    const size_t scale_stride = scale_shape.data[scale_shape.ndim - 1];
    std::unique_ptr<OutputType[]> ref_output =
      std::make_unique<OutputType[]>(rows * cols);
    constexpr float full_scale_max =
      std::is_same_v<ScaleType, fp8ue5m3> ? 114688.0f : 448.0f;
    const float scale_max =
      TypeInfo<ScaleType>::dtype == DType::kFloat8E4M3 ? e4m3_max : full_scale_max;
    compute_ref_dequantize_nvfp4<OutputType, ScaleType>(
      fp4_data, scales, amax_vals, ref_output.get(),
      rows, cols, scale_stride, scale_max);

    // Compare results from TE and reference impls
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_nvfp4", output, ref_output.get(), true, atol, rtol);
}

// Dequantize NVFP4 with GEMM-swizzled scales and compare against compact path.
template <typename OutputType, typename ScaleType = fp8e4m3>
void performTest_dequantize_nvfp4_swizzled(const size_t rows, const size_t cols,
                                           const bool row_scaled_nvfp4,
                                           const NVTENVFP44Over6Mode mode,
                                           const int e4m3_max) {
    using namespace test;
    DType otype = TypeInfo<OutputType>::dtype;

    Tensor input("input", std::vector<size_t>{rows, cols}, otype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    Tensor quantized_compact("quantized_compact", std::vector<size_t>{rows, cols},
                             DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING,
                             TypeInfo<ScaleType>::dtype);
    quantized_compact.set_nvfp4_e4m3_max(e4m3_max);
    ASSERT_EQ(quantized_compact.nvfp4_e4m3_max(), e4m3_max);
    if (row_scaled_nvfp4) {
        quantized_compact.set_row_scaled_nvfp4(true);
    } else if (rows > 0 && cols > 0) {
        quantized_compact.set_amax(compute_amax<OutputType>(input, rows, cols));
    } else {
        quantized_compact.set_amax(0.0f);
    }

    if (rows > 0 && cols > 0) {
        QuantizationConfigWrapper quant_config;
        quant_config.set_nvfp4_4over6_mode(mode);
        nvte_quantize_v2(input.data(), quantized_compact.data(), quant_config, 0);
        cudaDeviceSynchronize();
    }

    // Dequantize with compact scales to get the reference output.
    Tensor output_compact("output_compact", std::vector<size_t>{rows, cols}, otype, true, false);
    nvte_dequantize(quantized_compact.data(), output_compact.data(), 0);
    cudaDeviceSynchronize();

    // Create tensor with same FP4 data but swizzled scales
    Tensor quantized_swizzled("quantized_swizzled", std::vector<size_t>{rows, cols},
                              DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING,
                              TypeInfo<ScaleType>::dtype);
    quantized_swizzled.set_nvfp4_e4m3_max(e4m3_max);
    ASSERT_EQ(quantized_swizzled.nvfp4_e4m3_max(), e4m3_max);
    if (row_scaled_nvfp4) {
        quantized_swizzled.set_row_scaled_nvfp4(true);
    } else {
        quantized_swizzled.set_amax(0.0f);
    }
    quantized_swizzled.set_with_gemm_swizzled_scales(true);

    // Copy amax and scale from compact to swizzled before FP4 data,
    // since from_cpu() uploads all CPU buffers (including zero-init data).
    quantized_compact.to_cpu();
    if (row_scaled_nvfp4) {
        const auto *src = quantized_compact.cpu_rowwise_amax_ptr<float>();
        auto *dst = quantized_swizzled.cpu_rowwise_amax_ptr<float>();
        std::copy(src, src + rows, dst);
        quantized_swizzled.from_cpu();
    } else {
        quantized_swizzled.set_amax(quantized_compact.amax());
    }

    // Copy FP4 data after from_cpu() to avoid being overwritten
    const size_t data_bytes = rows * cols / 2;
    if (data_bytes > 0) {
        cudaMemcpy(quantized_swizzled.rowwise_dptr(), quantized_compact.rowwise_dptr(),
                   data_bytes, cudaMemcpyDeviceToDevice);
    }

    // Swizzle scales
    if (data_bytes > 0) {
        nvte_swizzle_scaling_factors(quantized_compact.data(), quantized_swizzled.data(), 0);
    }

    // Dequantize with swizzled scales
    Tensor output_swizzled("output_swizzled", std::vector<size_t>{rows, cols}, otype, true, false);
    nvte_dequantize(quantized_swizzled.data(), output_swizzled.data(), 0);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Read compact output as reference
    const size_t num_elems = rows * cols;
    std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(num_elems);
    if (num_elems > 0) {
        cudaMemcpy(ref_output.get(), output_compact.rowwise_dptr(),
                   num_elems * sizeof(OutputType), cudaMemcpyDeviceToHost);
    }

    auto [atol, rtol] = getTolerances(otype);
    if (num_elems > 0) {
        compareResults("output_nvfp4_swizzled", output_swizzled,
                       ref_output.get(), true, atol, rtol);
    }
}

std::vector<std::pair<size_t, size_t>> nvfp4_tensor_dims = {
    {0, 128},
    {0, 256},
    {32, 32},
    {32, 64},
    {64, 96},
    {128, 128},
    {128, 256},
    {256, 256},
    {256, 512},
    {512, 1024},
    {992, 512},
    {768, 1024},
};

}  // namespace

class DequantizeNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                transformer_engine::DType,
                bool,
                NVFP4DequantizeTestConfig>> {};

TEST_P(DequantizeNVFP4TestSuite, TestDequantizeNVFP4)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const bool row_scaled_nvfp4 = std::get<2>(GetParam());
    const NVFP4DequantizeTestConfig config = std::get<3>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest_dequantize_nvfp4<OutputType>(
            tensor_size.first, tensor_size.second, row_scaled_nvfp4, config.mode,
            config.e4m3_max);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(nvfp4_tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Bool(),
        ::testing::Values(NVFP4DequantizeTestConfig{},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 448},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 256})),
    [](const testing::TestParamInfo<DequantizeNVFP4TestSuite::ParamType>& info)
    {
        const NVFP4DequantizeTestConfig config = std::get<3>(info.param);
        const bool use_4over6 = config.mode != kNVTENVFP44Over6Disabled;
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           test::typeName(std::get<1>(info.param)) + "X" +
                           (std::get<2>(info.param) ? "RowScaled" : "PerTensor") + "X" +
                           (use_4over6 ? "FourOverSix" : "Default") + "X" +
                           (config.e4m3_max == 256 ? "E4M3Max256" : "E4M3Max448");
        return name;
    }
);

#if CUDA_VERSION >= 13040
TEST(DequantizeNVFP4Test, UE5M3Scales)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    performTest_dequantize_nvfp4<fp32, fp8ue5m3>(
        32, 64, false, kNVTENVFP44Over6Disabled, 448);
    performTest_dequantize_nvfp4<bf16, fp8ue5m3>(
        32, 64, true, kNVTENVFP44Over6Disabled, 448);
    performTest_dequantize_nvfp4_swizzled<fp32, fp8ue5m3>(
        32, 64, false, kNVTENVFP44Over6Disabled, 448);
    performTest_dequantize_nvfp4_swizzled<bf16, fp8ue5m3>(
        32, 64, true, kNVTENVFP44Over6Disabled, 448);
}

TEST(NVFP4RecipeTest, UE5M3ScaleUtilities)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    Tensor global_amax("global_amax", std::vector<size_t>{1}, DType::kFloat32);
    Tensor global_scale("global_scale", std::vector<size_t>{1}, DType::kFloat32);
    global_amax.rowwise_cpu_dptr<float>()[0] = 12.0f;
    global_amax.from_cpu();
    nvte_nvfp4_compute_global_scale(
        global_amax.data(), global_scale.data(), 0, kNVTEFloat8UE5M3);
    global_scale.to_cpu();
    EXPECT_FLOAT_EQ(global_scale.rowwise_cpu_dptr<float>()[0], 6.0f * 114688.0f / 12.0f);

    Tensor block_amax("block_amax", std::vector<size_t>{1, 2}, DType::kFloat32);
    Tensor block_scale("block_scale", std::vector<size_t>{1, 2}, DType::kFloat32);
    block_amax.rowwise_cpu_dptr<float>()[0] = 3.0f;
    block_amax.rowwise_cpu_dptr<float>()[1] = 6.0f;
    block_amax.from_cpu();
    nvte_nvfp4_compute_per_block_scale(
        block_amax.data(), block_scale.data(), global_amax.data(), 0, kNVTEFloat8UE5M3);
    block_scale.to_cpu();
    EXPECT_FLOAT_EQ(block_scale.rowwise_cpu_dptr<float>()[0], 3.0f * 114688.0f / 12.0f);
    EXPECT_FLOAT_EQ(block_scale.rowwise_cpu_dptr<float>()[1], 6.0f * 114688.0f / 12.0f);

    Tensor expanded_scale("expanded_scale", std::vector<size_t>{16, 2}, DType::kByte);
    nvte_nvfp4_expand_scale_to_fp8(
        block_scale.data(), expanded_scale.data(), 1, 2, 16, 16, 0, kNVTEFloat8UE5M3);
    expanded_scale.to_cpu();
    const auto *scales = reinterpret_cast<const fp8ue5m3 *>(
        expanded_scale.rowwise_cpu_dptr<byte>());
    for (size_t row = 0; row < 16; ++row) {
        EXPECT_FLOAT_EQ(static_cast<float>(scales[row * 2]),
                        static_cast<float>(fp8ue5m3(3.0f * 114688.0f / 12.0f)));
        EXPECT_FLOAT_EQ(static_cast<float>(scales[row * 2 + 1]),
                        static_cast<float>(fp8ue5m3(6.0f * 114688.0f / 12.0f)));
    }
}

TEST(NVFP4RecipeTest, UE5M3PerTensorScale)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    Tensor input_a("input_a", std::vector<size_t>{16, 16}, DType::kFloat4E2M1,
                   true, true, NVTE_NVFP4_1D_SCALING, DType::kFloat8UE5M3);
    Tensor input_b("input_b", std::vector<size_t>{16, 16}, DType::kFloat4E2M1,
                   true, true, NVTE_NVFP4_1D_SCALING, DType::kFloat8UE5M3);
    Tensor alpha_out("alpha_out", std::vector<size_t>{1}, DType::kFloat32);

    constexpr float amax_a = 12.0f;
    constexpr float amax_b = 18.0f;
    constexpr float alpha_in = 2.0f;
    constexpr float fp4_max = 6.0f;
    constexpr float ue5m3_max = 114688.0f;
    input_a.set_amax(amax_a);
    input_b.set_tensor_amax_columnwise(amax_b);

    nvte_nvfp4_compute_per_tensor_scale(
        input_a.data(), true, input_b.data(), false, alpha_in, alpha_out.data(), 0);
    alpha_out.to_cpu();

    const float factor_inv =
        1.0f / (fp4_max * fp4_max * ue5m3_max * ue5m3_max);
    const float expected = alpha_in * amax_a * amax_b * factor_inv;
    EXPECT_FLOAT_EQ(alpha_out.rowwise_cpu_dptr<float>()[0], expected);
}
#endif

class DequantizeNVFP4SwizzledTestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                transformer_engine::DType,
                bool,
                NVFP4DequantizeTestConfig>> {};

TEST_P(DequantizeNVFP4SwizzledTestSuite, TestDequantizeNVFP4Swizzled)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const bool row_scaled_nvfp4 = std::get<2>(GetParam());
    const NVFP4DequantizeTestConfig config = std::get<3>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest_dequantize_nvfp4_swizzled<OutputType>(
            tensor_size.first, tensor_size.second, row_scaled_nvfp4, config.mode,
            config.e4m3_max);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4SwizzledTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(nvfp4_tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Bool(),
        ::testing::Values(NVFP4DequantizeTestConfig{},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 448},
                          NVFP4DequantizeTestConfig{kNVTENVFP44Over6MinMAE, 256})),
    [](const testing::TestParamInfo<DequantizeNVFP4SwizzledTestSuite::ParamType>& info)
    {
        const NVFP4DequantizeTestConfig config = std::get<3>(info.param);
        const bool use_4over6 = config.mode != kNVTENVFP44Over6Disabled;
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           test::typeName(std::get<1>(info.param)) + "X" +
                           (std::get<2>(info.param) ? "RowScaled" : "PerTensor") + "X" +
                           (use_4over6 ? "FourOverSix" : "Default") + "X" +
                           (config.e4m3_max == 256 ? "E4M3Max256" : "E4M3Max448") + "X" +
                           "Swizzled";
        return name;
    }
);

#endif  // FP4_TYPE_SUPPORTED
