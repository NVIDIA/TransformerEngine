/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

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

float2 cvt_fp4x2_to_float2(fp4e2m1x2 fp4_pair) {
    const __half2_raw raw =
        __nv_cvt_fp4x2_to_halfraw2(
            *reinterpret_cast<__nv_fp4x2_storage_t *>(&fp4_pair), __NV_E2M1);
    const __half2 h2(raw);
    return {static_cast<float>(h2.x), static_cast<float>(h2.y)};
}

template <typename OType>
void compute_ref_dequantize_nvfp4(const uint8_t *packed_data,
                                  const fp8e4m3 *scales,
                                  float amax,
                                  OType *output,
                                  size_t rows,
                                  size_t cols,
                                  size_t scale_stride) {
    constexpr float factor_inv = 1.0f / (6.0f * 448.0f);
    constexpr size_t BLOCK_SIZE = 16;
    const size_t Mread = cols / BLOCK_SIZE;
    const size_t bytes_per_block = BLOCK_SIZE / 2;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < Mread; ++block) {
            const fp8e4m3 scale = scales[row * scale_stride + block];
            const float final_scale = static_cast<float>(scale) * amax * factor_inv;

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
float compute_amax(const test::Tensor &t, size_t rows, size_t cols) {
    t.to_cpu();
    const auto *data = t.rowwise_cpu_dptr<OutputType>();
    float amax = 0.0f;
    for (size_t i = 0; i < rows * cols; ++i) {
        amax = std::max(amax, std::abs(static_cast<float>(data[i])));
    }
    return amax;
}

// Quantize a high-precision input to NVFP4, then dequantize and compare
// against a CPU reference computed from the quantized data.
template <typename OutputType>
void performTest_dequantize_nvfp4(const size_t rows, const size_t cols) {
    using namespace test;
    DType otype = TypeInfo<OutputType>::dtype;

    Tensor input("input", std::vector<size_t>{rows, cols}, otype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    Tensor quantized("quantized", std::vector<size_t>{rows, cols},
                     DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING);
    if (rows > 0 && cols > 0) {
        quantized.set_tensor_amax(compute_amax<OutputType>(input, rows, cols));
    } else {
        quantized.set_tensor_amax(0.0f);
    }

    if (rows > 0 && cols > 0) {
        nvte_quantize(input.data(), quantized.data(), 0);
        cudaDeviceSynchronize();
    }

    Tensor output("output", std::vector<size_t>{rows, cols}, otype, true, false);
    nvte_dequantize(quantized.data(), output.data(), 0);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    if (rows > 0 && cols > 0) {
        quantized.to_cpu();
        const uint8_t *fp4_data =
            reinterpret_cast<const uint8_t *>(quantized.rowwise_cpu_dptr<fp4e2m1>());
        const fp8e4m3 *scales = quantized.rowwise_cpu_scale_inv_ptr<fp8e4m3>();
        const float amax_val = quantized.amax();
        const NVTEShape scale_shape = quantized.rowwise_scale_inv_shape();
        const size_t scale_stride = scale_shape.data[scale_shape.ndim - 1];

        std::unique_ptr<OutputType[]> ref_output =
            std::make_unique<OutputType[]>(rows * cols);
        compute_ref_dequantize_nvfp4<OutputType>(
            fp4_data, scales, amax_val, ref_output.get(),
            rows, cols, scale_stride);

        auto [atol, rtol] = getTolerances(otype);
        compareResults("output_nvfp4", output, ref_output.get(), true, atol, rtol);
    }
}

// Dequantize NVFP4 with GEMM-swizzled scales and compare against compact path.
template <typename OutputType>
void performTest_dequantize_nvfp4_swizzled(const size_t rows, const size_t cols) {
    using namespace test;
    DType otype = TypeInfo<OutputType>::dtype;

    Tensor input("input", std::vector<size_t>{rows, cols}, otype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    Tensor quantized_compact("quantized_compact", std::vector<size_t>{rows, cols},
                             DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING);
    if (rows > 0 && cols > 0) {
        quantized_compact.set_tensor_amax(compute_amax<OutputType>(input, rows, cols));
    } else {
        quantized_compact.set_tensor_amax(0.0f);
    }

    if (rows > 0 && cols > 0) {
        nvte_quantize(input.data(), quantized_compact.data(), 0);
        cudaDeviceSynchronize();
    }

    // Dequantize with compact scales → reference output
    Tensor output_compact("output_compact", std::vector<size_t>{rows, cols}, otype, true, false);
    nvte_dequantize(quantized_compact.data(), output_compact.data(), 0);
    cudaDeviceSynchronize();

    // Create tensor with same FP4 data but swizzled scales
    Tensor quantized_swizzled("quantized_swizzled", std::vector<size_t>{rows, cols},
                              DType::kFloat4E2M1, true, false, NVTE_NVFP4_1D_SCALING);
    quantized_swizzled.set_tensor_amax(0.0f);
    quantized_swizzled.set_with_gemm_swizzled_scales(true);

    // Copy amax and scale from compact to swizzled before FP4 data,
    // since from_cpu() uploads all CPU buffers (including zero-init data).
    quantized_compact.to_cpu();
    quantized_swizzled.set_tensor_amax(quantized_compact.amax());

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
                transformer_engine::DType>> {};

TEST_P(DequantizeNVFP4TestSuite, TestDequantizeNVFP4)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest_dequantize_nvfp4<OutputType>(
            tensor_size.first, tensor_size.second);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(nvfp4_tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<DequantizeNVFP4TestSuite::ParamType>& info)
    {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           test::typeName(std::get<1>(info.param));
        return name;
    }
);

class DequantizeNVFP4SwizzledTestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                transformer_engine::DType>> {};

TEST_P(DequantizeNVFP4SwizzledTestSuite, TestDequantizeNVFP4Swizzled)
{
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    const auto tensor_size = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
        performTest_dequantize_nvfp4_swizzled<OutputType>(
            tensor_size.first, tensor_size.second);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeNVFP4SwizzledTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(nvfp4_tensor_dims),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16)),
    [](const testing::TestParamInfo<DequantizeNVFP4SwizzledTestSuite::ParamType>& info)
    {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           test::typeName(std::get<1>(info.param)) + "X" +
                           "Swizzled";
        return name;
    }
);

#endif  // FP4_TYPE_SUPPORTED
