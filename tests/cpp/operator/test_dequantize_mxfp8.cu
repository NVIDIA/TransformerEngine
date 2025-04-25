/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename InputType, typename OutputType>
void dequantize_block(const InputType* input,
                      OutputType* output,
                      fp8e8m0* scales,
                      const size_t scale_idx,
                      const size_t i_min,
                      const size_t i_max,
                      const size_t j_min,
                      const size_t j_max,
                      const size_t cols)
{
    const fp8e8m0 biased_exponent = scales[scale_idx];
    const float block_scale = exp2f(static_cast<float>(biased_exponent) - FP32_EXPONENT_BIAS);
    const float elem_scale = block_scale;

    // Dequantize elements in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            const size_t idx = i * cols + j;
            const float elt = static_cast<float>(input[idx]);
            output[idx] = static_cast<OutputType>(elt * elem_scale);
        }
    }
}

template <typename InputType, typename OutputType>
void compute_ref_x1(const InputType* input,
                    OutputType* output,
                    fp8e8m0* scales,
                    const size_t rows,
                    const size_t cols,
                    const size_t block_size_Y,
                    const size_t block_size_X,
                    const size_t scales_stride)
{
    const size_t tile_size_Y = std::max(32lu, block_size_Y);
    const size_t tile_size_X = std::max(64lu, block_size_X);
    const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
    const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;
    const size_t blocks_per_tile_Y = tile_size_Y / block_size_Y;
    const size_t blocks_per_tile_X = tile_size_X / block_size_X;

    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t t = 0; t < tiles_num_Y * tiles_num_X; ++t) {
        const size_t tile_Y = t / tiles_num_X;
        const size_t tile_X = t % tiles_num_X;
        const size_t tile_offset_Y = tile_Y * tile_size_Y;
        const size_t tile_offset_X = tile_X * tile_size_X;

        for (size_t ii = 0; ii < blocks_per_tile_Y; ++ii) {
            const size_t block_idx_Y = tile_Y * blocks_per_tile_Y + ii;
            const size_t block_offset_Y = ii * block_size_Y;
            const size_t i_min = tile_offset_Y + block_offset_Y;
            if (i_min >= rows) continue;
            const size_t i_max = std::min(i_min + block_size_Y, rows);

            for (size_t jj = 0; jj < blocks_per_tile_X; ++jj) {
                const size_t block_idx_X = tile_X * blocks_per_tile_X + jj;
                const size_t block_offset_X = jj * block_size_X;
                const size_t j_min = tile_offset_X + block_offset_X;
                if (j_min >= cols) continue;
                const size_t j_max = std::min(j_min + block_size_X, cols);

                const size_t scale_idx = block_idx_Y * scales_stride + block_idx_X;
                dequantize_block<InputType, OutputType>(
                    input, output, scales, scale_idx, i_min, i_max, j_min, j_max, cols);
            }
        }
    }
}

template <typename InputType, typename OutputType>
void compute_ref_x2(const InputType* input,
                    OutputType* output_rowwise,
                    OutputType* output_colwise,
                    fp8e8m0* scales_rowwise,
                    fp8e8m0* scales_colwise,
                    const size_t rows,
                    const size_t cols,
                    const size_t block_size_Y,
                    const size_t block_size_X,
                    const size_t scales_stride_rowwise,
                    const size_t scales_stride_colwise)
{
    compute_ref_x1<InputType, OutputType>(input, output_rowwise, scales_rowwise, rows, cols, 1, block_size_X, scales_stride_rowwise);
    compute_ref_x1<InputType, OutputType>(input, output_colwise, scales_colwise, rows, cols, block_size_Y, 1, scales_stride_colwise);
}

void generate_scales(fp8e8m0 * const scales_ref,
                     fp8e8m0 * const scales,
                     const size_t blocks_num,
                     std::mt19937& gen,
                     std::uniform_int_distribution<fp8e8m0> dis)
{
    for (size_t i = 0; i < blocks_num; ++i) {
        const fp8e8m0 val = dis(gen);
        scales_ref[i] = val;
        scales[i] = val;
    }
}

template<typename InputType>
void generate_data(InputType * const data,
                   const size_t rows,
                   const size_t cols,
                   std::mt19937& gen,
                   std::uniform_real_distribution<>& dis,
                   std::uniform_real_distribution<>& dis_sign)
{
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            const bool is_negative = (dis_sign(gen) < 0.0);
            double val = dis(gen);
            if (is_negative) {
                val = -val;
            }
            data[idx] = static_cast<InputType>(val);
        }
    }
}

template<typename InputType>
void fill_tensor_data(Tensor& input,
                      fp8e8m0 * const scales_rowwise,
                      fp8e8m0 * const scales_colwise,
                      const bool is_rowwise_scaling,
                      const bool is_colwise_scaling,
                      const size_t rows,
                      const size_t cols,
                      const size_t blocks_num_rowwise,
                      const size_t blocks_num_colwise)
{
    const double minAbs = Numeric_Traits<InputType>::minNorm;
    const double maxAbs = Numeric_Traits<InputType>::maxNorm;
    static std::mt19937 gen(12345);
    std::uniform_real_distribution<> dis(minAbs, maxAbs);
    std::uniform_real_distribution<> dis_sign(-1.0, 1.0);
    std::uniform_int_distribution<fp8e8m0> int_dis(0, 255);

    if (is_rowwise_scaling) {
        generate_scales(scales_rowwise, input.rowwise_cpu_scale_inv_ptr<fp8e8m0>(), blocks_num_rowwise, gen, int_dis);
        generate_data(input.rowwise_cpu_dptr<InputType>(), rows, cols, gen, dis, dis_sign);
    }

    if (is_colwise_scaling) {
        generate_scales(scales_colwise, input.columnwise_cpu_scale_inv_ptr<fp8e8m0>(), blocks_num_colwise, gen, int_dis);
        generate_data(input.columnwise_cpu_dptr<InputType>(), rows, cols, gen, dis, dis_sign);
    }

    input.from_cpu();
}

// Dequantize along single dimension (either row- or columnwise)
template <typename InputType, typename OutputType>
void performTest_x1(const size_t rows,
                    const size_t cols,
                    const bool rowwise,
                    const bool colwise)
{
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t block_size_rows = rowwise ? 1 : 32;
    const size_t block_size_cols = colwise ? 1 : 32;

    const size_t unpadded_blocks_Y_rowwise = rows;
    const size_t unpadded_blocks_X_rowwise = divide_round_up(cols, block_size_cols);
    const size_t unpadded_blocks_Y_colwise = divide_round_up(rows, block_size_rows);
    const size_t unpadded_blocks_X_colwise = cols;

    const size_t blocks_Y_rowwise = round_up_to_nearest_multiple(unpadded_blocks_Y_rowwise,
                                                                 scale_tensor_alignment_Y_rowwise);
    const size_t blocks_X_rowwise = round_up_to_nearest_multiple(unpadded_blocks_X_rowwise,
                                                                 scale_tensor_alignment_X_rowwise);
    const size_t blocks_Y_colwise = round_up_to_nearest_multiple(unpadded_blocks_Y_colwise,
                                                                 scale_tensor_alignment_Y_colwise);
    const size_t blocks_X_colwise = round_up_to_nearest_multiple(unpadded_blocks_X_colwise,
                                                                 scale_tensor_alignment_X_colwise);

    const size_t blocks_num_rowwise = blocks_Y_rowwise * blocks_X_rowwise;
    const size_t blocks_num_colwise = blocks_Y_colwise * blocks_X_colwise;

    const size_t blocks_num = rowwise ? blocks_num_rowwise : blocks_num_colwise;
    const size_t scales_stride = rowwise ? blocks_X_rowwise : blocks_X_colwise;

    Tensor input("input", std::vector<size_t>{ rows, cols }, itype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);

    // Output data are written to the rowwise ptr regardless of the scaling direction
    Tensor output("output", std::vector<size_t>{ rows, cols }, otype, true, false);

    std::unique_ptr<OutputType[]> ref_output = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<fp8e8m0[]> scales = std::make_unique<fp8e8m0[]>(blocks_num);

    fill_tensor_data<InputType>(input, scales.get(), scales.get(), rowwise, colwise, rows, cols,
                                blocks_num_rowwise, blocks_num_colwise);

    nvte_dequantize(input.data(), output.data(), 0);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    InputType * data_ptr = rowwise
                           ? input.rowwise_cpu_dptr<InputType>()
                           : input.columnwise_cpu_dptr<InputType>();

    compute_ref_x1<InputType, OutputType>(data_ptr,
                                          ref_output.get(),
                                          scales.get(),
                                          rows,
                                          cols,
                                          block_size_rows,
                                          block_size_cols,
                                          scales_stride);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output", output, ref_output.get(), true, atol, rtol);
}

// Dequantize along single dimension (either row- or columnwise)
template <typename InputType, typename IntermediateType>
void performTest_quantize_then_dequantize(const size_t rows,
                                          const size_t cols,
                                          const bool rowwise,
                                          const bool colwise)
{
    using namespace test;
    using EncodingType = fp32;
    DType in_type = TypeInfo<InputType>::dtype;
    DType intermed_type = TypeInfo<IntermediateType>::dtype;
    DType out_type = TypeInfo<InputType>::dtype;

    std::unique_ptr<InputType[]> input_cpu = std::make_unique<InputType[]>(rows * cols);
    std::unique_ptr<IntermediateType[]> quantized_cpu = std::make_unique<IntermediateType[]>(rows * cols);
    std::unique_ptr<InputType[]> output_cpu = std::make_unique<InputType[]>(rows * cols);

    // input --> quantized --> output (dequantized)
    // input == output
    Tensor input("input", std::vector<size_t>{ rows, cols }, in_type);
    Tensor quantized("quantized", std::vector<size_t>{ rows, cols }, intermed_type, rowwise, colwise, NVTE_MXFP8_1D_SCALING);

    // Output data are written to the rowwise ptr regardless of the scaling direction
    Tensor output("output", std::vector<size_t>{ rows, cols }, out_type, true, false);

    // fillCase<EncodingType>(&input, InputsFillCase::minNorm_to_maxNorm);
    fillCase<EncodingType>(&input, InputsFillCase::uniform);

    const size_t copy_size = sizeof(InputType) * rows * cols;
    cudaMemcpy(input_cpu.get(), input.rowwise_dptr(), copy_size, cudaMemcpyDeviceToHost);

    nvte_quantize(input.data(), quantized.data(), 0);
    cudaDeviceSynchronize();

    const size_t copy_size_quantized = sizeof(IntermediateType) * rows * cols;
    if (rowwise) {
        cudaMemcpy(quantized_cpu.get(), quantized.rowwise_dptr(), copy_size_quantized, cudaMemcpyDeviceToHost);
    }
    if (colwise) {
        cudaMemcpy(quantized_cpu.get(), quantized.columnwise_dptr(), copy_size_quantized, cudaMemcpyDeviceToHost);
    }

    nvte_dequantize(quantized.data(), output.data(), 0);
    cudaDeviceSynchronize();

    cudaMemcpy(output_cpu.get(), output.rowwise_dptr(), copy_size, cudaMemcpyDeviceToHost);

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto [atol, rtol] = getTolerances(intermed_type);
    compareResults("Quantize-Dequantize", input, output_cpu.get(), true, atol, rtol);
}

// Dequantize along both dimensions (row- and columnwise)
template <typename InputType, typename OutputType>
void performTest_x2(const size_t rows,
                    const size_t cols,
                    const size_t block_size_rows,
                    const size_t block_size_cols)
{
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t unpadded_blocks_Y_rowwise = rows;
    const size_t unpadded_blocks_X_rowwise = divide_round_up(cols, block_size_cols);
    const size_t unpadded_blocks_Y_colwise = divide_round_up(rows, block_size_rows);
    const size_t unpadded_blocks_X_colwise = cols;

    const size_t blocks_Y_rowwise = round_up_to_nearest_multiple(unpadded_blocks_Y_rowwise,
                                                                 scale_tensor_alignment_Y_rowwise);
    const size_t blocks_X_rowwise = round_up_to_nearest_multiple(unpadded_blocks_X_rowwise,
                                                                 scale_tensor_alignment_X_rowwise);
    const size_t blocks_Y_colwise = round_up_to_nearest_multiple(unpadded_blocks_Y_colwise,
                                                                 scale_tensor_alignment_Y_colwise);
    const size_t blocks_X_colwise = round_up_to_nearest_multiple(unpadded_blocks_X_colwise,
                                                                 scale_tensor_alignment_X_colwise);

    const size_t scales_stride_rowwise = blocks_X_rowwise;
    const size_t scales_stride_colwise = blocks_X_colwise;
    const size_t blocks_num_rowwise = blocks_Y_rowwise * blocks_X_rowwise;
    const size_t blocks_num_colwise = blocks_Y_colwise * blocks_X_colwise;

    Tensor input("input", std::vector<size_t>{ rows, cols }, itype, true, true, NVTE_MXFP8_1D_SCALING);
    Tensor output("output", std::vector<size_t>{ rows, cols }, otype);

    std::unique_ptr<OutputType[]> ref_output_rowwise = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<OutputType[]> ref_output_colwise = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<fp8e8m0[]> ref_scales_rowwise = std::make_unique<fp8e8m0[]>(blocks_num_rowwise);
    std::unique_ptr<fp8e8m0[]> ref_scales_colwise = std::make_unique<fp8e8m0[]>(blocks_num_colwise);

    constexpr bool rowwise = true;
    constexpr bool colwise = true;
    fill_tensor_data<InputType>(input, ref_scales_rowwise.get(), ref_scales_colwise.get(),
                                rowwise, colwise, rows, cols, blocks_num_rowwise, blocks_num_colwise);

    nvte_dequantize(input.data(), output.data(), 0);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref_x2<InputType, OutputType>(input.rowwise_cpu_dptr<InputType>(),
                                          ref_output_rowwise.get(),
                                          ref_output_colwise.get(),
                                          ref_scales_rowwise.get(),
                                          ref_scales_colwise.get(),
                                          rows,
                                          cols,
                                          block_size_rows,
                                          block_size_cols,
                                          scales_stride_rowwise,
                                          scales_stride_colwise);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_rowwise", output, ref_output_rowwise.get(), true, atol, rtol);
    compareResults("output_colwise", output, ref_output_colwise.get(), false, atol, rtol);
}

std::vector<std::pair<size_t, size_t>> tensor_dims = {
    {1, 16},
    {16, 48},
    {65, 96},
    {128, 128},
    {256, 256},
    {993, 512},
    {768, 1024},
    // {2048, 12288},
    // {65536, 128},
    // {16384, 1632},
    // {16384, 6144},
};

std::vector<std::pair<size_t, size_t>> block_sizes = {
    {1, 32},
    {32, 1},
    // {32, 32},
};

}  // namespace

class DequantizeMXFP8TestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                std::pair<size_t, size_t>,
                transformer_engine::DType,
                transformer_engine::DType,
                bool>> {};

TEST_P(DequantizeMXFP8TestSuite, TestDequantizeMXFP8)
{
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const auto tensor_size = std::get<0>(GetParam());
    const auto block_size = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const DType output_type = std::get<3>(GetParam());
    const bool quantize_then_dequantize = std::get<4>(GetParam());

    const bool rowwise = block_size.second != 1;
    const bool colwise = block_size.first != 1;

    // Skip tests for dequantization along both dimensions
    if (rowwise && colwise) {
        GTEST_SKIP();
    }

    // Skip cases with invalid alignment
    if (rowwise && tensor_size.second % 32 != 0) {
        GTEST_SKIP();
    }
    if (colwise && tensor_size.first % 32 != 0) {
        GTEST_SKIP();
    }

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(input_type, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(output_type, OutputType,
            if (quantize_then_dequantize) {
                // Mind the order of the Output/Input template parameters
                performTest_quantize_then_dequantize<OutputType, InputType>(
                    tensor_size.first, tensor_size.second, rowwise, colwise);
            } else {
                if (block_size.first == 1 || block_size.second == 1) {
                    performTest_x1<InputType, OutputType>(tensor_size.first, tensor_size.second,
                                                        rowwise, colwise);
                } else {
                    performTest_x2<InputType, OutputType>(tensor_size.first, tensor_size.second,
                                                        block_size.first, block_size.second);
                }
            }
        );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DequantizeMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(tensor_dims),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(false)),
    [](const testing::TestParamInfo<DequantizeMXFP8TestSuite::ParamType>& info)
    {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           std::to_string(std::get<1>(info.param).first) + "X" +
                           std::to_string(std::get<1>(info.param).second) + "X" +
                           test::typeName(std::get<2>(info.param)) + "X" +
                           test::typeName(std::get<3>(info.param)) + "X" +
                           (std::get<4>(info.param) ? "QD" : "D");
        return name;
    }
);
