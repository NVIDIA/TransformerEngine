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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <bool IS_DGATED, typename IType, typename OType>
void scale_block(const IType* grad,
                 const IType* input,
                 OType* output,
                 fp8e8m0* output_scales,
                 const size_t scale_idx,
                 float& thread_amax,
                 const size_t i_min,
                 const size_t i_max,
                 const size_t j_min,
                 const size_t j_max,
                 const size_t cols) {

    float block_amax = 0.0f;
    const size_t stride = cols * 2;

    // Find the absolute maximum value in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            float silu_elt = static_cast<float>(input[i * stride + j]);
            float gate_elt = static_cast<float>(input[i * stride + cols + j]);
            float gated_amax;

            if constexpr (IS_DGATED) {
                const float grad_elt = static_cast<float>(grad[i * cols + j]);
                const float after_dsilu = dsilu(silu_elt) * grad_elt * gate_elt;
                const float after_dgate = silu(silu_elt) * grad_elt;
                gated_amax = max(abs(after_dsilu), abs(after_dgate));
            } else {
                const float after_silu = silu(silu_elt) * gate_elt;
                gated_amax = abs(after_silu);
            }

            if (abs(gated_amax) > block_amax) { block_amax = abs(gated_amax); }
        }
    }

    const fp8e8m0 biased_exponent = float_to_e8m0(block_amax * Quantized_Limits<OType>::max_reciprocal());
    const float scale_reciprocal = exp2f_rcp(biased_exponent);
    output_scales[scale_idx] = biased_exponent;

    // Quantize elements in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            float silu_elt = static_cast<float>(input[i * stride + j]);
            float gate_elt = static_cast<float>(input[i * stride + cols + j]);

            if constexpr (IS_DGATED) {
                const float grad_elt = static_cast<float>(grad[i * cols + j]);
                const float after_dsilu = dsilu(silu_elt) * grad_elt * gate_elt;
                const float after_dgate = silu(silu_elt) * grad_elt;
                output[i * stride + j] = static_cast<OType>(after_dsilu * scale_reciprocal);
                output[i * stride + cols + j] = static_cast<OType>(after_dgate * scale_reciprocal);
            } else {
                const float after_silu = silu(silu_elt) * gate_elt;
                output[i * cols + j] = static_cast<OType>(after_silu * scale_reciprocal);
            }

        }
    }
    thread_amax = std::max(thread_amax, block_amax);
}

template <bool IS_DGATED, typename IType, typename OType>
void compute_ref_x1(const IType* grad,
                    const IType* input,
                    OType* output,
                    fp8e8m0* output_scales,
                    float& ref_amax,
                    const size_t rows,
                    const size_t cols,
                    const size_t block_size_Y,
                    const size_t block_size_X) {
    const size_t tile_size_Y = std::max(32lu, block_size_Y);
    const size_t tile_size_X = std::max(64lu, block_size_X);
    const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
    const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;
    const size_t blocks_per_tile_Y = tile_size_Y / block_size_Y;
    const size_t blocks_per_tile_X = tile_size_X / block_size_X;
    const size_t blocks_per_row = (cols + block_size_X - 1) / block_size_X;

    float amax = 0;
    #pragma omp parallel reduction(max: amax) proc_bind(spread)
    {
        float thread_amax = 0;
        #pragma omp for schedule(static)
        for (size_t t = 0; t < tiles_num_Y * tiles_num_X; ++t) {
            const size_t tile_Y = t / tiles_num_X;
            const size_t tile_X = t % tiles_num_X;
            const size_t tile_offset_Y = tile_Y * tile_size_Y;
            const size_t tile_offset_X = tile_X * tile_size_X;

            for (size_t ii = 0; ii < blocks_per_tile_Y; ++ii) {
                const size_t block_idx_Y = tile_Y * blocks_per_tile_Y + ii;
                const size_t block_offset_Y = ii * block_size_Y;
                const size_t i_min = tile_offset_Y + block_offset_Y;
                const size_t i_max = std::min(i_min + block_size_Y, rows);

                for (size_t jj = 0; jj < blocks_per_tile_X; ++jj) {
                    const size_t block_idx_X = tile_X * blocks_per_tile_X + jj;
                    const size_t block_offset_X = jj * block_size_X;
                    const size_t j_min = tile_offset_X + block_offset_X;
                    const size_t j_max = std::min(j_min + block_size_X, cols);

                    const size_t mx_scale_idx = block_idx_Y * blocks_per_row + block_idx_X;
                    scale_block<IS_DGATED, IType, OType>(
                        grad, input, output, output_scales, mx_scale_idx,
                        thread_amax, i_min, i_max, j_min, j_max, cols);
                }
            }
        }
        if (thread_amax > amax) {
            amax = thread_amax;
        }
    }
    ref_amax = amax;
}

template <bool IS_DGATED, typename IType, typename OType>
void compute_ref_x2(const IType* grad,
                    const IType* input,
                    OType* output_rowwise,
                    OType* output_colwise,
                    fp8e8m0* scales_rowwise,
                    fp8e8m0* scales_colwise,
                    float& ref_amax,
                    const size_t rows,
                    const size_t cols,
                    const size_t block_size_Y,
                    const size_t block_size_X) {
    compute_ref_x1<IS_DGATED, IType, OType>(
        grad, input, output_rowwise, scales_rowwise, ref_amax, rows, cols, 1, block_size_X);
    compute_ref_x1<IS_DGATED, IType, OType>(
        grad, input, output_colwise, scales_colwise, ref_amax, rows, cols, block_size_Y, 1);
}

/**
 * Scaling along single dimension (either rows or columns)
 * Produces one set of output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *       OR
 * 2) Scaled columns + column-wise scaling factors
 */

template <bool IS_DGATED, typename IType, typename OType>
void performTest_x1(const size_t rows,
                    const size_t cols,
                    const size_t block_size_rows,
                    const size_t block_size_cols,
                    InputsFillCase fill_case) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<IType>::dtype;
    DType otype = TypeInfo<OType>::dtype;

    bool rowwise = false, colwise = false;
    if (block_size_rows == 1 && block_size_cols == 32) rowwise = true;
    if (block_size_rows == 32 && block_size_cols == 1) colwise = true;
    NVTE_CHECK(rowwise || colwise);

    const size_t blocks_Y = (rows + block_size_rows - 1) / block_size_rows;
    const size_t blocks_X = (cols + block_size_cols - 1) / block_size_cols;
    const size_t blocks_num = blocks_Y * blocks_X;

    Tensor grad({ rows, cols }, itype);
    Tensor input({ rows, cols * 2 }, itype);

    const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;
    Tensor output(std::vector<size_t>{ rows, output_cols }, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);

    std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(rows * output_cols);
    std::unique_ptr<fp8e8m0[]> ref_output_scales = std::make_unique<fp8e8m0[]>(blocks_Y * blocks_X);

    // fillCase<EncodingType>(&grad, fill_case);
    if constexpr (IS_DGATED) {
        fillUniform(&grad);
    }
    fillUniform(&input);

    if constexpr (IS_DGATED) {
        nvte_dswiglu(grad.data(), input.data(), output.data(), 0);
    } else {
        nvte_swiglu(input.data(), output.data(), 0);
    }
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float ref_amax = 0;
    compute_ref_x1<IS_DGATED, IType, OType>(grad.rowwise_cpu_dptr<IType>(),
                                            input.rowwise_cpu_dptr<IType>(),
                                            ref_output.get(),
                                            ref_output_scales.get(),
                                            ref_amax,
                                            rows,
                                            cols,
                                            block_size_rows,
                                            block_size_cols);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output", output, ref_output.get(), rowwise, atol, rtol);
    if (rowwise) {
      compare_e8m0_scaling_factors("scales", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(), ref_output_scales.get(), blocks_num);
    } else {
      compare_e8m0_scaling_factors("scales", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(), ref_output_scales.get(), blocks_num);
    }
}

/**
 * Scaling along both dimensions (rows and columns)
 * Produces two sets of scaled output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *      AND
 * 2) Scaled columns + column-wise scaling factors
 */
template <bool IS_DGATED, typename IType, typename OType>
void performTest_x2(const size_t rows,
                    const size_t cols,
                    const size_t block_size_rows,
                    const size_t block_size_cols,
                    InputsFillCase fill_case) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<IType>::dtype;
    DType otype = TypeInfo<OType>::dtype;

    const size_t blocks_Y = (rows + block_size_rows - 1) / block_size_rows;
    const size_t blocks_X = (cols + block_size_cols - 1) / block_size_cols;
    const size_t blocks_num_rowwise = rows * blocks_X;
    const size_t blocks_num_colwise = blocks_Y * cols;

    Tensor grad({ rows, cols }, itype);
    Tensor input({ rows, cols * 2 }, itype);

    const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;
    Tensor output(std::vector<size_t>{ rows, output_cols }, otype, true, true, NVTE_MXFP8_1D_SCALING);

    std::unique_ptr<OType[]> ref_output_rowwise = std::make_unique<OType[]>(rows * output_cols);
    std::unique_ptr<OType[]> ref_output_colwise = std::make_unique<OType[]>(rows * output_cols);
    std::unique_ptr<fp8e8m0[]> ref_scales_rowwise = std::make_unique<fp8e8m0[]>(rows * blocks_X);
    std::unique_ptr<fp8e8m0[]> ref_scales_colwise = std::make_unique<fp8e8m0[]>(blocks_Y * cols);

    // fillCase<EncodingType>(&grad, fill_case);
    if constexpr (IS_DGATED) {
        fillUniform(&grad);
    }
    fillUniform(&input);

    if constexpr (IS_DGATED) {
        nvte_dswiglu(grad.data(), input.data(), output.data(), 0);
    } else {
        nvte_swiglu(input.data(), output.data(), 0);
    }
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float ref_amax = 0;
    compute_ref_x2<IS_DGATED, IType, OType>(grad.rowwise_cpu_dptr<IType>(),
                                            input.rowwise_cpu_dptr<IType>(),
                                            ref_output_rowwise.get(),
                                            ref_output_colwise.get(),
                                            ref_scales_rowwise.get(),
                                            ref_scales_colwise.get(),
                                            ref_amax,
                                            rows,
                                            cols,
                                            block_size_rows,
                                            block_size_cols);

    auto [atol, rtol] = getTolerances(otype);
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("output_c_rowwise", output, ref_output_rowwise.get(), true, atol, rtol);
    compareResults("output_c_colwise", output, ref_output_colwise.get(), false, atol, rtol);
    compare_e8m0_scaling_factors("scales_rowwise", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_rowwise.get(), blocks_num_rowwise);
    compare_e8m0_scaling_factors("scales_colwise", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_colwise.get(), blocks_num_colwise);
}

std::vector<std::pair<size_t, size_t>> matrix_sizes = {
    {128, 128},
    {256, 256},
    {768, 1024},
    {256, 65536},
    // {2048, 12288},
    // {65536, 128},
    // {16384, 6144},
};

std::vector<std::pair<size_t, size_t>> block_sizes = {
    {1, 32},
    {32, 1},
    {32, 32},
};

std::vector<InputsFillCase> input_scenarios = {
    InputsFillCase::uniform,
    // InputsFillCase::zeros,
    // InputsFillCase::zero_to_minNorm,
    // InputsFillCase::minNorm_to_maxNorm,
    // InputsFillCase::maxNorm_to_inf
};

std::vector<bool> is_dgated_op = {
    true,
    false
};

}  // namespace

class CastMXFP8_GatedActTestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                std::pair<size_t, size_t>,
                transformer_engine::DType,
                transformer_engine::DType,
                InputsFillCase,
                bool>> {};

TEST_P(CastMXFP8_GatedActTestSuite, TestCastMXFP8Swiglu) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const auto matrix_size = std::get<0>(GetParam());
    const auto block_size = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const DType output_type = std::get<3>(GetParam());
    const InputsFillCase fill_case = std::get<4>(GetParam());
    const bool IS_DGATED = std::get<5>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OType,
            if (block_size.first == 1 || block_size.second == 1) {
                if (IS_DGATED) {
                    performTest_x1<true, IType, OType>(matrix_size.first, matrix_size.second,
                        block_size.first, block_size.second, fill_case);
                } else {
                    performTest_x1<false, IType, OType>(matrix_size.first, matrix_size.second,
                        block_size.first, block_size.second, fill_case);
                }
            } else {
                if (IS_DGATED) {
                    performTest_x2<true, IType, OType>(matrix_size.first, matrix_size.second,
                        block_size.first, block_size.second, fill_case);
                } else {
                    performTest_x2<false, IType, OType>(matrix_size.first, matrix_size.second,
                        block_size.first, block_size.second, fill_case);
                }
            }
        );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastMXFP8_GatedActTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(input_scenarios),
        ::testing::ValuesIn(is_dgated_op)),
    [](const testing::TestParamInfo<CastMXFP8_GatedActTestSuite::ParamType>& info) {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           std::to_string(std::get<1>(info.param).first) + "X" +
                           std::to_string(std::get<1>(info.param).second) + "X" +
                           test::typeName(std::get<2>(info.param)) + "X" +
                           test::typeName(std::get<3>(info.param)) + "X" +
                           test::caseName(std::get<4>(info.param)) + "X" +
                           (std::get<5>(info.param) ? "DGATED" : "GATED");
        return name;
    });
