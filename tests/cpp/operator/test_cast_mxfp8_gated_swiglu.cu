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

#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

template <typename IType, typename OType>
void scale_block(const IType* grad,
                 const IType* input,
                 OType* output,
                 e8m0_t* output_scales,
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
            float grad_elt = static_cast<float>(grad[i * cols + j]);
            float silu_elt = static_cast<float>(input[i * stride + j]);
            float gate_elt = static_cast<float>(input[i * stride + cols + j]);

            float after_dsilu = dsilu(silu_elt) * grad_elt * gate_elt;
            float after_dgate = silu(silu_elt) * grad_elt;

            const float gated_amax = max(abs(after_dsilu), abs(after_dgate));

            if (abs(gated_amax) > block_amax) { block_amax = abs(gated_amax); }
        }
    }

    const e8m0_t biased_exponent = float_to_e8m0(block_amax * Quantized_Limits<OType>::max_reciprocal());
    const float scale_reciprocal = exp2f_rcp(biased_exponent);
    output_scales[scale_idx] = biased_exponent;

    // Quantize elements in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            float grad_elt = static_cast<float>(grad[i * cols + j]);
            float silu_elt = static_cast<float>(input[i * stride + j]);
            float gate_elt = static_cast<float>(input[i * stride + cols + j]);

            float after_dsilu = dsilu(silu_elt) * grad_elt * gate_elt;
            float after_dgate = silu(silu_elt) * grad_elt;

            output[i * stride + j] = static_cast<OType>(after_dsilu * scale_reciprocal);
            output[i * stride + cols + j] = static_cast<OType>(after_dgate * scale_reciprocal);
        }
    }
    thread_amax = std::max(thread_amax, block_amax);
}

template <typename IType, typename OType>
void compute_ref_x1(const IType* grad,
                    const IType* input,
                    OType* output,
                    e8m0_t* output_scales,
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
                    scale_block<IType, OType>(grad, input, output, output_scales, mx_scale_idx,
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

template <typename IType, typename OType>
void compute_ref_x2(const IType* grad,
                    const IType* input,
                    OType* output_rowwise,
                    OType* output_colwise,
                    e8m0_t* scales_rowwise,
                    e8m0_t* scales_colwise,
                    float& ref_amax,
                    const size_t rows,
                    const size_t cols,
                    const size_t block_size_Y,
                    const size_t block_size_X) {
    compute_ref_x1<IType, OType>(
        grad, input, output_rowwise, scales_rowwise, ref_amax, rows, cols, 1, block_size_X);
    compute_ref_x1<IType, OType>(
        grad, input, output_colwise, scales_colwise, ref_amax, rows, cols, block_size_Y, 1);
}

/**
 * Scaling along single dimension (either rows or columns)
 * Produces one set of output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *       OR
 * 2) Scaled columns + column-wise scaling factors
 */

template <typename IType, typename OType>
void performTest_x1(const size_t rows,
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
    const size_t blocks_num = blocks_Y * blocks_X;

    const int block_rows_dim = static_cast<int>(block_size_rows);
    const int block_cols_dim = static_cast<int>(block_size_cols);
    const int is_delayed_scaling = false;

    Tensor grad({ rows, cols }, itype);
    Tensor input({ rows, cols * 2 }, itype);
    Tensor output({ rows, cols * 2 }, otype, { block_rows_dim, block_cols_dim, is_delayed_scaling});

    std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(rows * cols * 2);
    std::unique_ptr<e8m0_t[]> ref_output_scales = std::make_unique<e8m0_t[]>(blocks_Y * blocks_X);

    // fillCase<EncodingType>(&grad, fill_case);
    fillUniform(&grad);
    fillUniform(&input);

    nvte_fp8_quantize_swiglu(grad.data(), input.data(), output.data(), 0);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float ref_amax = 0;
    compute_ref_x1<IType, OType>(grad.cpu_dptr<IType>(),
                                 input.cpu_dptr<IType>(),
                                 ref_output.get(),
                                 ref_output_scales.get(),
                                 ref_amax,
                                 rows,
                                 cols,
                                 block_size_rows,
                                 block_size_cols);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output", output, ref_output.get(), atol, rtol);
    compare_e8m0_scaling_factors("scales", output.cpu_scale_inv_ptr<e8m0_t>(), ref_output_scales.get(), blocks_num);
}

/**
 * Scaling along both dimensions (rows and columns)
 * Produces two sets of scaled output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *      AND
 * 2) Scaled columns + column-wise scaling factors
 */
template <typename IType, typename OType>
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

    const int block_rows_dim = static_cast<int>(block_size_rows);
    const int block_cols_dim = static_cast<int>(block_size_cols);
    const int is_delayed_scaling = false;

    Tensor grad({ rows, cols }, itype);
    Tensor input({ rows, cols * 2 }, itype);
    Tensor output_rowwise({ rows, cols * 2 }, otype, { 1, block_cols_dim, is_delayed_scaling});
    Tensor output_colwise({ rows, cols * 2 }, otype, { block_rows_dim, 1, is_delayed_scaling});

    std::unique_ptr<OType[]> ref_output_rowwise = std::make_unique<OType[]>(rows * cols * 2);
    std::unique_ptr<OType[]> ref_output_colwise = std::make_unique<OType[]>(rows * cols * 2);
    std::unique_ptr<e8m0_t[]> ref_scales_rowwise = std::make_unique<e8m0_t[]>(rows * blocks_X);
    std::unique_ptr<e8m0_t[]> ref_scales_colwise = std::make_unique<e8m0_t[]>(blocks_Y * cols);

    // fillCase<EncodingType>(&grad, fill_case);
    fillUniform(&grad);
    fillUniform(&input);

    nvte_fp8_quantize_swiglu_x2(grad.data(), input.data(), output_rowwise.data(), output_colwise.data(), 0);
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float ref_amax = 0;
    compute_ref_x2<IType, OType>(grad.cpu_dptr<IType>(),
                                 input.cpu_dptr<IType>(),
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
    compareResults("output_c_rowwise", output_rowwise, ref_output_rowwise.get(), atol, rtol);
    compareResults("output_c_colwise", output_colwise, ref_output_colwise.get(), atol, rtol);
    compare_e8m0_scaling_factors("scales_rowwise", output_rowwise.cpu_scale_inv_ptr<e8m0_t>(),
                                 ref_scales_rowwise.get(), blocks_num_rowwise);
    compare_e8m0_scaling_factors("scales_colwise", output_colwise.cpu_scale_inv_ptr<e8m0_t>(),
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

}  // namespace

class CastMXFP8SwiGLUTestSuite : public ::testing::TestWithParam
    <std::tuple<std::pair<size_t, size_t>,
                std::pair<size_t, size_t>,
                transformer_engine::DType,
                transformer_engine::DType,
                InputsFillCase>> {};

TEST_P(CastMXFP8SwiGLUTestSuite, TestCastMXFP8Swiglu) {
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

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OType,
            if (block_size.first == 1 || block_size.second == 1) {
                performTest_x1<IType, OType>(matrix_size.first, matrix_size.second,
                    block_size.first, block_size.second, fill_case);
            } else {
                performTest_x2<IType, OType>(matrix_size.first, matrix_size.second,
                    block_size.first, block_size.second, fill_case);
            }
        );
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastMXFP8SwiGLUTestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        // ::testing::Values(DType::kBFloat16),
        // ::testing::Values(DType::kFloat8E4M3),
        ::testing::ValuesIn(input_scenarios)),
    [](const testing::TestParamInfo<CastMXFP8SwiGLUTestSuite::ParamType>& info) {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           std::to_string(std::get<1>(info.param).first) + "X" +
                           std::to_string(std::get<1>(info.param).second) + "X" +
                           test::typeName(std::get<2>(info.param)) + "X" +
                           test::typeName(std::get<3>(info.param)) + "X" +
                           test::caseName(std::get<4>(info.param));
        return name;
    });
