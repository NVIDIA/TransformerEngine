/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

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
                 const size_t scale_idx_gate,
                 float& thread_amax,
                 const size_t i_min,
                 const size_t i_max,
                 const size_t j_min,
                 const size_t j_max,
                 const size_t cols) {

    float block_amax = 0.0f;
    float block_amax_gate = 0.0f;
    const size_t stride = cols * 2;

    // Find the absolute maximum value in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            float silu_elt = static_cast<float>(input[i * stride + j]);
            float gate_elt = static_cast<float>(input[i * stride + cols + j]);
            float gated_amax_act = 0;
            float gated_amax_gate = 0;

            if constexpr (IS_DGATED) {
                const float grad_elt = static_cast<float>(grad[i * cols + j]);
                const float after_dsilu = dsilu(silu_elt) * grad_elt * gate_elt;
                const float after_dgate = silu(silu_elt) * grad_elt;
                gated_amax_act = abs(after_dsilu);
                gated_amax_gate = abs(after_dgate);
            } else {
                const float after_silu = silu(silu_elt) * gate_elt;
                gated_amax_act = abs(after_silu);
            }

            if (gated_amax_act > block_amax) { block_amax = gated_amax_act; }
            if (gated_amax_gate > block_amax_gate) { block_amax_gate = gated_amax_gate; }
        }
    }

    const fp8e8m0 biased_exponent = float_to_e8m0(block_amax *
                                                  Quantized_Limits<OType>::max_reciprocal());
    const float scale_reciprocal = exp2f_rcp(biased_exponent);
    output_scales[scale_idx] = biased_exponent;
    float scale_reciprocal_gate = 1;
    if constexpr (IS_DGATED) {
      const fp8e8m0 biased_exponent = float_to_e8m0(block_amax_gate *
                                                    Quantized_Limits<OType>::max_reciprocal());
      scale_reciprocal_gate = exp2f_rcp(biased_exponent);
      output_scales[scale_idx_gate] = biased_exponent;
    }


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
                output[i * stride + cols + j] = static_cast<OType>(after_dgate *
                                                                   scale_reciprocal_gate);
            } else {
                const float after_silu = silu(silu_elt) * gate_elt;
                output[i * cols + j] = static_cast<OType>(after_silu * scale_reciprocal);
            }

        }
    }
    thread_amax = std::max(thread_amax, block_amax);
    thread_amax = std::max(thread_amax, block_amax_gate);
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
                    const size_t block_size_X,
                    const size_t scales_stride) {
    const size_t tile_size_Y = std::max(32lu, block_size_Y);
    const size_t tile_size_X = std::max(64lu, block_size_X);
    const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
    const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;
    const size_t blocks_per_tile_Y = tile_size_Y / block_size_Y;
    const size_t blocks_per_tile_X = tile_size_X / block_size_X;

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
                if (i_min >= rows) continue;
                const size_t i_max = std::min(i_min + block_size_Y, rows);

                for (size_t jj = 0; jj < blocks_per_tile_X; ++jj) {
                    const size_t block_idx_X = tile_X * blocks_per_tile_X + jj;
                    const size_t block_offset_X = jj * block_size_X;
                    const size_t j_min = tile_offset_X + block_offset_X;
                    if (j_min >= cols) continue;
                    const size_t j_max = std::min(j_min + block_size_X, cols);

                    const size_t mx_scale_idx = block_idx_Y * scales_stride + block_idx_X;
                    const size_t mx_scale_idx_gate = block_idx_Y * scales_stride + block_idx_X +
                                                     cols / block_size_X;
                    scale_block<IS_DGATED, IType, OType>(
                        grad, input, output, output_scales, mx_scale_idx, mx_scale_idx_gate,
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
                    const size_t block_size_X,
                    const size_t scales_stride_rowwise,
                    const size_t scales_stride_colwise) {
    compute_ref_x1<IS_DGATED, IType, OType>(
        grad, input, output_rowwise, scales_rowwise, ref_amax, rows, cols, 1, block_size_X, scales_stride_rowwise);
    compute_ref_x1<IS_DGATED, IType, OType>(
        grad, input, output_colwise, scales_colwise, ref_amax, rows, cols, block_size_Y, 1, scales_stride_colwise);
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

    const bool rowwise = (block_size_rows == 1) && (block_size_cols == 32);
    const bool colwise = (block_size_rows == 32) && (block_size_cols == 1);
    NVTE_CHECK(rowwise || colwise);

    // std::cout << "unpadded_blocks_Y: " << unpadded_blocks_Y << std::endl;
    // std::cout << "unpadded_blocks_X: " << unpadded_blocks_X << std::endl;
    // std::cout << "blocks_Y: " << blocks_Y << std::endl;
    // std::cout << "blocks_X: " << blocks_X << std::endl;
    // std::cout << "scales_stride: " << scales_stride << std::endl;

    Tensor grad("grad", std::vector<size_t>{ rows, cols }, itype);
    Tensor input("input", std::vector<size_t>{ rows, cols * 2 }, itype);

    const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, output_cols, block_size_rows,
                                                                  block_size_cols);

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    Tensor output("output", std::vector<size_t>{ rows, output_cols }, otype,
                  rowwise, colwise, NVTE_MXFP8_1D_SCALING);

    std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(rows * output_cols);
    std::unique_ptr<fp8e8m0[]> ref_output_scales = std::make_unique<fp8e8m0[]>(blocks_Y * blocks_X);

    for (size_t i = 0; i < blocks_Y * blocks_X; ++i) {
      ref_output_scales[i] = 0;
    }

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
                                            block_size_cols,
                                            scales_stride);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output", output, ref_output.get(), rowwise, atol, rtol);

    const uint8_t * const gpu_scales_ptr = rowwise
                                           ? output.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                           : output.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
    if (rowwise) {
      compare_e8m0_scaling_factors("rowwise scales", gpu_scales_ptr, ref_output_scales.get(),
                                   unpadded_blocks_Y, unpadded_blocks_X, scales_stride);
    } else {
      compare_e8m0_scaling_factors("colwise scales", gpu_scales_ptr, ref_output_scales.get(),
                                   unpadded_blocks_Y, unpadded_blocks_X, scales_stride);
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

    Tensor grad("grad", std::vector<size_t>{ rows, cols }, itype);
    Tensor input("input", std::vector<size_t>{ rows, cols * 2 }, itype);

    const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

    const std::array<size_t,4> scale_dims_rowwise = get_scale_tensor_dims(rows, output_cols, 1, 32);
    const std::array<size_t,4> scale_dims_colwise = get_scale_tensor_dims(rows, output_cols, 32, 1);

    const size_t unpadded_blocks_Y_rowwise = scale_dims_rowwise[0];
    const size_t unpadded_blocks_X_rowwise = scale_dims_rowwise[1];
    const size_t blocks_Y_rowwise = scale_dims_rowwise[2];
    const size_t blocks_X_rowwise = scale_dims_rowwise[3];
    const size_t scales_stride_rowwise = blocks_X_rowwise;

    const size_t unpadded_blocks_Y_colwise = scale_dims_colwise[0];
    const size_t unpadded_blocks_X_colwise = scale_dims_colwise[1];
    const size_t blocks_Y_colwise = scale_dims_colwise[2];
    const size_t blocks_X_colwise = scale_dims_colwise[3];
    const size_t scales_stride_colwise = blocks_X_colwise;

    Tensor output("output", std::vector<size_t>{ rows, output_cols }, otype,
                  true, true, NVTE_MXFP8_1D_SCALING);

    std::unique_ptr<OType[]> ref_output_rowwise = std::make_unique<OType[]>(rows * output_cols);
    std::unique_ptr<OType[]> ref_output_colwise = std::make_unique<OType[]>(rows * output_cols);
    std::unique_ptr<fp8e8m0[]> ref_scales_rowwise = std::make_unique<fp8e8m0[]>(blocks_Y_rowwise * blocks_X_rowwise);
    std::unique_ptr<fp8e8m0[]> ref_scales_colwise = std::make_unique<fp8e8m0[]>(blocks_Y_colwise * blocks_X_colwise);

    for (size_t i = 0; i < blocks_Y_rowwise * blocks_X_rowwise; ++i) {
      ref_scales_rowwise[i] = 0;
    }
    for (size_t i = 0; i < blocks_Y_colwise * blocks_X_colwise; ++i) {
      ref_scales_colwise[i] = 0;
    }

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
                                            block_size_cols,
                                            scales_stride_rowwise,
                                            scales_stride_colwise);

    auto [atol, rtol] = getTolerances(otype);
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("output_c_rowwise", output, ref_output_rowwise.get(), true, atol, rtol);
    compareResults("output_c_colwise", output, ref_output_colwise.get(), false, atol, rtol);
    compare_e8m0_scaling_factors("scales_rowwise", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_rowwise.get(), unpadded_blocks_Y_rowwise,
                                 unpadded_blocks_X_rowwise, scales_stride_rowwise);
    compare_e8m0_scaling_factors("scales_colwise", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_colwise.get(), unpadded_blocks_Y_colwise,
                                 unpadded_blocks_X_colwise, scales_stride_colwise);
}

std::vector<std::pair<size_t, size_t>> matrix_sizes = {
    {1, 32},
    {16, 64},
    {65, 96},
    {128, 128},
    {256, 256},
    {993, 512},
    {768, 1024},
    {65536, 128},
    {16384, 1632},
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
