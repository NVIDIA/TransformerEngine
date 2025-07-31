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

template <typename IType, typename OType>
void compute_ref(const IType* grad,
                 const IType* input,
                 OType* output_rowwise,
                 OType* output_colwise,
                 fp8e8m0* output_scales_rowwise,
                 fp8e8m0* output_scales_colwise,
                 float& ref_amax,
                 const bool IS_DGATED,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride_rowwise,
                 const size_t scales_stride_colwise,
                 const bool is_rowwise,
                 const bool is_colwise) {
    constexpr size_t tile_size_Y = 32;
    constexpr size_t tile_size_X = 32;
    const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
    const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;
    float amax = 0;
    #pragma omp parallel reduction(max: amax) proc_bind(spread)
    {
        // Buffers to cache intermediate computations
        std::vector<float> cache_buffer_act(tile_size_Y * tile_size_X);
        std::vector<float> cache_buffer_gate(tile_size_Y * tile_size_X);
        float thread_amax = 0.0f;
        #pragma omp for schedule(static)
        for (size_t t = 0; t < tiles_num_Y * tiles_num_X; ++t) {
            const size_t tile_Y = t / tiles_num_X;
            const size_t tile_X = t % tiles_num_X;
            const size_t tile_offset_Y = tile_Y * tile_size_Y;
            const size_t tile_offset_X = tile_X * tile_size_X;

            const size_t stride = cols * 2;

            const size_t i_min = tile_offset_Y;
            const size_t i_max = std::min(rows, tile_offset_Y + tile_size_Y);
            const size_t j_min = tile_offset_X;
            const size_t j_max = std::min(cols, tile_offset_X + tile_size_X);

            // Compute and cache activations for the entire tile
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; ++j) {
                    float silu_elt = static_cast<float>(input[i * stride + j]);
                    float gate_elt = static_cast<float>(input[i * stride + cols + j]);

                    const size_t cached_idx = (i - i_min) * tile_size_X + (j - j_min);

                    if (IS_DGATED) {
                        const float x = silu_elt;
                        const float s = sigmoid(x);
                        const float act_x = x * s;
                        const float dact_x = x * s * (1 - s) + s;

                        const float grad_elt = static_cast<float>(grad[i * cols + j]);
                        float after_dsilu = dact_x * grad_elt * gate_elt;
                        float after_dgate = act_x * grad_elt;

                        // Numerical truncation: after downcast to IType (BF16/FP16), upcast it back to FP32
                        after_dsilu = static_cast<float>(static_cast<IType>(after_dsilu));
                        after_dgate = static_cast<float>(static_cast<IType>(after_dgate));

                        cache_buffer_act[cached_idx] = after_dsilu;
                        cache_buffer_gate[cached_idx] = after_dgate;
                        thread_amax = std::max(thread_amax, std::abs(after_dsilu));
                        thread_amax = std::max(thread_amax, std::abs(after_dgate));
                    } else {
                        float after_silu = silu(silu_elt) * gate_elt;

                        // Numerical truncation: after downcast to IType (BF16/FP16), upcast it back to FP32
                        after_silu = static_cast<float>(static_cast<IType>(after_silu));

                        cache_buffer_act[cached_idx] = after_silu;
                        thread_amax = std::max(thread_amax, std::abs(after_silu));
                    }
                }
            }

            if (is_rowwise) {
                for (size_t i = i_min; i < i_max; ++i) {
                    float block_amax_act = 0.0f;
                    float block_amax_gate = 0.0f;
                    for (size_t j = j_min; j < j_max; ++j) {
                        const size_t cached_idx = (i - i_min) * tile_size_X + (j - j_min);
                        block_amax_act = std::max(block_amax_act, std::abs(cache_buffer_act[cached_idx]));
                        if (IS_DGATED) {
                            block_amax_gate = std::max(block_amax_gate, std::abs(cache_buffer_gate[cached_idx]));
                        }
                    }
                    const fp8e8m0 biased_exponent_act = float_to_e8m0(block_amax_act * Quantized_Limits<OType>::max_reciprocal());
                    const float scale_reciprocal_act = exp2f_rcp(biased_exponent_act);
                    const size_t scale_idx_act = i * scales_stride_rowwise + tile_X;
                    output_scales_rowwise[scale_idx_act] = biased_exponent_act;

                    float scale_reciprocal_gate;
                    if (IS_DGATED) {
                        const fp8e8m0 biased_exponent_gate = float_to_e8m0(block_amax_gate * Quantized_Limits<OType>::max_reciprocal());
                        scale_reciprocal_gate = exp2f_rcp(biased_exponent_gate);
                        const size_t scale_idx_gate = scale_idx_act + (cols + 32 - 1) / 32;
                        output_scales_rowwise[scale_idx_gate] = biased_exponent_gate;
                    }
                    for (size_t j = j_min; j < j_max; ++j) {
                        const size_t cached_idx = (i - i_min) * tile_size_X + (j - j_min);
                        const float after_act = cache_buffer_act[cached_idx] * scale_reciprocal_act;

                        if (IS_DGATED) {
                            const float after_gate = cache_buffer_gate[cached_idx] * scale_reciprocal_gate;
                            output_rowwise[i * stride + j] = static_cast<OType>(after_act);
                            output_rowwise[i * stride + cols + j] = static_cast<OType>(after_gate);
                        } else {
                            output_rowwise[i * cols + j] = static_cast<OType>(after_act);
                        }
                    }
                }
            }

            if (is_colwise) {
                for (size_t j = j_min; j < j_max; ++j) {
                    float block_amax_act = 0.0f;
                    float block_amax_gate = 0.0f;
                    for (size_t i = i_min; i < i_max; ++i) {
                        const size_t cached_idx = (i - i_min) * tile_size_X + (j - j_min);
                        block_amax_act = std::max(block_amax_act, std::abs(cache_buffer_act[cached_idx]));
                        if (IS_DGATED) {
                            block_amax_gate = std::max(block_amax_gate, std::abs(cache_buffer_gate[cached_idx]));
                        }
                    }
                    const fp8e8m0 biased_exponent_act = float_to_e8m0(block_amax_act * Quantized_Limits<OType>::max_reciprocal());
                    const float scale_reciprocal_act = exp2f_rcp(biased_exponent_act);
                    const size_t scale_idx_act = tile_Y * scales_stride_colwise + j;
                    output_scales_colwise[scale_idx_act] = biased_exponent_act;

                    float scale_reciprocal_gate;
                    if (IS_DGATED) {
                        const fp8e8m0 biased_exponent_gate = float_to_e8m0(block_amax_gate * Quantized_Limits<OType>::max_reciprocal());
                        const size_t scale_idx_gate = scale_idx_act + cols;
                        scale_reciprocal_gate = exp2f_rcp(biased_exponent_gate);
                        output_scales_colwise[scale_idx_gate] = biased_exponent_gate;
                    }
                    for (size_t i = i_min; i < i_max; ++i) {
                        const size_t cached_idx = (i - i_min) * tile_size_X + (j - j_min);
                        const float after_act = cache_buffer_act[cached_idx] * scale_reciprocal_act;

                        if (IS_DGATED) {
                            const float after_gate = cache_buffer_gate[cached_idx] * scale_reciprocal_gate;
                            output_colwise[i * stride + j] = static_cast<OType>(after_act);
                            output_colwise[i * stride + cols + j] = static_cast<OType>(after_gate);
                        } else {
                            output_colwise[i * cols + j] = static_cast<OType>(after_act);
                        }
                    }
                }
            }
        }
        if (thread_amax > amax) {
            amax = thread_amax;
        }
    }
    ref_amax = amax;
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
                    InputsFillCase fill_case,
                    const bool IS_DGATED) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<IType>::dtype;
    DType otype = TypeInfo<OType>::dtype;

    const bool rowwise = (block_size_rows == 1) && (block_size_cols == 32);
    const bool colwise = (block_size_rows == 32) && (block_size_cols == 1);
    NVTE_CHECK(rowwise || colwise);

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
    if (IS_DGATED) {
        fillUniform(&grad);
    }
    fillUniform(&input);

    if (IS_DGATED) {
        nvte_dswiglu(grad.data(), input.data(), output.data(), 0);
    } else {
        nvte_swiglu(input.data(), output.data(), 0);
    }
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float ref_amax = 0;
    compute_ref<IType, OType>(grad.rowwise_cpu_dptr<IType>(),
                              input.rowwise_cpu_dptr<IType>(),
                              ref_output.get(),
                              ref_output.get(),
                              ref_output_scales.get(),
                              ref_output_scales.get(),
                              ref_amax,
                              IS_DGATED,
                              rows,
                              cols,
                              scales_stride,
                              scales_stride,
                              rowwise,
                              colwise);

    size_t mismatches_scales = 0;
    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 1.0;
    const double rel_tolerable_mismatches_limit = 1.0e-4;

    const uint8_t * const gpu_scales_ptr = rowwise
                                           ? output.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                           : output.columnwise_cpu_scale_inv_ptr<fp8e8m0>();
    if (rowwise) {
      compare_e8m0_scaling_factors("rowwise scales", gpu_scales_ptr, ref_output_scales.get(),
                                   unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
                                   mismatches_scales,
                                   scale_diff_abs_tolerance,
                                   abs_tolerable_mismatches_limit,
                                   rel_tolerable_mismatches_limit);
    } else {
      compare_e8m0_scaling_factors("colwise scales", gpu_scales_ptr, ref_output_scales.get(),
                                   unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
                                   mismatches_scales,
                                   scale_diff_abs_tolerance,
                                   abs_tolerable_mismatches_limit,
                                   rel_tolerable_mismatches_limit);
    }

    const size_t mismatches_elts = 32 * mismatches_scales;
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output", output, ref_output.get(), rowwise, atol, rtol, true, mismatches_elts);
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
                    InputsFillCase fill_case,
                    const bool IS_DGATED) {
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
    if (IS_DGATED) {
        fillUniform(&grad);
    }
    fillUniform(&input);

    if (IS_DGATED) {
        nvte_dswiglu(grad.data(), input.data(), output.data(), 0);
    } else {
        nvte_swiglu(input.data(), output.data(), 0);
    }
    cudaDeviceSynchronize();

    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    float ref_amax = 0;
    compute_ref<IType, OType>(grad.rowwise_cpu_dptr<IType>(),
                              input.rowwise_cpu_dptr<IType>(),
                              ref_output_rowwise.get(),
                              ref_output_colwise.get(),
                              ref_scales_rowwise.get(),
                              ref_scales_colwise.get(),
                              ref_amax,
                              IS_DGATED,
                              rows,
                              cols,
                              scales_stride_rowwise,
                              scales_stride_colwise,
                              true,
                              true);

    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 1.0;
    const double rel_tolerable_mismatches_limit = 1.0e-4;

    size_t mismatches_scales_rowwise = 0;
    compare_e8m0_scaling_factors("scales_rowwise", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_rowwise.get(), unpadded_blocks_Y_rowwise,
                                 unpadded_blocks_X_rowwise, scales_stride_rowwise,
                                 mismatches_scales_rowwise,
                                 scale_diff_abs_tolerance,
                                 abs_tolerable_mismatches_limit,
                                 rel_tolerable_mismatches_limit);
    size_t mismatches_scales_colwise = 0;
    compare_e8m0_scaling_factors("scales_colwise", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_colwise.get(), unpadded_blocks_Y_colwise,
                                 unpadded_blocks_X_colwise, scales_stride_colwise,
                                 mismatches_scales_colwise,
                                 scale_diff_abs_tolerance,
                                 abs_tolerable_mismatches_limit,
                                 rel_tolerable_mismatches_limit);

    const size_t mismatches_elts_rowwise = 32 * mismatches_scales_rowwise;
    const size_t mismatches_elts_colwise = 32 * mismatches_scales_colwise;

    auto [atol, rtol] = getTolerances(otype);
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("output_c_rowwise", output, ref_output_rowwise.get(), true, atol, rtol, true, mismatches_elts_rowwise);
    compareResults("output_c_colwise", output, ref_output_colwise.get(), false, atol, rtol, true, mismatches_elts_colwise);
}

std::vector<std::pair<size_t, size_t>> matrix_sizes = {
    {1, 32},
    {16, 64},
    {65, 96},
    {128, 128},
    {256, 256},
    {993, 512},
    {768, 1024},
    {8192, 128},
    {577, 1632},
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

std::vector<bool> is_bwd_op = {
    false,
    true
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
                performTest_x1<IType, OType>(matrix_size.first, matrix_size.second,
                    block_size.first, block_size.second, fill_case, IS_DGATED);
            } else {
                performTest_x2<IType, OType>(matrix_size.first, matrix_size.second,
                    block_size.first, block_size.second, fill_case, IS_DGATED);
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
        ::testing::ValuesIn(is_bwd_op)),
    [](const testing::TestParamInfo<CastMXFP8_GatedActTestSuite::ParamType>& info) {
        std::string name = std::to_string(std::get<0>(info.param).first) + "X" +
                           std::to_string(std::get<0>(info.param).second) + "X" +
                           std::to_string(std::get<1>(info.param).first) + "X" +
                           std::to_string(std::get<1>(info.param).second) + "X" +
                           test::typeName(std::get<2>(info.param)) + "X" +
                           test::typeName(std::get<3>(info.param)) + "X" +
                           test::caseName(std::get<4>(info.param)) + "X" +
                           (std::get<5>(info.param) ? "BWD" : "FWD");
        return name;
    });
