/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

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

enum ProcessingMethod {
    CAST_ONLY,
    CAST_DBIAS,
    CAST_DBIAS_DACT,
    CAST_DACT,
    CAST_ACT
};

enum ActivationType {
    Identity,
    GeLU,
    SiLU,
    ReLU,
    QGeLU,
    SReLU
};

template <typename InputType, typename OutputType>
void compute_ref(const ProcessingMethod processing_method,
                 float (*OP)(const float),
                 const bool rowwise,
                 const bool colwise,
                 const InputType* input,
                 const InputType* grad,
                 OutputType* output_rowwise,
                 OutputType* output_colwise,
                 fp8e8m0* output_scales_rowwise,
                 fp8e8m0* output_scales_colwise,
                 InputType* output_dbias,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride_rowwise,
                 const size_t scales_stride_colwise)
{
    const size_t tile_size_Y = 32;
    const size_t tile_size_X = 32;
    const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
    const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;

    std::vector<float> output_dbias_fp32(cols, 0);
    #pragma omp parallel proc_bind(spread)
    {
        // Buffers to cache intermediate computations
        std::vector<float> cache_buffer(tile_size_Y * tile_size_X);

        std::vector<float> thread_dbias(cols, 0);
        #pragma omp for schedule(static)
        for (size_t t = 0; t < tiles_num_Y * tiles_num_X; ++t) {
            const size_t tile_Y = t / tiles_num_X;
            const size_t tile_X = t % tiles_num_X;
            const size_t tile_offset_Y = tile_Y * tile_size_Y;
            const size_t tile_offset_X = tile_X * tile_size_X;

            const size_t i_min = tile_offset_Y;
            const size_t i_max = std::min(i_min + tile_size_Y, rows);

            const size_t j_min = tile_offset_X;
            const size_t j_max = std::min(j_min + tile_size_X, cols);

            // Cache computations
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; ++j) {
                    const size_t idx = i * cols + j;
                    const size_t cache_idx = (i - i_min) * tile_size_X + (j - j_min);

                    float elt = static_cast<float>(input[idx]);
                    if (processing_method == ProcessingMethod::CAST_DBIAS) {
                        // grad is the input
                        elt = static_cast<float>(grad[idx]);
                    }
                    if (processing_method != ProcessingMethod::CAST_ONLY
                        && processing_method != ProcessingMethod::CAST_DBIAS) {
                        elt = OP(elt);
                    }
                    if (processing_method == ProcessingMethod::CAST_DACT ||
                        processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
                        elt *= static_cast<float>(grad[idx]);
                    }
                    thread_dbias[j] += elt;

                    // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                    elt = static_cast<float>(static_cast<InputType>(elt));

                    cache_buffer[cache_idx] = elt;
                    if (isinf(elt) || isnan(elt)) {
                        continue;
                    }
                }
            }

            if (rowwise) {
                for (size_t i = i_min; i < i_max; ++i) {
                    float block_amax = 0.0f;

                    for (size_t j = j_min; j < j_max; ++j) {
                        const size_t cache_idx = (i - i_min) * tile_size_X + (j - j_min);
                        block_amax = std::max(block_amax, std::abs(cache_buffer[cache_idx]));
                    }

                    const fp8e8m0 biased_exponent = float_to_e8m0(block_amax * Quantized_Limits<OutputType>::max_reciprocal());
                    const size_t scale_idx = i * scales_stride_rowwise + tile_X;
                    output_scales_rowwise[scale_idx] = biased_exponent;
                    const float scale_reciprocal = exp2f_rcp(biased_exponent);

                    for (size_t j = j_min; j < j_max; ++j) {
                        const size_t idx = i * cols + j;
                        const size_t cache_idx = (i - i_min) * tile_size_X + (j - j_min);
                        output_rowwise[idx] = static_cast<OutputType>(cache_buffer[cache_idx] * scale_reciprocal);
                    }
                }
            }
            if (colwise) {
                for (size_t j = j_min; j < j_max; ++j) {
                    float block_amax = 0.0f;

                    for (size_t i = i_min; i < i_max; ++i) {
                        const size_t cache_idx = (i - i_min) * tile_size_X + (j - j_min);
                        block_amax = std::max(block_amax, std::abs(cache_buffer[cache_idx]));
                    }

                    const fp8e8m0 biased_exponent = float_to_e8m0(block_amax * Quantized_Limits<OutputType>::max_reciprocal());
                    const size_t scale_idx = tile_Y * scales_stride_colwise + j;
                    output_scales_colwise[scale_idx] = biased_exponent;
                    const float scale_reciprocal = exp2f_rcp(biased_exponent);

                    for (size_t i = i_min; i < i_max; ++i) {
                        const size_t idx = i * cols + j;
                        const size_t cache_idx = (i - i_min) * tile_size_X + (j - j_min);
                        output_colwise[idx] = static_cast<OutputType>(cache_buffer[cache_idx] * scale_reciprocal);
                    }
                }
            }
        }
        #pragma omp critical
        {
            for (size_t j = 0; j < cols; ++j) {
                output_dbias_fp32[j] += thread_dbias[j];
            }
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        output_dbias[j] = static_cast<InputType>(output_dbias_fp32[j]);
    }
}

/**
 * Scaling along single dimension (either rows or columns)
 * Produces one set of output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *       OR
 * 2) Scaled columns + column-wise scaling factors
 */

template <typename InputType, typename OutputType>
void performTest_x1(const ProcessingMethod processing_method,
                    float (*OP)(const float),
                    const std::vector<size_t>& shape,
                    const bool rowwise,
                    const bool colwise,
                    InputsFillCase fill_case) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    if (shape.size() < 2 && colwise) {
      GTEST_SKIP();
    }

    const size_t block_size_rows = rowwise ? 1 : 32;
    const size_t block_size_cols = colwise ? 1 : 32;

    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, block_size_rows,
                                                                  block_size_cols);

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    Tensor input("input", shape, itype);
    Tensor grad("grad", shape, itype);
    Tensor output_c("output_c", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
    Tensor output_dbias("output_dbias", std::vector<size_t>{ cols }, itype);

    std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<InputType[]> ref_output_dbias = std::make_unique<InputType[]>(cols);
    std::unique_ptr<fp8e8m0[]> ref_output_scales = std::make_unique<fp8e8m0[]>(blocks_Y * blocks_X);

    fillCase<EncodingType>(&input, fill_case);
    fillUniform(&grad);

    Tensor workspace;
    switch (processing_method) {
        case ProcessingMethod::CAST_ONLY: {
            nvte_quantize(input.data(), output_c.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS: {
            nvte_quantize_dbias(grad.data(),
                                output_c.data(),
                                output_dbias.data(),
                                workspace.data(),
                                0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

            nvte_quantize_dbias(grad.data(),
                                output_c.data(),
                                output_dbias.data(),
                                workspace.data(),
                                0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS_DACT: {
            auto nvte_quantize_dbias_dact = &nvte_quantize_dbias_dgelu;
            if (OP == &dsilu)       { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsilu; }
            else if (OP == &drelu)  { nvte_quantize_dbias_dact = &nvte_quantize_dbias_drelu; }
            else if (OP == &dqgelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dqgelu; }
            else if (OP == &dsrelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsrelu; }

            nvte_quantize_dbias_dact(grad.data(),
                                     input.data(),
                                     output_c.data(),
                                     output_dbias.data(),
                                     workspace.data(),
                                     0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

            nvte_quantize_dbias_dact(grad.data(),
                                     input.data(),
                                     output_c.data(),
                                     output_dbias.data(),
                                     workspace.data(),
                                     0);
            break;
        }
        case ProcessingMethod::CAST_DACT: {
            auto nvte_dact = &nvte_dgelu;
            if (OP == &dsilu)       { nvte_dact = &nvte_dsilu; }
            else if (OP == &drelu)  { nvte_dact = &nvte_drelu; }
            else if (OP == &dqgelu) { nvte_dact = &nvte_dqgelu; }
            else if (OP == &dsrelu) { nvte_dact = &nvte_dsrelu; }

            nvte_dact(grad.data(), input.data(), output_c.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_ACT: {
            auto nvte_act = &nvte_gelu;
            if (OP == &silu)       { nvte_act = &nvte_silu; }
            else if (OP == &relu)  { nvte_act = &nvte_relu; }
            else if (OP == &qgelu) { nvte_act = &nvte_qgelu; }
            else if (OP == &srelu) { nvte_act = &nvte_srelu; }

            nvte_act(input.data(), output_c.data(), 0);
            break;
        }
    }

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref<InputType, OutputType>(processing_method,
                                       OP,
                                       rowwise,
                                       colwise,
                                       input.rowwise_cpu_dptr<InputType>(),
                                       grad.rowwise_cpu_dptr<InputType>(),
                                       ref_output_c.get(),
                                       ref_output_c.get(),
                                       ref_output_scales.get(),
                                       ref_output_scales.get(),
                                       ref_output_dbias.get(),
                                       rows,
                                       cols,
                                       scales_stride,
                                       scales_stride);

    const uint8_t * const gpu_scales_ptr = rowwise
                                           ? output_c.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                           : output_c.columnwise_cpu_scale_inv_ptr<fp8e8m0>();

    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 0.0;
    const double rel_tolerable_mismatches_limit = 0.0;

    size_t mismatches_scales = 0;
    compare_e8m0_scaling_factors("scales", gpu_scales_ptr, ref_output_scales.get(),
                                 unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
                                 mismatches_scales,
                                 scale_diff_abs_tolerance,
                                 abs_tolerable_mismatches_limit,
                                 rel_tolerable_mismatches_limit);

    const size_t mismatches_elts = 32 * mismatches_scales;
    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_c", output_c, ref_output_c.get(), rowwise, atol, rtol, true, mismatches_elts);

    if (processing_method == ProcessingMethod::CAST_DBIAS
        || processing_method == ProcessingMethod::CAST_DBIAS_DACT)
    {
        auto [atol_dbias, rtol_dbias] = getTolerances(itype);
        if (itype == DType::kFloat32) {
            atol_dbias = 1e-4;
            rtol_dbias *= sqrt(static_cast<double>(rows)) ;
        } else {
            rtol_dbias *= 4;
        }
        compareResults("output_dbias", output_dbias, ref_output_dbias.get(), true, atol_dbias, rtol_dbias);
    }
}

/**
 * Scaling along both dimensions (rows and columns)
 * Produces two sets of scaled output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *      AND
 * 2) Scaled columns + column-wise scaling factors
 */
template <typename InputType, typename OutputType>
void performTest_x2(const ProcessingMethod processing_method,
                    float (*OP)(const float),
                    const std::vector<size_t>& shape,
                    const size_t block_size_rows,
                    const size_t block_size_cols,
                    InputsFillCase fill_case) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    if (shape.size() < 2) {
      GTEST_SKIP();
    }

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    const std::array<size_t,4> scale_dims_rowwise = get_scale_tensor_dims(rows, cols, 1, 32);
    const std::array<size_t,4> scale_dims_colwise = get_scale_tensor_dims(rows, cols, 32, 1);

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

    Tensor input("input", shape, itype);
    Tensor grad("grad", shape, itype);
    Tensor output("output", shape, otype, true, true, NVTE_MXFP8_1D_SCALING);
    Tensor output_dbias("output_dbias", std::vector<size_t>{ cols }, itype);

    std::unique_ptr<OutputType[]> ref_output_c_rowwise = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<OutputType[]> ref_output_c_colwise = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<fp8e8m0[]> ref_scales_rowwise = std::make_unique<fp8e8m0[]>(blocks_Y_rowwise * blocks_X_rowwise);
    std::unique_ptr<fp8e8m0[]> ref_scales_colwise = std::make_unique<fp8e8m0[]>(blocks_Y_colwise * blocks_X_colwise);
    std::unique_ptr<InputType[]> ref_output_dbias = std::make_unique<InputType[]>(cols);

    fillCase<EncodingType>(&input, fill_case);
    fillUniform(&grad);

    Tensor workspace;
    switch (processing_method) {
        case ProcessingMethod::CAST_ONLY: {
            nvte_quantize(input.data(), output.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS: {
            nvte_quantize_dbias(grad.data(),
                                output.data(),
                                output_dbias.data(),
                                workspace.data(),
                                0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

            nvte_quantize_dbias(grad.data(),
                                output.data(),
                                output_dbias.data(),
                                workspace.data(),
                                0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS_DACT: {
            auto nvte_quantize_dbias_dact = &nvte_quantize_dbias_dgelu;
            if (OP == &dsilu)       { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsilu; }
            else if (OP == &drelu)  { nvte_quantize_dbias_dact = &nvte_quantize_dbias_drelu; }
            else if (OP == &dqgelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dqgelu; }
            else if (OP == &dsrelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsrelu; }

            nvte_quantize_dbias_dact(grad.data(),
                                     input.data(),
                                     output.data(),
                                     output_dbias.data(),
                                     workspace.data(),
                                     0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

            nvte_quantize_dbias_dact(grad.data(),
                                     input.data(),
                                     output.data(),
                                     output_dbias.data(),
                                     workspace.data(),
                                     0);
            break;
        }
        case ProcessingMethod::CAST_DACT: {
            auto nvte_dact = &nvte_dgelu;
            if (OP == &dsilu)       { nvte_dact = &nvte_dsilu; }
            else if (OP == &drelu)  { nvte_dact = &nvte_drelu; }
            else if (OP == &dqgelu) { nvte_dact = &nvte_dqgelu; }
            else if (OP == &dsrelu) { nvte_dact = &nvte_dsrelu; }

            nvte_dact(grad.data(), input.data(), output.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_ACT: {
            auto nvte_act = &nvte_gelu;
            if (OP == &silu)       { nvte_act = &nvte_silu; }
            else if (OP == &relu)  { nvte_act = &nvte_relu; }
            else if (OP == &qgelu) { nvte_act = &nvte_qgelu; }
            else if (OP == &srelu) { nvte_act = &nvte_srelu; }

            nvte_act(input.data(), output.data(), 0);
            break;
        }
    }

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref<InputType, OutputType>(processing_method,
                                       OP,
                                       true,
                                       true,
                                       input.rowwise_cpu_dptr<InputType>(),
                                       grad.rowwise_cpu_dptr<InputType>(),
                                       ref_output_c_rowwise.get(),
                                       ref_output_c_colwise.get(),
                                       ref_scales_rowwise.get(),
                                       ref_scales_colwise.get(),
                                       ref_output_dbias.get(),
                                       rows,
                                       cols,
                                       scales_stride_rowwise,
                                       scales_stride_colwise);

    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 0.0;
    const double rel_tolerable_mismatches_limit = 0.0;

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
    compareResults("output_c_rowwise", output, ref_output_c_rowwise.get(), true, atol, rtol, true, mismatches_elts_rowwise);
    compareResults("output_c_colwise", output, ref_output_c_colwise.get(), false, atol, rtol, true, mismatches_elts_colwise);

    if (processing_method == ProcessingMethod::CAST_DBIAS
        || processing_method == ProcessingMethod::CAST_DBIAS_DACT)
    {
        auto [atol_dbias, rtol_dbias] = getTolerances(itype);
        if (itype == DType::kFloat32) {
            atol_dbias = 1e-4;
            rtol_dbias *= sqrt(static_cast<double>(rows)) ;
        } else {
            rtol_dbias *= 4;
        }
        compareResults("output_dbias", output_dbias, ref_output_dbias.get(), true, atol_dbias, rtol_dbias);
    }
}

std::vector<std::vector<size_t>> matrix_sizes = {
    {1, 16},
    {16, 48},
    {65, 96},
    {128, 128},
    {256, 256},
    {993, 512},
    {511, 6144},
    {8192, 128},
    {2048, 160},
    {577, 1632},
    {1024},
    {8, 32, 1024},
    {16, 8, 4, 512},
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

std::vector<ProcessingMethod> processing_methods = {
    ProcessingMethod::CAST_ONLY,
    ProcessingMethod::CAST_DBIAS,
    ProcessingMethod::CAST_DBIAS_DACT,
    ProcessingMethod::CAST_DACT,
    ProcessingMethod::CAST_ACT,
};

// Only GeLU activation tests are supported
std::vector<ActivationType> Activation_types = {
    ActivationType::Identity,
    ActivationType::GeLU,
    // ActivationType::SiLU,
    // ActivationType::ReLU,
    // ActivationType::QGeLU,
    // ActivationType::SReLU,
};

}  // namespace

class FusedCastMXFP8TestSuite : public ::testing::TestWithParam
    <std::tuple<ProcessingMethod,
                ActivationType,
                std::vector<size_t>,
                std::pair<size_t, size_t>,
                transformer_engine::DType,
                transformer_engine::DType,
                InputsFillCase>> {};

TEST_P(FusedCastMXFP8TestSuite, TestFusedCastMXFP8) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const ProcessingMethod processing_method = std::get<0>(GetParam());
    const ActivationType Act_type = std::get<1>(GetParam());
    const auto matrix_size = std::get<2>(GetParam());
    const auto block_size = std::get<3>(GetParam());
    const DType input_type = std::get<4>(GetParam());
    const DType output_type = std::get<5>(GetParam());
    const InputsFillCase fill_case = std::get<6>(GetParam());

    // Skips non Act tests if the Activation type is not an identity
    if ((processing_method == ProcessingMethod::CAST_ONLY || processing_method == ProcessingMethod::CAST_DBIAS)
        && Act_type != ActivationType::Identity) {
        GTEST_SKIP();
    }
    // Skips Act tests if the Activation is an identity
    if ((processing_method == ProcessingMethod::CAST_DBIAS_DACT
        || processing_method == ProcessingMethod::CAST_DACT
        || processing_method == ProcessingMethod::CAST_ACT) && (Act_type == ActivationType::Identity)) {
        GTEST_SKIP();
    }

    const bool rowwise = block_size.second != 1;
    const bool colwise = block_size.first != 1;
    if (processing_method == ProcessingMethod::CAST_ACT) {
        // Forward activations
        auto OP = &identity;
        switch (Act_type) {
            case ActivationType::GeLU: OP = &gelu; break;
            case ActivationType::SiLU: OP = &silu; break;
            case ActivationType::ReLU: OP = &relu; break;
            case ActivationType::QGeLU: OP = &qgelu; break;
            case ActivationType::SReLU: OP = &srelu; break;
        }

        TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
                if (block_size.first == 1 || block_size.second == 1) {
                    performTest_x1<InputType, OutputType>(
                        processing_method, OP, matrix_size,
                        rowwise, colwise, fill_case);
                } else {
                    performTest_x2<InputType, OutputType>(
                        processing_method, OP, matrix_size,
                        block_size.first, block_size.second, fill_case);
                }
            );
        );
    } else {
        auto OP = &identity;
        switch (Act_type) {
            case ActivationType::GeLU: OP = &dgelu; break;
            case ActivationType::SiLU: OP = &dsilu; break;
            case ActivationType::ReLU: OP = &drelu; break;
            case ActivationType::QGeLU: OP = &dqgelu; break;
            case ActivationType::SReLU: OP = &dsrelu; break;
        }
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
                if (block_size.first == 1 || block_size.second == 1) {
                    performTest_x1<InputType, OutputType>(
                        processing_method, OP, matrix_size,
                        rowwise, colwise, fill_case);
                } else {
                    performTest_x2<InputType, OutputType>(
                        processing_method, OP, matrix_size,
                        block_size.first, block_size.second, fill_case);
                }
            );
        );
    }
}

std::string to_string(const ProcessingMethod method) {
    switch (method) {
        case ProcessingMethod::CAST_ONLY:       return "CAST_ONLY";
        case ProcessingMethod::CAST_DBIAS:      return "CAST_DBIAS";
        case ProcessingMethod::CAST_DBIAS_DACT: return "CAST_DBIAS_DACT";
        case ProcessingMethod::CAST_DACT:       return "CAST_DACT";
        case ProcessingMethod::CAST_ACT:        return "CAST_ACT";
        default: return "";
    }
}

std::string to_string(const ActivationType Act_type) {
    switch (Act_type) {
        case ActivationType::Identity:  return "Identity";
        case ActivationType::GeLU:      return "GeLU";
        case ActivationType::SiLU:      return "SiLU";
        case ActivationType::ReLU:      return "ReLU";
        case ActivationType::QGeLU:     return "QGeLU";
        case ActivationType::SReLU:     return "SReLU";
        default: return "";
    }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    FusedCastMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(processing_methods),
        ::testing::ValuesIn(Activation_types),
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(input_scenarios)),
    [](const testing::TestParamInfo<FusedCastMXFP8TestSuite::ParamType>& info) {
        std::string name = to_string(std::get<0>(info.param)) + "X" +
                           to_string(std::get<1>(info.param));
      const auto& shape = std::get<2>(info.param);
      for ( const auto& s: shape) {
        name += "X" + std::to_string(s);
      }
      name += "X" + std::to_string(std::get<3>(info.param).first) +
              "X" + std::to_string(std::get<3>(info.param).second) +
              "X" + test::typeName(std::get<4>(info.param)) +
              "X" + test::typeName(std::get<5>(info.param)) +
              "X" + test::caseName(std::get<6>(info.param));
        return name;
    });
