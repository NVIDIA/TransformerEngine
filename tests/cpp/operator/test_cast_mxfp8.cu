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

template <typename InputType, typename OutputType, float (*OP)(const float)>
void scale_block(const ProcessingMethod processing_method,
                 const InputType* input,
                 const InputType* grad,
                 OutputType* output_c,
                 float* dbias,
                 fp8e8m0* output_scales,
                 const size_t scale_idx,
                 const size_t i_min,
                 const size_t i_max,
                 const size_t j_min,
                 const size_t j_max,
                 const size_t cols) {
    float amax = 0.0f;

    // Find the absolute maximum value in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            const size_t idx = i * cols + j;
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
            dbias[j] += elt;
            if (isinf(elt) || isnan(elt)) {
                continue;
            }
            amax = std::max(amax, std::abs(elt));
        }
    }

    const fp8e8m0 biased_exponent = float_to_e8m0(amax * Quantized_Limits<OutputType>::max_reciprocal());
    const float scale_reciprocal = exp2f_rcp(biased_exponent);
    output_scales[scale_idx] = biased_exponent;

    // Quantize elements in the block
    for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = j_min; j < j_max; ++j) {
            const size_t idx = i * cols + j;
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
            output_c[idx] = static_cast<OutputType>(elt * scale_reciprocal);
        }
    }
}

template <typename InputType, typename OutputType, float (*OP)(const float)>
void compute_ref_x1(const ProcessingMethod processing_method,
                    const InputType* input,
                    const InputType* grad,
                    OutputType* output_c,
                    fp8e8m0* output_scales,
                    InputType* output_dbias,
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

    std::vector<float> output_dbias_fp32(cols, 0);
    #pragma omp parallel proc_bind(spread)
    {
        std::vector<float> thread_dbias(cols, 0);
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

                    const size_t scale_idx = block_idx_Y * scales_stride + block_idx_X;
                    scale_block<InputType, OutputType, OP>(
                        processing_method, input, grad, output_c, thread_dbias.data(),
                        output_scales, scale_idx, i_min, i_max, j_min, j_max, cols);
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

template <typename InputType, typename OutputType, float (*OP)(const float)>
void compute_ref_x2(const ProcessingMethod processing_method,
                    const InputType* input,
                    const InputType* grad,
                    OutputType* output_rowwise,
                    OutputType* output_colwise,
                    fp8e8m0* scales_rowwise,
                    fp8e8m0* scales_colwise,
                    InputType* output_dbias,
                    const size_t rows,
                    const size_t cols,
                    const size_t block_size_Y,
                    const size_t block_size_X,
                    const size_t scales_stride_rowwise,
                    const size_t scales_stride_colwise) {
    compute_ref_x1<InputType, OutputType, OP>(
        processing_method, input, grad, output_rowwise, scales_rowwise, output_dbias,
        rows, cols, 1, block_size_X, scales_stride_rowwise);
    compute_ref_x1<InputType, OutputType, OP>(
        processing_method, input, grad, output_colwise, scales_colwise, output_dbias,
        rows, cols, block_size_Y, 1, scales_stride_colwise);
}

/**
 * Scaling along single dimension (either rows or columns)
 * Produces one set of output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *       OR
 * 2) Scaled columns + column-wise scaling factors
 */

template <typename InputType, typename OutputType, float (*OP)(const float)>
void performTest_x1(const ProcessingMethod processing_method,
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
            nvte_quantize_dbias_dgelu(grad.data(),
                                      input.data(),
                                      output_c.data(),
                                      output_dbias.data(),
                                      workspace.data(),
                                      0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

            nvte_quantize_dbias_dgelu(grad.data(),
                                      input.data(),
                                      output_c.data(),
                                      output_dbias.data(),
                                      workspace.data(),
                                      0);
            break;
        }
        case ProcessingMethod::CAST_DACT: {
            nvte_dgelu(grad.data(), input.data(), output_c.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_ACT: {
            nvte_gelu(input.data(), output_c.data(), 0);
            break;
        }
    }

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref_x1<InputType, OutputType, OP>(processing_method,
                                              input.rowwise_cpu_dptr<InputType>(),
                                              grad.rowwise_cpu_dptr<InputType>(),
                                              ref_output_c.get(),
                                              ref_output_scales.get(),
                                              ref_output_dbias.get(),
                                              rows,
                                              cols,
                                              block_size_rows,
                                              block_size_cols,
                                              scales_stride);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_c", output_c, ref_output_c.get(), rowwise, atol, rtol);

    const uint8_t * const gpu_scales_ptr = rowwise
                                           ? output_c.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
                                           : output_c.columnwise_cpu_scale_inv_ptr<fp8e8m0>();

    compare_e8m0_scaling_factors("scales", gpu_scales_ptr, ref_output_scales.get(),
                                 unpadded_blocks_Y, unpadded_blocks_X, scales_stride);

    if (processing_method == ProcessingMethod::CAST_DBIAS || processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
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
template <typename InputType, typename OutputType, float (*OP)(const float)>
void performTest_x2(const ProcessingMethod processing_method,
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
            nvte_quantize_dbias_dgelu(grad.data(),
                                      input.data(),
                                      output.data(),
                                      output_dbias.data(),
                                      workspace.data(),
                                      0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

            nvte_quantize_dbias_dgelu(grad.data(),
                                      input.data(),
                                      output.data(),
                                      output_dbias.data(),
                                      workspace.data(),
                                      0);
            break;
        }
        case ProcessingMethod::CAST_DACT: {
            nvte_dgelu(grad.data(), input.data(), output.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_ACT: {
            nvte_gelu(input.data(), output.data(), 0);
            break;
        }
    }

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref_x2<InputType, OutputType, OP>(processing_method,
                                              input.rowwise_cpu_dptr<InputType>(),
                                              grad.rowwise_cpu_dptr<InputType>(),
                                              ref_output_c_rowwise.get(),
                                              ref_output_c_colwise.get(),
                                              ref_scales_rowwise.get(),
                                              ref_scales_colwise.get(),
                                              ref_output_dbias.get(),
                                              rows,
                                              cols,
                                              block_size_rows,
                                              block_size_cols,
                                              scales_stride_rowwise,
                                              scales_stride_colwise);

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_c_rowwise", output, ref_output_c_rowwise.get(), true, atol, rtol);
    compareResults("output_c_colwise", output, ref_output_c_colwise.get(), false, atol, rtol);
    compare_e8m0_scaling_factors("scales_rowwise", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_rowwise.get(), unpadded_blocks_Y_rowwise,
                                 unpadded_blocks_X_rowwise, scales_stride_rowwise);
    compare_e8m0_scaling_factors("scales_colwise", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
                                 ref_scales_colwise.get(), unpadded_blocks_Y_colwise,
                                 unpadded_blocks_X_colwise, scales_stride_colwise);

    if (processing_method == ProcessingMethod::CAST_DBIAS || processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
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
    {256, 65536},
    {2048, 6144},
    {16384, 128},
    {32768, 160},
    {4096, 1632},
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

#define DACT_FUNC_SWITCH(OP_FUNC_TYPE, OP, ...) \
switch (OP_FUNC_TYPE) { \
    case ActivationType::Identity: { constexpr auto OP = &identity; { __VA_ARGS__ } } break; \
    case ActivationType::GeLU:     { constexpr auto OP = &dgelu;    { __VA_ARGS__ } } break; \
    case ActivationType::SiLU:     { constexpr auto OP = &dsilu;    { __VA_ARGS__ } } break; \
    case ActivationType::ReLU:     { constexpr auto OP = &drelu;    { __VA_ARGS__ } } break; \
    case ActivationType::QGeLU:    { constexpr auto OP = &dqgelu;   { __VA_ARGS__ } } break; \
    case ActivationType::SReLU:    { constexpr auto OP = &dsrelu;   { __VA_ARGS__ } } break; \
}

#define ACT_FUNC_SWITCH(OP_FUNC_TYPE, OP, ...) \
switch (OP_FUNC_TYPE) { \
    case ActivationType::Identity: { constexpr auto OP = &identity; { __VA_ARGS__ } } break; \
    case ActivationType::GeLU:     { constexpr auto OP = &gelu;    { __VA_ARGS__ } } break; \
    case ActivationType::SiLU:     { constexpr auto OP = &silu;    { __VA_ARGS__ } } break; \
    case ActivationType::ReLU:     { constexpr auto OP = &relu;    { __VA_ARGS__ } } break; \
    case ActivationType::QGeLU:    { constexpr auto OP = &qgelu;   { __VA_ARGS__ } } break; \
    case ActivationType::SReLU:    { constexpr auto OP = &srelu;   { __VA_ARGS__ } } break; \
}

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
        ACT_FUNC_SWITCH(Act_type, OP,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
                TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
                    if (block_size.first == 1 || block_size.second == 1) {
                        performTest_x1<InputType, OutputType, OP>(
                            processing_method, matrix_size,
                            rowwise, colwise, fill_case);
                    } else {
                        performTest_x2<InputType, OutputType, OP>(
                            processing_method, matrix_size,
                            block_size.first, block_size.second, fill_case);
                    }
                );
            );
        );
    } else {
        DACT_FUNC_SWITCH(Act_type, OP,
            TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
                TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
                    if (block_size.first == 1 || block_size.second == 1) {
                        performTest_x1<InputType, OutputType, OP>(
                            processing_method, matrix_size,
                            rowwise, colwise, fill_case);
                    } else {
                        performTest_x2<InputType, OutputType, OP>(
                            processing_method, matrix_size,
                            block_size.first, block_size.second, fill_case);
                    }
                );
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
