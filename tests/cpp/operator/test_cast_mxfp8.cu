/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Compute the pre-quantization float values that the GPU kernel processes before
// casting to FP8. These are used as the reference for dequantized output comparison.
template <typename InputType>
void compute_ref(const ProcessingMethod processing_method,
                 float (*OP)(const float),
                 const InputType* input,
                 const InputType* grad,
                 float* ref_output,
                 InputType* output_dbias,
                 const size_t rows,
                 const size_t cols)
{
    std::vector<float> output_dbias_fp32(cols, 0);
    #pragma omp parallel proc_bind(spread)
    {
        std::vector<float> thread_dbias(cols, 0);
        #pragma omp for schedule(static) collapse(2)
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                const size_t idx = i * cols + j;
                const float in = static_cast<float>(input[idx]);
                float out;
                switch (processing_method) {
                case ProcessingMethod::CAST_ONLY:
                case ProcessingMethod::CAST_DBIAS:
                    out = in;
                    break;
                case ProcessingMethod::CAST_ACT:
                    out = OP(in);
                    break;
                case ProcessingMethod::CAST_DBIAS_DACT:
                case ProcessingMethod::CAST_DACT:
                    out = OP(in) * static_cast<float>(grad[idx]);
                    break;
                default:
                    NVTE_ERROR("Invalid processing mode (",
                               static_cast<int>(processing_method), ").");
                }
                thread_dbias[j] += out;
                // Match GPU: downcast to InputType then back to float before quantization.
                ref_output[idx] = static_cast<float>(static_cast<InputType>(out));
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
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    NVTE_CHECK(rowwise != colwise,
               "Expected either rowwise or columnwise scaling (rowwise=", rowwise,
               ", colwise=", colwise, ").");
    if (shape.size() < 2 && colwise) {
        GTEST_SKIP();
    }

    const size_t block_size_rows = rowwise ? 1 : 32;
    const size_t block_size_cols = colwise ? 1 : 32;
    const size_t scales_stride = get_scale_tensor_dims(rows, cols, block_size_rows, block_size_cols)[3];

    Tensor input("input", shape, itype);
    Tensor grad("grad", shape, itype);
    Tensor output_c("output_c", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
    Tensor output_dbias("output_dbias", std::vector<size_t>{ cols }, itype);

    std::vector<float> ref_output(rows * cols);
    std::unique_ptr<InputType[]> ref_output_dbias = std::make_unique<InputType[]>(cols);

    fillCase<fp32>(&input, fill_case);
    fillUniform(&grad);

    Tensor workspace;
    switch (processing_method) {
        case ProcessingMethod::CAST_ONLY: {
            nvte_quantize(input.data(), output_c.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS: {
            nvte_quantize_dbias(grad.data(), output_c.data(), output_dbias.data(),
                                workspace.data(), 0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
            nvte_quantize_dbias(grad.data(), output_c.data(), output_dbias.data(),
                                workspace.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS_DACT: {
            auto nvte_quantize_dbias_dact = &nvte_quantize_dbias_dgelu;
            if (OP == &dsilu)       { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsilu; }
            else if (OP == &drelu)  { nvte_quantize_dbias_dact = &nvte_quantize_dbias_drelu; }
            else if (OP == &dqgelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dqgelu; }
            else if (OP == &dsrelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsrelu; }
            nvte_quantize_dbias_dact(grad.data(), input.data(), output_c.data(),
                                     output_dbias.data(), workspace.data(), 0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
            nvte_quantize_dbias_dact(grad.data(), input.data(), output_c.data(),
                                     output_dbias.data(), workspace.data(), 0);
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

    // CPU reference: pre-quantization float values that the GPU kernel operates on.
    compute_ref<InputType>(processing_method, OP,
                           input.rowwise_cpu_dptr<InputType>(),
                           grad.rowwise_cpu_dptr<InputType>(),
                           ref_output.data(), ref_output_dbias.get(),
                           rows, cols);

    // Dequantize GPU output using the GPU's own scales, then compare to the
    // pre-quantization reference.  This is valid regardless of whether the GPU
    // and reference rounded a near-power-of-2 amax to the same E8M0 exponent.
    std::vector<float> dequant_output(rows * cols);
    if (rowwise) {
        dequantize_mxfp8_rowwise(
            output_c.rowwise_cpu_dptr<OutputType>(),
            output_c.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_output.data(), rows, cols, scales_stride);
    } else {
        dequantize_mxfp8_colwise(
            output_c.columnwise_cpu_dptr<OutputType>(),
            output_c.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_output.data(), rows, cols, scales_stride);
    }

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output_c", dequant_output.data(), ref_output.data(), rows * cols, atol, rtol);

    if (processing_method == ProcessingMethod::CAST_DBIAS
        || processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
        auto [atol_dbias, rtol_dbias] = getTolerances(itype);
        if (itype == DType::kFloat32) {
            atol_dbias = 1e-4;
            rtol_dbias *= sqrt(static_cast<double>(rows));
        } else {
            rtol_dbias *= 4;
        }
        compareResults("output_dbias", output_dbias, ref_output_dbias.get(), true,
                       atol_dbias, rtol_dbias);
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
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    if (shape.size() < 2) {
        GTEST_SKIP();
    }

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    const size_t scales_stride_rowwise = get_scale_tensor_dims(rows, cols, 1, 32)[3];
    const size_t scales_stride_colwise = get_scale_tensor_dims(rows, cols, 32, 1)[3];

    Tensor input("input", shape, itype);
    Tensor grad("grad", shape, itype);
    Tensor output("output", shape, otype, true, true, NVTE_MXFP8_1D_SCALING);
    Tensor output_dbias("output_dbias", std::vector<size_t>{ cols }, itype);

    // Single reference array serves both rowwise and colwise comparisons: the
    // pre-quantization float values are independent of the scaling direction.
    std::vector<float> ref_output(rows * cols);
    std::unique_ptr<InputType[]> ref_output_dbias = std::make_unique<InputType[]>(cols);

    fillCase<fp32>(&input, fill_case);
    fillUniform(&grad);

    Tensor workspace;
    switch (processing_method) {
        case ProcessingMethod::CAST_ONLY: {
            nvte_quantize(input.data(), output.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS: {
            nvte_quantize_dbias(grad.data(), output.data(), output_dbias.data(),
                                workspace.data(), 0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
            nvte_quantize_dbias(grad.data(), output.data(), output_dbias.data(),
                                workspace.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS_DACT: {
            auto nvte_quantize_dbias_dact = &nvte_quantize_dbias_dgelu;
            if (OP == &dsilu)       { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsilu; }
            else if (OP == &drelu)  { nvte_quantize_dbias_dact = &nvte_quantize_dbias_drelu; }
            else if (OP == &dqgelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dqgelu; }
            else if (OP == &dsrelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsrelu; }
            nvte_quantize_dbias_dact(grad.data(), input.data(), output.data(),
                                     output_dbias.data(), workspace.data(), 0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
            nvte_quantize_dbias_dact(grad.data(), input.data(), output.data(),
                                     output_dbias.data(), workspace.data(), 0);
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

    compute_ref<InputType>(processing_method, OP,
                           input.rowwise_cpu_dptr<InputType>(),
                           grad.rowwise_cpu_dptr<InputType>(),
                           ref_output.data(), ref_output_dbias.get(),
                           rows, cols);

    auto [atol, rtol] = getTolerances(otype);

    // Dequantize rowwise GPU output and compare to reference.
    {
        std::vector<float> dequant_rowwise(rows * cols);
        dequantize_mxfp8_rowwise(
            output.rowwise_cpu_dptr<OutputType>(),
            output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_rowwise.data(), rows, cols, scales_stride_rowwise);
        compareResults("output_rowwise", dequant_rowwise.data(), ref_output.data(),
                       rows * cols, atol, rtol);
    }

    // Dequantize colwise GPU output and compare to reference.
    {
        std::vector<float> dequant_colwise(rows * cols);
        dequantize_mxfp8_colwise(
            output.columnwise_cpu_dptr<OutputType>(),
            output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_colwise.data(), rows, cols, scales_stride_colwise);
        compareResults("output_colwise", dequant_colwise.data(), ref_output.data(),
                       rows * cols, atol, rtol);
    }

    if (processing_method == ProcessingMethod::CAST_DBIAS
        || processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
        auto [atol_dbias, rtol_dbias] = getTolerances(itype);
        if (itype == DType::kFloat32) {
            atol_dbias = 1e-4;
            rtol_dbias *= sqrt(static_cast<double>(rows));
        } else {
            rtol_dbias *= 4;
        }
        compareResults("output_dbias", output_dbias, ref_output_dbias.get(), true,
                       atol_dbias, rtol_dbias);
    }
}

std::vector<std::vector<size_t>> matrix_sizes = {
    {1, 16},
    {16, 48},
    {128, 128},
    {993, 512},
    {1024},
    {8, 32, 1024},
    {16, 8, 4, 512},
    {8192, 7168},
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

std::string test_name_generator(
    const testing::TestParamInfo<FusedCastMXFP8TestSuite::ParamType>& info) {
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
}

}  // namespace

// Test cases with only cast kernels
INSTANTIATE_TEST_SUITE_P(
    OperatorTest_FusedCastMXFP8_CastOnly,
    FusedCastMXFP8TestSuite,
    ::testing::Combine(
        ::testing::Values(ProcessingMethod::CAST_ONLY),
        ::testing::Values(ActivationType::Identity),
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(input_scenarios)),
    test_name_generator);

// Test cases with varying matrix shapes and block shapes
INSTANTIATE_TEST_SUITE_P(
    OperatorTest_FusedCastMXFP8_Sizes,
    FusedCastMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(processing_methods),
        ::testing::ValuesIn(Activation_types),
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        ::testing::Values(DType::kBFloat16),
        ::testing::Values(DType::kFloat8E4M3),
        ::testing::ValuesIn(input_scenarios)),
    test_name_generator);

// Test cases with varying dtypes
INSTANTIATE_TEST_SUITE_P(
    OperatorTest_FusedCastMXFP8_Dtypes,
    FusedCastMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(processing_methods),
        ::testing::ValuesIn(Activation_types),
        ::testing::Values(std::vector<size_t>{256, 384}),
        ::testing::Values(std::pair<size_t, size_t>{32, 32}),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::ValuesIn(input_scenarios)),
    test_name_generator);
