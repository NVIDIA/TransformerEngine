/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Compute the pre-quantization float values for the gated-SwiGLU kernel.
// Output layout: rows × output_cols, where output_cols = IS_DGATED ? 2*cols : cols.
// For IS_DGATED, the act part occupies [0, cols) and the gate part [cols, 2*cols).
template <typename IType>
void compute_ref(const IType* grad,
                 const IType* input,
                 float* ref_output,
                 const bool IS_DGATED,
                 const size_t rows,
                 const size_t cols)   // "half" cols (before gating)
{
    const size_t output_cols = IS_DGATED ? 2 * cols : cols;
    const size_t input_stride = cols * 2;  // input has shape [rows, 2*cols]

    #pragma omp parallel for proc_bind(spread) schedule(static) collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const float silu_elt = static_cast<float>(input[i * input_stride + j]);
            const float gate_elt = static_cast<float>(input[i * input_stride + cols + j]);

            if (IS_DGATED) {
                const float s = sigmoid(silu_elt);
                const float dact = silu_elt * s * (1 - s) + s;  // dsilu
                const float grad_elt = static_cast<float>(grad[i * cols + j]);

                float after_dsilu = dact * grad_elt * gate_elt;
                float after_dgate = silu(silu_elt) * grad_elt;

                // Match GPU: downcast to IType then back to float before quantization.
                after_dsilu = static_cast<float>(static_cast<IType>(after_dsilu));
                after_dgate = static_cast<float>(static_cast<IType>(after_dgate));

                ref_output[i * output_cols + j]        = after_dsilu;
                ref_output[i * output_cols + cols + j] = after_dgate;
            } else {
                float after_silu = silu(silu_elt) * gate_elt;
                after_silu = static_cast<float>(static_cast<IType>(after_silu));
                ref_output[i * cols + j] = after_silu;
            }
        }
    }
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
    DType itype = TypeInfo<IType>::dtype;
    DType otype = TypeInfo<OType>::dtype;

    const bool rowwise = (block_size_rows == 1) && (block_size_cols == 32);
    const bool colwise = (block_size_rows == 32) && (block_size_cols == 1);
    NVTE_CHECK(rowwise != colwise,
               "Expected either rowwise or columnwise scaling (rowwise=", rowwise,
               ", colwise=", colwise, ").");

    const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;
    const size_t scales_stride = get_scale_tensor_dims(rows, output_cols,
                                                       block_size_rows, block_size_cols)[3];

    Tensor grad("grad", std::vector<size_t>{ rows, cols }, itype);
    Tensor input("input", std::vector<size_t>{ rows, cols * 2 }, itype);
    Tensor output("output", std::vector<size_t>{ rows, output_cols }, otype,
                  rowwise, colwise, NVTE_MXFP8_1D_SCALING);

    std::vector<float> ref_output(rows * output_cols);

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

    compute_ref<IType>(grad.rowwise_cpu_dptr<IType>(),
                       input.rowwise_cpu_dptr<IType>(),
                       ref_output.data(), IS_DGATED, rows, cols);

    std::vector<float> dequant_output(rows * output_cols);
    if (rowwise) {
        dequantize_mxfp8_rowwise(
            output.rowwise_cpu_dptr<OType>(),
            output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_output.data(), rows, output_cols, scales_stride);
    } else {
        dequantize_mxfp8_colwise(
            output.columnwise_cpu_dptr<OType>(),
            output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_output.data(), rows, output_cols, scales_stride);
    }

    auto [atol, rtol] = getTolerances(otype);
    compareResults("output", dequant_output.data(), ref_output.data(),
                   rows * output_cols, atol, rtol);
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
    DType itype = TypeInfo<IType>::dtype;
    DType otype = TypeInfo<OType>::dtype;

    const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;
    const size_t scales_stride_rowwise = get_scale_tensor_dims(rows, output_cols, 1, 32)[3];
    const size_t scales_stride_colwise = get_scale_tensor_dims(rows, output_cols, 32, 1)[3];

    Tensor grad("grad", std::vector<size_t>{ rows, cols }, itype);
    Tensor input("input", std::vector<size_t>{ rows, cols * 2 }, itype);
    Tensor output("output", std::vector<size_t>{ rows, output_cols }, otype,
                  true, true, NVTE_MXFP8_1D_SCALING);

    // Pre-quantization float reference (scale-direction independent).
    std::vector<float> ref_output(rows * output_cols);

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

    compute_ref<IType>(grad.rowwise_cpu_dptr<IType>(),
                       input.rowwise_cpu_dptr<IType>(),
                       ref_output.data(), IS_DGATED, rows, cols);

    auto [atol, rtol] = getTolerances(otype);

    // Dequantize rowwise GPU output and compare to reference.
    {
        std::vector<float> dequant_rowwise(rows * output_cols);
        dequantize_mxfp8_rowwise(
            output.rowwise_cpu_dptr<OType>(),
            output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_rowwise.data(), rows, output_cols, scales_stride_rowwise);
        compareResults("output_rowwise", dequant_rowwise.data(), ref_output.data(),
                       rows * output_cols, atol, rtol);
    }

    // Dequantize colwise GPU output and compare to reference.
    {
        std::vector<float> dequant_colwise(rows * output_cols);
        dequantize_mxfp8_colwise(
            output.columnwise_cpu_dptr<OType>(),
            output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
            dequant_colwise.data(), rows, output_cols, scales_stride_colwise);
        compareResults("output_colwise", dequant_colwise.data(), ref_output.data(),
                       rows * output_cols, atol, rtol);
    }
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

}  // namespace

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
