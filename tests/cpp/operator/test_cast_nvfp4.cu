/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

enum ActivationType {
    Identity,
    GeLU,
    SiLU,
    ReLU,
    QGeLU,
    SReLU
};

template <typename InputType, typename OutputType>
void compute_ref(const bool rowwise,
                 const bool colwise,
                 float (*OP)(const float),
                 const InputType* input,
                 OutputType* output_rowwise_mxfp8,
                 OutputType* output_colwise_mxfp8,
                 fp4e2m1x2* output_rowwise_nvfp4,
                 fp4e2m1x2* output_colwise_nvfp4,
                 fp8e8m0* scales_rowwise_mxfp8,
                 fp8e8m0* scales_colwise_mxfp8,
                 fp8e4m3* scales_rowwise_nvfp4,
                 fp8e4m3* scales_colwise_nvfp4,
                 const float global_prev_amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride_rowwise,
                 const size_t scales_stride_colwise)
{
    const size_t tile_size_Y = 16;
    const size_t tile_size_X = 16;
    const size_t tiles_num_Y = (rows + tile_size_Y - 1) / tile_size_Y;
    const size_t tiles_num_X = (cols + tile_size_X - 1) / tile_size_X;

    // Compute a global encoding/decoding scaling factor for all S_dec_b 
    const float S_enc = 6.0f * 448.0f / global_prev_amax;

    #pragma omp parallel proc_bind(spread)
    {
        // Buffers to cache intermediate computations
        std::vector<float> cache_buffer(tile_size_Y * tile_size_X);

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
                    const int idx = i * cols + j;
                    const int cache_idx = (i - i_min) * tile_size_X + (j - j_min);

                    const float input_elt = static_cast<float>(input[idx]);
                    const float act_elt = OP(input_elt);


                    // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                    const float elt = static_cast<float>(static_cast<InputType>(act_elt));

                    cache_buffer[cache_idx] = elt;
                    // printf("Idx: %d Input: %f, Act: %f, Truncated: %f\n", idx, input_elt, act_elt, elt);
                    if (isinf(elt) || isnan(elt)) {
                        continue;
                    }
                }
                // printf("--------------------------------------------------------------------------\n\n");
            }

            if (rowwise) {
                for (size_t i = i_min; i < i_max; ++i) {
                    float block_amax = 0.0f;

                    for (size_t j = j_min; j < j_max; ++j) {
                        const int cache_idx = (i - i_min) * tile_size_X + (j - j_min);
                        block_amax = std::max(block_amax, std::abs(cache_buffer[cache_idx]));
                    }

                    // 2. Compute E4M3 scaling factor
                    // Compute per-block encoding/decoding scaling factor
                    const float S_dec_b = block_amax / 6.0f;

                    // Scale & Store per-block decoding scaling factor
                    const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(S_dec_b * S_enc);

                    // Compute "correct" per-block encoding scaling factor
                    const float S_enc_b_fp8 = S_enc / static_cast<float>(S_dec_b_fp8);

                    const int scale_idx = i * scales_stride_rowwise + tile_X;
                    scales_rowwise_nvfp4[scale_idx] = S_dec_b_fp8;
                    const float scale_reciprocal = S_enc_b_fp8;

                    // MXFP8 Scaling Type
                    // const fp8e8m0 biased_exponent = float_to_e8m0(block_amax * Quantized_Limits<OutputType>::max_reciprocal());
                    // const int scale_idx = i * scales_stride_rowwise + tile_X;
                    // scales_rowwise_mxfp8[scale_idx] = biased_exponent;
                    // const float scale_reciprocal = exp2f_rcp(biased_exponent);

                    // printf("Scale Reciprocal: %f\n", scale_reciprocal);
                    for (size_t j = j_min; j < j_max; j += 2) {
                        const int idx_pair = (i * cols + j) / 2;
                        // const int idx_x = i * cols + j;
                        // const int idx_y = i * cols + j + 1;
                        const int cache_idx_x = (i - i_min) * tile_size_X + (j     - j_min);
                        const int cache_idx_y = (i - i_min) * tile_size_X + (j + 1 - j_min);
                        const float cached_x = cache_buffer[cache_idx_x];
                        const float cached_y = cache_buffer[cache_idx_y];
                        const float scaled_elt_x = cached_x * scale_reciprocal;
                        const float scaled_elt_y = cached_y * scale_reciprocal;
                        const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                        fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);

                        const __half2_raw raw_truncated_to_fp4e2m1_pair =
                            __nv_cvt_fp4x2_to_halfraw2(*reinterpret_cast<__nv_fp4x2_storage_t*>(&casted_to_e2m1_pair),
                                                        __NV_E2M1);

                        const __half2 truncated_to_fp4e2m1_pair(raw_truncated_to_fp4e2m1_pair);
                        const float truncated_to_fp4e2m1_x = static_cast<float>(truncated_to_fp4e2m1_pair.x);
                        const float truncated_to_fp4e2m1_y = static_cast<float>(truncated_to_fp4e2m1_pair.y);

                        // output_rowwise_nvfp4[idx_pair] = *(reinterpret_cast<fp4e2m1*>(&casted_to_e2m1_pair));
                        output_rowwise_nvfp4[idx_pair] = casted_to_e2m1_pair;

                        // output_rowwise_nvfp4[idx] = static_cast<fp4e2m1>(cache_buffer[cache_idx] * scale_reciprocal);
                        // printf("Idx: %d Cached: %f, Scaled: %f, Truncated to E2M1: %f\n", idx_x, cached_x, scaled_elt_x, truncated_to_fp4e2m1_x);
                        // printf("Idx: %d Cached: %f, Scaled: %f, Truncated to E2M1: %f\n", idx_y, cached_y, scaled_elt_y, truncated_to_fp4e2m1_y);
                    }
                    // printf("--------------------------------------------------------------------------\n\n");
                }
            }
            // if (colwise) {
            //     for (size_t j = j_min; j < j_max; ++j) {
            //         float block_amax = 0.0f;

            //         for (size_t i = i_min; i < i_max; ++i) {
            //             const int cache_idx = (i - i_min) * tile_size_X + (j - j_min);
            //             block_amax = std::max(block_amax, std::abs(cache_buffer[cache_idx]));
            //         }

            //         const fp8e8m0 biased_exponent = float_to_e8m0(block_amax * Quantized_Limits<OutputType>::max_reciprocal());
            //         const int scale_idx = tile_Y * scales_stride_colwise + j;
            //         scales_colwise_e8m0[scale_idx] = biased_exponent;
            //         const float scale_reciprocal = exp2f_rcp(biased_exponent);

            //         for (size_t i = i_min; i < i_max; ++i) {
            //             const int idx = i * cols + j;
            //             const int cache_idx = (i - i_min) * tile_size_X + (j - j_min);
            //             output_colwise[idx] = static_cast<OutputType>(cache_buffer[cache_idx] * scale_reciprocal);
            //         }
            //     }
            // }
        }
    }
}


void compareResults_nvfp4(const std::string &name, const Tensor &test,
                          const void *ref, const bool rowwise,
                          double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true) {

  const std::string direction = rowwise ? "rowwise" : "colwise";

  if (if_on_gpus) test.to_cpu();
  const auto& shape = rowwise ? test.rowwise_shape() : test.columnwise_shape();
  const size_t N = product(shape);
  const fp4e2m1 *test_data = rowwise
                             ? test.rowwise_cpu_dptr<fp4e2m1>()
                             : test.columnwise_cpu_dptr<fp4e2m1>();
  const fp4e2m1 *ref_data = reinterpret_cast<const fp4e2m1*>(ref);
  for (size_t i = 0; i < N; i += 2) {

    const __nv_fp4x2_storage_t* test_raw_storage = reinterpret_cast<const __nv_fp4x2_storage_t*>(&test_data[i/2]);
    const __nv_fp4x2_storage_t* ref_raw_storage = reinterpret_cast<const __nv_fp4x2_storage_t*>(&ref_data[i/2]);

    const __half2_raw test_data_pair_raw = __nv_cvt_fp4x2_to_halfraw2(*test_raw_storage, __NV_E2M1);
    const __half2_raw ref_data_pair_raw = __nv_cvt_fp4x2_to_halfraw2(*ref_raw_storage, __NV_E2M1);

    const __half2 test_data_pair(test_data_pair_raw);
    const __half2 ref_data_pair(ref_data_pair_raw);

    for (int k = 0; k < 2; ++k) {
        const double t = static_cast<double>(k == 0 ? test_data_pair.x : test_data_pair.y);
        const double r = static_cast<double>(k == 0 ? ref_data_pair.x : ref_data_pair.y);

        bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
        /* For Float32 the floating point comparison is enough to error out */
        bool assertion = mismatch && test.dtype() == DType::kFloat32;
        if (mismatch && !assertion) {
          /* Check if it is just a failure of round to nearest choosing different
             side of the real value */
          const double mean = (t + r) / 2;
          const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
          const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
          const double cast_mean_p = static_cast<double>(static_cast<fp4e2m1>(mean_p));
          const double cast_mean_m = static_cast<double>(static_cast<fp4e2m1>(mean_m));
          assertion = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
        }
        // printf("%lu GPU: %f CPU: %f\n", i + k, t, r);
        if (assertion) {
            ASSERT_FALSE(assertion) << "Error in tensor " << name << " in "
                                    << direction << " direction." << std::endl
                                    << "Mismatch at place " 
                                    << " (" << std::to_string(i + k) << "): "
                                    << t << " vs " << r;
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

template <typename InputType, typename OutputType>
void performTest_x1(float (*OP)(const float),
                    const std::vector<size_t>& shape,
                    const bool rowwise,
                    const bool colwise,
                    InputsFillCase fill_case) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    // DType otype = TypeInfo<OutputType>::dtype;
    
    DType otype = TypeInfo<fp4e2m1>::dtype;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    const size_t block_size_rows = rowwise ? 1 : 16;
    const size_t block_size_cols = colwise ? 1 : 16;

    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, block_size_rows, block_size_cols);

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    Tensor input("input", shape, itype);

    std::vector<size_t> shape_nvfp4 = shape;
    // shape.back()

    Tensor output("output", shape, otype, rowwise, colwise, NVTE_FWD_NVFP4_BWD_MXFP8_SCALING);

    const float global_prev_amax = 6.0f * 448.0f;

    std::unique_ptr<OutputType[]> ref_output_mxfp8 = std::make_unique<OutputType[]>(rows * cols);
    std::unique_ptr<fp4e2m1x2[]> ref_output_nvfp4 = std::make_unique<fp4e2m1x2[]>(rows * cols / 2);
    std::unique_ptr<fp8e8m0[]> ref_scales_mxfp8 = std::make_unique<fp8e8m0[]>(blocks_Y * blocks_X);
    std::unique_ptr<fp8e4m3[]> ref_scales_nvfp4 = std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);

    fillCase<EncodingType>(&input, fill_case);

    auto nvte_quantize_operation = &nvte_quantize;
    if (OP == &gelu)       { nvte_quantize_operation = &nvte_gelu; }
    else if (OP == &silu)  { nvte_quantize_operation = &nvte_silu; }
    else if (OP == &relu)  { nvte_quantize_operation = &nvte_relu; }
    else if (OP == &qgelu) { nvte_quantize_operation = &nvte_qgelu; }
    else if (OP == &srelu) { nvte_quantize_operation = &nvte_srelu; }

    nvte_quantize_operation(input.data(), output.data(), 0);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    compute_ref<InputType, OutputType>(rowwise,
                                       colwise,
                                       OP,
                                       input.rowwise_cpu_dptr<InputType>(),
                                       ref_output_mxfp8.get(),
                                       ref_output_mxfp8.get(),
                                       ref_output_nvfp4.get(),
                                       ref_output_nvfp4.get(),
                                       ref_scales_mxfp8.get(),
                                       ref_scales_mxfp8.get(),
                                       ref_scales_nvfp4.get(),
                                       ref_scales_nvfp4.get(),
                                       global_prev_amax,
                                       rows,
                                       cols,
                                       scales_stride,
                                       scales_stride);

    // auto [atol, rtol] = getTolerances(otype);
    const double atol = 0.125;
    const double rtol = 0.5;
    compareResults_nvfp4("output", output, ref_output_nvfp4.get(), rowwise, atol, rtol);

    const fp8e4m3* const gpu_scales_ptr = rowwise
                                          ? output.rowwise_cpu_scale_inv_ptr<fp8e4m3>()
                                          : output.columnwise_cpu_scale_inv_ptr<fp8e4m3>();

    size_t scale_mismatches_num = 0;
    compare_scaling_factors("scales", gpu_scales_ptr, ref_scales_nvfp4.get(),
                            unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
                            scale_mismatches_num);
}

/**
 * Scaling along both dimensions (rows and columns)
 * Produces two sets of scaled output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *      AND
 * 2) Scaled columns + column-wise scaling factors
 */
// template <typename InputType, typename OutputType>
// void performTest_x2(float (*OP)(const float),
//                     const std::vector<size_t>& shape,
//                     const size_t block_size_rows,
//                     const size_t block_size_cols,
//                     InputsFillCase fill_case) {
//     using namespace test;
//     using EncodingType = fp32;
//     DType itype = TypeInfo<InputType>::dtype;
//     DType otype = TypeInfo<OutputType>::dtype;

//     const size_t rows = first_dimension(shape);
//     const size_t cols = last_dimension(shape);

//     const std::array<size_t,4> scale_dims_rowwise = get_scale_tensor_dims(rows, cols, 1, 32);
//     const std::array<size_t,4> scale_dims_colwise = get_scale_tensor_dims(rows, cols, 32, 1);

//     const size_t unpadded_blocks_Y_rowwise = scale_dims_rowwise[0];
//     const size_t unpadded_blocks_X_rowwise = scale_dims_rowwise[1];
//     const size_t blocks_Y_rowwise = scale_dims_rowwise[2];
//     const size_t blocks_X_rowwise = scale_dims_rowwise[3];
//     const size_t scales_stride_rowwise = blocks_X_rowwise;

//     const size_t unpadded_blocks_Y_colwise = scale_dims_colwise[0];
//     const size_t unpadded_blocks_X_colwise = scale_dims_colwise[1];
//     const size_t blocks_Y_colwise = scale_dims_colwise[2];
//     const size_t blocks_X_colwise = scale_dims_colwise[3];
//     const size_t scales_stride_colwise = blocks_X_colwise;

//     Tensor input("input", shape, itype);
//     Tensor output("output", shape, otype, true, true, NVTE_FWD_NVFP4_BWD_MXFP8_SCALING);
    
//     const float global_prev_amax = 6.0f * 448.0f;

//     std::unique_ptr<OutputType[]> ref_output_c_rowwise = std::make_unique<OutputType[]>(rows * cols);
//     std::unique_ptr<OutputType[]> ref_output_c_colwise = std::make_unique<OutputType[]>(rows * cols);
//     std::unique_ptr<fp8e8m0[]> ref_scales_rowwise = std::make_unique<fp8e8m0[]>(blocks_Y_rowwise * blocks_X_rowwise);
//     std::unique_ptr<fp8e8m0[]> ref_scales_colwise = std::make_unique<fp8e8m0[]>(blocks_Y_colwise * blocks_X_colwise);

//     fillCase<EncodingType>(&input, fill_case);

//     auto nvte_quantize_operation = &nvte_quantize;
//     if (OP == &gelu)       { nvte_quantize_operation = &nvte_gelu; }
//     else if (OP == &silu)  { nvte_quantize_operation = &nvte_silu; }
//     else if (OP == &relu)  { nvte_quantize_operation = &nvte_relu; }
//     else if (OP == &qgelu) { nvte_quantize_operation = &nvte_qgelu; }
//     else if (OP == &srelu) { nvte_quantize_operation = &nvte_srelu; }

//     nvte_quantize_operation(input.data(), output.data(), 0);

//     cudaDeviceSynchronize();
//     auto err = cudaGetLastError();
//     ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

//     compute_ref<InputType, OutputType>(true,
//                                        true,
//                                        OP,
//                                        input.rowwise_cpu_dptr<InputType>(),
//                                        ref_output_c_rowwise.get(),
//                                        ref_output_c_colwise.get(),
//                                        ref_scales_rowwise.get(),
//                                        ref_scales_colwise.get(),
//                                        global_prev_amax,
//                                        rows,
//                                        cols,
//                                        scales_stride_rowwise,
//                                        scales_stride_colwise);

//     auto [atol, rtol] = getTolerances(otype);
//     compareResults("output_c_rowwise", output, ref_output_c_rowwise.get(), true, atol, rtol);
//     compareResults("output_c_colwise", output, ref_output_c_colwise.get(), false, atol, rtol);
//     compare_e8m0_scaling_factors("scales_rowwise", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
//                                  ref_scales_rowwise.get(), unpadded_blocks_Y_rowwise,
//                                  unpadded_blocks_X_rowwise, scales_stride_rowwise);
//     compare_e8m0_scaling_factors("scales_colwise", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
//                                  ref_scales_colwise.get(), unpadded_blocks_Y_colwise,
//                                  unpadded_blocks_X_colwise, scales_stride_colwise);
// }

std::vector<std::vector<size_t>> matrix_sizes = {
    // {1, 32},
    // {16, 48},    
    // {65, 96},
    // {128, 128},
    // {256, 256},
    // {993, 512},
    // {256, 65536},
    // {2048, 6144},
    // {16384, 128},
    // {32768, 160},
    // {4096, 1632},
    // {1024},
    // {8, 32, 1024},
    // {16, 8, 4, 512},
    {1024, 16384},
    {4096, 13312},  
};

std::vector<std::pair<size_t, size_t>> block_sizes = {
    {1, 16},
    // {16, 1},
    // {16, 32},
    // {32, 16},
};

std::vector<InputsFillCase> input_scenarios = {
    InputsFillCase::uniform,
    // InputsFillCase::zeros,
    // InputsFillCase::zero_to_minNorm,
    // InputsFillCase::minNorm_to_maxNorm,
    // InputsFillCase::maxNorm_to_inf
};

// Only GeLU activation tests are supported
std::vector<ActivationType> Activation_types = {
    ActivationType::Identity,
    // ActivationType::GeLU,
    // ActivationType::SiLU,
    // ActivationType::ReLU,
    // ActivationType::QGeLU,
    // ActivationType::SReLU,
};

}  // namespace

class FusedCastNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<ActivationType,
                std::vector<size_t>,
                std::pair<size_t, size_t>,
                transformer_engine::DType,
                transformer_engine::DType,
                InputsFillCase>> {};

TEST_P(FusedCastNVFP4TestSuite, TestFusedCastNVFP4) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const ActivationType Act_type = std::get<0>(GetParam());
    const auto tensor_dims = std::get<1>(GetParam());
    const auto block_size = std::get<2>(GetParam());
    const DType input_type = std::get<3>(GetParam());
    const DType output_type = std::get<4>(GetParam());
    const InputsFillCase fill_case = std::get<5>(GetParam());

    const bool rowwise = block_size.second != 1;
    const bool colwise = block_size.first != 1;

    // Skip tests with colwise scaling, if the input tensor is 1D  
    if (tensor_dims.size() < 2 && colwise) {
        GTEST_SKIP();
    }

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
            if (rowwise && colwise) {
                // performTest_x2<InputType, OutputType>(OP, tensor_dims, block_size.first, block_size.second, fill_case);
            } else {
                performTest_x1<InputType, OutputType>(OP, tensor_dims, rowwise, colwise, fill_case);
            }
        );
    );
}

std::string to_string(const ActivationType Act_type) {
    switch (Act_type) {
        case ActivationType::Identity:  return "CAST_ONLY";
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
    FusedCastNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),
        ::testing::ValuesIn(matrix_sizes),
        ::testing::ValuesIn(block_sizes),
        // ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        // ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2),
        ::testing::Values(DType::kBFloat16),
        ::testing::Values(DType::kFloat8E4M3),
        ::testing::ValuesIn(input_scenarios)),
    [](const testing::TestParamInfo<FusedCastNVFP4TestSuite::ParamType>& info) {
        std::string name = to_string(std::get<0>(info.param));
      const auto& shape = std::get<1>(info.param);
      for ( const auto& s: shape) {
        name += "X" + std::to_string(s);
      }
      name += "X" + std::to_string(std::get<2>(info.param).first) +
              "X" + std::to_string(std::get<2>(info.param).second) +
              "X" + test::typeName(std::get<3>(info.param)) +
              "X" + test::typeName(std::get<4>(info.param)) +
              "X" + test::caseName(std::get<5>(info.param));
        return name;
    });
