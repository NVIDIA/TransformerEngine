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

enum ActivationKind {
    Identity,
    GeLU,
    SiLU,
    ReLU,
    QGeLU,
    SReLU
};

enum ShapeRepresentation {
  SAME_BOTH_DIMS    = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM  = 2,
  VARYING_BOTH_DIMS = 3
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
                    // if (processing_method == ProcessingMethod::CAST_DBIAS) {
                    //     // grad is the input
                    //     elt = static_cast<float>(grad[idx]);
                    // }
                    if (processing_method != ProcessingMethod::CAST_ONLY
                        && processing_method != ProcessingMethod::CAST_DBIAS) {
                        elt = OP(elt);
                    }
                    // if (processing_method == ProcessingMethod::CAST_DACT ||
                    //     processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
                    //     elt *= static_cast<float>(grad[idx]);
                    // }
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
    // for (size_t j = 0; j < cols; ++j) {
    //     output_dbias[j] = static_cast<InputType>(output_dbias_fp32[j]);
    // }
}

template <typename T>
void compare_scaled_elts(const std::string &name,
                         const T* ref_data,
                         const T* test_data,
                         const size_t rows,
                         const size_t cols,
                         const bool rowwise,
                         const size_t tolerable_mismatches_limit = 0,
                         const double atol = 1e-5,
                         const double rtol = 1e-8) {
    size_t mismatches_num = 0;
    int first_mismatch_idx = -1;

    for (size_t i = 0; i < rows * cols; ++i) {
        double t = static_cast<double>(test_data[i]);
        double r = static_cast<double>(ref_data[i]);
        bool mismatch = fabs(t - r) > atol && (r == 0 || fabs((t - r) / r) > rtol);
        /* For Float32 the floating point comparison is enough to error out */
        bool assertion = false;
        if (mismatch && !assertion) {
            /* Check if it is just a failure of round to nearest choosing different
                side of the real value */
            const double mean = (t + r) / 2;
            const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
            const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
            const double cast_mean_p = static_cast<double>(static_cast<T>(mean_p));
            const double cast_mean_m = static_cast<double>(static_cast<T>(mean_m));
            assertion = !(cast_mean_m == std::min(t,r) && cast_mean_p == std::max(t,r));
        }
        std::string direction = rowwise ? "rowwise" : "columnwise";
        if (assertion) {
            mismatches_num++;
            if (first_mismatch_idx == -1) {
                first_mismatch_idx = i;
            }
        }
        if (mismatches_num > tolerable_mismatches_limit) {
            const double first_mismatch_t = static_cast<double>(test_data[first_mismatch_idx]);
            const double first_mismatch_r = static_cast<double>(ref_data[first_mismatch_idx]);

            GTEST_FAIL() << mismatches_num << " mismatche(s) which is more than tolerable mismatch limit of "
                        << tolerable_mismatches_limit << "." << std::endl
                        << "Error in tensor " << name << " in "
                        << direction << " direction." << std::endl
                        << "First mismatch at place " << first_mismatch_idx
                        << " (" << std::to_string(first_mismatch_idx) << "): "
                        << first_mismatch_t << " vs " << first_mismatch_r;
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
void performTest_x1(const ProcessingMethod processing_method,
                    float (*OP)(const float),
                    const ShapeRepresentation shape_rep,
                    const size_t num_tensors,
                    const std::vector<size_t>& logical_shape_vec,
                    const std::vector<size_t>& first_dims_h,
                    const std::vector<size_t>& last_dims_h,
                    const bool rowwise,
                    const bool colwise) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t rows = logical_shape_vec[0];
    const size_t cols = logical_shape_vec[1];

    std::vector<size_t> scales_rowwise_shape = {rows, cols / 32};
    std::vector<size_t> scales_colwise_shape = {rows / 32, cols};

    const size_t scales_stride_rowwise = scales_rowwise_shape[1];
    const size_t scales_stride_colwise = scales_colwise_shape[1];

    const size_t elts_num = rows * cols;
    const size_t sfs_num = (rows * cols) / 32;

    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-2.0, 1.0);

    std::vector<InputType> in_data(elts_num);

    std::vector<OutputType> out_data_rowwise_h(rowwise ? elts_num : 0);
    std::vector<OutputType> out_data_colwise_h(colwise ? elts_num : 0);
    std::vector<fp8e8m0> out_scales_rowwise_h(rowwise ? sfs_num : 0);
    std::vector<fp8e8m0> out_scales_colwise_h(colwise ? sfs_num : 0);
    
    std::vector<OutputType> out_data_rowwise_ref(rowwise ? elts_num : 0);
    std::vector<OutputType> out_data_colwise_ref(colwise ? elts_num : 0);
    std::vector<fp8e8m0> out_scales_rowwise_ref(rowwise ? sfs_num : 0);
    std::vector<fp8e8m0> out_scales_colwise_ref(colwise ? sfs_num : 0);

    std::vector<size_t> offsets_h(num_tensors + 1);
    for (size_t t = 0; t < num_tensors + 1; ++t) {
        if (t == 0) {
            offsets_h[t] = 0;
        } else {
            offsets_h[t] = offsets_h[t-1] + (first_dims_h[t-1] * last_dims_h[t-1]);
        }
    }

    for (size_t i = 0; i < elts_num; ++i) {
        const float val = dis(gen);
        in_data[i] = static_cast<InputType>(val);
    }

    if (rowwise) {
        for (size_t i = 0; i < elts_num; ++i) {
            out_data_rowwise_h[i] = static_cast<OutputType>(0.0f);
            out_data_rowwise_ref[i] = static_cast<OutputType>(0.0f);
        }
        for (size_t i = 0; i < sfs_num; ++i) {
            out_scales_rowwise_h[i] = static_cast<fp8e8m0>(0.0f);
            out_scales_rowwise_ref[i] = static_cast<fp8e8m0>(0.0f);
        }
    }
    if (colwise) {
        for (size_t i = 0; i < elts_num; ++i) {
            out_data_colwise_h[i] = static_cast<OutputType>(0.0f);
            out_data_colwise_ref[i] = static_cast<OutputType>(0.0f);
        }
        for (size_t i = 0; i < sfs_num; ++i) {
            out_scales_colwise_h[i] = static_cast<fp8e8m0>(0.0f);
            out_scales_colwise_ref[i] = static_cast<fp8e8m0>(0.0f);
        }
    }

    const size_t in_data_size = elts_num * sizeof(InputType);
    const size_t out_data_size = elts_num * sizeof(OutputType);
    const size_t out_scales_size = sfs_num * sizeof(fp8e8m0);

    const size_t first_dims_size = num_tensors * sizeof(size_t);
    const size_t last_dims_size = num_tensors * sizeof(size_t);
    const size_t offsets_size = (num_tensors + 1) * sizeof(size_t);

    InputType* in_data_d;
    OutputType* out_data_rowwise_d;
    OutputType* out_data_colwise_d;
    fp8e8m0* out_scales_rowwise_d;
    fp8e8m0* out_scales_colwise_d;
    size_t* first_dims_d;
    size_t* last_dims_d;
    size_t* offsets_d;

    cudaMalloc((void**)&in_data_d, in_data_size);
    cudaMalloc((void**)&first_dims_d, first_dims_size);
    cudaMalloc((void**)&last_dims_d, last_dims_size);
    cudaMalloc((void**)&offsets_d, offsets_size);
    
    cudaMemcpy(in_data_d, in_data.data(), in_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(first_dims_d, first_dims_h.data(), first_dims_size, cudaMemcpyHostToDevice);
    cudaMemcpy(last_dims_d, last_dims_h.data(), last_dims_size, cudaMemcpyHostToDevice);
    cudaMemcpy(offsets_d, offsets_h.data(), offsets_size, cudaMemcpyHostToDevice);

    NVTEShape logical_shape_ = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());

    NVTEShape first_dims_shape_;
    NVTEShape last_dims_shape_;
    NVTEShape offsets_shape_;

    first_dims_shape_.ndim = 1;
    last_dims_shape_.ndim = 1;
    offsets_shape_.ndim = 1;

    first_dims_shape_.data[0] = num_tensors;
    last_dims_shape_.data[0] = num_tensors;
    offsets_shape_.data[0] = num_tensors + 1;

    NVTEGroupedTensor in_group_tensor = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape_);
    NVTEGroupedTensor out_group_tensor = nvte_create_grouped_tensor(NVTE_MXFP8_1D_SCALING, num_tensors, logical_shape_);

    NVTEBasicTensor in_data_tensor = {in_data_d, static_cast<NVTEDType>(itype), logical_shape_};
    nvte_set_grouped_tensor_param(&in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData, &in_data_tensor);

    if ((shape_rep == VARYING_FIRST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
        NVTEBasicTensor first_dims_tensor = {first_dims_d, kNVTEInt64, first_dims_shape_};
        nvte_set_grouped_tensor_param(&in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims, &first_dims_tensor);
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims, &first_dims_tensor);
    }

    if ((shape_rep == VARYING_LAST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
        NVTEBasicTensor last_dims_tensor = {last_dims_d, kNVTEInt64, last_dims_shape_};
        nvte_set_grouped_tensor_param(&in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims, &last_dims_tensor);
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims, &last_dims_tensor);
    }

    if (shape_rep != SAME_BOTH_DIMS) {
        NVTEBasicTensor offsets_tensor = {offsets_d, kNVTEInt64, offsets_shape_};
        nvte_set_grouped_tensor_param(&in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets, &offsets_tensor);
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets, &offsets_tensor);
    }

    if (rowwise) {
        cudaMalloc((void**)&out_data_rowwise_d, out_data_size);
        cudaMalloc((void**)&out_scales_rowwise_d, out_scales_size);
        cudaMemset(out_data_rowwise_d, 0, out_data_size);
        cudaMemset(out_scales_rowwise_d, 0, out_scales_size);
        NVTEBasicTensor out_data_rowwise_tensor = {out_data_rowwise_d, static_cast<NVTEDType>(otype), logical_shape_};
        NVTEShape scales_rowwise_shape_ = nvte_make_shape(scales_rowwise_shape.data(), scales_rowwise_shape.size());
        NVTEBasicTensor out_scales_rowwise_tensor = {out_scales_rowwise_d, NVTEDType::kNVTEFloat8E8M0, scales_rowwise_shape_};
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData, &out_data_rowwise_tensor);
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseScaleInv, &out_scales_rowwise_tensor);
    }

    if (colwise) {
        cudaMalloc((void**)&out_data_colwise_d, out_data_size);
        cudaMalloc((void**)&out_scales_colwise_d, out_scales_size);
        cudaMemset(out_data_colwise_d, 0, out_data_size);
        cudaMemset(out_scales_colwise_d, 0, out_scales_size);
        NVTEBasicTensor out_data_colwise_tensor = {out_data_colwise_d, static_cast<NVTEDType>(otype), logical_shape_};
        NVTEShape scales_colwise_shape_ = nvte_make_shape(scales_colwise_shape.data(), scales_colwise_shape.size());
        NVTEBasicTensor out_scales_colwise_tensor = {out_scales_colwise_d, NVTEDType::kNVTEFloat8E8M0, scales_colwise_shape_};
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedColumnwiseData, &out_data_colwise_tensor);
        nvte_set_grouped_tensor_param(&out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedColumnwiseScaleInv, &out_scales_colwise_tensor);
    }

    // Reference (CPU)
    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t M = first_dims_h[t];
        const size_t K = last_dims_h[t];

        const size_t scales_stride_rowwise = K / 32;
        const size_t scales_stride_colwise = K;
        const size_t data_offset = offsets_h[t];
        const size_t sfs_offset = data_offset / 32;

        const InputType* const in_ptr = in_data.data() + data_offset;
        OutputType* const out_data_rowwise_ptr = out_data_rowwise_ref.data() + data_offset;
        OutputType* const out_data_colwise_ptr = out_data_colwise_ref.data() + data_offset;
        fp8e8m0* const out_scales_rowwise_ptr = out_scales_rowwise_ref.data() + sfs_offset;
        fp8e8m0* const out_scales_colwise_ptr = out_scales_colwise_ref.data() + sfs_offset;
    
        compute_ref<InputType, OutputType>(
            processing_method, OP, rowwise, colwise, in_ptr, /*grad=*/ nullptr,
            out_data_rowwise_ptr, out_data_colwise_ptr,
            out_scales_rowwise_ptr, out_scales_colwise_ptr,
            /*output_dbias=*/ nullptr, M, K,
            scales_stride_rowwise,
            scales_stride_colwise);
    }

    // GPU
    nvte_quantize_grouped(in_group_tensor, out_group_tensor, 0);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto [atol, rtol] = getTolerances(otype);
    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 0.0;
    const double rel_tolerable_mismatches_limit = 0.0;

    if (rowwise) {
        cudaMemcpy(out_data_rowwise_h.data(), out_data_rowwise_d, out_data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_scales_rowwise_h.data(), out_scales_rowwise_d, out_scales_size, cudaMemcpyDeviceToHost);

        size_t mismatches_scales = 0;
        compare_scaling_factors("rowwise_scales", out_scales_rowwise_h.data(), out_scales_rowwise_ref.data(),
                                scales_rowwise_shape[0], scales_rowwise_shape[1], scales_stride_rowwise,
                                mismatches_scales,
                                scale_diff_abs_tolerance,
                                abs_tolerable_mismatches_limit,
                                rel_tolerable_mismatches_limit);

        const size_t mismatches_elts = 32 * mismatches_scales;

        compare_scaled_elts<OutputType>("rowwise_output", out_data_rowwise_ref.data(),
                                        out_data_rowwise_h.data(), rows, cols, true, mismatches_elts);
    }

    if (colwise) {
        cudaMemcpy(out_data_colwise_h.data(), out_data_colwise_d, out_data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_scales_colwise_h.data(), out_scales_colwise_d, out_scales_size, cudaMemcpyDeviceToHost);

        size_t mismatches_scales = 0;
        compare_scaling_factors("colwise_scales", out_scales_colwise_h.data(), out_scales_colwise_ref.data(),
                                scales_colwise_shape[0], scales_colwise_shape[1], scales_stride_colwise,
                                mismatches_scales,
                                scale_diff_abs_tolerance,
                                abs_tolerable_mismatches_limit,
                                rel_tolerable_mismatches_limit);

        const size_t mismatches_elts = 32 * mismatches_scales;

        compare_scaled_elts<OutputType>("colwise_output", out_data_colwise_ref.data(),
                                        out_data_colwise_h.data(), rows, cols, false, mismatches_elts);
    }

    cudaFree(in_data_d);
    cudaFree(first_dims_d);
    cudaFree(last_dims_d);
    cudaFree(offsets_d);
    if (rowwise) {
        cudaFree(out_data_rowwise_d);
        cudaFree(out_scales_rowwise_d);
    }
    if (colwise) {
        cudaFree(out_data_colwise_d);
        cudaFree(out_scales_colwise_d);
    }
}

std::vector<ProcessingMethod> processing_methods = {
    ProcessingMethod::CAST_ONLY,
    // ProcessingMethod::CAST_DBIAS,
    // ProcessingMethod::CAST_DBIAS_DACT,
    // ProcessingMethod::CAST_DACT,
    // ProcessingMethod::CAST_ACT,
};

// Only GeLU activation tests are supported
std::vector<ActivationKind> activation_kinds = {
    ActivationKind::Identity,
    // ActivationKind::GeLU,
    // ActivationKind::SiLU,
    // ActivationKind::ReLU,
    // ActivationKind::QGeLU,
    // ActivationKind::SReLU,
};

enum ScalingDirection {
    ROWWISE = 0,
    COLWISE = 1,
    BOTH = 2
};

std::vector<ScalingDirection> scaling_directions = {
    ScalingDirection::ROWWISE,
    // ScalingDirection::COLWISE,
    // ScalingDirection::BOTH,
};

// {shape_representation, num_tensors, [logical_shape_M, logical_shape_K], [M_i], [K_i]}
std::vector<std::vector<size_t>> input_config = {
    {SAME_BOTH_DIMS,        1,      128,128},
    {SAME_BOTH_DIMS,        2,      256,128},
    {VARYING_FIRST_DIM,     2,      512,128,                    128,512-128},
    {VARYING_BOTH_DIMS,     2,      1,(128*128)+(256*256),      128,256,        128,256},
};

}  // namespace

class GroupedFusedCastMXFP8TestSuite : public ::testing::TestWithParam
    <std::tuple<ProcessingMethod,
                ActivationKind,
                ScalingDirection,
                std::vector<size_t>,        // Config
                transformer_engine::DType,  // InputType
                transformer_engine::DType   // OutputType
                >> {};

TEST_P(GroupedFusedCastMXFP8TestSuite, Test) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;
   
    const ProcessingMethod processing_method = std::get<0>(GetParam());
    const ActivationKind activation = std::get<1>(GetParam());
    const ScalingDirection scaling_direction = std::get<2>(GetParam());
    const std::vector<size_t> input_config = std::get<3>(GetParam());

    const ShapeRepresentation shape_rep = static_cast<ShapeRepresentation>(input_config[0]);
    const size_t num_tensors = input_config[1];
    const std::vector<size_t> logical_shape = {input_config[2], input_config[3]};
    std::vector<size_t> first_dims(num_tensors);
    std::vector<size_t> last_dims(num_tensors);
    for (size_t t = 0; t < num_tensors; ++t) {
        switch (shape_rep) {
            case SAME_BOTH_DIMS: {
                first_dims[t] = logical_shape[0] / num_tensors;
                last_dims[t] = logical_shape[1];
                break;
            }
            case VARYING_FIRST_DIM: {
                first_dims[t] = input_config[t + 4];
                last_dims[t] = logical_shape[1];
                break;
            }
            case VARYING_LAST_DIM: {
                first_dims[t] = logical_shape[0];
                last_dims[t] = input_config[t + (4 + num_tensors)];
                break;
            }
            case VARYING_BOTH_DIMS: {
                first_dims[t] = input_config[t + 4];
                last_dims[t] = input_config[t + (4 + num_tensors)];
                break;
            }
        }
        // Skips tests if tensor dims are not multiples of 128
        if ((first_dims[t] % 128 != 0) || (last_dims[t] % 128 != 0)) {
            GTEST_SKIP();
        }
    }

    // Skips non Act tests if the Activation type is not an identity
    if ((processing_method == ProcessingMethod::CAST_ONLY || processing_method == ProcessingMethod::CAST_DBIAS)
        && activation != ActivationKind::Identity) {
        GTEST_SKIP();
    }
    // Skips Act tests if the Activation is an identity
    if ((processing_method == ProcessingMethod::CAST_DBIAS_DACT
        || processing_method == ProcessingMethod::CAST_DACT
        || processing_method == ProcessingMethod::CAST_ACT) && (activation == ActivationKind::Identity)) {
        GTEST_SKIP();
    }

    bool rowwise = false;
    bool colwise = false;
    switch (scaling_direction) {
        case ScalingDirection::ROWWISE: rowwise = true; break;
        case ScalingDirection::COLWISE: colwise = true; break;
        case ScalingDirection::BOTH:    rowwise = true; colwise = true; break;
    }

    auto OP = &identity;
    performTest_x1<bf16, fp8e4m3>(processing_method, OP, shape_rep, num_tensors, logical_shape,
                                  first_dims, last_dims, rowwise, colwise);

    // if (processing_method == ProcessingMethod::CAST_ACT) {
    //     // Forward activations
    //     auto OP = &identity;
    //     switch (activation) {
    //         case ActivationKind::GeLU: OP = &gelu; break;
    //         case ActivationKind::SiLU: OP = &silu; break;
    //         case ActivationKind::ReLU: OP = &relu; break;
    //         case ActivationKind::QGeLU: OP = &qgelu; break;
    //         case ActivationKind::SReLU: OP = &srelu; break;
    //     }

    //     TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
    //         TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
    //             if (scaling_direction == ScalingDirection::BOTH) {
    //                 performTest_x2<InputType, OutputType>(
    //                     processing_method, OP, tensor_logical_shape, M_i, Offset_i);
    //             } else {
    //                 performTest_x1<InputType, OutputType>(
    //                     processing_method, OP, tensor_logical_shape, M_i, Offset_i, rowwise, colwise);
    //             }
    //         );
    //     );
    // } else {
    //     auto OP = &identity;
    //     switch (activation) {
    //         case ActivationKind::GeLU: OP = &dgelu; break;
    //         case ActivationKind::SiLU: OP = &dsilu; break;
    //         case ActivationKind::ReLU: OP = &drelu; break;
    //         case ActivationKind::QGeLU: OP = &dqgelu; break;
    //         case ActivationKind::SReLU: OP = &dsrelu; break;
    //     }
    //     TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
    //         TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
    //             if (scaling_direction == ScalingDirection::BOTH) {
    //                 performTest_x2<InputType, OutputType>(
    //                     processing_method, OP, tensor_logical_shape, M_i, Offset_i);
    //             } else {
    //                 performTest_x1<InputType, OutputType>(
    //                     processing_method, OP, tensor_logical_shape, M_i, Offset_i, rowwise, colwise);
    //             }
    //         );
    //     );
    // }
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

std::string to_string(const ActivationKind activation) {
    switch (activation) {
        case ActivationKind::Identity:  return "Identity";
        case ActivationKind::GeLU:      return "GeLU";
        case ActivationKind::SiLU:      return "SiLU";
        case ActivationKind::ReLU:      return "ReLU";
        case ActivationKind::QGeLU:     return "QGeLU";
        case ActivationKind::SReLU:     return "SReLU";
        default: return "";
    }
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    GroupedFusedCastMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(processing_methods),
        ::testing::ValuesIn(activation_kinds),
        ::testing::ValuesIn(scaling_directions),
        ::testing::ValuesIn(input_config),
        ::testing::Values(DType::kBFloat16),
        ::testing::Values(DType::kFloat8E4M3)),
        // ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        // ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    [](const testing::TestParamInfo<GroupedFusedCastMXFP8TestSuite::ParamType>& info) {
        const ProcessingMethod method = std::get<0>(info.param);
        std::string name = to_string(method);
        if (method != ProcessingMethod::CAST_ONLY && method != ProcessingMethod::CAST_DBIAS) {
            name += "X" + to_string(std::get<1>(info.param));
        }

        switch (std::get<2>(info.param)) {
            case ScalingDirection::ROWWISE: name += "_ROWWISE"; break;
            case ScalingDirection::COLWISE: name += "_COLWISE"; break;
            case ScalingDirection::BOTH:    name += "_BOTH"; break;
        }

        const std::vector<size_t> input = std::get<3>(info.param);
        name += "_Shape_";
        switch(static_cast<ShapeRepresentation>(input[0])) {
            case ShapeRepresentation::SAME_BOTH_DIMS:       name += "SAME_BOTH_DIMS"; break;
            case ShapeRepresentation::VARYING_FIRST_DIM:    name += "VARYING_FIRST_DIM"; break;
            case ShapeRepresentation::VARYING_LAST_DIM:     name += "VARYING_LAST_DIM"; break;
            case ShapeRepresentation::VARYING_BOTH_DIMS:    name += "VARYING_BOTH_DIMS"; break;
        };

        name += "_N_" + std::to_string(input[1]);

        name += "_Shape_" +
                std::to_string(input[2]) +
                "X" + std::to_string(input[3]);

        // name += "_DimsM_";
        // const auto& M_i_ = std::get<5>(info.param);
        // for (size_t i = 0; i < M_i_.size(); ++i) {
        //     const size_t m = M_i_[i];
        //     name += std::to_string(m);
        //     if (i < M_i_.size() - 1) {
        //         name += "X";
        //     }
        // }
        // name += "_Offsets_";
        // const auto& Offset_i_ = std::get<6>(info.param);
        // for (size_t i = 0; i < Offset_i_.size(); ++i) {
        //     const size_t offset = Offset_i_[i];
        //     name += std::to_string(offset);
        //     if (i < Offset_i_.size() - 1) {
        //         name += "X";
        //     }
        // }
        name += "_" + test::typeName(std::get<4>(info.param)) +
                "_" + test::typeName(std::get<5>(info.param));
        return name;
    });
