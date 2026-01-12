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
    SAME_MK = 0,
    VARYING_M = 1,
    VARYING_K = 2,
    VARYING_MK = 3
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
                    const size_t num_tensors,
                    const std::vector<size_t>& logical_shape_vec,
                    const bool rowwise,
                    const bool colwise) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t rows = logical_shape_vec[0];
    const size_t cols = logical_shape_vec[1];

    const size_t M = rows / num_tensors;
    const size_t K = cols;

    std::vector<size_t> scales_rowwise_shape = {rows, cols / 32};
    std::vector<size_t> scales_colwise_shape = {rows / 32, cols};

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

    size_t tensor_elts[2] = {128 * 128, 128 * 128};
    std::vector<size_t> offsets_h(num_tensors);
    offsets_h[0] = 0;
    for (size_t t = 1; t < num_tensors; ++t) {
        offsets_h[t] = offsets_h[t-1] + tensor_elts[t-1];
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

    InputType* in_data_d;
    OutputType* out_data_rowwise_d;
    OutputType* out_data_colwise_d;
    fp8e8m0* out_scales_rowwise_d;
    fp8e8m0* out_scales_colwise_d;
    size_t* offsets_d;

    cudaMalloc((void**)&in_data_d, in_data_size);
    cudaMemcpy(in_data_d, in_data.data(), in_data_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&offsets_d, in_data_size);
    cudaMemcpy(offsets_d, offsets_h.data(), num_tensors * sizeof(size_t), cudaMemcpyHostToDevice);

    NVTEShape logical_shape_ = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());
    NVTEShape offsets_shape_;
    offsets_shape_.data[0] = num_tensors;
    offsets_shape_.ndim = 1;

    NVTEGroupedTensor in_group_tensor = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape_);
    NVTEGroupedTensor out_group_tensor = nvte_create_grouped_tensor(NVTE_MXFP8_1D_SCALING, num_tensors, logical_shape_);

    NVTEBasicTensor in_data_tensor = {in_data_d, static_cast<NVTEDType>(itype), logical_shape_};
    nvte_set_grouped_tensor_param(&in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData, &in_data_tensor);
    
    NVTEBasicTensor offsets_tensor = {offsets_d, kNVTEInt64, offsets_shape_};
    nvte_set_grouped_tensor_param(&in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets, &offsets_tensor);

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

    /* DO STUFF */
    // Reference (CPU)
    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t scales_stride_rowwise = K / 32;
        const size_t scales_stride_colwise = K;
        const size_t data_offset = t * (M * K);
        const size_t sfs_offset = t * (M * K / 32);

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

    if (rowwise) {
        cudaMemcpy(out_data_rowwise_h.data(), out_data_rowwise_d, out_data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_scales_rowwise_h.data(), out_scales_rowwise_d, out_scales_size, cudaMemcpyDeviceToHost);
    }

    if (colwise) {
        cudaMemcpy(out_data_colwise_h.data(), out_data_colwise_d, out_data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_scales_colwise_h.data(), out_scales_colwise_d, out_scales_size, cudaMemcpyDeviceToHost);
    }


    cudaFree(in_data_d);
    cudaFree(offsets_d);
    if (rowwise) {
        cudaFree(out_data_rowwise_d);
        cudaFree(out_scales_rowwise_d);
    }
    if (colwise) {
        cudaFree(out_data_colwise_d);
        cudaFree(out_scales_colwise_d);
    }

    // const size_t block_size_rows = rowwise ? 1 : 32;
    // const size_t block_size_cols = colwise ? 1 : 32;

    // const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, block_size_rows,
    //                                                               block_size_cols);

    // const size_t unpadded_blocks_Y = scale_dims[0];
    // const size_t unpadded_blocks_X = scale_dims[1];
    // const size_t blocks_Y = scale_dims[2];
    // const size_t blocks_X = scale_dims[3];
    // const size_t scales_stride = blocks_X;

    // Tensor input("input", shape, itype);
    // Tensor grad("grad", shape, itype);
    // Tensor output_c("output_c", shape, otype, rowwise, colwise, NVTE_MXFP8_1D_SCALING);
    // Tensor output_dbias("output_dbias", std::vector<size_t>{ cols }, itype);

    // std::unique_ptr<OutputType[]> ref_output_c = std::make_unique<OutputType[]>(rows * cols);
    // std::unique_ptr<InputType[]> ref_output_dbias = std::make_unique<InputType[]>(cols);
    // std::unique_ptr<fp8e8m0[]> ref_output_scales = std::make_unique<fp8e8m0[]>(blocks_Y * blocks_X);

    // fillCase<EncodingType>(&input, InputsFillCase::uniform);
    // fillUniform(&grad);

    // Tensor workspace;
    // switch (processing_method) {
    //     case ProcessingMethod::CAST_ONLY: {
    //         nvte_quantize(input.data(), output_c.data(), 0);
    //         break;
    //     }
    //     case ProcessingMethod::CAST_DBIAS: {
    //         nvte_quantize_dbias(grad.data(),
    //                             output_c.data(),
    //                             output_dbias.data(),
    //                             workspace.data(),
    //                             0);
    //         workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

    //         nvte_quantize_dbias(grad.data(),
    //                             output_c.data(),
    //                             output_dbias.data(),
    //                             workspace.data(),
    //                             0);
    //         break;
    //     }
    //     case ProcessingMethod::CAST_DBIAS_DACT: {
    //         auto nvte_quantize_dbias_dact = &nvte_quantize_dbias_dgelu;
    //         if (OP == &dsilu)       { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsilu; }
    //         else if (OP == &drelu)  { nvte_quantize_dbias_dact = &nvte_quantize_dbias_drelu; }
    //         else if (OP == &dqgelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dqgelu; }
    //         else if (OP == &dsrelu) { nvte_quantize_dbias_dact = &nvte_quantize_dbias_dsrelu; }

    //         nvte_quantize_dbias_dact(grad.data(),
    //                                  input.data(),
    //                                  output_c.data(),
    //                                  output_dbias.data(),
    //                                  workspace.data(),
    //                                  0);
    //         workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());

    //         nvte_quantize_dbias_dact(grad.data(),
    //                                  input.data(),
    //                                  output_c.data(),
    //                                  output_dbias.data(),
    //                                  workspace.data(),
    //                                  0);
    //         break;
    //     }
    //     case ProcessingMethod::CAST_DACT: {
    //         auto nvte_dact = &nvte_dgelu;
    //         if (OP == &dsilu)       { nvte_dact = &nvte_dsilu; }
    //         else if (OP == &drelu)  { nvte_dact = &nvte_drelu; }
    //         else if (OP == &dqgelu) { nvte_dact = &nvte_dqgelu; }
    //         else if (OP == &dsrelu) { nvte_dact = &nvte_dsrelu; }

    //         nvte_dact(grad.data(), input.data(), output_c.data(), 0);
    //         break;
    //     }
    //     case ProcessingMethod::CAST_ACT: {
    //         auto nvte_act = &nvte_gelu;
    //         if (OP == &silu)       { nvte_act = &nvte_silu; }
    //         else if (OP == &relu)  { nvte_act = &nvte_relu; }
    //         else if (OP == &qgelu) { nvte_act = &nvte_qgelu; }
    //         else if (OP == &srelu) { nvte_act = &nvte_srelu; }

    //         nvte_act(input.data(), output_c.data(), 0);
    //         break;
    //     }
    // }

    // cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // compute_ref<InputType, OutputType>(processing_method,
    //                                    OP,
    //                                    rowwise,
    //                                    colwise,
    //                                    input.rowwise_cpu_dptr<InputType>(),
    //                                    grad.rowwise_cpu_dptr<InputType>(),
    //                                    ref_output_c.get(),
    //                                    ref_output_c.get(),
    //                                    ref_output_scales.get(),
    //                                    ref_output_scales.get(),
    //                                    ref_output_dbias.get(),
    //                                    rows,
    //                                    cols,
    //                                    scales_stride,
    //                                    scales_stride);

    // const uint8_t * const gpu_scales_ptr = rowwise
    //                                        ? output_c.rowwise_cpu_scale_inv_ptr<fp8e8m0>()
    //                                        : output_c.columnwise_cpu_scale_inv_ptr<fp8e8m0>();

    // const size_t scale_diff_abs_tolerance = 0;
    // const double abs_tolerable_mismatches_limit = 0.0;
    // const double rel_tolerable_mismatches_limit = 0.0;

    // size_t mismatches_scales = 0;

    // compare_scaling_factors("scales", gpu_scales_ptr, ref_output_scales.get(),
    //                         unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
    //                         mismatches_scales,
    //                         scale_diff_abs_tolerance,
    //                         abs_tolerable_mismatches_limit,
    //                         rel_tolerable_mismatches_limit);

    // const size_t mismatches_elts = 32 * mismatches_scales;
    // auto [atol, rtol] = getTolerances(otype);
    // compareResults("output_c", output_c, ref_output_c.get(), rowwise, atol, rtol, true, mismatches_elts);

    // if (processing_method == ProcessingMethod::CAST_DBIAS
    //     || processing_method == ProcessingMethod::CAST_DBIAS_DACT)
    // {
    //     auto [atol_dbias, rtol_dbias] = getTolerances(itype);
    //     if (itype == DType::kFloat32) {
    //         atol_dbias = 1e-4;
    //         rtol_dbias *= sqrt(static_cast<double>(rows)) ;
    //     } else {
    //         rtol_dbias *= 4;
    //     }
    //     compareResults("output_dbias", output_dbias, ref_output_dbias.get(), true, atol_dbias, rtol_dbias);
    // }
}

/**
 * Scaling along both dimensions (rows and columns)
 * Produces two sets of scaled output data and the corresponding data of the fused operation (dbias):
 * 1) Scaled rows + row-wise scaling factors
 *      AND
 * 2) Scaled columns + column-wise scaling factors
 */
/*
template <typename InputType, typename OutputType>
void performTest_x2(const ProcessingMethod processing_method,
                    float (*OP)(const float),
                    const std::pair<size_t, size_t>& shape,
                    const std::vector<size_t>& M_i,
                    const std::vector<size_t>& Offset_i) {
    using namespace test;
    using EncodingType = fp32;
    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const size_t rows = shape.first;
    const size_t cols = shape.second;

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

    fillCase<EncodingType>(&input, InputsFillCase::uniform);
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
    compare_scaling_factors("scales_rowwise", output.rowwise_cpu_scale_inv_ptr<fp8e8m0>(),
                            ref_scales_rowwise.get(), unpadded_blocks_Y_rowwise,
                            unpadded_blocks_X_rowwise, scales_stride_rowwise,
                            mismatches_scales_rowwise,
                            scale_diff_abs_tolerance,
                            abs_tolerable_mismatches_limit,
                            rel_tolerable_mismatches_limit);

    size_t mismatches_scales_colwise = 0;
    compare_scaling_factors("scales_colwise", output.columnwise_cpu_scale_inv_ptr<fp8e8m0>(),
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
*/

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

// {num_tensors, logical_shape_M, logical_shape_K, [M_i], [K_i], [Offset_i]}
std::vector<std::vector<size_t>> input_config = {
    {1, 128, 128},
    {2, 256, 128},
    // {3, 128 * 3, 256},
    // {5, 256 * 5, 256},
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

    const size_t num_tensors = input_config[0];
    const std::vector<size_t> logical_shape = {input_config[1], input_config[2]};
  
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
    performTest_x1<bf16, fp8e4m3>(processing_method, OP, num_tensors, logical_shape, rowwise, colwise);

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
        name += "_N_" + std::to_string(input[0]);

        name += "_Shape_" +
                std::to_string(input[1]) +
                "X" + std::to_string(input[2]);

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
