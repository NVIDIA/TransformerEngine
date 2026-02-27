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
void performTest(const ProcessingMethod processing_method,
                 float (*OP)(const float),
                 const ShapeRepresentation shape_rep,
                 const size_t num_tensors,
                 const std::vector<size_t>& logical_shape_vec,
                 const std::vector<size_t>& first_dims_h,
                 const std::vector<size_t>& last_dims_h,
                 const std::vector<size_t>& offsets_h,
                 const bool rowwise,
                 const bool colwise) {
    using namespace test;

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    const bool compute_dbias = (processing_method == ProcessingMethod::CAST_DBIAS
                                || processing_method == ProcessingMethod::CAST_DBIAS_DACT);

    const size_t rows = logical_shape_vec[0];
    const size_t cols = logical_shape_vec[1];

    size_t elts_num = 0;
    size_t rowwise_sfs_num = 0;
    size_t colwise_sfs_num = 0;
    size_t sum_of_last_dims = 0;

    std::vector<size_t> rowwise_scales_first_dim(num_tensors, 0);
    std::vector<size_t> rowwise_scales_last_dim(num_tensors, 0);
    std::vector<size_t> rowwise_scales_offset(num_tensors + 1, 0);
    std::vector<size_t> colwise_scales_first_dim(num_tensors, 0);
    std::vector<size_t> colwise_scales_last_dim(num_tensors, 0);
    std::vector<size_t> colwise_scales_offset(num_tensors + 1, 0);
    std::vector<size_t> dbias_offsets(num_tensors + 1, 0);

    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t M = first_dims_h[t];
        const size_t K = last_dims_h[t];
        const size_t elts = M * K;
        elts_num += elts;

        const size_t unpadded_rowwise_blocks_Y = M;
        const size_t unpadded_rowwise_blocks_X = divide_round_up(K, 32);
        const size_t unpadded_colwise_blocks_Y = divide_round_up(M, 32);
        const size_t unpadded_colwise_blocks_X = K;

        rowwise_scales_first_dim[t] = round_up_to_nearest_multiple(unpadded_rowwise_blocks_Y, 128);
        rowwise_scales_last_dim[t] = round_up_to_nearest_multiple(unpadded_rowwise_blocks_X, 4);
        colwise_scales_first_dim[t] = round_up_to_nearest_multiple(unpadded_colwise_blocks_Y, 4);
        colwise_scales_last_dim[t] = round_up_to_nearest_multiple(unpadded_colwise_blocks_X, 128);

        const size_t rowwise_sfs = rowwise_scales_first_dim[t] * rowwise_scales_last_dim[t];
        const size_t colwise_sfs = colwise_scales_first_dim[t] * colwise_scales_last_dim[t];

        rowwise_sfs_num += rowwise_sfs;
        colwise_sfs_num += colwise_sfs;
        sum_of_last_dims += K;

        rowwise_scales_offset[t+1] = rowwise_sfs_num;
        colwise_scales_offset[t+1] = colwise_sfs_num;
        dbias_offsets[t+1] = sum_of_last_dims;
    }

    std::vector<size_t> scales_rowwise_shape = {rowwise_sfs_num};
    std::vector<size_t> scales_colwise_shape = {colwise_sfs_num};

    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-2.0, 1.0);

    std::vector<InputType> in_data(elts_num);
    std::vector<InputType> grad_data(elts_num);

    std::vector<OutputType> out_data_rowwise_h(rowwise ? elts_num : 0);
    std::vector<OutputType> out_data_colwise_h(colwise ? elts_num : 0);
    std::vector<fp8e8m0> out_scales_rowwise_h(rowwise ? rowwise_sfs_num : 0);
    std::vector<fp8e8m0> out_scales_colwise_h(colwise ? colwise_sfs_num : 0);

    std::vector<OutputType> out_data_rowwise_ref(rowwise ? elts_num : 0);
    std::vector<OutputType> out_data_colwise_ref(colwise ? elts_num : 0);
    std::vector<fp8e8m0> out_scales_rowwise_ref(rowwise ? rowwise_sfs_num : 0);
    std::vector<fp8e8m0> out_scales_colwise_ref(colwise ? colwise_sfs_num : 0);

    std::vector<InputType> ref_output_dbias(sum_of_last_dims, static_cast<InputType>(0.0f));

    for (size_t i = 0; i < elts_num; ++i) {
        const float val = dis(gen);
        grad_data[i] = static_cast<InputType>(val);
        in_data[i] = static_cast<InputType>(val);
    }

    const OutputType zero_elt = static_cast<OutputType>(0.0f);
    const fp8e8m0 zero_SF = static_cast<fp8e8m0>(0.0f);
    if (rowwise) {
        std::fill(out_data_rowwise_h.begin(), out_data_rowwise_h.end(), zero_elt);
        std::fill(out_data_rowwise_ref.begin(), out_data_rowwise_ref.end(), zero_elt);
        std::fill(out_scales_rowwise_h.begin(), out_scales_rowwise_h.end(), zero_SF);
        std::fill(out_scales_rowwise_ref.begin(), out_scales_rowwise_ref.end(), zero_SF);
    }
    if (colwise) {
        std::fill(out_data_colwise_h.begin(), out_data_colwise_h.end(), zero_elt);
        std::fill(out_data_colwise_ref.begin(), out_data_colwise_ref.end(), zero_elt);
        std::fill(out_scales_colwise_h.begin(), out_scales_colwise_h.end(), zero_SF);
        std::fill(out_scales_colwise_ref.begin(), out_scales_colwise_ref.end(), zero_SF);
    }

    const size_t in_data_size = elts_num * sizeof(InputType);
    const size_t out_data_size = elts_num * sizeof(OutputType);
    const size_t dbias_data_size = sum_of_last_dims * sizeof(InputType);
    const size_t rowwise_scales_size = rowwise_sfs_num * sizeof(fp8e8m0);
    const size_t colwise_scales_size = colwise_sfs_num * sizeof(fp8e8m0);

    const size_t first_dims_size = num_tensors * sizeof(size_t);
    const size_t last_dims_size = num_tensors * sizeof(size_t);
    const size_t offsets_size = (num_tensors + 1) * sizeof(size_t);

    InputType* grad_data_d = nullptr;
    InputType* in_data_d = nullptr;
    InputType* dbias_out_data_d = nullptr;
    OutputType* out_data_rowwise_d = nullptr;
    OutputType* out_data_colwise_d = nullptr;
    fp8e8m0* out_scales_rowwise_d = nullptr;
    fp8e8m0* out_scales_colwise_d = nullptr;
    size_t* first_dims_d = nullptr;
    size_t* last_dims_d = nullptr;
    size_t* offsets_d = nullptr;

    cudaMalloc((void**)&grad_data_d, in_data_size);
    cudaMalloc((void**)&in_data_d, in_data_size);
    cudaMalloc((void**)&first_dims_d, first_dims_size);
    cudaMalloc((void**)&last_dims_d, last_dims_size);
    cudaMalloc((void**)&offsets_d, offsets_size);

    cudaMemcpy(grad_data_d, grad_data.data(), in_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(in_data_d, in_data.data(), in_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(first_dims_d, first_dims_h.data(), first_dims_size, cudaMemcpyHostToDevice);
    cudaMemcpy(last_dims_d, last_dims_h.data(), last_dims_size, cudaMemcpyHostToDevice);
    cudaMemcpy(offsets_d, offsets_h.data(), offsets_size, cudaMemcpyHostToDevice);

    NVTEShape logical_shape_ = nvte_make_shape(logical_shape_vec.data(), logical_shape_vec.size());

    std::vector<size_t> dbias_logical_shape_vec= {num_tensors, cols};
    NVTEShape dbias_logical_shape_ = nvte_make_shape(dbias_logical_shape_vec.data(),
                                                     dbias_logical_shape_vec.size());

    NVTEShape first_dims_shape_;
    NVTEShape last_dims_shape_;
    NVTEShape offsets_shape_;

    first_dims_shape_.ndim = 1;
    last_dims_shape_.ndim = 1;
    offsets_shape_.ndim = 1;

    first_dims_shape_.data[0] = num_tensors;
    last_dims_shape_.data[0] = num_tensors;
    offsets_shape_.data[0] = num_tensors + 1;

    NVTEGroupedTensor grad_group_tensor = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape_);
    NVTEGroupedTensor in_group_tensor = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, logical_shape_);
    NVTEGroupedTensor out_group_tensor = nvte_create_grouped_tensor(NVTE_MXFP8_1D_SCALING, num_tensors, logical_shape_);
    NVTEGroupedTensor output_dbias_tensor = nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, dbias_logical_shape_);

    NVTEBasicTensor grad_data_tensor = {grad_data_d, static_cast<NVTEDType>(itype), logical_shape_};
    NVTEBasicTensor in_data_tensor = {in_data_d, static_cast<NVTEDType>(itype), logical_shape_};
    nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData,
                                  &in_data_tensor, sizeof(in_data_tensor));
    nvte_set_grouped_tensor_param(grad_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData,
                                  &grad_data_tensor, sizeof(grad_data_tensor));

    if ((shape_rep == VARYING_FIRST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
        NVTEBasicTensor first_dims_tensor = {first_dims_d, kNVTEInt64, first_dims_shape_};
        nvte_set_grouped_tensor_param(grad_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                      &first_dims_tensor, sizeof(first_dims_tensor));
        nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                      &first_dims_tensor, sizeof(first_dims_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                      &first_dims_tensor, sizeof(first_dims_tensor));
    }

    if ((shape_rep == VARYING_LAST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
        NVTEBasicTensor last_dims_tensor = {last_dims_d, kNVTEInt64, last_dims_shape_};
        nvte_set_grouped_tensor_param(grad_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims,
                                      &last_dims_tensor, sizeof(last_dims_tensor));
        nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims,
                                      &last_dims_tensor, sizeof(last_dims_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims,
                                      &last_dims_tensor, sizeof(last_dims_tensor));
    }

    if (shape_rep != SAME_BOTH_DIMS) {
        NVTEBasicTensor offsets_tensor = {offsets_d, kNVTEInt64, offsets_shape_};
        nvte_set_grouped_tensor_param(grad_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                      &offsets_tensor, sizeof(offsets_tensor));
        nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                      &offsets_tensor, sizeof(offsets_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                      &offsets_tensor, sizeof(offsets_tensor));
    }

    if (rowwise) {
        cudaMalloc((void**)&out_data_rowwise_d, out_data_size);
        cudaMalloc((void**)&out_scales_rowwise_d, rowwise_scales_size);
        cudaMemset(out_data_rowwise_d, 0, out_data_size);
        cudaMemset(out_scales_rowwise_d, 0, rowwise_scales_size);
        NVTEBasicTensor out_data_rowwise_tensor = {out_data_rowwise_d, static_cast<NVTEDType>(otype), logical_shape_};
        NVTEShape scales_rowwise_shape_ = nvte_make_shape(scales_rowwise_shape.data(), scales_rowwise_shape.size());
        NVTEBasicTensor out_scales_rowwise_tensor = {out_scales_rowwise_d, NVTEDType::kNVTEFloat8E8M0, scales_rowwise_shape_};
        nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData,
                                      &out_data_rowwise_tensor, sizeof(out_data_rowwise_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedRowwiseScaleInv,
                                      &out_scales_rowwise_tensor, sizeof(out_scales_rowwise_tensor));
    }

    if (colwise) {
        cudaMalloc((void**)&out_data_colwise_d, out_data_size);
        cudaMalloc((void**)&out_scales_colwise_d, colwise_scales_size);
        cudaMemset(out_data_colwise_d, 0, out_data_size);
        cudaMemset(out_scales_colwise_d, 0, colwise_scales_size);
        NVTEBasicTensor out_data_colwise_tensor = {out_data_colwise_d, static_cast<NVTEDType>(otype), logical_shape_};
        NVTEShape scales_colwise_shape_ = nvte_make_shape(scales_colwise_shape.data(), scales_colwise_shape.size());
        NVTEBasicTensor out_scales_colwise_tensor = {out_scales_colwise_d, NVTEDType::kNVTEFloat8E8M0, scales_colwise_shape_};
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedColumnwiseData,
                                      &out_data_colwise_tensor, sizeof(out_data_colwise_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedColumnwiseScaleInv,
                                      &out_scales_colwise_tensor, sizeof(out_scales_colwise_tensor));
    }

    if (compute_dbias) {
        cudaMalloc((void**)&dbias_out_data_d, dbias_data_size);
        cudaMemset(dbias_out_data_d, 0, dbias_data_size);
        NVTEBasicTensor output_dbias_data_tensor = {dbias_out_data_d, static_cast<NVTEDType>(itype), dbias_logical_shape_};
        nvte_set_grouped_tensor_param(output_dbias_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData, &output_dbias_data_tensor);
    }

    // Reference (CPU)
    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t M = first_dims_h[t];
        const size_t K = last_dims_h[t];

        const size_t scales_stride_rowwise = rowwise_scales_last_dim[t];
        const size_t scales_stride_colwise = colwise_scales_last_dim[t];
        const size_t data_offset = offsets_h[t];
        const size_t rowwise_sfs_offset = rowwise_scales_offset[t];
        const size_t colwise_sfs_offset = colwise_scales_offset[t];
        const size_t dbias_offset = dbias_offsets[t];

        const InputType* const grad_ptr = grad_data.data() + data_offset;
        const InputType* const in_ptr = in_data.data() + data_offset;
        OutputType* const out_data_rowwise_ptr = out_data_rowwise_ref.data() + data_offset;
        OutputType* const out_data_colwise_ptr = out_data_colwise_ref.data() + data_offset;
        fp8e8m0* const out_scales_rowwise_ptr = out_scales_rowwise_ref.data() + rowwise_sfs_offset;
        fp8e8m0* const out_scales_colwise_ptr = out_scales_colwise_ref.data() + colwise_sfs_offset;
        InputType* const ref_output_dbias_ptr = ref_output_dbias.data() + dbias_offset;

        compute_ref<InputType, OutputType>(
            processing_method, OP, rowwise, colwise, in_ptr, grad_ptr,
            out_data_rowwise_ptr, out_data_colwise_ptr,
            out_scales_rowwise_ptr, out_scales_colwise_ptr,
            ref_output_dbias_ptr, M, K,
            scales_stride_rowwise,
            scales_stride_colwise);
    }

    // GPU
    Tensor workspace;
    switch (processing_method) {
        case ProcessingMethod::CAST_ONLY: {
            nvte_group_quantize(in_group_tensor, out_group_tensor, 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS: {
            nvte_group_quantize_dbias(grad_group_tensor, out_group_tensor, output_dbias_tensor, workspace.data(), 0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
            nvte_group_quantize_dbias(grad_group_tensor, out_group_tensor, output_dbias_tensor, workspace.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_DBIAS_DACT: {
            auto nvte_group_quantize_dbias_dact = &nvte_group_quantize_dbias_dgelu;
            if (OP == &dsilu)       { nvte_group_quantize_dbias_dact = &nvte_group_quantize_dbias_dsilu; }
            else if (OP == &drelu)  { nvte_group_quantize_dbias_dact = &nvte_group_quantize_dbias_drelu; }
            else if (OP == &dqgelu) { nvte_group_quantize_dbias_dact = &nvte_group_quantize_dbias_dqgelu; }
            else if (OP == &dsrelu) { nvte_group_quantize_dbias_dact = &nvte_group_quantize_dbias_dsrelu; }

            nvte_group_quantize_dbias_dact(grad_group_tensor, in_group_tensor, out_group_tensor,
                                           output_dbias_tensor, workspace.data(), 0);
            workspace = Tensor("workspace", workspace.rowwise_shape(), workspace.dtype());
            nvte_group_quantize_dbias_dact(grad_group_tensor, in_group_tensor, out_group_tensor,
                                           output_dbias_tensor, workspace.data(), 0);
            break;
        }
        case ProcessingMethod::CAST_ACT: {
            auto nvte_group_act = &nvte_group_gelu;
            if (OP == &silu)       { nvte_group_act = &nvte_group_silu; }
            else if (OP == &relu)  { nvte_group_act = &nvte_group_relu; }
            else if (OP == &qgelu) { nvte_group_act = &nvte_group_qgelu; }
            else if (OP == &srelu) { nvte_group_act = &nvte_group_srelu; }
            nvte_group_act(in_group_tensor, out_group_tensor, 0);
            break;
        }
        case ProcessingMethod::CAST_DACT: {
            auto nvte_group_dact = &nvte_group_dgelu;
            if (OP == &dsilu)       { nvte_group_dact = &nvte_group_dsilu; }
            else if (OP == &drelu)  { nvte_group_dact = &nvte_group_drelu; }
            else if (OP == &dqgelu) { nvte_group_dact = &nvte_group_dqgelu; }
            else if (OP == &dsrelu) { nvte_group_dact = &nvte_group_dsrelu; }
            nvte_group_dact(grad_group_tensor, in_group_tensor, out_group_tensor, 0);
            break;
        }
    }
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto [atol, rtol] = getTolerances(otype);
    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 0.0;
    const double rel_tolerable_mismatches_limit = 0.0;

    if (rowwise) {
        cudaMemcpy(out_data_rowwise_h.data(), out_data_rowwise_d, out_data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_scales_rowwise_h.data(), out_scales_rowwise_d, rowwise_scales_size, cudaMemcpyDeviceToHost);

        size_t mismatches_scales = 0;
        compare_scaling_factors("rowwise_scales", out_scales_rowwise_h.data(), out_scales_rowwise_ref.data(),
                                1, rowwise_sfs_num, rowwise_sfs_num, mismatches_scales, scale_diff_abs_tolerance,
                                abs_tolerable_mismatches_limit, rel_tolerable_mismatches_limit);

        const size_t mismatches_elts = 32 * mismatches_scales;

        compare_scaled_elts<OutputType>("rowwise_output", out_data_rowwise_ref.data(),
                                        out_data_rowwise_h.data(), rows, cols, true, mismatches_elts);
    }

    if (colwise) {
        cudaMemcpy(out_data_colwise_h.data(), out_data_colwise_d, out_data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(out_scales_colwise_h.data(), out_scales_colwise_d, colwise_scales_size, cudaMemcpyDeviceToHost);

        size_t mismatches_scales = 0;
        compare_scaling_factors("colwise_scales", out_scales_colwise_h.data(), out_scales_colwise_ref.data(),
                                1, colwise_sfs_num, colwise_sfs_num, mismatches_scales, scale_diff_abs_tolerance,
                                abs_tolerable_mismatches_limit, rel_tolerable_mismatches_limit);

        const size_t mismatches_elts = 32 * mismatches_scales;

        compare_scaled_elts<OutputType>("colwise_output", out_data_colwise_ref.data(),
                                        out_data_colwise_h.data(), rows, cols, false, mismatches_elts);
    }

    if (compute_dbias) {
        Tensor output_dbias("output_dbias", std::vector<size_t>{ sum_of_last_dims }, itype);
        cudaMemcpy(output_dbias.rowwise_dptr(), dbias_out_data_d, dbias_data_size, cudaMemcpyDeviceToDevice);

        auto [atol_dbias, rtol_dbias] = getTolerances(itype);
        if (itype == DType::kFloat32) {
            atol_dbias = 1e-4;
            rtol_dbias *= sqrt(static_cast<double>(rows)) ;
        } else {
            rtol_dbias *= 4;
        }
        compareResults("output_dbias", output_dbias, ref_output_dbias.data(), true, atol_dbias, rtol_dbias);
    }

    cudaFree(grad_data_d);
    cudaFree(in_data_d);
    cudaFree(dbias_out_data_d);
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
    ProcessingMethod::CAST_DBIAS,
    ProcessingMethod::CAST_DBIAS_DACT,
    ProcessingMethod::CAST_DACT,
    ProcessingMethod::CAST_ACT,
};

std::vector<ActivationKind> activation_kinds = {
    ActivationKind::Identity,
    ActivationKind::GeLU,
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
    ScalingDirection::COLWISE,
    ScalingDirection::BOTH,
};

// {shape_representation, num_tensors, [logical_shape_M, logical_shape_K], [M_i], [K_i]}
std::vector<std::vector<size_t>> input_config = {
    {SAME_BOTH_DIMS,        1,      128,128},
    {SAME_BOTH_DIMS,        2,      256,128},
    {VARYING_FIRST_DIM,     2,      512,128,                    128,384},
    {VARYING_FIRST_DIM,     3,      1024,144,                   128,384,512},
    {VARYING_FIRST_DIM,     4,      1536,160,                   128,384,512,512},
    {VARYING_FIRST_DIM,     5,      4096,512,                   128,256,384,1024,2304},
    {VARYING_LAST_DIM,      3,      256,896,                    128,256,512},
    {VARYING_BOTH_DIMS,     2,      1,(128*128)+(256*256),      128,256,        128,256},
    {VARYING_BOTH_DIMS,     2,      1,(256*128)+(512*640),      256,512,        128,640},
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
    const DType input_type = std::get<4>(GetParam());
    const DType output_type = std::get<5>(GetParam());

    const ShapeRepresentation shape_rep = static_cast<ShapeRepresentation>(input_config[0]);
    const bool is_single_tensor = (shape_rep == SAME_BOTH_DIMS) || (shape_rep == VARYING_FIRST_DIM);

    const size_t num_tensors = input_config[1];
    const std::vector<size_t> logical_shape = {input_config[2], input_config[3]};
    std::vector<size_t> first_dims(num_tensors);
    std::vector<size_t> last_dims(num_tensors);
    std::vector<size_t> offsets(num_tensors + 1, 0);
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
                last_dims[t] = input_config[t + 4];
                break;
            }
            case VARYING_BOTH_DIMS: {
                first_dims[t] = input_config[t + 4];
                last_dims[t] = input_config[t + (4 + num_tensors)];
                break;
            }
        }
        offsets[t+1] = offsets[t] + first_dims[t] * last_dims[t];
        // Skip tests when the tensor shape is incompatible with the kernel.
        // The TMA engine requires strides to be 16-byte aligned.
        if ((first_dims[t] % 128 != 0) || (last_dims[t] % 16 != 0)) {
            GTEST_SKIP();
        }
        // If a grouped tensor has a varying last dimension, it must be a multiple of 128.
        // Otherwise, computing the grid size adds runtime overhead in the non-persistent kernel,
        // since the relevant tensor metadata resides in device memory.
        constexpr size_t CHUNK_DIM_X = 128;
        if (!is_single_tensor && (last_dims[t] % CHUNK_DIM_X != 0)) {
            GTEST_SKIP();
        }
    }
    // Skip dBias tests when tensors in the group have different last dimensions.
    if ((processing_method == ProcessingMethod::CAST_DBIAS || processing_method == ProcessingMethod::CAST_DBIAS_DACT)
        && !is_single_tensor) {
        GTEST_SKIP();
    }

    // Skip non-activation tests when the activation type is not Identity.
    if ((processing_method == ProcessingMethod::CAST_ONLY || processing_method == ProcessingMethod::CAST_DBIAS)
        && activation != ActivationKind::Identity) {
        GTEST_SKIP();
    }
    // Skip activation tests when the activation type is Identity.
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

    if (processing_method == ProcessingMethod::CAST_ACT) {
        switch (activation) {
            case ActivationKind::GeLU: OP = &gelu; break;
            case ActivationKind::SiLU: OP = &silu; break;
            case ActivationKind::ReLU: OP = &relu; break;
            case ActivationKind::QGeLU: OP = &qgelu; break;
            case ActivationKind::SReLU: OP = &srelu; break;
        }
    } else if (processing_method == ProcessingMethod::CAST_DACT
               || processing_method == ProcessingMethod::CAST_DBIAS_DACT) {
        switch (activation) {
            case ActivationKind::GeLU: OP = &dgelu; break;
            case ActivationKind::SiLU: OP = &dsilu; break;
            case ActivationKind::ReLU: OP = &drelu; break;
            case ActivationKind::QGeLU: OP = &dqgelu; break;
            case ActivationKind::SReLU: OP = &dsrelu; break;
        }
    }

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
            performTest<InputType, OutputType>(processing_method, OP, shape_rep, num_tensors,
                                               logical_shape, first_dims, last_dims, offsets,
                                               rowwise, colwise);
        );
    );
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
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    [](const testing::TestParamInfo<GroupedFusedCastMXFP8TestSuite::ParamType>& info) {
        const ProcessingMethod method = std::get<0>(info.param);
        std::string name = to_string(method);
        name += "X" + to_string(std::get<1>(info.param));

        switch (std::get<2>(info.param)) {
            case ScalingDirection::ROWWISE: name += "_ROWWISE_"; break;
            case ScalingDirection::COLWISE: name += "_COLWISE_"; break;
            case ScalingDirection::BOTH:    name += "_BIDIMENSIONAL_"; break;
        }

        const std::vector<size_t> input = std::get<3>(info.param);

        switch(static_cast<ShapeRepresentation>(input[0])) {
            case ShapeRepresentation::SAME_BOTH_DIMS:       name += "SAME_BOTH_DIMS"; break;
            case ShapeRepresentation::VARYING_FIRST_DIM:    name += "VARYING_FIRST_DIM"; break;
            case ShapeRepresentation::VARYING_LAST_DIM:     name += "VARYING_LAST_DIM"; break;
            case ShapeRepresentation::VARYING_BOTH_DIMS:    name += "VARYING_BOTH_DIMS"; break;
        };

        name += "_N_" + std::to_string(input[1]);

        name += "_SHAPE_" +
                std::to_string(input[2]) +
                "X" + std::to_string(input[3]);

        name += "_" + test::typeName(std::get<4>(info.param)) +
                "_" + test::typeName(std::get<5>(info.param));
        return name;
    });
