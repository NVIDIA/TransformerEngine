/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/cast.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

using namespace transformer_engine;
using namespace test;

namespace {

enum ShapeRepresentation {
  SAME_BOTH_DIMS    = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM  = 2,
  VARYING_BOTH_DIMS = 3
};

using nvfp4_scale_t = fp8e4m3;

double2 cvt_fp4x2_to_double2(fp4e2m1x2 fp4_pair) {
    const __half2_raw raw_truncated_to_fp4e2m1_pair =
        __nv_cvt_fp4x2_to_halfraw2(*reinterpret_cast<__nv_fp4x2_storage_t*>(&fp4_pair), __NV_E2M1);

    const __half2 truncated_to_fp4e2m1_pair(raw_truncated_to_fp4e2m1_pair);
    const double truncated_to_fp4e2m1_x = static_cast<double>(truncated_to_fp4e2m1_pair.x);
    const double truncated_to_fp4e2m1_y = static_cast<double>(truncated_to_fp4e2m1_pair.y);
    return {truncated_to_fp4e2m1_x, truncated_to_fp4e2m1_y};
}

template <typename InputType>
std::vector<InputType> create_transpose(const InputType* const input, const size_t rows, size_t cols) {
    std::vector<InputType> input_t(cols * rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t idx = i * cols + j;
            const size_t idx_t = j * rows + i;
            input_t[idx_t] = input[idx];
        }
    }
    return input_t;
}

template <typename T>
struct TypeExtrema;

template <>
struct TypeExtrema<bf16> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <typename T>
struct TypeExtrema {
  static constexpr float max = std::numeric_limits<T>::max();
};

// Compute "correct" per-block encoding scaling factor
float compute_scaling_coefficient(const nvfp4_scale_t S_dec_block, const float S_enc,
                                  const bool use_fast_math) {
    const float S_dec_block_as_fp32 = static_cast<float>(S_dec_block);
    float scale_rcp = 0.0f;
    if (use_fast_math) {
        scale_rcp = fminf(S_enc / S_dec_block_as_fp32, TypeExtrema<bf16>::max);
        scale_rcp = static_cast<float>(static_cast<bf16>(scale_rcp));
    } else {
        const float S_dec = 1.0f / S_enc;
        scale_rcp = fminf(1.0f / (S_dec_block_as_fp32 * S_dec), TypeExtrema<float>::max);
    }
    return scale_rcp;
}

nvfp4_scale_t compute_decoding_scaling_factor(const float block_amax, const float S_enc) {
  constexpr float fp4_max = 6.0f;
  const float S_dec_b = block_amax / fp4_max * S_enc;
  return static_cast<nvfp4_scale_t>(fminf(S_dec_b, TypeExtrema<float>::max));
}

// Compute the global encode scale factor for a given global amax
float compute_global_encode_scaling_factor(const float global_amax) {
  constexpr float fp8_max = 448.0f;     // 448.0f;
  constexpr float fp4_max = 6.0f;       // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;

  global_encode_scale = fminf(global_encode_scale, Numeric_Traits<float>::maxNorm);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

// 1D Scaling: Original implementation with 1x16 blocks
template <typename InputType>
void quantize_nvfp4(const InputType* const input,
                    fp4e2m1x2* const output,
                    fp8e4m3* const scales,
                    const size_t rows,
                    const size_t cols,
                    const size_t scales_stride,
                    const float global_amax,
                    const bool use_fast_math) {

    // Compute a global encoding/decoding scaling factor for all S_dec_b
    const float S_enc = compute_global_encode_scaling_factor(global_amax);

    constexpr size_t block_size_X = 16;
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    std::array<float, block_size_X> cache_buffer;
    for (size_t i = 0; i < block_size_X; ++i) {
        cache_buffer[i] = 0.0f;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = j_min + block_size_X;

            // Find block amax
            float block_amax = 0.0f;
            for (size_t j = j_min; j < j_max; ++j) {
                const size_t idx = i * cols + j;
                const size_t cache_idx = j - j_min;

                const float input_elt = static_cast<float>(input[idx]);
                const float act_elt = input_elt;

                // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                cache_buffer[cache_idx] = elt;
                block_amax = std::max(block_amax, std::abs(elt));
            }

            // Compute E4M3 scaling factor
            const nvfp4_scale_t S_dec_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);
            const float SFcoefficient = compute_scaling_coefficient(S_dec_b_fp8, S_enc, use_fast_math);

            const size_t scale_idx = i * scales_stride + block_X;
            scales[scale_idx] = S_dec_b_fp8;

            for (size_t j = j_min; j < j_max; j += 2) {
                const int idx_pair = (i * cols + j) / 2;
                const int cache_idx_x = j - j_min;
                const int cache_idx_y = cache_idx_x + 1;
                const float cached_x = cache_buffer[cache_idx_x];
                const float cached_y = cache_buffer[cache_idx_y];
                const float scaled_elt_x = cached_x * SFcoefficient;
                const float scaled_elt_y = cached_y * SFcoefficient;
                const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                output[idx_pair] = casted_to_e2m1_pair;

                // const double2 truncated_pair = cvt_fp4x2_to_double2(casted_to_e2m1_pair);
            }
        }
    }
}

template <typename InputType>
void compute_ref(const InputType* input,
                 fp4e2m1x2* output,
                 fp4e2m1x2* output_t,
                 fp8e4m3* scales,
                 fp8e4m3* scales_t,
                 const float global_amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride,
                 const size_t scales_stride_t,
                 const bool use_fast_math)
{
    std::vector<InputType> input_t = create_transpose(input, rows, cols);

    quantize_nvfp4<InputType>(input, output, scales, rows, cols, scales_stride, global_amax, use_fast_math);
    quantize_nvfp4<InputType>(input_t.data(), output_t, scales_t, cols, rows, scales_stride_t, global_amax, use_fast_math);
}

void compare_nvfp4_tensors(const std::string& name,
                           const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                           const int rows, const int cols,
                           double atol = 1e-5, double rtol = 1e-8) {
    constexpr int max_mismatches_to_print = 3;

    std::vector<std::string> mismatch_messages;
    size_t total_mismatches = 0;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 2) {
            const int idx = i * cols + j;
            double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
            double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));

            for (int k = 0; k < 2; ++k) {
                const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);

                const bool mismatch = fabs(t - r) > (atol + fabs(r) * rtol);
                if (mismatch) {
                    total_mismatches++;
                    // Optional: limit number of detailed messages to avoid overwhelming output
                    if (total_mismatches <= max_mismatches_to_print) {
                        std::string msg = "Mismatch at place (" + std::to_string(idx + k) + "): " +
                                          std::to_string(t) + " vs " + std::to_string(r) +
                                          " (abs_diff: " + std::to_string(fabs(t - r)) +
                                          ", rel_diff: " + std::to_string(r == 0 ? 0.0 : fabs((t - r) / r)) + ")";
                        mismatch_messages.push_back(msg);
                        std::cout << "Error in tensor " << name << ": " << msg << std::endl;
                    }
                }
            }
        }
    }

    bool print_detailed_summary = false;
    if (print_detailed_summary) {
        // Always report summary - either success or failure
        std::cout << "=== SUMMARY for tensor " << name << " ===" << std::endl;
        std::cout << "Total elements checked: " << (rows * cols) << std::endl;

        if (total_mismatches > 0) {
            std::cout << "STATUS: FAILED for output" << std::endl;
            std::cout << "Total mismatches found: " << total_mismatches << std::endl;
            std::cout << "Mismatch rate: " << (100.0 * total_mismatches) / (rows * cols) << "%" << std::endl;
            if (total_mismatches > max_mismatches_to_print) {
                std::cout << "... and " << (total_mismatches - max_mismatches_to_print)
                << " more mismatches (showing first " << max_mismatches_to_print << ")" << std::endl;
            }
            std::cout << "============================" << std::endl;

            GTEST_FAIL() << "Found " << total_mismatches << " mismatches in tensor " << name;
        } else {
            std::cout << "STATUS: PASSED for output" << std::endl;
            std::cout << "All elements match within tolerance!" << std::endl;
            std::cout << "Tensor " << name << " is IDENTICAL to reference" << std::endl;
            std::cout << "============================" << std::endl;
        }
    } else {
        if (total_mismatches > 0) {
            GTEST_FAIL() << "Found " << total_mismatches << " mismatches in tensor " << name;
        }
    }
}

template <typename InputType>
void performTest(const ShapeRepresentation shape_rep,
                 const size_t num_tensors,
                 const std::vector<size_t>& logical_shape,
                 const std::vector<size_t>& first_dims,
                 const std::vector<size_t>& last_dims,
                 const std::vector<size_t>& offsets,
                 const bool use_fast_math) {
    using namespace test;

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = DType::kFloat4E2M1;
    const size_t total_elts = offsets.back();
    std::vector<InputType> grouped_input(total_elts);

    // Validate logical shape against the offsets-based flattened size.
    size_t expected_total_elts = logical_shape[0] * logical_shape[1];
    if (shape_rep == VARYING_LAST_DIM) {
        expected_total_elts = logical_shape[0]
                              * std::accumulate(last_dims.begin(), last_dims.end(), static_cast<size_t>(0));
    }
    ASSERT_GE(expected_total_elts, total_elts);

    Tensor grouped_input_tensor("grouped_input", std::vector<size_t>{total_elts}, itype);
    fillCase<fp32>(&grouped_input_tensor, InputsFillCase::uniform);
    std::copy(grouped_input_tensor.rowwise_cpu_dptr<InputType>(),
              grouped_input_tensor.rowwise_cpu_dptr<InputType>() + total_elts,
              grouped_input.begin());

    const double atol = 1.0E-6;
    const double rtol = 1.0E-6;

    std::vector<size_t> rowwise_scales_stride(num_tensors, 0);
    std::vector<size_t> colwise_scales_stride(num_tensors, 0);
    std::vector<size_t> rowwise_unpadded_blocks_X(num_tensors, 0);
    std::vector<size_t> colwise_unpadded_blocks_X(num_tensors, 0);
    std::vector<size_t> rowwise_scale_offsets(num_tensors, 0);
    std::vector<size_t> colwise_scale_offsets(num_tensors, 0);

    size_t rowwise_scales_num = 0;
    size_t colwise_scales_num = 0;

    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t rows = first_dims[t];
        const size_t cols = last_dims[t];
        rowwise_unpadded_blocks_X[t] = divide_round_up(cols, static_cast<size_t>(16));
        colwise_unpadded_blocks_X[t] = divide_round_up(rows, static_cast<size_t>(16));

        rowwise_scales_stride[t] = round_up_to_nearest_multiple(rowwise_unpadded_blocks_X[t], static_cast<size_t>(4));
        colwise_scales_stride[t] = round_up_to_nearest_multiple(colwise_unpadded_blocks_X[t], static_cast<size_t>(4));

        rowwise_scale_offsets[t] = rowwise_scales_num;
        colwise_scale_offsets[t] = colwise_scales_num;

        rowwise_scales_num += rows * rowwise_scales_stride[t];
        colwise_scales_num += cols * colwise_scales_stride[t];
    }

    std::vector<fp4e2m1> out_data_rowwise_h(total_elts / 2);
    std::vector<fp4e2m1> out_data_colwise_h(total_elts / 2);
    std::vector<fp8e4m3> out_scales_rowwise_h(rowwise_scales_num);
    std::vector<fp8e4m3> out_scales_colwise_h(colwise_scales_num);

    std::vector<fp4e2m1> out_data_rowwise_ref(total_elts / 2);
    std::vector<fp4e2m1> out_data_colwise_ref(total_elts / 2);
    std::vector<std::vector<fp8e4m3>> out_scales_rowwise_ref(num_tensors);
    std::vector<std::vector<fp8e4m3>> out_scales_colwise_ref(num_tensors);
    std::vector<float> amax_per_tensor(num_tensors, 0.0f);

    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t rows = first_dims[t];
        const size_t cols = last_dims[t];
        const size_t tensor_offset = offsets[t];
        const size_t tensor_numel = rows * cols;
        ASSERT_EQ(offsets[t + 1] - offsets[t], tensor_numel);
        ASSERT_LE(tensor_offset + tensor_numel, total_elts);
        ASSERT_EQ(tensor_numel % 2, 0U);

        float amax = 0.0f;
        for (size_t i = 0; i < tensor_numel; ++i) {
            amax = fmaxf(amax, fabs(static_cast<float>(grouped_input[tensor_offset + i])));
        }
        amax_per_tensor[t] = amax;

        std::unique_ptr<fp4e2m1x2[]> ref_output = std::make_unique<fp4e2m1x2[]>(tensor_numel / 2);
        std::unique_ptr<fp4e2m1x2[]> ref_output_t = std::make_unique<fp4e2m1x2[]>(tensor_numel / 2);
        std::unique_ptr<fp8e4m3[]> ref_scales = std::make_unique<fp8e4m3[]>(rows * rowwise_scales_stride[t]);
        std::unique_ptr<fp8e4m3[]> ref_scales_t = std::make_unique<fp8e4m3[]>(cols * colwise_scales_stride[t]);

        compute_ref<InputType>(grouped_input.data() + tensor_offset,
                               ref_output.get(),
                               ref_output_t.get(),
                               ref_scales.get(),
                               ref_scales_t.get(),
                               amax_per_tensor[t],
                               rows,
                               cols,
                               rowwise_scales_stride[t],
                               colwise_scales_stride[t],
                               use_fast_math);

        std::memcpy(out_data_rowwise_ref.data() + tensor_offset / 2, ref_output.get(),
                    (tensor_numel / 2) * sizeof(fp4e2m1x2));
        std::memcpy(out_data_colwise_ref.data() + tensor_offset / 2, ref_output_t.get(),
                    (tensor_numel / 2) * sizeof(fp4e2m1x2));

        out_scales_rowwise_ref[t].assign(ref_scales.get(), ref_scales.get() + rows * rowwise_scales_stride[t]);
        out_scales_colwise_ref[t].assign(ref_scales_t.get(), ref_scales_t.get() + cols * colwise_scales_stride[t]);
    }

    const size_t in_data_size = total_elts * sizeof(InputType);
    const size_t out_data_size = (total_elts * typeToNumBits(otype)) / 8;
    const size_t rowwise_scales_size = rowwise_scales_num * sizeof(fp8e4m3);
    const size_t colwise_scales_size = colwise_scales_num * sizeof(fp8e4m3);
    const size_t amax_size = num_tensors * sizeof(float);

    std::vector<int64_t> first_dims_h(num_tensors, 0);
    std::vector<int64_t> last_dims_h(num_tensors, 0);
    std::vector<int64_t> offsets_h(num_tensors + 1, 0);
    for (size_t t = 0; t < num_tensors; ++t) {
        first_dims_h[t] = static_cast<int64_t>(first_dims[t]);
        last_dims_h[t] = static_cast<int64_t>(last_dims[t]);
    }
    for (size_t t = 0; t < num_tensors + 1; ++t) {
        offsets_h[t] = static_cast<int64_t>(offsets[t]);
    }

    InputType* in_data_d = nullptr;
    fp4e2m1* out_data_rowwise_d = nullptr;
    fp4e2m1* out_data_colwise_d = nullptr;
    fp8e4m3* out_scales_rowwise_d = nullptr;
    fp8e4m3* out_scales_colwise_d = nullptr;
    float* out_amax_rowwise_d = nullptr;
    float* out_amax_colwise_d = nullptr;
    int64_t* first_dims_d = nullptr;
    int64_t* last_dims_d = nullptr;
    int64_t* offsets_d = nullptr;

    cudaMalloc((void**)&in_data_d, in_data_size);
    cudaMalloc((void**)&out_data_rowwise_d, out_data_size);
    cudaMalloc((void**)&out_data_colwise_d, out_data_size);
    cudaMalloc((void**)&out_scales_rowwise_d, rowwise_scales_size);
    cudaMalloc((void**)&out_scales_colwise_d, colwise_scales_size);
    cudaMalloc((void**)&out_amax_rowwise_d, amax_size);
    cudaMalloc((void**)&out_amax_colwise_d, amax_size);

    cudaMalloc((void**)&first_dims_d, num_tensors * sizeof(int64_t));
    cudaMalloc((void**)&last_dims_d, num_tensors * sizeof(int64_t));
    cudaMalloc((void**)&offsets_d, (num_tensors + 1) * sizeof(int64_t));

    cudaMemcpy(in_data_d, grouped_input.data(), in_data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(out_amax_rowwise_d, amax_per_tensor.data(), amax_size, cudaMemcpyHostToDevice);
    cudaMemcpy(out_amax_colwise_d, amax_per_tensor.data(), amax_size, cudaMemcpyHostToDevice);
    cudaMemcpy(first_dims_d, first_dims_h.data(), num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(last_dims_d, last_dims_h.data(), num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(offsets_d, offsets_h.data(), (num_tensors + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaMemset(out_data_rowwise_d, 0, out_data_size);
    cudaMemset(out_data_colwise_d, 0, out_data_size);
    cudaMemset(out_scales_rowwise_d, 0, rowwise_scales_size);
    cudaMemset(out_scales_colwise_d, 0, colwise_scales_size);

    NVTEShape logical_shape_ = nvte_make_shape(logical_shape.data(), logical_shape.size());

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
    NVTEGroupedTensor out_group_tensor = nvte_create_grouped_tensor(NVTE_NVFP4_1D_SCALING, num_tensors, logical_shape_);

    NVTEBasicTensor in_data_tensor = {in_data_d, static_cast<NVTEDType>(itype), logical_shape_};
    nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData,
                                  &in_data_tensor, sizeof(in_data_tensor));

    NVTEBasicTensor out_data_rowwise_tensor = {out_data_rowwise_d, NVTEDType::kNVTEFloat4E2M1, logical_shape_};
    NVTEBasicTensor out_data_colwise_tensor = {out_data_colwise_d, NVTEDType::kNVTEFloat4E2M1, logical_shape_};
    nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData,
                                  &out_data_rowwise_tensor, sizeof(out_data_rowwise_tensor));
    nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedColumnwiseData,
                                  &out_data_colwise_tensor, sizeof(out_data_colwise_tensor));

    std::vector<size_t> rowwise_scales_shape = {rowwise_scales_num};
    std::vector<size_t> colwise_scales_shape = {colwise_scales_num};
    std::vector<size_t> amax_shape = {num_tensors};
    NVTEShape rowwise_scales_shape_ = nvte_make_shape(rowwise_scales_shape.data(), rowwise_scales_shape.size());
    NVTEShape colwise_scales_shape_ = nvte_make_shape(colwise_scales_shape.data(), colwise_scales_shape.size());
    NVTEShape amax_shape_ = nvte_make_shape(amax_shape.data(), amax_shape.size());
    NVTEBasicTensor out_scales_rowwise_tensor = { out_scales_rowwise_d, NVTEDType::kNVTEFloat8E4M3, rowwise_scales_shape_};
    NVTEBasicTensor out_scales_colwise_tensor = { out_scales_colwise_d, NVTEDType::kNVTEFloat8E4M3, colwise_scales_shape_};
    NVTEBasicTensor out_amax_rowwise_tensor = {out_amax_rowwise_d, NVTEDType::kNVTEFloat32, amax_shape_};
    NVTEBasicTensor out_amax_colwise_tensor = {out_amax_colwise_d, NVTEDType::kNVTEFloat32, amax_shape_};
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedRowwiseScaleInv,
                                  &out_scales_rowwise_tensor, sizeof(out_scales_rowwise_tensor));
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedColumnwiseScaleInv,
                                  &out_scales_colwise_tensor, sizeof(out_scales_colwise_tensor));
    nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedAmax,
                                  &out_amax_rowwise_tensor, sizeof(out_amax_rowwise_tensor));
    nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedColumnwiseAmax,
                                  &out_amax_colwise_tensor, sizeof(out_amax_colwise_tensor));

    if ((shape_rep == VARYING_FIRST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
        NVTEBasicTensor first_dims_tensor = {first_dims_d, kNVTEInt64, first_dims_shape_};
        nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                      &first_dims_tensor, sizeof(first_dims_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                      &first_dims_tensor, sizeof(first_dims_tensor));
    }

    if ((shape_rep == VARYING_LAST_DIM) || (shape_rep == VARYING_BOTH_DIMS)) {
        NVTEBasicTensor last_dims_tensor = {last_dims_d, kNVTEInt64, last_dims_shape_};
        nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims,
                                      &last_dims_tensor, sizeof(last_dims_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedLastDims,
                                      &last_dims_tensor, sizeof(last_dims_tensor));
    }

    if (shape_rep != SAME_BOTH_DIMS) {
        NVTEBasicTensor offsets_tensor = {offsets_d, kNVTEInt64, offsets_shape_};
        nvte_set_grouped_tensor_param(in_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                      &offsets_tensor, sizeof(offsets_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                      &offsets_tensor, sizeof(offsets_tensor));
    }

    QuantizationConfigWrapper quant_config;
    quant_config.set_use_fast_math(use_fast_math);
    quant_config.set_stochastic_rounding(false);

    nvte_group_quantize_v2(in_group_tensor, out_group_tensor, quant_config, 0);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    cudaMemcpy(out_data_rowwise_h.data(), out_data_rowwise_d, out_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_data_colwise_h.data(), out_data_colwise_d, out_data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_scales_rowwise_h.data(), out_scales_rowwise_d, rowwise_scales_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_scales_colwise_h.data(), out_scales_colwise_d, colwise_scales_size, cudaMemcpyDeviceToHost);

    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t rows = first_dims[t];
        const size_t cols = last_dims[t];
        const size_t tensor_offset = offsets[t];

        const fp4e2m1* test_output = out_data_rowwise_h.data() + tensor_offset / 2;
        const fp4e2m1* ref_output = out_data_rowwise_ref.data() + tensor_offset / 2;
        const fp4e2m1* test_output_t = out_data_colwise_h.data() + tensor_offset / 2;
        const fp4e2m1* ref_output_t = out_data_colwise_ref.data() + tensor_offset / 2;

        compare_nvfp4_tensors("output_" + std::to_string(t), test_output, ref_output,
                              static_cast<int>(rows), static_cast<int>(cols), atol, rtol);
        compare_nvfp4_tensors("output_t_" + std::to_string(t), test_output_t, ref_output_t,
                              static_cast<int>(cols), static_cast<int>(rows), atol, rtol);

        size_t scale_mismatches_num = 0;
        compare_scaling_factors<fp8e4m3>(
            "scales_" + std::to_string(t),
            out_scales_rowwise_h.data() + rowwise_scale_offsets[t],
            out_scales_rowwise_ref[t].data(),
            rows, rowwise_unpadded_blocks_X[t], rowwise_scales_stride[t], scale_mismatches_num);

        compare_scaling_factors<fp8e4m3>(
            "scales_t_" + std::to_string(t),
            out_scales_colwise_h.data() + colwise_scale_offsets[t],
            out_scales_colwise_ref[t].data(),
            cols, colwise_unpadded_blocks_X[t], colwise_scales_stride[t], scale_mismatches_num);
    }

    nvte_destroy_grouped_tensor(in_group_tensor);
    nvte_destroy_grouped_tensor(out_group_tensor);

    cudaFree(in_data_d);
    cudaFree(out_data_rowwise_d);
    cudaFree(out_data_colwise_d);
    cudaFree(out_scales_rowwise_d);
    cudaFree(out_scales_colwise_d);
    cudaFree(out_amax_rowwise_d);
    cudaFree(out_amax_colwise_d);
    cudaFree(first_dims_d);
    cudaFree(last_dims_d);
    cudaFree(offsets_d);
}

// {shape_representation, num_tensors, [logical_shape_M, logical_shape_K], [M_i], [K_i]}
std::vector<std::vector<size_t>> grouped_input_config = {
    {SAME_BOTH_DIMS,        1,      128,128},
    {SAME_BOTH_DIMS,        2,      256,128},
    {VARYING_FIRST_DIM,     2,      512,128,                    128,384},
    {VARYING_FIRST_DIM,     3,      1024,160,                   128,384,512},
    {VARYING_FIRST_DIM,     4,      1536,160,                   128,384,512,512},
    {VARYING_FIRST_DIM,     5,      4096,256,                   128,256,384,1024,2304},
    {VARYING_FIRST_DIM,     5,      8192,256,                   128,256,384,1024,2304},
    {VARYING_LAST_DIM,      3,      256,896,                    128,256,512},
    {VARYING_BOTH_DIMS,     2,      1,(128*128)+(256*256),      128,256,        128,256},
    {VARYING_BOTH_DIMS,     2,      1,(256*128)+(512*640),      256,512,        128,640},
};

}  // namespace

class GroupedFusedCastTransposeNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<std::vector<size_t>,        // Config
                transformer_engine::DType,
                bool>> {};

TEST_P(GroupedFusedCastTransposeNVFP4TestSuite, TestFusedCastTransposeNVFP4) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const std::vector<size_t> input_config = std::get<0>(GetParam());
    const DType input_type = std::get<1>(GetParam());
    const bool use_fast_math = std::get<2>(GetParam());

    const ShapeRepresentation shape_rep = static_cast<ShapeRepresentation>(input_config[0]);
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
        offsets[t + 1] = offsets[t] + first_dims[t] * last_dims[t];

        if (first_dims[t] % 128 != 0) {
            GTEST_SKIP();
        }

        if (shape_rep == VARYING_LAST_DIM || shape_rep == VARYING_BOTH_DIMS) {
            if (last_dims[t] % 128 != 0) {
                GTEST_SKIP();
            }
        }
    }

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
        performTest<InputType>(shape_rep, num_tensors, logical_shape,
                               first_dims, last_dims, offsets, use_fast_math);
    );
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    GroupedFusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(grouped_input_config),
        ::testing::Values(DType::kBFloat16),
        ::testing::Values(true, false)),
    [](const testing::TestParamInfo<GroupedFusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        std::string name = "CAST_ONLY";
        const std::vector<size_t> input = std::get<0>(info.param);

        switch (static_cast<ShapeRepresentation>(input[0])) {
            case ShapeRepresentation::SAME_BOTH_DIMS:       name += "_SAME_BOTH_DIMS"; break;
            case ShapeRepresentation::VARYING_FIRST_DIM:    name += "_VARYING_FIRST_DIM"; break;
            case ShapeRepresentation::VARYING_LAST_DIM:     name += "_VARYING_LAST_DIM"; break;
            case ShapeRepresentation::VARYING_BOTH_DIMS:    name += "_VARYING_BOTH_DIMS"; break;
        };

        name += "_N_" + std::to_string(input[1]);
        name += "_SHAPE_" + std::to_string(input[2]) + "X" + std::to_string(input[3]);
        name += "_" + test::typeName(std::get<1>(info.param));
        if (std::get<2>(info.param)) {
            name += "_FAST_SCALING";
        }
        return name;
    });
