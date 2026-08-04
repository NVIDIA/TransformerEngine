/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <transformer_engine/activation.h>
#include "../test_common.h"
#include "transformer_engine/transformer_engine.h"

using namespace transformer_engine;
using namespace test;

namespace {

// Only the two grouped layouts with a uniform last dim are supported: the output is
// [T, F] with F shared by every expert.
enum ShapeRepresentation {
    SAME_BOTH_DIMS    = 0,
    VARYING_FIRST_DIM = 1
};

constexpr size_t SCALE_DIM_Y = 32;

// Host mirror of mxfp8::swizzle::gemm_swizzled_scale_idx. The FC2 wgrad GEMM reads this
// operand transposed, so its scale matrix is the [cols, rows/32] transpose of the compact
// one, tiled 128x4:
// https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
size_t gemm_swizzled_scale_idx(const size_t i, const size_t j, const size_t num_tiles_X) {
    constexpr size_t TILE_DIM_X = 4;
    constexpr size_t TILE_DIM_Y = 128;
    constexpr size_t TILE_SIZE = TILE_DIM_X * TILE_DIM_Y;
    const size_t tile_idx_X = j / TILE_DIM_X;
    const size_t tile_idx_Y = i / TILE_DIM_Y;
    const size_t idx_in_tile_X = j % TILE_DIM_X;
    const size_t idx_in_tile_Y = i % TILE_DIM_Y;
    size_t idx = (tile_idx_Y * num_tiles_X + tile_idx_X) * TILE_SIZE;
    idx += (idx_in_tile_Y % 32) * 16 + (idx_in_tile_Y / 32) * 4 + idx_in_tile_X;
    return idx;
}

/**
 * Reference for a single expert: (silu(act) * gate) * prob, then columnwise MXFP8.
 *   input  : [rows, 2 * cols], last dim = [act | gate]
 *   prob   : [rows], per-token router weight in the input dtype
 *   output : [rows, cols]
 *   scales : this expert's block of e8m0 exponents, compact or GEMM-swizzled
 */
template <typename InputType, typename OutputType>
void compute_ref(const InputType* input,
                 const InputType* prob,
                 OutputType* output,
                 fp8e8m0* scales,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride,
                 const bool with_gemm_swizzled_scales) {
    const size_t blocks_Y = divide_round_up(rows, SCALE_DIM_Y);
    // Number of 4-wide tiles along the swizzled matrix's column axis (which is rows / 32).
    const size_t swizzled_tiles_X = divide_round_up(rows, scale_tensor_alignment_Y_rowwise);
    const size_t input_stride = 2 * cols;

    #pragma omp parallel proc_bind(spread)
    {
        // Buffer to cache the weighted activation of one 32-element block
        std::vector<float> cache(SCALE_DIM_Y);
        #pragma omp for schedule(static)
        for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y) {
            const size_t i_min = block_Y * SCALE_DIM_Y;
            const size_t i_max = std::min(rows, i_min + SCALE_DIM_Y);

            for (size_t j = 0; j < cols; ++j) {
                float block_amax = 0.0f;
                for (size_t i = i_min; i < i_max; ++i) {
                    const float act_elt = static_cast<float>(input[i * input_stride + j]);
                    const float gate_elt = static_cast<float>(input[i * input_stride + cols + j]);
                    const float prob_elt = static_cast<float>(prob[i]);
                    // Numerical truncation: the kernel rounds the weighted activation back
                    // through InputType before quantizing, so the reference must too.
                    const float elt = static_cast<float>(
                        static_cast<InputType>(silu(act_elt) * gate_elt * prob_elt));
                    cache[i - i_min] = elt;
                    block_amax = std::max(block_amax, std::abs(elt));
                }

                const fp8e8m0 biased_exponent =
                    float_to_e8m0(block_amax * Quantized_Limits<OutputType>::max_reciprocal());
                const size_t scale_idx = with_gemm_swizzled_scales
                                         ? gemm_swizzled_scale_idx(j, block_Y, swizzled_tiles_X)
                                         : block_Y * scales_stride + j;
                scales[scale_idx] = biased_exponent;

                const float scale_reciprocal = exp2f_rcp(biased_exponent);
                for (size_t i = i_min; i < i_max; ++i) {
                    output[i * cols + j] =
                        static_cast<OutputType>(cache[i - i_min] * scale_reciprocal);
                }
            }
        }
    }
}

template <typename T>
void compare_quantized_elts(const std::string& name,
                            const T* ref_data,
                            const T* test_data,
                            const size_t numel,
                            const size_t tolerable_mismatches_limit) {
    size_t mismatches_num = 0;
    int64_t first_mismatch_idx = -1;

    for (size_t i = 0; i < numel; ++i) {
        const double t = static_cast<double>(test_data[i]);
        const double r = static_cast<double>(ref_data[i]);
        if (t == r) {
            continue;
        }
        // Tolerate round-to-nearest picking the other side of the real value: the kernel's
        // silu intrinsic and the CPU reference can disagree in the last ULP, which flips
        // codes that sit on a rounding boundary.
        const double mean = (t + r) / 2;
        const double mean_p = mean >= 0 ? mean * (1 + 1e-6) : mean * (1 - 1e-6);
        const double mean_m = mean >= 0 ? mean * (1 - 1e-6) : mean * (1 + 1e-6);
        const double cast_mean_p = static_cast<double>(static_cast<T>(mean_p));
        const double cast_mean_m = static_cast<double>(static_cast<T>(mean_m));
        if (cast_mean_m == std::min(t, r) && cast_mean_p == std::max(t, r)) {
            continue;
        }

        mismatches_num++;
        if (first_mismatch_idx == -1) {
            first_mismatch_idx = static_cast<int64_t>(i);
        }
        if (mismatches_num > tolerable_mismatches_limit) {
            GTEST_FAIL() << mismatches_num << " mismatch(es) in " << name
                         << ", more than the tolerable limit of "
                         << tolerable_mismatches_limit << "." << std::endl
                         << "First mismatch at " << first_mismatch_idx << ": "
                         << static_cast<double>(test_data[first_mismatch_idx]) << " vs "
                         << static_cast<double>(ref_data[first_mismatch_idx]);
        }
    }
}

template <typename InputType, typename OutputType>
void performTest(const ShapeRepresentation shape_rep,
                 const size_t num_tensors,
                 const std::vector<size_t>& rows_per_tensor,
                 const size_t F,
                 const bool with_gemm_swizzled_scales,
                 const bool expect_rejection) {
    using namespace test;

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = TypeInfo<OutputType>::dtype;

    size_t T = 0;
    for (size_t t = 0; t < num_tensors; ++t) {
        T += rows_per_tensor[t];
    }

    const size_t in_elts = T * 2 * F;
    const size_t out_elts = T * F;
    const size_t scales_stride = round_up_to_nearest_multiple(F, scale_tensor_alignment_X_colwise);

    // Element offsets into the [T, F] output, and e8m0 offsets into the scale buffer. Both
    // layouts share these offsets: per-expert row counts are 128-aligned, so a compact and
    // a swizzled block for the same expert occupy the same number of scales.
    std::vector<int64_t> data_offsets(num_tensors + 1, 0);
    std::vector<size_t> scale_offsets(num_tensors + 1, 0);
    std::vector<int64_t> first_dims(num_tensors, 0);
    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t M = rows_per_tensor[t];
        first_dims[t] = static_cast<int64_t>(M);
        data_offsets[t + 1] = data_offsets[t] + static_cast<int64_t>(M * F);
        const size_t blocks_Y = round_up_to_nearest_multiple(divide_round_up(M, SCALE_DIM_Y),
                                                             scale_tensor_alignment_Y_colwise);
        scale_offsets[t + 1] = scale_offsets[t] + blocks_Y * scales_stride;
    }
    const size_t sfs_num = scale_offsets[num_tensors];

    std::mt19937 gen;
    std::uniform_real_distribution<> dis(-2.0, 1.0);
    std::vector<InputType> in_data(in_elts);
    for (size_t i = 0; i < in_elts; ++i) {
        in_data[i] = static_cast<InputType>(dis(gen));
    }

    // prob follows TE's cuDNN fc1_prob_tensor convention: model (input) dtype.
    Tensor prob("prob", std::vector<size_t>{T}, itype);
    fillUniform(&prob);

    const size_t in_data_size = in_elts * sizeof(InputType);
    const size_t out_data_size = out_elts * sizeof(OutputType);
    const size_t scales_size = sfs_num * sizeof(fp8e8m0);

    auto in_data_d = cuda_alloc<InputType>(in_data_size);
    auto out_data_d = cuda_alloc<OutputType>(out_data_size);
    auto out_scales_d = cuda_alloc<fp8e8m0>(scales_size);
    auto first_dims_d = cuda_alloc<int64_t>(num_tensors * sizeof(int64_t));
    auto offsets_d = cuda_alloc<int64_t>((num_tensors + 1) * sizeof(int64_t));

    NVTE_CHECK_CUDA(cudaMemcpy(in_data_d.get(), in_data.data(), in_data_size,
                               cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(first_dims_d.get(), first_dims.data(),
                               num_tensors * sizeof(int64_t), cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(offsets_d.get(), data_offsets.data(),
                               (num_tensors + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemset(out_data_d.get(), 0, out_data_size));
    NVTE_CHECK_CUDA(cudaMemset(out_scales_d.get(), 0, scales_size));

    std::vector<size_t> in_logical_shape_vec = {T, 2 * F};
    std::vector<size_t> out_logical_shape_vec = {T, F};
    std::vector<size_t> scales_shape_vec = {sfs_num};
    NVTEShape in_logical_shape = nvte_make_shape(in_logical_shape_vec.data(),
                                                in_logical_shape_vec.size());
    NVTEShape out_logical_shape = nvte_make_shape(out_logical_shape_vec.data(),
                                                 out_logical_shape_vec.size());
    NVTEShape scales_shape = nvte_make_shape(scales_shape_vec.data(), scales_shape_vec.size());

    NVTEShape first_dims_shape;
    NVTEShape offsets_shape;
    first_dims_shape.ndim = 1;
    offsets_shape.ndim = 1;
    first_dims_shape.data[0] = num_tensors;
    offsets_shape.data[0] = num_tensors + 1;

    NVTEGroupedTensor in_group_tensor =
        nvte_create_grouped_tensor(NVTE_DELAYED_TENSOR_SCALING, num_tensors, in_logical_shape);
    NVTEGroupedTensor out_group_tensor =
        nvte_create_grouped_tensor(NVTE_MXFP8_1D_SCALING, num_tensors, out_logical_shape);

    NVTEBasicTensor in_data_tensor = {in_data_d.get(), static_cast<NVTEDType>(itype),
                                     in_logical_shape};
    nvte_set_grouped_tensor_param(in_group_tensor, NVTEGroupedTensorParam::kNVTEGroupedRowwiseData,
                                  &in_data_tensor, sizeof(in_data_tensor));

    // Columnwise only: the MoE FC2 weight-gradient GEMM is the sole consumer.
    NVTEBasicTensor out_data_tensor = {out_data_d.get(), static_cast<NVTEDType>(otype),
                                      out_logical_shape};
    NVTEBasicTensor out_scales_tensor = {out_scales_d.get(), NVTEDType::kNVTEFloat8E8M0,
                                        scales_shape};
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedColumnwiseData,
                                  &out_data_tensor, sizeof(out_data_tensor));
    nvte_set_grouped_tensor_param(out_group_tensor,
                                  NVTEGroupedTensorParam::kNVTEGroupedColumnwiseScaleInv,
                                  &out_scales_tensor, sizeof(out_scales_tensor));

    // The launcher derives the grouped layout from the output metadata: leaving first_dims
    // unset means SAME_BOTH_DIMS, setting it means VARYING_FIRST_DIM.
    if (shape_rep == VARYING_FIRST_DIM) {
        NVTEBasicTensor first_dims_tensor = {first_dims_d.get(), kNVTEInt64, first_dims_shape};
        NVTEBasicTensor offsets_tensor = {offsets_d.get(), kNVTEInt64, offsets_shape};
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedFirstDims,
                                      &first_dims_tensor, sizeof(first_dims_tensor));
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedTensorOffsets,
                                      &offsets_tensor, sizeof(offsets_tensor));
    }

    if (with_gemm_swizzled_scales) {
        const uint8_t flag = 1;
        nvte_set_grouped_tensor_param(out_group_tensor,
                                      NVTEGroupedTensorParam::kNVTEGroupedWithGEMMSwizzledScales,
                                      &flag, sizeof(flag));
    }

    if (expect_rejection) {
        EXPECT_THROW(nvte_group_swiglu_quantize(in_group_tensor, prob.data(), out_group_tensor, 0),
                     std::runtime_error);
        nvte_destroy_grouped_tensor(in_group_tensor);
        nvte_destroy_grouped_tensor(out_group_tensor);
        return;
    }

    // Reference (CPU), one expert at a time.
    std::vector<OutputType> out_data_ref(out_elts, static_cast<OutputType>(0.0f));
    std::vector<fp8e8m0> out_scales_ref(sfs_num, static_cast<fp8e8m0>(0));
    const InputType* const prob_ptr = prob.rowwise_cpu_dptr<InputType>();
    size_t row_base = 0;
    for (size_t t = 0; t < num_tensors; ++t) {
        const size_t M = rows_per_tensor[t];
        if (M == 0) {
            continue;
        }
        // data_offsets are F-based, so the [T, 2F] input offset is twice as large.
        compute_ref<InputType, OutputType>(in_data.data() + 2 * data_offsets[t],
                                           prob_ptr + row_base,
                                           out_data_ref.data() + data_offsets[t],
                                           out_scales_ref.data() + scale_offsets[t],
                                           M, F, scales_stride, with_gemm_swizzled_scales);
        row_base += M;
    }

    // GPU
    nvte_group_swiglu_quantize(in_group_tensor, prob.data(), out_group_tensor, 0);
    NVTE_CHECK_CUDA(cudaDeviceSynchronize());
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    std::vector<OutputType> out_data_h(out_elts);
    std::vector<fp8e8m0> out_scales_h(sfs_num);
    NVTE_CHECK_CUDA(cudaMemcpy(out_data_h.data(), out_data_d.get(), out_data_size,
                               cudaMemcpyDeviceToHost));
    NVTE_CHECK_CUDA(cudaMemcpy(out_scales_h.data(), out_scales_d.get(), scales_size,
                               cudaMemcpyDeviceToHost));

    // A last-ULP silu difference can push a block amax onto the next e8m0 exponent, so a
    // few scale mismatches are tolerated; every element of such a block is then allowed to
    // differ as well.
    const size_t scale_diff_abs_tolerance = 0;
    const double abs_tolerable_mismatches_limit = 1.0;
    const double rel_tolerable_mismatches_limit = 1.0e-4;

    size_t mismatches_scales = 0;
    compare_scaling_factors("colwise_scales", out_scales_h.data(), out_scales_ref.data(),
                            1, sfs_num, sfs_num, mismatches_scales, scale_diff_abs_tolerance,
                            abs_tolerable_mismatches_limit, rel_tolerable_mismatches_limit);

    compare_quantized_elts<OutputType>("colwise_output", out_data_ref.data(), out_data_h.data(),
                                       out_elts, 32 * mismatches_scales);

    nvte_destroy_grouped_tensor(in_group_tensor);
    nvte_destroy_grouped_tensor(out_group_tensor);
}

// {shape_representation, num_tensors, F, rows_of_each_expert...}
// Per-expert row counts are multiples of 128, which the kernel requires.
std::vector<std::vector<size_t>> input_configs = {
    {SAME_BOTH_DIMS,    1,  128,    128},
    {SAME_BOTH_DIMS,    2,  256,    128, 128},
    {VARYING_FIRST_DIM, 2,  128,    128, 384},
    {VARYING_FIRST_DIM, 3,  256,    128, 384, 512},
    // Empty expert in the middle must not terminate the persistent work loop.
    {VARYING_FIRST_DIM, 4,  256,    128, 384, 0, 512},
    // F is not a multiple of the 128-wide chunk, exercising the partial-tile bounds check.
    {VARYING_FIRST_DIM, 4,  160,    128, 384, 512, 512},
    {VARYING_FIRST_DIM, 5,  512,    128, 256, 384, 1024, 2304},
};

std::vector<std::vector<size_t>> input_configs_small = {
    {SAME_BOTH_DIMS,    1,  128,    128},
    {VARYING_FIRST_DIM, 3,  256,    128, 384, 512},
    {VARYING_FIRST_DIM, 4,  160,    128, 384, 512, 512},
};

}  // namespace

class GroupedSwigluQuantizeMXFP8TestSuite : public ::testing::TestWithParam
    <std::tuple<std::vector<size_t>,        // Config
                bool,                       // GEMM-swizzled scales
                transformer_engine::DType,  // InputType
                transformer_engine::DType   // OutputType
                >> {};

TEST_P(GroupedSwigluQuantizeMXFP8TestSuite, Test) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const std::vector<size_t> config = std::get<0>(GetParam());
    const bool with_gemm_swizzled_scales = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const DType output_type = std::get<3>(GetParam());

    const ShapeRepresentation shape_rep = static_cast<ShapeRepresentation>(config[0]);
    const size_t num_tensors = config[1];
    const size_t F = config[2];
    const std::vector<size_t> rows_per_tensor(config.begin() + 3, config.end());

    // The swizzled layout tiles the scale matrix 128-wide along F, and each expert owns a
    // block sized by its own token count. Configs that violate either requirement must be
    // rejected by the launcher rather than silently produce a wrong layout.
    const bool expect_rejection = with_gemm_swizzled_scales
                                  && ((F % 128 != 0)
                                      || (num_tensors > 1 && shape_rep == SAME_BOTH_DIMS));

    TRANSFORMER_ENGINE_TYPE_SWITCH_FP16_FP32_ONLY(input_type, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8_ONLY(output_type, OutputType,
            performTest<InputType, OutputType>(shape_rep, num_tensors, rows_per_tensor, F,
                                               with_gemm_swizzled_scales, expect_rejection);
        );
    );
}

namespace {

std::string MakeGroupedSwigluQuantizeMXFP8TestName(
    const testing::TestParamInfo<GroupedSwigluQuantizeMXFP8TestSuite::ParamType>& info) {
    const std::vector<size_t> config = std::get<0>(info.param);

    std::string name;
    switch (static_cast<ShapeRepresentation>(config[0])) {
        case ShapeRepresentation::SAME_BOTH_DIMS:    name = "SAME_BOTH_DIMS";    break;
        case ShapeRepresentation::VARYING_FIRST_DIM: name = "VARYING_FIRST_DIM"; break;
    }

    name += "_N_" + std::to_string(config[1]);
    name += "_F_" + std::to_string(config[2]);
    for (size_t i = 3; i < config.size(); ++i) {
        name += (i == 3 ? "_ROWS_" : "X") + std::to_string(config[i]);
    }

    name += std::get<1>(info.param) ? "_SWIZZLED" : "_COMPACT";
    name += "_" + test::typeName(std::get<2>(info.param)) +
            "_" + test::typeName(std::get<3>(info.param));

    return name;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest_GroupedSwigluQuantizeMXFP8_Shapes,
    GroupedSwigluQuantizeMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(input_configs),
        ::testing::Values(false, true),
        ::testing::Values(DType::kBFloat16),
        ::testing::Values(DType::kFloat8E4M3)),
    MakeGroupedSwigluQuantizeMXFP8TestName);

INSTANTIATE_TEST_SUITE_P(
    OperatorTest_GroupedSwigluQuantizeMXFP8_Dtypes,
    GroupedSwigluQuantizeMXFP8TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(input_configs_small),
        ::testing::Values(false, true),
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E4M3, DType::kFloat8E5M2)),
    MakeGroupedSwigluQuantizeMXFP8TestName);
