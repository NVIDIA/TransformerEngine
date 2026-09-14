/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

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

// Compute the global encode scale factor for a given global amax
float compute_global_encode_scaling_factor_FP4(const float global_amax, const bool use_fast_math,
                                               const int e4m3_max = 448) {
  NVTE_CHECK(e4m3_max == 448 || e4m3_max == 256, "Unsupported NVFP4 E4M3 max.");
  const float fp8_max = static_cast<float>(e4m3_max);
  constexpr float fp4_max = 6.0f;       // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return the max normalized value
  const float max_norm_clamp = (use_fast_math && e4m3_max == 448)
                               ? Numeric_Traits<bf16>::maxNorm
                               : Numeric_Traits<float>::maxNorm;

  global_encode_scale = fminf(global_encode_scale, max_norm_clamp);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

struct NVFP4FourOverSixQuantization {
  fp8e4m3 scale_map4;
  fp8e4m3 scale_map6;
  float reciprocal_map4;
  float reciprocal_map6;
  fp4e2m1x2 quantized_map4;
  fp4e2m1x2 quantized_map6;
};

enum class NVFP4FourOverSixCandidate {
  Map4,
  Map6,
};

enum class NVFP4ScalingMode {
  Block1D,
  RowScaled1D,
  Block2D,
};

struct NVFP4FourOverSixTestConfig {
  NVTENVFP44Over6Mode mode = kNVTENVFP44Over6Disabled;
  int e4m3_max = 448;
  bool err_use_fast_math = false;
};

bool use_2d_quantization(const NVFP4ScalingMode scaling_mode) {
  return scaling_mode == NVFP4ScalingMode::Block2D;
}

NVFP4FourOverSixQuantization compute_4over6_quantization_scales(
    const float block_amax, const float global_encode_scale) {
  constexpr float fp4_max = 6.0f;
  constexpr float fp8_max = 448.0f;
  constexpr float scale_expansion_factor = 1.5f;
  const float base_sf_high_precision = block_amax / fp4_max * global_encode_scale;
  const float sf_high_precision_map4 =
      fminf(base_sf_high_precision * scale_expansion_factor, fp8_max);
  const float sf_high_precision_map6 = fminf(base_sf_high_precision, fp8_max);
  const fp8e4m3 scale_map4 = static_cast<fp8e4m3>(sf_high_precision_map4);
  const fp8e4m3 scale_map6 = static_cast<fp8e4m3>(sf_high_precision_map6);

  const float global_decode_scale = 1.0f / global_encode_scale;
  const float scale_map4_fp32 = static_cast<float>(scale_map4);
  const float reciprocal_map4 =
      fminf(1.0f / (scale_map4_fp32 * global_decode_scale), Numeric_Traits<float>::maxNorm);
  const float scale_map6_fp32 = static_cast<float>(scale_map6);
  const float reciprocal_map6 =
      fminf(1.0f / (scale_map6_fp32 * global_decode_scale), Numeric_Traits<float>::maxNorm);

  const float2 zero = {0.0f, 0.0f};
  return {
      scale_map4,
      scale_map6,
      reciprocal_map4,
      reciprocal_map6,
      fp4e2m1x2(zero),
      fp4e2m1x2(zero),
  };
}

fp8e4m3 select_4over6_scale(const NVFP4FourOverSixQuantization& quantization,
                            const NVFP4FourOverSixCandidate candidate) {
  if (candidate == NVFP4FourOverSixCandidate::Map4) {
    return quantization.scale_map4;
  }
  return quantization.scale_map6;
}

fp4e2m1x2 select_4over6_quantized_pair(const NVFP4FourOverSixQuantization& quantization,
                                       const NVFP4FourOverSixCandidate candidate) {
  if (candidate == NVFP4FourOverSixCandidate::Map4) {
    return quantization.quantized_map4;
  }
  return quantization.quantized_map6;
}

NVFP4FourOverSixQuantization quantize_4over6_pair(
    const float x, const float y, const NVFP4FourOverSixQuantization& quantization) {
  const float2 scaled_map4 = {x * quantization.reciprocal_map4,
                              y * quantization.reciprocal_map4};
  const fp4e2m1x2 quantized_map4(scaled_map4);

  const float2 scaled_map6 = {x * quantization.reciprocal_map6,
                              y * quantization.reciprocal_map6};
  const fp4e2m1x2 quantized_map6(scaled_map6);

  return {
      quantization.scale_map4,
      quantization.scale_map6,
      quantization.reciprocal_map4,
      quantization.reciprocal_map6,
      quantized_map4,
      quantized_map6,
  };
}

// 1D Scaling: Original implementation with 1x16 blocks
template <typename InputType>
void quantize_nvfp4_1d(float (*OP)(const float),
                       const InputType* const input,
                       fp4e2m1x2* const output,
                       fp8e4m3* const scales,
                       const size_t rows,
                       const size_t cols,
                       const size_t scales_stride,
                       const float global_amax,
                       const bool use_fast_math,
                       const bool use_4over6 = false,
                       const int e4m3_max = 448,
                       const NVFP4FourOverSixCandidate four_over_six_candidate =
                           NVFP4FourOverSixCandidate::Map6) {

    // Compute a global encoding/decoding scaling factor for all S_dec_b
    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax, use_fast_math,
                                                                 e4m3_max);

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
                const float act_elt = OP(input_elt);

                // Numerical truncation: after downcast to InputType (BF16/FP16), upcast it back to FP32
                const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                cache_buffer[cache_idx] = elt;
                block_amax = std::max(block_amax, std::abs(elt));
            }

            const size_t scale_idx = i * scales_stride + block_X;

            if (use_4over6) {
                const NVFP4FourOverSixQuantization quantization =
                    compute_4over6_quantization_scales(block_amax, S_enc);
                scales[scale_idx] = select_4over6_scale(quantization, four_over_six_candidate);

                for (size_t j = j_min; j < j_max; j += 2) {
                    const int idx_pair = (i * cols + j) / 2;
                    const int cache_idx_x = j - j_min;
                    const int cache_idx_y = cache_idx_x + 1;
                    const float cached_x = cache_buffer[cache_idx_x];
                    const float cached_y = cache_buffer[cache_idx_y];
                    const NVFP4FourOverSixQuantization pair_quantization =
                        quantize_4over6_pair(cached_x, cached_y, quantization);
                    output[idx_pair] =
                        select_4over6_quantized_pair(pair_quantization, four_over_six_candidate);
                }
                continue;
            }

            // Compute and store the per-block FP8 decode scale
            const float S_dec_b = block_amax * (S_enc * (1.0f / 6.0f));
            const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(fminf(S_dec_b, Numeric_Traits<float>::maxNorm));
            const float S_dec_b_fp32 = static_cast<float>(S_dec_b_fp8);

            // Compute "correct" per-block encoding scaling factor
            const float S_enc_b_fp8 = S_dec_b_fp32 == 0.f ? 0.f :
                fminf(1.0f / (S_dec_b_fp32 * (1.0f / S_enc)), Numeric_Traits<float>::maxNorm);

            scales[scale_idx] = S_dec_b_fp8;

            float scale_reciprocal = S_enc_b_fp8;
            if (use_fast_math) {
                // Numerical truncation to match GPU implementation, if mixed precision FMA instruction is used
                scale_reciprocal = static_cast<float>(static_cast<bf16>(scale_reciprocal));
            }

            for (size_t j = j_min; j < j_max; j += 2) {
                const int idx_pair = (i * cols + j) / 2;
                const int cache_idx_x = j - j_min;
                const int cache_idx_y = cache_idx_x + 1;
                const float cached_x = cache_buffer[cache_idx_x];
                const float cached_y = cache_buffer[cache_idx_y];
                const float scaled_elt_x = cached_x * scale_reciprocal;
                const float scaled_elt_y = cached_y * scale_reciprocal;
                const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                output[idx_pair] = casted_to_e2m1_pair;

                const double2 truncated_pair = cvt_fp4x2_to_double2(casted_to_e2m1_pair);
            }
        }
    }
}

// Compute 2D mathematical scaling factors (8x8 for 128x128 input)
template <typename InputType>
void compute_2d_mathematical_scales(float (*OP)(const float),
                                   const InputType* const input,
                                   const size_t rows,
                                   const size_t cols,
                                   const float global_amax,
                                   std::vector<std::vector<fp8e4m3>>& math_scales,
                                   const bool use_fast_math,
                                   const bool use_4over6 = false,
                                   const int e4m3_max = 448,
                                   const NVFP4FourOverSixCandidate four_over_six_candidate =
                                       NVFP4FourOverSixCandidate::Map6) {

    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax, use_fast_math,
                                                                 e4m3_max);
    constexpr size_t block_size_Y = 16;
    constexpr size_t block_size_X = 16;
    const size_t blocks_Y = divide_round_up(rows, block_size_Y);
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    math_scales.resize(blocks_Y, std::vector<fp8e4m3>(blocks_X));

    for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t i_min = block_Y * block_size_Y;
            const size_t i_max = std::min(i_min + block_size_Y, rows);
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = std::min(j_min + block_size_X, cols);

            // Find 2D block amax over entire 16x16 region
            float block_amax = 0.0f;
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; ++j) {
                    const size_t idx = i * cols + j;
                    const float input_elt = static_cast<float>(input[idx]);
                    const float act_elt = OP(input_elt);
                    const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                    block_amax = std::max(block_amax, std::abs(elt));
                }
            }

            // Compute E4M3 scaling factor for this 16x16 block
            if (use_4over6) {
                const NVFP4FourOverSixQuantization quantization =
                    compute_4over6_quantization_scales(block_amax, S_enc);
                math_scales[block_Y][block_X] =
                    select_4over6_scale(quantization, four_over_six_candidate);
            } else {
                const float S_dec_b = block_amax / 6.0f * S_enc;
                const fp8e4m3 S_dec_b_fp8_map6 = static_cast<fp8e4m3>(S_dec_b);
                math_scales[block_Y][block_X] = S_dec_b_fp8_map6;
            }
        }
    }
}

// 2D Scaling: NEW implementation with proper replication
template <typename InputType>
void quantize_nvfp4_2d(float (*OP)(const float),
                       const InputType* const input,
                       fp4e2m1x2* const output,
                       fp8e4m3* const scales,
                       const size_t rows,
                       const size_t cols,
                       const size_t scales_stride,
                       const float global_amax,
                       const bool use_fast_math,
                       const bool use_4over6 = false,
                       const int e4m3_max = 448,
                       const NVFP4FourOverSixCandidate four_over_six_candidate =
                           NVFP4FourOverSixCandidate::Map6) {

    // Step 1: Compute mathematical 8x8 scaling factors
    std::vector<std::vector<fp8e4m3>> math_scales;
    compute_2d_mathematical_scales(OP, input, rows, cols, global_amax, math_scales, use_fast_math,
                                   use_4over6, e4m3_max, four_over_six_candidate);

    const float S_enc = compute_global_encode_scaling_factor_FP4(global_amax, use_fast_math,
                                                                 e4m3_max);
    constexpr size_t block_size_Y = 16;
    constexpr size_t block_size_X = 16;
    const size_t blocks_Y = divide_round_up(rows, block_size_Y);
    const size_t blocks_X = divide_round_up(cols, block_size_X);

    // Step 2: Replicate scaling factors row-wise (128×8 storage) - only if scales is not nullptr
    if (scales != nullptr) {
        // Each of the 128 rows gets scaling factors from its corresponding 16×16 block
        for (size_t i = 0; i < rows; ++i) {
            const size_t block_Y = i / block_size_Y;
            for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
                const size_t scale_idx = i * scales_stride + block_X;
                scales[scale_idx] = math_scales[block_Y][block_X];
            }
        }
    }

    // Step 3: Apply quantization using the mathematical scaling factors
    std::array<std::array<float, block_size_X>, block_size_Y> cache_buffer;

    for (size_t block_Y = 0; block_Y < blocks_Y; ++block_Y) {
        for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
            const size_t i_min = block_Y * block_size_Y;
            const size_t i_max = std::min(i_min + block_size_Y, rows);
            const size_t j_min = block_X * block_size_X;
            const size_t j_max = std::min(j_min + block_size_X, cols);

            // Get the scaling factor for this block
            const float S_dec_b_fp8 = static_cast<float>(math_scales[block_Y][block_X]);
            const float S_enc_b_fp8 = S_dec_b_fp8 == 0.0f ? 0.0f : S_enc / S_dec_b_fp8;
            const float scale_reciprocal = S_enc_b_fp8;

            // Process and cache data for this 16x16 block
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; ++j) {
                    const size_t idx = i * cols + j;
                    const size_t cache_idx_y = i - i_min;
                    const size_t cache_idx_x = j - j_min;

                    const float input_elt = static_cast<float>(input[idx]);
                    const float act_elt = OP(input_elt);
                    const float elt = static_cast<float>(static_cast<InputType>(act_elt));
                    cache_buffer[cache_idx_y][cache_idx_x] = elt;
                }
            }

            // Apply scaling to all elements in this 16x16 block
            for (size_t i = i_min; i < i_max; ++i) {
                for (size_t j = j_min; j < j_max; j += 2) {
                    const int idx_pair = (i * cols + j) / 2;
                    const size_t cache_idx_y = i - i_min;
                    const size_t cache_idx_x1 = j - j_min;
                    const size_t cache_idx_x2 = std::min(cache_idx_x1 + 1, block_size_X - 1);

                    const float cached_x = cache_buffer[cache_idx_y][cache_idx_x1];
                    const float cached_y = ((j + 1) < j_max && cache_idx_x2 < block_size_X) ?
                                          cache_buffer[cache_idx_y][cache_idx_x2] : 0.0f;

                    const float scaled_elt_x = cached_x * scale_reciprocal;
                    const float scaled_elt_y = cached_y * scale_reciprocal;
                    const float2 scaled_elt_pair = {scaled_elt_x, scaled_elt_y};

                    fp4e2m1x2 casted_to_e2m1_pair(scaled_elt_pair);
                    output[idx_pair] = casted_to_e2m1_pair;
                }
            }
        }
    }
}

// Wrapper function that calls appropriate implementation based on 2D flag
template <typename InputType>
void quantize_nvfp4(float (*OP)(const float),
                    const InputType* const input,
                    fp4e2m1x2* const output,
                    fp8e4m3* const scales,
                    const size_t rows,
                    const size_t cols,
                    const size_t scales_stride,
                    const float global_amax,
                    const bool use_fast_math,
                    const bool use_2d_quantization = false,
                    const bool use_4over6 = false,
                    const int e4m3_max = 448,
                    const NVFP4FourOverSixCandidate four_over_six_candidate =
                        NVFP4FourOverSixCandidate::Map6) {
    if (use_2d_quantization) {
        quantize_nvfp4_2d(OP, input, output, scales, rows, cols, scales_stride, global_amax,
                          use_fast_math, use_4over6, e4m3_max, four_over_six_candidate);
    } else {
        quantize_nvfp4_1d(OP, input, output, scales, rows, cols, scales_stride, global_amax,
                          use_fast_math, use_4over6, e4m3_max, four_over_six_candidate);
    }
}

template <typename InputType>
void compute_ref(float (*OP)(const float),
                 const InputType* input,
                 fp4e2m1x2* output,
                 fp4e2m1x2* output_t,
                 fp8e4m3* scales,
                 fp8e4m3* scales_t,
                 const float* amax,
                 const size_t rows,
                 const size_t cols,
                 const size_t scales_stride,
                 const size_t scales_stride_t,
                 const bool use_fast_math,
                 const bool use_2d_quantization = false,
                 const bool row_scaled_nvfp4 = false,
                 const bool use_4over6 = false,
                 const int e4m3_max = 448,
                 const NVFP4FourOverSixCandidate four_over_six_candidate =
                     NVFP4FourOverSixCandidate::Map6)
{
    std::vector<InputType> input_t = create_transpose(input, rows, cols);
    NVTE_CHECK(!(use_2d_quantization && row_scaled_nvfp4),
               "2D quantization and row-scaling are not supported together.");

    // Ref impl for 2D quantization
    if (use_2d_quantization) {
        // Step 1: Compute mathematical 8×8 scaling factors
        std::vector<std::vector<fp8e4m3>> math_scales;
        compute_2d_mathematical_scales(OP, input, rows, cols, *amax, math_scales, use_fast_math,
                                       use_4over6, e4m3_max, four_over_six_candidate);

        constexpr size_t block_size_Y = 16;
        constexpr size_t block_size_X = 16;
        const size_t blocks_Y = divide_round_up(rows, block_size_Y);
        const size_t blocks_X = divide_round_up(cols, block_size_X);

        // Step 2: Generate scales (128×8) by replicating row-wise
        for (size_t i = 0; i < rows; ++i) {
            const size_t block_Y = i / block_size_Y;
            for (size_t block_X = 0; block_X < blocks_X; ++block_X) {
                const size_t scale_idx = i * scales_stride + block_X;
                scales[scale_idx] = math_scales[block_Y][block_X];
            }
        }

        // Step 3: Generate scales_t (128×8) with proper transposed block mapping
        for (size_t i = 0; i < cols; ++i) {  // cols = 128, which becomes rows of transposed data
            const size_t block_X_orig = i / block_size_X;  // i was column index in original, so maps to block_X
            for (size_t block_Y_new = 0; block_Y_new < blocks_Y; ++block_Y_new) {  // block in transposed coordinate
                const size_t scale_idx = i * scales_stride_t + block_Y_new;
                scales_t[scale_idx] = math_scales[block_Y_new][block_X_orig];
            }
        }

        // Step 4: Process quantized outputs using the same algorithm as quantize_nvfp4_2d
        // (This part processes the actual FP4 data using the mathematical scaling factors)
        quantize_nvfp4_2d(OP, input, output, nullptr, rows, cols, scales_stride, *amax,
                          use_fast_math, use_4over6, e4m3_max,
                          four_over_six_candidate); // scales already filled
        quantize_nvfp4_2d(OP, input_t.data(), output_t, nullptr, cols, rows, scales_stride_t, *amax,
                          use_fast_math, use_4over6, e4m3_max,
                          four_over_six_candidate); // scales_t already filled

        return;
    }

    // Ref impl for row-scaling
    if (row_scaled_nvfp4) {
        for (size_t row = 0; row < rows; ++row) {
            quantize_nvfp4(OP,
                           input + row * cols,
                           output + row * (cols / 2),
                           scales + row * scales_stride,
                           1,
                           cols,
                           scales_stride,
                           amax[row],
                           use_fast_math,
                           use_2d_quantization,
                           use_4over6,
                           e4m3_max,
                           four_over_six_candidate);
        }
        return;
    }

    // Ref impl for basic NVFP4
    quantize_nvfp4(OP, input, output, scales, rows, cols, scales_stride, *amax,
                   use_fast_math, use_2d_quantization, use_4over6, e4m3_max,
                   four_over_six_candidate);
    quantize_nvfp4(OP, input_t.data(), output_t, scales_t, cols, rows, scales_stride_t, *amax,
                   use_fast_math, use_2d_quantization, use_4over6, e4m3_max,
                   four_over_six_candidate);
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

    // Always report summary - either success or failure
    std::cout << "=== SUMMARY for tensor " << name << " ===" << std::endl;
    std::cout << "Total elements checked: " << (rows * cols) << std::endl;

    if (total_mismatches > 0) {
        std::cout << "STATUS: FAILED for output" << std::endl;
        std::cout << "Total mismatches found: " << total_mismatches << std::endl;
        std::cout << "Mismatch rate: " << (100.0 * total_mismatches) / (rows * cols) << "%" << std::endl;
        if (mismatch_messages.size() > max_mismatches_to_print) {
            std::cout << "... and " << (mismatch_messages.size() - max_mismatches_to_print)
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
}

// Optional: Function to dump tensor data to files for detailed analysis
void dump_nvfp4_tensor_data(const std::string& prefix,
                            const fp4e2m1 *test_data, const fp4e2m1 *ref_data,
                            const int rows, const int cols) {
    std::string test_file = prefix + "_test.txt";
    std::string ref_file = prefix + "_ref.txt";
    std::string diff_file = prefix + "_diff.txt";

    std::ofstream test_out(test_file);
    std::ofstream ref_out(ref_file);
    std::ofstream diff_out(diff_file);

    if (test_out.is_open() && ref_out.is_open() && diff_out.is_open()) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; j += 2) {
                const int idx = i * cols + j;
                double2 test_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx/2]));
                double2 ref_data_pair = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx/2]));

                for (int k = 0; k < 2; ++k) {
                    const double t = (k == 0 ? test_data_pair.x : test_data_pair.y);
                    const double r = (k == 0 ? ref_data_pair.x : ref_data_pair.y);
                    const int pos = idx + k;

                    test_out << "pos[" << pos << "] = " << t << std::endl;
                    ref_out << "pos[" << pos << "] = " << r << std::endl;
                    diff_out << "pos[" << pos << "] test=" << t << " ref=" << r
                            << " abs_diff=" << fabs(t - r)
                            << " rel_diff=" << (r == 0 ? 0.0 : fabs((t - r) / r)) << std::endl;
                }
            }
        }
        std::cout << "DEBUG: Dumped tensor data to files: " << test_file << ", " << ref_file << ", " << diff_file << std::endl;
    } else {
        std::cout << "WARNING: Could not open files for tensor data dump" << std::endl;
    }
}

void compareResults_nvfp4(Tensor &test,
                          const void *ref, const void *ref_t, const int rows, const int cols,
                          double atol = 1e-5, double rtol = 1e-8, bool if_on_gpus = true,
                          bool dump_data = false, bool compare_columnwise = true) {
    if (if_on_gpus) test.to_cpu();

    const fp4e2m1 *test_data = test.rowwise_cpu_dptr<fp4e2m1>();
    const fp4e2m1 *ref_data = reinterpret_cast<const fp4e2m1*>(ref);

    // Optionally dump tensor data to files for detailed analysis
    if (dump_data) {
        dump_nvfp4_tensor_data("output", test_data, ref_data, rows, cols);
    }

    compare_nvfp4_tensors("output", test_data, ref_data, rows, cols, atol, rtol);
    if (compare_columnwise) {
        const fp4e2m1 *test_data_t = test.columnwise_cpu_dptr<fp4e2m1>();
        const fp4e2m1 *ref_data_t = reinterpret_cast<const fp4e2m1*>(ref_t);
        if (dump_data) {
            dump_nvfp4_tensor_data("output_t", test_data_t, ref_data_t, cols, rows);
        }
        compare_nvfp4_tensors("output_t", test_data_t, ref_data_t, cols, rows, atol, rtol);
    }
}

template <typename T>
bool bitwise_equal(const T& x, const T& y) {
    const auto *x_bytes = reinterpret_cast<const unsigned char*>(&x);
    const auto *y_bytes = reinterpret_cast<const unsigned char*>(&y);
    for (size_t i = 0; i < sizeof(T); ++i) {
        if (x_bytes[i] != y_bytes[i]) {
            return false;
        }
    }
    return true;
}

bool nvfp4_output_block_matches(const fp4e2m1x2* const test_data,
                                const fp4e2m1x2* const ref_data,
                                const size_t row,
                                const size_t cols,
                                const size_t block_x) {
    constexpr size_t block_size_X = 16;
    const size_t j_min = block_x * block_size_X;
    const size_t j_max = std::min(j_min + block_size_X, cols);
    for (size_t j = j_min; j < j_max; j += 2) {
        const size_t idx_pair = (row * cols + j) / 2;
        if (!bitwise_equal(test_data[idx_pair], ref_data[idx_pair])) {
            return false;
        }
    }
    return true;
}

void compare_nvfp4_4over6_candidates(const std::string& name,
                                     const fp4e2m1* const test_data,
                                     const fp8e4m3* const test_scales,
                                     const fp4e2m1x2* const ref_data_map4,
                                     const fp8e4m3* const ref_scales_map4,
                                     const fp4e2m1x2* const ref_data_map6,
                                     const fp8e4m3* const ref_scales_map6,
                                     const size_t rows,
                                     const size_t cols,
                                     const size_t blocks_X,
                                     const size_t scales_stride) {
    constexpr int max_mismatches_to_print = 3;
    const auto* const test_data_pairs = reinterpret_cast<const fp4e2m1x2*>(test_data);
    size_t total_mismatches = 0;

    for (size_t row = 0; row < rows; ++row) {
        for (size_t block_x = 0; block_x < blocks_X; ++block_x) {
            const size_t scale_idx = row * scales_stride + block_x;
            const bool scale_matches_map4 =
                bitwise_equal(test_scales[scale_idx], ref_scales_map4[scale_idx]);
            const bool data_matches_map4 =
                nvfp4_output_block_matches(test_data_pairs, ref_data_map4, row, cols, block_x);
            const bool scale_matches_map6 =
                bitwise_equal(test_scales[scale_idx], ref_scales_map6[scale_idx]);
            const bool data_matches_map6 =
                nvfp4_output_block_matches(test_data_pairs, ref_data_map6, row, cols, block_x);

            if ((scale_matches_map4 && data_matches_map4) ||
                (scale_matches_map6 && data_matches_map6)) {
                continue;
            }

            ++total_mismatches;
            if (total_mismatches <= max_mismatches_to_print) {
                std::cout << "Error in tensor " << name << ": 4over6 block mismatch at row "
                          << row << ", block_x " << block_x
                          << ". The output did not match either map-to-4 or map-to-6 exactly."
                          << std::endl;
            }
        }
    }

    std::cout << "=== SUMMARY for tensor " << name << " ===" << std::endl;
    std::cout << "Total 4over6 blocks checked: " << (rows * blocks_X) << std::endl;
    if (total_mismatches > 0) {
        std::cout << "STATUS: FAILED for output" << std::endl;
        std::cout << "Total mismatched 4over6 blocks found: " << total_mismatches << std::endl;
        std::cout << "============================" << std::endl;
        GTEST_FAIL() << "Found " << total_mismatches << " 4over6 block mismatches in tensor "
                     << name;
    }

    std::cout << "STATUS: PASSED for output" << std::endl;
    std::cout << "Each 4over6 block matched either map-to-4 or map-to-6 exactly" << std::endl;
    std::cout << "============================" << std::endl;
}

void compare_rowwise_amax(Tensor &output, const std::vector<float> &ref_amax) {
    ASSERT_EQ(output.rowwise_amax_size(), ref_amax.size());
    const auto *amax_ptr = output.cpu_rowwise_amax_ptr<float>();
    const std::vector<float> test_amax_data(amax_ptr, amax_ptr + ref_amax.size());
    for (size_t row = 0; row < ref_amax.size(); ++row) {
        ASSERT_EQ(test_amax_data[row], ref_amax[row])
            << "Row-scaled amax mismatch at row " << row;
    }
}

template <typename InputType>
void performTest(float (*OP)(const float),
                 const std::vector<size_t>& shape,
                 const bool use_fast_math,
                 const NVFP4ScalingMode scaling_mode = NVFP4ScalingMode::Block1D,
                 const NVTENVFP44Over6Mode mode = kNVTENVFP44Over6Disabled,
                 const int e4m3_max = 448,
                 const bool use_4over6_err_use_fast_math = false) {
    using namespace test;
    const bool use_4over6 = mode != kNVTENVFP44Over6Disabled;

    if (use_4over6 && use_fast_math) {
        std::cout << "WARNING: Plain NVFP4 fast math is ignored for 4over6. "
                     "Use use_4over6_err_use_fast_math to test the 4over6 candidate "
                     "error fast-math path."
                  << std::endl;
    }

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = DType::kFloat4E2M1;

    const bool is_2d_quantization = use_2d_quantization(scaling_mode);
    const bool row_scaled_nvfp4 = scaling_mode == NVFP4ScalingMode::RowScaled1D;
    const bool rowwise = true;
    const bool columnwise = !row_scaled_nvfp4;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    // Use get_scale_tensor_dims for NVFP4 scale tensor dimensions
    // Now that CheckScaleTensorShape is fixed, this should work correctly
    const std::array<size_t,4> scale_dims = get_scale_tensor_dims(rows, cols, 1, 16);
    const std::array<size_t,4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16);

    const size_t unpadded_blocks_Y = scale_dims[0];
    const size_t unpadded_blocks_X = scale_dims[1];
    const size_t blocks_Y = scale_dims[2];
    const size_t blocks_X = scale_dims[3];
    const size_t scales_stride = blocks_X;

    const size_t unpadded_blocks_Y_t = scale_dims_t[0];
    const size_t unpadded_blocks_X_t = scale_dims_t[1];
    const size_t blocks_Y_t = scale_dims_t[2];
    const size_t blocks_X_t = scale_dims_t[3];
    const size_t scales_stride_t = blocks_X_t;

    Tensor input("input", shape, itype);
    Tensor output("output", shape, otype, rowwise, columnwise, NVTE_NVFP4_1D_SCALING);
    output.set_nvfp4_e4m3_max(e4m3_max);

    std::unique_ptr<fp4e2m1x2[]> ref_output   = std::make_unique<fp4e2m1x2[]>(rows * (cols / 2));
    std::unique_ptr<fp4e2m1x2[]> ref_output_t = std::make_unique<fp4e2m1x2[]>(cols * (rows / 2));
    std::unique_ptr<fp8e4m3[]> ref_scales     = std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);
    std::unique_ptr<fp8e4m3[]> ref_scales_t   = std::make_unique<fp8e4m3[]>(blocks_Y_t * blocks_X_t);
    std::unique_ptr<fp4e2m1x2[]> ref_output_map6;
    std::unique_ptr<fp4e2m1x2[]> ref_output_t_map6;
    std::unique_ptr<fp8e4m3[]> ref_scales_map6;
    std::unique_ptr<fp8e4m3[]> ref_scales_t_map6;

    fillCase<fp32>(&input, InputsFillCase::uniform);

    if (use_4over6 && row_scaled_nvfp4) {
        const float target_row_amax = static_cast<float>(e4m3_max) * 6.0f * 8.0f;
        auto *input_vals = input.rowwise_cpu_dptr<InputType>();
        for (size_t row = 0; row < rows; ++row) {
            float row_amax = 0.0f;
            size_t max_col = 0;
            for (size_t col = 0; col < cols; ++col) {
                const float val = static_cast<float>(input_vals[row * cols + col]);
                const float abs_val = fabsf(val);
                if (abs_val > row_amax) {
                    row_amax = abs_val;
                    max_col = col;
                }
            }

            if (row_amax == 0.0f) {
                continue;
            }

            const float row_scale = target_row_amax / row_amax;
            for (size_t col = 0; col < cols; ++col) {
                float scaled = static_cast<float>(input_vals[row * cols + col]) * row_scale;
                scaled = fminf(fmaxf(scaled, -target_row_amax), target_row_amax);
                input_vals[row * cols + col] = static_cast<InputType>(scaled);
            }

            const float max_val = static_cast<float>(input_vals[row * cols + max_col]);
            input_vals[row * cols + max_col] =
                static_cast<InputType>(max_val < 0.0f ? -target_row_amax : target_row_amax);
        }
        input.from_cpu();
    }

    // Compute 2nd stage NVFP4 scaling factor
    std::vector<float> ref_amax;
    if (row_scaled_nvfp4) {
        // Compute per-row amaxes
        const auto *input_vals = input.rowwise_cpu_dptr<InputType>();
        for (size_t row = 0; row < rows; ++row){
            float row_amax = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                row_amax = fmaxf(row_amax, fabsf(static_cast<float>(input_vals[row * cols + col])));
            }
            ref_amax.push_back(row_amax);
        }

        // Update tensor
        // Note: No need to update amax like standard NVFP4, amaxes
        // are computed during quantization.
        output.set_row_scaled_nvfp4(row_scaled_nvfp4);
    } else {
        // Golden value of amax chosen to make the 2nd-stage scaling mantissa zero and avoid rounding issues
        if (use_4over6) {
            ref_amax.assign(1, static_cast<float>(e4m3_max) * 6.0f * 8.0f);
        } else {
            ref_amax.assign(1, 448.0f * 6.0f * 8.0f);
        }

        // Update tensor
        if (rowwise) {
            std::copy(ref_amax.begin(), ref_amax.end(), output.cpu_rowwise_amax_ptr<float>());
        }
        if (columnwise) {
            std::copy(ref_amax.begin(), ref_amax.end(), output.cpu_columnwise_amax_ptr<float>());
        }
        output.from_cpu();
    }

    if (use_4over6) {
        ref_output_map6 = std::make_unique<fp4e2m1x2[]>(rows * (cols / 2));
        ref_output_t_map6 = std::make_unique<fp4e2m1x2[]>(cols * (rows / 2));
        ref_scales_map6 = std::make_unique<fp8e4m3[]>(blocks_Y * blocks_X);
        ref_scales_t_map6 = std::make_unique<fp8e4m3[]>(blocks_Y_t * blocks_X_t);

        compute_ref<InputType>(OP,
                               input.rowwise_cpu_dptr<InputType>(),
                               ref_output.get(),
                               ref_output_t.get(),
                               ref_scales.get(),
                               ref_scales_t.get(),
                               ref_amax.data(),
                               rows,
                               cols,
                               scales_stride,
                               scales_stride_t,
                               use_fast_math,
                               is_2d_quantization,
                               row_scaled_nvfp4,
                               use_4over6,
                               e4m3_max,
                               NVFP4FourOverSixCandidate::Map4);
        compute_ref<InputType>(OP,
                               input.rowwise_cpu_dptr<InputType>(),
                               ref_output_map6.get(),
                               ref_output_t_map6.get(),
                               ref_scales_map6.get(),
                               ref_scales_t_map6.get(),
                               ref_amax.data(),
                               rows,
                               cols,
                               scales_stride,
                               scales_stride_t,
                               use_fast_math,
                               is_2d_quantization,
                               row_scaled_nvfp4,
                               use_4over6,
                               e4m3_max,
                               NVFP4FourOverSixCandidate::Map6);
    } else {
        compute_ref<InputType>(OP,
                               input.rowwise_cpu_dptr<InputType>(),
                               ref_output.get(),
                               ref_output_t.get(),
                               ref_scales.get(),
                               ref_scales_t.get(),
                               ref_amax.data(),
                               rows,
                               cols,
                               scales_stride,
                               scales_stride_t,
                               use_fast_math,
                               is_2d_quantization,
                               row_scaled_nvfp4,
                               use_4over6);
    }

    // Initialize stochastic rounding
    Tensor rng_state("rng_state", std::vector<size_t>{2}, DType::kInt64);
    rng_state.rowwise_cpu_dptr<int64_t>()[0] = 123;  // rng_seed
    rng_state.rowwise_cpu_dptr<int64_t>()[1] = 321;  // rng_sequence
    rng_state.from_cpu();

    // Quantization options
    QuantizationConfigWrapper quant_config;
    quant_config.set_use_fast_math(use_fast_math && !use_4over6);
    quant_config.set_stochastic_rounding(false);
    quant_config.set_rng_state(rng_state.data());
    quant_config.set_nvfp4_2d_quantization(is_2d_quantization);
    quant_config.set_nvfp4_4over6_mode(mode);
    quant_config.set_nvfp4_4over6_err_use_fast_math(use_4over6 && use_4over6_err_use_fast_math);

    // Call appropriate function based on operation type
    // Activation functions take 3 parameters (input, output, stream)
    // nvte_quantize_v2 takes 4 parameters (input, output, quant_config, stream)
    if (OP == &gelu) {
        nvte_gelu(input.data(), output.data(), 0);
    } else if (OP == &silu) {
        nvte_silu(input.data(), output.data(), 0);
    } else if (OP == &relu) {
        nvte_relu(input.data(), output.data(), 0);
    } else if (OP == &qgelu) {
        nvte_qgelu(input.data(), output.data(), 0);
    } else if (OP == &srelu) {
        nvte_srelu(input.data(), output.data(), 0);
    } else {
        nvte_quantize_v2(input.data(), output.data(), quant_config, 0);
    }

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("DEBUG: CUDA error detected: %s\n", cudaGetErrorString(err));
    }
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    const double atol = 1.0E-6;
    const double rtol = 1.0E-6;

    if (use_4over6) {
        output.to_cpu();
        compare_nvfp4_4over6_candidates("output",
                                        output.rowwise_cpu_dptr<fp4e2m1>(),
                                        output.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                        ref_output.get(),
                                        ref_scales.get(),
                                        ref_output_map6.get(),
                                        ref_scales_map6.get(),
                                        rows,
                                        cols,
                                        unpadded_blocks_X,
                                        scales_stride);
        if (!row_scaled_nvfp4) {
            compare_nvfp4_4over6_candidates("output_t",
                                            output.columnwise_cpu_dptr<fp4e2m1>(),
                                            output.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                            ref_output_t.get(),
                                            ref_scales_t.get(),
                                            ref_output_t_map6.get(),
                                            ref_scales_t_map6.get(),
                                            cols,
                                            rows,
                                            unpadded_blocks_X_t,
                                            scales_stride_t);
        }
    } else {
        // Set dump_data=true to enable dumping tensor data to files for analysis
        compareResults_nvfp4(output, ref_output.get(), ref_output_t.get(), rows, cols, atol, rtol,
                             true, false, !row_scaled_nvfp4);

        size_t scale_mismatches_num = 0;
        compare_scaling_factors<fp8e4m3>("scales", output.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                          ref_scales.get(),
                                          unpadded_blocks_Y, unpadded_blocks_X, scales_stride,
                                          scale_mismatches_num);

        if (!row_scaled_nvfp4) {
            compare_scaling_factors<fp8e4m3>("scales_t",
                                              output.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                              ref_scales_t.get(),
                                              unpadded_blocks_Y_t, unpadded_blocks_X_t,
                                              scales_stride_t, scale_mismatches_num);
        }
    }

    compare_rowwise_amax(output, ref_amax);
}

// Columnwise-only 2D NVFP4 must match the columnwise half of both-directions output
template <typename InputType>
void performTestColumnwiseOnly2D(const std::vector<size_t>& shape) {
    using namespace test;

    DType itype = TypeInfo<InputType>::dtype;
    DType otype = DType::kFloat4E2M1;

    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    // Columnwise (transposed) scale-tensor dimensions.
    const std::array<size_t, 4> scale_dims_t = get_scale_tensor_dims(cols, rows, 1, 16);
    const size_t unpadded_blocks_Y_t = scale_dims_t[0];
    const size_t unpadded_blocks_X_t = scale_dims_t[1];
    const size_t scales_stride_t = scale_dims_t[3];

    Tensor input("input", shape, itype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    // Golden amax chosen so the 2nd-stage scaling mantissa is zero (avoids rounding noise).
    const float golden_amax = 448.0f * 6.0f * 8.0f;

    // Reference: both directions produced in a single kernel call (rowwise + columnwise).
    Tensor output_both("output_both", shape, otype, /*rowwise=*/true, /*columnwise=*/true,
                       NVTE_NVFP4_1D_SCALING);
    output_both.cpu_rowwise_amax_ptr<float>()[0] = golden_amax;
    output_both.cpu_columnwise_amax_ptr<float>()[0] = golden_amax;
    output_both.from_cpu();

    // System under test: columnwise-only output (no rowwise data allocated).
    Tensor output_col("output_col", shape, otype, /*rowwise=*/false, /*columnwise=*/true,
                      NVTE_NVFP4_1D_SCALING);
    output_col.cpu_columnwise_amax_ptr<float>()[0] = golden_amax;
    output_col.from_cpu();

    QuantizationConfigWrapper quant_config;
    quant_config.set_stochastic_rounding(false);
    quant_config.set_nvfp4_2d_quantization(true);

    nvte_quantize_v2(input.data(), output_both.data(), quant_config, 0);
    nvte_quantize_v2(input.data(), output_col.data(), quant_config, 0);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    output_both.to_cpu();
    output_col.to_cpu();

    // Columnwise FP4 data must match bitwise (atol = rtol = 0).
    compare_nvfp4_tensors("columnwise_only_data",
                          output_col.columnwise_cpu_dptr<fp4e2m1>(),
                          output_both.columnwise_cpu_dptr<fp4e2m1>(),
                          static_cast<int>(cols), static_cast<int>(rows),
                          /*atol=*/0.0, /*rtol=*/0.0);

    // Columnwise scale factors must match over the in-bounds region.
    size_t scale_mismatches = 0;
    compare_scaling_factors<fp8e4m3>("columnwise_only_scales",
                                     output_col.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     output_both.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     unpadded_blocks_Y_t, unpadded_blocks_X_t, scales_stride_t,
                                     scale_mismatches);
    ASSERT_EQ(scale_mismatches, 0u);

    // The columnwise-only tensor must not allocate rowwise output.
    EXPECT_FALSE(output_col.rowwise());
}

std::vector<std::vector<size_t>> tensor_dims = {
    {32, 32},
    {32, 64},
    {64, 32},
    {64, 96},
    {128, 128},
    {256, 256},
    {512, 512},
    {1024, 1024},
    {2048, 2048},
    {128, 256},
    {8192, 128},
    {2048, 160},
    {8, 32, 1024},
    {16, 8, 4, 512},
    {1024, 16384},
    {4096, 13312},
};

// Only the Identity activation is currently supported.
std::vector<ActivationType> Activation_types = {
    ActivationType::Identity
};

// Element-level FP4 code differences between two compact NVFP4 buffers of the
// same logical (rows, cols) shape (cols must be even; FP4 is packed 2/byte).
inline size_t count_nvfp4_code_mismatches(const fp4e2m1* test_data, const fp4e2m1* ref_data,
                                          int rows, int cols) {
    size_t mismatches = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; j += 2) {
            const int idx = i * cols + j;
            const double2 t = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&test_data[idx / 2]));
            const double2 r = cvt_fp4x2_to_double2(*reinterpret_cast<const fp4e2m1x2*>(&ref_data[idx / 2]));
            if (t.x != r.x) ++mismatches;
            if (t.y != r.y) ++mismatches;
        }
    }
    return mismatches;
}

// Row-scaled transpose NVFP4 cast: emits rowwise (per-row amax) and columnwise
// (per-col amax) NVFP4 outputs in one pass. Cross-validated against the merged
// row-scaled 1D kernel (PR #2931, NVFP4ScalingMode::RowScaled1D): rowwise vs
// RowScaled1D(input), columnwise vs RowScaled1D(input^T). Amaxes and FP8 block
// scales must match bitwise; FP4 codes may differ only at rounding-midpoint
// ties (< 0.1%). bf16 input, rows/cols multiples of 128.
template <typename InputType>
void performTestRowScaledTranspose(const std::vector<size_t>& shape) {
    using namespace test;

    const DType itype = TypeInfo<InputType>::dtype;
    const DType otype = DType::kFloat4E2M1;
    const size_t rows = first_dimension(shape);
    const size_t cols = last_dimension(shape);

    Tensor input("input", shape, itype);
    fillCase<fp32>(&input, InputsFillCase::uniform);

    // System under test: row-scaled NVFP4 with both directions requested in one
    // call. The transpose kernel is selected by the generic nvte_quantize_v2
    // dispatch when a row-scaled tensor (set_row_scaled_nvfp4) also allocates a
    // columnwise output -- no dedicated config flag.
    Tensor output("output", shape, otype, /*rowwise=*/true, /*columnwise=*/true,
                  NVTE_NVFP4_1D_SCALING);
    // Marking the tensor row-scaled with a columnwise output allocated selects the
    // transpose kernel and sizes the per-row (rowwise) and per-col (columnwise)
    // amax vectors (default is a single scalar amax).
    output.set_row_scaled_nvfp4(true);
    QuantizationConfigWrapper quant_config;
    quant_config.set_stochastic_rounding(false);
    nvte_quantize_v2(input.data(), output.data(), quant_config, 0);

    // Reference (rowwise direction): trusted row-scaled 1D kernel on the input.
    Tensor ref_row("ref_row", shape, otype, /*rowwise=*/true, /*columnwise=*/false,
                   NVTE_NVFP4_1D_SCALING);
    ref_row.set_row_scaled_nvfp4(true);
    QuantizationConfigWrapper ref_config;
    ref_config.set_stochastic_rounding(false);
    nvte_quantize_v2(input.data(), ref_row.data(), ref_config, 0);

    // Reference (columnwise direction): trusted row-scaled 1D kernel on input^T.
    input.to_cpu();
    std::vector<InputType> input_t_host =
        create_transpose(input.rowwise_cpu_dptr<InputType>(), rows, cols);
    const std::vector<size_t> shape_t = {cols, rows};
    Tensor input_t("input_t", shape_t, itype);
    std::copy(input_t_host.begin(), input_t_host.end(), input_t.rowwise_cpu_dptr<InputType>());
    input_t.from_cpu();
    Tensor ref_col("ref_col", shape_t, otype, /*rowwise=*/true, /*columnwise=*/false,
                   NVTE_NVFP4_1D_SCALING);
    ref_col.set_row_scaled_nvfp4(true);
    nvte_quantize_v2(input_t.data(), ref_col.data(), ref_config, 0);

    cudaDeviceSynchronize();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << cudaGetErrorString(cudaGetLastError());

    output.to_cpu();
    ref_row.to_cpu();
    ref_col.to_cpu();

    // FP4 codes must match the trusted kernel except at rounding-midpoint ties.
    // Cap the tolerated disagreement well below what a real bug would produce.
    constexpr double kMaxCodeMismatchRate = 5e-3;  // 0.5% (observed < 0.1%)
    const size_t row_mismatches = count_nvfp4_code_mismatches(
        output.rowwise_cpu_dptr<fp4e2m1>(), ref_row.rowwise_cpu_dptr<fp4e2m1>(),
        static_cast<int>(rows), static_cast<int>(cols));
    EXPECT_LT(static_cast<double>(row_mismatches) / static_cast<double>(rows * cols),
              kMaxCodeMismatchRate)
        << "rowwise FP4 disagreement " << row_mismatches << "/" << (rows * cols);
    const size_t col_mismatches = count_nvfp4_code_mismatches(
        output.columnwise_cpu_dptr<fp4e2m1>(), ref_col.rowwise_cpu_dptr<fp4e2m1>(),
        static_cast<int>(cols), static_cast<int>(rows));
    EXPECT_LT(static_cast<double>(col_mismatches) / static_cast<double>(cols * rows),
              kMaxCodeMismatchRate)
        << "columnwise FP4 disagreement " << col_mismatches << "/" << (cols * rows);

    // FP8 e4m3 block scale factors must match (compact layout on both sides).
    const std::array<size_t, 4> sd = get_scale_tensor_dims(rows, cols, 1, 16);
    const std::array<size_t, 4> sd_t = get_scale_tensor_dims(cols, rows, 1, 16);
    size_t scale_mismatches = 0;
    compare_scaling_factors<fp8e4m3>("rowwise_scales",
                                     output.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     ref_row.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     sd[0], sd[1], sd[3], scale_mismatches);
    compare_scaling_factors<fp8e4m3>("columnwise_scales",
                                     output.columnwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     ref_col.rowwise_cpu_scale_inv_ptr<fp8e4m3>(),
                                     sd_t[0], sd_t[1], sd_t[3], scale_mismatches);
    ASSERT_EQ(scale_mismatches, 0u);

    // Per-row / per-col amaxes must match the reference amaxes exactly.
    ASSERT_EQ(output.rowwise_amax_size(), rows);
    const float* row_amax = output.cpu_rowwise_amax_ptr<float>();
    const float* ref_row_amax = ref_row.cpu_rowwise_amax_ptr<float>();
    for (size_t i = 0; i < rows; ++i) {
        ASSERT_EQ(row_amax[i], ref_row_amax[i]) << "rowwise amax mismatch at row " << i;
    }
    const float* col_amax = output.cpu_columnwise_amax_ptr<float>();
    const float* ref_col_amax = ref_col.cpu_rowwise_amax_ptr<float>();
    for (size_t j = 0; j < cols; ++j) {
        ASSERT_EQ(col_amax[j], ref_col_amax[j]) << "columnwise amax mismatch at col " << j;
    }
}

// Row-scaled transpose requires 128-aligned rows and cols (bf16 only).
std::vector<std::vector<size_t>> row_scaled_transpose_dims = {
    {128, 128},
    {256, 256},
    {128, 256},
    {256, 512},
    {512, 512},
    {384, 1024},
    {2048, 256},
};

}  // namespace

class FusedCastTransposeNVFP4TestSuite : public ::testing::TestWithParam
    <std::tuple<ActivationType,
                std::vector<size_t>,
                transformer_engine::DType,
                bool,
                NVFP4ScalingMode,
                NVFP4FourOverSixTestConfig>> {};

TEST_P(FusedCastTransposeNVFP4TestSuite, TestFusedCastTransposeNVFP4) {
    // Skip tests for pre-Blackwell architectures
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }

    using namespace transformer_engine;
    using namespace test;

    const ActivationType Act_type = std::get<0>(GetParam());
    const auto tensor_dims = std::get<1>(GetParam());
    const DType input_type = std::get<2>(GetParam());
    const bool use_fast_math = std::get<3>(GetParam());
    const NVFP4ScalingMode scaling_mode = std::get<4>(GetParam());
    const NVFP4FourOverSixTestConfig config = std::get<5>(GetParam());

    // Skip tests if the input tensor is 1D
    if (tensor_dims.size() < 2) {
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
        performTest<InputType>(OP, tensor_dims, use_fast_math, scaling_mode, config.mode,
                               config.e4m3_max,
                               config.err_use_fast_math);
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

std::string to_string(const NVFP4ScalingMode scaling_mode) {
    switch (scaling_mode) {
        case NVFP4ScalingMode::Block1D:     return "";
        case NVFP4ScalingMode::RowScaled1D: return "XROW_SCALED";
        case NVFP4ScalingMode::Block2D:     return "X2D";
        default: return "";
    }
}

std::string test_name(const FusedCastTransposeNVFP4TestSuite::ParamType& param) {
    std::string name = to_string(std::get<0>(param));
    const auto& shape = std::get<1>(param);
    for (const auto& s: shape) {
        name += "X" + std::to_string(s);
    }
    name += "X" + test::typeName(std::get<2>(param));
    if (std::get<3>(param)) {
        name += "X_FAST_SCALING";
    }
    name += to_string(std::get<4>(param));
    const NVFP4FourOverSixTestConfig& config = std::get<5>(param);
    if (config.mode != kNVTENVFP44Over6Disabled) {
        name += "X4OVER6";
        if (config.e4m3_max == 448) {
            name += "XE4M3_MAX_448";
        } else {
            name += "XE4M3_MAX_256";
        }
        if (config.mode == kNVTENVFP44Over6MinMSE) {
            name += "XMSE";
        } else if (config.mode == kNVTENVFP44Over6MinMAE) {
            name += "XMAE";
        } else {
            name += "XINVALID_MODE";
        }
        if (config.err_use_fast_math) {
            name += "XERR_USE_FAST_MATH";
        }
    }
    return name;
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),           // activation_type
        ::testing::ValuesIn(tensor_dims),                // tensor_dims
        ::testing::Values(DType::kBFloat16),             // input_type
        ::testing::Values(false),                       // use_fast_math
        ::testing::Values(NVFP4ScalingMode::Block1D),   // scaling_mode
        ::testing::Values(NVFP4FourOverSixTestConfig{})), // four_over_six_config
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        return test_name(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTestRowScaled,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),               // activation_type
        ::testing::ValuesIn(tensor_dims),                    // tensor_dims
        ::testing::Values(DType::kBFloat16, DType::kFloat32), // input_type
        ::testing::Values(false),                           // use_fast_math
        ::testing::Values(NVFP4ScalingMode::RowScaled1D),   // scaling_mode
        ::testing::Values(NVFP4FourOverSixTestConfig{})),   // four_over_six_config
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        return test_name(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest4Over6,
    FusedCastTransposeNVFP4TestSuite,
    ::testing::Combine(
        ::testing::ValuesIn(Activation_types),           // activation_type
        ::testing::ValuesIn(tensor_dims),                // tensor_dims
        ::testing::Values(DType::kBFloat16, DType::kFloat32), // input_type
        ::testing::Values(false),                       // use_fast_math
        ::testing::Values(NVFP4ScalingMode::Block1D,
                          NVFP4ScalingMode::RowScaled1D,
                          NVFP4ScalingMode::Block2D),   // scaling_mode
        ::testing::Values(
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 448, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 448, true},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 448, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 448, true},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 256, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMAE, 256, true},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 256, false},
            NVFP4FourOverSixTestConfig{kNVTENVFP44Over6MinMSE, 256, true})), // four_over_six_config
    [](const testing::TestParamInfo<FusedCastTransposeNVFP4TestSuite::ParamType>& info) {
        return test_name(info.param);
    });

class CastNVFP4ColumnwiseOnly2DTestSuite : public ::testing::TestWithParam<std::vector<size_t>> {};

TEST_P(CastNVFP4ColumnwiseOnly2DTestSuite, ColumnwiseOnlyMatchesBothDirections) {
    // The optimized NVFP4 quantize-transpose kernel requires Blackwell.
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }
    performTestColumnwiseOnly2D<bf16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastNVFP4ColumnwiseOnly2DTestSuite,
    // Include rectangular 128-multiple shapes to guard transposed data/scale indexing.
    ::testing::Values(
        std::vector<size_t>{128, 128},
        std::vector<size_t>{256, 512},
        std::vector<size_t>{384, 1024},
        std::vector<size_t>{2048, 256}),
    [](const testing::TestParamInfo<CastNVFP4ColumnwiseOnly2DTestSuite::ParamType>& info) {
        std::string name;
        for (const auto& s : info.param) {
            name += "X" + std::to_string(s);
        }
        return name;
    });

class CastNVFP4RowScaledTransposeTestSuite : public ::testing::TestWithParam<std::vector<size_t>> {};

TEST_P(CastNVFP4RowScaledTransposeTestSuite, MatchesRowScaledReference) {
    // The row-scaled transpose NVFP4 cast kernel requires Blackwell.
    if (getDeviceComputeCapability() < blackwellComputeCapability) {
        GTEST_SKIP();
    }
    performTestRowScaledTranspose<bf16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    CastNVFP4RowScaledTransposeTestSuite,
    ::testing::ValuesIn(row_scaled_transpose_dims),
    [](const testing::TestParamInfo<CastNVFP4RowScaledTransposeTestSuite::ParamType>& info) {
        std::string name;
        for (const auto& s : info.param) {
            name += "X" + std::to_string(s);
        }
        return name;
    });
