/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_4over6_nvfp4.cuh
 *  \brief Helpers used by NVFP4 4over6 quantization.
 *
 *  4over6 evaluates two TE-style NVFP4 encodings for each 1x16 block. The
 *  map-to-6 candidate uses the normal block scale. The map-to-4 candidate uses
 *  a 1.5x expanded block scale, which maps the FP4 value 4 to the same dynamic
 *  range as FP4 value 6. The selected candidate is the one with lower MSE after
 *  dequantizing back to the original input domain; ties select map-to-6.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace core {

#if FP4_TYPE_SUPPORTED

__device__ __forceinline__ void compute_4over6_decoding_scaling_factors(
    const float block_amax, const float S_enc, nvfp4_scale_t &S_dec_b_fp8_map4,
    nvfp4_scale_t &S_dec_b_fp8_map6) {
  constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f
  const float sf_high_precision = block_amax / fp4_max * S_enc;
  S_dec_b_fp8_map4 = static_cast<nvfp4_scale_t>(sf_high_precision * 1.5f);
  S_dec_b_fp8_map6 = static_cast<nvfp4_scale_t>(sf_high_precision);
}

template <typename scaling_coeff_type>
struct QuantizationScales4Over6 {
  nvfp4_scale_t S_dec_b_fp8_map4;
  nvfp4_scale_t S_dec_b_fp8_map6;
  scaling_coeff_type SFcoefficient_map4;
  scaling_coeff_type SFcoefficient_map6;
};

template <typename scaling_coeff_type>
__device__ __forceinline__ scaling_coeff_type
compute_4over6_nvfp4_scaling_coefficient(const nvfp4_scale_t S_dec_block, const float S_enc) {
  if constexpr (std::is_same_v<scaling_coeff_type, float>) {
    const float S_dec = 1.0f / S_enc;
    const float scale_rcp =
        fminf(1.0f / (static_cast<float>(S_dec_block) * S_dec), detail::TypeExtrema<float>::max);
    return scale_rcp;
  } else if constexpr (std::is_same_v<scaling_coeff_type, bf16>) {
    const float scale_rcp =
        fminf(S_enc / static_cast<float>(S_dec_block), detail::TypeExtrema<bf16>::max);
    return static_cast<bf16>(scale_rcp);
  } else {
    NVTE_DEVICE_ERROR("Unsupported scaling-factor type. Only FP32 and BF16 are supported.");
    return scaling_coeff_type{};
  }
}

template <typename scaling_coeff_type>
__device__ __forceinline__ QuantizationScales4Over6<scaling_coeff_type>
compute_4over6_nvfp4_quantization_scaling_factors(const float block_amax, const float S_enc) {
  QuantizationScales4Over6<scaling_coeff_type> scaling_factors;
  compute_4over6_decoding_scaling_factors(block_amax, S_enc, scaling_factors.S_dec_b_fp8_map4,
                                          scaling_factors.S_dec_b_fp8_map6);
  scaling_factors.SFcoefficient_map4 = compute_4over6_nvfp4_scaling_coefficient<scaling_coeff_type>(
      scaling_factors.S_dec_b_fp8_map4, S_enc);
  scaling_factors.SFcoefficient_map6 = compute_4over6_nvfp4_scaling_coefficient<scaling_coeff_type>(
      scaling_factors.S_dec_b_fp8_map6, S_enc);
  return scaling_factors;
}

__device__ __forceinline__ QuantizationScales4Over6<float>
compute_4over6_fp4_encode_quantization_scaling_factors(const float block_amax,
                                                       const float global_encode_scale,
                                                       const float global_decode_scale) {
  QuantizationScales4Over6<float> scaling_factors;
  compute_4over6_decoding_scaling_factors(block_amax, global_encode_scale,
                                          scaling_factors.S_dec_b_fp8_map4,
                                          scaling_factors.S_dec_b_fp8_map6);
  scaling_factors.SFcoefficient_map4 =
      fminf(1.0f / (static_cast<float>(scaling_factors.S_dec_b_fp8_map4) * global_decode_scale),
            detail::TypeExtrema<float>::max);
  scaling_factors.SFcoefficient_map6 =
      fminf(1.0f / (static_cast<float>(scaling_factors.S_dec_b_fp8_map6) * global_decode_scale),
            detail::TypeExtrema<float>::max);
  return scaling_factors;
}

template <bool USE_FAST_MATH = false>
__device__ __forceinline__ uint32_t cvt_fp32_to_fp4_8x_with_mse_rn(const float (&x)[8],
                                                                   const float block_scale_inverse,
                                                                   const nvfp4_scale_t S_dec_b_fp8,
                                                                   const float global_amax,
                                                                   float *err) {
  uint32_t out = 0;
  uint32_t out_dequant_1 = 0;
  uint32_t out_dequant_2 = 0;
  uint32_t out_dequant_3 = 0;
  uint32_t out_dequant_4 = 0;

  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    float x_scaled[8];
    if constexpr (USE_FAST_MATH) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        x_scaled[i] = x[i] * block_scale_inverse;
      }
    } else {
      x_scaled[0] = __fmul_rn(x[0], block_scale_inverse);
      x_scaled[1] = __fmul_rn(x[1], block_scale_inverse);
      x_scaled[2] = __fmul_rn(x[2], block_scale_inverse);
      x_scaled[3] = __fmul_rn(x[3], block_scale_inverse);
      x_scaled[4] = __fmul_rn(x[4], block_scale_inverse);
      x_scaled[5] = __fmul_rn(x[5], block_scale_inverse);
      x_scaled[6] = __fmul_rn(x[6], block_scale_inverse);
      x_scaled[7] = __fmul_rn(x[7], block_scale_inverse);
    }

    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
        "}"
        : "=r"(out), "=r"(out_dequant_1), "=r"(out_dequant_2), "=r"(out_dequant_3),
          "=r"(out_dequant_4)
        : "f"(x_scaled[0]), "f"(x_scaled[1]), "f"(x_scaled[2]), "f"(x_scaled[3]), "f"(x_scaled[4]),
          "f"(x_scaled[5]), "f"(x_scaled[6]), "f"(x_scaled[7]));

    const uint16_t out_dequant_1_hi = (out_dequant_1 >> 16) & 0xFFFF;
    const uint16_t out_dequant_1_lo = out_dequant_1 & 0xFFFF;
    const uint16_t out_dequant_2_hi = (out_dequant_2 >> 16) & 0xFFFF;
    const uint16_t out_dequant_2_lo = out_dequant_2 & 0xFFFF;
    const uint16_t out_dequant_3_hi = (out_dequant_3 >> 16) & 0xFFFF;
    const uint16_t out_dequant_3_lo = out_dequant_3 & 0xFFFF;
    const uint16_t out_dequant_4_hi = (out_dequant_4 >> 16) & 0xFFFF;
    const uint16_t out_dequant_4_lo = out_dequant_4 & 0xFFFF;

    constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f
    constexpr float fp8_4over6_max = 256.0f;
    constexpr float mse_denom = fp4_max * fp8_4over6_max;
    const float sf = static_cast<float>(S_dec_b_fp8);
    if constexpr (USE_FAST_MATH) {
      const float dequant[8] = {
          __half2float(__ushort_as_half(out_dequant_1_lo)),
          __half2float(__ushort_as_half(out_dequant_1_hi)),
          __half2float(__ushort_as_half(out_dequant_2_lo)),
          __half2float(__ushort_as_half(out_dequant_2_hi)),
          __half2float(__ushort_as_half(out_dequant_3_lo)),
          __half2float(__ushort_as_half(out_dequant_3_hi)),
          __half2float(__ushort_as_half(out_dequant_4_lo)),
          __half2float(__ushort_as_half(out_dequant_4_hi)),
      };
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float val = dequant[i] * sf * global_amax / mse_denom;
        const float diff = val - x[i];
        *err += diff * diff;
      }
    } else {
      const float val0 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_1_lo)), sf), global_amax),
          mse_denom);
      const float val1 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_1_hi)), sf), global_amax),
          mse_denom);
      const float val2 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_2_lo)), sf), global_amax),
          mse_denom);
      const float val3 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_2_hi)), sf), global_amax),
          mse_denom);
      const float val4 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_3_lo)), sf), global_amax),
          mse_denom);
      const float val5 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_3_hi)), sf), global_amax),
          mse_denom);
      const float val6 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_4_lo)), sf), global_amax),
          mse_denom);
      const float val7 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_4_hi)), sf), global_amax),
          mse_denom);

      const float diff0 = __fsub_rn(val0, x[0]);
      const float diff1 = __fsub_rn(val1, x[1]);
      const float diff2 = __fsub_rn(val2, x[2]);
      const float diff3 = __fsub_rn(val3, x[3]);
      const float diff4 = __fsub_rn(val4, x[4]);
      const float diff5 = __fsub_rn(val5, x[5]);
      const float diff6 = __fsub_rn(val6, x[6]);
      const float diff7 = __fsub_rn(val7, x[7]);

      *err = __fadd_rn(*err, __fmul_rn(diff0, diff0));
      *err = __fadd_rn(*err, __fmul_rn(diff1, diff1));
      *err = __fadd_rn(*err, __fmul_rn(diff2, diff2));
      *err = __fadd_rn(*err, __fmul_rn(diff3, diff3));
      *err = __fadd_rn(*err, __fmul_rn(diff4, diff4));
      *err = __fadd_rn(*err, __fmul_rn(diff5, diff5));
      *err = __fadd_rn(*err, __fmul_rn(diff6, diff6));
      *err = __fadd_rn(*err, __fmul_rn(diff7, diff7));
    }
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }

  return out;
}

template <bool USE_FAST_MATH, bool REVERSE_PACK_ORDER = false, typename scaling_coeff_type>
__device__ __forceinline__ void quantize_4over6_16x(
    const float (&first_half)[8], const float (&second_half)[8],
    const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors, const float global_amax,
    float &err_map4, float &err_map6, uint32_t (&rOut_map4)[2], uint32_t (&rOut_map6)[2]) {
  if constexpr (REVERSE_PACK_ORDER) {
    rOut_map4[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
    rOut_map4[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
  } else {
    rOut_map4[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
    rOut_map4[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
  }
}

template <bool USE_FAST_MATH, bool REVERSE_PACK_ORDER = false, typename scaling_coeff_type>
__device__ __forceinline__ void quantize_4over6_16x(
    const float (&first_half)[8], const float (&second_half)[8],
    const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors, const float global_amax,
    nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float err_map4 = 0.0f;
  float err_map6 = 0.0f;
  __align__(8) uint32_t rOut_map4[2];
  __align__(8) uint32_t rOut_map6[2];

  quantize_4over6_16x<USE_FAST_MATH, REVERSE_PACK_ORDER>(first_half, second_half, scaling_factors,
                                                         global_amax, err_map4, err_map6, rOut_map4,
                                                         rOut_map6);

  if (err_map4 < err_map6) {
    S_dec_b_fp8 = scaling_factors.S_dec_b_fp8_map4;
    rOut[0] = rOut_map4[0];
    rOut[1] = rOut_map4[1];
  } else {
    S_dec_b_fp8 = scaling_factors.S_dec_b_fp8_map6;
    rOut[0] = rOut_map6[0];
    rOut[1] = rOut_map6[1];
  }
}

struct QuantizationCandidates4Over6 {
  float err_map4;
  float err_map6;
  uint32_t rOut_map4[2];
  uint32_t rOut_map6[2];
};

template <bool USE_FAST_MATH, typename scaling_coeff_type>
__device__ __forceinline__ void quantize_4over6_candidates_16x(
    const float (&x)[16], const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors,
    const float global_amax, QuantizationCandidates4Over6 &candidates) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = x[i];
    second_half[i] = x[i + 8];
  }

  candidates.err_map4 = 0.0f;
  candidates.err_map6 = 0.0f;
  quantize_4over6_16x<USE_FAST_MATH>(first_half, second_half, scaling_factors, global_amax,
                                     candidates.err_map4, candidates.err_map6, candidates.rOut_map4,
                                     candidates.rOut_map6);
}

template <size_t BLOCK_DIM, size_t BLOCKS_PER_TILE_Y, size_t BLOCKS_PER_TILE_X>
__device__ __forceinline__ void reduce_4over6_2d_block_selection(
    const size_t block_in_tile_y, const size_t reduce_thread_idx, const float global_encode_scale,
    const float global_decode_scale,
    float (&block_amax_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1],
    float (&err_map4_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM],
    float (&err_map6_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM],
    uint8_t (&pick_map4_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X],
    nvfp4_scale_t (&selected_scale_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X]) {
  if (reduce_thread_idx < BLOCKS_PER_TILE_X) {
    const size_t reduce_block_x = reduce_thread_idx;
    float block_err_map4 = 0.0f;
    float block_err_map6 = 0.0f;
#pragma unroll
    for (int i = 0; i < BLOCK_DIM; ++i) {
      block_err_map4 += err_map4_matrix[block_in_tile_y][reduce_block_x][i];
      block_err_map6 += err_map6_matrix[block_in_tile_y][reduce_block_x][i];
    }

    const auto scaling_factors = compute_4over6_fp4_encode_quantization_scaling_factors(
        block_amax_matrix[block_in_tile_y][reduce_block_x], global_encode_scale,
        global_decode_scale);
    if (block_err_map4 < block_err_map6) {
      pick_map4_matrix[block_in_tile_y][reduce_block_x] = 1;
      selected_scale_matrix[block_in_tile_y][reduce_block_x] = scaling_factors.S_dec_b_fp8_map4;
    } else {
      pick_map4_matrix[block_in_tile_y][reduce_block_x] = 0;
      selected_scale_matrix[block_in_tile_y][reduce_block_x] = scaling_factors.S_dec_b_fp8_map6;
    }
  }
}

template <size_t BLOCK_DIM, size_t BLOCKS_PER_TILE_Y, size_t BLOCKS_PER_TILE_X>
__device__ __forceinline__ void record_and_reduce_4over6_2d_block_selection(
    const float block_amax, const float global_encode_scale, const float global_decode_scale,
    const size_t block_in_tile_y, const size_t block_in_tile_x, const size_t participant_idx,
    float (&err_map4_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM],
    float (&err_map6_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM],
    uint8_t (&pick_map4_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X],
    nvfp4_scale_t (&selected_scale_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X],
    const QuantizationCandidates4Over6 &candidates) {
  err_map4_matrix[block_in_tile_y][block_in_tile_x][participant_idx] = candidates.err_map4;
  err_map6_matrix[block_in_tile_y][block_in_tile_x][participant_idx] = candidates.err_map6;
  __syncthreads();

  if (participant_idx == 0) {
    float block_err_map4 = 0.0f;
    float block_err_map6 = 0.0f;
#pragma unroll
    for (int i = 0; i < BLOCK_DIM; ++i) {
      block_err_map4 += err_map4_matrix[block_in_tile_y][block_in_tile_x][i];
      block_err_map6 += err_map6_matrix[block_in_tile_y][block_in_tile_x][i];
    }

    const auto scaling_factors = compute_4over6_fp4_encode_quantization_scaling_factors(
        block_amax, global_encode_scale, global_decode_scale);
    if (block_err_map4 < block_err_map6) {
      pick_map4_matrix[block_in_tile_y][block_in_tile_x] = 1;
      selected_scale_matrix[block_in_tile_y][block_in_tile_x] = scaling_factors.S_dec_b_fp8_map4;
    } else {
      pick_map4_matrix[block_in_tile_y][block_in_tile_x] = 0;
      selected_scale_matrix[block_in_tile_y][block_in_tile_x] = scaling_factors.S_dec_b_fp8_map6;
    }
  }
  __syncthreads();
}

template <bool USE_FAST_MATH, size_t BLOCK_DIM, size_t BLOCKS_PER_TILE_Y, size_t BLOCKS_PER_TILE_X>
__device__ __forceinline__ void quantize_4over6_2d_block_candidate(
    const float (&x)[16], const float block_amax, const float global_encode_scale,
    const float global_decode_scale, const float global_amax, const size_t block_in_tile_y,
    const size_t block_in_tile_x, const size_t participant_idx, const size_t reduce_thread_idx,
    float (&block_amax_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X + 1],
    float (&err_map4_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM],
    float (&err_map6_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM],
    uint8_t (&pick_map4_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X],
    nvfp4_scale_t (&selected_scale_matrix)[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X],
    QuantizationCandidates4Over6 &candidates) {
  const auto scaling_factors = compute_4over6_fp4_encode_quantization_scaling_factors(
      block_amax, global_encode_scale, global_decode_scale);
  quantize_4over6_candidates_16x<USE_FAST_MATH>(x, scaling_factors, global_amax, candidates);

  err_map4_matrix[block_in_tile_y][block_in_tile_x][participant_idx] = candidates.err_map4;
  err_map6_matrix[block_in_tile_y][block_in_tile_x][participant_idx] = candidates.err_map6;
  __syncthreads();

  reduce_4over6_2d_block_selection<BLOCK_DIM, BLOCKS_PER_TILE_Y, BLOCKS_PER_TILE_X>(
      block_in_tile_y, reduce_thread_idx, global_encode_scale, global_decode_scale,
      block_amax_matrix, err_map4_matrix, err_map6_matrix, pick_map4_matrix, selected_scale_matrix);
  __syncthreads();
}

__device__ __forceinline__ uint32_t *selected_4over6_packed(
    const bool pick_map4, QuantizationCandidates4Over6 &candidates) {
  if (pick_map4) {
    return candidates.rOut_map4;
  }
  return candidates.rOut_map6;
}

template <typename output_type>
__device__ __forceinline__ void store_4over6_colwise_packed_16x(
    const bool pick_map4, QuantizationCandidates4Over6 &candidates, const int thread_lane,
    output_type *out_t_data_sh, const size_t shmem_offset_base_colwise_out_t) {
  uint32_t *regs_4x = selected_4over6_packed(pick_map4, candidates);
  const int group = thread_lane / 16;
  uint32_t val[2];
  switch (group) {
    case 0:
      val[0] = regs_4x[0];
      val[1] = regs_4x[1];
      break;
    case 1:
      val[0] = regs_4x[1];
      val[1] = regs_4x[0];
      break;
  }
  uint32_t *out_t_data_sh_as_uint32_t =
      reinterpret_cast<uint32_t *>(&out_t_data_sh[shmem_offset_base_colwise_out_t]);
  out_t_data_sh_as_uint32_t[group] = val[0];            // idx1 = (group + 0) % 2;
  out_t_data_sh_as_uint32_t[(group + 1) & 1] = val[1];  // idx2 = (group + 1) % 2;
}

template <size_t WAVES, size_t PACK_SIZE, size_t SCALE_DIM, typename output_type>
__device__ __forceinline__ void store_4over6_rowwise_packed_16x(
    const bool pick_map4, QuantizationCandidates4Over6 &candidates, const int bank_group,
    const size_t thread_offset_X_rowwise, const size_t shmem_offset_base_rowwise_out,
    output_type *out_data_sh) {
  uint32_t *packed = selected_4over6_packed(pick_map4, candidates);
#pragma unroll
  for (int w = 0; w < WAVES; ++w) {
    const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
    const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
    const size_t shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;
    uint32_t *out_data_sh_as_uint32_t =
        reinterpret_cast<uint32_t *>(&out_data_sh[shmem_offset_rowwise]);
    out_data_sh_as_uint32_t[0] = packed[swizzled_group_idx / PACK_SIZE];
  }
}

template <typename output_vec_type>
__device__ __forceinline__ void store_4over6_packed_16x(const uint32_t (&packed)[2],
                                                        output_vec_type &output_vec) {
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[0]) = packed[0];
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[4]) = packed[1];
}

template <typename output_vec_type>
__device__ __forceinline__ void store_selected_4over6_packed_16x(
    const bool pick_map4, QuantizationCandidates4Over6 &candidates, output_vec_type &output_vec) {
  uint32_t *packed = selected_4over6_packed(pick_map4, candidates);
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[0]) = packed[0];
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[4]) = packed[1];
}

template <bool USE_FAST_MATH, bool REVERSE_PACK_ORDER = false, typename scaling_coeff_type,
          typename input_type>
__device__ __forceinline__ void quantize_4over6_contiguous_16x(
    const input_type *x, const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i]);
    second_half[i] = static_cast<float>(x[i + 8]);
  }

  quantize_4over6_16x<USE_FAST_MATH, REVERSE_PACK_ORDER>(first_half, second_half, scaling_factors,
                                                         global_amax, S_dec_b_fp8, rOut);
}

template <bool USE_FAST_MATH, bool REVERSE_PACK_ORDER = false, typename scaling_coeff_type,
          typename pair_type>
__device__ __forceinline__ void quantize_4over6_pair_array_16x(
    const pair_type (&x)[2][4], const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[0][i].x);
    first_half[2 * i + 1] = static_cast<float>(x[0][i].y);
    second_half[2 * i] = static_cast<float>(x[1][i].x);
    second_half[2 * i + 1] = static_cast<float>(x[1][i].y);
  }

  quantize_4over6_16x<USE_FAST_MATH, REVERSE_PACK_ORDER>(first_half, second_half, scaling_factors,
                                                         global_amax, S_dec_b_fp8, rOut);
}

template <bool USE_FAST_MATH, typename scaling_coeff_type, typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec2_array_candidates_16x(
    const vec_type (&x)[8], const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors,
    const float global_amax, QuantizationCandidates4Over6 &candidates) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[i].data.elt[0]);
    first_half[2 * i + 1] = static_cast<float>(x[i].data.elt[1]);
    second_half[2 * i] = static_cast<float>(x[i + 4].data.elt[0]);
    second_half[2 * i + 1] = static_cast<float>(x[i + 4].data.elt[1]);
  }

  candidates.err_map4 = 0.0f;
  candidates.err_map6 = 0.0f;
  quantize_4over6_16x<USE_FAST_MATH>(first_half, second_half, scaling_factors, global_amax,
                                     candidates.err_map4, candidates.err_map6, candidates.rOut_map4,
                                     candidates.rOut_map6);
}

template <bool USE_FAST_MATH, bool REVERSE_PACK_ORDER = false, typename scaling_coeff_type,
          typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec2_array_16x(
    const vec_type (&x)[8], const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[i].data.elt[0]);
    first_half[2 * i + 1] = static_cast<float>(x[i].data.elt[1]);
    second_half[2 * i] = static_cast<float>(x[i + 4].data.elt[0]);
    second_half[2 * i + 1] = static_cast<float>(x[i + 4].data.elt[1]);
  }

  quantize_4over6_16x<USE_FAST_MATH, REVERSE_PACK_ORDER>(first_half, second_half, scaling_factors,
                                                         global_amax, S_dec_b_fp8, rOut);
}

template <bool USE_FAST_MATH, typename scaling_coeff_type, typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec_index_candidates_16x(
    const vec_type (&x)[16], const int idx,
    const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors, const float global_amax,
    QuantizationCandidates4Over6 &candidates) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i].data.elt[idx]);
    second_half[i] = static_cast<float>(x[i + 8].data.elt[idx]);
  }

  candidates.err_map4 = 0.0f;
  candidates.err_map6 = 0.0f;
  quantize_4over6_16x<USE_FAST_MATH>(first_half, second_half, scaling_factors, global_amax,
                                     candidates.err_map4, candidates.err_map6, candidates.rOut_map4,
                                     candidates.rOut_map6);
}

template <bool USE_FAST_MATH, bool REVERSE_PACK_ORDER = false, typename scaling_coeff_type,
          typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec_index_16x(
    const vec_type (&x)[16], const int idx,
    const QuantizationScales4Over6<scaling_coeff_type> &scaling_factors, const float global_amax,
    nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i].data.elt[idx]);
    second_half[i] = static_cast<float>(x[i + 8].data.elt[idx]);
  }

  quantize_4over6_16x<USE_FAST_MATH, REVERSE_PACK_ORDER>(first_half, second_half, scaling_factors,
                                                         global_amax, S_dec_b_fp8, rOut);
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
