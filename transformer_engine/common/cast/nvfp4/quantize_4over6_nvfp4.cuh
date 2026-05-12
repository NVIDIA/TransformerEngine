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
 *  range as FP4 value 6. The selected candidate is the one with lower configured
 *  error after dequantizing back to the original input domain; ties select map-to-6.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace core {

#if FP4_TYPE_SUPPORTED

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_ERR_MODE_SWITCH(ERR_MODE, ERR_MODE_CONST, ...) \
  switch (ERR_MODE) {                                                                  \
    case kNVTENVFP44Over6ErrMAE: {                                                     \
      constexpr NVTENVFP44Over6ErrMode ERR_MODE_CONST = kNVTENVFP44Over6ErrMAE;        \
      { __VA_ARGS__ }                                                                  \
    } break;                                                                           \
    case kNVTENVFP44Over6ErrMSE: {                                                     \
      constexpr NVTENVFP44Over6ErrMode ERR_MODE_CONST = kNVTENVFP44Over6ErrMSE;        \
      { __VA_ARGS__ }                                                                  \
    } break;                                                                           \
    default: {                                                                         \
      NVTE_ERROR("Unsupported NVFP4 4over6 error mode.");                              \
    }                                                                                  \
  }

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_E4M3_MAX_SWITCH(E4M3_MAX_VALUE, E4M3_MAX_CONST, ...) \
  if ((E4M3_MAX_VALUE) == 256) {                                                             \
    constexpr int E4M3_MAX_CONST = 256;                                                      \
    { __VA_ARGS__ }                                                                          \
  } else {                                                                                   \
    NVTE_CHECK((E4M3_MAX_VALUE) == 448, "Unsupported NVFP4 E4M3 max.");                      \
    constexpr int E4M3_MAX_CONST = 448;                                                      \
    { __VA_ARGS__ }                                                                          \
  }

template <bool kEnabled, NVTENVFP44Over6ErrMode kErrMode = kNVTENVFP44Over6ErrMAE,
          bool kErrUseFastMath = false>
struct NVFP44Over6Config {
  static constexpr bool enabled = kEnabled;
  static constexpr NVTENVFP44Over6ErrMode err_mode = kErrMode;
  static constexpr bool err_use_fast_math = kErrUseFastMath;
};

using NVFP44Over6DisabledConfig = NVFP44Over6Config<false>;

#define TRANSFORMER_ENGINE_NVFP4_4OVER6_CONFIG_SWITCH(USE_4OVER6_VALUE, ERR_MODE_VALUE,           \
                                                      ERR_USE_FAST_MATH_VALUE, CONFIG_CONST, ...) \
  if (USE_4OVER6_VALUE) {                                                                         \
    TRANSFORMER_ENGINE_NVFP4_4OVER6_ERR_MODE_SWITCH(                                              \
        ERR_MODE_VALUE, ERR_MODE_CONST,                                                           \
        TRANSFORMER_ENGINE_SWITCH_CONDITION(ERR_USE_FAST_MATH_VALUE, ERR_USE_FAST_MATH_CONST, {   \
          using CONFIG_CONST = NVFP44Over6Config<true, ERR_MODE_CONST, ERR_USE_FAST_MATH_CONST>;  \
          { __VA_ARGS__ }                                                                         \
        }););                                                                                     \
  } else {                                                                                        \
    using CONFIG_CONST = NVFP44Over6DisabledConfig;                                               \
    { __VA_ARGS__ }                                                                               \
  }

__device__ __forceinline__ void compute_4over6_decoding_scaling_factors(
    const float block_amax, const float S_enc, nvfp4_scale_t &S_dec_b_fp8_map4,
    nvfp4_scale_t &S_dec_b_fp8_map6) {
  constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f
  constexpr float fp8_max = detail::TypeExtrema<fp8e4m3>::max;  // 448.0f
  constexpr float scale_expansion_factor = 1.5f;
  const float base_sf_high_precision = block_amax / fp4_max * S_enc;
  const float sf_high_precision_map4 =
      fminf(base_sf_high_precision * scale_expansion_factor, fp8_max);
  const float sf_high_precision_map6 = fminf(base_sf_high_precision, fp8_max);
  S_dec_b_fp8_map4 = static_cast<nvfp4_scale_t>(sf_high_precision_map4);
  S_dec_b_fp8_map6 = static_cast<nvfp4_scale_t>(sf_high_precision_map6);
}

struct QuantizationScales4Over6 {
  nvfp4_scale_t S_dec_b_fp8_map4;
  nvfp4_scale_t S_dec_b_fp8_map6;
  float SFcoefficient_map4;
  float SFcoefficient_map6;
};

__device__ __forceinline__ float compute_4over6_nvfp4_scaling_coefficient(
    const nvfp4_scale_t S_dec_block, const float S_enc) {
  const float S_dec = 1.0f / S_enc;
  return fminf(1.0f / (static_cast<float>(S_dec_block) * S_dec), detail::TypeExtrema<float>::max);
}

__device__ __forceinline__ QuantizationScales4Over6
compute_4over6_nvfp4_quantization_scaling_factors(const float block_amax, const float S_enc) {
  QuantizationScales4Over6 scaling_factors;
  compute_4over6_decoding_scaling_factors(block_amax, S_enc, scaling_factors.S_dec_b_fp8_map4,
                                          scaling_factors.S_dec_b_fp8_map6);
  scaling_factors.SFcoefficient_map4 =
      compute_4over6_nvfp4_scaling_coefficient(scaling_factors.S_dec_b_fp8_map4, S_enc);
  scaling_factors.SFcoefficient_map6 =
      compute_4over6_nvfp4_scaling_coefficient(scaling_factors.S_dec_b_fp8_map6, S_enc);
  return scaling_factors;
}

__device__ __forceinline__ QuantizationScales4Over6
compute_4over6_fp4_encode_quantization_scaling_factors(const float block_amax,
                                                       const float global_encode_scale,
                                                       const float global_decode_scale) {
  QuantizationScales4Over6 scaling_factors;
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

template <NVTENVFP44Over6ErrMode ERR_MODE>
__device__ __forceinline__ float compute_4over6_error_rn(const float diff) {
  if constexpr (ERR_MODE == kNVTENVFP44Over6ErrMSE) {
    return __fmul_rn(diff, diff);
  } else if constexpr (ERR_MODE == kNVTENVFP44Over6ErrMAE) {
    return fabsf(diff);
  } else {
    NVTE_DEVICE_ERROR("Unsupported NVFP4 4over6 error mode.");
    return fabsf(diff);
  }
}

template <NVTENVFP44Over6ErrMode ERR_MODE>
__device__ __forceinline__ float compute_4over6_error(const float diff) {
  if constexpr (ERR_MODE == kNVTENVFP44Over6ErrMSE) {
    return diff * diff;
  } else if constexpr (ERR_MODE == kNVTENVFP44Over6ErrMAE) {
    return fabsf(diff);
  } else {
    NVTE_DEVICE_ERROR("Unsupported NVFP4 4over6 error mode.");
    return fabsf(diff);
  }
}

template <typename FourOverSixConfig, int E4M3_MAX>
__device__ __forceinline__ uint32_t cvt_fp32_to_fp4_8x_with_error_rn(
    const float (&x)[8], const float block_scale_inverse, const nvfp4_scale_t S_dec_b_fp8,
    const float global_amax, float *err) {
  static_assert(FourOverSixConfig::enabled,
                "4over6 conversion helpers require an enabled 4over6 config.");
  uint32_t out = 0;
  uint32_t out_dequant_1 = 0;
  uint32_t out_dequant_2 = 0;
  uint32_t out_dequant_3 = 0;
  uint32_t out_dequant_4 = 0;

  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    float x_scaled[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      x_scaled[i] = __fmul_rn(x[i], block_scale_inverse);
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
    static_assert(E4M3_MAX == 448 || E4M3_MAX == 256, "Unsupported NVFP4 E4M3 max.");
    constexpr float fp8_4over6_max = static_cast<float>(E4M3_MAX);
    constexpr float err_denom = fp4_max * fp8_4over6_max;
    const float sf = static_cast<float>(S_dec_b_fp8);
    if constexpr (FourOverSixConfig::err_use_fast_math) {
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
        const float val = dequant[i] * sf * global_amax / err_denom;
        const float diff = val - x[i];
        *err += compute_4over6_error<FourOverSixConfig::err_mode>(diff);
      }
    } else {
      const float val0 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_1_lo)), sf), global_amax),
          err_denom);
      const float val1 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_1_hi)), sf), global_amax),
          err_denom);
      const float val2 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_2_lo)), sf), global_amax),
          err_denom);
      const float val3 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_2_hi)), sf), global_amax),
          err_denom);
      const float val4 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_3_lo)), sf), global_amax),
          err_denom);
      const float val5 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_3_hi)), sf), global_amax),
          err_denom);
      const float val6 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_4_lo)), sf), global_amax),
          err_denom);
      const float val7 = __fdiv_rn(
          __fmul_rn(__fmul_rn(__half2float(__ushort_as_half(out_dequant_4_hi)), sf), global_amax),
          err_denom);

      const float diff0 = __fsub_rn(val0, x[0]);
      const float diff1 = __fsub_rn(val1, x[1]);
      const float diff2 = __fsub_rn(val2, x[2]);
      const float diff3 = __fsub_rn(val3, x[3]);
      const float diff4 = __fsub_rn(val4, x[4]);
      const float diff5 = __fsub_rn(val5, x[5]);
      const float diff6 = __fsub_rn(val6, x[6]);
      const float diff7 = __fsub_rn(val7, x[7]);

      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff0));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff1));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff2));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff3));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff4));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff5));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff6));
      *err = __fadd_rn(*err, compute_4over6_error_rn<FourOverSixConfig::err_mode>(diff7));
    }
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }

  return out;
}

template <typename FourOverSixConfig, int E4M3_MAX, bool REVERSE_PACK_ORDER>
__device__ __forceinline__ void quantize_4over6_16x(const float (&first_half)[8],
                                                    const float (&second_half)[8],
                                                    const QuantizationScales4Over6 &scaling_factors,
                                                    const float global_amax, float &err_map4,
                                                    float &err_map6, uint32_t (&rOut_map4)[2],
                                                    uint32_t (&rOut_map6)[2]) {
  if constexpr (REVERSE_PACK_ORDER) {
    rOut_map4[1] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[1] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
    rOut_map4[0] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[0] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
  } else {
    rOut_map4[0] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[0] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        first_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
    rOut_map4[1] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map4),
        scaling_factors.S_dec_b_fp8_map4, global_amax, &err_map4);
    rOut_map6[1] = cvt_fp32_to_fp4_8x_with_error_rn<FourOverSixConfig, E4M3_MAX>(
        second_half, static_cast<float>(scaling_factors.SFcoefficient_map6),
        scaling_factors.S_dec_b_fp8_map6, global_amax, &err_map6);
  }
}

__device__ __forceinline__ bool pick_4over6_map4(const float err_map4, const float err_map6) {
  return err_map4 < err_map6;
}

__device__ __forceinline__ nvfp4_scale_t
selected_4over6_scale(const bool pick_map4, const QuantizationScales4Over6 &scaling_factors) {
  if (pick_map4) {
    return scaling_factors.S_dec_b_fp8_map4;
  }
  return scaling_factors.S_dec_b_fp8_map6;
}

template <typename FourOverSixConfig, int E4M3_MAX, bool REVERSE_PACK_ORDER>
__device__ __forceinline__ void quantize_4over6_16x(const float (&first_half)[8],
                                                    const float (&second_half)[8],
                                                    const QuantizationScales4Over6 &scaling_factors,
                                                    const float global_amax,
                                                    nvfp4_scale_t &S_dec_b_fp8,
                                                    uint32_t (&rOut)[2]) {
  float err_map4 = 0.0f;
  float err_map6 = 0.0f;
  __align__(8) uint32_t rOut_map4[2];
  __align__(8) uint32_t rOut_map6[2];

  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, REVERSE_PACK_ORDER>(
      first_half, second_half, scaling_factors, global_amax, err_map4, err_map6, rOut_map4,
      rOut_map6);

  const bool pick_map4 = pick_4over6_map4(err_map4, err_map6);
  S_dec_b_fp8 = selected_4over6_scale(pick_map4, scaling_factors);
  if (pick_map4) {
    rOut[0] = rOut_map4[0];
    rOut[1] = rOut_map4[1];
  } else {
    rOut[0] = rOut_map6[0];
    rOut[1] = rOut_map6[1];
  }
}

struct QuantizationCandidates4Over6 {
  float err_map4;
  float err_map6;
  uint32_t rOut_map4[2];
  uint32_t rOut_map6[2];

  __device__ __forceinline__ void reset_errors() {
    err_map4 = 0.0f;
    err_map6 = 0.0f;
  }

  __device__ __forceinline__ const uint32_t *selected_packed(const bool pick_map4) const {
    if (pick_map4) {
      return rOut_map4;
    }
    return rOut_map6;
  }
};

template <size_t BLOCK_DIM, size_t BLOCKS_PER_TILE_Y, size_t BLOCKS_PER_TILE_X>
struct alignas(16) QuantizationScratch4Over6 {
  alignas(16) float err_map4_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM];
  alignas(16) float err_map6_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X][BLOCK_DIM];
  alignas(16) uint8_t pick_map4_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X];
  alignas(16) nvfp4_scale_t selected_scale_matrix[BLOCKS_PER_TILE_Y][BLOCKS_PER_TILE_X];

  template <bool USE_2D_QUANTIZATION, bool USE_4OVER6>
  static constexpr size_t dynamic_shared_memory_size() {
    if constexpr (USE_2D_QUANTIZATION && USE_4OVER6) {
      return ((sizeof(QuantizationScratch4Over6) + TMA_SHMEM_ALIGNMENT - 1) / TMA_SHMEM_ALIGNMENT) *
             TMA_SHMEM_ALIGNMENT;
    }
    return 0;
  }
};

template <typename input_type>
__device__ __forceinline__ void load_4over6_contiguous_halves_16x(const input_type *x,
                                                                  float (&first_half)[8],
                                                                  float (&second_half)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i]);
    second_half[i] = static_cast<float>(x[i + 8]);
  }
}

template <typename pair_type>
__device__ __forceinline__ void load_4over6_pair_array_halves_16x(const pair_type (&x)[2][4],
                                                                  float (&first_half)[8],
                                                                  float (&second_half)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[0][i].x);
    first_half[2 * i + 1] = static_cast<float>(x[0][i].y);
    second_half[2 * i] = static_cast<float>(x[1][i].x);
    second_half[2 * i + 1] = static_cast<float>(x[1][i].y);
  }
}

template <typename vec_type>
__device__ __forceinline__ void load_4over6_vec2_array_halves_16x(const vec_type (&x)[8],
                                                                  float (&first_half)[8],
                                                                  float (&second_half)[8]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[i].data.elt[0]);
    first_half[2 * i + 1] = static_cast<float>(x[i].data.elt[1]);
    second_half[2 * i] = static_cast<float>(x[i + 4].data.elt[0]);
    second_half[2 * i + 1] = static_cast<float>(x[i + 4].data.elt[1]);
  }
}

template <typename vec_type>
__device__ __forceinline__ void load_4over6_vec_index_halves_16x(const vec_type (&x)[16],
                                                                 const int idx,
                                                                 float (&first_half)[8],
                                                                 float (&second_half)[8]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i].data.elt[idx]);
    second_half[i] = static_cast<float>(x[i + 8].data.elt[idx]);
  }
}

template <typename FourOverSixConfig, int E4M3_MAX>
__device__ __forceinline__ void quantize_4over6_candidates_16x(
    const float (&x)[16], const QuantizationScales4Over6 &scaling_factors, const float global_amax,
    QuantizationCandidates4Over6 &candidates) {
  float first_half[8];
  float second_half[8];
  load_4over6_contiguous_halves_16x(x, first_half, second_half);

  candidates.reset_errors();
  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, false>(
      first_half, second_half, scaling_factors, global_amax, candidates.err_map4,
      candidates.err_map6, candidates.rOut_map4, candidates.rOut_map6);
}

template <size_t BLOCK_DIM, size_t BLOCKS_PER_TILE_Y, size_t BLOCKS_PER_TILE_X>
__device__ __forceinline__ bool record_and_select_4over6_2d_block(
    const QuantizationScales4Over6 &scaling_factors, const size_t block_in_tile_y,
    const size_t block_in_tile_x, const size_t participant_idx,
    QuantizationScratch4Over6<BLOCK_DIM, BLOCKS_PER_TILE_Y, BLOCKS_PER_TILE_X> &scratch,
    nvfp4_scale_t &S_dec_b_fp8, const QuantizationCandidates4Over6 &candidates) {
  scratch.err_map4_matrix[block_in_tile_y][block_in_tile_x][participant_idx] = candidates.err_map4;
  scratch.err_map6_matrix[block_in_tile_y][block_in_tile_x][participant_idx] = candidates.err_map6;
  __syncthreads();

  if (participant_idx == 0) {
    float block_err_map4 = 0.0f;
    float block_err_map6 = 0.0f;
#pragma unroll
    for (int i = 0; i < BLOCK_DIM; ++i) {
      block_err_map4 += scratch.err_map4_matrix[block_in_tile_y][block_in_tile_x][i];
      block_err_map6 += scratch.err_map6_matrix[block_in_tile_y][block_in_tile_x][i];
    }

    const bool pick_map4 = pick_4over6_map4(block_err_map4, block_err_map6);
    if (pick_map4) {
      scratch.pick_map4_matrix[block_in_tile_y][block_in_tile_x] = 1;
    } else {
      scratch.pick_map4_matrix[block_in_tile_y][block_in_tile_x] = 0;
    }
    scratch.selected_scale_matrix[block_in_tile_y][block_in_tile_x] =
        selected_4over6_scale(pick_map4, scaling_factors);
  }
  __syncthreads();
  S_dec_b_fp8 = scratch.selected_scale_matrix[block_in_tile_y][block_in_tile_x];
  return scratch.pick_map4_matrix[block_in_tile_y][block_in_tile_x] != 0;
}

template <typename FourOverSixConfig, int E4M3_MAX, size_t BLOCK_DIM, size_t BLOCKS_PER_TILE_Y,
          size_t BLOCKS_PER_TILE_X>
__device__ __forceinline__ bool quantize_and_select_4over6_2d_block_16x(
    const float (&x)[16], const float block_amax, const float global_encode_scale,
    const float global_decode_scale, const float global_amax, const size_t block_in_tile_y,
    const size_t block_in_tile_x, const size_t participant_idx,
    QuantizationScratch4Over6<BLOCK_DIM, BLOCKS_PER_TILE_Y, BLOCKS_PER_TILE_X> &scratch,
    nvfp4_scale_t &S_dec_b_fp8, QuantizationCandidates4Over6 &candidates) {
  const auto scaling_factors = compute_4over6_fp4_encode_quantization_scaling_factors(
      block_amax, global_encode_scale, global_decode_scale);
  quantize_4over6_candidates_16x<FourOverSixConfig, E4M3_MAX>(x, scaling_factors, global_amax,
                                                              candidates);

  const bool pick_map4 =
      record_and_select_4over6_2d_block<BLOCK_DIM, BLOCKS_PER_TILE_Y, BLOCKS_PER_TILE_X>(
          scaling_factors, block_in_tile_y, block_in_tile_x, participant_idx, scratch, S_dec_b_fp8,
          candidates);
  return pick_map4;
}

template <typename output_type>
__device__ __forceinline__ void store_4over6_colwise_packed_16x(
    const bool pick_map4, const QuantizationCandidates4Over6 &candidates, const int thread_lane,
    output_type *out_t_data_sh, const size_t shmem_offset_base_colwise_out_t) {
  const uint32_t *regs_4x = candidates.selected_packed(pick_map4);
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
    const bool pick_map4, const QuantizationCandidates4Over6 &candidates, const int bank_group,
    const size_t thread_offset_X_rowwise, const size_t shmem_offset_base_rowwise_out,
    output_type *out_data_sh) {
  const uint32_t *packed = candidates.selected_packed(pick_map4);
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
__device__ __forceinline__ void store_4over6_packed_16x(const uint32_t *packed,
                                                        output_vec_type &output_vec) {
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[0]) = packed[0];
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[4]) = packed[1];
}

template <typename output_vec_type>
__device__ __forceinline__ void store_selected_4over6_packed_16x(
    const bool pick_map4, const QuantizationCandidates4Over6 &candidates,
    output_vec_type &output_vec) {
  store_4over6_packed_16x(candidates.selected_packed(pick_map4), output_vec);
}

template <typename FourOverSixConfig, int E4M3_MAX, bool REVERSE_PACK_ORDER, typename input_type>
__device__ __forceinline__ void quantize_4over6_contiguous_16x(
    const input_type *x, const QuantizationScales4Over6 &scaling_factors, const float global_amax,
    nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
  load_4over6_contiguous_halves_16x(x, first_half, second_half);

  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, REVERSE_PACK_ORDER>(
      first_half, second_half, scaling_factors, global_amax, S_dec_b_fp8, rOut);
}

template <typename FourOverSixConfig, int E4M3_MAX, bool REVERSE_PACK_ORDER, typename pair_type>
__device__ __forceinline__ void quantize_4over6_pair_array_16x(
    const pair_type (&x)[2][4], const QuantizationScales4Over6 &scaling_factors,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
  load_4over6_pair_array_halves_16x(x, first_half, second_half);

  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, REVERSE_PACK_ORDER>(
      first_half, second_half, scaling_factors, global_amax, S_dec_b_fp8, rOut);
}

template <typename FourOverSixConfig, int E4M3_MAX, typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec2_array_candidates_16x(
    const vec_type (&x)[8], const QuantizationScales4Over6 &scaling_factors,
    const float global_amax, QuantizationCandidates4Over6 &candidates) {
  float first_half[8];
  float second_half[8];
  load_4over6_vec2_array_halves_16x(x, first_half, second_half);

  candidates.reset_errors();
  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, false>(
      first_half, second_half, scaling_factors, global_amax, candidates.err_map4,
      candidates.err_map6, candidates.rOut_map4, candidates.rOut_map6);
}

template <typename FourOverSixConfig, int E4M3_MAX, bool REVERSE_PACK_ORDER, typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec2_array_16x(
    const vec_type (&x)[8], const QuantizationScales4Over6 &scaling_factors,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
  load_4over6_vec2_array_halves_16x(x, first_half, second_half);

  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, REVERSE_PACK_ORDER>(
      first_half, second_half, scaling_factors, global_amax, S_dec_b_fp8, rOut);
}

template <typename FourOverSixConfig, int E4M3_MAX, typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec_index_candidates_16x(
    const vec_type (&x)[16], const int idx, const QuantizationScales4Over6 &scaling_factors,
    const float global_amax, QuantizationCandidates4Over6 &candidates) {
  float first_half[8];
  float second_half[8];
  load_4over6_vec_index_halves_16x(x, idx, first_half, second_half);

  candidates.reset_errors();
  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, false>(
      first_half, second_half, scaling_factors, global_amax, candidates.err_map4,
      candidates.err_map6, candidates.rOut_map4, candidates.rOut_map6);
}

template <typename FourOverSixConfig, int E4M3_MAX, bool REVERSE_PACK_ORDER, typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec_index_16x(
    const vec_type (&x)[16], const int idx, const QuantizationScales4Over6 &scaling_factors,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
  load_4over6_vec_index_halves_16x(x, idx, first_half, second_half);

  quantize_4over6_16x<FourOverSixConfig, E4M3_MAX, REVERSE_PACK_ORDER>(
      first_half, second_half, scaling_factors, global_amax, S_dec_b_fp8, rOut);
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
