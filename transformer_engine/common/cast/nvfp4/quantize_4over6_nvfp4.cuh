/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_4over6_nvfp4.cuh
 *  \brief Helpers used by NVFP4 4over6 quantization.
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

__device__ __forceinline__ void compute_4over6_decoding_scaling_factors(
    const float block_amax, const float S_enc, nvfp4_scale_t &S_dec_b_fp8_map4,
    nvfp4_scale_t &S_dec_b_fp8_map6) {
  constexpr float fp4_max = detail::TypeExtrema<fp4e2m1>::max;  // 6.0f
  const float sf_high_precision = block_amax / fp4_max * S_enc;
  S_dec_b_fp8_map4 = static_cast<nvfp4_scale_t>(sf_high_precision * 1.5f);
  S_dec_b_fp8_map6 = static_cast<nvfp4_scale_t>(sf_high_precision);
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

template <bool USE_FAST_MATH, typename scaling_coeff_type, bool REVERSE_PACK_ORDER = false>
__device__ __forceinline__ void quantize_4over6_16x(
    const float (&first_half)[8], const float (&second_half)[8],
    const nvfp4_scale_t S_dec_b_fp8_map4, const nvfp4_scale_t S_dec_b_fp8_map6,
    const scaling_coeff_type SFcoefficient_map4, const scaling_coeff_type SFcoefficient_map6,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float err_map4 = 0.0f;
  float err_map6 = 0.0f;
  __align__(8) uint32_t rOut_map4[2];
  __align__(8) uint32_t rOut_map6[2];

  if constexpr (REVERSE_PACK_ORDER) {
    rOut_map4[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(SFcoefficient_map4), S_dec_b_fp8_map4, global_amax,
        &err_map4);
    rOut_map6[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(SFcoefficient_map6), S_dec_b_fp8_map6, global_amax,
        &err_map6);
    rOut_map4[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(SFcoefficient_map4), S_dec_b_fp8_map4, global_amax,
        &err_map4);
    rOut_map6[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(SFcoefficient_map6), S_dec_b_fp8_map6, global_amax,
        &err_map6);
  } else {
    rOut_map4[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(SFcoefficient_map4), S_dec_b_fp8_map4, global_amax,
        &err_map4);
    rOut_map6[0] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        first_half, static_cast<float>(SFcoefficient_map6), S_dec_b_fp8_map6, global_amax,
        &err_map6);
    rOut_map4[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(SFcoefficient_map4), S_dec_b_fp8_map4, global_amax,
        &err_map4);
    rOut_map6[1] = cvt_fp32_to_fp4_8x_with_mse_rn<USE_FAST_MATH>(
        second_half, static_cast<float>(SFcoefficient_map6), S_dec_b_fp8_map6, global_amax,
        &err_map6);
  }

  if (err_map4 < err_map6) {
    S_dec_b_fp8 = S_dec_b_fp8_map4;
    rOut[0] = rOut_map4[0];
    rOut[1] = rOut_map4[1];
  } else {
    S_dec_b_fp8 = S_dec_b_fp8_map6;
    rOut[0] = rOut_map6[0];
    rOut[1] = rOut_map6[1];
  }
}

template <typename output_vec_type>
__device__ __forceinline__ void store_4over6_packed_16x(const uint32_t (&packed)[2],
                                                        output_vec_type &output_vec) {
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[0]) = packed[0];
  *reinterpret_cast<uint32_t *>(&output_vec.data.elt[4]) = packed[1];
}

template <bool USE_FAST_MATH, typename scaling_coeff_type, bool REVERSE_PACK_ORDER = false,
          typename input_type>
__device__ __forceinline__ void quantize_4over6_contiguous_16x(
    const input_type *x, const nvfp4_scale_t S_dec_b_fp8_map4, const nvfp4_scale_t S_dec_b_fp8_map6,
    const scaling_coeff_type SFcoefficient_map4, const scaling_coeff_type SFcoefficient_map6,
    const float global_amax, nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i]);
    second_half[i] = static_cast<float>(x[i + 8]);
  }

  quantize_4over6_16x<USE_FAST_MATH, scaling_coeff_type, REVERSE_PACK_ORDER>(
      first_half, second_half, S_dec_b_fp8_map4, S_dec_b_fp8_map6, SFcoefficient_map4,
      SFcoefficient_map6, global_amax, S_dec_b_fp8, rOut);
}

template <bool USE_FAST_MATH, typename scaling_coeff_type, bool REVERSE_PACK_ORDER = false,
          typename pair_type>
__device__ __forceinline__ void quantize_4over6_pair_array_16x(
    const pair_type (&x)[2][4], const nvfp4_scale_t S_dec_b_fp8_map4,
    const nvfp4_scale_t S_dec_b_fp8_map6, const scaling_coeff_type SFcoefficient_map4,
    const scaling_coeff_type SFcoefficient_map6, const float global_amax,
    nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[0][i].x);
    first_half[2 * i + 1] = static_cast<float>(x[0][i].y);
    second_half[2 * i] = static_cast<float>(x[1][i].x);
    second_half[2 * i + 1] = static_cast<float>(x[1][i].y);
  }

  quantize_4over6_16x<USE_FAST_MATH, scaling_coeff_type, REVERSE_PACK_ORDER>(
      first_half, second_half, S_dec_b_fp8_map4, S_dec_b_fp8_map6, SFcoefficient_map4,
      SFcoefficient_map6, global_amax, S_dec_b_fp8, rOut);
}

template <bool USE_FAST_MATH, typename scaling_coeff_type, bool REVERSE_PACK_ORDER = false,
          typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec2_array_16x(
    const vec_type (&x)[8], const nvfp4_scale_t S_dec_b_fp8_map4,
    const nvfp4_scale_t S_dec_b_fp8_map6, const scaling_coeff_type SFcoefficient_map4,
    const scaling_coeff_type SFcoefficient_map6, const float global_amax,
    nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    first_half[2 * i] = static_cast<float>(x[i].data.elt[0]);
    first_half[2 * i + 1] = static_cast<float>(x[i].data.elt[1]);
    second_half[2 * i] = static_cast<float>(x[i + 4].data.elt[0]);
    second_half[2 * i + 1] = static_cast<float>(x[i + 4].data.elt[1]);
  }

  quantize_4over6_16x<USE_FAST_MATH, scaling_coeff_type, REVERSE_PACK_ORDER>(
      first_half, second_half, S_dec_b_fp8_map4, S_dec_b_fp8_map6, SFcoefficient_map4,
      SFcoefficient_map6, global_amax, S_dec_b_fp8, rOut);
}

template <bool USE_FAST_MATH, typename scaling_coeff_type, bool REVERSE_PACK_ORDER = false,
          typename vec_type>
__device__ __forceinline__ void quantize_4over6_vec_index_16x(
    const vec_type (&x)[16], const int idx, const nvfp4_scale_t S_dec_b_fp8_map4,
    const nvfp4_scale_t S_dec_b_fp8_map6, const scaling_coeff_type SFcoefficient_map4,
    const scaling_coeff_type SFcoefficient_map6, const float global_amax,
    nvfp4_scale_t &S_dec_b_fp8, uint32_t (&rOut)[2]) {
  float first_half[8];
  float second_half[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    first_half[i] = static_cast<float>(x[i].data.elt[idx]);
    second_half[i] = static_cast<float>(x[i + 8].data.elt[idx]);
  }

  quantize_4over6_16x<USE_FAST_MATH, scaling_coeff_type, REVERSE_PACK_ORDER>(
      first_half, second_half, S_dec_b_fp8_map4, S_dec_b_fp8_map6, SFcoefficient_map4,
      SFcoefficient_map6, global_amax, S_dec_b_fp8, rOut);
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_4OVER6_NVFP4_CUH_
