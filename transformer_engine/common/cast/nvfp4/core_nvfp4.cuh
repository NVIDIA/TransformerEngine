/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file core_nvfp4.cuh
 *  \brief Core functions used in NVFP4.
 */

#ifndef TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_CORE_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits>

#include "../../common.h"
#include "../../util/curanddx.hpp"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif  // FP4_TYPE_SUPPORTED

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

using nvfp4_scale_t = fp8e4m3;

namespace quantization_and_transposition_SF {
#if FP4_TYPE_SUPPORTED
// Used in transpose variant
// Compute per-block E4M3 encoding/decoding scaling factor
__device__ __forceinline__ nvfp4_scale_t compute_decoding_scaling_factor(const float block_amax,
                                                                         const float S_enc) {
  // constexpr float rcp_6f = 1.0f / 6.0f;
  // const float S_dec_b = block_amax * rcp_6f;
  // const nvfp4_scale_t S_dec_b_fp8 = static_cast<nvfp4_scale_t>(S_dec_b * S_enc);
  // return S_dec_b_fp8;
  // NOTE: Divide by 6.0f is not elegant and not efficient.
  // However, this is part of the emulation code to ensure exact match.
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  constexpr float fp4_max_inv = 1.0f / fp4_max;
  const float S_dec_b = block_amax * (S_enc * fp4_max_inv);
  return static_cast<nvfp4_scale_t>(fminf(S_dec_b, TypeExtrema<float>::max));
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace quantization_and_transposition_SF

namespace quantization_SF {
#if FP4_TYPE_SUPPORTED
// Used in non-transpose variant
// Compute per-block E4M3 encoding/decoding scaling factor
__device__ __forceinline__ fp8e4m3 compute_decoding_scaling_factor(const float block_amax,
                                                                   const float S_enc) {
  using namespace detail;
  constexpr float fp4_max_inv = 1.0f / TypeExtrema<fp4e2m1>::max;  // 1 / 6.0f
  // const float S_dec_b = block_amax * rcp_6f;
  // const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(S_dec_b * S_enc);
  // return S_dec_b_fp8;
  return static_cast<fp8e4m3>(block_amax * (S_enc * fp4_max_inv));
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace quantization_SF

namespace core {

#if FP4_TYPE_SUPPORTED
using namespace ptx;

// Compute the global encode scale factor for a given global amax
template <bool USE_4OVER6 = false>
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  constexpr float fp8_max = USE_4OVER6 ? 256.0f : TypeExtrema<fp8e4m3>::max;  // 448.0f;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;                        // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

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

__device__ __forceinline__ uint32_t get_rbits(
    transformer_engine::curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>
        &rng,
    // philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>: compile-time configurable rounds
    uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    random_uint4 = rng.generate4();
  }
  // Treat uint4 as an array of 4x uint32_t elements for indexing
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
