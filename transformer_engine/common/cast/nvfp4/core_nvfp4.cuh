/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime.h>

#include <limits>

#include "../../common.h"
#include "../../utils.cuh"
#include "../../util/math.h"
#include "../../util/ptx.cuh"

#include "curanddx.hpp"
#if CUDA_VERSION > 12080
#include <cuda_fp4.h>
#endif  // CUDA_VERSION > 12080


namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace quantization_and_transposition_SF {
  using nvfp4_scale_t = fp8e4m3;
  // Used in transpose variant
  // Compute per-block E4M3 encoding/decoding scaling factor
  __device__ __forceinline__ nvfp4_scale_t
  compute_decoding_scaling_factor(const float block_amax, const float S_enc) {
    // constexpr float rcp_6f = 1.0f / 6.0f;
    // const float S_dec_b = block_amax * rcp_6f;
    // const nvfp4_scale_t S_dec_b_fp8 = static_cast<nvfp4_scale_t>(S_dec_b * S_enc);
    // return S_dec_b_fp8;
    // NOTE: Divide by 6.0f is not elegant and not efficient.
    // However, this is part of the emulation code to ensure exact match.
    using namespace detail;
    constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
    const float S_dec_b = block_amax / fp4_max * S_enc;
    return static_cast<nvfp4_scale_t>(fminf(S_dec_b, TypeExtrema<float>::max));
  }
}

namespace quantization_SF {
  // Used in non-transpose variant
  // Compute per-block E4M3 encoding/decoding scaling factor
  __device__ __forceinline__ fp8e4m3
  compute_decoding_scaling_factor(const float block_amax, const float S_enc) {
    constexpr float rcp_6f = 1.0f / 6.0f;
    // const float S_dec_b = block_amax * rcp_6f;
    // const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(S_dec_b * S_enc);
    // return S_dec_b_fp8;
    return static_cast<fp8e4m3>(block_amax * rcp_6f * S_enc);
  }
}

namespace core {

#if CUDA_VERSION > 12080
using RNG = decltype(curanddx::Generator<curanddx::philox4_32>() + curanddx::PhiloxRounds<10>() +
                     curanddx::SM<800>() + curanddx::Thread());

using namespace ptx;

// Compute the global encode scale factor for a given global amax
__device__ __forceinline__ float
compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  constexpr float fp8_max = TypeExtrema<fp8e4m3>::max;  // 448.0f;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

__device__ __forceinline__ uint32_t
get_rbits(RNG &rng, uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    curanddx::uniform_bits dist;
    random_uint4 = dist.generate4(rng);
  }
  // Treat uint4 as an array of 4x uint32_t elements for indexing
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

__device__ __forceinline__ fp4e2m1x4
mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b16 v0_bf16; \n\t"
      ".reg.b16 v1_bf16; \n\t"
      ".reg.b16 v2_bf16; \n\t"
      ".reg.b16 v3_bf16; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %2; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %2; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %3; \n\t"  // mind the shuffled elements order
      "}"
      : "=h"(out_4x)
      : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4
mul_cvt_bf16_to_fp4_4x_with_rn(const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  // NOTE: rbits unused for rn.
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b16 v0_bf16; \n\t"
      ".reg.b16 v1_bf16; \n\t"
      ".reg.b16 v2_bf16; \n\t"
      ".reg.b16 v3_bf16; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
      "cvt.f32.bf16 v0, v0_bf16; \n\t"
      "cvt.f32.bf16 v1, v1_bf16; \n\t"
      "cvt.f32.bf16 v2, v2_bf16; \n\t"
      "cvt.f32.bf16 v3, v3_bf16; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %2; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %2; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4
mul_cvt_bf16_to_fp4_4x(const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(in_4x, scale, rbits);
  } else {
    return mul_cvt_bf16_to_fp4_4x_with_rn(in_4x, scale, rbits);
  }
}

__device__ __forceinline__ fp4e2m1x4
mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(const float2 in01, const float2 in23,
                                                const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      "mov.b64 {v0, v1} , %1; \n\t"
      "mov.b64 {v2, v3} , %2; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %3; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %3; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %4; \n\t"  // mind the shuffled elements order
      "}"
      : "=h"(out_4x)
      : "l"(reinterpret_cast<const uint64_t &>(in01)),
        "l"(reinterpret_cast<const uint64_t &>(in23)),
        "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4
mul_cvt_fp32_to_fp4_4x_with_rn(const float2 in01, const float2 in23,
                               const float2 scale, const uint32_t rbits) {
  // NOTE: rbits unused for rn.
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
#if CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "mov.b64 {v0, v1} , %1; \n\t"
      "mov.b64 {v2, v3} , %2; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %3; \n\t"  // mind the shuffled elements order
      "mul.f32x2 v23, v23, %3; \n\t"  // mind the shuffled elements order
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(reinterpret_cast<const uint64_t &>(in01)),
        "l"(reinterpret_cast<const uint64_t &>(in23)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#else
  NVTE_DEVICE_ERROR(
      "FP4 cvt PTX instructions are architecture-specific. "
      "Try recompiling with sm_XXXa instead of sm_XXX.");
#endif  // CUDA_ARCH_HAS_FEATURE_SM10X_ALL
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4
mul_cvt_fp32_to_fp4_4x(const float2 in01, const float2 in23,
                       const float2 scale, const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, scale, rbits);
  } else {
    return mul_cvt_fp32_to_fp4_4x_with_rn(in01, in23, scale, rbits);
  }
}

#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#endif  // CUDA_VERSION > 12080

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
