/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ptx_arch_spec.cuh
 *  \brief Architecture / family specific PTX helpers
 *
 *  This header should only be included by sources that are listed under
 *  `transformer_engine_cuda_arch_specific_sources` in CMakeLists.txt since these helper
 *  functions use architecture-specific instructions and must be compiled with
 *  corresponding flags (e.g. sm100f, sm100a, etc.).
 */

#ifndef TRANSFORMER_ENGINE_PTX_ARCH_SPEC_CUH_
#define TRANSFORMER_ENGINE_PTX_ARCH_SPEC_CUH_

#include "common/util/ptx.cuh"

namespace transformer_engine {

namespace ptx {

__device__ __forceinline__ void try_cancel_cta(uint64_t *mbar, __uint128_t *response_data_ptr) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
    uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::"
        "all.b128 "
        "[%0], [%1];" ::"r"(workID_response),
        "r"(mbar_ptr));
  } else {
    NVTE_DEVICE_ERROR(
        "Cluster Launch Control PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

__device__ __forceinline__ void get_cancelled_cta_id_2D(__uint128_t *response_data_ptr,
                                                        int32_t &ctaid_X, int32_t &ctaid_Y) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
    asm volatile(
        "{\n\t"
        ".reg .s32 x_ctaid; \n\t"
        ".reg .s32 y_ctaid; \n\t"
        "mov .s32 x_ctaid, -1; \n\t"
        "mov .s32 y_ctaid, -1; \n\t"
        ".reg.b128 try_cancel_response; \n\t"
        "ld.shared.b128 try_cancel_response, [%2]; \n\t"
        ".reg .pred P1; \n\t"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P1, try_cancel_response; \n\t"
        "@P1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x_ctaid, y_ctaid, _, "
        "_}, try_cancel_response; \n\t"
        "mov .s32 %0, x_ctaid; \n\t"
        "mov .s32 %1, y_ctaid; \n\t"
        "}\n\t"
        : "=r"(ctaid_X), "=r"(ctaid_Y)
        : "r"(workID_response)
        : "memory");
  } else {
    NVTE_DEVICE_ERROR(
        "Cluster Launch Control PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

__device__ __forceinline__ e8m0_t float_to_e8m0(float val) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint16_t out;
    asm volatile(
        "{\n"
        "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
        "}"
        : "=h"(out)
        : "f"(val));
    return *reinterpret_cast<e8m0_t *>(&out);
  } else {
    // TODO: nan/inf needs to be set for any value
    // of nan/inf in input not just amax.
    if (isnan(val)) {
      return 0xFF;
    }
    if (isinf(val)) {
      return 0xFE;
    }
    if (val == 0.0f) {
      return 0x00;
    }
    uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
    e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    // Round up exponent and deal with satfinite.
    if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) {
      ++exponent;
    }
    return exponent;
  }
}

__device__ __forceinline__ void reduce_sync_max_abs_f32(float &out, float const &in) {
  constexpr bool is_sm_100f = NVTE_CUDA_ARCH_MATCHES(ptx::FamilySpecific<100>);
  if constexpr (is_sm_100f) {
    asm volatile("redux.sync.max.abs.f32 %0, %1, 0xFFFFFFFF;" : "=f"(out) : "f"(in));
  } else {
    asm volatile(
        "{\n\t"
        ".reg.b32 val;\n"
        "abs.f32 val, %1;\n"
        "redux.sync.max.u32 %0, val, 0xFFFFFFFF;\n"
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t &>(out))
        : "f"(in));
  }
}

#if FP4_TYPE_SUPPORTED

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
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
  } else {
    // mul.f32x2 above applies scale.x to the even elements and scale.y to the odd ones.
    const bf16 *vals = reinterpret_cast<const bf16 *>(&in_4x);
    const float q0 = stochastic_round_fp4_e2m1(static_cast<float>(vals[0]) * scale.x, rbits);
    const float q1 = stochastic_round_fp4_e2m1(static_cast<float>(vals[1]) * scale.y, rbits >> 8);
    const float q2 = stochastic_round_fp4_e2m1(static_cast<float>(vals[2]) * scale.x, rbits >> 16);
    const float q3 = stochastic_round_fp4_e2m1(static_cast<float>(vals[3]) * scale.y, rbits >> 24);
    const fp4e2m1x4 packed(make_float4(q0, q1, q2, q3));
    out_4x = *reinterpret_cast<const uint16_t *>(&packed);
  }
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_rn(const uint64_t in_4x,
                                                                    const float2 scale,
                                                                    const uint32_t rbits) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
  if constexpr (is_blackwell) {
    // NOTE: rbits unused for rn.
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
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x(const uint64_t in_4x,
                                                            const float2 scale,
                                                            const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(in_4x, scale, rbits);
  } else {
    return mul_cvt_bf16_to_fp4_4x_with_rn(in_4x, scale, rbits);
  }
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
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
  } else {
    const float q0 = stochastic_round_fp4_e2m1(in01.x * scale.x, rbits);
    const float q1 = stochastic_round_fp4_e2m1(in01.y * scale.y, rbits >> 8);
    const float q2 = stochastic_round_fp4_e2m1(in23.x * scale.x, rbits >> 16);
    const float q3 = stochastic_round_fp4_e2m1(in23.y * scale.y, rbits >> 24);
    const fp4e2m1x4 packed(make_float4(q0, q1, q2, q3));
    out_4x = *reinterpret_cast<const uint16_t *>(&packed);
  }
  return *reinterpret_cast<fp4e2m1x4 *>(&out_4x);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_rn(const float2 in01,
                                                                    const float2 in23,
                                                                    const float2 scale,
                                                                    const uint32_t rbits) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  uint32_t out_4x = 0;  // Only need 16 bit. Using 32 bit container for packing.
  if constexpr (is_blackwell) {
    // NOTE: rbits unused for rn.
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
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return reinterpret_cast<fp4e2m1x4 *>(&out_4x)[0];
}

template <bool USE_STOCHASTIC_ROUNDING>
__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x(const float2 in01, const float2 in23,
                                                            const float2 scale,
                                                            const uint32_t rbits) {
  if constexpr (USE_STOCHASTIC_ROUNDING) {
    return mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(in01, in23, scale, rbits);
  } else {
    return mul_cvt_fp32_to_fp4_4x_with_rn(in01, in23, scale, rbits);
  }
}

template <typename SCALING_COEFFICIENT_TYPE>
__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_round_to_nearest(
    const uint64_t in03, const uint64_t in47, const SCALING_COEFFICIENT_TYPE scaling_coefficient) {
  uint32_t out_8x = 0;
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, bf16>::value) {
      asm volatile(
          "{\n"
          ".reg.f32 zero; \n\t"
          "mov.b32 zero, 0; \n\t"
          ".reg.b16 scaling_coeff; \n\t"
          "mov.b16 scaling_coeff, %3; \n\t"
          ".reg.b16 v0_h, v1_h, v2_h, v3_h, v4_h, v5_h, v6_h, v7_h; \n\t"
          "mov.b64 {v0_h, v1_h, v2_h, v3_h}, %1; \n\t"
          "mov.b64 {v4_h, v5_h, v6_h, v7_h}, %2; \n\t"

          ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
          "fma.rn.f32.bf16 v0, v0_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v1, v1_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v2, v2_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v3, v3_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v4, v4_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v5, v5_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v6, v6_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v7, v7_h, scaling_coeff, zero; \n\t"

          ".reg.b8 f0, f1, f2, f3; \n\t"
          // Elements reordered to match e2m1x4 packing order (v1,v0)
          "cvt.rn.satfinite.e2m1x2.f32 f0, v1, v0;\n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f1, v3, v2;\n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f2, v5, v4;\n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f3, v7, v6;\n\t"
          "mov.b32 %0, {f0, f1, f2, f3};\n"
          "}"
          : "=r"(out_8x)
          : "l"(in03), "l"(in47), "h"(reinterpret_cast<const uint16_t &>(scaling_coefficient)));
    } else if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, float>::value) {
      asm volatile(
          "{\n"
          ".reg.b64 scaling_coeff_2x; \n\t"
          "mov.b64 scaling_coeff_2x, {%3, %3}; \n\t"
          ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
          "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
          "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"

          ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
          "cvt.f32.bf16 v0, v0_bf16; \n\t"
          "cvt.f32.bf16 v1, v1_bf16; \n\t"
          "cvt.f32.bf16 v2, v2_bf16; \n\t"
          "cvt.f32.bf16 v3, v3_bf16; \n\t"
          "cvt.f32.bf16 v4, v4_bf16; \n\t"
          "cvt.f32.bf16 v5, v5_bf16; \n\t"
          "cvt.f32.bf16 v6, v6_bf16; \n\t"
          "cvt.f32.bf16 v7, v7_bf16; \n\t"

          ".reg.b64 v01, v23, v45, v67; \n\t"
          "mov.b64 v01, {v0, v1}; \n\t"
          "mov.b64 v23, {v2, v3}; \n\t"
          "mov.b64 v45, {v4, v5}; \n\t"
          "mov.b64 v67, {v6, v7}; \n\t"
          "mul.f32x2 v01, v01, scaling_coeff_2x; \n\t"
          "mul.f32x2 v23, v23, scaling_coeff_2x; \n\t"
          "mul.f32x2 v45, v45, scaling_coeff_2x; \n\t"
          "mul.f32x2 v67, v67, scaling_coeff_2x; \n\t"
          // Elements reordered to match the packing order (v1,v0)
          "mov.b64 {v1, v0}, v01; \n\t"
          "mov.b64 {v3, v2}, v23; \n\t"
          "mov.b64 {v5, v4}, v45; \n\t"
          "mov.b64 {v7, v6}, v67; \n\t"

          ".reg.b8 f0, f1, f2, f3; \n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f2, v4, v5;\n\t"
          "cvt.rn.satfinite.e2m1x2.f32 f3, v6, v7;\n\t"
          "mov.b32 %0, {f0, f1, f2, f3};\n\t"
          "}"
          : "=r"(out_8x)
          : "l"(in03), "l"(in47), "f"(scaling_coefficient));
    } else {
      NVTE_DEVICE_ERROR("Not supported scaling coefficient type.");
    }
  } else {
    NVTE_DEVICE_ERROR(
        "FP4 cvt PTX instructions are architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
  return out_8x;
}

template <typename SCALING_COEFFICIENT_TYPE>
__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_stochastic_rounding(
    const uint64_t in03, const uint64_t in47, const SCALING_COEFFICIENT_TYPE scaling_coefficient,
    const uint32_t rbits03, const uint32_t rbits47) {
  uint32_t out_8x = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, bf16>::value) {
      asm volatile(
          "{\n"
          ".reg.f32 zero; \n\t"
          "mov.b32 zero, 0; \n\t"
          ".reg.b16 scaling_coeff; \n\t"
          "mov.b16 scaling_coeff, %3; \n\t"
          ".reg.b16 v0_h, v1_h, v2_h, v3_h, v4_h, v5_h, v6_h, v7_h; \n\t"
          "mov.b64 {v0_h, v1_h, v2_h, v3_h}, %1; \n\t"
          "mov.b64 {v4_h, v5_h, v6_h, v7_h}, %2; \n\t"

          ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
          "fma.rn.f32.bf16 v0, v0_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v1, v1_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v2, v2_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v3, v3_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v4, v4_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v5, v5_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v6, v6_h, scaling_coeff, zero; \n\t"
          "fma.rn.f32.bf16 v7, v7_h, scaling_coeff, zero; \n\t"

          ".reg.b16 b03, b47; \n\t"
          // Elements reordered to match e2m1x4 packing order (v3,v2,v1,v0)
          "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %4; \n\t"
          "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %5; \n\t"
          "mov.b32 %0, {b03, b47};\n"
          "}"
          : "=r"(out_8x)
          : "l"(in03), "l"(in47), "h"(reinterpret_cast<const uint16_t &>(scaling_coefficient)),
            "r"(rbits03), "r"(rbits47));
    } else if constexpr (std::is_same<SCALING_COEFFICIENT_TYPE, float>::value) {
      asm volatile(
          "{\n"
          ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
          "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
          "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"

          ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
          "cvt.f32.bf16 v0, v0_bf16; \n\t"
          "cvt.f32.bf16 v1, v1_bf16; \n\t"
          "cvt.f32.bf16 v2, v2_bf16; \n\t"
          "cvt.f32.bf16 v3, v3_bf16; \n\t"
          "cvt.f32.bf16 v4, v4_bf16; \n\t"
          "cvt.f32.bf16 v5, v5_bf16; \n\t"
          "cvt.f32.bf16 v6, v6_bf16; \n\t"
          "cvt.f32.bf16 v7, v7_bf16; \n\t"

          "mul.f32 v0, v0, %3; \n\t"
          "mul.f32 v1, v1, %3; \n\t"
          "mul.f32 v2, v2, %3; \n\t"
          "mul.f32 v3, v3, %3; \n\t"
          "mul.f32 v4, v4, %3; \n\t"
          "mul.f32 v5, v5, %3; \n\t"
          "mul.f32 v6, v6, %3; \n\t"
          "mul.f32 v7, v7, %3; \n\t"
          ".reg.b16 b03, b47; \n\t"
          // Elements reordered to match e2m1x4 packing order (v3,v2,v1,v0)
          "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %4; \n\t"
          "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %5; \n\t"
          "mov.b32 %0, {b03, b47};\n"
          "}"
          : "=r"(out_8x)
          : "l"(in03), "l"(in47), "f"(scaling_coefficient), "r"(rbits03), "r"(rbits47));
    } else {
      NVTE_DEVICE_ERROR("Not supported scaling coefficient type.");
    }
  } else {
    constexpr bool known_coeff = std::is_same<SCALING_COEFFICIENT_TYPE, bf16>::value ||
                                 std::is_same<SCALING_COEFFICIENT_TYPE, float>::value;
    if constexpr (known_coeff) {
      const float coeff = static_cast<float>(scaling_coefficient);
      const bf16 *vals03 = reinterpret_cast<const bf16 *>(&in03);
      const bf16 *vals47 = reinterpret_cast<const bf16 *>(&in47);
      float q[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q[i] = stochastic_round_fp4_e2m1(static_cast<float>(vals03[i]) * coeff, rbits03 >> (8 * i));
        q[i + 4] =
            stochastic_round_fp4_e2m1(static_cast<float>(vals47[i]) * coeff, rbits47 >> (8 * i));
      }
      const fp4e2m1x4 lo(make_float4(q[0], q[1], q[2], q[3]));
      const fp4e2m1x4 hi(make_float4(q[4], q[5], q[6], q[7]));
      out_8x = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&lo)) |
               (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(&hi)) << 16);
    } else {
      NVTE_DEVICE_ERROR("Not supported scaling coefficient type.");
    }
  }
  return out_8x;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace ptx
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PTX_ARCH_SPEC_CUH_
