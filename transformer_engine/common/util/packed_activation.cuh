/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file packed_activation.cuh
 *  \brief GeLU and dGeLU on packed FP32 pairs, rounding exactly as util/math.h.
 *
 *  util/math.h is compiled with -fmad=true, so nvcc contracts some of its
 *  multiply-adds; these write the same contractions out as FMAs.  tanh follows
 *  libdevice tanhf step for step.  Two elements go through each FP32x2
 *  instruction instead of one.
 */

#ifndef TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_
#define TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_

#include <cuda_runtime.h>

#include "math.h"

namespace transformer_engine {
namespace packed_act {

//! A packed FP32 pair, low element in the low half.
using f32x2 = unsigned long long;

__device__ __forceinline__ f32x2 make_f32x2(float lo, float hi) {
  f32x2 d;
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(lo), "f"(hi));
  return d;
}
__device__ __forceinline__ void unpack_f32x2(f32x2 a, float &lo, float &hi) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(a));
}
__device__ __forceinline__ f32x2 splat_f32x2(float c) { return make_f32x2(c, c); }
__device__ __forceinline__ f32x2 add_f32x2(f32x2 a, f32x2 b) {
  f32x2 d;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
__device__ __forceinline__ f32x2 mul_f32x2(f32x2 a, f32x2 b) {
  f32x2 d;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(d) : "l"(a), "l"(b));
  return d;
}
__device__ __forceinline__ f32x2 fma_f32x2(f32x2 a, f32x2 b, f32x2 c) {
  f32x2 d;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(d) : "l"(a), "l"(b), "l"(c));
  return d;
}

constexpr float kTanhLog2eX2 = 0x1.715476p+1f;
constexpr float kTanhPoly4 = 0x1.01e104p-6f;
constexpr float kTanhPoly3 = -0x1.ac795cp-5f;
constexpr float kTanhPoly2 = 0x1.10b282p-3f;
constexpr float kTanhPoly1 = -0x1.5553dap-2f;
//! 0.6f squared; libdevice switches from the polynomial at |x| >= 0.6.
constexpr float kTanhBranchXSq = 0x1.70a3d8p-2f;

__device__ __forceinline__ float copysign_nonneg(float a, float b) {
  unsigned d;
  asm("lop3.b32 %0, %1, %2, 0x80000000, 0xec;"
      : "=r"(d)
      : "r"(__float_as_uint(b)), "r"(__float_as_uint(a)));
  return __uint_as_float(d);
}

__device__ __forceinline__ f32x2 tanh_f32x2(f32x2 u) {
  float ua, ub;
  unpack_f32x2(u, ua, ub);
  float e0, e1;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(__fmul_rn(fabsf(ua), kTanhLog2eX2)));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(__fmul_rn(fabsf(ub), kTanhLog2eX2)));
  float f0, f1;
  unpack_f32x2(add_f32x2(make_f32x2(e0, e1), splat_f32x2(1.0f)), f0, f1);
  float r0, r1;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(f0));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(f1));
  // Past libdevice's |x| >= 9.010914 clamp, fma(r, -2, 1) already rounds to 1.
  float g0, g1;
  unpack_f32x2(fma_f32x2(make_f32x2(r0, r1), splat_f32x2(-2.0f), splat_f32x2(1.0f)), g0, g1);
  const float xa = copysign_nonneg(g0, ua);
  const float xb = copysign_nonneg(g1, ub);
  const f32x2 s = mul_f32x2(u, u);
  f32x2 p = fma_f32x2(s, splat_f32x2(kTanhPoly4), splat_f32x2(kTanhPoly3));
  p = fma_f32x2(p, s, splat_f32x2(kTanhPoly2));
  p = fma_f32x2(p, s, splat_f32x2(kTanhPoly1));
  p = mul_f32x2(p, s);
  p = fma_f32x2(p, u, u);
  float pa, pb, sa, sb;
  unpack_f32x2(p, pa, pb);
  unpack_f32x2(s, sa, sb);
  return make_f32x2(sa >= kTanhBranchXSq ? xa : pa, sb >= kTanhBranchXSq ? xb : pb);
}

__device__ __forceinline__ f32x2 gelu_f32x2(f32x2 v) {
  const f32x2 u =
      mul_f32x2(v, fma_f32x2(mul_f32x2(splat_f32x2(0.03567741f), v), v, splat_f32x2(0.79788456f)));
  const f32x2 t = tanh_f32x2(u);
  // 0.5f * t is exact, so this fma rounds like 0.5f + 0.5f * t.
  return mul_f32x2(v, fma_f32x2(splat_f32x2(0.5f), t, splat_f32x2(0.5f)));
}

__device__ __forceinline__ f32x2 dgelu_f32x2(f32x2 v) {
  const f32x2 a3 = fma_f32x2(mul_f32x2(splat_f32x2(0.044715f), v), v, splat_f32x2(1.0f));
  const f32x2 a5 = mul_f32x2(mul_f32x2(splat_f32x2(0.79788456f), v), a3);
  const f32x2 t = tanh_f32x2(a5);
  // fma(t, t, -1) is exactly -fma(-t, t, 1) (round-to-nearest is sign-symmetric);
  // the sign folds into the -0.5 below.
  const f32x2 c2_neg = fma_f32x2(t, t, splat_f32x2(-1.0f));
  const f32x2 d3 = fma_f32x2(mul_f32x2(splat_f32x2(0.1070322243f), v), v, splat_f32x2(0.79788456f));
  const f32x2 f = mul_f32x2(mul_f32x2(splat_f32x2(-0.5f), v), mul_f32x2(c2_neg, d3));
  // Halving is exact, so this fma rounds like 0.5f * (1 + t).
  const f32x2 h = fma_f32x2(splat_f32x2(0.5f), t, splat_f32x2(0.5f));
  return add_f32x2(f, h);
}

//! Which util/math.h activation OP is, identified by template matching.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool is_gelu = false;
template <>
constexpr bool is_gelu<Empty, gelu<float, float>> = true;
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool is_dgelu = false;
template <>
constexpr bool is_dgelu<Empty, dgelu<float, float>> = true;

//! OP applied to a packed pair, for the ops that have a packed form.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ f32x2 apply_f32x2(f32x2 v) {
  if constexpr (is_gelu<ParamOP, OP>) {
    return gelu_f32x2(v);
  } else {
    static_assert(is_dgelu<ParamOP, OP>, "No packed form for this activation.");
    return dgelu_f32x2(v);
  }
}

}  // namespace packed_act
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_
