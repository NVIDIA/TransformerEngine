/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file packed_activation.cuh
 *  \brief util/math.h activations applied to a pair of FP32 values.
 *
 *  activation_2x<ParamOP, OP> returns exactly {OP(x), OP(y)}. Where an
 *  activation has a packed form, the pair goes through FP32x2 instructions
 *  instead of two scalar evaluations; the other activations are evaluated one
 *  lane at a time.
 *
 *  The packed forms round exactly like util/math.h. That file is compiled with
 *  -fmad=true, so nvcc contracts some of its multiply-adds; the packed forms
 *  write the same contractions out as FMAs, and follow libdevice tanhf and expf
 *  step by step. tests/cpp/util/test_packed_activation.cu checks the match.
 */

#ifndef TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_
#define TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_

#include <cuda_runtime.h>

#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {
namespace packed_activation {

// FP32x2 add/mul/fma instructions in ptx.cuh require SM 10.0+. On older targets the
// activation falls back to the scalar OP. Under --use_fast_math (NVTE_USE_FAST_MATH, set by
// CMakeLists.txt), util/math.h uses the approximate tanhf and flushes denormals, which the
// packed forms do not reproduce, so every activation also falls back to the scalar OP.
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && !(defined NVTE_USE_FAST_MATH)
constexpr bool kEnabled = true;
#else
constexpr bool kEnabled = false;
#endif

__device__ __forceinline__ ptx::floatx2 duplicate(const float c) { return {c, c}; }

__device__ __forceinline__ float ex2_approx_ftz(const float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float rcp_approx_ftz(const float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// tanh computed as libdevice's tanhf computes it, so the two agree bit for bit:
//   |x| >= 0.6:  tanh(x) = copysign(1 - 2 / (2^(2*log2(e)*|x|) + 1), x)
//   |x| <  0.6:  tanh(x) = x + x^3 * (P1 + P2*x^2 + P3*x^4 + P4*x^6)
// The polynomial is libdevice's minimax fit, and the constants below are its
// coefficients, written as hex floats so they are exact. The branch compares x^2,
// which the polynomial needs anyway, against 0.6^2 = 0.36.
constexpr float kTanhTwoLog2e = 0x1.715476p+1f;  // 2 * log2(e)
constexpr float kTanhP4 = 0x1.01e104p-6f;
constexpr float kTanhP3 = -0x1.ac795cp-5f;
constexpr float kTanhP2 = 0x1.10b282p-3f;
constexpr float kTanhP1 = -0x1.5553dap-2f;
constexpr float kTanhPolyMaxXSq = 0x1.70a3d8p-2f;  // 0.6f * 0.6f

__device__ __forceinline__ ptx::floatx2 tanh_2x(const ptx::floatx2 &x) {
  // |x| >= 0.6. Past libdevice's |x| >= 9.010914 clamp, fma(r, -2, 1) already rounds to 1.
  const ptx::floatx2 e = {ex2_approx_ftz(__fmul_rn(fabsf(x.x), kTanhTwoLog2e)),
                          ex2_approx_ftz(__fmul_rn(fabsf(x.y), kTanhTwoLog2e))};
  const ptx::floatx2 d = ptx::add_2x(e, duplicate(1.0f));
  const ptx::floatx2 r = {rcp_approx_ftz(d.x), rcp_approx_ftz(d.y)};
  const ptx::floatx2 large = ptx::fma_2x(r, duplicate(-2.0f), duplicate(1.0f));

  // |x| < 0.6
  const ptx::floatx2 x_sq = ptx::mul_2x(x, x);
  ptx::floatx2 small = ptx::fma_2x(x_sq, duplicate(kTanhP4), duplicate(kTanhP3));
  small = ptx::fma_2x(small, x_sq, duplicate(kTanhP2));
  small = ptx::fma_2x(small, x_sq, duplicate(kTanhP1));
  small = ptx::mul_2x(small, x_sq);
  small = ptx::fma_2x(small, x, x);

  return {x_sq.x >= kTanhPolyMaxXSq ? copysignf(large.x, x.x) : small.x,
          x_sq.y >= kTanhPolyMaxXSq ? copysignf(large.y, x.y) : small.y};
}

// 1 + expf(-x) as nvcc emits it for util/math.h's sigmoid: libdevice expf splits -x * log2(e)
// into an integer k and a remainder f, computes 2^f with ex2, and nvcc contracts the final
// 2^f * 2^k with the + 1 into one fma. k is formed in the mantissa of a float rounded toward
// -infinity (the 12582913 = 1.5 * 2^23 + 1 offset), clamped by saturating x * -log2(e) / 252.
constexpr float kExpNegClampScale = -0x1.77313ap-8f;  // -log2(e) / 252
constexpr float kExpKRange = 252.0f;
constexpr float kExpKBias = 12582913.0f;    // 1.5 * 2^23 + 1
constexpr float kExpKUnbias = 12583039.0f;  // kExpKBias + 126
constexpr float kExpNegLog2eHi = -0x1.715476p+0f;
constexpr float kExpNegLog2eLo = -0x1.4ae0c0p-26f;

__device__ __forceinline__ ptx::floatx2 one_plus_exp_neg_2x(const ptx::floatx2 &x) {
  ptx::floatx2 t = ptx::fma_2x(x, duplicate(kExpNegClampScale), duplicate(0.5f));
  t = {__saturatef(t.x), __saturatef(t.y)};
  const ptx::floatx2 j = ptx::fma_rm_2x(t, duplicate(kExpKRange), duplicate(kExpKBias));
  // -k, exactly: rounding to nearest is sign-symmetric.
  const ptx::floatx2 neg_k = ptx::fma_2x(j, duplicate(-1.0f), duplicate(kExpKUnbias));
  ptx::floatx2 f = ptx::fma_2x(x, duplicate(kExpNegLog2eHi), neg_k);
  f = ptx::fma_2x(x, duplicate(kExpNegLog2eLo), f);
  const ptx::floatx2 two_k = {__uint_as_float(__float_as_uint(j.x) << 23),
                              __uint_as_float(__float_as_uint(j.y) << 23)};
  const ptx::floatx2 e = {ex2_approx_ftz(f.x), ex2_approx_ftz(f.y)};
  return ptx::fma_2x(e, two_k, duplicate(1.0f));
}

//! sigmoid<float, float> on both lanes.
__device__ __forceinline__ ptx::floatx2 sigmoid_2x(const ptx::floatx2 &x) {
  const ptx::floatx2 d = one_plus_exp_neg_2x(x);
  return {__frcp_rn(d.x), __frcp_rn(d.y)};
}

//! silu<float, float> on both lanes.
__device__ __forceinline__ ptx::floatx2 silu_2x(const ptx::floatx2 &x) {
  return ptx::mul_2x(x, sigmoid_2x(x));
}

//! dsilu<float, float> on both lanes: x * s * (1 - s) + s, as x * (s * (1 - s)) + s.
__device__ __forceinline__ ptx::floatx2 dsilu_2x(const ptx::floatx2 &x) {
  const ptx::floatx2 s = sigmoid_2x(x);
  // fma(s, -1, 1) rounds exactly like 1 - s.
  const ptx::floatx2 one_minus_s = ptx::fma_2x(s, duplicate(-1.0f), duplicate(1.0f));
  return ptx::fma_2x(x, ptx::mul_2x(s, one_minus_s), s);
}

//! gelu<float, float> on both lanes.
__device__ __forceinline__ ptx::floatx2 gelu_2x(const ptx::floatx2 &x) {
  const ptx::floatx2 u = ptx::mul_2x(
      x, ptx::fma_2x(ptx::mul_2x(duplicate(0.03567741f), x), x, duplicate(0.79788456f)));
  const ptx::floatx2 t = tanh_2x(u);
  // 0.5f * t is exact, so this fma rounds like 0.5f + 0.5f * t.
  return ptx::mul_2x(x, ptx::fma_2x(duplicate(0.5f), t, duplicate(0.5f)));
}

//! dgelu<float, float> on both lanes.
__device__ __forceinline__ ptx::floatx2 dgelu_2x(const ptx::floatx2 &x) {
  const ptx::floatx2 a3 = ptx::fma_2x(ptx::mul_2x(duplicate(0.044715f), x), x, duplicate(1.0f));
  const ptx::floatx2 a5 = ptx::mul_2x(ptx::mul_2x(duplicate(0.79788456f), x), a3);
  const ptx::floatx2 t = tanh_2x(a5);
  // fma(t, t, -1) is exactly -fma(-t, t, 1) (round-to-nearest is sign-symmetric);
  // the sign folds into the -0.5 below.
  const ptx::floatx2 c2_neg = ptx::fma_2x(t, t, duplicate(-1.0f));
  const ptx::floatx2 d3 =
      ptx::fma_2x(ptx::mul_2x(duplicate(0.1070322243f), x), x, duplicate(0.79788456f));
  const ptx::floatx2 f = ptx::mul_2x(ptx::mul_2x(duplicate(-0.5f), x), ptx::mul_2x(c2_neg, d3));
  // nvcc rounds util/math.h's product f on its own and contracts the addition with the
  // 0.5f * (1 + t) multiply instead: fma(1 + t, 0.5f, f). Halving is exact, so that rounds
  // like this fma followed by the add.
  const ptx::floatx2 h = ptx::fma_2x(duplicate(0.5f), t, duplicate(0.5f));
  return ptx::add_2x(f, h);
}

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool is_gelu = false;
template <>
constexpr bool is_gelu<Empty, gelu<float, float>> = true;

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool is_dgelu = false;
template <>
constexpr bool is_dgelu<Empty, dgelu<float, float>> = true;

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool is_silu = false;
template <>
constexpr bool is_silu<Empty, silu<float, float>> = true;

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
constexpr bool is_dsilu = false;
template <>
constexpr bool is_dsilu<Empty, dsilu<float, float>> = true;

}  // namespace packed_activation

//! {OP(x.x, p), OP(x.y, p)}, through the packed form when OP has one.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ ptx::floatx2 activation_2x(const ptx::floatx2 &x, const ParamOP &p) {
  namespace pa = packed_activation;
  if constexpr (pa::kEnabled && pa::is_gelu<ParamOP, OP>) {
    return pa::gelu_2x(x);
  } else if constexpr (pa::kEnabled && pa::is_dgelu<ParamOP, OP>) {
    return pa::dgelu_2x(x);
  } else if constexpr (pa::kEnabled && pa::is_silu<ParamOP, OP>) {
    return pa::silu_2x(x);
  } else if constexpr (pa::kEnabled && pa::is_dsilu<ParamOP, OP>) {
    return pa::dsilu_2x(x);
  } else {
    return {OP(x.x, p), OP(x.y, p)};
  }
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_
