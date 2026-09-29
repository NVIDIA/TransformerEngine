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
 *  write the same contractions out as FMAs, and follow libdevice tanhf step by
 *  step. tests/cpp/util/test_packed_activation.cu checks the match.
 */

#ifndef TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_
#define TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_

#include <cuda_runtime.h>

#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {
namespace packed_activation {

// Under --use_fast_math (NVTE_USE_FAST_MATH, set by CMakeLists.txt), util/math.h uses the
// approximate tanhf and flushes denormals, which the packed forms do not reproduce, so every
// activation falls back to the scalar OP.
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
  // Halving is exact, so this fma rounds like 0.5f * (1 + t).
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

}  // namespace packed_activation

//! {OP(x.x, p), OP(x.y, p)}, through the packed form when OP has one.
template <typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ ptx::floatx2 activation_2x(const ptx::floatx2 &x, const ParamOP &p) {
  namespace pa = packed_activation;
  if constexpr (pa::kEnabled && pa::is_gelu<ParamOP, OP>) {
    return pa::gelu_2x(x);
  } else if constexpr (pa::kEnabled && pa::is_dgelu<ParamOP, OP>) {
    return pa::dgelu_2x(x);
  } else {
    return {OP(x.x, p), OP(x.y, p)};
  }
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_PACKED_ACTIVATION_CUH_
