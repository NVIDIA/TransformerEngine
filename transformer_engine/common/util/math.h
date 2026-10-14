/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_

namespace transformer_engine {

struct Empty {};

struct ClampedSwiGLUParam {
  float limit;
  float alpha = 1.702f;            // Default value for QuickGELU
  float glu_linear_offset = 1.0f;  // Offset added to the linear (gate) component after clamping
};

struct SiTUGLUParam {
  float beta1 = 4.0f;
  float beta2 = 25.0f;
};

template <typename OType, typename IType>
__device__ inline OType gelu(const IType val, const Empty&) {
  const float cval = val;
  return cval * (0.5F + 0.5F * tanhf(cval * (0.79788456F + 0.03567741F * cval * cval)));
}

template <typename OType, typename IType>
__device__ inline OType dgelu(const IType val, const Empty&) {
  const float cval = val;
  const float tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
  return 0.5f * cval * ((1.f - tanh_out * tanh_out) * (0.79788456f + 0.1070322243f * cval * cval)) +
         0.5f * (1.f + tanh_out);
}

template <typename OType, typename IType>
__device__ inline OType sigmoid(const IType val, const Empty&) {
  const float cval = val;
  return 1.f / (1.f + expf(-cval));
}

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

template <typename OType, typename IType>
__device__ inline OType dsigmoid(const IType val, const Empty& e) {
  const float cval = val;
  const float s = sigmoid<float, float>(cval, e);
  return s * (1.f - s);
}

template <typename OType, typename IType>
__device__ inline OType qgelu_with_alpha(const IType val, const float alpha) {
  const float cval = val;
  Empty e = {};
  return cval * sigmoid<float, float>(alpha * cval, e);
}

template <typename OType, typename IType>
__device__ inline OType qgelu(const IType val, const Empty& e) {
  return qgelu_with_alpha<OType, IType>(val, 1.702f);
}

template <typename OType, typename IType>
__device__ inline OType dqgelu_with_alpha(const IType val, const float alpha) {
  const float cval = val;
  Empty e = {};
  return alpha * cval * dsigmoid<float, float>(alpha * cval, e) +
         sigmoid<float, float>(alpha * cval, e);
}

template <typename OType, typename IType>
__device__ inline OType dqgelu(const IType val, const Empty& e) {
  return dqgelu_with_alpha<OType, IType>(val, 1.702f);
}

template <typename OType, typename IType>
__device__ inline OType silu(const IType val, const Empty& e) {
  const float cval = val;
  return cval * sigmoid<float, float>(cval, e);
}

template <typename OType, typename IType>
__device__ inline OType clamped_silu(const IType val, const ClampedSwiGLUParam& p) {
  const float cval = min(p.limit, static_cast<float>(val));  // Clamping
  return qgelu_with_alpha<OType, float>(cval, p.alpha);
}

// clamp(val, -limit, limit). Requires limit > 0: the single-instruction form below
// derives the lower bound from the upper one's sign.
__device__ inline float clamp_symmetric(const float val, const float limit) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 860)
  float clamped;
  asm("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(clamped) : "f"(val), "f"(limit));
  return clamped;
#else
  return min(max(-limit, val), limit);
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 860)
}

// Twice the SiLU, from the identity 2 * silu(x) = x * (1 + tanh(x / 2)).
//
// Returning twice the value is what makes this cheaper than silu: a caller that already
// scales the result by a per-row factor can fold the compensating 0.5 into that factor,
// so the halving costs nothing per element. That leaves one transcendental per element
// against the exponential and reciprocal the exact sigmoid needs. Approximate, so only
// worth using where the output is quantized to few mantissa bits.
template <typename OType, typename IType>
__device__ inline OType silu_approx_x2(const IType val, const Empty&) {
  const float cval = val;
  float tanh_half;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_half) : "f"(0.5f * cval));
#else
  tanh_half = tanhf(0.5f * cval);
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  return fmaf(cval, tanh_half, cval);
}

// Twice the clamped SiLU; see silu_approx_x2 for why it returns twice the value.
template <typename OType, typename IType>
__device__ inline OType clamped_silu_approx_x2(const IType val, const ClampedSwiGLUParam& p) {
  const float cval = min(p.limit, static_cast<float>(val));  // Clamping
  const float half_alpha = 0.5f * p.alpha;
  float tanh_half;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  asm("tanh.approx.f32 %0, %1;" : "=f"(tanh_half) : "f"(half_alpha * cval));
#else
  tanh_half = tanhf(half_alpha * cval);
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  return fmaf(cval, tanh_half, cval);
}

template <typename OType, typename IType>
__device__ inline OType dsilu(const IType val, const Empty& e) {
  const float cval = val;
  return cval * dsigmoid<float, float>(cval, e) + sigmoid<float, float>(cval, e);
}

template <typename OType, typename IType>
__device__ inline OType situ_gate(const IType val, const SiTUGLUParam& p) {
  const float x = static_cast<float>(val);
  Empty e = {};
  return p.beta1 * tanhf(x / p.beta1) * sigmoid<float, float>(x, e);
}

template <typename OType, typename IType>
__device__ inline OType dsitu_gate(const IType val, const SiTUGLUParam& p) {
  const float x = static_cast<float>(val);
  const float t = tanhf(x / p.beta1);
  Empty e = {};
  const float s = sigmoid<float, float>(x, e);
  return (1.0f - t * t) * s + p.beta1 * t * s * (1.0f - s);
}

template <typename OType, typename IType>
__device__ inline OType situ_up(const IType val, const SiTUGLUParam& p) {
  const float x = static_cast<float>(val);
  return p.beta2 * tanhf(x / p.beta2);
}

template <typename OType, typename IType>
__device__ inline OType dsitu_up(const IType val, const SiTUGLUParam& p) {
  const float x = static_cast<float>(val);
  const float t = tanhf(x / p.beta2);
  return 1.0f - t * t;
}

template <typename OType, typename IType>
__device__ inline OType clamped_dsilu(const IType val, const ClampedSwiGLUParam& p) {
  const bool dclamp_val = static_cast<float>(val) <= p.limit;
  const float clamp_val = min(static_cast<float>(val), p.limit);
  const float dsilu_val = dqgelu_with_alpha<OType, float>(clamp_val, p.alpha);
  return dclamp_val ? dsilu_val : 0.0f;
}

template <typename OType, typename IType>
__device__ inline OType relu(IType value, const Empty&) {
  return fmaxf(value, 0.f);
}

template <typename OType, typename IType>
__device__ inline OType drelu(IType value, const Empty&) {
  return value > 0.f ? 1.f : 0.f;
}

template <typename OType, typename IType>
__device__ inline OType srelu(IType value, const Empty&) {
  return value > 0 ? value * value : 0.f;
}

template <typename OType, typename IType>
__device__ inline OType dsrelu(IType value, const Empty&) {
  return fmaxf(2.f * value, 0.f);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_
