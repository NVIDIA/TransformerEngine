/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_MATH_H_

namespace transformer_engine {

struct Empty {};

struct ClampedSwiGLUParam {
  float limit;
  float alpha = 1.702f;  // Default value for QuickGELU
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

template <typename OType, typename IType>
__device__ inline OType dsilu(const IType val, const Empty& e) {
  const float cval = val;
  return cval * dsigmoid<float, float>(cval, e) + sigmoid<float, float>(cval, e);
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
