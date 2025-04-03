/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_RECIPE_RECIPE_COMMON_CUH_
#define TRANSFORMER_ENGINE_RECIPE_RECIPE_COMMON_CUH_

#include "common/common.h"

namespace transformer_engine {

__device__ __forceinline__ float compute_scale_from_amax(float amax, float max_fp8,
                                                         bool force_pow_2_scales, float epsilon,
                                                         float value_for_inf) {
  // NOTE: NAN amax evaluates false for <, handled further down.
  if (amax < epsilon) {
    amax = epsilon;
  }

  float scale = 1.f;

  if (isinf(amax) || amax == 0.f || isnan(amax)) {
    return scale;
  }

  // Here we don't use "scale = max_fp8 / amax" because it has different results with/without
  // "--use_fast_math".
  // "__fdiv_rn" has the same behavior with "max_fp8 / amax" when not using fast math.
  scale = __fdiv_rn(max_fp8, amax);

  // The amax is too small that the scale becoming infinite in FP32. In other word,
  // the scale is not representable in FP32.
  if (isinf(scale)) {
    // use fp32 max to represent the scale
    scale = value_for_inf;
  }
  if (force_pow_2_scales) {
    uint32_t scale_bits = *reinterpret_cast<uint32_t *>(&scale);
    scale_bits &= 0xFF800000;
    // If the exponent was zero, we have a logic error.
    __builtin_assume(scale_bits != 0 || scale == 0.0);
    __builtin_assume(scale_bits != 0x80000000);
    scale = *reinterpret_cast<float *>(&scale_bits);
  }

  return scale;
}

// Calculate the quantization scale for an individual data element
// given the amax(abs(tile)) value for a given quantization tile.
//
//
// Arguments:
// IType: data type of the tensor being quantized (float or bf16)
// OType: quantized data type (e4m3 or e5m2)
// amax: The evaluation of amax(abs(tile)) for the quantization tile.
// eps: An epsilon used as a floor for amax.
// pow_2_scaling: Whether to force the scale to be a power of 2.
template <typename IType, typename OType>
__device__ __forceinline__ float compute_scale_from_types(const float amax, const float eps,
                                                          const float pow_2_scaling) {
  constexpr float fp8_max = TypeInfo<OType>::max_finite_value;
  // NOTE: We're relying on compute_scale_from_amax to have behavior where it
  // clips the mantissa of the max_finite_value if power of 2 scaling applies.
  constexpr float value_for_inf = TypeInfo<IType>::max_finite_value;
  return compute_scale_from_amax(amax, fp8_max, pow_2_scaling, eps, value_for_inf);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_RECIPE_RECIPE_COMMON_CUH_
