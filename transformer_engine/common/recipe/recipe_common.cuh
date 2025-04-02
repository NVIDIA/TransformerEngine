/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_RECIPE_RECIPE_COMMON_CUH_
#define TRANSFORMER_ENGINE_RECIPE_RECIPE_COMMON_CUH_

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

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_RECIPE_RECIPE_COMMON_CUH_
