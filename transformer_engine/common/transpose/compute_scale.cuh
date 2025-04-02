/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMPUTE_SCALE_CUH_
#define TRANSFORMER_ENGINE_COMPUTE_SCALE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <limits>

#include "../recipe/recipe_common.cuh"

namespace transformer_engine {

// Type trait for extreme values of fp8 types.
// Used in the calculation of scale factors
// as a constexpr lookup from e4m3 or e5m2 to
// the max finite value.
template <typename T>
struct F8LimitsTrait;

template <>
struct F8LimitsTrait<__nv_fp8_e4m3> {
  static constexpr float max = 448.0f;
};

template <>
struct F8LimitsTrait<__nv_fp8_e5m2> {
  static constexpr float max = 57344.0f;
};

// Type trait to resolve the max finite value
// represented by a input type to quantization.
// Or to represent max representable power of 2
// finite value.
template <typename T, bool ForcePow2>
struct HighPrecisionFloatScaleLimitsTrait;

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, false> {
  static constexpr float max = std::numeric_limits<float>::max();
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<float, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, false> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<nv_bfloat16, true> {
  // Hex float format of 1.0 * 2 ^ 127
  static constexpr float max = 0x1.0p127;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, false> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <>
struct HighPrecisionFloatScaleLimitsTrait<half, true> {
  // Hex float format of 1.0 * 2 ^ 15
  static constexpr float max = 0x1.0p15;
};

// Calculate the quantization scale for an individual data element
// given the amax(abs(tile)) value for a given quantization tile.
//
//
// Arguments:
// IType: data type of the tensor being quantized (float or bf16)
// OType: quantized data type (e4m3 or e5m2)
// pow_2_scaling: Whether to force the scale to be a power of 2.
// amax: The evaluation of amax(abs(tile)) for the quantization tile.
// eps: An epsilon used as a floor for amax.
template <typename IType, typename OType, bool Power2Scaling>
__device__ __forceinline__ float ComputeScale(const float amax, const float eps) {
  constexpr float fp8_max = F8LimitsTrait<OType>::max;
  constexpr float value_for_inf = HighPrecisionFloatScaleLimitsTrait<IType, Power2Scaling>::max;
  return compute_scale_from_amax(amax, fp8_max, Power2Scaling, eps, value_for_inf);
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMPUTE_SCALE_CUH_
