#ifndef TRANSFORMER_ENGINE_COMPUTE_SCALE_CUH_
#define TRANSFORMER_ENGINE_COMPUTE_SCALE_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <limits>

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

  // Clamping amax to avoid division by small numbers
  float amax_mod = fmaxf(amax, eps);

  // Handle overflow cases for non-clamped amax (eps is 0 or very small)
  if (amax_mod == 0.f) {
    // If amax is 0, return 1
    return 1.f;
  }
  // Compute scale factor
  float scale = fp8_max / amax_mod;

  if (isinf(scale)) {
    // If scale is infinity, return max value of IType
    return HighPrecisionFloatScaleLimitsTrait<IType, Power2Scaling>::max;
  }
  if (scale == 0.0) {
    // Case that amax is "inf". The frexp, ldexp logic changes 0.0 scales.
    // Return 0.0 for 0.0 scale here is consistent with non-Power2Scaling model.
    // quantization will remove signal from the tensor,
    // this is bad for the model, but define pow2Scale behavior
    // as returning 0.0 scale. amax calculation can
    // improve the situation to avoid this by taking largest finite.
    return scale;
  }
  if constexpr (Power2Scaling) {
    // NOTE: using bit fiddling based on advice of Asit in this
    // thread: https://nvidia.slack.com/archives/C06EDT7LZEW/p1738274404153439

    // inf scales already early returned, as did nan scales.
    // The cases to consider here are normals, zero, and subnormals.
    // zero is not possible with current math as
    // 448.0 / float_max == 1.31655e-36, which is the smallest
    // possible scale given current dtypes. It is still in the normal
    // fp32 range with an exponent of -120, so subnormals are also
    // not possible. To handle normals, we can simply mask off the
    // mantissa.
    uint32_t scale_bits = *reinterpret_cast<uint32_t*>(&scale);
    scale_bits &= 0xFF800000;
    // If the exponent was zero, we have a logic error.
    __builtin_assume(scale_bits != 0);
    __builtin_assume(scale_bits != 0x80000000);
    scale = *reinterpret_cast<float*>(&scale_bits);
  }
  return scale;
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMPUTE_SCALE_CUH_
