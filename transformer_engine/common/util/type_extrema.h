/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file type_extrema.h
 *  \brief Device-safe FP4 aliases and TypeExtrema specializations.
 *
 *  Include from inside namespace transformer_engine after fp8e4m3 / fp8e5m2 /
 *  bf16 / fp16 aliases are available. Kept free of host-only dependencies
 *  (no STL) so the same definitions can be used by static builds and by
 *  NVRTC-compiled device kernels.
 *
 *  Requires FP4_TYPE_SUPPORTED to be defined (as in common.h / ptx.cuh) and,
 *  when it is true, <cuda_fp4.h> to be visible.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_TYPE_EXTREMA_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_TYPE_EXTREMA_H_

#if FP4_TYPE_SUPPORTED
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
#endif  // FP4_TYPE_SUPPORTED

namespace detail {

template <typename T>
struct TypeExtrema;

#if FP4_TYPE_SUPPORTED
template <>
struct TypeExtrema<fp4e2m1> {
  static constexpr float max = 6.0f;
  static constexpr float max_inverse = 1.0 / max;
};
#endif  // FP4_TYPE_SUPPORTED

template <>
struct TypeExtrema<fp8e4m3> {
  static constexpr float max = 448.0f;
  static constexpr float max_inverse = 1.0 / max;
};

template <>
struct TypeExtrema<fp8e5m2> {
  static constexpr float max = 57344.0f;
  static constexpr float max_inverse = 1.0 / max;
};

template <>
struct TypeExtrema<bf16> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct TypeExtrema<fp16> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <>
struct TypeExtrema<float> {
  static constexpr float max = 0x1.fffffep127f;  // FLT_MAX
};

}  // namespace detail

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_TYPE_EXTREMA_H_
