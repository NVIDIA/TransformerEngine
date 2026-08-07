/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file core_nvfp4.cuh
 *  \brief Core functions used in NVFP4.
 */

#ifndef TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_CORE_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <limits>

#include "../../common.h"
#include "../../util/curanddx.hpp"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif  // FP4_TYPE_SUPPORTED

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

// Central runtime-to-compile-time dispatch for NVFP4 scale storage types.
// SWITCH_FP8UE5M3_TYPE_HANDLE adds UE5M3 when the CUDA toolkit supports it.
#define TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(SCALE_DTYPE, SCALE_TYPE, ...)          \
  switch (SCALE_DTYPE) {                                                                  \
    case DType::kFloat8E4M3: {                                                            \
      using SCALE_TYPE = fp8e4m3;                                                         \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
      SWITCH_FP8UE5M3_TYPE_HANDLE(SCALE_TYPE, __VA_ARGS__)                                \
    default: {                                                                            \
      NVTE_ERROR("Unsupported NVFP4 scale dtype ", to_string(SCALE_DTYPE),                \
                 ". Expected Float8E4M3, or Float8UE5M3 when compiled with CUDA 13.4+."); \
    }                                                                                     \
  }

namespace core {

#if FP4_TYPE_SUPPORTED
using namespace ptx;

// Scale-format-specific behavior belongs here rather than in individual kernels.
template <typename ScaleType>
struct NVFP4ScaleTraits {
  static constexpr bool is_supported = false;
  static constexpr bool supports_fp16_error_path = false;
  static constexpr float expected_max = 0.0f;
  static constexpr float headroom_max = 0.0f;
};

template <>
struct NVFP4ScaleTraits<fp8e4m3> {
  // E4M3 scales fit in FP16 and can use the packed E4M3-to-FP16 PTX fast
  // path. UE5M3 scales can exceed the FP16 range, so they retain the generic
  // FP32 error path.
  static constexpr bool is_supported = true;
  static constexpr bool supports_fp16_error_path = true;
  static constexpr float expected_max = 448.0f;
  static constexpr float headroom_max = 256.0f;
};

#if CUDA_VERSION >= 13040
template <>
struct NVFP4ScaleTraits<fp8ue5m3> {
  static constexpr bool is_supported = true;
  static constexpr bool supports_fp16_error_path = false;
  static constexpr float expected_max = 114688.0f;
  static constexpr float headroom_max = 65536.0f;
};
#endif

// Return the effective maximum used to derive the global NVFP4 encode scale.
// SCALE_TYPE_MAX is the resolved maximum for ScaleType (e.g., 448 for E4M3
// or 114688 for UE5M3). The headroom maximum keeps the 1.5x map-to-4 scale
// used by 4over6 within the scale format's representable range.
template <typename ScaleType,
          int SCALE_TYPE_MAX = static_cast<int>(NVFP4ScaleTraits<ScaleType>::expected_max)>
__host__ __device__ constexpr float scale_max() {
  using ScaleTraits = NVFP4ScaleTraits<ScaleType>;
  static_assert(ScaleTraits::is_supported, "Unsupported NVFP4 scale type.");
  if constexpr (ScaleTraits::is_supported) {
    static_assert(detail::TypeExtrema<ScaleType>::max == ScaleTraits::expected_max,
                  "Unexpected NVFP4 scale type maximum.");
    static_assert(SCALE_TYPE_MAX == static_cast<int>(ScaleTraits::expected_max) ||
                      SCALE_TYPE_MAX == static_cast<int>(ScaleTraits::headroom_max),
                  "Unsupported NVFP4 scale type maximum.");
    static_assert(ScaleTraits::headroom_max * 1.5f <= ScaleTraits::expected_max,
                  "NVFP4 4over6 scale headroom exceeds scale type maximum.");
    return static_cast<float>(SCALE_TYPE_MAX);
  } else {
    return 0.0f;
  }
}

// Return the full-range maximum for a runtime scale dtype.
inline float scale_max(const DType scale_dtype) {
  float result = 0.0f;
  TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(scale_dtype, ScaleType,
                                             result = scale_max<ScaleType>();)
  return result;
}

// Return and validate a user-provided maximum for a runtime scale dtype.
inline float scale_max(const DType scale_dtype, const int scale_type_max) {
  float result = 0.0f;
  TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(scale_dtype, ScaleType, {
    using ScaleTraits = NVFP4ScaleTraits<ScaleType>;
    NVTE_CHECK(scale_type_max == static_cast<int>(ScaleTraits::expected_max) ||
                   scale_type_max == static_cast<int>(ScaleTraits::headroom_max),
               "Unsupported maximum for NVFP4 scale dtype.");
    result = static_cast<float>(scale_type_max);
  })
  return result;
}

template <typename ScaleType>
__device__ __forceinline__ ScaleType
compute_decoding_scaling_factor(const float block_amax, const float global_encode_scale) {
  // Compute the per-block decode scale in the selected scale storage type:
  //
  //   block_decode_scale = block_amax / fp4_max
  //   stored_decode_scale = block_decode_scale * global_encode_scale
  //
  // An equivalent, more literal implementation is:
  //
  //   constexpr float rcp_6f = 1.0f / 6.0f;
  //   const float block_decode_scale = block_amax * rcp_6f;
  //   return static_cast<ScaleType>(block_decode_scale * global_encode_scale);
  //
  // Keep the multiplication order below to match the emulation code exactly,
  // while avoiding a direct division by the FP4 maximum.
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f
  constexpr float fp4_max_inv = 1.0f / fp4_max;
  const float decode_scale = block_amax * (global_encode_scale * fp4_max_inv);
  return static_cast<ScaleType>(fminf(decode_scale, TypeExtrema<float>::max));
}

// Compute the global encode scale factor for a given global amax.
// NVFP4 uses the full scale-type range by default. The explicit SCALE_MAX
// template argument lets recipes such as 4over6 reserve encoding headroom.
template <typename ScaleType, int SCALE_MAX = static_cast<int>(detail::TypeExtrema<ScaleType>::max)>
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  static_assert(SCALE_MAX > 0, "NVFP4 scale maximum must be positive.");
  constexpr float fp8_max = static_cast<float>(SCALE_MAX);
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  float global_encode_scale = fp8_max * fp4_max / global_amax;
  // If scale is infinity, return max value of float32
  global_encode_scale = fminf(global_encode_scale, TypeExtrema<float>::max);
  // If global amax is 0 or infinity, return 1
  if (global_amax == 0.0f || global_encode_scale == 0.0f) {
    return 1.0f;
  }
  return global_encode_scale;
}

__device__ __forceinline__ uint32_t get_rbits(
    transformer_engine::curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>
        &rng,
    // philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>: compile-time configurable rounds
    uint4 &random_uint4, int &rnd_idx) {
  if (rnd_idx == 4) {
    rnd_idx = 0;
    random_uint4 = rng.generate4();
  }
  // Treat uint4 as an array of 4x uint32_t elements for indexing
  const uint32_t *const rbits_arr = reinterpret_cast<uint32_t *>(&random_uint4);
  const uint32_t rbits = rbits_arr[rnd_idx++];
  return rbits;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CORE_NVFP4_CUH_
