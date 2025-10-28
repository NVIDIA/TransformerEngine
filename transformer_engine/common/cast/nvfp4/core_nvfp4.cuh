/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

using nvfp4_scale_t = fp8e4m3;

namespace quantization_and_transposition_SF {
#if FP4_TYPE_SUPPORTED
// Used in transpose variant
// Compute per-block E4M3 encoding/decoding scaling factor
__device__ __forceinline__ nvfp4_scale_t compute_decoding_scaling_factor(const float block_amax,
                                                                         const float S_enc) {
  // constexpr float rcp_6f = 1.0f / 6.0f;
  // const float S_dec_b = block_amax * rcp_6f;
  // const nvfp4_scale_t S_dec_b_fp8 = static_cast<nvfp4_scale_t>(S_dec_b * S_enc);
  // return S_dec_b_fp8;
  // NOTE: Divide by 6.0f is not elegant and not efficient.
  // However, this is part of the emulation code to ensure exact match.
  using namespace detail;
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f;
  const float S_dec_b = block_amax / fp4_max * S_enc;
  return static_cast<nvfp4_scale_t>(fminf(S_dec_b, TypeExtrema<float>::max));
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace quantization_and_transposition_SF

namespace quantization_SF {
#if FP4_TYPE_SUPPORTED
// Used in non-transpose variant
// Compute per-block E4M3 encoding/decoding scaling factor
__device__ __forceinline__ fp8e4m3 compute_decoding_scaling_factor(const float block_amax,
                                                                   const float S_enc) {
  constexpr float rcp_6f = 1.0f / 6.0f;
  // const float S_dec_b = block_amax * rcp_6f;
  // const fp8e4m3 S_dec_b_fp8 = static_cast<fp8e4m3>(S_dec_b * S_enc);
  // return S_dec_b_fp8;
  return static_cast<fp8e4m3>(block_amax * rcp_6f * S_enc);
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace quantization_SF

namespace core {

#if FP4_TYPE_SUPPORTED
using namespace ptx;

// Compute the global encode scale factor for a given global amax
__device__ __forceinline__ float compute_global_encode_scaling_factor_FP4(const float global_amax) {
  using namespace detail;
  constexpr float fp8_max = TypeExtrema<fp8e4m3>::max;  // 448.0f;
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

__device__ __forceinline__ uint32_t
get_rbits(transformer_engine::curanddx::detail::philox4x32_native_state<10> &rng,
          // philox4x32_native_state<10>: 10 rounds of philox4_32
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
