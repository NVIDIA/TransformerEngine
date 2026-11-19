/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cutedsl_gemm_swizzled_scales.h
 *  \brief Which NVFP4 1D quantizations the CuTeDSL backend serves with GEMM-swizzled scales.
 *
 *  Host-only, so that the framework, which decides whether the kernel gets the GEMM-swizzled scale
 *  buffers, and the dispatch that accepts them apply the same rule.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_CUTEDSL_GEMM_SWIZZLED_SCALES_H_
#define TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_CUTEDSL_GEMM_SWIZZLED_SCALES_H_

#include <transformer_engine/transformer_engine.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace transformer_engine {
namespace cutedsl_backend {

// Flattened row and column alignment at which the CuTeDSL backend, when enabled, serves an NVFP4
// 1D quantization with GEMM-swizzled scales, or none if it does not serve it. Multiples of 128
// fill whole 128x4 scale tiles, so no padding needs zeroing.
inline std::optional<std::pair<size_t, size_t>> nvfp4_gemm_swizzled_scales_alignment(
    bool with_rht, bool with_2d_quantization, bool row_scaled_nvfp4, bool nvfp4_use_4over6,
    int sm_arch, DType scale_dtype, bool with_amax, bool rowwise, bool stochastic_rounding) {
  if (sm_arch < 100 || sm_arch > 110) {
    return std::nullopt;
  }
  // The stochastic rounding kernels use cvt.rs, which only SM 10.x has.
  if (stochastic_rounding && sm_arch / 10 != 10) {
    return std::nullopt;
  }
  if (with_rht || with_2d_quantization || row_scaled_nvfp4 || nvfp4_use_4over6) {
    return std::nullopt;
  }
  if (scale_dtype != DType::kFloat8E4M3 || !with_amax || !rowwise) {
    return std::nullopt;
  }
  return std::pair<size_t, size_t>{128, 128};
}

}  // namespace cutedsl_backend
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_CUTEDSL_GEMM_SWIZZLED_SCALES_H_
