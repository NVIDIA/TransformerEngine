/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC source file for the 32x32 bidimensional (rowwise+colwise) MXFP8
// cast-only kernel (non-warp-specialized variant). The concrete input/output
// element types are substituted at runtime via the __ITYPE__/__OTYPE__
// placeholders (see rtc_dispatch). CUtensorMap is provided as an opaque struct
// by the included header; the host builds the real TMA descriptors.

#include "specialized_quantize_mxfp8.cuh"

using namespace transformer_engine;
namespace specialized =
    transformer_engine::dispatch::mxfp8::quantize_kernel::specialized;  // NOLINT(*)

namespace {
// Substituted at compile time by the host dispatch. Defaults (2, 4, true)
// reproduce the shipped CastTraits<...,true,true> tiling.
using IType = __ITYPE__;
using OType = __OTYPE__;
using BidimTraits =
    specialized::BidimTunableTraits<IType, OType, __NUM_STAGES__, __ITER_N__, __USE_CVT_4X__>;
}  // namespace

// Non-template entry point so the host can request it by a stable name.
__global__ void __launch_bounds__(BidimTraits::numThreads)
    quantize_mxfp8_bidimensional_rtc_kernel(
        const __grid_constant__ CUtensorMap tensor_map_input,
        const __grid_constant__ CUtensorMap tensor_map_rowwise_output,
        const __grid_constant__ CUtensorMap tensor_map_colwise_output, e8m0_t *scales_rowwise,
        e8m0_t *scales_colwise, const float *noop, int32_t rows, int32_t cols,
        int32_t scale_stride_rowwise, int32_t scale_stride_colwise) {
  specialized::quantize_mxfp8_bidimensional_cast_only_body<BidimTraits>(
      tensor_map_input, tensor_map_rowwise_output, tensor_map_colwise_output, scales_rowwise,
      scales_colwise, noop, rows, cols, scale_stride_rowwise, scale_stride_colwise);
}
