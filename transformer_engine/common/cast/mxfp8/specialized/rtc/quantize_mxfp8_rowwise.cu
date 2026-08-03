/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC source file for the 1x32 rowwise MXFP8 cast-only kernel. The host
// bundles this (and the specialized kernel headers it needs) as in-memory
// strings; the concrete input/output element types are substituted at runtime
// via the __ITYPE__/__OTYPE__ placeholders (see rtc_dispatch).

#include "specialized_quantize_mxfp8.cuh"

using namespace transformer_engine;
namespace specialized =
    transformer_engine::dispatch::mxfp8::quantize_kernel::specialized;  // NOLINT(*)

namespace {
// Substituted at compile time by the host dispatch.
using IType = __ITYPE__;
using OType = __OTYPE__;
using RowwiseTraits = specialized::CastTraits<IType, OType, /*rowwise=*/true, /*colwise=*/false>;
}  // namespace

// Non-template entry point so the host can request it by a stable name.
__global__ void __launch_bounds__(RowwiseTraits::numThreads) quantize_mxfp8_rowwise_rtc_kernel(
    IType *__restrict__ input, OType *__restrict__ output, e8m0_t *__restrict__ scales_rowwise,
    const float *noop, int32_t rows, int32_t cols, int32_t scale_stride_rowwise,
    int32_t scale_stride_colwise) {
  specialized::quantize_mxfp8_rowwise_cast_only_body<RowwiseTraits>(
      input, output, scales_rowwise, noop, rows, cols, scale_stride_rowwise,
      scale_stride_colwise);
}
