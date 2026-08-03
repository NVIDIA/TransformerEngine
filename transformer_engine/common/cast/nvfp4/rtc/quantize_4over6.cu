/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC source file for the NVFP4 4over6 quantize kernel

#include "quantize_4over6_nvfp4.cuh"

using namespace transformer_engine;
using namespace transformer_engine::dispatch::nvfp4;
using namespace transformer_engine::dispatch::nvfp4::quantize_4over6_kernel;

namespace {
// Substituted at compile time by the host dispatch.
using IType = __ITYPE__;
using Cfg = Config<__MODE__, __ERR_FAST_MATH__>;
}  // namespace

__global__ void __launch_bounds__(kThreads) quantize_4over6_rtc_kernel(
    const IType *input, fp4e2m1x2 *output, fp4e2m1x2 *output_t, nvfp4_scale_t *scales,
    nvfp4_scale_t *scales_t, const float *amax_rowwise, const float *amax_colwise,
    const size_t rows, const size_t cols, const size_t scale_stride, const size_t scale_stride_t,
    const float *noop) {
  quantize_4over6_body<__USE_2D__, __RETURN_IDENTITY__, __RETURN_TRANSPOSE__, __ROW_SCALED__, Cfg,
                       __E4M3_MAX__, IType>(input, output, output_t, scales, scales_t, amax_rowwise,
                                            amax_colwise, rows, cols, scale_stride, scale_stride_t,
                                            noop);
}
