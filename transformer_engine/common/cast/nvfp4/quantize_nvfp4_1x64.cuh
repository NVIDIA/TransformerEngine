/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_nvfp4_1x64.cuh
 *  \brief NVFP4 hierarchical 1x64 cast (rowwise + optional transposed columnwise),
 *         with per-1x64-K-window S_enc and per-1x16 sub-block E4M3 s_dec.
 *
 *  The kernel produces, for an (M, N) input, any non-empty subset of:
 *    * rowwise NVFP4 data + 1x16 E4M3 scales + (M, N/64) FP32 window amax
 *    * columnwise (transposed) NVFP4 data + 1x16 E4M3 scales + (N, M/64) FP32 window amax
 *
 *  The "window amax" tensors are stored on the existing ``amax`` /
 *  ``columnwise_amax`` slots of the C++ ``Tensor`` (their shape is upgraded
 *  from a scalar (1,) -- as used by the per-tensor NVFP4 path -- to a 2D
 *  per-window buffer in 1x64 mode). Consumers that need the global tensor
 *  amax can take ``max`` over the per-window buffer at trivial cost.
 *
 *  Non-RHT, non-2D, non-stochastic-rounding only. Both M and N are
 *  required to be multiples of 64 by the dispatcher.
 */
#ifndef TRANSFORMER_ENGINE_QUANTIZE_NVFP4_1X64_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_NVFP4_1X64_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

// Hierarchical 1x64 + 1x16 NVFP4 cast. Routes to the fused rowwise+columnwise
// kernel; populates whichever of ``data`` / ``columnwise_data`` are present
// on ``output`` (and the matching scales + window amax buffers).
void quantize_1x64_local_encode(const Tensor& input, const Tensor& noop, Tensor* output,
                                const QuantizationConfig& quant_config, cudaStream_t stream);

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_NVFP4_1X64_CUH_
