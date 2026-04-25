/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_nvfp4_1x64_rowwise.cuh
 *  \brief NVFP4 rowwise cast with per-1x64-K-tile S_enc (non-RHT path; no columnwise / GEMM).
 */
#ifndef TRANSFORMER_ENGINE_QUANTIZE_NVFP4_1X64_ROWWISE_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_NVFP4_1X64_ROWWISE_CUH_

#include <transformer_engine/transformer_engine.h>

#include "../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

// Same TE NVFP4 math as quantize_transpose / vector_blockwise, but
// S_enc = (fp8_max*fp4_max)/max|x| over the current 1x64 K-tile in each row
// (per row, for each 64-stride K window).
void quantize_rowwise_1x64_local_encode(const Tensor& input, const Tensor& noop, Tensor* output,
                                        const QuantizationConfig& quant_config, cudaStream_t stream);

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_NVFP4_1X64_ROWWISE_CUH_
