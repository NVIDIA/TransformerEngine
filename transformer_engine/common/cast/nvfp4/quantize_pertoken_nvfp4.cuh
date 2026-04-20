/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_pertoken_nvfp4.cuh
 *  \brief CUDA kernels to cast to NVFP4 with per-token (per-row) global scaling.
 *
 *  Unlike standard NVFP4 quantization which uses a single per-tensor global scale
 *  (amax / (fp8_max * fp4_max)), per-token NVFP4 computes a separate global scale
 *  for each row (token) of the input tensor. This preserves more dynamic range
 *  information per token, improving accuracy for MoE grouped GEMM workloads.
 *
 *  Scaling hierarchy:
 *    x_quantized = round_to_fp4(x / (global_scale[row] * block_scale[row, block]))
 *    x_dequantized = x_quantized * block_scale[row, block] * global_scale[row]
 *
 *  Where:
 *    - global_scale[row] = row_amax / (fp8_max * fp4_max)     [FP32, per-row]
 *    - block_scale[row, block] = block_amax / (fp4_max * global_scale[row])  [FP8 E4M3, per-16-element block]
 *
 *  Output tensors:
 *    - data: uint8 packed FP4 (same as standard NVFP4)
 *    - block_scales: uint8 reinterpreted as FP8 E4M3 (same layout as standard NVFP4)
 *    - per_token_scales: float32 tensor of shape (num_rows,) containing global_scale per row
 *
 *  TODO: Implement the CUDA kernel. The kernel should:
 *    1. Compute per-row amax via parallel reduction
 *    2. Derive per-row global_scale = row_amax / (fp8_max * fp4_max)
 *    3. For each 16-element block: compute block_amax, derive block_scale, quantize to FP4
 *    4. Store per-row global_scale to output tensor
 *
 *  For now, per-token scaling is approximated by using the per-tensor amax
 *  broadcast to all rows. The fused grouped MLP path in TransformerEngine
 *  handles this via the global_scale_tensor parameter in cuDNN Frontend.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace quantize_pertoken_kernel {

using namespace core;

/*
 * Per-token NVFP4 quantization kernel placeholder.
 *
 * Parameters:
 *   input          - Input tensor (rows x cols), high-precision (BF16/FP32)
 *   output_data    - Output packed FP4 data (rows x cols/2), uint8
 *   output_scales  - Output block scales (rows x ceil(cols/16)), FP8 E4M3
 *   output_per_token_scales - Output per-row global scales (rows,), FP32
 *   rows           - Number of rows (tokens)
 *   cols           - Number of columns (hidden dim), must be multiple of 16
 *
 * TODO: Implement kernel body. See quantize_nvfp4.cuh for reference implementation
 *       of the per-tensor variant.
 */

}  // namespace quantize_pertoken_kernel
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
