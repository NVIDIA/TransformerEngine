/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_pertoken_nvfp4.cuh
 *  \brief CUDA kernels to cast to NVFP4 with per-token (per-row) global scaling.
 *
 *  Unlike standard NVFP4 quantization which uses a single per-tensor global scale,
 *  per-token NVFP4 computes a separate global scale for each row. This preserves
 *  more dynamic range per token, improving accuracy for MoE workloads.
 *
 *  Scaling hierarchy:
 *    global_scale[row] = row_amax / (fp8_max * fp4_max)
 *    block_scale[row, block] = block_amax / (fp4_max * global_scale[row])
 *    x_fp4 = quantize_to_fp4(x / (global_scale[row] * block_scale[row, block]))
 *
 *  Based on the approach from FlashInfer (flashinfer-ai/flashinfer#3027):
 *  two-pass design with one CUDA block per row.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace quantize_pertoken_kernel {

using namespace core;

constexpr int PERTOKEN_BLOCK_SIZE = 256;
constexpr int PERTOKEN_SF_VEC_SIZE = 16;

/*
 * Per-token NVFP4 quantization kernel.
 *
 * One CUDA block per row. Two passes:
 *   Pass 1: Vectorized load + per-row amax reduction via cub::BlockReduce
 *   Pass 2: Reload data, compute per-block E4M3 scale, quantize to FP4
 *
 * Template parameters:
 *   IType           - Input type (half, __nv_bfloat16)
 *   BLOCK_SIZE      - Threads per block
 *
 * Parameters:
 *   num_rows        - Number of rows (tokens)
 *   num_cols        - Number of columns (hidden dim), must be multiple of 16
 *   input           - Input tensor (num_rows, num_cols), IType
 *   row_offsets     - Optional row index remapping (for MoE expert routing), or nullptr
 *   output_data     - Output packed FP4 data (num_rows, num_cols/2), uint8
 *   output_scales   - Output block scales, fp8e4m3
 *   output_per_token_scales - Output per-row global scales (num_rows,), FP32
 *   scale_stride    - Stride of scale factor output (number of SF vectors per row)
 */
template <typename IType, int BLOCK_SIZE>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__launch_bounds__(BLOCK_SIZE)
#endif
    quantize_pertoken_nvfp4_kernel(
        const int num_rows, const int num_cols, const IType *__restrict__ input,
        const int *__restrict__ row_offsets,  // optional: nullptr for identity mapping
        uint8_t *__restrict__ output_data, fp8e4m3 *__restrict__ output_scales,
        float *__restrict__ output_per_token_scales, const int scale_stride) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  using namespace detail;
  constexpr float fp8_max = TypeExtrema<fp8e4m3>::max;  // 448.0f
  constexpr float fp4_max = TypeExtrema<fp4e2m1>::max;  // 6.0f
  constexpr float fp4_max_inv = 1.0f / fp4_max;

  // Packed type: 4 elements per float2 pair for FP4 conversion
  using IType2 =
      typename std::conditional<std::is_same<IType, half>::value, half2, __nv_bfloat162>::type;

  const int row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  // Optional row remapping (for MoE routing)
  const int actual_row = (row_offsets != nullptr) ? row_offsets[row_idx] : row_idx;
  if (actual_row < 0) return;

  const int num_vec2 = num_cols / 2;  // number of IType2 elements per row
  const IType2 *input_row = reinterpret_cast<const IType2 *>(input + actual_row * num_cols);

  // =========================================================================
  // Pass 1: Per-row amax reduction
  // =========================================================================
  float thread_max = 0.0f;
  for (int i = threadIdx.x; i < num_vec2; i += BLOCK_SIZE) {
    IType2 val = input_row[i];
    float2 fval;
    if constexpr (std::is_same_v<IType, half>) {
      fval = __half22float2(val);
    } else {
      fval = __bfloat1622float2(val);
    }
    thread_max = fmaxf(thread_max, fabsf(fval.x));
    thread_max = fmaxf(thread_max, fabsf(fval.y));
  }

  // Block-wide max reduction
  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float row_amax =
      BlockReduce(temp_storage).Reduce(thread_max, [](float a, float b) { return fmaxf(a, b); });

  // Compute and store per-token global scale
  // global_scale = row_amax / (fp8_max * fp4_max)
  // S_enc = fp8_max * fp4_max / row_amax  (encoding scale, inverse of global_scale)
  __shared__ float shared_s_enc;
  if (threadIdx.x == 0) {
    float s_enc = compute_global_encode_scaling_factor_FP4(row_amax);
    float global_scale = (s_enc > 0.0f) ? (1.0f / s_enc) : 0.0f;
    output_per_token_scales[row_idx] = global_scale;
    shared_s_enc = s_enc;
  }
  __syncthreads();
  const float S_enc = shared_s_enc;

  // =========================================================================
  // Pass 2: Compute block scales and quantize to FP4
  // =========================================================================
  // Each thread processes one 16-element block: computes block amax,
  // derives E4M3 block scale, then quantizes 16 elements to 8 packed FP4 bytes.
  const int num_sf_blocks = num_cols / PERTOKEN_SF_VEC_SIZE;

  for (int sf_idx = threadIdx.x; sf_idx < num_sf_blocks; sf_idx += BLOCK_SIZE) {
    const int col_start = sf_idx * PERTOKEN_SF_VEC_SIZE;

    // Load 16 elements and find block amax
    float block_max = 0.0f;
    float vals[PERTOKEN_SF_VEC_SIZE];
    for (int j = 0; j < PERTOKEN_SF_VEC_SIZE; j++) {
      if constexpr (std::is_same_v<IType, half>) {
        vals[j] = __half2float(input[actual_row * num_cols + col_start + j]);
      } else {
        vals[j] = __bfloat162float(input[actual_row * num_cols + col_start + j]);
      }
      block_max = fmaxf(block_max, fabsf(vals[j]));
    }

    // Compute per-block E4M3 scale factor: S_dec_b = block_max / (fp4_max / S_enc)
    fp8e4m3 S_dec_b = quantization_SF::compute_decoding_scaling_factor(block_max, S_enc);
    float S_dec_b_f = static_cast<float>(S_dec_b);

    // Store block scale (LINEAR layout: row-major)
    output_scales[row_idx * scale_stride + sf_idx] = S_dec_b;

    // Compute encoding scale for this block: maps input range to [-6, 6] (FP4 range)
    float block_encode_scale = (S_dec_b_f != 0.0f)
                                   ? __fdividef(S_enc, S_dec_b_f)
                                   : 0.0f;

    // Scale values and pack to FP4 using PTX cvt.rn.satfinite.e2m1x2
    // Process 8 elements (4 pairs) at a time -> 4 bytes -> 1 uint32_t
    // Matching FlashInfer's fp32_vec_to_e2m1 pattern.
    uint8_t *out_ptr = output_data + actual_row * (num_cols / 2) + col_start / 2;
    for (int j = 0; j < PERTOKEN_SF_VEC_SIZE; j += 8) {
      float s0 = vals[j]     * block_encode_scale;
      float s1 = vals[j + 1] * block_encode_scale;
      float s2 = vals[j + 2] * block_encode_scale;
      float s3 = vals[j + 3] * block_encode_scale;
      float s4 = vals[j + 4] * block_encode_scale;
      float s5 = vals[j + 5] * block_encode_scale;
      float s6 = vals[j + 6] * block_encode_scale;
      float s7 = vals[j + 7] * block_encode_scale;
      uint32_t packed;
      asm volatile(
          "{\n"
          ".reg .b8 byte0, byte1, byte2, byte3;\n"
          "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
          "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
          "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
          "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
          "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
          "}\n"
          : "=r"(packed)
          : "f"(s0), "f"(s1), "f"(s2), "f"(s3),
            "f"(s4), "f"(s5), "f"(s6), "f"(s7));
      reinterpret_cast<uint32_t *>(out_ptr)[j / 8] = packed;
    }
    // Handle remaining 8 elements (PERTOKEN_SF_VEC_SIZE=16, so exactly 2 iterations of 8)
    // The loop above covers j=0..7 and j=8..15, so all 16 elements are handled.
  }
#endif  // __CUDA_ARCH__ >= 1000
}

/*
 * Host-side launcher for per-token NVFP4 quantization.
 */
template <typename IType>
void launch_quantize_pertoken_nvfp4(const int num_rows, const int num_cols, const IType *input,
                                    const int *row_offsets, uint8_t *output_data,
                                    fp8e4m3 *output_scales, float *output_per_token_scales,
                                    cudaStream_t stream) {
  if (num_rows == 0 || num_cols == 0) return;

  NVTE_CHECK(num_cols % PERTOKEN_SF_VEC_SIZE == 0, "num_cols must be a multiple of ",
             PERTOKEN_SF_VEC_SIZE, " for per-token NVFP4 quantization, got ", num_cols);

  const int scale_stride = num_cols / PERTOKEN_SF_VEC_SIZE;
  dim3 grid(num_rows);
  dim3 block(PERTOKEN_BLOCK_SIZE);

  quantize_pertoken_nvfp4_kernel<IType, PERTOKEN_BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(num_rows, num_cols, input, row_offsets, output_data,
                                   output_scales, output_per_token_scales, scale_stride);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace quantize_pertoken_kernel
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_PERTOKEN_NVFP4_CUH_
