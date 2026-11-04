/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp4.cuh
 *  \brief CUDA kernels to quantize to MXFP4 (E2M1 data, E8M0 scale per 32 elements).
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/ptx.cuh"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "../mxfp8/swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace quantize_kernel {

constexpr size_t kBlockSize = 32;
constexpr size_t kThreadsPerBlock = 256;
constexpr float kE2M1MaxNormRcp = 1.0f / 6.0f;

__device__ __forceinline__ size_t scale_idx(size_t i, size_t j, size_t stride, size_t num_tiles_X,
                                            bool swizzled) {
  return swizzled ? mxfp8::swizzle::gemm_swizzled_scale_idx(i, j, num_tiles_X) : i * stride + j;
}

#if FP4_TYPE_SUPPORTED

// One thread quantizes the 32 consecutive elements of one row.
template <typename IType, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(kThreadsPerBlock)
    quantize_mxfp4_rowwise_kernel(const IType *__restrict__ input, uint8_t *__restrict__ output,
                                  e8m0_t *__restrict__ scales, const float *__restrict__ noop,
                                  const size_t rows, const size_t cols,
                                  const size_t scales_stride) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const size_t blocks_per_row = cols / kBlockSize;
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= rows * blocks_per_row) {
    return;
  }
  const size_t row = idx / blocks_per_row;
  const size_t block = idx % blocks_per_row;
  const IType *in = input + row * cols + block * kBlockSize;

  float vals[kBlockSize];
  float amax = 0.0f;
#pragma unroll
  for (size_t i = 0; i < kBlockSize; ++i) {
    vals[i] = static_cast<float>(in[i]);
    amax = fmaxf(amax, fabsf(vals[i]));
  }
  const e8m0_t biased_exponent = ptx::float_to_e8m0(amax * kE2M1MaxNormRcp);
  const float scale_rcp = ptx::exp2f_rcp<float>(biased_exponent);

  alignas(16) ptx::fp4e2m1x4 packed[kBlockSize / 4];
#pragma unroll
  for (size_t i = 0; i < kBlockSize / 4; ++i) {
    packed[i] =
        ptx::fp4e2m1x4(make_float4(vals[4 * i] * scale_rcp, vals[4 * i + 1] * scale_rcp,
                                   vals[4 * i + 2] * scale_rcp, vals[4 * i + 3] * scale_rcp));
  }
  uint4 *out = reinterpret_cast<uint4 *>(output + (row * cols + block * kBlockSize) / 2);
  out[0] = reinterpret_cast<const uint4 *>(packed)[0];

  const size_t num_tiles_X = DIVUP(cols, static_cast<size_t>(128));
  scales[scale_idx(row, block, scales_stride, num_tiles_X, WITH_GEMM_SWIZZLED_SCALES)] =
      biased_exponent;
}

// One thread quantizes a pair of adjacent columns over 32 consecutive rows.
template <typename IType, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(kThreadsPerBlock)
    quantize_mxfp4_columnwise_kernel(const IType *__restrict__ input, uint8_t *__restrict__ output,
                                     e8m0_t *__restrict__ scales, const float *__restrict__ noop,
                                     const size_t rows, const size_t cols,
                                     const size_t scales_stride) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  const size_t col_pairs = cols / 2;
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= (rows / kBlockSize) * col_pairs) {
    return;
  }
  const size_t block = idx / col_pairs;
  const size_t col = 2 * (idx % col_pairs);
  const IType *in = input + block * kBlockSize * cols + col;

  float vals[kBlockSize][2];
  float amax0 = 0.0f;
  float amax1 = 0.0f;
#pragma unroll
  for (size_t i = 0; i < kBlockSize; ++i) {
    vals[i][0] = static_cast<float>(in[i * cols]);
    vals[i][1] = static_cast<float>(in[i * cols + 1]);
    amax0 = fmaxf(amax0, fabsf(vals[i][0]));
    amax1 = fmaxf(amax1, fabsf(vals[i][1]));
  }
  const e8m0_t biased_exponent0 = ptx::float_to_e8m0(amax0 * kE2M1MaxNormRcp);
  const e8m0_t biased_exponent1 = ptx::float_to_e8m0(amax1 * kE2M1MaxNormRcp);
  const float scale_rcp0 = ptx::exp2f_rcp<float>(biased_exponent0);
  const float scale_rcp1 = ptx::exp2f_rcp<float>(biased_exponent1);

  uint8_t *out = output + (block * kBlockSize * cols + col) / 2;
#pragma unroll
  for (size_t i = 0; i < kBlockSize; ++i) {
    const ptx::fp4e2m1x2 packed(make_float2(vals[i][0] * scale_rcp0, vals[i][1] * scale_rcp1));
    out[i * cols / 2] = *reinterpret_cast<const uint8_t *>(&packed);
  }

  const size_t num_tiles_X = DIVUP(rows, static_cast<size_t>(128));
  if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
    scales[scale_idx(col, block, scales_stride, num_tiles_X, true)] = biased_exponent0;
    scales[scale_idx(col + 1, block, scales_stride, num_tiles_X, true)] = biased_exponent1;
  } else {
    scales[block * scales_stride + col] = biased_exponent0;
    scales[block * scales_stride + col + 1] = biased_exponent1;
  }
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace quantize_kernel

inline void quantize(const Tensor &input, const Tensor *noop, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_kernel;
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output", false);

  const bool use_rowwise = output->has_data();
  const bool use_colwise = output->has_columnwise_data();
  NVTE_CHECK(!use_rowwise || is_fp4_dtype(output->data.dtype),
             "MXFP4 quantization requires FP4 rowwise output.");
  NVTE_CHECK(!use_colwise || is_fp4_dtype(output->columnwise_data.dtype),
             "MXFP4 quantization requires FP4 columnwise output.");
  NVTE_CHECK(!is_fp8_dtype(input.dtype()) && !is_fp4_dtype(input.dtype()),
             "MXFP4 quantization requires high-precision input.");

  const auto [rows, cols] = input.flat_2d_dims();
  if (rows == 0 || cols == 0) {
    return;
  }
  NVTE_CHECK(cols % kBlockSize == 0, "MXFP4 quantization requires the last dimension (", cols,
             ") to be a multiple of ", kBlockSize, ".");
  NVTE_CHECK(!use_colwise || rows % kBlockSize == 0,
             "MXFP4 columnwise quantization requires the first dimensions (", rows,
             ") to be a multiple of ", kBlockSize, ".");

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const bool swizzled = output->with_gemm_swizzled_scales;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          swizzled, WITH_GEMM_SWIZZLED_SCALES,
          const IType *in_ptr = reinterpret_cast<const IType *>(input.data.dptr);
          if (use_rowwise) {
            const size_t num_threads = rows * (cols / kBlockSize);
            const size_t scales_stride = output->scale_inv.shape.back();
            quantize_mxfp4_rowwise_kernel<IType, WITH_GEMM_SWIZZLED_SCALES>
                <<<DIVUP(num_threads, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                    in_ptr, reinterpret_cast<uint8_t *>(output->data.dptr),
                    reinterpret_cast<e8m0_t *>(output->scale_inv.dptr), noop_ptr, rows, cols,
                    scales_stride);
            NVTE_CHECK_CUDA(cudaGetLastError());
          } if (use_colwise) {
            const size_t num_threads = (rows / kBlockSize) * (cols / 2);
            const size_t scales_stride = output->columnwise_scale_inv.shape.back();
            quantize_mxfp4_columnwise_kernel<IType, WITH_GEMM_SWIZZLED_SCALES>
                <<<DIVUP(num_threads, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                    in_ptr, reinterpret_cast<uint8_t *>(output->columnwise_data.dptr),
                    reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr), noop_ptr, rows,
                    cols, scales_stride);
            NVTE_CHECK_CUDA(cudaGetLastError());
          });  // NOLINT(*)
  );           // NOLINT(*)
#else
  NVTE_ERROR("MXFP4 quantization requires CUDA 12.8 or newer.");
#endif
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP4_CUH_
