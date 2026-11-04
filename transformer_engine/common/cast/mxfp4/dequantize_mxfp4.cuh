/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_mxfp4.cuh
 *  \brief CUDA kernels to dequantize from MXFP4.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_MXFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../mxfp8/swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace dequantize_kernel {

constexpr size_t kBlockSize = 32;
constexpr size_t kThreadsPerBlock = 256;

#if FP4_TYPE_SUPPORTED

// One thread dequantizes one byte (two adjacent elements of a row).
template <typename OType, bool ROWWISE_SCALING, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(kThreadsPerBlock)
    dequantize_mxfp4_kernel(const uint8_t *__restrict__ input, const e8m0_t *__restrict__ scales,
                            OType *__restrict__ output, const size_t rows, const size_t cols,
                            const size_t scales_stride) {
  const size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (idx >= rows * cols / 2) {
    return;
  }
  const size_t row = (2 * idx) / cols;
  const size_t col = (2 * idx) % cols;

  e8m0_t exponent0, exponent1;
  if constexpr (ROWWISE_SCALING) {
    const size_t block = col / kBlockSize;
    const size_t num_tiles_X = DIVUP(cols, static_cast<size_t>(128));
    exponent0 = scales[WITH_GEMM_SWIZZLED_SCALES
                           ? mxfp8::swizzle::gemm_swizzled_scale_idx(row, block, num_tiles_X)
                           : row * scales_stride + block];
    exponent1 = exponent0;
  } else {
    const size_t block = row / kBlockSize;
    const size_t num_tiles_X = DIVUP(rows, static_cast<size_t>(128));
    if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
      exponent0 = scales[mxfp8::swizzle::gemm_swizzled_scale_idx(col, block, num_tiles_X)];
      exponent1 = scales[mxfp8::swizzle::gemm_swizzled_scale_idx(col + 1, block, num_tiles_X)];
    } else {
      exponent0 = scales[block * scales_stride + col];
      exponent1 = scales[block * scales_stride + col + 1];
    }
  }

  const uint8_t byte = input[idx];
  ptx::fp4e2m1x2 packed;
  *reinterpret_cast<uint8_t *>(&packed) = byte;
  const float2 vals = static_cast<float2>(packed);
  const float scale0 = exp2f(static_cast<float>(exponent0) - 127.0f);
  const float scale1 = exp2f(static_cast<float>(exponent1) - 127.0f);
  output[row * cols + col] = static_cast<OType>(vals.x * scale0);
  output[row * cols + col + 1] = static_cast<OType>(vals.y * scale1);
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace dequantize_kernel

inline void dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace dequantize_kernel;
  const bool use_rowwise = input.has_data();
  NVTE_CHECK(use_rowwise || input.has_columnwise_data(), "Input MXFP4 tensor has no data.");
  const SimpleTensor &data = use_rowwise ? input.data : input.columnwise_data;
  const SimpleTensor &scale_inv = use_rowwise ? input.scale_inv : input.columnwise_scale_inv;
  NVTE_CHECK(is_fp4_dtype(data.dtype), "Input must have FP4 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype) && !is_fp4_dtype(output->data.dtype),
             "Output must be in higher precision.");
  NVTE_CHECK(output->shape() == input.shape(), "Input and output shapes need to match.");

  const auto [rows, cols] = input.flat_2d_dims();
  if (rows == 0 || cols == 0) {
    return;
  }
  NVTE_CHECK(cols % 2 == 0, "MXFP4 dequantization requires an even last dimension.");
  const size_t scales_stride = scale_inv.shape.back();
  const size_t num_threads = rows * cols / 2;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->data.dtype, OType,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_rowwise, ROWWISE_SCALING,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              input.with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,
              dequantize_mxfp4_kernel<OType, ROWWISE_SCALING, WITH_GEMM_SWIZZLED_SCALES>
              <<<DIVUP(num_threads, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                  reinterpret_cast<const uint8_t *>(data.dptr),
                  reinterpret_cast<const e8m0_t *>(scale_inv.dptr),
                  reinterpret_cast<OType *>(output->data.dptr), rows, cols,
                  scales_stride););  // NOLINT(*)
      );                             // NOLINT(*)
  );                                 // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("MXFP4 dequantization requires CUDA 12.8 or newer.");
#endif
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_MXFP4_CUH_
