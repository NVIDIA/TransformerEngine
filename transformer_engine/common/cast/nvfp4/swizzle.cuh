/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file swizzle.cuh
 *  \brief Helper for emitting GEMM-swizzled NVFP4 scale factors in-kernel.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_SWIZZLE_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_SWIZZLE_CUH_

#include <cstddef>

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace swizzle {

/*! \brief Map compact NVFP4 scale-factor indices to the GEMM-swizzled offset.
 *
 *  cuBLAS block-scaled GEMM consumes scale factors in a "swizzled" 512B-base-block
 *  layout (128 rows x 4 cols per base block, itself split into 4 column blocks of
 *  32 rows x 4 cols):
 *  https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
 *
 *  This converts an index (row_idx, col_idx) in the compact scale matrix -- the
 *  layout that matches the FP4 data -- into the flat offset of the swizzled buffer.
 *  \p col_length is the number of scale columns of the compact matrix (i.e. the
 *  last-dim tile count, K/16 for the rowwise operand or M/16 for the columnwise
 *  operand). This is byte-compatible with nvte_swizzle_scaling_factors and with
 *  the fallback kernel's scale_factor_swizzled_offset.
 */
__device__ __forceinline__ size_t gemm_swizzled_scale_idx(size_t row_idx, size_t col_idx,
                                                          size_t col_length) {
  constexpr size_t kRowsPerBaseBlock = 128;
  constexpr size_t kRowsPerBaseBlockCol = 32;
  constexpr size_t kColsPerBaseBlockCol = 4;

  const size_t rb = row_idx / kRowsPerBaseBlock;
  const size_t rem = row_idx % kRowsPerBaseBlock;
  const size_t d4 = rem / kRowsPerBaseBlockCol;
  const size_t d3 = rem % kRowsPerBaseBlockCol;
  const size_t cbg = col_idx / kColsPerBaseBlockCol;
  const size_t d5 = col_idx % kColsPerBaseBlockCol;

  const size_t cbg_cnt = (col_length + kColsPerBaseBlockCol - 1) / kColsPerBaseBlockCol;
  // Row-major offset in the logical (rb_cnt, cbg_cnt, 32, 4, 4) shape.
  return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol + d5;
}

}  // namespace swizzle
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_SWIZZLE_CUH_
