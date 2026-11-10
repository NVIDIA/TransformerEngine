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
#include <cstdint>

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

/*! \brief Store \p count consecutive compact scales of one row at their GEMM-swizzled offsets.
 *
 *  \p src holds the scales of compact columns \p col_idx .. \p col_idx + \p count - 1 of row
 *  \p row_idx. In the swizzled layout the scales of columns 4c .. 4c+3 of a row are 4 contiguous bytes
 *  at a 4-aligned offset, so whole groups of four are written with one aligned 32-bit store: the same
 *  bytes at the same addresses as per-scale stores. Misaligned cases fall back to byte stores.
 */
template <typename ScaleT>
__device__ __forceinline__ void store_gemm_swizzled_scales(ScaleT *dst, const ScaleT *src,
                                                           size_t row_idx, size_t col_idx,
                                                           size_t col_length, int count) {
  static_assert(sizeof(ScaleT) == 1, "NVFP4 scales are 1 byte wide.");
  int k = 0;
  if (col_idx % 4 == 0 && reinterpret_cast<uintptr_t>(src) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(dst) % 4 == 0) {
    for (; k + 4 <= count; k += 4) {
      const size_t off = gemm_swizzled_scale_idx(row_idx, col_idx + k, col_length);
      *reinterpret_cast<uint32_t *>(dst + off) = *reinterpret_cast<const uint32_t *>(src + k);
    }
  }
  for (; k < count; ++k) {
    dst[gemm_swizzled_scale_idx(row_idx, col_idx + k, col_length)] = src[k];
  }
}

}  // namespace swizzle
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_NVFP4_SWIZZLE_CUH_
