/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file swizzle.cuh
 *  \brief Helper function for GEMM-swizzled scales
 */

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_SWIZZLE_CUH_
#define TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_SWIZZLE_CUH_

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace swizzle {

/*! \brief Convert compact scale indices into GEMM swizzled scale index
 *
 *  MXFP8 GEMM expects scaling factors to be in a "swizzled" order
 *  (https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout).
 *  This function converts indices from "compact" order (i.e. matching
 *  the FP8 data) to swizzled order.
 *
 */
__device__ __forceinline__ size_t gemm_swizzled_scale_idx(size_t i, size_t j, size_t num_tiles_X) {
  constexpr size_t TILE_DIM_X = 4;  // Tile dim in scale buffer
  constexpr size_t TILE_DIM_Y = 128;
  constexpr size_t TILE_SIZE = TILE_DIM_X * TILE_DIM_Y;
  const size_t tile_idx_X = j / TILE_DIM_X;
  const size_t tile_idx_Y = i / TILE_DIM_Y;
  const size_t idx_in_tile_X = j % TILE_DIM_X;
  const size_t idx_in_tile_Y = i % TILE_DIM_Y;
  size_t idx = (tile_idx_Y * num_tiles_X + tile_idx_X) * TILE_SIZE;
  idx += (idx_in_tile_Y % 32) * 16 + (idx_in_tile_Y / 32) * 4 + idx_in_tile_X;
  return idx;
}

}  // namespace swizzle
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_MXFP8_SWIZZLE_CUH_
