/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_dequantize_fp8_blockwise.cuh
 *  \brief CUDA kernels to dequantize grouped tensors from FP8 with 1D/2D
 *  block scaling (rowwise or columnwise) back to BF16 / FP16 / FP32. Mirrors
 *  the per-expert layouts written by ``group_quantize_fp8_blockwise``.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_FP8_BLOCKWISE_CUH_
#define TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_FP8_BLOCKWISE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../utils.cuh"
#include "../core/common.cuh"
#include "group_quantize_fp8_blockwise.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8_blockwise {

namespace group_dequantize_kernel {

// Resolve which expert a tile (blocks_X x total_row_blocks grid) belongs to and its row range.
// tensor_M (the expert's first-dim length) addresses the per-expert (K, M_t) transposed block
// in the columnwise modes.
struct TileExpertInfo {
  size_t tensor_id;
  size_t tensor_block_y_base;
  size_t tensor_row_blocks;
  size_t tensor_row_base;
  size_t tensor_M;
  bool in_bounds;
};

template <bool kSameBothDims>
__device__ __forceinline__ TileExpertInfo resolve_tile_expert(
    size_t tile_y_global, size_t num_tensors, size_t common_first_dim_blocks, size_t K,
    size_t total_row_blocks, const int64_t* __restrict__ tensor_offsets_ptr) {
  TileExpertInfo info{};
  info.in_bounds = false;
  if (tile_y_global >= total_row_blocks) return info;
  const size_t tile_row_stride = static_cast<size_t>(kTileDim) * K;
  info.tensor_id = find_tensor_id_by_block_y<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
  info.tensor_block_y_base =
      kSameBothDims
          ? (info.tensor_id * common_first_dim_blocks)
          : tensor_block_y_base_from_offsets(info.tensor_id, tensor_offsets_ptr, tile_row_stride);
  info.tensor_row_blocks = kSameBothDims
                               ? common_first_dim_blocks
                               : (tensor_block_y_base_from_offsets(
                                      info.tensor_id + 1, tensor_offsets_ptr, tile_row_stride) -
                                  info.tensor_block_y_base);
  if (tile_y_global >= info.tensor_block_y_base + info.tensor_row_blocks) return info;
  info.tensor_row_base = info.tensor_block_y_base * kTileDim;
  info.tensor_M = info.tensor_row_blocks * kTileDim;
  info.in_bounds = true;
  return info;
}

// ===== 1D rowwise =====
// Per-expert scale layout: (blocks_X, roundup(M_t, 4)) floats.
// scale[expert_off + tile_x * roundup(M_t, 4) + r_local]
template <bool kSameBothDims, typename IType, typename OType, typename CType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_dequantize_blockwise_1d_rw_kernel(const IType* __restrict__ input_base,
                                            OType* __restrict__ output_base,
                                            const CType* __restrict__ scale_inv_base,
                                            const int64_t* __restrict__ tensor_offsets_ptr,
                                            const size_t num_tensors,
                                            const size_t common_first_dim_blocks, const size_t K,
                                            const size_t total_row_blocks, const size_t R_total) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  const auto info = resolve_tile_expert<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, K, total_row_blocks, tensor_offsets_ptr);
  if (!info.in_bounds) return;

  const size_t blocks_X = DIVUP(K, static_cast<size_t>(kTileDim));
  const size_t tile_row_stride = static_cast<size_t>(kTileDim) * K;
  const size_t expert_offset = expert_scale_offset_1d_rowwise<kSameBothDims>(
      info.tensor_id, blocks_X, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
  const size_t per_expert_stride = DIVUP_TO_MULTIPLE(info.tensor_M, kScaleColAlign);
  const CType* const tile_scale_inv_base =
      scale_inv_base + expert_offset + tile_x * per_expert_stride;

  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  constexpr int kThreadsPerRow = 8;
  constexpr int kVec = 16;
  constexpr int kRowsPerIter = kThreadsPerBlock / kThreadsPerRow;  // 32
  constexpr int kIters = kTileDim / kRowsPerIter;                  // 4

  const int tid = threadIdx.x;
  const int thr_col = tid % kThreadsPerRow;
  const int thr_row = tid / kThreadsPerRow;
  const size_t c = global_col_base + static_cast<size_t>(thr_col) * kVec;

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int row_local = thr_row + it * kRowsPerIter;
    const size_t r_global = global_row_base + row_local;
    if (r_global >= R_total) continue;

    const size_t r_local = r_global - info.tensor_row_base;
    const CType s_inv = tile_scale_inv_base[r_local];

    Vec<IType, kVec> in_vec;
    if (c + kVec <= K) {
      in_vec.load_from(input_base + r_global * K + c);
    } else if (c < K) {
      in_vec.load_from_elts(input_base + r_global * K + c, 0, K - c);
    } else {
      continue;
    }

    Vec<OType, kVec> out_vec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      out_vec.data.elt[e] = static_cast<OType>(static_cast<float>(in_vec.data.elt[e]) * s_inv);
    }

    if (c + kVec <= K) {
      out_vec.store_to(output_base + r_global * K + c);
    } else if (c < K) {
      out_vec.store_to_elts(output_base + r_global * K + c, 0, K - c);
    }
  }
#endif
}

// ===== 1D columnwise =====
// Data layout per-expert: (K, M_t) transposed -- element (r, c) is at
//   input[tensor_row_base * K + c * tensor_M + r_local].
// Scale layout GLOBAL: (total_row_blocks, roundup(K, 4)) floats.
//   scale_inv[tile_y_global * scale_t_stride_aligned_K + c]
template <bool kSameBothDims, typename IType, typename OType, typename CType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_dequantize_blockwise_1d_cw_kernel(const IType* __restrict__ input_base,
                                            OType* __restrict__ output_base,
                                            const CType* __restrict__ scale_inv_base,
                                            const size_t scale_t_stride_aligned_K,
                                            const int64_t* __restrict__ tensor_offsets_ptr,
                                            const size_t num_tensors,
                                            const size_t common_first_dim_blocks, const size_t K,
                                            const size_t total_row_blocks, const size_t R_total) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  const auto info = resolve_tile_expert<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, K, total_row_blocks, tensor_offsets_ptr);
  if (!info.in_bounds) return;

  const size_t expert_data_off = info.tensor_row_base * K;
  const CType* const tile_scale_base = scale_inv_base + tile_y_global * scale_t_stride_aligned_K;

  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  constexpr int kThreadsPerRow = 8;
  constexpr int kVec = 16;
  constexpr int kRowsPerIter = kThreadsPerBlock / kThreadsPerRow;
  constexpr int kIters = kTileDim / kRowsPerIter;

  const int tid = threadIdx.x;
  const int thr_col = tid % kThreadsPerRow;
  const int thr_row = tid / kThreadsPerRow;
  const size_t c = global_col_base + static_cast<size_t>(thr_col) * kVec;

  // 1D columnwise has one scale per column. Pre-load this thread's 16 columns.
  CType s_inv[kVec];
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    s_inv[e] = (c + e < K) ? tile_scale_base[c + e] : static_cast<CType>(0.f);
  }

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int row_local = thr_row + it * kRowsPerIter;
    const size_t r_global = global_row_base + row_local;
    if (r_global >= R_total) continue;

    const size_t r_local = r_global - info.tensor_row_base;

    // Per-expert (K, M_t) transposed input: strided by M_t per column (no vector load).
    // K % 128 == 0 so c+e < K always holds; the explicit else just keeps correctness
    // independent of that invariant.
    Vec<IType, kVec> in_vec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      in_vec.data.elt[e] = (c + e < K)
                               ? input_base[expert_data_off + (c + e) * info.tensor_M + r_local]
                               : static_cast<IType>(0);
    }

    Vec<OType, kVec> out_vec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      out_vec.data.elt[e] = static_cast<OType>(static_cast<float>(in_vec.data.elt[e]) * s_inv[e]);
    }

    if (c + kVec <= K) {
      out_vec.store_to(output_base + r_global * K + c);
    } else if (c < K) {
      out_vec.store_to_elts(output_base + r_global * K + c, 0, K - c);
    }
  }
#endif
}

// ===== 2D rowwise =====
// Data layout: (M, K) flat. One scale per 128x128 tile.
// Scale layout GLOBAL: (total_row_blocks, roundup(blocks_X, 4)) floats.
//   scale[tile_y_global * scale_stride_y + tile_x]
template <bool kSameBothDims, typename IType, typename OType, typename CType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_dequantize_blockwise_2d_rw_kernel(const IType* __restrict__ input_base,
                                            OType* __restrict__ output_base,
                                            const CType* __restrict__ scale_inv_base,
                                            const size_t scale_stride_y,
                                            const int64_t* __restrict__ tensor_offsets_ptr,
                                            const size_t num_tensors,
                                            const size_t common_first_dim_blocks, const size_t K,
                                            const size_t total_row_blocks, const size_t R_total) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  const auto info = resolve_tile_expert<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, K, total_row_blocks, tensor_offsets_ptr);
  if (!info.in_bounds) return;

  // 2D: one scale per tile.
  const CType s_inv = scale_inv_base[tile_y_global * scale_stride_y + tile_x];

  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  constexpr int kThreadsPerRow = 8;
  constexpr int kVec = 16;
  constexpr int kRowsPerIter = kThreadsPerBlock / kThreadsPerRow;
  constexpr int kIters = kTileDim / kRowsPerIter;

  const int tid = threadIdx.x;
  const int thr_col = tid % kThreadsPerRow;
  const int thr_row = tid / kThreadsPerRow;
  const size_t c = global_col_base + static_cast<size_t>(thr_col) * kVec;

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int row_local = thr_row + it * kRowsPerIter;
    const size_t r_global = global_row_base + row_local;
    if (r_global >= R_total) continue;

    Vec<IType, kVec> in_vec;
    if (c + kVec <= K) {
      in_vec.load_from(input_base + r_global * K + c);
    } else if (c < K) {
      in_vec.load_from_elts(input_base + r_global * K + c, 0, K - c);
    } else {
      continue;
    }

    Vec<OType, kVec> out_vec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      out_vec.data.elt[e] = static_cast<OType>(static_cast<float>(in_vec.data.elt[e]) * s_inv);
    }

    if (c + kVec <= K) {
      out_vec.store_to(output_base + r_global * K + c);
    } else if (c < K) {
      out_vec.store_to_elts(output_base + r_global * K + c, 0, K - c);
    }
  }
#endif
}

// ===== 2D columnwise =====
// Data layout per-expert: (K, M_t) transposed.
// Scale layout per-expert: (blocks_X, roundup(blocks_y_t, 4)) floats.
//   scale[expert_off + tile_x * roundup(blocks_y_t, 4) + local_tile_y]
template <bool kSameBothDims, typename IType, typename OType, typename CType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_dequantize_blockwise_2d_cw_kernel(const IType* __restrict__ input_base,
                                            OType* __restrict__ output_base,
                                            const CType* __restrict__ scale_inv_base,
                                            const int64_t* __restrict__ tensor_offsets_ptr,
                                            const size_t num_tensors,
                                            const size_t common_first_dim_blocks, const size_t K,
                                            const size_t total_row_blocks, const size_t R_total) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  const auto info = resolve_tile_expert<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, K, total_row_blocks, tensor_offsets_ptr);
  if (!info.in_bounds) return;

  // `info.in_bounds` is derived from blockIdx.y and is uniform across the CTA,
  // so the early return above never strands a sibling thread inside
  // compute_2d_cw_expert_offset's __syncthreads().
  __shared__ size_t warp_offset_partials[kNumWarps];

  const int tid = threadIdx.x;
  const int warp_id = tid / kThreadsPerWarp;
  const int lane = tid % kThreadsPerWarp;

  const size_t blocks_X = DIVUP(K, static_cast<size_t>(kTileDim));
  const size_t tile_row_stride = static_cast<size_t>(kTileDim) * K;
  const size_t expert_offset = compute_2d_cw_expert_offset<kSameBothDims>(
      info.tensor_id, blocks_X, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr,
      warp_offset_partials, tid, warp_id, lane);
  const size_t per_expert_stride_t = DIVUP_TO_MULTIPLE(info.tensor_row_blocks, kScaleColAlign);
  const size_t local_tile_y = tile_y_global - info.tensor_block_y_base;
  const CType s_inv = scale_inv_base[expert_offset + tile_x * per_expert_stride_t + local_tile_y];

  const size_t expert_data_off = info.tensor_row_base * K;
  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  constexpr int kThreadsPerRow = 8;
  constexpr int kVec = 16;
  constexpr int kRowsPerIter = kThreadsPerBlock / kThreadsPerRow;
  constexpr int kIters = kTileDim / kRowsPerIter;

  const int thr_col = tid % kThreadsPerRow;
  const int thr_row = tid / kThreadsPerRow;
  const size_t c = global_col_base + static_cast<size_t>(thr_col) * kVec;

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int row_local = thr_row + it * kRowsPerIter;
    const size_t r_global = global_row_base + row_local;
    if (r_global >= R_total) continue;

    const size_t r_local = r_global - info.tensor_row_base;

    // Explicit else zeroes out-of-range lanes (c+e < K always holds since K % 128 == 0;
    // this keeps correctness independent of that invariant).
    Vec<IType, kVec> in_vec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      in_vec.data.elt[e] = (c + e < K)
                               ? input_base[expert_data_off + (c + e) * info.tensor_M + r_local]
                               : static_cast<IType>(0);
    }

    Vec<OType, kVec> out_vec;
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      out_vec.data.elt[e] = static_cast<OType>(static_cast<float>(in_vec.data.elt[e]) * s_inv);
    }

    if (c + kVec <= K) {
      out_vec.store_to(output_base + r_global * K + c);
    } else if (c < K) {
      out_vec.store_to_elts(output_base + r_global * K + c, 0, K - c);
    }
  }
#endif
}

}  // namespace group_dequantize_kernel

// Host-side dispatcher. Supports all four combinations of {1D, 2D} block
// scaling x {rowwise, columnwise} data, matching the layouts written by
// ``group_quantize_fp8_blockwise``. The input GroupedTensor must have exactly
// one of rowwise / columnwise data populated (the dequantize API rejects
// both).
inline void group_dequantize(const GroupedTensor* input, GroupedTensor* output,
                             cudaStream_t stream) {
  using namespace group_dequantize_kernel;

  const int sm = transformer_engine::cuda::sm_arch();
  NVTE_CHECK(sm >= 90 && sm < 100,
             "Grouped FP8 block-scaling dequantize is only supported on Hopper (SM90-SM99); "
             "got SM",
             sm, ".");
  NVTE_CHECK(
      input->scaling_mode == NVTE_BLOCK_SCALING_1D || input->scaling_mode == NVTE_BLOCK_SCALING_2D,
      "Grouped FP8 block-scaling dequantize requires 1D or 2D block scaling "
      "(got scaling_mode=",
      to_string(input->scaling_mode), ").");
  NVTE_CHECK(is_fp8_dtype(input->dtype()), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->dtype()), "Output must be in higher precision.");
  NVTE_CHECK(!is_fp4_dtype(output->dtype()), "Output must not be FP4.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must match.");

  const bool use_rowwise = input->has_data();
  const bool use_colwise = input->has_columnwise_data();
  NVTE_CHECK(use_rowwise || use_colwise, "Input must have rowwise or columnwise data populated.");
  NVTE_CHECK(!(use_rowwise && use_colwise),
             "Grouped FP8 block-scaling dequantize accepts exactly one direction at a "
             "time (not both rowwise and columnwise simultaneously).");
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Grouped FP8 block-scaling dequantize requires compact (un-swizzled) scales.");

  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  if (first_logical_dim == 0 || last_logical_dim == 0) return;

  const bool same_both_dims = input->all_same_shape();
  const bool varying_first_dim = (!input->all_same_first_dim()) && input->all_same_last_dim();
  NVTE_CHECK(same_both_dims || varying_first_dim,
             "Grouped FP8 block-scaling dequantize supports only SAME_BOTH_DIMS and "
             "VARYING_FIRST_DIM shape representations.");

  const size_t num_tensors = input->num_tensors;
  const size_t K = last_logical_dim;
  NVTE_CHECK(K % kTileDim == 0,
             "Last dim must be a multiple of 128 for FP8 block-scaling dequantize (got ", K, ").");

  size_t common_first_dim_blocks = 0;
  if (same_both_dims) {
    const size_t common_first_dim = input->get_common_first_dim();
    NVTE_CHECK(common_first_dim % kTileDim == 0,
               "SAME_BOTH_DIMS first dim must be multiple of 128 (got ", common_first_dim, ").");
    common_first_dim_blocks = common_first_dim / kTileDim;
  }
  const size_t total_row_blocks = DIVUP(first_logical_dim, static_cast<size_t>(kTileDim));
  const size_t blocks_X = K / kTileDim;

  const int64_t* tensor_offsets_ptr =
      same_both_dims ? nullptr : reinterpret_cast<const int64_t*>(input->tensor_offsets.dptr);
  if (!same_both_dims) {
    NVTE_CHECK(tensor_offsets_ptr != nullptr,
               "VARYING_FIRST_DIM requires tensor_offsets to be set on the input.");
  }

  const dim3 grid(blocks_X, total_row_blocks);
  const dim3 block(kThreadsPerBlock);
  const bool is_1d = (input->scaling_mode == NVTE_BLOCK_SCALING_1D);

  // Pick the populated direction's data + scale buffer. The global-layout strides (1D cw, 2D rw)
  // are derived from K / blocks_X exactly as the quantize launcher does.
  const SimpleTensor& input_data = use_rowwise ? input->data : input->columnwise_data;
  const SimpleTensor& input_scale_inv =
      use_rowwise ? input->scale_inv : input->columnwise_scale_inv;
  const size_t scale_t_stride_aligned_K = DIVUP_TO_MULTIPLE(K, kScaleColAlign);
  const size_t scale_stride_y = DIVUP_TO_MULTIPLE(blocks_X, kScaleColAlign);

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->dtype(), OType, using CType = float;
          const IType* const input_dptr = reinterpret_cast<const IType*>(input_data.dptr);
          OType* const output_dptr = reinterpret_cast<OType*>(output->data.dptr);
          const CType* const scale_inv_dptr = reinterpret_cast<const CType*>(input_scale_inv.dptr);

          if (is_1d && use_rowwise) {
            if (same_both_dims) {
              group_dequantize_blockwise_1d_rw_kernel<true, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(
                      input_dptr, output_dptr, scale_inv_dptr, tensor_offsets_ptr, num_tensors,
                      common_first_dim_blocks, K, total_row_blocks, first_logical_dim);
            } else {
              group_dequantize_blockwise_1d_rw_kernel<false, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(
                      input_dptr, output_dptr, scale_inv_dptr, tensor_offsets_ptr, num_tensors,
                      common_first_dim_blocks, K, total_row_blocks, first_logical_dim);
            }
          } else if (is_1d && use_colwise) {
            if (same_both_dims) {
              group_dequantize_blockwise_1d_cw_kernel<true, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(input_dptr, output_dptr, scale_inv_dptr,
                                               scale_t_stride_aligned_K, tensor_offsets_ptr,
                                               num_tensors, common_first_dim_blocks, K,
                                               total_row_blocks, first_logical_dim);
            } else {
              group_dequantize_blockwise_1d_cw_kernel<false, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(input_dptr, output_dptr, scale_inv_dptr,
                                               scale_t_stride_aligned_K, tensor_offsets_ptr,
                                               num_tensors, common_first_dim_blocks, K,
                                               total_row_blocks, first_logical_dim);
            }
          } else if (!is_1d && use_rowwise) {
            if (same_both_dims) {
              group_dequantize_blockwise_2d_rw_kernel<true, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(
                      input_dptr, output_dptr, scale_inv_dptr, scale_stride_y, tensor_offsets_ptr,
                      num_tensors, common_first_dim_blocks, K, total_row_blocks, first_logical_dim);
            } else {
              group_dequantize_blockwise_2d_rw_kernel<false, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(
                      input_dptr, output_dptr, scale_inv_dptr, scale_stride_y, tensor_offsets_ptr,
                      num_tensors, common_first_dim_blocks, K, total_row_blocks, first_logical_dim);
            }
          } else {  // 2D columnwise
            if (same_both_dims) {
              group_dequantize_blockwise_2d_cw_kernel<true, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(
                      input_dptr, output_dptr, scale_inv_dptr, tensor_offsets_ptr, num_tensors,
                      common_first_dim_blocks, K, total_row_blocks, first_logical_dim);
            } else {
              group_dequantize_blockwise_2d_cw_kernel<false, IType, OType, CType>
                  <<<grid, block, 0, stream>>>(
                      input_dptr, output_dptr, scale_inv_dptr, tensor_offsets_ptr, num_tensors,
                      common_first_dim_blocks, K, total_row_blocks, first_logical_dim);
            }
          });  // NOLINT(*)
  );           // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8_blockwise
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_FP8_BLOCKWISE_CUH_
