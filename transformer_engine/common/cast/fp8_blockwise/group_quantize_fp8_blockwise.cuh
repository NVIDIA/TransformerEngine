/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8_blockwise.cuh
 *  \brief CUDA kernels to quantize grouped tensors with FP8 1D and 2D
 *  block scaling. A single launch walks 128x128 tiles across every tensor
 *  in the group, with each CTA decoding its owning tensor from the device-side
 *  GroupedTensor metadata. Supports SAME_BOTH_DIMS and VARYING_FIRST_DIM.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <vector>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../transpose/cast_transpose.h"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8_blockwise {

using transformer_engine::detail::FP8BlockwiseColumnwiseOption;
using transformer_engine::detail::FP8BlockwiseRowwiseOption;

constexpr int kTileDim = 128;
constexpr int kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock = 256;
constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;

// ---- Per-expert scale layout helpers --------------------------------------------
//
// cuBLAS grouped FP8 block-scaling GEMM expects each expert's scales to live
// in a contiguous per-expert sub-block of the global scale buffer:
//
//   1D rowwise CW (op_rowwise=true)   per expert: (blocks_X, roundup(M_t, 4))    floats
//   1D columnwise (op_rowwise=false)  per expert: (blocks_y_t, roundup(K, 4))    floats
//   2D rowwise (op_rowwise=true)      per expert: (blocks_y_t, roundup(blocks_X, 4))
//   2D columnwise (op_rowwise=false)  per expert: (blocks_X, roundup(blocks_y_t, 4))
//
// The grouped kernel writes the GLOBAL buffer in tile-stride order, but the
// position assigned to each tile must map into the per-expert contiguous
// sub-block. These helpers compute the per-expert cumulative byte/float
// offset and the per-expert local stride. SAME_BOTH_DIMS lets us derive
// without walking offsets; VARYING_FIRST_DIM walks `tensor_offsets_ptr` once
// per block (only the writing thread).

__device__ __host__ constexpr size_t kScaleColAlign = 4;

// 2D columnwise per-expert scale offset. Per-expert layout is
// (blocks_X, roundup(blocks_y_t, 4)). Two paths, picked at call sites:
//   - SAME_BOTH_DIMS: direct formula (no walk, no reduction).
//   - VARYING_FIRST_DIM: CTA-cooperative prefix sum. Each thread accumulates a
//     partial over a strided subset of the tensors-before-this-one, a
//     warp-shuffle reduces inside each warp, then all threads sum the per-warp
//     partials read from a kNumWarps-element smem buffer to obtain the same
//     total. The non-linear DIVUP_TO_MULTIPLE on each per-tensor blocks_y_t
//     prevents a closed form.
// ALL threads of the CTA must call this in lock-step (the cooperative path
// contains __syncthreads()).
template <bool kSameBothDims>
__device__ __forceinline__ size_t compute_2d_cw_expert_offset(
    const size_t tensor_id, const size_t blocks_X, const size_t common_first_dim_blocks,
    const size_t tile_row_stride, const int64_t* __restrict__ tensor_offsets_ptr,
    size_t* warp_partials_smem, const int tid, const int warp_id, const int lane) {
  if constexpr (kSameBothDims) {
    return tensor_id * blocks_X * DIVUP_TO_MULTIPLE(common_first_dim_blocks, kScaleColAlign);
  }
  size_t my_partial = 0;
  for (size_t i = static_cast<size_t>(tid); i < tensor_id;
       i += static_cast<size_t>(kThreadsPerBlock)) {
    const size_t blocks_y_i =
        static_cast<size_t>(tensor_offsets_ptr[i + 1] - tensor_offsets_ptr[i]) / tile_row_stride;
    my_partial += DIVUP_TO_MULTIPLE(blocks_y_i, kScaleColAlign);
  }
  my_partial = warp_allreduce_sum(my_partial);
  if (lane == 0) warp_partials_smem[warp_id] = my_partial;
  __syncthreads();
  size_t total = 0;
#pragma unroll
  for (int w = 0; w < kNumWarps; ++w) {
    total += warp_partials_smem[w];
  }
  return total * blocks_X;
}

// 1D rowwise: per-expert layout (blocks_X, roundup(M_t, 4)).
template <bool kSameBothDims>
__device__ __forceinline__ size_t expert_scale_offset_1d_rowwise(
    size_t tensor_id, size_t blocks_X, size_t common_first_dim_blocks, size_t tile_row_stride,
    const int64_t* __restrict__ tensor_offsets_ptr) {
  if constexpr (kSameBothDims) {
    const size_t M = common_first_dim_blocks * kTileDim;
    return tensor_id * blocks_X * DIVUP_TO_MULTIPLE(M, kScaleColAlign);
  } else {
    // Each M_i is enforced to be a multiple of kTileDim (=128), hence a
    // multiple of kScaleColAlign (=4), so DIVUP_TO_MULTIPLE(M_i, 4) == M_i and
    // sum_{i<tensor_id} M_i = tensor_offsets_ptr[tensor_id] / K. Computing the
    // offset in O(1) avoids a per-CTA O(num_tensors) walk that otherwise
    // dominates wall-clock for jagged routings at high expert counts.
    const size_t K = tile_row_stride / kTileDim;
    const size_t total_M_before = static_cast<size_t>(tensor_offsets_ptr[tensor_id]) / K;
    return blocks_X * total_M_before;
  }
}

// ---- Tensor-lookup helpers ----------------------------------------------------

// Map a global tile-row index to its owning tensor. Delegates to the shared
// `common::get_current_tensor_id` helper from `cast/core/common.cuh`. The
// helper is parameterized by total `first_logical_dim` rather than per-tensor
// block count, so we reconstruct it here for the SAME_BOTH_DIMS specialization
// (VARYING_FIRST_DIM ignores it and uses `current_offset` + `offsets_ptr`).
template <bool kSameBothDims>
__device__ __forceinline__ size_t find_tensor_id_by_block_y(
    const size_t block_y_global, const size_t num_tensors, const size_t common_first_dim_blocks,
    const size_t tile_row_stride, const int64_t* __restrict__ tensor_offsets_ptr) {
  constexpr auto shape_rep =
      kSameBothDims ? ShapeRepresentation::SAME_BOTH_DIMS : ShapeRepresentation::VARYING_FIRST_DIM;
  const size_t first_logical_dim = num_tensors * common_first_dim_blocks * kTileDim;
  const size_t tensor_id = common::get_current_tensor_id<shape_rep, kTileDim>(
      num_tensors, block_y_global * tile_row_stride, block_y_global, first_logical_dim,
      /*last_logical_dim=*/0, tensor_offsets_ptr);
  if constexpr (!kSameBothDims) {
    // tensor_offsets_ptr carries cumulative element counts; tile_row_stride =
    // kTileDim * K, so the per-tensor element span is divisible by
    // tile_row_stride iff first_dim is a multiple of kTileDim.
    if (tensor_id < num_tensors) {
      const size_t span =
          static_cast<size_t>(tensor_offsets_ptr[tensor_id + 1] - tensor_offsets_ptr[tensor_id]);
      if (span % tile_row_stride != 0) {
        NVTE_DEVICE_ERROR(
            "Grouped FP8 block-scaling quantize: each tensor's first dimension must be a "
            "multiple of 128 (VARYING_FIRST_DIM).");
      }
    }
  }
  return tensor_id;
}

// Per-tensor block-y base for VARYING_FIRST_DIM (in 128-row block units).
__device__ __forceinline__ size_t tensor_block_y_base_from_offsets(
    const size_t tensor_id, const int64_t* __restrict__ tensor_offsets_ptr,
    const size_t tile_row_stride) {
  return static_cast<size_t>(tensor_offsets_ptr[tensor_id]) / tile_row_stride;
}

// Per-vector amax. Uses bf16x2 `max.xorsign.abs` on sm_89+; FP32 fallback otherwise.
template <typename IType, typename CType, int kVec>
__device__ __forceinline__ CType compute_row_amax(const Vec<IType, kVec>& v) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  if constexpr (std::is_same_v<IType, bf16>) {
    static_assert(kVec % 2 == 0, "kVec must be even for packed bf16x2 amax");
    const ptx::bf16x2* pairs = reinterpret_cast<const ptx::bf16x2*>(&v.data.elt[0]);
    ptx::bf16x2 amax_x2{static_cast<bf16>(0.f), static_cast<bf16>(0.f)};
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p) {
      ptx::abs_max_2x(amax_x2, amax_x2, pairs[p]);
    }
    return static_cast<CType>(__hmax(__habs(amax_x2.x), __habs(amax_x2.y)));
  }
#endif
  CType amax = 0.f;
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    amax = fmaxf(amax, fabsf(static_cast<CType>(v.data.elt[e])));
  }
  return amax;
}

// Per-tile column sum of the high-precision input -> one fp32 row at
// dbias_workspace[tile_y_global * K + col]. 2 threads/column sum 64 rows each (combined via
// shfl_xor); grouped_reduce_dbias later sums each expert's row-blocks. Tiles are always full
// (experts are 128-row aligned), so all 128 rows are summed.
template <typename IType>
__device__ __forceinline__ void write_tile_dbias_partial(const IType smem_tile[][kTileDim],
                                                         const int tid,
                                                         const size_t global_col_base,
                                                         const size_t K, const size_t tile_y_global,
                                                         float* __restrict__ dbias_workspace) {
  constexpr int kThreadsPerColDB = 2;
  constexpr int kRowsPerThreadDB = kTileDim / kThreadsPerColDB;  // 64
  const int col_local = tid / kThreadsPerColDB;                  // 0..127
  const int sub = tid % kThreadsPerColDB;
  const int row_start = sub * kRowsPerThreadDB;
  float partial = 0.f;
#pragma unroll
  for (int e = 0; e < kRowsPerThreadDB; ++e) {
    partial += static_cast<float>(smem_tile[row_start + e][col_local]);
  }
  partial += __shfl_xor_sync(0xffffffff, partial, 1);
  const size_t c_global = global_col_base + col_local;
  if (sub == 0 && c_global < K) {
    dbias_workspace[tile_y_global * K + c_global] = partial;
  }
}

// Per-vector multiply-and-quantize via fp32 intermediates.
template <typename IType, typename OType, typename CType, int kVec>
__device__ __forceinline__ void quantize_row_vec(Vec<OType, kVec>& out, const Vec<IType, kVec>& in,
                                                 CType scale) {
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    out.data.elt[e] = static_cast<OType>(static_cast<CType>(in.data.elt[e]) * scale);
  }
}

// Bank-conflict swizzle delta for the 2D smem_T staging buffer. delta carries
// bits 2..5 only so each 4-byte sub-chunk is preserved.
__device__ __forceinline__ int smem_t_swz_delta(int smem_t_row) {
  return ((smem_t_row >> 3) & 0xf) << 2;
}

// Drain smem_T to gmem for the CW path: 4 cols/warp, 8 lanes/col, each lane
// stores a 16-row chunk so the 8 lanes of a col emit one 128 B gmem line.
//
// Columnwise data is stored per-expert contiguously as a (K, M_t) transposed
// block at element offset K * tensor_row_base, matching cuBLAS grouped GEMM's
// per-expert data pointer (compute_grouped_tensor_offset = sum_i M_i * K).
// `tensor_row_base` is expert t's first global row; `tensor_M` is M_t.
//
// When `kSwizzledStaging` is true, the writer applied smem_t_swz_delta to the
// inner index, so the read must XOR back to recover the logical (row, col).
// Per-byte unswizzled reads are required because the smem_T row stride is not
// 16 B aligned for arbitrary column offsets.
template <bool kSwizzledStaging, typename OType, int kSMemTRowStride>
__device__ __forceinline__ void drain_smem_t_to_gmem(
    OType (&smem_T)[kTileDim][kSMemTRowStride], OType* __restrict__ output_t_base,
    const size_t global_col_base, const size_t global_row_base, const size_t tensor_row_base,
    const size_t tensor_M, const size_t K, const int tid) {
  constexpr int kStorePerChunk = 16;
  constexpr int kRowChunksPerCol = kTileDim / kStorePerChunk;        // 8
  constexpr int kColsPerIter = kThreadsPerBlock / kRowChunksPerCol;  // 32
  constexpr int kColIters = kTileDim / kColsPerIter;                 // 4
  const int warp_id = tid / kThreadsPerWarp;
  const int lane = tid % kThreadsPerWarp;
  const int col_in_warp = lane / kRowChunksPerCol;
  const int row_chunk = lane % kRowChunksPerCol;
  const int out_row_off = row_chunk * kStorePerChunk;
  const size_t expert_data_off = tensor_row_base * K;
  const size_t m_local_base = global_row_base - tensor_row_base;
#pragma unroll
  for (int it = 0; it < kColIters; ++it) {
    const int out_col_local = it * kColsPerIter + warp_id * 4 + col_in_warp;
    const size_t out_col_global = global_col_base + out_col_local;
    if (out_col_global < K) {
      // Per-expert (K, M_t): index = expert_off + k * M_t + m_local.
      OType* out_ptr =
          output_t_base + expert_data_off + out_col_global * tensor_M + m_local_base + out_row_off;
      const int swz_delta_r = kSwizzledStaging ? smem_t_swz_delta(out_col_local) : 0;
      Vec<OType, kStorePerChunk> v;
#pragma unroll
      for (int e = 0; e < kStorePerChunk; ++e) {
        v.data.elt[e] = smem_T[out_col_local][(out_row_off + e) ^ swz_delta_r];
      }
      v.store_to(out_ptr);
    }
  }
}

// ----- 2D block scaling kernel with TMA input ------------------------------------------------
// Pass 1: amax over a 128x128 TMA-loaded tile, with input vectors staged in
// registers. Pass 2: quantize from registers, emit rowwise output and the
// transposed smem_T tile, then drain smem_T to gmem.

template <bool kSameBothDims, bool kReturnRowwise, bool kReturnColwise, typename CType,
          typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock, 4) group_block_scaled_2d_tma_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input, OType* __restrict__ output_base,
    OType* __restrict__ output_t_base, CType* __restrict__ scale_inv_base,
    CType* __restrict__ scale_inv_t_base, const int64_t* __restrict__ tensor_offsets_ptr,
    const size_t num_tensors, const size_t common_first_dim_blocks, const size_t K,
    const size_t total_row_blocks, const size_t blocks_X, const size_t scale_stride_y,
    const float epsilon, const bool pow_2_scales, const float* __restrict__ noop_ptr,
    float* __restrict__ dbias_workspace) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) return;

  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  if (tile_y_global >= total_row_blocks) return;

  const size_t tile_row_stride = static_cast<size_t>(kTileDim) * K;
  const size_t tensor_id = find_tensor_id_by_block_y<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
  const size_t tensor_block_y_base =
      kSameBothDims
          ? (tensor_id * common_first_dim_blocks)
          : tensor_block_y_base_from_offsets(tensor_id, tensor_offsets_ptr, tile_row_stride);
  const size_t tensor_row_blocks =
      kSameBothDims
          ? common_first_dim_blocks
          : (tensor_block_y_base_from_offsets(tensor_id + 1, tensor_offsets_ptr, tile_row_stride) -
             tensor_block_y_base);
  if (tile_y_global >= tensor_block_y_base + tensor_row_blocks) return;

  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  // Dynamic smem holds the IType input tile (TMA dest, must be 128 B aligned).
  // warp_amaxes and tma_mbar are static smem.
  extern __shared__ unsigned char smem_raw_2d_tma[];
  IType(*smem_in)[kTileDim] = reinterpret_cast<IType(*)[kTileDim]>(
      common::align_smem_ptr_per_TMA_requirements(smem_raw_2d_tma));

  __shared__ CType warp_amaxes[kNumWarps];
  __shared__ size_t warp_offset_partials[kNumWarps];
  __shared__ uint64_t tma_mbar;

  const int tid = threadIdx.x;
  const bool leading_thread = (tid == 0);

  // ---- TMA async load of the input tile ----
  if (leading_thread) {
    ptx::mbarrier_init(&tma_mbar, 1);
    // Fence so the TMA engine (async proxy) and the threads (generic proxy)
    // both observe the just-initialized mbarrier consistently.
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (leading_thread) {
    constexpr uint32_t tx_bytes = kTileDim * kTileDim * sizeof(IType);
    ptx::mbarrier_arrive_expect_tx(&tma_mbar, tx_bytes);
    ptx::cp_async_bulk_tensor_2d_global_to_shared_cta(
        reinterpret_cast<uint64_t*>(smem_in), reinterpret_cast<const uint64_t*>(&tensor_map_input),
        static_cast<uint32_t>(global_col_base), static_cast<uint32_t>(global_row_base), &tma_mbar);
  }
  ptx::mbarrier_wait_parity(&tma_mbar, 0);
  if (leading_thread) ptx::mbarrier_invalid(&tma_mbar);
  __syncthreads();

  // ---- Optional dbias: per-tile column sum of the high-precision input (smem-resident) ----
  if (dbias_workspace != nullptr) {
    write_tile_dbias_partial<IType>(smem_in, tid, global_col_base, K, tile_y_global,
                                    dbias_workspace);
  }

  // ---- Pass 1: tile amax, staging input vectors in registers for reuse in pass 2 ----
  constexpr int kEltsPerThread = 8;
  constexpr int kThreadsPerRow = kTileDim / kEltsPerThread;        // 16
  constexpr int kRowsPerIter = kThreadsPerBlock / kThreadsPerRow;  // 16
  constexpr int kIters = kTileDim / kRowsPerIter;                  // 8

  const int thr_col = tid % kThreadsPerRow;
  const int thr_row = tid / kThreadsPerRow;

  using IVec = Vec<IType, kEltsPerThread>;
  using OVec = Vec<OType, kEltsPerThread>;

  IVec staged[kIters];
  CType thr_amax = 0.f;
#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int r_local = thr_row + it * kRowsPerIter;
    staged[it].load_from(&smem_in[r_local][thr_col * kEltsPerThread]);
    thr_amax = fmaxf(thr_amax, compute_row_amax<IType, CType, kEltsPerThread>(staged[it]));
  }
  CType warp_amax = warp_reduce_max<kThreadsPerWarp>(thr_amax);
  const int warp_id = tid / kThreadsPerWarp;
  const int lane = tid % kThreadsPerWarp;
  if (lane == 0) warp_amaxes[warp_id] = warp_amax;
  __syncthreads();

  CType block_amax = warp_amaxes[0];
#pragma unroll
  for (int w = 1; w < kNumWarps; ++w) {
    block_amax = fmaxf(block_amax, warp_amaxes[w]);
  }
  const CType scale =
      compute_scale_from_types<IType, OType>(block_amax, epsilon, pow_2_scales);

  // The 2D colwise per-expert scale offset requires a CTA-cooperative prefix
  // sum in the VARYING_FIRST_DIM case, so compute it across all threads before
  // the leading-thread-only store. All threads end up with the same value;
  // only thread 0 reads it below.
  size_t expert_offset_t = 0;
  if constexpr (kReturnColwise) {
    expert_offset_t = compute_2d_cw_expert_offset<kSameBothDims>(
        tensor_id, blocks_X, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr,
        warp_offset_partials, tid, warp_id, lane);
  }

  if (leading_thread) {
    const CType scale_inv = 1.f / scale;
    if constexpr (kReturnRowwise) {
      // 2D rowwise: kernel-global stride matches per-expert layout naturally.
      // Expert t's rows occupy [tensor_block_y_base, +blocks_y_t) of the buffer,
      // each row sized roundup(blocks_X, 4), which equals the dispatcher's
      // cumulative per-expert offset.
      scale_inv_base[tile_y_global * scale_stride_y + tile_x] = scale_inv;
    }
    if constexpr (kReturnColwise) {
      // 2D colwise: rewrite into per-expert sub-block matching cuBLAS grouped
      // GEMM's per-expert layout (blocks_X, roundup(blocks_y_t, 4)).
      const size_t local_tile_y = tile_y_global - tensor_block_y_base;
      const size_t per_expert_stride_t = DIVUP_TO_MULTIPLE(tensor_row_blocks, kScaleColAlign);
      scale_inv_t_base[expert_offset_t + tile_x * per_expert_stride_t + local_tile_y] = scale_inv;
    }
  }

  // ---- Pass 2: quantize from register-staged inputs, emit rowwise + colwise outputs ----
  // 2D block-scaling uses a per-tile (128x128) scalar scale, so the row-wise and
  // column-wise quantized bytes are identical -- only the gmem layout differs. The
  // columnwise buffer is physically transposed (cuBLAS FP8 block-scaling GEMM is
  // TN-only), so we stage into smem_T and drain to the per-expert (K, M_t) transposed block.
  if constexpr (kReturnColwise) {
    constexpr int kSMemTRowStride = kTileDim + 4;
    __shared__ OType smem_T[kTileDim][kSMemTRowStride];
    // Same delta for all 8 elements this thread writes since (thr_col*8 + e) >> 3 == thr_col.
    const int swz_delta_w = smem_t_swz_delta(thr_col * kEltsPerThread);
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int row_local = thr_row + it * kRowsPerIter;
      const size_t r = global_row_base + row_local;
      const size_t c = global_col_base + thr_col * kEltsPerThread;
      OVec qo;
      quantize_row_vec<IType, OType, CType, kEltsPerThread>(qo, staged[it], scale);
      if constexpr (kReturnRowwise) {
        if (c < K) {
          const size_t count = (c + kEltsPerThread <= K) ? kEltsPerThread : (K - c);
          qo.store_to_elts(output_base + r * K + c, 0, count);
        }
      }
      const int c_phys = row_local ^ swz_delta_w;
#pragma unroll
      for (int e = 0; e < kEltsPerThread; ++e) {
        smem_T[thr_col * kEltsPerThread + e][c_phys] = qo.data.elt[e];
      }
    }
    __syncthreads();

    const size_t tensor_row_base = tensor_block_y_base * kTileDim;
    const size_t tensor_M = tensor_row_blocks * kTileDim;
    drain_smem_t_to_gmem<true, OType, kSMemTRowStride>(
        smem_T, output_t_base, global_col_base, global_row_base, tensor_row_base, tensor_M, K, tid);
  } else if constexpr (kReturnRowwise) {
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      const int row_local = thr_row + it * kRowsPerIter;
      const size_t r = global_row_base + row_local;
      const size_t c = global_col_base + thr_col * kEltsPerThread;
      OVec qo;
      quantize_row_vec<IType, OType, CType, kEltsPerThread>(qo, staged[it], scale);
      if (c < K) {
        const size_t count = (c + kEltsPerThread <= K) ? kEltsPerThread : (K - c);
        qo.store_to_elts(output_base + r * K + c, 0, count);
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
}

// ----- 1D block scaling rowwise-only kernel ----------------------------------------------------
// No smem cache. Each thread loads 16 cols/row, reduces amax across the 8
// row-mates with shfl_xor, then quantizes and stores.

template <bool kSameBothDims, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    group_block_scaled_1d_rw_kernel(const IType* __restrict__ input_base,
                                    OType* __restrict__ output_base,
                                    CType* __restrict__ scale_inv_base,
                                    const int64_t* __restrict__ tensor_offsets_ptr,
                                    const size_t num_tensors, const size_t common_first_dim_blocks,
                                    const size_t K, const size_t total_row_blocks,
                                    const size_t R_total, const float epsilon,
                                    const bool pow_2_scales,
                                    const float* __restrict__ noop_ptr) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) return;

  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  if (tile_y_global >= total_row_blocks) return;

  const size_t tile_row_stride = static_cast<size_t>(kTileDim) * K;
  const size_t tensor_id = find_tensor_id_by_block_y<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
  const size_t tensor_block_y_base =
      kSameBothDims
          ? (tensor_id * common_first_dim_blocks)
          : tensor_block_y_base_from_offsets(tensor_id, tensor_offsets_ptr, tile_row_stride);
  const size_t tensor_row_blocks =
      kSameBothDims
          ? common_first_dim_blocks
          : (tensor_block_y_base_from_offsets(tensor_id + 1, tensor_offsets_ptr, tile_row_stride) -
             tensor_block_y_base);
  if (tile_y_global >= tensor_block_y_base + tensor_row_blocks) return;

  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  // 8 threads per row x 16 cols/thread = one 128 B gmem cache line per row, 4 iters per tile.
  constexpr int kThreadsPerRow = 8;
  constexpr int kVec = 16;
  constexpr int kRowsPerIter = kThreadsPerBlock / kThreadsPerRow;  // 32
  constexpr int kIters = kTileDim / kRowsPerIter;                  // 4

  const int tid = threadIdx.x;
  const int thr_col = tid % kThreadsPerRow;  // 0..7
  const int thr_row = tid / kThreadsPerRow;  // 0..31 (row index within an iter)
  const size_t c = global_col_base + static_cast<size_t>(thr_col) * kVec;

  Vec<IType, kVec> in_vec[kIters];

#pragma unroll
  for (int it = 0; it < kIters; ++it) {
    const int row_local = thr_row + it * kRowsPerIter;
    const size_t r_global = global_row_base + row_local;

    // Load this thread's 16 cols of row `row_local`.
    if (c + kVec <= K) {
      in_vec[it].load_from(input_base + r_global * K + c);
    } else if (c < K) {
      in_vec[it].load_from_elts(input_base + r_global * K + c, 0, K - c);
    } else {
      in_vec[it].clear();
    }

    CType amax = compute_row_amax<IType, CType, kVec>(in_vec[it]);
    amax = subwarp_reduce_max_broadcast<kThreadsPerRow>(amax);

    const CType scale =
        compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scales);
    const CType scale_inv = 1.f / scale;
    if (thr_col == 0 && r_global < R_total) {
      // Per-expert layout: (blocks_X, roundup(M_t, 4)). Compute expert base
      // offset + local stride matching cuBLAS grouped GEMM's per-expert view.
      const size_t blocks_X = DIVUP(K, static_cast<size_t>(kTileDim));
      const size_t expert_offset = expert_scale_offset_1d_rowwise<kSameBothDims>(
          tensor_id, blocks_X, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
      const size_t tensor_M = tensor_row_blocks * kTileDim;
      const size_t per_expert_stride = DIVUP_TO_MULTIPLE(tensor_M, kScaleColAlign);
      const size_t tensor_row_base = tensor_block_y_base * kTileDim;
      const size_t r_local = r_global - tensor_row_base;
      scale_inv_base[expert_offset + tile_x * per_expert_stride + r_local] = scale_inv;
    }

    if (r_global < R_total) {
      Vec<OType, kVec> out_vec;
      quantize_row_vec<IType, OType, CType, kVec>(out_vec, in_vec[it], scale);
      if (c + kVec <= K) {
        out_vec.store_to(output_base + r_global * K + c);
      } else if (c < K) {
        out_vec.store_to_elts(output_base + r_global * K + c, 0, K - c);
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
}

// ----- 1D block scaling kernel with TMA input ------------------------------------------------
// CW and BOTH path. TMA fills a 128x128 smem input cache. RW pass reads rows
// and stores quantized output. CW pass stages a column slice in registers,
// computes the per-column amax there, then fills smem_T and drains to gmem.

template <bool kSameBothDims, bool kReturnRowwise, bool kReturnColwise, typename CType,
          typename IType, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) group_block_scaled_1d_tma_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input, OType* __restrict__ output_base,
    OType* __restrict__ output_t_base, CType* __restrict__ scale_inv_base,
    CType* __restrict__ scale_inv_t_base, const int64_t* __restrict__ tensor_offsets_ptr,
    const size_t num_tensors, const size_t common_first_dim_blocks, const size_t K,
    const size_t total_row_blocks, const size_t blocks_X, const size_t scale_t_stride_aligned_K,
    const size_t R_total, const float epsilon, const bool pow_2_scales,
    const float* __restrict__ noop_ptr, float* __restrict__ dbias_workspace) {
#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) return;

  const size_t tile_x = blockIdx.x;
  const size_t tile_y_global = blockIdx.y;
  if (tile_y_global >= total_row_blocks) return;

  const size_t tile_row_stride = static_cast<size_t>(kTileDim) * K;
  const size_t tensor_id = find_tensor_id_by_block_y<kSameBothDims>(
      tile_y_global, num_tensors, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
  const size_t tensor_block_y_base =
      kSameBothDims
          ? (tensor_id * common_first_dim_blocks)
          : tensor_block_y_base_from_offsets(tensor_id, tensor_offsets_ptr, tile_row_stride);
  const size_t tensor_row_blocks =
      kSameBothDims
          ? common_first_dim_blocks
          : (tensor_block_y_base_from_offsets(tensor_id + 1, tensor_offsets_ptr, tile_row_stride) -
             tensor_block_y_base);
  if (tile_y_global >= tensor_block_y_base + tensor_row_blocks) return;

  const size_t global_row_base = tile_y_global * kTileDim;
  const size_t global_col_base = tile_x * kTileDim;

  // Dynamic smem: IType[kTileDim][kTileDim], 128 B aligned for TMA. Static smem
  // (smem_T when CW, tma_mbar) lives outside the dynamic region.
  extern __shared__ unsigned char smem_raw_1d_tma[];
  unsigned char* smem_base = common::align_smem_ptr_per_TMA_requirements(smem_raw_1d_tma);
  IType(*smem)[kTileDim] = reinterpret_cast<IType(*)[kTileDim]>(smem_base);

  __shared__ uint64_t tma_mbar;
  const int tid = threadIdx.x;
  const bool leading_thread = (tid == 0);

  // ---- TMA async load of the input tile ----
  if (leading_thread) {
    ptx::mbarrier_init(&tma_mbar, 1);
    // Fence so the TMA engine (async proxy) and the threads (generic proxy)
    // both observe the just-initialized mbarrier consistently.
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (leading_thread) {
    constexpr uint32_t tx_bytes = kTileDim * kTileDim * sizeof(IType);
    ptx::mbarrier_arrive_expect_tx(&tma_mbar, tx_bytes);
    ptx::cp_async_bulk_tensor_2d_global_to_shared_cta(
        reinterpret_cast<uint64_t*>(smem_base),
        reinterpret_cast<const uint64_t*>(&tensor_map_input),
        static_cast<uint32_t>(global_col_base), static_cast<uint32_t>(global_row_base), &tma_mbar);
  }
  ptx::mbarrier_wait_parity(&tma_mbar, 0);
  if (leading_thread) ptx::mbarrier_invalid(&tma_mbar);
  __syncthreads();

  // ---- Optional dbias: per-tile column sum of the high-precision input (smem-resident) ----
  if (dbias_workspace != nullptr) {
    write_tile_dbias_partial<IType>(smem, tid, global_col_base, K, tile_y_global, dbias_workspace);
  }

  // ---- RW pass (1x128 scale per row) ----
  // 8 t/row, vec-16 reads from smem; emits rowwise gmem directly. Only entered
  // when CW is also requested (BOTH) -- RW-only requests use the dedicated
  // group_block_scaled_1d_rw_kernel which skips the TMA load.
  if constexpr (kReturnRowwise) {
    constexpr int kThreadsPerRowRW = 8;
    constexpr int kVec = 16;
    constexpr int kRowsPerIterRW = kThreadsPerBlock / kThreadsPerRowRW;  // 32
    constexpr int kRwIters = kTileDim / kRowsPerIterRW;                  // 4

    const int rw_thr_col = tid % kThreadsPerRowRW;
    const int rw_thr_row = tid / kThreadsPerRowRW;
    const int col_local = rw_thr_col * kVec;

#pragma unroll
    for (int it = 0; it < kRwIters; ++it) {
      const int row_local = rw_thr_row + it * kRowsPerIterRW;
      Vec<IType, kVec> in_vec;
      in_vec.load_from(&smem[row_local][col_local]);

      CType amax = compute_row_amax<IType, CType, kVec>(in_vec);
      amax = subwarp_reduce_max_broadcast<kThreadsPerRowRW>(amax);

      const CType scale =
          compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scales);
      const CType scale_inv = 1.f / scale;

      const size_t r_global = global_row_base + row_local;
      const bool row_in_bounds = (r_global < R_total);

      if (row_in_bounds && rw_thr_col == 0) {
        // Per-expert layout matches the 1D RW kernel.
        const size_t blocks_X_eff = DIVUP(K, static_cast<size_t>(kTileDim));
        const size_t expert_offset = expert_scale_offset_1d_rowwise<kSameBothDims>(
            tensor_id, blocks_X_eff, common_first_dim_blocks, tile_row_stride, tensor_offsets_ptr);
        const size_t tensor_M = tensor_row_blocks * kTileDim;
        const size_t per_expert_stride = DIVUP_TO_MULTIPLE(tensor_M, kScaleColAlign);
        const size_t tensor_row_base = tensor_block_y_base * kTileDim;
        const size_t r_local = r_global - tensor_row_base;
        scale_inv_base[expert_offset + tile_x * per_expert_stride + r_local] = scale_inv;
      }

      if (row_in_bounds) {
        const size_t cc = global_col_base + col_local;
        Vec<OType, kVec> out_vec;
        quantize_row_vec<IType, OType, CType, kVec>(out_vec, in_vec, scale);
        if (cc + kVec <= K) {
          out_vec.store_to(output_base + r_global * K + cc);
        } else if (cc < K) {
          out_vec.store_to_elts(output_base + r_global * K + cc, 0, K - cc);
        }
      }
    }
  }

  if constexpr (kReturnRowwise && kReturnColwise) {
    __syncthreads();
  }

  // ---- CW pass (128x1 scale per column) ----
  // CW-only: 2 t/col, single column pass per CTA, 64-row reg_data per thread.
  // BOTH:    4 t/col, two column passes per CTA, 32-row reg_data per thread.
  //
  // In BOTH, the RW pass's per-expert offset arithmetic raises register
  // footprint past the 85-reg/thread threshold for 3 CTAs/SM on sm_90, so we
  // halve the reg_data stage (and accept the extra XOR-reduce stage plus
  // doubled column pass count) to recover that occupancy. CW-only is not
  // occupancy-bound, so its 2 t/col path stays — 4 t/col increases the
  // bank-conflict footprint on the smem load, which costs more than the extra
  // CTA/SM gains.
  //
  // The columnwise buffer is physically transposed (cuBLAS FP8 block-scaling
  // GEMM is TN-only); we stage in smem_T and drain to the per-expert (K, M_t)
  // transposed block.
  if constexpr (kReturnColwise) {
    constexpr int kSMemTRowStride = kTileDim + 4;
    __shared__ OType smem_T[kTileDim][kSMemTRowStride];

    constexpr int kThreadsPerColCW = kReturnRowwise ? 4 : 2;
    constexpr int kRowsPerThreadCW = kTileDim / kThreadsPerColCW;        // 32 (BOTH) / 64 (CW)
    constexpr int kColsPerCWPass = kThreadsPerBlock / kThreadsPerColCW;  // 64 (BOTH) / 128 (CW)
    constexpr int kCWPasses = kTileDim / kColsPerCWPass;                 // 2 (BOTH) / 1 (CW)

    const int sub = tid % kThreadsPerColCW;
    const int col_in_pass = tid / kThreadsPerColCW;
    const int row_start = sub * kRowsPerThreadCW;

#pragma unroll
    for (int pass = 0; pass < kCWPasses; ++pass) {
      const int col_local = pass * kColsPerCWPass + col_in_pass;

      CType reg_data[kRowsPerThreadCW];
      CType amax = 0.f;
#pragma unroll
      for (int e = 0; e < kRowsPerThreadCW; ++e) {
        reg_data[e] = static_cast<CType>(smem[row_start + e][col_local]);
        amax = fmaxf(amax, fabsf(reg_data[e]));
      }
      amax = subwarp_reduce_max_broadcast<kThreadsPerColCW>(amax);

      const CType scale =
          compute_scale_from_types<IType, OType>(amax, epsilon, pow_2_scales);
      const CType scale_inv = 1.f / scale;

      const size_t c_global = global_col_base + col_local;
      const bool col_in_bounds = (c_global < K);

      if (col_in_bounds && sub == 0) {
        scale_inv_t_base[c_global + tile_y_global * scale_t_stride_aligned_K] = scale_inv;
      }

      if (col_in_bounds) {
#pragma unroll
        for (int e = 0; e < kRowsPerThreadCW; ++e) {
          smem_T[col_local][row_start + e] = static_cast<OType>(reg_data[e] * scale);
        }
      }
    }
    __syncthreads();

    const size_t tensor_row_base = tensor_block_y_base * kTileDim;
    const size_t tensor_M = tensor_row_blocks * kTileDim;
    drain_smem_t_to_gmem<false, OType, kSMemTRowStride>(
        smem_T, output_t_base, global_col_base, global_row_base, tensor_row_base, tensor_M, K, tid);
  }
#endif  // __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
}

// ----- Host-side dispatchers --------------------------------------------------------------------

struct GroupedBlockwiseLaunchInfo {
  size_t num_tensors;
  size_t K;
  size_t R_total;
  size_t common_first_dim_blocks;
  size_t total_row_blocks;
  size_t blocks_X;
  bool same_both_dims;
  const int64_t* tensor_offsets_d = nullptr;
};

inline GroupedBlockwiseLaunchInfo prepare_grouped_blockwise_launch(const GroupedTensor* output) {
  GroupedBlockwiseLaunchInfo info{};
  const bool same_both_dims = output->all_same_shape();
  const bool varying_first_dim = (!output->all_same_first_dim()) && output->all_same_last_dim();
  NVTE_CHECK(same_both_dims || varying_first_dim,
             "Grouped FP8 block-scaling supports only SAME_BOTH_DIMS and VARYING_FIRST_DIM "
             "shape representations.");

  info.same_both_dims = same_both_dims;
  info.num_tensors = output->num_tensors;
  info.K = output->get_common_last_dim();
  NVTE_CHECK(info.K % TMA_GMEM_ALIGNMENT == 0,
             "Last dim must be a multiple of TMA_GMEM_ALIGNMENT (", TMA_GMEM_ALIGNMENT,
             ") for FP8 alignment.");

  if (same_both_dims) {
    const size_t common_first_dim = output->get_common_first_dim();
    NVTE_CHECK(common_first_dim % kTileDim == 0,
               "SAME_BOTH_DIMS first dim must be multiple of 128.");
    info.common_first_dim_blocks = common_first_dim / kTileDim;
    info.R_total = info.num_tensors * common_first_dim;
  } else {
    info.common_first_dim_blocks = 0;
    info.R_total = output->logical_shape.data[0];
    info.tensor_offsets_d = reinterpret_cast<const int64_t*>(output->tensor_offsets.dptr);
    NVTE_CHECK(info.tensor_offsets_d != nullptr,
               "VARYING_FIRST_DIM requires tensor_offsets to be set on the GroupedTensor.");
  }
  info.total_row_blocks = DIVUP(info.R_total, static_cast<size_t>(kTileDim));
  info.blocks_X = DIVUP(info.K, static_cast<size_t>(kTileDim));
  return info;
}

// Public dispatch — 2D block scaling.
//
// When `dbias` is non-null, also accumulate the bias gradient into `workspace` (reduced per
// expert by grouped_reduce_dbias). Two-call protocol: the sizing pass (workspace unallocated)
// reports the [total_row_blocks, K] fp32 shape and returns without launching.
inline void group_quantize_blockwise_2d(const GroupedTensor* input, GroupedTensor* output,
                                        const Tensor* noop, const float epsilon,
                                        const bool pow_2_scales, cudaStream_t stream,
                                        GroupedTensor* dbias = nullptr,
                                        Tensor* workspace = nullptr) {
  const int sm = transformer_engine::cuda::sm_arch();
  NVTE_CHECK(sm >= 90 && sm < 100,
             "Grouped FP8 block-scaling quantize is only supported on Hopper (SM90-SM99); "
             "use MXFP8 on Blackwell (SM100) or newer. Got SM",
             sm, ".");
  const bool use_rowwise = output->has_data();
  const bool use_colwise = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise || use_colwise,
             "Either rowwise or columnwise output data must be allocated.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must be FP8.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Input and output must have same num_tensors.");

  auto info = prepare_grouped_blockwise_launch(output);
  if (info.R_total == 0 || info.K == 0) return;

  float* dbias_workspace = nullptr;
  if (dbias != nullptr) {
    NVTE_CHECK(workspace != nullptr, "Workspace required for grouped FP8 block-scaling dbias.");
    NVTE_CHECK(dbias->dtype() == input->dtype(), "dbias must have the same dtype as the input.");
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {info.total_row_blocks, info.K};
      workspace->data.dtype = DType::kFloat32;
      return;  // sizing pass
    }
    dbias_workspace = reinterpret_cast<float*>(workspace->data.dptr);
  }

  using CType = float;
  const float* noop_ptr =
      (noop != nullptr) ? reinterpret_cast<const float*>(noop->data.dptr) : nullptr;

  const size_t scale_stride_y = DIVUP_TO_MULTIPLE(info.blocks_X, 4);

  dim3 grid(info.blocks_X, info.total_row_blocks, 1);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              info.same_both_dims, kSameBothDims,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_rowwise, kRowwise,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      use_colwise, kColwise, if constexpr (kRowwise || kColwise) {
                        CUtensorMap tensor_map_input{};
                        create_2D_tensor_map(tensor_map_input, input->data, info.R_total, info.K,
                                             kTileDim, kTileDim, info.K, 0, sizeof(IType) * 8);
                        auto tma_kernel =
                            group_block_scaled_2d_tma_kernel<kSameBothDims, kRowwise, kColwise,
                                                             CType, IType, OType>;
                        const size_t smem_bytes =
                            kTileDim * kTileDim * sizeof(IType) + TMA_SHMEM_ALIGNMENT - 1;
                        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                            tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(smem_bytes)));
                        tma_kernel<<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
                            tensor_map_input,
                            kRowwise ? reinterpret_cast<OType*>(output->data.dptr) : nullptr,
                            kColwise ? reinterpret_cast<OType*>(output->columnwise_data.dptr)
                                     : nullptr,
                            kRowwise ? reinterpret_cast<CType*>(output->scale_inv.dptr) : nullptr,
                            kColwise ? reinterpret_cast<CType*>(output->columnwise_scale_inv.dptr)
                                     : nullptr,
                            info.tensor_offsets_d, info.num_tensors, info.common_first_dim_blocks,
                            info.K, info.total_row_blocks, info.blocks_X, scale_stride_y, epsilon,
                            pow_2_scales, noop_ptr, dbias_workspace);
                        if (dbias_workspace != nullptr) {
                          const ShapeRepresentation shape_rep =
                              info.same_both_dims ? ShapeRepresentation::SAME_BOTH_DIMS
                                                  : ShapeRepresentation::VARYING_FIRST_DIM;
                          common::grouped_reduce_dbias<IType>(
                              shape_rep, info.num_tensors, info.R_total, info.K,
                              reinterpret_cast<const int64_t*>(output->tensor_offsets.dptr),
                              reinterpret_cast<const int64_t*>(output->first_dims.dptr),
                              reinterpret_cast<const int64_t*>(output->last_dims.dptr), dbias,
                              dbias_workspace, kTileDim, stream);
                        }
                      })))));

  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Public dispatch — 1D block scaling.
//
// See group_quantize_blockwise_2d for dbias/workspace semantics. RW-only without dbias uses the
// no-smem fast path; RW-only with dbias routes through the TMA kernel (input in smem) so the
// per-tile column partial can be computed.
inline void group_quantize_blockwise_1d(const GroupedTensor* input, GroupedTensor* output,
                                        const Tensor* noop, const float epsilon,
                                        const bool pow_2_scales, cudaStream_t stream,
                                        GroupedTensor* dbias = nullptr,
                                        Tensor* workspace = nullptr) {
  const int sm = transformer_engine::cuda::sm_arch();
  NVTE_CHECK(sm >= 90 && sm < 100,
             "Grouped FP8 block-scaling quantize is only supported on Hopper (SM90-SM99); "
             "use MXFP8 on Blackwell (SM100) or newer. Got SM",
             sm, ".");
  const bool use_rowwise = output->has_data();
  const bool use_colwise = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise || use_colwise,
             "Either rowwise or columnwise output data must be allocated.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must be FP8.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Input and output must have same num_tensors.");

  auto info = prepare_grouped_blockwise_launch(output);
  if (info.R_total == 0 || info.K == 0) return;

  float* dbias_workspace = nullptr;
  if (dbias != nullptr) {
    NVTE_CHECK(workspace != nullptr, "Workspace required for grouped FP8 block-scaling dbias.");
    NVTE_CHECK(dbias->dtype() == input->dtype(), "dbias must have the same dtype as the input.");
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {info.total_row_blocks, info.K};
      workspace->data.dtype = DType::kFloat32;
      return;  // sizing pass
    }
    dbias_workspace = reinterpret_cast<float*>(workspace->data.dptr);
  }

  using CType = float;
  const float* noop_ptr =
      (noop != nullptr) ? reinterpret_cast<const float*>(noop->data.dptr) : nullptr;

  const size_t scale_t_stride_aligned_K = DIVUP_TO_MULTIPLE(info.K, 4);

  dim3 grid(info.blocks_X, info.total_row_blocks, 1);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              info.same_both_dims, kSameBothDims,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  use_rowwise, kRowwise,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      use_colwise, kColwise,
                      // RW-only without dbias uses the no-smem fast path; all else uses TMA.
                      constexpr bool kRwOnly = kRowwise && !kColwise;
                      const bool use_rw_fast_path = kRwOnly && (dbias_workspace == nullptr);
                      if (use_rw_fast_path) {
                        if constexpr (kRwOnly) {
                          group_block_scaled_1d_rw_kernel<kSameBothDims, CType, IType, OType>
                              <<<grid, kThreadsPerBlock, 0, stream>>>(
                                  reinterpret_cast<const IType*>(input->data.dptr),
                                  reinterpret_cast<OType*>(output->data.dptr),
                                  reinterpret_cast<CType*>(output->scale_inv.dptr),
                                  info.tensor_offsets_d, info.num_tensors,
                                  info.common_first_dim_blocks, info.K, info.total_row_blocks,
                                  info.R_total, epsilon, pow_2_scales, noop_ptr);
                        }
                      } else if constexpr (kRowwise || kColwise) {
                        // CW-only, BOTH, or RW-only WITH dbias: smem-cached TMA kernel.
                        const size_t smem_bytes = kTileDim * kTileDim * sizeof(IType);
                        constexpr size_t kStaticSmemCWBytes =
                            (kTileDim * (kTileDim + 4)) * sizeof(OType);
                        const size_t static_smem_bytes = kColwise ? kStaticSmemCWBytes : 0;
                        const size_t tma_smem_bytes = smem_bytes + TMA_SHMEM_ALIGNMENT - 1;
                        const size_t total_smem_tma = tma_smem_bytes + static_smem_bytes;
                        auto tma_kernel =
                            group_block_scaled_1d_tma_kernel<kSameBothDims, kRowwise, kColwise,
                                                             CType, IType, OType>;
                        if (total_smem_tma >= 48 * 1024) {
                          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                              tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(tma_smem_bytes)));
                        }
                        CUtensorMap tensor_map_input{};
                        create_2D_tensor_map(tensor_map_input, input->data, info.R_total, info.K,
                                             kTileDim, kTileDim, info.K, 0, sizeof(IType) * 8);
                        tma_kernel<<<grid, kThreadsPerBlock, tma_smem_bytes, stream>>>(
                            tensor_map_input,
                            kRowwise ? reinterpret_cast<OType*>(output->data.dptr) : nullptr,
                            kColwise ? reinterpret_cast<OType*>(output->columnwise_data.dptr)
                                     : nullptr,
                            kRowwise ? reinterpret_cast<CType*>(output->scale_inv.dptr) : nullptr,
                            kColwise ? reinterpret_cast<CType*>(output->columnwise_scale_inv.dptr)
                                     : nullptr,
                            info.tensor_offsets_d, info.num_tensors, info.common_first_dim_blocks,
                            info.K, info.total_row_blocks, info.blocks_X, scale_t_stride_aligned_K,
                            info.R_total, epsilon, pow_2_scales, noop_ptr, dbias_workspace);
                        if (dbias_workspace != nullptr) {
                          const ShapeRepresentation shape_rep =
                              info.same_both_dims ? ShapeRepresentation::SAME_BOTH_DIMS
                                                  : ShapeRepresentation::VARYING_FIRST_DIM;
                          common::grouped_reduce_dbias<IType>(
                              shape_rep, info.num_tensors, info.R_total, info.K,
                              reinterpret_cast<const int64_t*>(output->tensor_offsets.dptr),
                              reinterpret_cast<const int64_t*>(output->first_dims.dptr),
                              reinterpret_cast<const int64_t*>(output->last_dims.dptr), dbias,
                              dbias_workspace, kTileDim, stream);
                        }
                      })))));

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fp8_blockwise
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_BLOCKWISE_CUH_
