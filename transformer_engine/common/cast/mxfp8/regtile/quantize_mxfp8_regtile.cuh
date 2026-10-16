/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8_regtile.cuh
 *  \brief Register-resident MXFP8 bidimensional quantize fused with GeLU or SiLU, or
 *         their derivatives, optionally with dbias.
 *
 *  Rowwise and colwise MX scales need an amax over the same data in two
 *  directions.  The generic kernel stages each tile through shared memory and
 *  walks it twice.  Here a CTA covers whole 32-row bands, the colwise block
 *  height, so the tile is read once into registers and both directions are
 *  quantized from there; shared memory holds only the cross-warp colwise
 *  partials, the row scales and the activation table.
 *
 *  The activation or derivative is read from the table the caller passes (see
 *  util/activation_table.cuh), copied into shared memory once per CTA, or, without a
 *  table, computed with util/math.h. Either way this kernel produces exactly the bytes the
 *  generic kernel produces for the same request.
 *
 *  It replaces the generic kernel, not the specialized ones: cast-only requests
 *  are served before the generic dispatch by the specialized kernels, one of
 *  which (specialized/cast_bidim.cuh) is also register-resident.  Anything this
 *  kernel does not implement (see can_use) takes the generic kernel.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#include "../../../common.h"
#include "../../../util/activation_table.cuh"
#include "../../../util/cuda_runtime.h"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"
#include "../../../util/ptx_arch_spec.cuh"
#include "../../../utils.cuh"
#include "../swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace regtile {

namespace ptx = transformer_engine::ptx;

namespace {

// The datapath works on 32-bit words, each holding a BF16 column pair.

/*! \brief Elementwise max(|a|, |b|) on a packed BF16 pair, as a raw word. */
__device__ __forceinline__ unsigned abs_max_bf16x2(unsigned a, unsigned b) {
  ptx::bf16x2 d;
  ptx::abs_max_2x(d, reinterpret_cast<const ptx::bf16x2&>(a),
                  reinterpret_cast<const ptx::bf16x2&>(b));
  return reinterpret_cast<const unsigned&>(d);
}

/*! \brief Widen a BF16 pair word to FP32, low half to x.  Same result as ptx::up_cast, whose
 *  volatile asm the compiler cannot schedule as freely in the derivative loop. */
__device__ __forceinline__ ptx::floatx2 widen_bf16x2(const unsigned v) {
  return {__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u)};
}

/*! \brief Round an FP32 pair to a BF16 pair word, low lane in the low half. */
__device__ __forceinline__ unsigned to_bf16x2(const ptx::floatx2& v) {
  const __nv_bfloat162 r = __floats2bfloat162_rn(v.x, v.y);
  return reinterpret_cast<const unsigned&>(r);
}

/*! \brief The reciprocal of an E8M0 scale, 2^(127-e), broadcast to a BF16 pair. */
__device__ __forceinline__ unsigned e8m0_to_bf16x2_reciprocal(unsigned e) {
  const ptx::bf16x2 r = ptx::exp2f_rcp_2x(static_cast<e8m0_t>(e));
  return reinterpret_cast<const unsigned&>(r);
}

/*! \brief E8M0 scales for two block amaxes, packed as ptx::float_to_e8m0_2x.
 *
 * The generic kernel's amax is a 0-seeded max that ignores NaN, so a block of
 * only NaNs gets amax 0; the fmaxf reproduces that for the data-seeded maxima
 * here.  The conversion saturates, so the scale never reaches 255.
 */
template <typename OType>
__device__ __forceinline__ unsigned amax_to_e8m0_2x(float amax_hi, float amax_lo) {
  constexpr float kMaxNormRcp = Quantized_Limits<OType>::max_norm_rcp;
  return ptx::float_to_e8m0_2x(fmaxf(amax_hi * kMaxNormRcp, 0.0f),
                               fmaxf(amax_lo * kMaxNormRcp, 0.0f));
}

/*! \brief Table<kind>'s activation computed with util/math.h instead of read from a table:
 *         the generic kernel's output without tables (NVTE_ACTIVATION_TABLE=0). No table is
 *         copied.
 */
template <activation_table::Kind kKind>
struct ComputedTable : activation_table::Table<kKind> {
  static constexpr bool kComputed = true;
  static constexpr size_t kBytes = 16;  // unused shared-memory slot
};

/*! \brief The derivative of a BF16 pair, from the table in shared memory. */
template <typename TableT>
__device__ __forceinline__ ptx::floatx2 dact_word(unsigned x,
                                                  const unsigned char* __restrict__ table) {
  if constexpr (TableT::kComputed) {
    const ptx::floatx2 v = widen_bf16x2(x);
    return {TableT::op(v.x), TableT::op(v.y)};
  } else {
    const float2 d = activation_table::derivative_2x<TableT>(x, table);
    return {d.x, d.y};
  }
}

/*! \brief Start copying a table of kBytes into shared memory with one bulk copy.
 *
 *  Called by every thread of the CTA; contains a __syncthreads.
 */
template <size_t kBytes>
__device__ __forceinline__ void load_table_async(unsigned char* smem, uint64_t* barrier,
                                                 const void* table, const int tid) {
  static_assert(kBytes % 16 == 0, "A bulk copy moves whole 16-byte units.");
  if (tid == 0) {
    ptx::mbarrier_init(barrier, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (tid == 0) {
    ptx::mbarrier_arrive_expect_tx(barrier, kBytes);
    ptx::cp_async_bulk_tensor_1d_global_to_shared(reinterpret_cast<uint64_t*>(smem),
                                                  reinterpret_cast<const uint64_t*>(table), kBytes,
                                                  barrier);
  }
}

// Tile geometry.  A CTA covers 256 columns and walks 32-row sub-tiles; 32 rows
// is the colwise MX block height, so the colwise reduction closes in the CTA.

//! Columns a CTA covers.
constexpr int kTileCols = 256;
//! Elements in one MX block, and equivalently the colwise block height.
constexpr int kRowsPerMxBlock = 32;
//! MX groups (32 columns each) across the tile.
constexpr int kMxGroupsPerRow = kTileCols / 32;
//! A register-resident tile spreads its 256 columns over the 32 lanes of a warp.
constexpr int kColsPerLane = kTileCols / 32;
//! ...which a lane holds as BF16 pairs, one 32-bit word each.
constexpr int kWordsPerLane = kColsPerLane / 2;
//! ...read as 128-bit vectors, so one per row.
constexpr int kVecLoadsPerRow = kWordsPerLane / 4;
//! Lanes that have to cooperate to cover one 32-column MX group.
constexpr int kLanesPerMxGroup = 32 / kColsPerLane;

/*! \brief Scale two BF16 pairs, each by its own reciprocal scale pair, to four OType (E4M3 or
 *         E5M2). */
template <typename OType>
__device__ __forceinline__ unsigned quantize_x4(unsigned in01, unsigned in23, unsigned scale01,
                                                unsigned scale23) {
  ptx::FPx4<OType> out;
  ptx::mul_cvt_4x(out, reinterpret_cast<const ptx::bf16x2&>(in01),
                  reinterpret_cast<const ptx::bf16x2&>(scale01),
                  reinterpret_cast<const ptx::bf16x2&>(in23),
                  reinterpret_cast<const ptx::bf16x2&>(scale23));
  return reinterpret_cast<const unsigned&>(out);
}

/*! \brief Store one lane's quantized row: kColsPerLane FP8 bytes, one STG.64. */
__device__ __forceinline__ void store_quantized(unsigned char* p, const unsigned* v) {
  static_assert(kColsPerLane == 8, "One STG.64 covers exactly eight FP8 bytes.");
  *(uint2*)p = *(const uint2*)v;
}

// Minimum resident CTAs per SM, the second __launch_bounds__ argument, capped
// at what the SM can hold: sm_75 and sm_107 have half the threads per SM of sm_100.
constexpr int kMinBlocksAct = 8;
constexpr int kMinBlocksDact = 6;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 1070)
constexpr int kMaxThreadsPerSm = 1024;
#else
constexpr int kMaxThreadsPerSm = 2048;
#endif

// Minimum resident CTAs per SM with dbias: the 32 KB staging buffer allows about four.
constexpr int kMinBlocksDbias = 4;

template <bool IS_DACT, bool IS_ACT, typename TableT, bool kDbias = false>
struct TileConfig {
  static_assert(IS_ACT != IS_DACT, "Exactly one of the activation and its derivative is fused.");
  static constexpr int kThreadsPerCta = IS_DACT ? 256 : 128;
  static constexpr int kWarpsPerCta = kThreadsPerCta / 32;
  static constexpr int kRowsPerWarp = 32 / kWarpsPerCta;
  // The rowwise half is quantized before the cross-warp barrier and the row
  // scales are drained inside the colwise barrier interval, saving one barrier
  // per 32-row tile.  A tile too narrow to leave idle threads hands the drain to
  // the threads that just folded the colwise partials.
  static constexpr int kDrainTidBase =
      (kThreadsPerCta >= kTileCols / 2 + kTileCols / 4) ? kTileCols / 2 : 0;
  static constexpr int kRowsInFlight = 2;
  static constexpr int kRequestedMinBlocks =
      kDbias ? kMinBlocksDbias : (IS_DACT ? kMinBlocksDact : kMinBlocksAct);
  static constexpr int kMinBlocksPerSm = kRequestedMinBlocks < kMaxThreadsPerSm / kThreadsPerCta
                                             ? kRequestedMinBlocks
                                             : kMaxThreadsPerSm / kThreadsPerCta;
  static constexpr int kTableBytes = static_cast<int>(TableT::kBytes);
  //! Dynamic shared memory: with dbias, the band's FP32 products before rounding.
  static constexpr int kDbiasStageBytes = kDbias ? kRowsPerMxBlock * kTileCols * 4 : 0;
  static_assert(!kDbias || (IS_DACT && kThreadsPerCta == kTileCols),
                "dbias: one thread per column sums the column.");
};

// Register-resident tiling: the 32x256 tile stays in registers; only the
// colwise partials go through shared memory.
// kTails: the grid may cover elements past M or K (see shape_supported). Without tails the
// bounds checks compile away, which matters at the kernel's register budget.
//
// kDbias: also write the per-64-row-chunk column sums of the FP32 products (before rounding to
// BF16) into dbias_workspace[chunk][K], as the generic kernel does; common::reduce_dbias then
// adds the chunks. The generic kernel sums each column of a chunk strictly in row order, so a
// CTA here takes the chunk's two 32-row bands back to back, stages each band's products in
// shared memory, and one thread per column adds them in row order.
//
// kAmax: also fold the amax of the output into *amax_ptr.
template <bool IS_DACT, bool IS_ACT, typename TableT, typename OType,
          bool WITH_GEMM_SWIZZLED_SCALES, bool kTails, bool kDbias, bool kAmax>
__device__ __forceinline__ void quantize_regtile(
    const uint4* __restrict__ input, const uint4* __restrict__ act_input,
    unsigned char* __restrict__ rowwise_out, unsigned char* __restrict__ rowwise_scales,
    unsigned char* __restrict__ colwise_out, unsigned char* __restrict__ colwise_scales, int M,
    int K, int rowwise_scale_stride, int colwise_scale_stride, int iters, const void* table,
    float* __restrict__ dbias_workspace, float* __restrict__ amax_ptr) {
  using C = TileConfig<IS_DACT, IS_ACT, TableT, kDbias>;
  constexpr int kWarpsPerCta = C::kWarpsPerCta, kRowsPerWarp = C::kRowsPerWarp,
                kRowsInFlight = C::kRowsInFlight;

  __shared__ __align__(16) unsigned col_amax_partials[kWarpsPerCta][kTileCols / 2];
  __shared__ __align__(16) unsigned col_scale_rcp_smem[kTileCols / 2];
  __shared__ __align__(8) unsigned char row_scale_bytes[kRowsPerMxBlock * kMxGroupsPerRow];
  __shared__ __align__(16) unsigned char table_smem[C::kTableBytes];
  __shared__ __align__(8) uint64_t table_barrier;
  extern __shared__ __align__(16) float dbias_stage[];  // [kRowsPerMxBlock][kTileCols], dbias only
  float dbias_partial = 0.0f;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col0 = blockIdx.x * kTileCols;
  const int vecs_per_row = K >> 3;  // uint4 (8 bf16 values) per row
  // Tails: the last column tile may end inside the tile, and row bands may end past M. Out-of-range
  // elements load as 0, which every activation maps to 0, so they never raise an amax. The
  // generic kernel writes scales for its whole 64x64 tiles, and so does this one: rowwise scales
  // for every row of the grid's bands, for the groups that start inside the tensor; colwise
  // scales for the columns below K rounded up to 64.
  const bool full_width = !kTails || col0 + kTileCols <= K;
  const bool lane_in_cols = col0 + kColsPerLane * lane < K;
  const int colwise_scale_cols = (K + 63) / 64 * 64;

  // The table copy overlaps the first input loads; see the wait below.
  if constexpr (!TableT::kComputed) {
    load_table_async<C::kTableBytes>(table_smem, &table_barrier, table, tid);
  }

  float thread_amax = 0.0f;

#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    // Row blocks are taken grid-strided, so the CTAs on the same step cover one
    // contiguous row band.
    // With dbias a CTA takes the two bands of a 64-row chunk on consecutive steps.
    const int row0 = kDbias
                         ? (((it >> 1) * gridDim.y + blockIdx.y) * 2 + (it & 1)) * kRowsPerMxBlock
                         : (it * gridDim.y + blockIdx.y) * kRowsPerMxBlock;
    if constexpr (kDbias) {
      if ((it & 1) == 0) dbias_partial = 0.0f;
    }
    // The band's work. A band inside the tensor runs it without bounds checks; only bands at
    // the edges of the tensor (kEdge) pay for them.
    const auto band = [&](auto edge) {
      constexpr bool kEdge = decltype(edge)::value;
      const int warp_row0 = row0 + warp * kRowsPerWarp;
      // Whether this lane's elements of row warp_row0 + j are inside the tensor.
      const auto in_bounds = [&](int j) { return !kEdge || (lane_in_cols && warp_row0 + j < M); };
      // This thread's kRowsPerWarp x kColsPerLane sub-tile, as BF16 pairs.
      unsigned tile_words[kRowsPerWarp * kWordsPerLane];
      const size_t load_offset = (size_t)(row0 + warp * kRowsPerWarp) * vecs_per_row + (col0 >> 3) +
                                 (size_t)kVecLoadsPerRow * lane;
      const uint4* input_ptr = input + load_offset;
      const uint4* act_input_ptr = act_input + load_offset;

      // ---- load + activate, kRowsInFlight rows in flight at a time --------------------
#pragma unroll
      for (int h = 0; h < kRowsPerWarp / kRowsInFlight; ++h) {
        uint4 x_vec[kRowsInFlight][kVecLoadsPerRow], grad_vec[kRowsInFlight][kVecLoadsPerRow];
        asm volatile("" ::: "memory");
#pragma unroll
        for (int t = 0; t < kRowsInFlight; ++t)
#pragma unroll
          for (int n = 0; n < kVecLoadsPerRow; ++n)
            x_vec[t][n] = in_bounds(h * kRowsInFlight + t)
                              ? act_input_ptr[(size_t)(h * kRowsInFlight + t) * vecs_per_row + n]
                              : make_uint4(0, 0, 0, 0);
        if constexpr (IS_DACT) {
#pragma unroll
          for (int t = 0; t < kRowsInFlight; ++t)
#pragma unroll
            for (int n = 0; n < kVecLoadsPerRow; ++n)
              grad_vec[t][n] = in_bounds(h * kRowsInFlight + t)
                                   ? input_ptr[(size_t)(h * kRowsInFlight + t) * vecs_per_row + n]
                                   : make_uint4(0, 0, 0, 0);
        }
        // Keep the load group's loads together, between the two empty asm statements: issued
        // back to back, their latencies overlap. Otherwise register pressure leads the compiler
        // to spread them between the previous group's table lookups, each then stalling on its
        // own.
        asm volatile("" ::: "memory");
        if constexpr (!TableT::kComputed) {
          if (it == 0 && h == 0) ptx::mbarrier_wait_parity(&table_barrier, 0);
        }
        // Convert the load group into the BF16 pairs the epilogue quantizes.
        if constexpr (!IS_DACT) {
#pragma unroll
          for (int t = 0; t < kRowsInFlight; ++t) {
#pragma unroll
            for (int n = 0; n < kVecLoadsPerRow; ++n) {
              const unsigned* x_words = (const unsigned*)&x_vec[t][n];
              const int tile_word_base = (h * kRowsInFlight + t) * kWordsPerLane + n * 4;
#pragma unroll
              for (int m = 0; m < 4; ++m) {
                if constexpr (TableT::kComputed) {
                  // util/math.h, then rounded to BF16 as the generic kernel rounds it.
                  const ptx::floatx2 v = widen_bf16x2(x_words[m]);
                  tile_words[tile_word_base + m] = to_bf16x2({TableT::op(v.x), TableT::op(v.y)});
                } else {
                  tile_words[tile_word_base + m] = activation_table::forward_2x<TableT>(
                      x_words[m], reinterpret_cast<const __nv_bfloat16*>(table_smem));
                }
              }
            }
          }
        } else {
#define MXFP8_DACT_WORD(T, M)                                                                \
  {                                                                                          \
    const ptx::floatx2 product =                                                             \
        ptx::mul_2x(widen_bf16x2(grad_words[M]), dact_word<TableT>(x_words[M], table_smem)); \
    tile_words[(h * kRowsInFlight + (T)) * kWordsPerLane + (M)] = to_bf16x2(product);        \
    products[2 * (M)] = product.x;                                                           \
    products[2 * (M) + 1] = product.y;                                                       \
  }
#define MXFP8_DACT_ROW(T)                                                                       \
  {                                                                                             \
    const unsigned* x_words = (const unsigned*)&x_vec[T][0];                                    \
    const unsigned* grad_words = (const unsigned*)&grad_vec[T][0];                              \
    float products[2 * kWordsPerLane];                                                          \
    MXFP8_DACT_WORD(T, 0)                                                                       \
    MXFP8_DACT_WORD(T, 1)                                                                       \
    MXFP8_DACT_WORD(T, 2)                                                                       \
    MXFP8_DACT_WORD(T, 3)                                                                       \
    if constexpr (kDbias) {                                                                     \
      float4* stage =                                                                           \
          (float4*)(dbias_stage + (warp * kRowsPerWarp + h * kRowsInFlight + (T)) * kTileCols + \
                    kColsPerLane * lane);                                                       \
      stage[0] = make_float4(products[0], products[1], products[2], products[3]);               \
      stage[1] = make_float4(products[4], products[5], products[6], products[7]);               \
    }                                                                                           \
  }
          static_assert(kRowsInFlight == 2, "A load group is two rows.");
          MXFP8_DACT_ROW(0)
          MXFP8_DACT_ROW(1)
#undef MXFP8_DACT_ROW
#undef MXFP8_DACT_WORD
        }
      }

      // The rowwise half does not depend on the colwise scales, so it runs before
      // the cross-warp barrier (see TileConfig).
      const size_t store_offset =
          (size_t)(row0 + warp * kRowsPerWarp) * K + col0 + kColsPerLane * lane;
      auto row_scale = [&](const unsigned* row, int j) {
        unsigned amax =
            abs_max_bf16x2(abs_max_bf16x2(row[0], row[1]), abs_max_bf16x2(row[2], row[3]));
#pragma unroll
        for (int m = 4; m < kWordsPerLane; m += 2)
          amax = abs_max_bf16x2(amax, abs_max_bf16x2(row[m], row[m + 1]));
          // A 32-value rowwise block spans kLanesPerMxGroup lanes.
#pragma unroll
        for (int lane_mask = 1; lane_mask < kLanesPerMxGroup; lane_mask <<= 1)
          amax = abs_max_bf16x2(amax, __shfl_xor_sync(0xffffffffu, amax, lane_mask));
        // Fold the two halves with one PRMT and a packed max, which leaves the
        // magnitude in both halves; the high half is then the FP32 bit pattern.
        amax = abs_max_bf16x2(amax, __byte_perm(amax, amax, 0x1032));
        const unsigned scale_byte =
            amax_to_e8m0_2x<OType>(0.f, __uint_as_float(amax & 0x7fff0000u));
        if ((lane & (kLanesPerMxGroup - 1)) == 0)
          row_scale_bytes[(warp * kRowsPerWarp + j) * kMxGroupsPerRow + (lane / kLanesPerMxGroup)] =
              (unsigned char)scale_byte;
        return e8m0_to_bf16x2_reciprocal(scale_byte);
      };
      // On the 128-thread activation tile, two rows share one butterfly and conversion.
      auto row_amax = [&](const unsigned* row) {
        unsigned amax =
            abs_max_bf16x2(abs_max_bf16x2(row[0], row[1]), abs_max_bf16x2(row[2], row[3]));
#pragma unroll
        for (int m = 4; m < kWordsPerLane; m += 2)
          amax = abs_max_bf16x2(amax, abs_max_bf16x2(row[m], row[m + 1]));
        return abs_max_bf16x2(amax, __byte_perm(amax, amax, 0x1032));
      };
      auto row_scale_pair = [&](const unsigned* row_a, const unsigned* row_b, int j,
                                unsigned& row_scale_rcp0, unsigned& row_scale_rcp1) {
        unsigned amax_pair = __byte_perm(row_amax(row_a), row_amax(row_b), 0x5410) & 0x7fff7fffu;
#pragma unroll
        for (int lane_mask = 1; lane_mask < kLanesPerMxGroup; lane_mask <<= 1)
          amax_pair = abs_max_bf16x2(amax_pair, __shfl_xor_sync(0xffffffffu, amax_pair, lane_mask));
        const ptx::floatx2 amax_2x = widen_bf16x2(amax_pair);
        const unsigned scale_pair = amax_to_e8m0_2x<OType>(amax_2x.y, amax_2x.x);
        const unsigned scale_byte0 = scale_pair & 0xffu, scale_byte1 = (scale_pair >> 8) & 0xffu;
        if ((lane & (kLanesPerMxGroup - 1)) == 0) {
          unsigned char* scale_dst = row_scale_bytes + (warp * kRowsPerWarp + j) * kMxGroupsPerRow +
                                     (lane / kLanesPerMxGroup);
          scale_dst[0] = (unsigned char)scale_byte0;
          scale_dst[kMxGroupsPerRow] = (unsigned char)scale_byte1;
        }
        row_scale_rcp0 = e8m0_to_bf16x2_reciprocal(scale_byte0);
        row_scale_rcp1 = e8m0_to_bf16x2_reciprocal(scale_byte1);
      };
      // Colwise amax: per-warp partial in registers, cross-warp fold in shared.
      auto publish_col_amax_partials = [&]() {
        unsigned col_amax[kWordsPerLane];
#pragma unroll
        for (int m = 0; m < kWordsPerLane; ++m) col_amax[m] = tile_words[m];
#pragma unroll
        for (int j = 1; j < kRowsPerWarp; ++j)
#pragma unroll
          for (int m = 0; m < kWordsPerLane; ++m)
            col_amax[m] = abs_max_bf16x2(col_amax[m], tile_words[j * kWordsPerLane + m]);
#pragma unroll
        for (int n = 0; n < kVecLoadsPerRow; ++n)
          *(uint4*)(&col_amax_partials[warp][kWordsPerLane * lane + 4 * n]) =
              *(const uint4*)(col_amax + 4 * n);
      };
      if constexpr (C::kThreadsPerCta == 128) {
        unsigned char* row_out_ptr = rowwise_out + store_offset;
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; j += 2) {
          const unsigned* row_a = tile_words + j * kWordsPerLane;
          const unsigned* row_b = row_a + kWordsPerLane;
          unsigned row_scale_rcp0, row_scale_rcp1;
          row_scale_pair(row_a, row_b, j, row_scale_rcp0, row_scale_rcp1);
          unsigned row_out[kWordsPerLane / 2];
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            row_out[m] =
                quantize_x4<OType>(row_a[2 * m], row_a[2 * m + 1], row_scale_rcp0, row_scale_rcp0);
          if (in_bounds(j)) store_quantized(row_out_ptr + (unsigned)j * (unsigned)K, row_out);
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            row_out[m] =
                quantize_x4<OType>(row_b[2 * m], row_b[2 * m + 1], row_scale_rcp1, row_scale_rcp1);
          if (in_bounds(j + 1))
            store_quantized(row_out_ptr + (unsigned)(j + 1) * (unsigned)K, row_out);
        }
      } else {
        unsigned char* row_out_ptr = rowwise_out + store_offset;
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; ++j) {
          const unsigned* v = tile_words + j * kWordsPerLane;
          const unsigned row_scale_rcp = row_scale(v, j);
          unsigned row_out[kWordsPerLane / 2];
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            row_out[m] = quantize_x4<OType>(v[2 * m], v[2 * m + 1], row_scale_rcp, row_scale_rcp);
          if (in_bounds(j)) store_quantized(row_out_ptr + (unsigned)j * (unsigned)K, row_out);
        }
      }

      publish_col_amax_partials();
      __syncthreads();
      if constexpr (kDbias) {
        // The generic kernel's order: a strictly sequential FP32 sum down the column.
        float sum = dbias_partial;
#pragma unroll 8
        for (int r = 0; r < kRowsPerMxBlock; ++r)
          sum = __fadd_rn(sum, dbias_stage[r * kTileCols + tid]);
        dbias_partial = sum;
      }
      if (tid < kTileCols / 2) {
        unsigned col_amax = col_amax_partials[0][tid];
#pragma unroll
        for (int w = 1; w < kWarpsPerCta; ++w)
          col_amax = abs_max_bf16x2(col_amax, col_amax_partials[w][tid]);
        col_amax &= 0x7fff7fffu;
        const ptx::floatx2 amax_2x = widen_bf16x2(col_amax);
        // The tensor amax, when asked for: the max of these column amaxes. Like the generic
        // kernel's per-element fmaxf, it ignores NaN (a column of only NaNs gives NaN here,
        // which fmaxf drops) and is unaffected by the zeros loaded past M or K.
        if constexpr (kAmax) thread_amax = fmaxf(thread_amax, fmaxf(amax_2x.x, amax_2x.y));
        const unsigned scale_pair = amax_to_e8m0_2x<OType>(amax_2x.y, amax_2x.x);
        if (!kEdge || col0 + 2 * tid < colwise_scale_cols) {
          if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
            // The colwise swizzle is the rowwise one with rows and columns swapped,
            // so the two columns of this pair land in different 16-byte rows.
            const size_t row_block_tiles = DIVUP(static_cast<size_t>(M), size_t{128});
            const size_t col = col0 + 2 * tid;
            colwise_scales[swizzle::gemm_swizzled_scale_idx(col, row0 / 32, row_block_tiles)] =
                static_cast<unsigned char>(scale_pair);
            colwise_scales[swizzle::gemm_swizzled_scale_idx(col + 1, row0 / 32, row_block_tiles)] =
                static_cast<unsigned char>(scale_pair >> 8);
          } else {
            *(unsigned short*)(colwise_scales + (size_t)(row0 / 32) * colwise_scale_stride + col0 +
                               2 * tid) = (unsigned short)scale_pair;
          }
        }
        const ptx::bf16x2 scale_rcp =
            ptx::exp2f_rcp_2x_per_lane(__byte_perm(scale_pair, 0, 0x4140));
        col_scale_rcp_smem[tid] = reinterpret_cast<const unsigned&>(scale_rcp);
      }
      auto drain_row_scales = [&]() {
        constexpr int drain_tid_base = C::kDrainTidBase;
        // A row's eight group scales are eight contiguous, 4-byte-aligned bytes of
        // rowwise_scales (the stride is a multiple of 4 groups), so one warp drains a
        // 32-row tile with two STG.32 per lane. In the GEMM-swizzled layout they are two
        // 4-byte runs in adjacent tiles. A tile that ends inside the tensor writes only
        // the groups that start inside it.
        if (tid >= drain_tid_base && tid < drain_tid_base + kRowsPerMxBlock) {
          const int r = tid - drain_tid_base;
          const unsigned char* bytes = row_scale_bytes + r * kMxGroupsPerRow;
          const size_t col_tiles = DIVUP(static_cast<size_t>(K), size_t{128});
          if (!kEdge || full_width) {
            const uint2 words = *(const uint2*)bytes;
            if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
              constexpr size_t kTileBytes =
                  swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_X * swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_Y;
              const size_t idx = swizzle::gemm_swizzled_scale_idx(row0 + r, col0 >> 5, col_tiles);
              *(unsigned*)(rowwise_scales + idx) = words.x;
              *(unsigned*)(rowwise_scales + idx + kTileBytes) = words.y;
            } else {
              unsigned* dst = (unsigned*)(rowwise_scales +
                                          (size_t)(row0 + r) * rowwise_scale_stride + (col0 >> 5));
              dst[0] = words.x;
              dst[1] = words.y;
            }
          } else {
            const int groups = (K - col0 + 31) / 32;
            for (int g = 0; g < groups; ++g) {
              size_t idx;
              if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
                idx = swizzle::gemm_swizzled_scale_idx(row0 + r, (col0 >> 5) + g, col_tiles);
              } else {
                idx = (size_t)(row0 + r) * rowwise_scale_stride + (col0 >> 5) + g;
              }
              rowwise_scales[idx] = bytes[g];
            }
          }
        }
      };
      drain_row_scales();
      __syncthreads();

      // ---- columnwise quantization --------------------------------------------
      {
        uint4 col_scale_rcp_vec[kVecLoadsPerRow];
#pragma unroll
        for (int n = 0; n < kVecLoadsPerRow; ++n)
          col_scale_rcp_vec[n] = *(const uint4*)(col_scale_rcp_smem + kWordsPerLane * lane + 4 * n);
        const unsigned* col_scale_rcp = (const unsigned*)col_scale_rcp_vec;
        unsigned char* col_out_ptr = colwise_out + store_offset;
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; ++j) {
          const unsigned* v = tile_words + j * kWordsPerLane;
          unsigned col_out[kWordsPerLane / 2];
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            col_out[m] = quantize_x4<OType>(v[2 * m], v[2 * m + 1], col_scale_rcp[2 * m],
                                            col_scale_rcp[2 * m + 1]);
          if (in_bounds(j)) store_quantized(col_out_ptr + (unsigned)j * (unsigned)K, col_out);
        }
      }
    };
    if constexpr (kTails) {
      if (full_width && row0 + kRowsPerMxBlock <= M) {
        band(std::false_type{});
      } else {
        band(std::true_type{});
      }
    } else {
      band(std::false_type{});
    }
    if constexpr (kDbias) {
      if ((it & 1) == 1 && col0 + tid < K) {
        dbias_workspace[(size_t)(row0 / (2 * kRowsPerMxBlock)) * K + col0 + tid] = dbias_partial;
      }
    }
  }

  if constexpr (kAmax) {
    const float block_amax = reduce_max<kWarpsPerCta>(thread_amax, warp);
    if (tid == 0) atomicMaxFloat(amax_ptr, block_amax);
  }
}

template <bool IS_DACT, bool IS_ACT, typename TableT, typename OType,
          bool WITH_GEMM_SWIZZLED_SCALES, bool kTails, bool kDbias, bool kAmax>
__global__ void __launch_bounds__(TileConfig<IS_DACT, IS_ACT, TableT, kDbias>::kThreadsPerCta,
                                  TileConfig<IS_DACT, IS_ACT, TableT, kDbias>::kMinBlocksPerSm)
    quantize_mxfp8_kernel(const unsigned* __restrict__ input,
                          const unsigned* __restrict__ act_input,
                          unsigned char* __restrict__ rowwise_out,
                          unsigned char* __restrict__ rowwise_scales,
                          unsigned char* __restrict__ colwise_out,
                          unsigned char* __restrict__ colwise_scales, int M, int K,
                          int rowwise_scale_stride, int colwise_scale_stride, int iters,
                          const void* __restrict__ table, float* __restrict__ dbias_workspace,
                          float* __restrict__ amax_ptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  quantize_regtile<IS_DACT, IS_ACT, TableT, OType, WITH_GEMM_SWIZZLED_SCALES, kTails, kDbias,
                   kAmax>((const uint4*)input, (const uint4*)act_input, rowwise_out, rowwise_scales,
                          colwise_out, colwise_scales, M, K, rowwise_scale_stride,
                          colwise_scale_stride, iters, table, dbias_workspace, amax_ptr);
#endif
}

// Row blocks per CTA.  A longer walk amortizes the per-CTA table copy; a
// shorter one keeps the grid deep enough to avoid a residency tail.
//! Grid size, in CTAs, that the walk length aims for.
constexpr long long kCtaTargetActivation = 16384;
//! Minimum walk for the instantiations that copy the table.
constexpr int kMinWalk = 2;
//! Maximum walk for the derivatives.
constexpr int kMaxWalkDact = 64;

static int pick_walk_length(int row_blocks, int grid_cols, long long target,
                            int min_walk = kMinWalk) {
  int walk = 1;
  while ((row_blocks / walk) % 2 == 0 && (long long)grid_cols * (row_blocks / walk) > target)
    walk *= 2;
  if (walk < min_walk && row_blocks % min_walk == 0) walk = min_walk;
  return walk;
}

// Shared-memory carveout, in percent, for the activations, whose footprint leaves room:
// the rest of the unified array goes to L1.
constexpr int kCarveoutPercentAct = 40;

template <bool IS_DACT, bool IS_ACT, typename TableT, typename OType,
          bool WITH_GEMM_SWIZZLED_SCALES, bool kTails, bool kDbias, bool kAmax>
static void set_carveout() {
  constexpr int kPercent = IS_ACT ? kCarveoutPercentAct : 0;
  constexpr int kDynamicBytes = TileConfig<IS_DACT, IS_ACT, TableT, kDbias>::kDbiasStageBytes;
  if constexpr (kPercent > 0 || kDynamicBytes > 0) {
    // Function attributes are per device, like the tables.
    static std::mutex mutex;
    static std::vector<bool> done;
    int device;
    NVTE_CHECK_CUDA(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (static_cast<size_t>(device) >= done.size()) done.resize(device + 1, false);
    if (!done[device]) {
      const void* kernel =
          (const void*)quantize_mxfp8_kernel<IS_DACT, IS_ACT, TableT, OType,
                                             WITH_GEMM_SWIZZLED_SCALES, kTails, kDbias, kAmax>;
      if constexpr (kPercent > 0) {
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, kPercent));
      }
      if constexpr (kDynamicBytes > 0) {
        // The staging buffer takes the kernel past the 48 KB of shared memory a CTA gets by
        // default.
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             kDynamicBytes));
      }
      done[device] = true;
    }
  }
}

//! 32-row bands the grid covers: whole 64-row tiles of the generic kernel, whose scales it
//! writes.
inline int row_bands(int M) { return (M + 63) / 64 * 2; }
//! 256-column tiles the grid covers.
inline int col_tiles(int K) { return (K + kTileCols - 1) / kTileCols; }

// Row blocks per CTA walk, in powers of two.
template <bool IS_DACT, bool kDbias = false>
int walk_length(int M, int K) {
  const int row_blocks = row_bands(M);
  const int grid_cols = col_tiles(K);
  // The minimum walk amortizes the table copy, but not at the price of leaving
  // SMs without a CTA.
  int min_walk = kMinWalk;
  if ((long long)grid_cols * (row_blocks / min_walk) < 2LL * cuda::sm_count()) min_walk = 1;
  int iters = pick_walk_length(row_blocks, grid_cols, kCtaTargetActivation, min_walk);
  if constexpr (IS_DACT) {
    if (iters > kMaxWalkDact) iters = kMaxWalkDact;
  }
  // With dbias a CTA walks whole 64-row chunks: two bands per chunk.
  if constexpr (kDbias) {
    if (iters % 2 != 0) iters *= 2;
  }
  return iters;
}

template <bool IS_DACT, bool IS_ACT, typename TableT, typename OType,
          bool WITH_GEMM_SWIZZLED_SCALES, bool kDbias>
static void launch(const void* input, const void* act_input, void* rowwise_out,
                   void* rowwise_scales, void* colwise_out, void* colwise_scales, int M, int K,
                   int rowwise_scale_stride, int colwise_scale_stride, const void* table,
                   float* dbias_workspace, float* amax, cudaStream_t stream) {
  using C = TileConfig<IS_DACT, IS_ACT, TableT, kDbias>;
  const int iters = walk_length<IS_DACT, kDbias>(M, K);
  dim3 grid(col_tiles(K), row_bands(M) / iters);
  // kAmax: a separate instantiation, so that the kernels without amax keep their register
  // budget (the derivatives run at the 40-register cap).
  const auto run_amax = [&](auto tails, auto with_amax) {
    constexpr bool kTails = decltype(tails)::value;
    constexpr bool kAmax = decltype(with_amax)::value;
    set_carveout<IS_DACT, IS_ACT, TableT, OType, WITH_GEMM_SWIZZLED_SCALES, kTails, kDbias,
                 kAmax>();
    quantize_mxfp8_kernel<IS_DACT, IS_ACT, TableT, OType, WITH_GEMM_SWIZZLED_SCALES, kTails, kDbias,
                          kAmax><<<grid, C::kThreadsPerCta, C::kDbiasStageBytes, stream>>>(
        (const unsigned*)input, (const unsigned*)act_input, (unsigned char*)rowwise_out,
        (unsigned char*)rowwise_scales, (unsigned char*)colwise_out, (unsigned char*)colwise_scales,
        M, K, rowwise_scale_stride, colwise_scale_stride, iters, table, dbias_workspace, amax);
  };
  const auto run = [&](auto tails) {
    if (amax != nullptr) {
      run_amax(tails, std::true_type{});
    } else {
      run_amax(tails, std::false_type{});
    }
  };
  // Whole 64-row tiles and 256-column tiles leave nothing past M or K. The derivatives take the
  // tail-capable kernel for every shape: with the band body behind its edge branch, the compiler
  // schedules the loads so that they wait less (GB200, 32768x7168 dGeLU: 239 -> 220 us, fewer
  // long-scoreboard stalls, DRAM throughput 73% -> 79%). The forward kernels do not gain from it.
  if (!IS_DACT && M % 64 == 0 && K % kTileCols == 0) {
    run(std::false_type{});
  } else {
    run(std::true_type{});
  }
}

// Any number of rows. Columns in multiples of 16, the shapes the generic kernel takes (its TMA
// store of the FP8 output needs 16-byte rows); this also keeps every input row 16-byte aligned.
// Output offsets within a 32-row band are 32-bit.
inline bool shape_supported(size_t rows, size_t cols) {
  constexpr size_t kMaxRows = std::numeric_limits<int>::max() - 64;
  constexpr size_t kMaxCols = std::numeric_limits<unsigned>::max() / kRowsPerMxBlock;
  return rows > 0 && cols > 0 && rows <= kMaxRows && cols <= kMaxCols && cols % 16 == 0;
}

}  // namespace

//! Whether this kernel implements the request type: GeLU or SiLU, or the derivative of
//! either, the derivative optionally with dbias. It reads the activation from a table, or
//! computes it with util/math.h when there is none.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP&)>
constexpr bool kImplements =
    (!IS_DBIAS || IS_DACT) && (IS_ACT != IS_DACT) && activation_table::kHasTable<ParamOP, OP> &&
    (IS_ACT == activation_table::Table<activation_table::kKindOf<ParamOP, OP>>::kForward);

// Whether this kernel takes a request of a type in kImplements, with BF16 input and E4M3 or
// E5M2 output.  Anything else takes the generic kernel, as does every request with
// NVTE_MXFP8_REGTILE=0.  A noop tensor is ignored, as the generic kernel ignores it for every
// request that computes an activation.
template <bool IS_DACT, bool IS_ACT, bool IS_DBIAS = false>
bool can_use(const Tensor& input, const Tensor* act_input, const Tensor& output, const Tensor& noop,
             bool use_2d_quantization, const activation_table::TableRef& table) {
  if (!getenv<bool>("NVTE_MXFP8_REGTILE", true)) return false;
  const auto [rows, cols] = input.flat_2d_dims();
  // Without a table the forward activation is compute-bound here, and on small tensors too
  // few CTAs hide its latency: GB200, 993x7200 GeLU is 0.93x of the generic kernel, 4160x2048
  // 1.26x. The derivatives stay memory-bound and win at every size.
  constexpr size_t kComputedForwardMinElements = size_t{1} << 23;
  if (table.use == activation_table::kNoTable && !IS_DACT &&
      rows * cols < kComputedForwardMinElements) {
    return false;
  }
  // Vector widths: 16-byte loads of both inputs, 8-byte stores of both outputs
  // and of each row's scales.
  const bool aligned = is_aligned_ptr(input.data.dptr, 16) &&
                       (!IS_DACT || is_aligned_ptr(act_input->data.dptr, 16)) &&
                       is_aligned_ptr(output.data.dptr, 8) &&
                       is_aligned_ptr(output.columnwise_data.dptr, 8) &&
                       is_aligned_ptr(output.scale_inv.dptr, 8) &&
                       is_aligned_ptr(output.columnwise_scale_inv.dptr, 2);
  if (!(transformer_engine::cuda::sm_arch() >= 100 && !use_2d_quantization && output.has_data() &&
        output.has_columnwise_data() && shape_supported(rows, cols) && aligned)) {
    return false;
  }
  constexpr size_t kMaxGridY = 65535;
  const int M = static_cast<int>(rows);
  return static_cast<size_t>(
             row_bands(M) / walk_length<IS_DACT, IS_DBIAS>(M, static_cast<int>(cols))) <= kMaxGridY;
}

//! dbias_workspace: [ceil(rows / 64), cols] FP32 column sums for common::reduce_dbias, or
//! nullptr without dbias.
template <bool IS_DACT, bool IS_ACT, bool IS_DBIAS, typename ParamOP,
          float (*OP)(float, const ParamOP&), typename OType, bool WITH_GEMM_SWIZZLED_SCALES>
void quantize(const Tensor& input, const Tensor* act_input, Tensor* output,
              const activation_table::TableRef& table, float* dbias_workspace,
              cudaStream_t stream) {
  using namespace activation_table;
  // Plain variables: lambdas cannot capture structured bindings before C++20.
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const void* act_ptr = IS_DACT ? act_input->data.dptr : input.data.dptr;
  const auto run = [&](auto table_type) {
    using TableT = typename decltype(table_type)::type;
    launch<IS_DACT, IS_ACT, TableT, OType, WITH_GEMM_SWIZZLED_SCALES, IS_DBIAS>(
        input.data.dptr, act_ptr, output->data.dptr, output->scale_inv.dptr,
        output->columnwise_data.dptr, output->columnwise_scale_inv.dptr, static_cast<int>(rows),
        static_cast<int>(cols), static_cast<int>(output->scale_inv.shape[1]),
        static_cast<int>(output->columnwise_scale_inv.shape[1]), table.ptr, dbias_workspace,
        static_cast<float*>(output->amax.dptr), stream);
  };
  constexpr Kind kKind = kKindOf<ParamOP, OP>;
  if (table.use == kNoTable) {
    run(std::common_type<ComputedTable<kKind>>{});
    NVTE_CHECK_CUDA(cudaGetLastError());
    return;
  }
  if constexpr (kKind == kDsilu) {
    if (table.use == kExactDsiluTable) {
      run(std::common_type<ExactDsiluTable>{});
      NVTE_CHECK_CUDA(cudaGetLastError());
      return;
    }
  }
  run(std::common_type<Table<kKind>>{});
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace regtile
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
