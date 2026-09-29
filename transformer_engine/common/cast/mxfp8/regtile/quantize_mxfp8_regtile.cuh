/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8_regtile.cuh
 *  \brief Register-resident MXFP8 bidimensional quantize fused with GeLU or dGeLU.
 *
 *  Rowwise and colwise MX scales need an amax over the same data in two
 *  directions.  The generic kernel stages each tile through shared memory and
 *  walks it twice.  Here a CTA covers whole 32-row bands, the colwise block
 *  height, so the tile is read once into registers and both directions are
 *  quantized from there; shared memory holds only the cross-warp colwise
 *  partials, the row scales and the dGeLU table.
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
#include "../../../util/cuda_runtime.h"
#include "../../../util/dgelu_table.cuh"
#include "../../../util/math.h"
#include "../../../util/packed_activation.cuh"
#include "../../../util/ptx.cuh"
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
 *  volatile asm the compiler cannot schedule as freely in the dGeLU loop. */
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
__device__ __forceinline__ unsigned amax_to_e8m0_2x(float amax_hi, float amax_lo) {
  constexpr float kMaxNormRcp = Quantized_Limits<fp8e4m3>::max_norm_rcp;
  return ptx::float_to_e8m0_2x(fmaxf(amax_hi * kMaxNormRcp, 0.0f),
                               fmaxf(amax_lo * kMaxNormRcp, 0.0f));
}

/*! \brief GeLU on a BF16 pair, returned as a BF16 pair. */
__device__ __forceinline__ unsigned gelu_bf16x2(unsigned x) {
  return to_bf16x2(activation_2x<Empty, gelu<float, float>>(widen_bf16x2(x), {}));
}

/*! \brief dGeLU of a BF16 pair, by table or by arithmetic.
 *  \tparam USE_TABLE  Which route this (row, word) slot takes; see LutSlot.
 */
template <bool USE_TABLE>
__device__ __forceinline__ ptx::floatx2 dgelu_word(unsigned x,
                                                   const unsigned char* __restrict__ table) {
  if constexpr (USE_TABLE) {
    return dgelu_table::lookup(x, table);
  } else {
    return activation_2x<Empty, dgelu<float, float>>(widen_bf16x2(x), {});
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

/*! \brief Scale two BF16 pairs, each by its own reciprocal scale pair, to four E4M3. */
__device__ __forceinline__ unsigned quantize_e4m3x4(unsigned in01, unsigned in23, unsigned scale01,
                                                    unsigned scale23) {
  ptx::fp8e4m3x4 out;
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

// Table/arithmetic routing per (row, word) slot of one load group: L of the S
// slots take the table, and the arithmetic slots are spread with a Bresenham
// step so their MUFU chains interleave with the table probes.
template <int kSlots, int kTableSlots, int kSlot>
struct LutSlot {
  static constexpr int kArithmeticSlots = kSlots - kTableSlots;
  static constexpr bool value = !((((kSlot + 1) * kArithmeticSlots) % kSlots) < kArithmeticSlots);
};
//! Of the eight slots in dGeLU's two-row load group, how many take the table.
constexpr int kLutSlotsDact = 6;

// Minimum resident CTAs per SM, the second __launch_bounds__ argument, capped
// at what the SM can hold: sm_107 has half the threads per SM of sm_100.
constexpr int kMinBlocksAct = 8;
constexpr int kMinBlocksDact = 6;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1070)
constexpr int kMaxThreadsPerSm = 1024;
#else
constexpr int kMaxThreadsPerSm = 2048;
#endif

template <bool IS_DACT, bool IS_ACT>
struct TileConfig {
  static_assert(IS_ACT != IS_DACT, "Exactly one of GeLU and dGeLU is fused.");
  static constexpr bool kNarrowTile = IS_ACT && !IS_DACT;
  static constexpr int kThreadsPerCta = kNarrowTile ? 128 : 256;
  static constexpr int kWarpsPerCta = kThreadsPerCta / 32;
  static constexpr int kRowsPerWarp = 32 / kWarpsPerCta;
  static constexpr bool kSlotSplit = IS_DACT || kNarrowTile;
  // The rowwise half is quantized before the cross-warp barrier and the row
  // scales are drained inside the colwise barrier interval, saving one barrier
  // per 32-row tile.  A tile too narrow to leave idle threads hands the drain to
  // the threads that just folded the colwise partials.
  static constexpr int kDrainTidBase =
      (kThreadsPerCta >= kTileCols / 2 + kTileCols / 4) ? kTileCols / 2 : 0;
  static constexpr int kRowsInFlight = kSlotSplit ? 2 : kRowsPerWarp;
  static constexpr int kSlots = kRowsInFlight * kWordsPerLane;
  static constexpr int kLutSlots = kLutSlotsDact;
  static constexpr int kRequestedMinBlocks = IS_DACT ? kMinBlocksDact : kMinBlocksAct;
  static constexpr int kMinBlocksPerSm = kRequestedMinBlocks < kMaxThreadsPerSm / kThreadsPerCta
                                             ? kRequestedMinBlocks
                                             : kMaxThreadsPerSm / kThreadsPerCta;
  //! Only dGeLU tabulates; GeLU evaluates its closed form for every word.
  static constexpr bool kNeedsLut = IS_DACT;
  //! A non-tabulating instantiation still declares the array, at a minimal size.
  static constexpr int kLutSharedBytes = kNeedsLut ? dgelu_table::kBytes : 16;
};

// Register-resident tiling: the 32x256 tile stays in registers; only the
// colwise partials go through shared memory.
template <bool IS_DACT, bool IS_ACT, bool WITH_GEMM_SWIZZLED_SCALES>
__device__ __forceinline__ void quantize_regtile(
    const uint4* __restrict__ input, const uint4* __restrict__ act_input,
    unsigned char* __restrict__ rowwise_out, unsigned char* __restrict__ rowwise_scales,
    unsigned char* __restrict__ colwise_out, unsigned char* __restrict__ colwise_scales, int K,
    int rowwise_scale_stride, int colwise_scale_stride, int iters) {
  using C = TileConfig<IS_DACT, IS_ACT>;
  constexpr int kWarpsPerCta = C::kWarpsPerCta, kRowsPerWarp = C::kRowsPerWarp,
                kRowsInFlight = C::kRowsInFlight;

  __shared__ __align__(16) unsigned col_amax_partials[kWarpsPerCta][kTileCols / 2];
  __shared__ __align__(16) unsigned col_scale_rcp_smem[kTileCols / 2];
  __shared__ __align__(8) unsigned char row_scale_bytes[kRowsPerMxBlock * kMxGroupsPerRow];
  __shared__ __align__(16) unsigned char dgelu_table_smem[C::kLutSharedBytes];
  __shared__ __align__(8) uint64_t dgelu_table_barrier;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col0 = blockIdx.x * kTileCols;
  const int vecs_per_row = K >> 3;  // uint4 (8 bf16 values) per row

  // The table copy overlaps the first input loads; see wait_table below.
  if constexpr (C::kNeedsLut)
    dgelu_table::load_table_async(dgelu_table_smem, &dgelu_table_barrier, tid);
  const unsigned char* __restrict__ dgelu_table = dgelu_table_smem;

#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    // Row blocks are taken grid-strided, so the CTAs on the same step cover one
    // contiguous row band.
    const int row0 = (it * gridDim.y + blockIdx.y) * kRowsPerMxBlock;
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
#pragma unroll
      for (int t = 0; t < kRowsInFlight; ++t)
#pragma unroll
        for (int n = 0; n < kVecLoadsPerRow; ++n)
          x_vec[t][n] = act_input_ptr[(size_t)(h * kRowsInFlight + t) * vecs_per_row + n];
      if constexpr (IS_DACT) {
#pragma unroll
        for (int t = 0; t < kRowsInFlight; ++t)
#pragma unroll
          for (int n = 0; n < kVecLoadsPerRow; ++n)
            grad_vec[t][n] = input_ptr[(size_t)(h * kRowsInFlight + t) * vecs_per_row + n];
      }
      if constexpr (C::kNeedsLut) {
        if (it == 0 && h == 0) dgelu_table::wait_table(&dgelu_table_barrier);
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
              tile_words[tile_word_base + m] = gelu_bf16x2(x_words[m]);
            }
          }
        }
      } else {
// Each (row, word) slot takes the table or the arithmetic body per LutSlot.
#define MXFP8_DACT_WORD(T, M)                                                                   \
  {                                                                                             \
    constexpr bool kUseLut = LutSlot<C::kSlots, C::kLutSlots, (T) * 4 + (M) + 2>::value;        \
    const ptx::floatx2 product =                                                                \
        ptx::mul_2x(widen_bf16x2(grad_words[M]), dgelu_word<kUseLut>(x_words[M], dgelu_table)); \
    tile_words[(h * kRowsInFlight + (T)) * kWordsPerLane + (M)] = to_bf16x2(product);           \
  }
#define MXFP8_DACT_ROW(T)                                          \
  {                                                                \
    const unsigned* x_words = (const unsigned*)&x_vec[T][0];       \
    const unsigned* grad_words = (const unsigned*)&grad_vec[T][0]; \
    MXFP8_DACT_WORD(T, 0)                                          \
    MXFP8_DACT_WORD(T, 1)                                          \
    MXFP8_DACT_WORD(T, 2)                                          \
    MXFP8_DACT_WORD(T, 3)                                          \
  }
        MXFP8_DACT_ROW(0)
        if constexpr (kRowsInFlight > 1) MXFP8_DACT_ROW(1)
        if constexpr (kRowsInFlight > 2) {
          MXFP8_DACT_ROW(2) MXFP8_DACT_ROW(3)
        }
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
      const unsigned scale_byte = amax_to_e8m0_2x(0.f, __uint_as_float(amax & 0x7fff0000u));
      if ((lane & (kLanesPerMxGroup - 1)) == 0)
        row_scale_bytes[(warp * kRowsPerWarp + j) * kMxGroupsPerRow + (lane / kLanesPerMxGroup)] =
            (unsigned char)scale_byte;
      return e8m0_to_bf16x2_reciprocal(scale_byte);
    };
    // On the 128-thread GeLU tile, two rows share one butterfly and conversion.
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
      const unsigned scale_pair = amax_to_e8m0_2x(amax_2x.y, amax_2x.x);
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
    constexpr bool kPairedRowScales =
        IS_ACT && !IS_DACT && C::kThreadsPerCta == 128 && kRowsPerWarp == 8;
    if constexpr (kPairedRowScales) {
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
              quantize_e4m3x4(row_a[2 * m], row_a[2 * m + 1], row_scale_rcp0, row_scale_rcp0);
        store_quantized(row_out_ptr + (unsigned)j * (unsigned)K, row_out);
#pragma unroll
        for (int m = 0; m < kWordsPerLane / 2; ++m)
          row_out[m] =
              quantize_e4m3x4(row_b[2 * m], row_b[2 * m + 1], row_scale_rcp1, row_scale_rcp1);
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
          row_out[m] = quantize_e4m3x4(v[2 * m], v[2 * m + 1], row_scale_rcp, row_scale_rcp);
        store_quantized(row_out_ptr + (unsigned)j * (unsigned)K, row_out);
      }
    }

    publish_col_amax_partials();
    __syncthreads();
    if (tid < kTileCols / 2) {
      unsigned col_amax = col_amax_partials[0][tid];
#pragma unroll
      for (int w = 1; w < kWarpsPerCta; ++w)
        col_amax = abs_max_bf16x2(col_amax, col_amax_partials[w][tid]);
      col_amax &= 0x7fff7fffu;
      const ptx::floatx2 amax_2x = widen_bf16x2(col_amax);
      const unsigned scale_pair = amax_to_e8m0_2x(amax_2x.y, amax_2x.x);
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        // The colwise swizzle is the rowwise one with rows and columns swapped,
        // so the two columns of this pair land in different 16-byte rows.  The
        // grid covers M exactly: gridDim.y * iters row blocks of 32.
        const size_t row_block_tiles =
            DIVUP(static_cast<size_t>(gridDim.y) * iters * kRowsPerMxBlock, size_t{128});
        const size_t col = col0 + 2 * tid;
        colwise_scales[swizzle::gemm_swizzled_scale_idx(col, row0 / 32, row_block_tiles)] =
            static_cast<unsigned char>(scale_pair);
        colwise_scales[swizzle::gemm_swizzled_scale_idx(col + 1, row0 / 32, row_block_tiles)] =
            static_cast<unsigned char>(scale_pair >> 8);
      } else {
        *(unsigned short*)(colwise_scales + (size_t)(row0 / 32) * colwise_scale_stride + col0 +
                           2 * tid) = (unsigned short)scale_pair;
      }
      const ptx::bf16x2 scale_rcp = ptx::exp2f_rcp_2x_per_lane(__byte_perm(scale_pair, 0, 0x4140));
      col_scale_rcp_smem[tid] = reinterpret_cast<const unsigned&>(scale_rcp);
    }
    auto drain_row_scales = [&]() {
      constexpr int drain_tid_base = C::kDrainTidBase;
      // A row's eight group scales are eight contiguous, 8-byte-aligned bytes of
      // rowwise_scales, so one warp drains a 32-row tile with one STG.64 per lane.
      // In the GEMM-swizzled layout they are two 4-byte runs in adjacent tiles.
      if (tid >= drain_tid_base && tid < drain_tid_base + kRowsPerMxBlock) {
        const int r = tid - drain_tid_base;
        if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
          constexpr size_t kTileBytes =
              swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_X * swizzle::GEMM_SWIZZLED_SCALE_TILE_DIM_Y;
          const size_t idx = swizzle::gemm_swizzled_scale_idx(row0 + r, col0 >> 5, K / 128);
          const uint2 bytes = *(const uint2*)(row_scale_bytes + r * kMxGroupsPerRow);
          *(unsigned*)(rowwise_scales + idx) = bytes.x;
          *(unsigned*)(rowwise_scales + idx + kTileBytes) = bytes.y;
        } else {
          *(uint2*)(rowwise_scales + (size_t)(row0 + r) * rowwise_scale_stride + (col0 >> 5)) =
              *(const uint2*)(row_scale_bytes + r * kMxGroupsPerRow);
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
          col_out[m] = quantize_e4m3x4(v[2 * m], v[2 * m + 1], col_scale_rcp[2 * m],
                                       col_scale_rcp[2 * m + 1]);
        store_quantized(col_out_ptr + (unsigned)j * (unsigned)K, col_out);
      }
    }
  }
}

template <bool IS_DACT, bool IS_ACT, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(TileConfig<IS_DACT, IS_ACT>::kThreadsPerCta,
                                  TileConfig<IS_DACT, IS_ACT>::kMinBlocksPerSm)
    quantize_mxfp8_kernel(const unsigned* __restrict__ input,
                          const unsigned* __restrict__ act_input,
                          unsigned char* __restrict__ rowwise_out,
                          unsigned char* __restrict__ rowwise_scales,
                          unsigned char* __restrict__ colwise_out,
                          unsigned char* __restrict__ colwise_scales, int K,
                          int rowwise_scale_stride, int colwise_scale_stride, int iters) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  quantize_regtile<IS_DACT, IS_ACT, WITH_GEMM_SWIZZLED_SCALES>(
      (const uint4*)input, (const uint4*)act_input, rowwise_out, rowwise_scales, colwise_out,
      colwise_scales, K, rowwise_scale_stride, colwise_scale_stride, iters);
#endif
}

// Row blocks per CTA.  A longer walk amortizes the per-CTA dGeLU table copy; a
// shorter one keeps the grid deep enough to avoid a residency tail.
//! Grid size, in CTAs, that the walk length aims for.
constexpr long long kCtaTargetActivation = 16384;
//! Minimum walk for the instantiations that copy the table.
constexpr int kMinWalk = 2;
//! Maximum walk for dGeLU.
constexpr int kMaxWalkDact = 64;

static int pick_walk_length(int row_blocks, int grid_cols, long long target,
                            int min_walk = kMinWalk) {
  int walk = 1;
  while ((row_blocks / walk) % 2 == 0 && (long long)grid_cols * (row_blocks / walk) > target)
    walk *= 2;
  if (walk < min_walk && row_blocks % min_walk == 0) walk = min_walk;
  return walk;
}

// Shared-memory carveout, in percent, for GeLU, whose footprint leaves room:
// the rest of the unified array goes to L1.
constexpr int kCarveoutPercentAct = 40;

template <bool IS_DACT, bool IS_ACT, bool WITH_GEMM_SWIZZLED_SCALES>
static void set_carveout() {
  constexpr int kPercent = IS_ACT ? kCarveoutPercentAct : 0;
  if constexpr (kPercent > 0) {
    // Function attributes are per device, like the dGeLU table.
    static std::mutex mutex;
    static std::vector<bool> done;
    int device;
    NVTE_CHECK_CUDA(cudaGetDevice(&device));
    std::lock_guard<std::mutex> lock(mutex);
    if (static_cast<size_t>(device) >= done.size()) done.resize(device + 1, false);
    if (!done[device]) {
      NVTE_CHECK_CUDA(cudaFuncSetAttribute(
          (const void*)quantize_mxfp8_kernel<IS_DACT, IS_ACT, WITH_GEMM_SWIZZLED_SCALES>,
          cudaFuncAttributePreferredSharedMemoryCarveout, kPercent));
      done[device] = true;
    }
  }
}

// Row blocks per CTA walk: M / 32, walked in powers of two.  The minimum walk
// exists for the table copy, so only the table-copying modes take it.
template <bool IS_DACT, bool IS_ACT>
int walk_length(int M, int K) {
  using C = TileConfig<IS_DACT, IS_ACT>;
  const int row_blocks = M / kRowsPerMxBlock;
  const int grid_cols = K / kTileCols;
  // The minimum walk amortizes the table copy, but not at the price of leaving
  // SMs without a CTA.
  int min_walk = C::kNeedsLut ? kMinWalk : 1;
  if ((long long)grid_cols * (row_blocks / min_walk) < 2LL * cuda::sm_count()) min_walk = 1;
  int iters = pick_walk_length(row_blocks, grid_cols, kCtaTargetActivation, min_walk);
  if constexpr (IS_DACT) {
    if (iters > kMaxWalkDact) iters = kMaxWalkDact;
  }
  return iters;
}

template <bool IS_DACT, bool IS_ACT, bool WITH_GEMM_SWIZZLED_SCALES>
static void launch(const void* input, const void* act_input, void* rowwise_out,
                   void* rowwise_scales, void* colwise_out, void* colwise_scales, int M, int K,
                   int rowwise_scale_stride, int colwise_scale_stride, cudaStream_t stream) {
  using C = TileConfig<IS_DACT, IS_ACT>;
  set_carveout<IS_DACT, IS_ACT, WITH_GEMM_SWIZZLED_SCALES>();
  const int iters = walk_length<IS_DACT, IS_ACT>(M, K);
  dim3 grid(K / kTileCols, M / kRowsPerMxBlock / iters);
  quantize_mxfp8_kernel<IS_DACT, IS_ACT, WITH_GEMM_SWIZZLED_SCALES>
      <<<grid, C::kThreadsPerCta, 0, stream>>>(
          (const unsigned*)input, (const unsigned*)act_input, (unsigned char*)rowwise_out,
          (unsigned char*)rowwise_scales, (unsigned char*)colwise_out,
          (unsigned char*)colwise_scales, K, rowwise_scale_stride, colwise_scale_stride, iters);
}

// The grid is derived by exact division, so columns must tile into 256.  Rows
// must be a multiple of the generic kernel's 64-row activation tile: on a
// 32-row tail that kernel writes zero scales into the padding past the last
// row, which this one does not.
constexpr size_t kGenericActivationTileRows = 64;

inline bool shape_supported(size_t rows, size_t cols) {
  constexpr size_t kMaxDim = std::numeric_limits<int>::max();
  return rows > 0 && cols > 0 && rows <= kMaxDim && cols <= kMaxDim &&
         rows % kGenericActivationTileRows == 0 && cols % kTileCols == 0;
}

}  // namespace

// Whether this kernel implements the request.  Anything else takes the generic
// kernel.
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP&)>
bool can_use(const Tensor& input, const Tensor* act_input, const Tensor& output, const Tensor& noop,
             bool use_2d_quantization) {
  constexpr bool op_supported =
      !IS_DBIAS && ((IS_ACT && !IS_DACT && packed_activation::is_gelu<ParamOP, OP>) ||
                    (IS_DACT && !IS_ACT && packed_activation::is_dgelu<ParamOP, OP>));
  if constexpr (!op_supported) {
    return false;
  } else {
    const auto [rows, cols] = input.flat_2d_dims();
    // Vector widths: 16-byte loads of both inputs, 8-byte stores of both outputs
    // and of each row's scales.
    const bool aligned = is_aligned_ptr(input.data.dptr, 16) &&
                         (!IS_DACT || is_aligned_ptr(act_input->data.dptr, 16)) &&
                         is_aligned_ptr(output.data.dptr, 8) &&
                         is_aligned_ptr(output.columnwise_data.dptr, 8) &&
                         is_aligned_ptr(output.scale_inv.dptr, 8) &&
                         is_aligned_ptr(output.columnwise_scale_inv.dptr, 2);
    if (!(transformer_engine::cuda::sm_arch() >= 100 && !use_2d_quantization && output.has_data() &&
          output.has_columnwise_data() && input.dtype() == DType::kBFloat16 &&
          output.dtype() == DType::kFloat8E4M3 && output.amax.dptr == nullptr &&
          noop.data.dptr == nullptr && shape_supported(rows, cols) && aligned)) {
      return false;
    }
    constexpr size_t kMaxGridY = 65535;
    const int M = static_cast<int>(rows);
    return static_cast<size_t>(M / kRowsPerMxBlock /
                               walk_length<IS_DACT, IS_ACT>(M, static_cast<int>(cols))) <=
           kMaxGridY;
  }
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT>
void quantize(const Tensor& input, const Tensor* act_input, Tensor* output,
              bool use_2d_quantization, cudaStream_t stream) {
  static_assert(!IS_DBIAS && IS_ACT != IS_DACT, "Only GeLU and dGeLU without dbias.");
  using C = TileConfig<IS_DACT, IS_ACT>;
  NVTE_CHECK(!use_2d_quantization,
             "Register-resident MXFP8 quantize does not implement 2D block scaling.");
  const auto [rows, cols] = input.flat_2d_dims();
  NVTE_CHECK(shape_supported(rows, cols),
             "Unsupported shape for register-resident MXFP8 quantize.");
  if constexpr (C::kNeedsLut) {
    dgelu_table::ensure_table(stream);
  }
  const void* act_ptr = IS_DACT ? act_input->data.dptr : input.data.dptr;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      output->with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,
      launch<IS_DACT, IS_ACT, WITH_GEMM_SWIZZLED_SCALES>(
          input.data.dptr, act_ptr, output->data.dptr, output->scale_inv.dptr,
          output->columnwise_data.dptr, output->columnwise_scale_inv.dptr, static_cast<int>(rows),
          static_cast<int>(cols), static_cast<int>(output->scale_inv.shape[1]),
          static_cast<int>(output->columnwise_scale_inv.shape[1]), stream););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace regtile
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
