/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_rowwise.cu
 *  \brief Register-resident rowwise MXFP8 quantization kernel.
 */

#include <cuda_runtime.h>

#include "../../../common.h"
#include "../../../util/ptx.cuh"
#include "../../../util/ptx_arch_spec.cuh"
#include "../../../utils.cuh"
#include "../swizzle.cuh"  // gemm_swizzled_scale_idx (GEMM scale layout)
#include "cast_rowwise.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

namespace ptx = transformer_engine::ptx;

// GEMM-swizzled scale index; see ../swizzle.cuh.  Not to be confused with
// specialized/swizzle.cuh, which is the TMA input bank-conflict swizzle.
using transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;

// This kernel casts BF16 to rowwise-scaled MXFP8: every run of 32 consecutive
// elements within a row forms one MX block that shares a single E8M0 scale,
// chosen so the block's largest magnitude lands at the top of the FP8E4M3
// range.
//
// Unlike the TMA-based kernel in mxfp8/specialized, this one keeps its tile
// entirely in registers.  There is no shared memory, no barrier, and no
// two-dimensional tiling: for a row-major tensor whose scale array is also
// contiguous, MX blocks never straddle a row boundary, so the whole tensor is
// just a flat sequence of M*(K/32) independent blocks.  That reduces the
// kernel to a pure streaming problem, and what is left to tune is how the
// input and output streams share L2.

// Elements in one MX block, all sharing a single E8M0 scale.
constexpr int32_t kBlockElems = 32;

// Bytes in one GEMM-swizzled scale tile (128 rows x 4 scale columns).
constexpr int32_t kSwzTileSize = 512;

// Two lanes cooperate on each MX block.  A lane's half of a block is 16 BF16
// values = 32 bytes = one 256-bit load, the widest the ISA offers; splitting
// the block any further would waste load width, and any less would exceed it.
constexpr int32_t kLanesPerBlock = 2;
constexpr int32_t kElemsPerLane = kBlockElems / kLanesPerBlock;

// Both tensors are addressed as 32-bit words: BF16 packs 2 elements per word,
// FP8 packs 4.  All the packed-math PTX below operates on those words.
constexpr int32_t kInElemsPerWord = sizeof(uint32_t) / sizeof(bf16);
// Every MXFP8 output type is a single byte, so a 32-bit word holds four.
constexpr int32_t kOutElemsPerWord = 4;

constexpr int32_t kInWordsPerLane = kElemsPerLane / kInElemsPerWord;    // 8 -> 256-bit load
constexpr int32_t kOutWordsPerLane = kElemsPerLane / kOutElemsPerWord;  // 4 -> 128-bit store
constexpr int32_t kInWordsPerBlock = kBlockElems / kInElemsPerWord;     // 16
constexpr int32_t kOutWordsPerBlock = kBlockElems / kOutElemsPerWord;   // 8

// MX blocks a single warp covers in one pass over its registers.
constexpr int32_t kBlocksPerWarp = THREADS_PER_WARP / kLanesPerBlock;  // 16

// The block-wide maximum is formed with a single shuffle that swaps a lane
// with its odd/even partner, which only covers a two-lane group.
static_assert(kLanesPerBlock == 2, "A wider lane group would need a multi-step reduction.");

/*! \brief Launch parameters for one tensor-size regime.
 *
 * The kernel is bandwidth-bound, so the best configuration tracks how the
 * working set compares with L2 rather than the shape itself.  These came from
 * an autotuning sweep; see kTierMaxBytes below for the one threshold that has
 * since been re-measured.
 */
struct LaunchConfig {
  //! CTA width.  Trades occupancy against per-CTA scheduling overhead.
  int32_t threads_per_cta;
  //! MX blocks each lane pair handles per launch.  Raising this unrolls the
  //! body, giving more independent loads in flight at the cost of registers.
  int32_t blocks_per_lane;
  //! Percentage of CTAs that let their input settle in L2 normally; the
  //! remainder tag their loads evict_first so the data streams past without
  //! displacing anything.  0 streams the entire input.
  //!
  //! Streaming everything is right once the input dwarfs L2, since nothing
  //! would survive to be reused anyway.  When the input is only a few times
  //! L2, holding part of it back leaves capacity for the output write-back
  //! instead of thrashing on input lines.
  int32_t l2_cached_cta_percent;
};

// Output bytes (one FP8 byte per element, i.e. M*K) separating the regimes.
//
// The first threshold is 12 MiB rather than the 24 MiB the original sweep
// picked: the single-block-per-lane configuration of tier 0 stops paying off
// well before 24 MiB, losing to tier 1 and to the staged kernel alike over the
// upper half of that range.  Re-measuring put the crossover here instead.
constexpr int64_t kTierMaxBytes[] = {12ll << 20, 48ll << 20, 96ll << 20};

constexpr LaunchConfig kTierConfigs[] = {
    {/*threads_per_cta=*/256, /*blocks_per_lane=*/1, /*l2_cached_cta_percent=*/0},
    {/*threads_per_cta=*/256, /*blocks_per_lane=*/2, /*l2_cached_cta_percent=*/0},
    {/*threads_per_cta=*/128, /*blocks_per_lane=*/2, /*l2_cached_cta_percent=*/40},
    {/*threads_per_cta=*/256, /*blocks_per_lane=*/2, /*l2_cached_cta_percent=*/40},
};
constexpr int32_t kNumTiers = sizeof(kTierConfigs) / sizeof(kTierConfigs[0]);
static_assert(kNumTiers == sizeof(kTierMaxBytes) / sizeof(kTierMaxBytes[0]) + 1,
              "Each size threshold must separate two tiers.");

namespace {

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

/*! \brief Reduce eight BF16 pairs to the largest magnitude among them. */
__device__ __forceinline__ ptx::bf16x2 block_half_amax(const uint32_t (&words)[kInWordsPerLane]) {
  const ptx::bf16x2 *pairs = reinterpret_cast<const ptx::bf16x2 *>(words);

  // Balanced tree: depth 3 instead of the 7 of a serial chain, so the
  // independent maxima issue back to back.
  ptx::bf16x2 level[kInWordsPerLane / 2];
#pragma unroll
  for (int32_t i = 0; i < kInWordsPerLane / 2; ++i) {
    ptx::abs_max_2x(level[i], pairs[2 * i], pairs[2 * i + 1]);
  }
#pragma unroll
  for (int32_t i = 0; i < kInWordsPerLane / 4; ++i) {
    ptx::abs_max_2x(level[i], level[i], level[i + kInWordsPerLane / 4]);
  }
  ptx::bf16x2 result;
  ptx::abs_max_2x(result, level[0], level[1]);
  return result;
}

/*! \brief Widen a BF16 pair's larger magnitude to FP32.
 *
 * `max.xorsign.abs` keeps the magnitude of the larger operand but sets the
 * result sign to the XOR of the input signs, so an accumulator built from it
 * can come out negative.  Only the magnitude means anything for a scale, and
 * feeding a negative value to the unsigned E8M0 conversion would saturate it
 * to zero, so the sign is cleared here.
 */
__device__ __forceinline__ float pair_amax_to_float(ptx::bf16x2 pair) {
  const uint32_t bits = reinterpret_cast<const uint32_t &>(pair);
  // Fold the two halves against each other, then keep the low BF16 sans sign.
  const uint32_t folded = __byte_perm(bits, bits, 0x1032);
  ptx::bf16x2 a, b;
  reinterpret_cast<uint32_t &>(a) = bits;
  reinterpret_cast<uint32_t &>(b) = folded;
  ptx::bf16x2 wide;
  ptx::abs_max_2x(wide, a, b);
  const uint32_t magnitude = reinterpret_cast<const uint32_t &>(wide) & 0x7FFFu;
  return __int_as_float(magnitude << 16);
}

/*! \brief Blank a lane's input registers so a dead block reduces harmlessly. */
__device__ __forceinline__ void zero_words(uint32_t (&words)[kInWordsPerLane]) {
#pragma unroll
  for (int32_t i = 0; i < kInWordsPerLane; ++i) {
    words[i] = 0;
  }
}

/*! \brief Scale and convert one lane's 16 BF16 values into 16 FP8E4M3 bytes. */
template <typename OType>
__device__ __forceinline__ void scale_and_convert(const uint32_t (&in)[kInWordsPerLane],
                                                  ptx::bf16x2 scale_reciprocal,
                                                  uint32_t (&out)[kOutWordsPerLane]) {
#pragma unroll
  for (int32_t i = 0; i < kOutWordsPerLane; ++i) {
    ptx::mul_cvt_4x(reinterpret_cast<ptx::FPx4<OType> &>(out[i]),
                    reinterpret_cast<const ptx::bf16x4 &>(in[2 * i]), scale_reciprocal);
  }
}

#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

}  // namespace

/*! \brief Quantize a contiguous run of MX blocks, two lanes per block.
 *
 * \tparam PARTIAL_L2_CACHING  When false every CTA streams its input, which
 *                             the ISA expresses as a static load modifier and
 *                             so costs no policy register.  When true the
 *                             decision varies per CTA and needs a runtime
 *                             policy token.  See LaunchConfig.
 * \tparam CHECK_BOUNDS        When true the grid is rounded up rather than
 *                             truncated, and every access is predicated on the
 *                             block existing.  Only the shapes whose block
 *                             count does not divide evenly among CTAs pay for
 *                             this; the rest instantiate it false and get the
 *                             same code as before.
 *
 * \param[in]  input                BF16 input, viewed as 32-bit words.
 * \param[out] output               FP8E4M3 output, viewed as 32-bit words.
 * \param[out] scales               One E8M0 byte per MX block.
 * \param[in]  first_streaming_cta  CTAs at or above this index stream their
 *                                  input; earlier ones cache normally.  Only
 *                                  read when PARTIAL_L2_CACHING is true.
 * \param[in]  num_blocks           Total MX blocks in the tensor.  Only read
 *                                  when CHECK_BOUNDS is true.
 * \param[in]  col_spans            Column spans per 128-row band, i.e. the
 *                                  factor of the grid the CTA index is
 *                                  decomposed by.  Computed by the launcher,
 *                                  which also sizes the grid with it, so the
 *                                  two cannot disagree.  Only read when
 *                                  SWIZZLED_SCALES is true.
 */
template <typename OType, int32_t THREADS_PER_CTA, int32_t BLOCKS_PER_LANE, bool PARTIAL_L2_CACHING,
          bool CHECK_BOUNDS, bool SWIZZLED_SCALES>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    quantize_contiguous_kernel(const uint32_t *__restrict__ input, uint32_t *__restrict__ output,
                               e8m0_t *__restrict__ scales, uint32_t first_streaming_cta,
                               int64_t num_blocks, int32_t blocks_per_row, int32_t num_tiles_X,
                               int32_t col_spans, int32_t rows) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int32_t kWarpsPerCta = THREADS_PER_CTA / THREADS_PER_WARP;
  constexpr int32_t kBlocksPerWarpPass = kBlocksPerWarp * BLOCKS_PER_LANE;

  const int32_t lane = threadIdx.x % THREADS_PER_WARP;
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerCta + threadIdx.x / THREADS_PER_WARP;
  const int64_t first_block = warp_id * kBlocksPerWarpPass;

  // Lanes pair up as (even, odd); the even lane of each pair owns the scale.
  const int32_t block_in_warp = lane / kLanesPerBlock;
  const bool owns_scale = (lane % kLanesPerBlock) == 0;

  // Swizzled scales need rows 32 apart to form a contiguous run, but putting that
  // spread inside a warp costs more than it saves: the warp then reads four
  // regions far apart and loses DRAM locality, moving the same bytes at a much
  // lower rate.  So the spread lives at CTA level.  Each warp keeps one
  // contiguous run of a single row, exactly as the packed shape does, and the
  // four sub-rows are covered by four different warps whose scale bytes meet in
  // shared memory.  The payload never does, which is what still separates this
  // kernel from the staged one.
  constexpr int32_t kSubRows = 4;                      // rows 32 apart, = 128/32
  constexpr int32_t kQuads = kWarpsPerCta / kSubRows;  // row offsets per CTA
  const int32_t warp_in_cta = threadIdx.x / THREADS_PER_WARP;
  const int32_t sub_row = warp_in_cta % kSubRows;  // s
  const int32_t quad = warp_in_cta / kSubRows;     // q
  const int32_t half_in_block = lane % kLanesPerBlock;

  int64_t swz_band = 0;
  int32_t swz_a = 0, swz_j0 = 0;
  if constexpr (SWIZZLED_SCALES) {
    // Column groups vary fastest, so concurrent CTAs walk adjacent columns of the
    // same rows.  col_spans comes from the launcher rather than being recomputed:
    // it must match the factor the grid was sized with exactly, or the CTA index
    // decomposes into a different (band, row slot, column span) than the grid was
    // built for.
    const int64_t r_slots = 32 / kQuads;
    const int64_t b = blockIdx.x;
    swz_j0 = static_cast<int32_t>(b % col_spans) * (kBlocksPerWarp * BLOCKS_PER_LANE);
    const int64_t t = b / col_spans;
    swz_a = static_cast<int32_t>(t % r_slots) * kQuads;
    swz_band = t / r_slots;
  }
  const int32_t swz_row_v = static_cast<int32_t>(swz_band) * 128 + 32 * sub_row + swz_a + quad;
  auto swz_row = [&](int32_t) -> int32_t { return swz_row_v; };
  auto swz_block = [&](int32_t u) -> int32_t {
    return swz_j0 + kBlocksPerWarp * u + block_in_warp;
  };

  // One 16-byte run per (pass, quad, tile-in-span); 4 words each.  256 bytes at
  // the widest configuration, flushed after a single barrier.
  __shared__ uint32_t swz_scratch[SWIZZLED_SCALES ? BLOCKS_PER_LANE * kQuads * 4 * kSubRows : 1];

  // Output is marked evict_last so its lines linger long enough to coalesce on
  // write-back.
  const uint64_t output_policy = ptx::create_l2_policy_evict_last();

  // Should be inlined by NVCC so there is never an actual function call
  auto is_live = [&](int64_t group_base) -> bool {
    if constexpr (CHECK_BOUNDS) {
      return group_base + block_in_warp < num_blocks;
    } else {
      return true;
    }
  };

  // The swizzled shape is indexed by row, so its bound is the row count; the
  // grid is rounded up to whole 128-row bands because the GEMM scale array is
  // padded to that anyway.
  auto swz_live = [&](int32_t u) -> bool {
    if constexpr (CHECK_BOUNDS) {
      // Both axes can be ragged: rows are padded to whole 128-row bands, and the
      // column span (16 * blocks_per_lane) need not divide blocks_per_row.
      return swz_row(u) < rows && swz_block(u) < blocks_per_row;
    } else {
      return true;
    }
  };
  auto in_offset = [&](int32_t u, int64_t group_base) -> int64_t {
    if constexpr (SWIZZLED_SCALES) {
      return static_cast<int64_t>(swz_row(u)) * blocks_per_row * kInWordsPerBlock +
             static_cast<int64_t>(swz_block(u)) * kInWordsPerBlock +
             half_in_block * kInWordsPerLane;
    } else {
      return group_base * kInWordsPerBlock + lane * kInWordsPerLane;
    }
  };
  auto out_offset = [&](int32_t u, int64_t group_base) -> int64_t {
    if constexpr (SWIZZLED_SCALES) {
      return static_cast<int64_t>(swz_row(u)) * blocks_per_row * kOutWordsPerBlock +
             static_cast<int64_t>(swz_block(u)) * kOutWordsPerBlock +
             half_in_block * kOutWordsPerLane;
    } else {
      return group_base * kOutWordsPerBlock + lane * kOutWordsPerLane;
    }
  };
  auto live = [&](int32_t u, int64_t group_base) -> bool {
    if constexpr (SWIZZLED_SCALES) {
      return swz_live(u);
    } else {
      return is_live(group_base);
    }
  };

  uint32_t in_words[BLOCKS_PER_LANE][kInWordsPerLane];
  if constexpr (PARTIAL_L2_CACHING) {
    const uint64_t input_policy =
        ptx::create_l2_policy_evict_first(blockIdx.x >= first_streaming_cta ? 1.0f : 0.0f);
#pragma unroll
    for (int32_t u = 0; u < BLOCKS_PER_LANE; ++u) {
      const int64_t group_base = first_block + static_cast<int64_t>(u) * kBlocksPerWarp;
      if (!live(u, group_base)) {
        zero_words(in_words[u]);
        continue;
      }
      ptx::ld_global_nc_b32x8(in_words[u], input + in_offset(u, group_base), input_policy);
    }
  } else {
#pragma unroll
    for (int32_t u = 0; u < BLOCKS_PER_LANE; ++u) {
      const int64_t group_base = first_block + static_cast<int64_t>(u) * kBlocksPerWarp;
      if (!live(u, group_base)) {
        zero_words(in_words[u]);
        continue;
      }
      ptx::ld_global_nc_evict_first_b32x8(in_words[u], input + in_offset(u, group_base));
    }
  }

#pragma unroll
  for (int32_t u = 0; u < BLOCKS_PER_LANE; ++u) {
    const int64_t group_base = first_block + static_cast<int64_t>(u) * kBlocksPerWarp;

    // Each lane reduces its own half, then swaps with its partner so both
    // arrive at the block-wide maximum.
    ptx::bf16x2 half_amax = block_half_amax(in_words[u]);
    ptx::bf16x2 partner;
    reinterpret_cast<uint32_t &>(partner) =
        __shfl_xor_sync(0xFFFFFFFFu, reinterpret_cast<const uint32_t &>(half_amax), /*laneMask=*/1);
    ptx::bf16x2 block_amax;
    ptx::abs_max_2x(block_amax, half_amax, partner);

    const e8m0_t biased_exponent =
        ptx::float_to_e8m0(pair_amax_to_float(block_amax) * Quantized_Limits<OType>::max_norm_rcp);
    const bool blk_live = live(u, group_base);
    if constexpr (SWIZZLED_SCALES) {
      // Collapse the four scale columns adjacent in the GEMM layout into one word:
      // lanes 8t,8t+2,8t+4,8t+6 hold j%4 = 0..3, so three shuffles suffice.  The
      // other axis of the run lives in sibling warps, so the word goes to scratch
      // and the flush below turns four of them into one 16-byte store.
      const uint32_t my_byte = static_cast<uint32_t>(biased_exponent);
      const uint32_t c1 = __shfl_down_sync(0xFFFFFFFFu, my_byte, kLanesPerBlock);
      const uint32_t c2 = __shfl_down_sync(0xFFFFFFFFu, my_byte, 2 * kLanesPerBlock);
      const uint32_t c3 = __shfl_down_sync(0xFFFFFFFFu, my_byte, 3 * kLanesPerBlock);
      if ((lane % (4 * kLanesPerBlock)) == 0) {
        const int32_t t = block_in_warp / 4;
        swz_scratch[((u * kQuads + quad) * 4 + t) * kSubRows + sub_row] =
            my_byte | (c1 << 8) | (c2 << 16) | (c3 << 24);
      }
    } else if (owns_scale && blk_live) {
      scales[group_base + block_in_warp] = biased_exponent;
    }

    uint32_t out_words[kOutWordsPerLane];
    scale_and_convert<OType>(in_words[u], ptx::exp2f_rcp_2x(biased_exponent), out_words);
    if (blk_live) {
      ptx::st_global_b32x4(output + out_offset(u, group_base), out_words, output_policy);
    }
  }

  if constexpr (SWIZZLED_SCALES) {
    // One barrier for the whole kernel: every pass has already deposited its words.
    __syncthreads();
    constexpr int32_t kRuns = BLOCKS_PER_LANE * kQuads * 4;
    for (int32_t r = threadIdx.x; r < kRuns; r += THREADS_PER_CTA) {
      const int32_t t = r % 4;
      const int32_t q = (r / 4) % kQuads;
      const int32_t u = r / (4 * kQuads);
      const int32_t row0 = static_cast<int32_t>(swz_band) * 128 + swz_a + q;
      const size_t base =
          (static_cast<size_t>(swz_band) * num_tiles_X + swz_j0 / 4 + 4 * u + t) * kSwzTileSize +
          static_cast<size_t>(swz_a + q) * 16;
      const uint32_t *src = &swz_scratch[((u * kQuads + q) * 4 + t) * kSubRows];
      bool all_rows = true;
      if constexpr (CHECK_BOUNDS) {
        if (swz_j0 + kBlocksPerWarp * u + 4 * t >= blocks_per_row) {
          continue;
        }
#pragma unroll
        for (int32_t sr = 0; sr < kSubRows; ++sr) {
          all_rows = all_rows && (row0 + 32 * sr < rows);
        }
      }
      if (all_rows) {
        uint32_t quad_words[4] = {src[0], src[1], src[2], src[3]};
        ptx::st_global_b32x4(reinterpret_cast<uint32_t *>(&scales[base]), quad_words,
                             output_policy);
      } else {
#pragma unroll
        for (int32_t sr = 0; sr < kSubRows; ++sr) {
          if (row0 + 32 * sr < rows) {
            *reinterpret_cast<uint32_t *>(&scales[base + sr * 4]) = src[sr];
          }
        }
      }
    }
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

namespace {

/*! \brief Launch quantize_contiguous_kernel for a configuration resolved at
 *         run time, instantiating only the combinations the tier table uses. */
template <typename OType, bool CHECK_BOUNDS, bool SWIZZLED_SCALES>
void launch_contiguous_checked(const LaunchConfig &config, int64_t grid,
                               uint32_t first_streaming_cta, int64_t num_blocks,
                               int32_t blocks_per_row, int32_t num_tiles_X, int32_t col_spans,
                               int32_t rows, const uint32_t *input, uint32_t *output,
                               e8m0_t *scales, cudaStream_t stream) {
  NVTE_CHECK(grid <= static_cast<int64_t>(UINT32_MAX), "MXFP8 rowwise cast needs ", grid,
             " CTAs, which exceeds the grid limit.");
  const dim3 blocks(static_cast<unsigned>(grid));
  const dim3 threads(static_cast<unsigned>(config.threads_per_cta));

  const bool partial = config.l2_cached_cta_percent != 0;

  if (config.threads_per_cta == 256 && config.blocks_per_lane == 1 && !partial) {
    quantize_contiguous_kernel<OType, 256, 1, false, CHECK_BOUNDS, SWIZZLED_SCALES>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta, num_blocks,
                                         blocks_per_row, num_tiles_X, col_spans, rows);
  } else if (config.threads_per_cta == 256 && config.blocks_per_lane == 2 && !partial) {
    quantize_contiguous_kernel<OType, 256, 2, false, CHECK_BOUNDS, SWIZZLED_SCALES>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta, num_blocks,
                                         blocks_per_row, num_tiles_X, col_spans, rows);
  } else if (config.threads_per_cta == 128 && config.blocks_per_lane == 2 && partial) {
    quantize_contiguous_kernel<OType, 128, 2, true, CHECK_BOUNDS, SWIZZLED_SCALES>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta, num_blocks,
                                         blocks_per_row, num_tiles_X, col_spans, rows);
  } else if (config.threads_per_cta == 256 && config.blocks_per_lane == 2 && partial) {
    quantize_contiguous_kernel<OType, 256, 2, true, CHECK_BOUNDS, SWIZZLED_SCALES>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta, num_blocks,
                                         blocks_per_row, num_tiles_X, col_spans, rows);
  } else {
    NVTE_ERROR("No quantize_contiguous_kernel instantiation for ", config.threads_per_cta,
               " threads, ", config.blocks_per_lane, " blocks per lane, partial L2 caching ",
               partial, ".");
  }
}

/*! \brief Pick the bounds-checked or unchecked instantiation. */
template <typename OType, bool SWIZZLED_SCALES>
void launch_contiguous(const LaunchConfig &config, int64_t grid, uint32_t first_streaming_cta,
                       int64_t num_blocks, int32_t blocks_per_row, int32_t num_tiles_X,
                       int32_t col_spans, int32_t rows, bool check_bounds, const uint32_t *input,
                       uint32_t *output, e8m0_t *scales, cudaStream_t stream) {
  if (check_bounds) {
    launch_contiguous_checked<OType, true, SWIZZLED_SCALES>(
        config, grid, first_streaming_cta, num_blocks, blocks_per_row, num_tiles_X, col_spans, rows,
        input, output, scales, stream);
  } else {
    launch_contiguous_checked<OType, false, SWIZZLED_SCALES>(
        config, grid, first_streaming_cta, num_blocks, blocks_per_row, num_tiles_X, col_spans, rows,
        input, output, scales, stream);
  }
}

}  // namespace

template <typename OType, bool SWIZZLED_SCALES>
void launch_cast_rowwise(const void *input, void *output, void *scales, int rows, int cols,
                         int scale_stride, cudaStream_t stream) {
  NVTE_CHECK(cols % kBlockElems == 0, "Rowwise MXFP8 requires the column count (", cols,
             ") to be a multiple of the MX block size (", kBlockElems, ").");

  const int32_t blocks_per_row = cols / kBlockElems;
  const uint32_t *in = reinterpret_cast<const uint32_t *>(input);
  uint32_t *out = reinterpret_cast<uint32_t *>(output);
  e8m0_t *scale_out = reinterpret_cast<e8m0_t *>(scales);

  // The scale array is treated as one flat image of the block sequence, so a
  // padded row stride would misplace every scale past the first row.  Dispatch
  // only admits cols % 128 == 0, which makes the allocators' round-up a no-op --
  // state that as a contract rather than carry an unreachable slow path.
  NVTE_CHECK(scale_stride == blocks_per_row, "Rowwise MXFP8 requires a packed scale array: stride ",
             scale_stride, " must equal cols/", kBlockElems, " = ", blocks_per_row, ".");

  const int64_t num_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  const int64_t output_bytes = static_cast<int64_t>(rows) * cols;

  int32_t tier = 0;
  while (tier < kNumTiers - 1 && output_bytes > kTierMaxBytes[tier]) {
    ++tier;
  }
  LaunchConfig config = kTierConfigs[tier];
  if constexpr (SWIZZLED_SCALES) {
    // The swizzled shape needs four warps just to cover the four sub-rows of one
    // row offset, so a 128-thread CTA has a single quad and nothing to amortise
    // its row span across.  Give it the wider CTA.
    if (config.threads_per_cta == 128) {
      config.threads_per_cta = 256;
    }
    // The rest of the tier is reused as-is, and the table was fit on the packed
    // layout.  Two of its assumptions are weaker here:
    //   - l2_cached_cta_percent splits CTAs by index, which in the packed layout
    //     is a prefix of the tensor.  Swizzled, the index decomposes into
    //     (band, row slot, column span) with spans varying fastest, so the same
    //     split selects rows interleaved across each band rather than a
    //     contiguous run.  It still covers roughly the same fraction of bytes.
    //   - tier 0 gives each lane a single block, which leaves a swizzled CTA
    //     with the least to amortise its four-sub-row span across; the smallest
    //     shapes are where this layout is furthest off the packed path.
    // Neither has been re-tuned for this layout.
  }

  // Every CTA covers a whole number of MX blocks.  When the count does not
  // divide evenly the grid is rounded up and the kernel predicates its accesses
  // instead, so there is always exactly one launch.
  const int64_t blocks_per_cta =
      static_cast<int64_t>(config.threads_per_cta) / kLanesPerBlock * config.blocks_per_lane;
  const int64_t warps_per_cta = config.threads_per_cta / THREADS_PER_WARP;

  int64_t grid;
  bool check_bounds;
  // Column spans per 128-row band.  The kernel decomposes its CTA index by this
  // exact value, so it is computed once, here, and passed down.
  int32_t col_spans = 1;
  if constexpr (SWIZZLED_SCALES) {
    // A CTA covers 4 sub-rows x (warps_per_cta / 4) row offsets and one span of
    // 16 * blocks_per_lane columns, so a 128-row band needs 32 / quads CTAs per
    // span.  Column spans vary fastest (see the kernel) to keep concurrent CTAs on
    // adjacent columns of the same rows.
    const int64_t quads = warps_per_cta / 4;
    const int64_t span = 16 * static_cast<int64_t>(config.blocks_per_lane);
    col_spans = static_cast<int32_t>(DIVUP(static_cast<int64_t>(blocks_per_row), span));
    const int64_t bands = DIVUP(rows, 128);
    grid = bands * (32 / quads) * static_cast<int64_t>(col_spans);
    check_bounds = (rows % 128) != 0 || (static_cast<int64_t>(blocks_per_row) % span) != 0;
  } else {
    check_bounds = (num_blocks % blocks_per_cta) != 0;
    grid = DIVUP(num_blocks, blocks_per_cta);
  }

  if (grid > 0) {
    const uint32_t first_streaming_cta =
        static_cast<uint32_t>(grid * config.l2_cached_cta_percent / 100);
    // Scale tiles across the row axis; only read when SWIZZLED_SCALES.
    const int32_t num_tiles_X = static_cast<int32_t>(DIVUP(cols, 128));

    if constexpr (SWIZZLED_SCALES) {
      // The packed 32-bit scale store assumes a 4-aligned group of blocks never
      // straddles a row, which needs blocks_per_row to be a multiple of 4.  The
      // dispatch only reaches here with cols % 128 == 0, which guarantees it, but
      // that is the caller's invariant and this kernel would corrupt scales
      // silently without it rather than fail.
      NVTE_CHECK(blocks_per_row % 4 == 0, "GEMM-swizzled MXFP8 scales require the column count (",
                 cols, ") to be a multiple of 128; blocks per row was ", blocks_per_row, ".");
      // Row and block indices inside the kernel are int32, and the CTA index is
      // decomposed from blockIdx.x, so the swizzled path is bounded by the row
      // count rather than by the block count the packed path carries.
      NVTE_CHECK(rows <= INT32_MAX - 128,
                 "GEMM-swizzled MXFP8 scales index rows with 32 bits; got ", rows);
    }
    launch_contiguous<OType, SWIZZLED_SCALES>(config, grid, first_streaming_cta, num_blocks,
                                              blocks_per_row, num_tiles_X, col_spans, rows,
                                              check_bounds, in, out, scale_out, stream);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

// The MXFP8 output types the specialized dispatch can reach; see hasSpec.
template void launch_cast_rowwise<fp8e4m3, false>(const void *, void *, void *, int, int, int,
                                                  cudaStream_t);
template void launch_cast_rowwise<fp8e5m2, false>(const void *, void *, void *, int, int, int,
                                                  cudaStream_t);
template void launch_cast_rowwise<fp8e4m3, true>(const void *, void *, void *, int, int, int,
                                                 cudaStream_t);
template void launch_cast_rowwise<fp8e5m2, true>(const void *, void *, void *, int, int, int,
                                                 cudaStream_t);

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
