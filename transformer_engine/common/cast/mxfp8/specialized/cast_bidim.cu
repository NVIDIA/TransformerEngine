/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_bidim.cu
 *  \brief Register-resident bidimensional MXFP8 quantization kernel.
 */

#include <cuda_runtime.h>

#include "../../../common.h"
#include "../../../util/cuda_runtime.h"
#include "../../../util/ptx.cuh"
#include "../../../utils.cuh"
#include "cast_bidim.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

namespace ptx = transformer_engine::ptx;

// Bidimensional MXFP8 produces two independent quantizations of one BF16
// tensor: a rowwise one, where 32 consecutive elements of a row share a scale,
// and a colwise one, where 32 consecutive elements of a column share a scale.
// Both outputs keep the input's [M, K] row-major layout -- the colwise result
// is not transposed, only scaled differently.
//
// A CTA owns a 32-row band crossed with a strip of columns.  Thirty-two rows is
// exactly the colwise block height, so the colwise reduction closes inside the
// CTA: the tile is read from global memory once, into registers, and drives
// both passes.  That is what lets this kernel drop the TMA pipeline the
// specialized kernel uses -- there is no second pass to stage for.
//
// Within a tile:
//   - each warp owns kRowsPerWarp consecutive rows of the band,
//   - each lane owns COLS_PER_LANE consecutive columns of those rows,
//   - the rowwise scale needs a reduction across the lanes that share a
//     32-column block, which shuffles handle,
//   - the colwise scale needs a reduction down all 32 rows, hence across warps,
//     which is the one thing shuffles cannot reach and the only reason this
//     kernel touches shared memory.

// Elements in one MX block, and equivalently the colwise block height.
constexpr int32_t kBlockElems = 32;
// Rows a CTA covers.  Must equal kBlockElems to keep the colwise reduction
// CTA-local.
constexpr int32_t kRowsPerTile = kBlockElems;

constexpr int32_t kWarpsPerCta = 8;
constexpr int32_t kThreadsPerCta = kWarpsPerCta * THREADS_PER_WARP;
constexpr int32_t kRowsPerWarp = kRowsPerTile / kWarpsPerCta;
static_assert(kRowsPerTile % kWarpsPerCta == 0, "Warps must divide the row band evenly.");

// BF16 packs two elements per 32-bit word, and all the packed math below works
// on those words, so a shared-memory "slot" holds one column pair.
constexpr int32_t kElemsPerWord = sizeof(uint32_t) / sizeof(bf16);

// Bytes in an L2 cache line, for sizing the software prefetch.
constexpr int32_t kCacheLineBytes = 128;

//
// MX scale arithmetic.
//
// The E8M0 scale of a block is the smallest power of two with
// amax / scale <= 448, i.e. biased_exponent = ceil(log2(amax / 448)) + 127.
// For a BF16 amax that has an exact closed form in the bit pattern.
//
// Write amax = m * 2^(e - 127), with the biased exponent e in bits 7..14 and
// the stored mantissa in bits 0..6.  Then
//
//     ceil(log2(amax / 448)) + 127  =  e - 8   for m <= 1.75
//                                   =  e - 7   for m >  1.75
//
// and "m > 1.75" is exactly "mantissa field >= 97", because 1.75 is mantissa 96
// and 97 is the next representable step.  Adding 31 to the mantissa field
// carries into the exponent for mantissa >= 97 and for nothing below it, so a
// single add and mask round e by precisely that rule.
//
// This is not an approximation of the usual amax * (1/448) route: it is exact,
// and better behaved, because 1/448 is not representable in FP32 and that
// product can round across the boundary.
//
// The helpers return the reciprocal scale, ready to multiply by, since that is
// what the packed conversion consumes.  mx_scale_byte recovers the stored E8M0
// byte from it.

constexpr uint32_t kMantissaRoundUp = 31u;
constexpr uint32_t kBf16ExponentMask = 0x7F80u;

// Exponent of the output type's largest normal: 8 for E4M3 (448 = 1.75 * 2^8)
// and 15 for E5M2 (57344 = 1.75 * 2^15).  Both share the 1.75 mantissa, which
// is exactly why the "mantissa field >= 97" rule above holds for either and
// only this offset has to change.
template <typename OType>
constexpr uint32_t kMaxNormExponent = Quantized_Limits<OType>::max_unbiased_exponent;

// Smallest exponent field allowed, so the reciprocal stays a normal BF16.
template <typename OType>
constexpr uint32_t kMinExponentField = kMaxNormExponent<OType> << 7;

// Encoding of 2^(127 + offset); subtracting the rounded exponent field yields
// the reciprocal scale directly.
template <typename OType>
constexpr uint32_t kReciprocalBias = (254u + kMaxNormExponent<OType>) << 7;
// BF16 magnitude at or above which a value is Inf or NaN.
constexpr uint32_t kBf16InfBits = 0x7F80u;
// Reciprocal paired with the E8M0 NaN scale (254): the smallest subnormal.
constexpr uint32_t kNaNReciprocal = 0x0040u;
constexpr uint32_t kBf16MagnitudeMask = 0x7FFFu;
// Bias such that kScaleByteBias - (reciprocal >> 7) is the E8M0 scale byte.
constexpr uint32_t kScaleByteBias = 254u;

/*! \brief Reciprocal MX scale for one BF16 amax, as BF16 bits.
 *  \param amax_bits  Block amax magnitude bits, sign already cleared. */
template <typename OType>
__device__ __forceinline__ uint32_t mx_scale_reciprocal(uint32_t amax_bits) {
  if (amax_bits >= kBf16InfBits) {
    return kNaNReciprocal;
  }
  const uint32_t rounded_exponent =
      max((amax_bits + kMantissaRoundUp) & kBf16ExponentMask, kMinExponentField<OType>);
  return kReciprocalBias<OType> - rounded_exponent;
}

/*! \brief E8M0 scale byte matching a reciprocal from mx_scale_reciprocal. */
__device__ __forceinline__ e8m0_t mx_scale_byte(uint32_t reciprocal_bits) {
  return static_cast<e8m0_t>(kScaleByteBias - (reciprocal_bits >> 7));
}

/*! \brief mx_scale_reciprocal applied to two columns at once.
 *
 * Valid only when neither half is Inf or NaN.  Callers test that first, which
 * costs one comparison for the pair.
 */
template <typename OType>
__device__ __forceinline__ uint32_t mx_scale_reciprocal_x2(uint32_t amax_pair) {
  constexpr uint32_t kMagnitudeMaskPair = 0x7FFF7FFFu;
  constexpr uint32_t kMantissaRoundUpPair = 0x001F001Fu;
  constexpr uint32_t kExponentMaskPair = 0xFF80FF80u;
  constexpr uint32_t kMinExponentPair = (kMinExponentField<OType> << 16) | kMinExponentField<OType>;
  constexpr uint32_t kReciprocalBiasPair = (kReciprocalBias<OType> << 16) | kReciprocalBias<OType>;

  // Each half is at most 0x7FFF, so adding 31 cannot carry out of the low half
  // into the high one; the two halves round independently.
  const uint32_t magnitudes = amax_pair & kMagnitudeMaskPair;
  ptx::bf16x2 rounded, floor_pair, clamped;
  reinterpret_cast<uint32_t &>(rounded) = (magnitudes + kMantissaRoundUpPair) & kExponentMaskPair;
  reinterpret_cast<uint32_t &>(floor_pair) = kMinExponentPair;
  // Both operands are positive BF16 patterns, so the magnitude-max doubles as a
  // per-half clamp against the minimum exponent.
  ptx::abs_max_2x(clamped, rounded, floor_pair);
  return kReciprocalBiasPair - reinterpret_cast<const uint32_t &>(clamped);
}

/*! \brief The two E8M0 scale bytes of a packed reciprocal pair, in the low
 *         16 bits of the result. */
__device__ __forceinline__ uint32_t mx_scale_byte_x2(uint32_t reciprocal_pair) {
  constexpr uint32_t kScaleByteBiasPair = 0x00FE00FEu;
  const uint32_t bytes = kScaleByteBiasPair - ((reciprocal_pair >> 7) & 0x00FF00FFu);
  // Gather the two scale bytes, at byte 0 and byte 2, into the low half.
  return __byte_perm(bytes, 0u, 0x4420);
}

/*! \brief Fold a BF16 pair's halves together and return the magnitude.
 *
 * `max.xorsign.abs` sets the result sign to the XOR of its inputs, so an
 * accumulator built from it can come out negative even when its magnitude is
 * right.  Only the magnitude matters for a scale.
 */
__device__ __forceinline__ uint32_t fold_pair_magnitude(uint32_t pair) {
  ptx::bf16x2 lo, hi, folded;
  reinterpret_cast<uint32_t &>(lo) = pair;
  reinterpret_cast<uint32_t &>(hi) = __byte_perm(pair, pair, 0x1032);
  ptx::abs_max_2x(folded, lo, hi);
  return reinterpret_cast<const uint32_t &>(folded) & kBf16MagnitudeMask;
}

/*! \brief Broadcast a BF16 reciprocal scale into both halves of a pair. */
__device__ __forceinline__ ptx::bf16x2 broadcast_pair(uint32_t scale_bits) {
  ptx::bf16x2 result;
  reinterpret_cast<uint32_t &>(result) = __byte_perm(scale_bits, 0u, 0x1010);
  return result;
}

/*! \brief Scale one lane's BF16 words and pack them into FP8E4M3 words.
 *
 * \param scales  One reciprocal per input word.  The rowwise pass passes the
 *                same broadcast value throughout; the colwise pass gives each
 *                column pair its own.
 */
template <typename OType, int32_t WORDS_IN>
__device__ __forceinline__ void scale_and_convert(const uint32_t (&in)[WORDS_IN],
                                                  const ptx::bf16x2 (&scales)[WORDS_IN],
                                                  uint32_t (&out)[WORDS_IN / 2]) {
#pragma unroll
  for (int32_t i = 0; i < WORDS_IN / 2; ++i) {
    ptx::mul_cvt_4x(reinterpret_cast<ptx::FPx4<OType> &>(out[i]),
                    reinterpret_cast<const ptx::bf16x2 &>(in[2 * i]), scales[2 * i],
                    reinterpret_cast<const ptx::bf16x2 &>(in[2 * i + 1]), scales[2 * i + 1]);
  }
}

/*! \brief Store a lane's output words with the shared L2 policy. */
template <int32_t WORDS>
__device__ __forceinline__ void store_words(void *dst, const uint32_t (&words)[WORDS],
                                            uint64_t policy) {
  static_assert(WORDS == 2 || WORDS == 4, "Output stores are 64- or 128-bit.");
  if constexpr (WORDS == 4) {
    ptx::st_global_b32x4(dst, words, policy);
  } else {
    ptx::st_global_b32x2(dst, words, policy);
  }
}

//
// Shared-memory access for the colwise fold, as one vector operation per call.
//
// WORDS is how many consecutive 32-bit slots a lane owns in each half of the
// slot array: 4 for the wide tile (LDS/STS.128) and 2 for the narrow one
// (LDS/STS.64).  Spelling the vector type out, rather than looping over
// scalars and hoping the compiler merges them, is what guarantees the single
// wide access the half-split layout was designed around.
//

template <int32_t WORDS>
__device__ __forceinline__ void store_shared_words(uint32_t *dst, const uint32_t *src) {
  static_assert(WORDS == 2 || WORDS == 4, "Shared-memory access is 64- or 128-bit.");
  if constexpr (WORDS == 4) {
    *reinterpret_cast<uint4 *>(dst) = make_uint4(src[0], src[1], src[2], src[3]);
  } else {
    *reinterpret_cast<uint2 *>(dst) = make_uint2(src[0], src[1]);
  }
}

template <int32_t WORDS>
__device__ __forceinline__ void load_shared_words(uint32_t *dst, const uint32_t *src) {
  static_assert(WORDS == 2 || WORDS == 4, "Shared-memory access is 64- or 128-bit.");
  if constexpr (WORDS == 4) {
    const uint4 v = *reinterpret_cast<const uint4 *>(src);
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
  } else {
    const uint2 v = *reinterpret_cast<const uint2 *>(src);
    dst[0] = v.x;
    dst[1] = v.y;
  }
}

/*! \brief Quantize a 32-row band crossed with a strip of columns.
 *
 * \tparam COLS_PER_LANE     Columns one lane owns.  16 gives a 32x512 tile read
 *                           with 256-bit loads; 8 gives 32x256 with 128-bit
 *                           loads, trading load width for more CTAs and so
 *                           better latency hiding on deep shapes.
 * \tparam K_COMPILE_TIME    When non-zero, the column count is a compile-time
 *                           constant, turning the row stride, the scale strides
 *                           and the grid width into immediates.  Zero takes
 *                           them at run time.
 * \tparam MIN_BLOCKS_PER_SM Occupancy target for __launch_bounds__.
 *
 * \param prefetch_distance_ctas  How far ahead, in CTAs, to prefetch input.
 *                                Set to one full resident wave so the next
 *                                wave's tile lands in L2 as this one drains;
 *                                0 disables the prefetch.
 */
template <typename OType, int32_t COLS_PER_LANE, int32_t K_COMPILE_TIME, int32_t MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(kThreadsPerCta, MIN_BLOCKS_PER_SM) void quantize_bidim_kernel(
    const uint8_t *__restrict__ input, uint8_t *__restrict__ output_rowwise,
    uint8_t *__restrict__ scales_rowwise, uint8_t *__restrict__ output_colwise,
    uint8_t *__restrict__ scales_colwise, int32_t cols_rt, int32_t scale_stride_rowwise_rt,
    int32_t scale_stride_colwise_rt, int32_t prefetch_distance_ctas) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int32_t cols = K_COMPILE_TIME ? K_COMPILE_TIME : cols_rt;
  const int32_t scale_stride_rowwise =
      K_COMPILE_TIME ? DIVUP_TO_MULTIPLE(K_COMPILE_TIME / kBlockElems, 4) : scale_stride_rowwise_rt;
  const int32_t scale_stride_colwise =
      K_COMPILE_TIME ? DIVUP_TO_MULTIPLE(K_COMPILE_TIME, 128) : scale_stride_colwise_rt;

  constexpr int32_t kWordsPerLane = COLS_PER_LANE / kElemsPerWord;
  constexpr int32_t kOutWordsPerLane = kWordsPerLane / 2;
  constexpr int32_t kColsPerTile = THREADS_PER_WARP * COLS_PER_LANE;
  // Lanes that must cooperate to cover one 32-column rowwise block.
  constexpr int32_t kLanesPerRowBlock = kBlockElems / COLS_PER_LANE;
  // Column pairs in the tile, i.e. shared-memory slots for the colwise fold.
  constexpr int32_t kColumnSlots = kColsPerTile / kElemsPerWord;
  // Half-split layout: a lane deposits its partials as two vector stores, one
  // into each half of the slot array, which keeps the store and the later
  // read-back free of bank conflicts.  Four slots per half for the wide tile
  // (COLS_PER_LANE 16, so 128-bit accesses) and two for the narrow one
  // (COLS_PER_LANE 8, so 64-bit) -- both instantiations are live, the narrow
  // tile being what deep grids and K-not-a-multiple-of-512 shapes use.
  constexpr int32_t kSlotsPerLaneHalf = kWordsPerLane / 2;
  constexpr int32_t kHalfSlots = kColumnSlots / 2;

  // Both arrays are indexed by column *pair*, one 32-bit slot each, because
  // that is the granularity the packed BF16 math works at.
  //
  // s_column_partial holds per-warp amax candidates: two BF16 magnitudes.
  // s_column_scale holds the finished *reciprocal* scales: again two BF16, one
  // per column, not the one-byte E8M0 encodings.  BF16 is what mul_cvt_4x
  // multiplies by, so keeping the reciprocal in that form avoids a byte-to-BF16
  // expansion inside the conversion loop; the E8M0 bytes are produced once, by
  // mx_scale_byte_x2, only when warp 0 writes the scale array out.
  __shared__ uint32_t s_column_partial[kWarpsPerCta][kColumnSlots];
  __shared__ uint32_t s_column_scale[kColumnSlots];

  const int32_t tid = threadIdx.x;
  const int32_t warp = tid / THREADS_PER_WARP;
  const int32_t lane = tid % THREADS_PER_WARP;

  const int32_t row0 = blockIdx.y * kRowsPerTile + warp * kRowsPerWarp;
  const int32_t col0 = blockIdx.x * kColsPerTile;

  // One policy for every stream.  Each byte is touched once, but marking them
  // evict_last measured better than streaming: holding the tile's lines through
  // the CTA's lifetime is what keeps the three write bursts coalesced.
  const uint64_t policy = ptx::create_l2_policy_evict_last();

  const size_t row_stride_bytes = static_cast<size_t>(cols) * sizeof(bf16);
  const size_t lane_offset = static_cast<size_t>(row0) * cols + col0 + COLS_PER_LANE * lane;
  const uint8_t *tile_in = input + lane_offset * sizeof(bf16);
  uint8_t *out_row = output_rowwise + lane_offset;
  uint8_t *out_col = output_colwise + lane_offset;
  uint8_t *scale_row = scales_rowwise + static_cast<size_t>(row0) * scale_stride_rowwise +
                       (col0 / kBlockElems) + (lane / kLanesPerRowBlock);
  // One lane of each cooperating group writes the shared rowwise scale byte.
  const bool owns_row_scale = (lane % kLanesPerRowBlock) == 0;

  // Read the whole tile into registers; both passes run off these.
  uint32_t tile[kRowsPerWarp][kWordsPerLane];
#pragma unroll
  for (int32_t i = 0; i < kRowsPerWarp; ++i) {
    const void *src = tile_in + i * row_stride_bytes;
    if constexpr (kWordsPerLane == 8) {
      ptx::ld_global_nc_b32x8(tile[i], src, policy);
    } else {
      ptx::ld_global_nc_b32x4(tile[i], src, policy);
    }
  }

  // ---- rowwise pass -------------------------------------------------------
#pragma unroll
  for (int32_t i = 0; i < kRowsPerWarp; ++i) {
    ptx::bf16x2 amax;
    reinterpret_cast<uint32_t &>(amax) = tile[i][0];
#pragma unroll
    for (int32_t k = 1; k < kWordsPerLane; ++k) {
      ptx::abs_max_2x(amax, amax, reinterpret_cast<const ptx::bf16x2 &>(tile[i][k]));
    }
    // Butterfly across the lanes that share this 32-column block.
#pragma unroll
    for (int32_t d = 1; d < kLanesPerRowBlock; d <<= 1) {
      ptx::bf16x2 partner;
      reinterpret_cast<uint32_t &>(partner) =
          __shfl_xor_sync(0xFFFFFFFFu, reinterpret_cast<const uint32_t &>(amax), d);
      ptx::abs_max_2x(amax, amax, partner);
    }

    const uint32_t reciprocal =
        mx_scale_reciprocal<OType>(fold_pair_magnitude(reinterpret_cast<const uint32_t &>(amax)));
    const ptx::bf16x2 broadcast = broadcast_pair(reciprocal);
    ptx::bf16x2 scales[kWordsPerLane];
#pragma unroll
    for (int32_t k = 0; k < kWordsPerLane; ++k) {
      scales[k] = broadcast;
    }
    uint32_t out_words[kOutWordsPerLane];
    scale_and_convert<OType>(tile[i], scales, out_words);
    store_words(out_row + static_cast<size_t>(i) * cols, out_words, policy);

    if (owns_row_scale) {
      ptx::st_global_b8(scale_row + static_cast<size_t>(i) * scale_stride_rowwise,
                        mx_scale_byte(reciprocal), policy);
    }
  }

  // ---- colwise pass -------------------------------------------------------
  // Each warp reduces its own rows first; the warps then meet in shared memory,
  // the only cross-warp step in the kernel.
  uint32_t column_partial[kWordsPerLane];
#pragma unroll
  for (int32_t k = 0; k < kWordsPerLane; ++k) {
    ptx::bf16x2 acc;
    reinterpret_cast<uint32_t &>(acc) = tile[0][k];
#pragma unroll
    for (int32_t i = 1; i < kRowsPerWarp; ++i) {
      ptx::abs_max_2x(acc, acc, reinterpret_cast<const ptx::bf16x2 &>(tile[i][k]));
    }
    column_partial[k] = reinterpret_cast<const uint32_t &>(acc);
  }

  store_shared_words<kSlotsPerLaneHalf>(&s_column_partial[warp][kSlotsPerLaneHalf * lane],
                                        &column_partial[0]);
  store_shared_words<kSlotsPerLaneHalf>(
      &s_column_partial[warp][kHalfSlots + kSlotsPerLaneHalf * lane],
      &column_partial[kSlotsPerLaneHalf]);
  __syncthreads();

  // Fold the per-warp partials into one scale per column pair.  Threads take
  // stride-1 slots so the strided reads across warps stay conflict-free.
  for (int32_t slot = tid; slot < kColumnSlots; slot += kThreadsPerCta) {
    const uint32_t *column = &s_column_partial[0][slot];
    ptx::bf16x2 acc;
    reinterpret_cast<uint32_t &>(acc) = column[0];
#pragma unroll
    for (int32_t w = 1; w < kWarpsPerCta; ++w) {
      ptx::abs_max_2x(acc, acc, reinterpret_cast<const ptx::bf16x2 &>(column[w * kColumnSlots]));
    }
    const uint32_t amax_pair = reinterpret_cast<const uint32_t &>(acc);

    uint32_t reciprocal_pair;
    if (__builtin_expect(fold_pair_magnitude(amax_pair) < kBf16InfBits, 1)) {
      reciprocal_pair = mx_scale_reciprocal_x2<OType>(amax_pair);
    } else {
      // At least one of the two columns is Inf or NaN; take them separately.
      const uint32_t lo = mx_scale_reciprocal<OType>(amax_pair & kBf16MagnitudeMask);
      const uint32_t hi = mx_scale_reciprocal<OType>((amax_pair >> 16) & kBf16MagnitudeMask);
      reciprocal_pair = lo | (hi << 16);
    }
    s_column_scale[slot] = reciprocal_pair;
  }
  __syncthreads();

  // Read back the reciprocals covering this lane's own columns.
  uint32_t column_reciprocal[kWordsPerLane];
  load_shared_words<kSlotsPerLaneHalf>(&column_reciprocal[0],
                                       &s_column_scale[kSlotsPerLaneHalf * lane]);
  load_shared_words<kSlotsPerLaneHalf>(&column_reciprocal[kSlotsPerLaneHalf],
                                       &s_column_scale[kHalfSlots + kSlotsPerLaneHalf * lane]);

  // Warp 0 emits the colwise scale row: one byte per column, one coalesced
  // vector store per lane.
  if (warp == 0) {
    uint32_t packed[kOutWordsPerLane];
#pragma unroll
    for (int32_t j = 0; j < kOutWordsPerLane; ++j) {
      packed[j] = mx_scale_byte_x2(column_reciprocal[2 * j]) |
                  (mx_scale_byte_x2(column_reciprocal[2 * j + 1]) << 16);
    }
    store_words(scales_colwise + static_cast<size_t>(blockIdx.y) * scale_stride_colwise + col0 +
                    COLS_PER_LANE * lane,
                packed, policy);
  }

  // Software L2 prefetch, issued here so the next wave's input arrives while
  // this CTA is still writing.  Pulling it earlier -- right after our own loads
  // -- measured worse: the lines then sit in L2 for the whole CTA lifetime and
  // crowd out the write bursts.
  if (prefetch_distance_ctas > 0) {
    const int32_t grid_x = K_COMPILE_TIME ? (K_COMPILE_TIME / kColsPerTile) : gridDim.x;
    const int32_t target = blockIdx.y * grid_x + blockIdx.x + prefetch_distance_ctas;
    if (target < grid_x * static_cast<int32_t>(gridDim.y)) {
      constexpr int32_t kLinesPerTile =
          kRowsPerTile * kColsPerTile * sizeof(bf16) / kCacheLineBytes;
      constexpr int32_t kLinesPerRow = kLinesPerTile / kRowsPerTile;
      const int32_t target_y = target / grid_x;
      const int32_t target_x = target - target_y * grid_x;
      for (int32_t line = tid; line < kLinesPerTile; line += kThreadsPerCta) {
        const size_t offset =
            (static_cast<size_t>(target_y * kRowsPerTile + line / kLinesPerRow) * cols +
             target_x * kColsPerTile) *
                sizeof(bf16) +
            (line % kLinesPerRow) * kCacheLineBytes;
        ptx::prefetch_l2_evict_last(input + offset);
      }
    }
  }

  // ---- colwise quantized output -------------------------------------------
  ptx::bf16x2 colwise_scales[kWordsPerLane];
#pragma unroll
  for (int32_t k = 0; k < kWordsPerLane; ++k) {
    reinterpret_cast<uint32_t &>(colwise_scales[k]) = column_reciprocal[k];
  }
#pragma unroll
  for (int32_t i = 0; i < kRowsPerWarp; ++i) {
    uint32_t out_words[kOutWordsPerLane];
    scale_and_convert<OType>(tile[i], colwise_scales, out_words);
    store_words(out_col + static_cast<size_t>(i) * cols, out_words, policy);
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

namespace {

// Tile shapes.  The wide tile reads 256 bits per lane and suits shallow grids;
// the narrow tile halves that to get twice the CTAs, which hides latency better
// once the grid is deep enough to keep every SM busy regardless.
constexpr int32_t kWideColsPerLane = 16;
constexpr int32_t kNarrowColsPerLane = 8;
constexpr int32_t kWideMinBlocksPerSm = 4;
constexpr int32_t kNarrowMinBlocksPerSm = 6;

// Thread-block cluster widths.  Clustering lets neighbouring CTAs, which read
// adjacent columns of the same rows, share L2 traffic.
constexpr int32_t kWideClusterShallow = 1;
constexpr int32_t kWideClusterDeep = 4;
constexpr int32_t kNarrowCluster = 8;
// Grid size, in CTAs, past which the wide tile switches to the deep cluster.
constexpr int64_t kWideDeepClusterFrom = 4096;
// Grid size, in resident waves, past which the narrow tile takes over.
constexpr int32_t kNarrowFromWaves = 8;

/*! \brief Launch quantize_bidim_kernel, specializing on K where we can.
 *
 * The handful of column counts below cover the shapes that matter in practice;
 * anything else takes the run-time path, which costs a few address
 * computations rather than a recompile.
 */
template <typename OType, int32_t COLS_PER_LANE, int32_t MIN_BLOCKS_PER_SM>
void launch_tiled(const void *input, void *output_rowwise, void *scales_rowwise,
                  void *output_colwise, void *scales_colwise, int32_t rows, int32_t cols,
                  int32_t scale_stride_rowwise, int32_t scale_stride_colwise, int32_t cluster_width,
                  int32_t prefetch_distance_ctas, cudaStream_t stream) {
  constexpr int32_t kColsPerTile = THREADS_PER_WARP * COLS_PER_LANE;
  const dim3 grid(cols / kColsPerTile, rows / kRowsPerTile);

  // A cluster launch is rejected outright unless its width divides the grid, so
  // narrow the requested width to the largest power of two that does.  Column
  // counts that are not a multiple of the tile width times the cluster width --
  // 7168 columns with the narrow tile, for instance, giving a grid of 28 -- are
  // otherwise a hard launch failure rather than a slow path.
  int32_t cluster_x = cluster_width;
  while (cluster_x > 1 && static_cast<int32_t>(grid.x) % cluster_x != 0) {
    --cluster_x;
  }

  cudaLaunchAttribute cluster_attr;
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = cluster_x;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreadsPerCta);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  const uint8_t *in = reinterpret_cast<const uint8_t *>(input);
  uint8_t *qrow = reinterpret_cast<uint8_t *>(output_rowwise);
  uint8_t *srow = reinterpret_cast<uint8_t *>(scales_rowwise);
  uint8_t *qcol = reinterpret_cast<uint8_t *>(output_colwise);
  uint8_t *scol = reinterpret_cast<uint8_t *>(scales_colwise);

#define NVTE_LAUNCH_BIDIM(K_CONST)                                                            \
  do {                                                                                        \
    auto kernel = quantize_bidim_kernel<OType, COLS_PER_LANE, K_CONST, MIN_BLOCKS_PER_SM>;    \
    if (cluster_x > 1) {                                                                      \
      NVTE_CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, in, qrow, srow, qcol, scol, cols,   \
                                         scale_stride_rowwise, scale_stride_colwise,          \
                                         prefetch_distance_ctas));                            \
    } else {                                                                                  \
      kernel<<<grid, kThreadsPerCta, 0, stream>>>(in, qrow, srow, qcol, scol, cols,           \
                                                  scale_stride_rowwise, scale_stride_colwise, \
                                                  prefetch_distance_ctas);                    \
      NVTE_CHECK_CUDA(cudaGetLastError());                                                    \
    }                                                                                         \
  } while (0)

  switch (cols) {
    case 2048:
      NVTE_LAUNCH_BIDIM(2048);
      break;
    case 4096:
      NVTE_LAUNCH_BIDIM(4096);
      break;
    case 7168:
      NVTE_LAUNCH_BIDIM(7168);
      break;
    case 8192:
      NVTE_LAUNCH_BIDIM(8192);
      break;
    case 16384:
      NVTE_LAUNCH_BIDIM(16384);
      break;
    case 32768:
      NVTE_LAUNCH_BIDIM(32768);
      break;
    default:
      NVTE_LAUNCH_BIDIM(0);
      break;
  }
#undef NVTE_LAUNCH_BIDIM
}

}  // namespace

template <typename OType>
void launch_cast_bidim(const void *input, void *output_rowwise, void *scales_rowwise,
                       void *output_colwise, void *scales_colwise, int rows, int cols,
                       int scale_stride_rowwise, int scale_stride_colwise, cudaStream_t stream) {
  constexpr int32_t kWideColsPerTile = THREADS_PER_WARP * kWideColsPerLane;
  constexpr int32_t kNarrowColsPerTile = THREADS_PER_WARP * kNarrowColsPerLane;

  NVTE_CHECK(rows % kRowsPerTile == 0, "Bidimensional MXFP8 requires the row count (", rows,
             ") to be a multiple of the MX block size (", kRowsPerTile, ").");
  NVTE_CHECK(cols % kNarrowColsPerTile == 0, "Bidimensional MXFP8 requires the column count (",
             cols, ") to be a multiple of ", kNarrowColsPerTile, ".");

  // One resident wave of CTAs, which is how far ahead the prefetch should run.
  const int32_t sm_count = cuda::sm_count();

  if (cols % kWideColsPerTile == 0) {
    const int64_t wide_ctas = static_cast<int64_t>(rows / kRowsPerTile) * (cols / kWideColsPerTile);
    const int64_t wide_resident = static_cast<int64_t>(sm_count) * kWideMinBlocksPerSm;
    if (wide_ctas < static_cast<int64_t>(kNarrowFromWaves) * wide_resident) {
      launch_tiled<OType, kWideColsPerLane, kWideMinBlocksPerSm>(
          input, output_rowwise, scales_rowwise, output_colwise, scales_colwise, rows, cols,
          scale_stride_rowwise, scale_stride_colwise,
          wide_ctas >= kWideDeepClusterFrom ? kWideClusterDeep : kWideClusterShallow,
          static_cast<int32_t>(wide_resident), stream);
      return;
    }
  }

  // Deep grid, or a column count that only the narrow tile divides.
  launch_tiled<OType, kNarrowColsPerLane, kNarrowMinBlocksPerSm>(
      input, output_rowwise, scales_rowwise, output_colwise, scales_colwise, rows, cols,
      scale_stride_rowwise, scale_stride_colwise, kNarrowCluster, sm_count * kNarrowMinBlocksPerSm,
      stream);
}

// The MXFP8 output types the specialized dispatch can reach; see hasSpec.
template void launch_cast_bidim<fp8e4m3>(const void *, void *, void *, void *, void *, int, int,
                                         int, int, cudaStream_t);
template void launch_cast_bidim<fp8e5m2>(const void *, void *, void *, void *, void *, int, int,
                                         int, int, cudaStream_t);

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
