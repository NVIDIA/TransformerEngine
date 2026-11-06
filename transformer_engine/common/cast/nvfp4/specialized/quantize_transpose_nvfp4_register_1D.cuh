/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_transpose_nvfp4_register_1D.cuh
 *  \brief Register-resident BF16 -> NVFP4 1D quantization, rowwise or rowwise + columnwise.
 *
 *  One kernel family, specialized at compile time over
 *      {rowwise, rowwise + columnwise} x {round-to-nearest-even, stochastic rounding}.
 *  The numeric core (block amax -> block scale -> element coefficient -> E2M1 payload) is
 *  shared by every specialization; only the tiling and the output staging differ.  No
 *  specialization stages the input through shared memory or TMA: the rowwise-only one streams
 *  each 16-element block straight from a 32-byte load, and the fused one reads every row twice,
 *  once per direction, with the second read served by L2.  Shared memory holds only the
 *  per-CTA coefficient tables and the transposed output.
 *
 *  Scale arithmetic is TE's own (core::compute_global_encode_scaling_factor_FP4,
 *  core::compute_decoding_scaling_factor, compute_nvfp4_scaling_coefficient<float>), so
 *  round-to-nearest output is bit-identical to quantize_transpose_tuned_1D for every scale
 *  type.  The per-block coefficient depends only on the block's scale byte, so its IEEE divide
 *  is evaluated once per CTA into a table indexed by that byte.
 *
 *  Stochastic rounding draws Philox4x32 (nvfp4_block_philox.cuh: bit-identical to curanddx for
 *  the same key and counter, with NVTE_BUILD_NUM_PHILOX_ROUNDS rounds) once per output block,
 *  keyed by the block's coordinates and direction.  The random bits for a given seed therefore
 *  differ from tuned_1D's, whose stream is keyed by thread id; both are valid stochastic
 *  rounding.
 *
 *  The tiling was tuned on B200.  Its tuning constants are collected in TunedConfig.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_NVFP4_REGISTER_1D_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_NVFP4_REGISTER_1D_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <climits>
#include <mutex>
#include <type_traits>
#include <vector>

#include "../../../common.h"
#include "../../../util/cuda_runtime.h"
#include "../../../util/ptx.cuh"
#include "../../../util/ptx_arch_spec.cuh"
#include "../../../utils.cuh"
#include "../core_nvfp4.cuh"
#include "../nvfp4_block_philox.cuh"
#include "quantize_transpose_nvfp4_tuned_1D.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace quantize_transpose_register_kernel {

#if FP4_TYPE_SUPPORTED

// Tuning constants, measured on B200 at 1500 MHz; retune per target.  Everything that depends
// on the SM count comes from per-device queries instead.
struct TunedConfig {
  // Rowwise-only: a CTA owns kRowBlocks consecutive 16-element blocks of kRowRows* rows.
  // 128 blocks = 2048 input bytes and 128 scale bytes per row per CTA.
  static constexpr int kRowBlocks = 128;
  static constexpr int kRowThreads = 128;
  // Rows per CTA.  RNE is memory-bound and 4 rows (4 loads in flight per thread) is where more
  // CTAs stop buying latency hiding.  SR is bound by the Philox chain, so taller CTAs amortize
  // the per-CTA prologue; the launcher doubles kRowRowsSr while the grid still fills the GPU.
  static constexpr int kRowRowsRne = 4;
  static constexpr int kRowRowsSr = 8;
  // Rows held in registers at once (one 32-byte load each).
  static constexpr int kRowWave = 4;

  // Rowwise + columnwise: a CTA owns kTileK columns and walks up to kMaxSlabs slabs of kSlabM
  // rows, keeping the whole transposed FP4 result for its columns in shared memory.
  static constexpr int kColThreads = 128;
  static constexpr int kTileK = 128;
  static constexpr int kSlabM = 64;
  static constexpr int kMaxSlabs = 4;
  // Adjacent CTAs that pool their rowwise work into one kGroupCTAs * kTileK wide strip, so a
  // warp writes 256 contiguous payload bytes and whole 32-byte scale sectors.
  static constexpr int kGroupCTAs = 4;
  // Cost model for the slab count: waves * (kSlabWork * slabs + kDrainWork), in memory
  // instructions per thread.
  static constexpr long long kSlabWork = 36;
  static constexpr long long kDrainWork = 51;

  // __launch_bounds__ minimum resident CTAs per SM.  Each is the occupancy the specialization
  // reaches anyway; stating it hands ptxas that whole register budget.  Rowwise SR is the
  // exception: 64 registers instead of 72 buys it a quarter more warps to hide Philox with.
  template <bool COLWISE, bool STOCHASTIC>
  static constexpr int min_ctas_per_sm() {
    if constexpr (COLWISE) return STOCHASTIC ? 5 : 7;
    return STOCHASTIC ? 8 : 10;
  }
};

constexpr int kBlockElems = 16;  // elements per NVFP4 1D block; index math below shifts by 4
constexpr int kBlocksPerSlab = TunedConfig::kSlabM / kBlockElems;
// Transposed staging buffer: one row of kStagingWords words per column.  kStagingWords % 32 == 2
// puts the sixteen lanes of an 8-byte shared access on sixteen disjoint bank pairs for both the
// staging store and the drain's gather, and an even stride keeps every slot 8-byte aligned.
constexpr int kStagingWords = 34;
static_assert(kStagingWords >= 2 * kBlocksPerSlab * TunedConfig::kMaxSlabs);
static_assert(kStagingWords % 32 == 2);
constexpr int kStagingScaleBytes = kBlocksPerSlab * TunedConfig::kMaxSlabs + 4;

// Scale bytes a block can produce.  E4M3 block scales are non-negative, so the sign bit is
// always clear; UE5M3 is unsigned and can use all 256 codes.
template <typename ScaleType>
constexpr int kScaleCodes = std::is_same_v<ScaleType, fp8e4m3> ? 128 : 256;

// Element offset of (row, pitch).  Both are 32-bit and their product is exact in 64 bits, so
// a u32 x u32 widening multiply is one IMAD.WIDE.U32 that also folds in the base pointer.
__device__ __forceinline__ size_t row_off(int row, int pitch) {
  return static_cast<size_t>(static_cast<unsigned>(row)) * static_cast<unsigned>(pitch);
}

// ---------------------------------------------------------------------------------------------
// Numeric core.  A block is 16 BF16 values in eight packed words; v[h] holds element 2h in its
// low half and 2h+1 in its high half.
// ---------------------------------------------------------------------------------------------

// Block scale byte for a block amax, by TE's formula, as its raw ScaleType bits.
//
// core::compute_decoding_scaling_factor clamps the product to FLT_MAX before the saturating
// conversion.  For E4M3 the clamp cannot change the result here: block_amax is an abs-max of
// BF16 values (max.xorsign.abs drops NaN operands, so it is never NaN) and S_enc / 6 is finite
// and positive, so the product is never NaN, and the one value the clamp could change, +inf,
// converts to the same saturated 448 that FLT_MAX does.  CLAMP = false drops that instruction
// from the issue-bound specializations; UE5M3 always keeps it.
template <typename ScaleType, bool CLAMP = true>
__device__ __forceinline__ uint32_t scale_code(float block_amax, float S_enc) {
  if constexpr (CLAMP || !std::is_same_v<ScaleType, fp8e4m3>) {
    return core::compute_decoding_scaling_factor<ScaleType>(block_amax, S_enc).__x;
  } else {
    constexpr float fp4_max_inv = 1.0f / detail::TypeExtrema<fp4e2m1>::max;
    return static_cast<fp8e4m3>(block_amax * (S_enc * fp4_max_inv)).__x;
  }
}

template <typename ScaleType>
__device__ __forceinline__ ScaleType scale_from_code(uint32_t code) {
  ScaleType s;
  s.__x = static_cast<__nv_fp8_storage_t>(code);
  return s;
}

template <typename ScaleType, bool CLAMP>
__device__ __forceinline__ uint32_t row_block_scale_code(const uint32_t (&v)[8], float S_enc) {
  uint32_t m0 = ptx::abs_max_bf16x2(v[0], v[1]);
  uint32_t m1 = ptx::abs_max_bf16x2(v[2], v[3]);
  uint32_t m2 = ptx::abs_max_bf16x2(v[4], v[5]);
  uint32_t m3 = ptx::abs_max_bf16x2(v[6], v[7]);
  m0 = ptx::abs_max_bf16x2(m0, m1);
  m2 = ptx::abs_max_bf16x2(m2, m3);
  m0 = ptx::abs_max_bf16x2(m0, m2);
  // Fold the two lanes with the same unit: swap them and take one more abs-max.
  const float block_amax =
      __uint_as_float(ptx::abs_max_bf16x2(m0, ptx::swap_halves_b32(m0)) & 0x7fff0000u);
  return scale_code<ScaleType, CLAMP>(block_amax, S_enc);
}

template <bool STOCHASTIC>
__device__ __forceinline__ uint32_t pack_half(const float (&f)[8], uint32_t r0, uint32_t r1) {
  if constexpr (STOCHASTIC) {
    return ptx::cvt_f32x8_to_fp4x8_rs(f, r0, r1);
  } else {
    return ptx::cvt_f32x8_to_fp4x8_rn(f);
  }
}

// Scale and pack one rowwise block, half a block at a time to keep 8 floats live, not 16.
template <bool STOCHASTIC>
__device__ __forceinline__ void pack_row_block(const uint32_t (&v)[8], uint64_t coeff2,
                                               const uint4 &rnd, uint32_t &w0, uint32_t &w1) {
  float f[8];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    ptx::unpack_f32x2(ptx::mul_f32x2(ptx::bf16x2_to_f32x2(v[h]), coeff2), f[2 * h], f[2 * h + 1]);
  }
  w0 = pack_half<STOCHASTIC>(f, rnd.x, rnd.y);
#pragma unroll
  for (int h = 4; h < 8; ++h) {
    ptx::unpack_f32x2(ptx::mul_f32x2(ptx::bf16x2_to_f32x2(v[h]), coeff2), f[2 * h - 8],
                      f[2 * h - 7]);
  }
  w1 = pack_half<STOCHASTIC>(f, rnd.z, rnd.w);
}

// Two rows of one column, scaled.  LOW selects the even column of each packed pair.
template <bool LOW>
__device__ __forceinline__ void scale_column_pair(uint32_t a, uint32_t b, uint64_t coeff2,
                                                  float &lo, float &hi) {
  const uint32_t ea = LOW ? (a << 16) : (a & 0xffff0000u);
  const uint32_t eb = LOW ? (b << 16) : (b & 0xffff0000u);
  ptx::unpack_f32x2(ptx::mul_f32x2((static_cast<uint64_t>(eb) << 32) | ea, coeff2), lo, hi);
}

// Columnwise twin of pack_row_block: col[t] is row t of a packed column pair.
template <bool STOCHASTIC, bool LOW>
__device__ __forceinline__ void pack_column_block(const uint32_t (&col)[16], uint64_t coeff2,
                                                  const uint4 &rnd, uint32_t &w0, uint32_t &w1) {
  float f[8];
#pragma unroll
  for (int h = 0; h < 4; ++h) {
    scale_column_pair<LOW>(col[2 * h], col[2 * h + 1], coeff2, f[2 * h], f[2 * h + 1]);
  }
  w0 = pack_half<STOCHASTIC>(f, rnd.x, rnd.y);
#pragma unroll
  for (int h = 4; h < 8; ++h) {
    scale_column_pair<LOW>(col[2 * h], col[2 * h + 1], coeff2, f[2 * h - 8], f[2 * h - 7]);
  }
  w1 = pack_half<STOCHASTIC>(f, rnd.z, rnd.w);
}

// Element coefficient for scale code `code`: compute_nvfp4_scaling_coefficient<float>'s
// formula, fminf(1 / (scale * (1 / S_enc)), FLT_MAX), with the inner reciprocal S_dec = 1 / S_enc
// computed once by the caller.  __frcp_rn is the correctly rounded reciprocal, so it equals
// the IEEE quotient 1.0f / x bit for bit.
template <typename ScaleType>
__device__ __forceinline__ float coefficient_of(uint32_t code, float S_dec) {
  const float scale = static_cast<float>(scale_from_code<ScaleType>(code));
  return fminf(__frcp_rn(scale * S_dec), detail::TypeExtrema<float>::max);
}

// Coefficient table entry.  Rowwise RNE consumes every coefficient as a packed pair, so it
// stores that form once; the others keep the narrower float (a 64-bit shared load cost the SR
// specialization more than the move it saves).
template <bool COLWISE, bool STOCHASTIC>
using CoeffEntry = std::conditional_t<!COLWISE && !STOCHASTIC, uint64_t, float>;

template <typename Entry>
__device__ __forceinline__ uint64_t coeff2_of(Entry e) {
  if constexpr (std::is_same_v<Entry, uint64_t>) {
    return e;
  } else {
    return ptx::splat_f32x2(e);
  }
}

// Fill the per-CTA coefficient tables: [0, kScaleCodes) rowwise and, for the fused
// specialization, [kScaleCodes, 2 * kScaleCodes) columnwise.  No barrier; callers place it.
// The trip count is a compile-time constant: a runtime-bounded loop around the divides made
// ptxas spill the already-loaded input across the divide's slow-path call.
template <typename ScaleType, bool COLWISE, bool STOCHASTIC, int THREADS>
__device__ __forceinline__ void write_coeff_tables(CoeffEntry<COLWISE, STOCHASTIC> *table,
                                                   float S_enc_row, float S_enc_col,
                                                   bool separate_col) {
  constexpr int kCodes = kScaleCodes<ScaleType>;
  static_assert(kCodes % THREADS == 0, "Each thread fills the same number of entries.");
  const float S_dec_row = __frcp_rn(S_enc_row);
  const float S_dec_col = separate_col ? __frcp_rn(S_enc_col) : S_dec_row;
#pragma unroll
  for (int k = 0; k < kCodes / THREADS; ++k) {
    const int code = threadIdx.x + k * THREADS;
    const float row_coeff = coefficient_of<ScaleType>(code, S_dec_row);
    if constexpr (COLWISE) {
      table[code] = row_coeff;
      table[kCodes + code] = separate_col ? coefficient_of<ScaleType>(code, S_dec_col) : row_coeff;
    } else if constexpr (!STOCHASTIC) {
      table[code] = ptx::splat_f32x2(row_coeff);
    } else {
      table[code] = row_coeff;
    }
  }
}

template <typename ScaleType, bool COLWISE, bool STOCHASTIC, int GRP, int ROWS>
__global__ void __launch_bounds__(COLWISE ? TunedConfig::kColThreads : TunedConfig::kRowThreads,
                                  TunedConfig::min_ctas_per_sm<COLWISE, STOCHASTIC>())
    quantize_transpose_nvfp4_register_1D_kernel(
        const uint16_t *__restrict__ x, const float *__restrict__ noop,
        const float *__restrict__ amax_row_ptr, const float *__restrict__ amax_col_ptr,
        const size_t *__restrict__ rng_state, uint8_t *__restrict__ q_row,
        uint8_t *__restrict__ s_row, uint8_t *__restrict__ q_col, uint8_t *__restrict__ s_col,
        const int M, const int K, const int s_row_pitch, const int s_col_pitch, const int nsubs) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int kCodes = kScaleCodes<ScaleType>;
  const int bpr = K >> 4;  // rowwise blocks per row (16 elements each)

  using Entry = CoeffEntry<COLWISE, STOCHASTIC>;
  __shared__ Entry coeff_table[COLWISE ? 2 * kCodes : kCodes];

  // Launch-wide scalars are READ here but consumed further down.  Each heads a dependency
  // chain off a global load (the encode-scale divides, the Philox key schedule), and a warp
  // issues in order, so only reading them here lets ptxas keep the loads at the top while the
  // arithmetic sinks below the first input loads, overlapping the two latencies.
  const bool skip = noop != nullptr && noop[0] == 1.0f;
  const float amax_row = amax_row_ptr != nullptr ? *amax_row_ptr : 0.0f;
  const float amax_col = (COLWISE && amax_col_ptr != nullptr) ? *amax_col_ptr : 0.0f;
  uint64_t rng_seed = 0, rng_offset = 0;
  if constexpr (STOCHASTIC) {
    if (rng_state != nullptr) {
      rng_seed = rng_state[0];
      rng_offset = rng_state[1];
    }
  }
  // Same rules as quantize_transpose_tuned_1D: no rowwise amax means S_enc = 1, no columnwise
  // amax means the columnwise output reuses the rowwise S_enc.
  auto encode_scales = [&](float &S_enc_row, float &S_enc_col) {
    S_enc_row = amax_row_ptr != nullptr
                    ? core::compute_global_encode_scaling_factor_FP4<ScaleType>(amax_row)
                    : 1.0f;
    // The usual caller passes the same global amax in both directions; recognizing that from
    // the raw bits skips a second divide chain in the CTA prologue.
    const bool same_amax =
        amax_row_ptr != nullptr && __float_as_uint(amax_col) == __float_as_uint(amax_row);
    S_enc_col = (COLWISE && amax_col_ptr != nullptr && !same_amax)
                    ? core::compute_global_encode_scaling_factor_FP4<ScaleType>(amax_col)
                    : S_enc_row;
  };

  if constexpr (!COLWISE) {
    // ---- rowwise only: pure streaming, one block per thread per row ----
    constexpr int kRowWave = TunedConfig::kRowWave;
    const int j0 = blockIdx.x * TunedConfig::kRowBlocks + threadIdx.x;
    const int j = j0 < bpr ? j0 : 0;  // clamp so the prefetch below stays in bounds
    const int r0 = blockIdx.y * ROWS;

    // The column term of all three addresses is a thread constant, folded into the base
    // pointers once, so each per-block address is one row_off() from them.  (Induction
    // pointers instead cost twelve live registers here, in the one register-bound
    // specialization, and measured 6% slower.)
    const uint16_t *const xj = x + static_cast<unsigned>(j) * kBlockElems;
    uint8_t *const qj = q_row + static_cast<unsigned>(j) * 8;
    uint8_t *const sj = s_row + static_cast<unsigned>(j);

    uint32_t v[kRowWave][8];
#pragma unroll
    for (int i = 0; i < kRowWave; ++i) {
      ptx::ld_global_nc_evict_first_b32x8_relaxed(v[i], xj + row_off(r0 + i, K));
    }
    if (skip) return;  // CTA-uniform, and no barrier has been reached yet

    // Resolve the RNG first: its offset-dependent multiply has the longer wait, and the
    // already-issued loads continue toward readiness behind it.  The block column is a thread
    // constant, so placing it in the subsequence's low word folds rounds 0..2 away.
    block_philox::PhiloxFold pf{};
    if constexpr (STOCHASTIC) {
      pf = block_philox::make_fold(block_philox::make_philox(rng_seed, rng_offset),
                                   block_philox::rowwise_sub_lo(j));
    }

    float S_enc_row, S_enc_col;
    encode_scales(S_enc_row, S_enc_col);

    // Written here; its barrier is deferred into the first wave, past that wave's amax trees,
    // which depend only on the input already in flight.
    write_coeff_tables<ScaleType, false, STOCHASTIC, TunedConfig::kRowThreads>(
        coeff_table, S_enc_row, S_enc_col, false);

#pragma unroll
    for (int w = 0; w < ROWS; w += kRowWave) {
      if (w) {
#pragma unroll
        for (int i = 0; i < kRowWave; ++i) {
          ptx::ld_global_nc_evict_first_b32x8_relaxed(v[i], xj + row_off(r0 + w + i, K));
        }
      }
      // Register-free L2 warm-up for the next wave.  The rows are inside this CTA's stripe
      // (ROWS divides 32 and M % 32 == 0), so no bounds test is needed.
      if (w + kRowWave < ROWS) {
#pragma unroll
        for (int i = 0; i < kRowWave; ++i) {
          ptx::prefetch_l2(xj + row_off(r0 + w + kRowWave + i, K));
        }
      }
      // The wave's first random pair needs none of the loaded data, so draw it before the
      // amax trees that stall on the loads.  Only the first pair: this specialization is
      // register-bound, so the rest stay next to their packs.
      uint4 rnd0[2] = {};
      if constexpr (STOCHASTIC) {
        uint32_t rows0[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) rows0[i] = block_philox::rowwise_sub_hi(r0 + w + i);
        block_philox::philox_fold_n<2>(rows0, pf, rnd0);
      }
      // Every block's amax tree and scale byte resolve before any of them packs.
      uint32_t codes[kRowWave];
#pragma unroll
      for (int i = 0; i < kRowWave; ++i) {
        codes[i] = row_block_scale_code<ScaleType, !STOCHASTIC>(v[i], S_enc_row);
      }
      if (w == 0) {
        __syncthreads();        // the coefficient table is read only from here on
        if (j0 >= bpr) return;  // no barrier follows, so a split warp is safe
      }
      if constexpr (!STOCHASTIC) {
        uint64_t coeff2[kRowWave];
#pragma unroll
        for (int i = 0; i < kRowWave; ++i) coeff2[i] = coeff_table[codes[i]];
#pragma unroll
        for (int i = 0; i < kRowWave; ++i) {
          const int row = r0 + w + i;
          uint32_t w0, w1;
          pack_row_block<false>(v[i], coeff2[i], make_uint4(0, 0, 0, 0), w0, w1);
          *reinterpret_cast<uint2 *>(qj + row_off(row, K >> 1)) = make_uint2(w0, w1);
          sj[row_off(row, s_row_pitch)] = static_cast<uint8_t>(codes[i]);
        }
      } else {
#pragma unroll
        for (int g = 0; g < kRowWave; g += 2) {
          uint64_t coeff2[2];
#pragma unroll
          for (int i = 0; i < 2; ++i) coeff2[i] = coeff2_of(coeff_table[codes[g + i]]);
          uint4 rnd[2];
          if (g == 0) {  // drawn above, ahead of the amax trees
            rnd[0] = rnd0[0];
            rnd[1] = rnd0[1];
          } else {
            uint32_t rows[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) rows[i] = block_philox::rowwise_sub_hi(r0 + w + g + i);
            block_philox::philox_fold_n<2>(rows, pf, rnd);
          }
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int row = r0 + w + g + i;
            uint32_t w0, w1;
            pack_row_block<true>(v[g + i], coeff2[i], rnd[i], w0, w1);
            *reinterpret_cast<uint2 *>(qj + row_off(row, K >> 1)) = make_uint2(w0, w1);
            sj[row_off(row, s_row_pitch)] = static_cast<uint8_t>(codes[g + i]);
          }
        }
      }
    }
  } else {
    // ---- rowwise + columnwise ----
    constexpr int kTileK = TunedConfig::kTileK;
    constexpr int kSlabM = TunedConfig::kSlabM;
    // Transposed result, column-major: staging[column slot][word].
    __shared__ uint32_t staging[kTileK][kStagingWords];
    __shared__ uint8_t staging_scales[kTileK][kStagingScaleBytes];

    const int c0 = blockIdx.x * kTileK;
    const int r00 = blockIdx.y * (nsubs * kSlabM);

    // Phase A lanes: kJA block columns per row x kRgA rows per pass, from the GRP-wide strip
    // this CTA belongs to (GRP == 1 is the plain per-tile mapping).
    constexpr int kJA = 8 * GRP;
    constexpr int kRgA = TunedConfig::kColThreads / kJA;
    constexpr int kRowsA = kSlabM / GRP;
    const int qa = static_cast<int>(blockIdx.x) & (GRP - 1);  // position inside the strip
    const int jbase = (static_cast<int>(blockIdx.x) - qa) * (kTileK / kBlockElems);
    const int jl = threadIdx.x & (kJA - 1);
    const int rl = threadIdx.x / kJA;
    const int j0 = jbase + jl;
    const int j = j0 < bpr ? j0 : jbase;
    // Phase B lanes: four adjacent columns per thread (a warp spans 128 columns, one 256-byte
    // request per row) and one of the slab's four column blocks.
    const int cp = threadIdx.x & 31;
    const int mb = threadIdx.x >> 5;
    const int c = c0 + 4 * cp;
    const bool cok = c < K;
    // A band of at most 32 rows that starts inside the matrix also ends inside it (M % 32 ==
    // 0), so clamping the band base is all the bounds work phase A's row walk needs.
    constexpr bool kBandSafe = kRowsA <= 32;
    const bool arow = j0 < bpr;
    // 32-bit row strides: ptxas folds a 32-bit pointer step into one IMAD.WIDE.
    const int xastep = kRgA * K, qastep = kRgA * (K >> 1), sastep = kRgA * s_row_pitch;
    const size_t xcol = static_cast<unsigned>(j * kBlockElems);
    const size_t qcol = static_cast<unsigned>(j * 8);

    // The first real load is inside the slab loop, so it cannot be hoisted above the
    // encode-scale chain; a prefetch can, and warms the first slab meanwhile.
    {
      const int pb = r00 + kRowsA * qa + rl;
      const uint16_t *pf =
          x +
          (static_cast<size_t>(kBandSafe && pb < M ? pb : r00) * static_cast<unsigned>(K) + xcol);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if constexpr (kBandSafe) {
          ptx::prefetch_l2(pf);
          pf += xastep;
        } else {
          const int r = pb + kRgA * i;
          ptx::prefetch_l2(
              x + (static_cast<size_t>(r < M ? r : r00) * static_cast<unsigned>(K) + xcol));
        }
      }
    }
    if (skip) return;  // CTA-uniform, and no barrier has been reached yet

    float S_enc_row, S_enc_col;
    encode_scales(S_enc_row, S_enc_col);
    // Equal encode scales (the usual case: one global amax for both directions) share the
    // rowwise table instead of evaluating a second set of divides.
    const bool separate_col = __float_as_uint(S_enc_row) != __float_as_uint(S_enc_col);
    // Both halves fold Philox rounds 0..2 against a low word that is constant for the CTA: the
    // rowwise half keys on its block column, the columnwise half on the base of its four
    // columns, with the column's offset in the quad folded into the high word.
    block_philox::PhiloxFold pf_r{}, pf_c{};
    if constexpr (STOCHASTIC) {
      const block_philox::PhiloxState ps = block_philox::make_philox(rng_seed, rng_offset);
      pf_r = block_philox::make_fold(ps, block_philox::rowwise_sub_lo(j));
      pf_c = block_philox::make_fold(ps, block_philox::columnwise_sub_lo(c));
    }

    write_coeff_tables<ScaleType, true, STOCHASTIC, TunedConfig::kColThreads>(
        coeff_table, S_enc_row, S_enc_col, separate_col);
    __syncthreads();

    for (int sub = 0; sub < nsubs; ++sub) {
      const int r0 = r00 + sub * kSlabM;
      if (r0 >= M) break;

      // Phase A: one rowwise block per thread per quarter-slab, all three streams advancing by
      // a constant row stride.  A full slab needs no per-row bounds test.
      const int rbase = r0 + kRowsA * qa + rl;
      const bool afull = arow && (r0 + kSlabM <= M);
      const uint16_t *xsafe = x + (static_cast<size_t>(r0) * static_cast<unsigned>(K) + xcol);
      const uint16_t *xr = kBandSafe && rbase < M
                               ? x + (static_cast<size_t>(rbase) * static_cast<unsigned>(K) + xcol)
                               : xsafe;
      int rows[4];
      uint32_t v[4][8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = rbase + kRgA * i;
        rows[i] = r;
        if constexpr (kBandSafe) {
          ptx::ld_global_evict_last_b32x8(v[i], xr);
        } else {
          ptx::ld_global_evict_last_b32x8(
              v[i], r < M ? x + (static_cast<size_t>(r) * static_cast<unsigned>(K) + xcol) : xsafe);
        }
        xr += xastep;
      }

      // Warm L2 for the next slab under SR only: RNE streams near its DRAM ceiling and a
      // lookahead competes with demand loads, while SR has the slack for it to land.
      if constexpr (STOCHASTIC) {
        if (sub + 1 < nsubs && r0 + 2 * kSlabM <= M) {
          const uint16_t *pn =
              x + (static_cast<size_t>(rbase + kSlabM) * static_cast<unsigned>(K) + xcol);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            ptx::prefetch_l2(pn);
            pn += xastep;
          }
        }
      }

      // The Philox chain depends only on the row, never on the data, so it goes before the
      // amax trees that stall on the loads.
      uint4 rnd_row[4] = {};
      if constexpr (STOCHASTIC) {
#pragma unroll
        for (int g = 0; g < 4; g += 2) {
          const uint32_t hi[2] = {block_philox::rowwise_sub_hi(rows[g]),
                                  block_philox::rowwise_sub_hi(rows[g + 1])};
          uint4 out[2];
          block_philox::philox_fold_n<2>(hi, pf_r, out);
          rnd_row[g] = out[0];
          rnd_row[g + 1] = out[1];
        }
      }
      uint32_t codes_a[4];
      uint64_t coeff2_a[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        codes_a[i] = row_block_scale_code<ScaleType, false>(v[i], S_enc_row);
#pragma unroll
      for (int i = 0; i < 4; ++i) coeff2_a[i] = coeff2_of(coeff_table[codes_a[i]]);
      uint8_t *qap = q_row + (static_cast<size_t>(rbase) * static_cast<unsigned>(K >> 1) + qcol);
      uint8_t *sap = s_row + (static_cast<size_t>(rbase) * static_cast<unsigned>(s_row_pitch) +
                              static_cast<unsigned>(j));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (afull || !(rows[i] >= M || !arow)) {
          uint32_t w0, w1;
          pack_row_block<STOCHASTIC>(v[i], coeff2_a[i], rnd_row[i], w0, w1);
          *reinterpret_cast<uint2 *>(qap) = make_uint2(w0, w1);
          *sap = static_cast<uint8_t>(codes_a[i]);
        }
        qap += qastep;
        sap += sastep;
      }

      // Phase B: the rows phase A just read are still in L2, so this column-major pass is a
      // cache re-read rather than a second DRAM stream.
      {
        const int rb = r0 + mb * kBlockElems;
        const bool bok = cok && (rb + kBlockElems <= M);
        const int cc = bok ? c : 0;
        const int rr = bok ? rb : r0;
        uint32_t raw[2][16];
        {
          const uint16_t *cp2 =
              x + (static_cast<size_t>(rr) * static_cast<unsigned>(K) + static_cast<unsigned>(cc));
#pragma unroll
          for (int t = 0; t < 16; ++t) {
            const uint2 rv = *reinterpret_cast<const uint2 *>(cp2);
            raw[0][t] = rv.x;
            raw[1][t] = rv.y;
            cp2 += K;
          }
        }

        const int wofs = (sub * kBlocksPerSlab + mb) * 2;
        const int bofs = sub * kBlocksPerSlab + mb;
        // High word = (row block, column inside the quad); the quad base is in the folded
        // low word, so the four columns differ only in two bits.
        uint4 rnd[4] = {};
        if constexpr (STOCHASTIC) {
          const uint32_t row_block = static_cast<uint32_t>(rb) >> 4;
#pragma unroll
          for (int g = 0; g < 4; g += 2) {
            const uint32_t hi[2] = {block_philox::columnwise_sub_hi(row_block, g),
                                    block_philox::columnwise_sub_hi(row_block, g + 1)};
            uint4 out[2];
            block_philox::philox_fold_n<2>(hi, pf_c, out);
            rnd[g] = out[0];
            rnd[g + 1] = out[1];
          }
        }
        // Both 16-row amax trees and all four coefficient lookups resolve before any column
        // packs, so the four shared loads are in flight together.
        uint32_t codes_b[4];
        uint64_t coeff2_b[4];
#pragma unroll
        for (int p = 0; p < 2; ++p) {  // one packed word holds a pair of adjacent columns
          uint32_t tr[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) tr[i] = ptx::abs_max_bf16x2(raw[p][2 * i], raw[p][2 * i + 1]);
#pragma unroll
          for (int i = 0; i < 4; ++i) tr[i] = ptx::abs_max_bf16x2(tr[2 * i], tr[2 * i + 1]);
          tr[0] = ptx::abs_max_bf16x2(tr[0], tr[1]);
          tr[1] = ptx::abs_max_bf16x2(tr[2], tr[3]);
          const uint32_t pair_amax = ptx::abs_max_bf16x2(tr[0], tr[1]) & 0x7fff7fffu;
          codes_b[2 * p] =
              scale_code<ScaleType, false>(__uint_as_float(pair_amax << 16), S_enc_col);
          codes_b[2 * p + 1] =
              scale_code<ScaleType, false>(__uint_as_float(pair_amax & 0xffff0000u), S_enc_col);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) coeff2_b[k] = coeff2_of(coeff_table[kCodes + codes_b[k]]);
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int cs = 2 * p + q;
            uint32_t w0, w1;
            if (q == 0) {
              pack_column_block<STOCHASTIC, true>(raw[p], coeff2_b[cs], rnd[cs], w0, w1);
            } else {
              pack_column_block<STOCHASTIC, false>(raw[p], coeff2_b[cs], rnd[cs], w0, w1);
            }
            const int slot = (cs << 5) | cp;  // one 8-byte staging store per column block
            *reinterpret_cast<uint2 *>(&staging[slot][wofs]) = make_uint2(w0, w1);
            staging_scales[slot][bofs] = static_cast<uint8_t>(codes_b[cs]);
          }
        }
      }
    }

    __syncthreads();

    // Phase C: four lanes cooperate on one column, each writing a whole 32-byte sector, so a
    // column's transposed run leaves as one coalesced request.
    {
      const int nb = min(nsubs * kBlocksPerSlab, (M - r00) >> 4);  // valid blocks
      const int ll = threadIdx.x & 3;
      const int cgrp = threadIdx.x >> 2;  // this lane group's column quad
      const int cbase = c0 + 4 * cgrp;
      const int nblk = nb - 4 * ll;  // payload blocks this lane owns (always even)
      const size_t qrow = static_cast<size_t>(M >> 1);
      const bool wide = (qrow & 31) == 0;  // 32-byte alignment of a transposed row
      uint8_t *qp = q_col + static_cast<size_t>(cbase) * qrow + (r00 >> 1) + 32 * ll;
      uint8_t *sp = s_col + static_cast<size_t>(cbase) * s_col_pitch + (r00 >> 4) + 4 * ll;
      int slot = cgrp;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (cbase + q < K && nblk > 0) {
          uint32_t d[8];
#pragma unroll
          for (int t = 0; t < 4; ++t) {
            const uint2 a = *reinterpret_cast<const uint2 *>(&staging[slot][8 * ll + 2 * t]);
            d[2 * t] = a.x;
            d[2 * t + 1] = a.y;
          }
          if (nblk >= 4 && wide) {
            ptx::st_global_b32x8(qp, d);  // one 32-byte sector
          } else if (nblk >= 4) {         // rows are only 16-byte aligned when M % 64 != 0
            *reinterpret_cast<uint4 *>(qp) = make_uint4(d[0], d[1], d[2], d[3]);
            *reinterpret_cast<uint4 *>(qp + 16) = make_uint4(d[4], d[5], d[6], d[7]);
          } else {  // tail slab: this lane owns exactly two blocks
            *reinterpret_cast<uint4 *>(qp) = make_uint4(d[0], d[1], d[2], d[3]);
          }
          const uint32_t sw = *reinterpret_cast<const uint32_t *>(&staging_scales[slot][4 * ll]);
          if (nblk >= 4) {
            *reinterpret_cast<uint32_t *>(sp) = sw;
          } else {
            *reinterpret_cast<uint16_t *>(sp) = static_cast<uint16_t>(sw);
          }
        }
        slot += 32;
        qp += qrow;
        sp += s_col_pitch;
      }
    }
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// Resident CTAs on the current device for one instantiation, from the driver.  Cached per
// device (and per instantiation), so a process driving several different devices gets each
// device's own value.
template <typename ScaleType, bool COLWISE, bool STOCHASTIC, int GRP, int ROWS>
int resident_ctas() {
  constexpr int threads = COLWISE ? TunedConfig::kColThreads : TunedConfig::kRowThreads;
  static std::vector<int> cache(cuda::num_devices(), 0);
  static std::vector<std::once_flag> flags(cuda::num_devices());
  const int device = cuda::current_device();
  std::call_once(flags[device], [&] {
    int per_sm = 0;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm,
        quantize_transpose_nvfp4_register_1D_kernel<ScaleType, COLWISE, STOCHASTIC, GRP, ROWS>,
        threads, 0));
    cache[device] = std::max(per_sm, 1) * cuda::sm_count(device);
  });
  return cache[device];
}

struct LaunchArgs {
  const uint16_t *x;
  const float *noop;
  const float *amax_row;
  const float *amax_col;
  const size_t *rng_state;
  uint8_t *q_row;
  uint8_t *s_row;
  uint8_t *q_col;
  uint8_t *s_col;
  int M;
  int K;
  int s_row_pitch;
  int s_col_pitch;
};

template <typename ScaleType, bool COLWISE, bool STOCHASTIC, int GRP, int ROWS>
void launch_shape(const LaunchArgs &a, cudaStream_t stream) {
  dim3 grid;
  int nsubs = 1;
  if constexpr (COLWISE) {
    // Taller CTAs make each transposed run longer (the drain costs one request per column
    // whatever the run length) but shrink the grid, and a grid that no longer covers the
    // device several times wastes its tail.  Pick the slab count by
    // waves * (slab streaming work + per-CTA drain).
    const int resident = resident_ctas<ScaleType, COLWISE, STOCHASTIC, GRP, ROWS>();
    const int gx = DIVUP(a.K, TunedConfig::kTileK);
    long long best = -1;
    for (int n = 1; n <= TunedConfig::kMaxSlabs; n <<= 1) {
      const long long gy =
          DIVUP(static_cast<long long>(a.M), static_cast<long long>(n) * TunedConfig::kSlabM);
      const long long g = static_cast<long long>(gx) * gy;
      const long long waves = DIVUP(g, static_cast<long long>(resident));
      const long long filled = std::min<long long>(g, resident);
      const long long cost =
          waves * (TunedConfig::kSlabWork * n + TunedConfig::kDrainWork) * resident / filled;
      if (best < 0 || cost < best) {
        best = cost;
        nsubs = n;
      }
    }
    grid = dim3(gx, DIVUP(a.M, nsubs * TunedConfig::kSlabM));
  } else {
    grid = dim3(DIVUP(a.K / kBlockElems, TunedConfig::kRowBlocks), DIVUP(a.M, ROWS));
  }
  constexpr int threads = COLWISE ? TunedConfig::kColThreads : TunedConfig::kRowThreads;
  quantize_transpose_nvfp4_register_1D_kernel<ScaleType, COLWISE, STOCHASTIC, GRP, ROWS>
      <<<grid, threads, 0, stream>>>(a.x, a.noop, a.amax_row, a.amax_col, a.rng_state, a.q_row,
                                     a.s_row, a.q_col, a.s_col, a.M, a.K, a.s_row_pitch,
                                     a.s_col_pitch, nsubs);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ScaleType, bool COLWISE, bool STOCHASTIC>
void launch(const LaunchArgs &a, cudaStream_t stream) {
  if constexpr (COLWISE) {
    // The strip remap needs every column of a kGroupCTAs-CTA strip to exist.
    if (a.K % (TunedConfig::kGroupCTAs * TunedConfig::kTileK) == 0) {
      launch_shape<ScaleType, true, STOCHASTIC, TunedConfig::kGroupCTAs, 1>(a, stream);
    } else {
      launch_shape<ScaleType, true, STOCHASTIC, 1, 1>(a, stream);
    }
  } else {
    // Under SR the Philox chain, not DRAM, sets the pace, so the per-CTA prologue (amax load,
    // divides, coefficient table, barrier) is a real fraction of each CTA.  Doubling the
    // stripe halves it, worth taking whenever the taller grid still fills the device.
    constexpr int kBase = STOCHASTIC ? TunedConfig::kRowRowsSr : TunedConfig::kRowRowsRne;
    if constexpr (STOCHASTIC) {
      const int gx = DIVUP(a.K / kBlockElems, TunedConfig::kRowBlocks);
      const int resident = resident_ctas<ScaleType, false, true, 1, kBase>();
      if (static_cast<long long>(gx) * DIVUP(a.M, 2 * kBase) >= resident) {
        launch_shape<ScaleType, false, true, 1, 2 * kBase>(a, stream);
        return;
      }
    }
    launch_shape<ScaleType, false, STOCHASTIC, 1, kBase>(a, stream);
  }
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace quantize_transpose_register_kernel

/*! \brief Whether quantize_transpose_register_1D can serve this request.
 *
 *  BF16 input, rowwise output (columnwise optional), compact scales, no row-scaled NVFP4, no
 *  fast math, rows and columns multiples of 32, 32-byte aligned buffers, on an SM 10.x GPU;
 *  stochastic rounding also needs a build with at least 3 Philox rounds.
 *  Everything else stays on quantize_transpose_tuned_1D.
 */
inline bool register_1D_supported(const Tensor &input, const Tensor &output,
                                  const QuantizationConfig *quant_config) {
#if FP4_TYPE_SUPPORTED
  using quantize_transpose_register_kernel::TunedConfig;
  if (cuda::sm_arch() / 10 != 10) return false;
  if (input.dtype() != DType::kBFloat16 || !input.has_data()) return false;
  if (!output.has_data() || output.row_scaled_nvfp4 || output.with_gemm_swizzled_scales) {
    return false;
  }
  if (quant_config != nullptr && quant_config->use_fast_math) return false;
  if (quant_config != nullptr && quant_config->stochastic_rounding &&
      !block_philox::kFoldSupported) {
    return false;
  }

  const auto [rows, cols] = input.flat_2d_dims();
  if (rows == 0 || cols == 0 || rows % 32 != 0 || cols % 32 != 0) return false;
  // 32-bit indexing and grid limits.
  if (cols > static_cast<size_t>(INT_MAX) / 16) return false;
  if (DIVUP(rows, static_cast<size_t>(TunedConfig::kRowRowsRne)) > 65535) return false;

  if (output.scale_inv.shape.size() != 2 || output.scale_inv.shape[1] < cols / 16 ||
      output.scale_inv.shape[1] > static_cast<size_t>(INT_MAX)) {
    return false;
  }
  if (!is_aligned_ptr(input.data.dptr, 32) || !is_aligned_ptr(output.data.dptr, 8)) return false;
  if (output.has_columnwise_data()) {
    const auto &col_scale = output.columnwise_scale_inv;
    if (col_scale.dptr == nullptr || col_scale.shape.size() != 2 ||
        col_scale.shape[1] < DIVUP(rows, static_cast<size_t>(16)) || col_scale.shape[1] % 4 != 0 ||
        col_scale.shape[1] > static_cast<size_t>(INT_MAX)) {
      return false;
    }
    if (!is_aligned_ptr(output.columnwise_data.dptr, 32) || !is_aligned_ptr(col_scale.dptr, 4)) {
      return false;
    }
  }
  return true;
#else
  return false;
#endif  // FP4_TYPE_SUPPORTED
}

/*! \brief Quantize BF16 to NVFP4 1D, rowwise and optionally columnwise, from registers.
 *
 *  Caller must have checked register_1D_supported.
 */
template <typename ScaleType>
void quantize_transpose_register_1D(const Tensor &input, const Tensor *noop, Tensor *output,
                                    const QuantizationConfig *quant_config, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_transpose_register_kernel;

  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  const bool colwise = output->has_columnwise_data();
  if (colwise) {
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Transposed output must have FP4 type.");
  }

  const bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;
  const size_t *rng_state = nullptr;
  const NVTETensor rng_state_tensor = (quant_config != nullptr) ? quant_config->rng_state : nullptr;
  if (rng_state_tensor != nullptr) {
    const Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    NVTE_CHECK(rng_state_te_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_te_tensor.data.shape == Shape{2},
               "Shape of the RNG state should be [2], but got ", rng_state_te_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr);
  }

  const auto [rows, cols] = input.flat_2d_dims();
  LaunchArgs a;
  a.x = reinterpret_cast<const uint16_t *>(input.data.dptr);
  a.noop = reinterpret_cast<const float *>(noop->data.dptr);
  a.amax_row = reinterpret_cast<const float *>(output->amax.dptr);
  a.amax_col = reinterpret_cast<const float *>(output->columnwise_amax.dptr);
  a.rng_state = rng_state;
  a.q_row = reinterpret_cast<uint8_t *>(output->data.dptr);
  a.s_row = reinterpret_cast<uint8_t *>(output->scale_inv.dptr);
  a.q_col = colwise ? reinterpret_cast<uint8_t *>(output->columnwise_data.dptr) : nullptr;
  a.s_col = colwise ? reinterpret_cast<uint8_t *>(output->columnwise_scale_inv.dptr) : nullptr;
  a.M = static_cast<int>(rows);
  a.K = static_cast<int>(cols);
  a.s_row_pitch = static_cast<int>(output->scale_inv.shape[1]);
  a.s_col_pitch = colwise ? static_cast<int>(output->columnwise_scale_inv.shape[1]) : 0;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      colwise, COLWISE,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(use_stochastic_rounding, STOCHASTIC,
                                          launch<ScaleType, COLWISE, STOCHASTIC>(a, stream);););
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_NVFP4_REGISTER_1D_CUH_
