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
 *  partials, the row scales and the dGeLU table.  Cast-only requests have their
 *  own register-resident kernel (specialized/cast_bidim.cuh); anything else this
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
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"

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

/*! \brief Widen the low / high BF16 of a word to FP32. */
__device__ __forceinline__ float bf16_lo(unsigned v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf16_hi(unsigned v) { return __uint_as_float(v & 0xffff0000u); }

/*! \brief The reciprocal of an E8M0 scale, 2^(127-e), broadcast to a BF16 pair. */
__device__ __forceinline__ unsigned e8m0_to_bf16x2_reciprocal(unsigned e) {
  const ptx::bf16x2 r = ptx::exp2f_rcp_2x(static_cast<e8m0_t>(e));
  return reinterpret_cast<const unsigned&>(r);
}

/*! \brief As e8m0_to_bf16x2_reciprocal, for a word holding one scale per half.
 *
 * e = 254 needs the subnormal 2^-127, as ptx::exp2f_rcp_2x gives, so that an
 * Inf saturates to +-448 instead of becoming NaN.  e = 255 does not occur: the
 * amax is clamped by amax_to_e8m0_2x.
 */
__device__ __forceinline__ unsigned e8m0x2_to_bf16x2_reciprocal(unsigned e2) {
  return ((0x00FE00FEu - e2) << 7) | (__vcmpeq2(e2, 0x00FE00FEu) & 0x00400040u);
}

/*! \brief E8M0 scales for two block amaxes, packed as ptx::float_to_e8m0_2x.
 *
 * The generic kernel's amax is a 0-seeded max that ignores NaN, so a block of
 * only NaNs gets amax 0; the fmaxf reproduces that for the data-seeded maxima
 * here.
 */
__device__ __forceinline__ unsigned amax_to_e8m0_2x(float amax_hi, float amax_lo) {
  return ptx::float_to_e8m0_2x(fmaxf(amax_hi * (1.0f / 448.0f), 0.0f),
                               fmaxf(amax_lo * (1.0f / 448.0f), 0.0f));
}

// The activations follow util/math.h's operation order, with the FMA
// contractions nvcc applies to it under -fmad=true written out as fmaf and _rn
// intrinsics, so they round the same way as the generic kernel.
__device__ __forceinline__ float act_dgelu(float v) {
  float a3 = fmaf(__fmul_rn(0.044715f, v), v, 1.0f);
  float a5 = __fmul_rn(__fmul_rn(0.79788456f, v), a3);
  float t = tanhf(a5);
  float c2 = fmaf(-t, t, 1.0f);
  float d3 = fmaf(__fmul_rn(0.1070322243f, v), v, 0.79788456f);
  float f = __fmul_rn(__fmul_rn(0.5f, v), __fmul_rn(c2, d3));
  float h = __fmul_rn(0.5f, __fadd_rn(1.0f, t));
  return __fadd_rn(f, h);
}

// A packed FP32 pair as the raw 64-bit register the .f32x2 instructions take.
using f32x2 = unsigned long long;

__device__ __forceinline__ f32x2 make_f32x2(float lo, float hi) {
  f32x2 d;
  asm("mov.b64 %0, {%1, %2};" : "=l"(d) : "f"(lo), "f"(hi));
  return d;
}
__device__ __forceinline__ void unpack_f32x2(f32x2 a, float& lo, float& hi) {
  asm("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(a));
}
__device__ __forceinline__ f32x2 splat_f32x2(float c) { return make_f32x2(c, c); }

#define NVTE_REGTILE_F32X2_OP(NAME, PTX_NAME)                                       \
  __device__ __forceinline__ f32x2 NAME(f32x2 a, f32x2 b) {                         \
    const ptx::floatx2 d = ptx::PTX_NAME(reinterpret_cast<const ptx::floatx2&>(a),  \
                                         reinterpret_cast<const ptx::floatx2&>(b)); \
    return reinterpret_cast<const f32x2&>(d);                                       \
  }
NVTE_REGTILE_F32X2_OP(add_f32x2, add_2x)
NVTE_REGTILE_F32X2_OP(mul_f32x2, mul_2x)
#undef NVTE_REGTILE_F32X2_OP

__device__ __forceinline__ f32x2 fma_f32x2(f32x2 a, f32x2 b, f32x2 c) {
  const ptx::floatx2 d = ptx::fma_2x(reinterpret_cast<const ptx::floatx2&>(a),
                                     reinterpret_cast<const ptx::floatx2&>(b),
                                     reinterpret_cast<const ptx::floatx2&>(c));
  return reinterpret_cast<const f32x2&>(d);
}

/*! \brief Widen both halves of a BF16 pair to a packed FP32 pair. */
__device__ __forceinline__ f32x2 bf16x2_to_f32x2(unsigned a) {
  return make_f32x2(bf16_lo(a), bf16_hi(a));
}

// Packed tanh, following libdevice tanhf step for step so it rounds the same.

//! 2 * log2(e), the argument scale of the exponential branch.
constexpr float kTanhLog2eX2 = 0x1.715476p+1f;
//! Coefficients of the |x| < 0.6 minimax polynomial, in Horner order.
constexpr float kTanhPoly4 = 0x1.01e104p-6f;
constexpr float kTanhPoly3 = -0x1.ac795cp-5f;
constexpr float kTanhPoly2 = 0x1.10b282p-3f;
constexpr float kTanhPoly1 = -0x1.5553dap-2f;
//! The branch threshold 0.6f, squared.  0.6f * 0.6f rounds exactly to 0.36f, so
//! the predicate can be tested on the already-computed square.
constexpr float kTanhBranchXSq = 0x1.70a3d8p-2f;

/*! \brief copysign(a, b) where a is known non-negative, as one LOP3. */
__device__ __forceinline__ float copysign_nonneg(float a, float b) {
  unsigned d;
  asm("lop3.b32 %0, %1, %2, 0x80000000, 0xec;"
      : "=r"(d)
      : "r"(__float_as_uint(b)), "r"(__float_as_uint(a)));
  return __uint_as_float(d);
}

__device__ __forceinline__ f32x2 tanh_f32x2(f32x2 u) {
  float ua, ub;
  unpack_f32x2(u, ua, ub);
  float e0, e1;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e0) : "f"(__fmul_rn(fabsf(ua), kTanhLog2eX2)));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(e1) : "f"(__fmul_rn(fabsf(ub), kTanhLog2eX2)));
  float f0, f1;
  unpack_f32x2(add_f32x2(make_f32x2(e0, e1), splat_f32x2(1.0f)), f0, f1);
  float r0, r1;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r0) : "f"(f0));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r1) : "f"(f1));
  // libdevice clamps |x| >= 9.010914 to exactly 1.0; there 2*rcp(ex2(2ln2|x|))
  // is already below half an ulp of 1.0f, so fma(r,-2,1) rounds to 1.0f on its
  // own and the clamp select is dropped.
  float g0, g1;
  unpack_f32x2(fma_f32x2(make_f32x2(r0, r1), splat_f32x2(-2.0f), splat_f32x2(1.0f)), g0, g1);
  const float xa = copysign_nonneg(g0, ua);
  const float xb = copysign_nonneg(g1, ub);
  const f32x2 s = mul_f32x2(u, u);
  f32x2 p = fma_f32x2(s, splat_f32x2(kTanhPoly4), splat_f32x2(kTanhPoly3));
  p = fma_f32x2(p, s, splat_f32x2(kTanhPoly2));
  p = fma_f32x2(p, s, splat_f32x2(kTanhPoly1));
  p = mul_f32x2(p, s);
  p = fma_f32x2(p, u, u);
  float pa, pb, sa, sb;
  unpack_f32x2(p, pa, pb);
  unpack_f32x2(s, sa, sb);
  return make_f32x2(sa >= kTanhBranchXSq ? xa : pa, sb >= kTanhBranchXSq ? xb : pb);
}

/*! \brief GeLU on a packed pair, matching util/math.h gelu. */
__device__ __forceinline__ f32x2 gelu_f32x2(f32x2 v) {
  const f32x2 u =
      mul_f32x2(v, fma_f32x2(mul_f32x2(splat_f32x2(0.03567741f), v), v, splat_f32x2(0.79788456f)));
  const f32x2 t = tanh_f32x2(u);
  // 0.5f * t is exact, so this fma rounds like 0.5f + 0.5f * t.
  return mul_f32x2(v, fma_f32x2(splat_f32x2(0.5f), t, splat_f32x2(0.5f)));
}

/*! \brief dGeLU on a packed pair, matching act_dgelu term for term. */
__device__ __forceinline__ f32x2 dgelu_f32x2(f32x2 v) {
  const f32x2 a3 = fma_f32x2(mul_f32x2(splat_f32x2(0.044715f), v), v, splat_f32x2(1.0f));
  const f32x2 a5 = mul_f32x2(mul_f32x2(splat_f32x2(0.79788456f), v), a3);
  const f32x2 t = tanh_f32x2(a5);
  // TE's fma(-t, t, 1), negated: round-to-nearest is sign-symmetric, so
  // fma(t, t, -1) is exactly -(1 - t*t), and the sign folds into the 0.5 below
  // for free instead of costing a negation per pair.
  const f32x2 c2_neg = fma_f32x2(t, t, splat_f32x2(-1.0f));
  const f32x2 d3 = fma_f32x2(mul_f32x2(splat_f32x2(0.1070322243f), v), v, splat_f32x2(0.79788456f));
  const f32x2 f = mul_f32x2(mul_f32x2(splat_f32x2(-0.5f), v), mul_f32x2(c2_neg, d3));
  // 0.5f*(1+t) and fma(0.5,t,0.5) round on the same grid (halving is exact).
  const f32x2 h = fma_f32x2(splat_f32x2(0.5f), t, splat_f32x2(0.5f));
  return add_f32x2(f, h);
}

/*! \brief GeLU on a BF16 pair, returned as a BF16 pair. */
__device__ __forceinline__ unsigned gelu_bf16x2(unsigned x_word) {
  float v0, v1;
  unpack_f32x2(gelu_f32x2(bf16x2_to_f32x2(x_word)), v0, v1);
  return ptx::cvt_bf16x2(v1, v0);
}

// dGeLU lookup table.
//
// The activation input is BF16, so dgelu is a function of a 16-bit key and the
// table holds exactly the FP32 values act_dgelu produces.  Some of each row's
// words (TileConfig::kLutSlots) take the table and the rest the arithmetic body, which
// spreads the work over the shared-memory and FP32/MUFU pipes.
//
// Range reduction keeps the table at 16 KB: magnitudes outside [2^-12, 8] are
// clamped into it, and one "did the clamp move the value" test sends both tails
// to dgelu_outside_window.

//! BF16 bits of 2^-12, the smallest tabulated magnitude.
constexpr unsigned kLutLoBits = 0x3980u;
//! BF16 bits of 8.0, the largest tabulated magnitude.
constexpr unsigned kLutHiBits = 0x4100u;
//! Entries per sign.  kLutHiBits - kLutLoBits + 1 = 1921 of them are reachable.
constexpr int kLutEntriesPerSign = 2048;
//! Both bounds broadcast, for the packed clamp.
constexpr unsigned kLutLoBitsX2 = 0x39803980u;
constexpr unsigned kLutHiBitsX2 = 0x41004100u;
//! Table size.  Reachable FP32 entries end at byte 15875; rounded up to the
//! bulk-copy granularity.
constexpr int kLutBytes = 15888;

__device__ __align__(16) unsigned char d_dgelu_table[kLutBytes];

/*! \brief Byte offsets of both halves of a BF16 pair into the dGeLU table.
 *
 * The sign bit is worth kLutEntriesPerSign four-byte entries, i.e. 0x8000, so
 * it needs no shift and each half stays below 0x10000: the packed add never
 * carries across the pair.
 *
 * \param[in,out] bad  Accumulates every bit the clamp moved, so several probes
 *                     can share one out-of-window test.
 */
__device__ __forceinline__ unsigned lut_byte_offsets(unsigned x_word, unsigned& bad) {
  const unsigned mag = x_word & 0x7fff7fffu;
  const unsigned c = ptx::min_bf16x2(ptx::max_bf16x2(mag, kLutLoBitsX2), kLutHiBitsX2);
  bad |= c ^ mag;
  return ((c - kLutLoBitsX2) << 2) + ((x_word ^ mag) >> 2);
}

/*! \brief Probe the dGeLU table for both halves of a BF16 pair.
 *  \param[out] out_of_window  True if either half fell outside the tabulated window.
 */
__device__ __forceinline__ f32x2 lut_dgelu(unsigned x_word, const unsigned char* __restrict__ table,
                                           bool& out_of_window) {
  unsigned bad = 0u;
  const unsigned d = lut_byte_offsets(x_word, bad);
  out_of_window = bad != 0u;
  return make_f32x2(*(const float*)(table + (d & 0xffffu)), *(const float*)(table + (d >> 16)));
}

/*! \brief dGeLU below the tabulated window.
 *
 * For |v| < 2^-12 tanh collapses to its argument, and this form reproduces
 * dgelu_f32x2's rounding sequence term for term.
 */
__device__ __forceinline__ float dgelu_tiny(float v) {
  const float t = __fmul_rn(0.79788456f, v);
  const float f = __fmul_rn(__fmul_rn(0.5f, v), 0.79788456f);
  return __fadd_rn(f, __fmaf_rn(0.5f, t, 0.5f));
}

//! BF16 bits of the smallest |v| at which util/math.h's dgelu overflows to NaN.
//! Between kLutHiBits and this, it equals the saturated last table entry.
constexpr unsigned kDgeluOverflowBits = 0x6044u;

/*! \brief dGeLU of one BF16 value whose probe may have been clamped.
 *
 * Above the window the clamped entry holds the saturated value, until the
 * formula overflows to NaN; that tail and Inf/NaN take the full formula.
 */
__device__ __forceinline__ float dgelu_outside_window(unsigned bits, float probe) {
  const unsigned mag = bits & 0x7fffu;
  const float v = __uint_as_float(bits << 16);
  if (mag < kLutLoBits) return dgelu_tiny(v);
  if (mag >= kDgeluOverflowBits) return act_dgelu(v);
  return probe;
}

/*! \brief Repair a clamped dGeLU probe. */
__device__ __forceinline__ f32x2 dgelu_lut_tail(unsigned x_word, f32x2 probe) {
  float d0, d1;
  unpack_f32x2(probe, d0, d1);
  return make_f32x2(dgelu_outside_window(x_word & 0xffffu, d0),
                    dgelu_outside_window(x_word >> 16, d1));
}

/*! \brief dGeLU of one BF16 pair, by table or by arithmetic.
 *  \tparam USE_LUT  Which route this (row, word) slot takes; see LutSlot.
 */
template <bool USE_LUT>
__device__ __forceinline__ f32x2 dgelu_word(unsigned x_word,
                                            const unsigned char* __restrict__ table) {
  if constexpr (USE_LUT) {
    bool out_of_window;
    const f32x2 d = lut_dgelu(x_word, table, out_of_window);
    if (__builtin_expect(!out_of_window, 1)) return d;
    return dgelu_lut_tail(x_word, d);
  }
  return dgelu_f32x2(bf16x2_to_f32x2(x_word));
}

/*! \brief Build the dGeLU table.  The contents depend only on the formula. */
__global__ void init_dgelu_table_kernel() {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * (int)(kLutHiBits - kLutLoBits + 1)) return;
  const int sgn = i > (int)(kLutHiBits - kLutLoBits);
  const unsigned mag = kLutLoBits + (unsigned)(sgn ? i - (int)(kLutHiBits - kLutLoBits + 1) : i);
  const unsigned bits = ((unsigned)sgn << 15) | mag;
  const unsigned idx = (mag - kLutLoBits) + (sgn ? (unsigned)kLutEntriesPerSign : 0u);
  *(float*)(d_dgelu_table + (idx << 2)) = act_dgelu(__uint_as_float(bits << 16));
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
  static constexpr int kLutSharedBytes = kNeedsLut ? kLutBytes : 16;
};

// Copy the dGeLU table into shared memory with one bulk-async (TMA) copy, so it
// costs no LSU wavefronts, and join on an mbarrier.
__device__ __forceinline__ void wait_lut(unsigned long long* bar) {
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  unsigned ok;
  do {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; "
        "selp.b32 %0, 1, 0, p; }"
        : "=r"(ok)
        : "r"(b)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void load_lut(unsigned char* dst, const void* src, int bytes,
                                         unsigned long long* bar, int tid) {
  const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  const unsigned b = (unsigned)__cvta_generic_to_shared(bar);
  if (tid == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(b) : "memory");
  __syncthreads();
  if (tid == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes)
                 : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];" ::"r"(d),
        "l"(__cvta_generic_to_global(src)), "r"(bytes), "r"(b)
        : "memory");
  }
  wait_lut(bar);
}

// Register-resident tiling: the 32x256 tile stays in registers; only the
// colwise partials go through shared memory.
template <bool IS_DACT, bool IS_ACT>
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
  __shared__ __align__(8) unsigned long long dgelu_table_barrier;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col0 = blockIdx.x * kTileCols;
  const int vecs_per_row = K >> 3;  // uint4 (8 bf16 values) per row

  if constexpr (C::kNeedsLut)
    load_lut(dgelu_table_smem, d_dgelu_table, C::kLutSharedBytes, &dgelu_table_barrier, tid);
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
#define MXFP8_DACT_WORD(T, M)                                                                    \
  {                                                                                              \
    constexpr bool kUseLut = LutSlot<C::kSlots, C::kLutSlots, (T) * 4 + (M) + 2>::value;         \
    const f32x2 product =                                                                        \
        mul_f32x2(dgelu_word<kUseLut>(x_words[M], dgelu_table), bf16x2_to_f32x2(grad_words[M])); \
    float product_lo, product_hi;                                                                \
    unpack_f32x2(product, product_lo, product_hi);                                               \
    tile_words[(h * kRowsInFlight + (T)) * kWordsPerLane + (M)] =                                \
        ptx::cvt_bf16x2(product_hi, product_lo);                                                 \
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
      const unsigned scale_pair = amax_to_e8m0_2x(bf16_hi(amax_pair), bf16_lo(amax_pair));
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
      const unsigned scale_pair = amax_to_e8m0_2x(bf16_hi(col_amax), bf16_lo(col_amax));
      *(unsigned short*)(colwise_scales + (size_t)(row0 / 32) * colwise_scale_stride + col0 +
                         2 * tid) = (unsigned short)scale_pair;
      col_scale_rcp_smem[tid] = e8m0x2_to_bf16x2_reciprocal(__byte_perm(scale_pair, 0, 0x4140));
    }
    auto drain_row_scales = [&]() {
      constexpr int drain_tid_base = C::kDrainTidBase;
      // A row's eight group scales are eight contiguous, 8-byte-aligned bytes of
      // rowwise_scales, so one warp drains a 32-row tile with one STG.64 per lane.
      if (tid >= drain_tid_base && tid < drain_tid_base + kRowsPerMxBlock) {
        const int r = tid - drain_tid_base;
        *(f32x2*)(rowwise_scales + (size_t)(row0 + r) * rowwise_scale_stride + (col0 >> 5)) =
            *(const f32x2*)(row_scale_bytes + r * kMxGroupsPerRow);
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

template <bool IS_DACT, bool IS_ACT>
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
  quantize_regtile<IS_DACT, IS_ACT>((const uint4*)input, (const uint4*)act_input, rowwise_out,
                                    rowwise_scales, colwise_out, colwise_scales, K,
                                    rowwise_scale_stride, colwise_scale_stride, iters);
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

template <bool IS_DACT, bool IS_ACT>
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
      NVTE_CHECK_CUDA(cudaFuncSetAttribute((const void*)quantize_mxfp8_kernel<IS_DACT, IS_ACT>,
                                           cudaFuncAttributePreferredSharedMemoryCarveout,
                                           kPercent));
      done[device] = true;
    }
  }
}

// Row blocks per CTA walk: M / 32, walked in powers of two.  The minimum walk
// exists for the table copy, so only the table-copying modes take it.
template <bool IS_DACT, bool IS_ACT>
int walk_length(int M, int K) {
  using C = TileConfig<IS_DACT, IS_ACT>;
  int iters = pick_walk_length(M / kRowsPerMxBlock, K / kTileCols, kCtaTargetActivation,
                               C::kNeedsLut ? kMinWalk : 1);
  if constexpr (IS_DACT) {
    if (iters > kMaxWalkDact) iters = kMaxWalkDact;
  }
  return iters;
}

template <bool IS_DACT, bool IS_ACT>
static void launch(const void* input, const void* act_input, void* rowwise_out,
                   void* rowwise_scales, void* colwise_out, void* colwise_scales, int M, int K,
                   int rowwise_scale_stride, int colwise_scale_stride, cudaStream_t stream) {
  using C = TileConfig<IS_DACT, IS_ACT>;
  set_carveout<IS_DACT, IS_ACT>();
  const int iters = walk_length<IS_DACT, IS_ACT>(M, K);
  dim3 grid(K / kTileCols, M / kRowsPerMxBlock / iters);
  quantize_mxfp8_kernel<IS_DACT, IS_ACT><<<grid, C::kThreadsPerCta, 0, stream>>>(
      (const unsigned*)input, (const unsigned*)act_input, (unsigned char*)rowwise_out,
      (unsigned char*)rowwise_scales, (unsigned char*)colwise_out, (unsigned char*)colwise_scales,
      K, rowwise_scale_stride, colwise_scale_stride, iters);
}

// The dGeLU table lives in an anonymous namespace, so there is one copy per
// translation unit, and as a __device__ array one per device.  The ready flags
// must match: this function has internal linkage and keeps one flag per device.
// A build issued under stream capture only runs when the graph does, so it does
// not mark the device ready.  Rebuilding is idempotent.
static void ensure_act_tables(cudaStream_t stream) {
  static std::mutex mutex;
  static std::vector<bool> ready;
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mutex);
  if (static_cast<size_t>(device) < ready.size() && ready[device]) return;
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  init_dgelu_table_kernel<<<(2 * kLutEntriesPerSign + 255) / 256, 256, 0, stream>>>();
  NVTE_CHECK_CUDA(cudaGetLastError());
  if (capture == cudaStreamCaptureStatusNone) {
    // Other streams on this device will skip the build, so it must complete.
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    if (static_cast<size_t>(device) >= ready.size()) ready.resize(device + 1, false);
    ready[device] = true;
  }
}

// Only GeLU and dGeLU are implemented.  The op is identified by template
// matching, since taking the address of a __device__ function in host code is
// not portable.
template <typename ParamOP, float (*OP)(float, const ParamOP&)>
constexpr bool is_gelu = false;
template <>
constexpr bool is_gelu<Empty, gelu<fp32, fp32>> = true;
template <typename ParamOP, float (*OP)(float, const ParamOP&)>
constexpr bool is_dgelu = false;
template <>
constexpr bool is_dgelu<Empty, dgelu<fp32, fp32>> = true;

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
  constexpr bool op_supported = !IS_DBIAS && ((IS_ACT && !IS_DACT && is_gelu<ParamOP, OP>) ||
                                              (IS_DACT && !IS_ACT && is_dgelu<ParamOP, OP>));
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
          output.has_columnwise_data() && !output.with_gemm_swizzled_scales &&
          input.dtype() == DType::kBFloat16 && output.dtype() == DType::kFloat8E4M3 &&
          output.amax.dptr == nullptr && noop.data.dptr == nullptr && shape_supported(rows, cols) &&
          aligned)) {
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
    ensure_act_tables(stream);
  }
  const void* act_ptr = IS_DACT ? act_input->data.dptr : input.data.dptr;
  launch<IS_DACT, IS_ACT>(input.data.dptr, act_ptr, output->data.dptr, output->scale_inv.dptr,
                          output->columnwise_data.dptr, output->columnwise_scale_inv.dptr,
                          static_cast<int>(rows), static_cast<int>(cols),
                          static_cast<int>(output->scale_inv.shape[1]),
                          static_cast<int>(output->columnwise_scale_inv.shape[1]), stream);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace regtile
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_REGTILE_CUH_
