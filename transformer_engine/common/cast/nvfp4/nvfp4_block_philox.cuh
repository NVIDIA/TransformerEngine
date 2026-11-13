/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_block_philox.cuh
 *  \brief Philox4x32 draws for NVFP4 stochastic rounding, one per output block, several
 *         streams at a time.
 *
 *  Each draw is exactly what curanddx's philox4x32_native_state<Rounds> returns from
 *  init(seed, subsequence, offset) followed by one generate4(): the counter is
 *  {offset_lo, offset_hi, subsequence_lo, subsequence_hi}.  Only the cost differs.  Callers
 *  give every output block its own subsequence and take one counter step from rng_state[1],
 *  so the counter never moves past the offset the framework reserved, whatever the grid.
 *
 *  Two savings over a plain per-thread generator:
 *   - Every round-0 term that reads only the launch-wide offset is evaluated once
 *     (make_philox).
 *   - When a thread draws for many blocks that share one subsequence word (its block column,
 *     say), placing that word in the subsequence's LOW half makes both of round 0's products
 *     thread constants, and the constant parts of rounds 0..2 fold away (make_fold).  The
 *     other half carries the varying coordinate XORed with the launch constant p.b, a
 *     bijection that cancels round 0's own XOR.
 *  Several streams advance together so the key schedule is paid once per batch.
 *
 *  The round count follows NVTE_BUILD_NUM_PHILOX_ROUNDS, as TE's other Philox users do.
 */

#ifndef TRANSFORMER_ENGINE_NVFP4_BLOCK_PHILOX_CUH_
#define TRANSFORMER_ENGINE_NVFP4_BLOCK_PHILOX_CUH_

#include <cuda_runtime.h>

#include <cstdint>

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace block_philox {

#ifdef NVTE_BUILD_NUM_PHILOX_ROUNDS
constexpr int kRounds = NVTE_BUILD_NUM_PHILOX_ROUNDS;
#else
constexpr int kRounds = 10;
#endif
// make_fold consumes rounds 0..2, so it only applies to builds with at least 3 rounds; callers
// must fall back to a plain generator otherwise.
constexpr bool kFoldSupported = kRounds >= 3;

// Subsequence of each output block, as (low word, logical high word).  The low word is the
// coordinate a thread keeps fixed while it walks the other, so make_fold can fold it.  Its
// parity tells the directions apart, so the rowwise and columnwise draws of one element never
// share a subsequence; within a direction the pair is the block's coordinates.
//   rowwise block (row, block_col):                  (2 * block_col,       row)
//   columnwise block (row_block, col), col = 4q + i: (2 * (4q) + 1, 4 * row_block + i)
// The column's offset inside its group of four moves to the high word because a thread owns
// four adjacent columns and folds against their base.
__host__ __device__ constexpr uint32_t rowwise_sub_lo(uint32_t block_col) { return block_col << 1; }
__host__ __device__ constexpr uint32_t rowwise_sub_hi(uint32_t row) { return row; }
__host__ __device__ constexpr uint32_t columnwise_sub_lo(uint32_t quad_base_col) {
  return (quad_base_col << 1) | 1u;
}
__host__ __device__ constexpr uint32_t columnwise_sub_hi(uint32_t row_block, uint32_t col_in_quad) {
  return (row_block << 2) | col_in_quad;
}

constexpr uint32_t kM0 = 0xD2511F53u;
constexpr uint32_t kM1 = 0xCD9E8D57u;
constexpr uint32_t kW0 = 0x9E3779B9u;
constexpr uint32_t kW1 = 0xBB67AE85u;

// Both halves of a 32 x 32 -> 64-bit product in one IMAD.WIDE.U32, written as PTX like
// curanddx's mulhilo32.  As inline asm the product chain keeps the place the kernels give it,
// right after their input loads, where it fills the wait for them; as plain C++ the compiler
// sank it past the loads' consumers and the kernels stalled on the loads instead.
__device__ __forceinline__ void mulhilo(uint32_t a, uint32_t b, uint32_t &lo, uint32_t &hi) {
  uint64_t p;
  asm("mul.wide.u32 %0, %1, %2;" : "=l"(p) : "r"(a), "r"(b));
  lo = static_cast<uint32_t>(p);
  hi = static_cast<uint32_t>(p >> 32);
}

// Three-way XOR in one LOP3.
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Round-0 terms that read only the key and the launch-wide offset.
struct PhiloxState {
  uint2 key;   // the key round 1 uses
  uint32_t a;  // offset_hi ^ k0         -> round-0 c.x = hi(sub_lo * M1) ^ a
  uint32_t b;  // hi(offset_lo * M0) ^ k1 -> round-0 c.z = b ^ sub_hi
  uint32_t w;  // lo(offset_lo * M0)      -> round-0 c.w
};

__device__ __forceinline__ PhiloxState make_philox(uint64_t seed, uint64_t offset) {
  const uint32_t k0 = static_cast<uint32_t>(seed);
  const uint32_t k1 = static_cast<uint32_t>(seed >> 32);
  uint32_t lo0, hi0;
  mulhilo(static_cast<uint32_t>(offset), kM0, lo0, hi0);
  PhiloxState p;
  p.a = static_cast<uint32_t>(offset >> 32) ^ k0;
  p.b = hi0 ^ k1;
  p.w = lo0;
  p.key = make_uint2(k0 + kW0, k1 + kW1);
  return p;
}

// Rounds 0..2 for a fixed subsequence low word.
struct PhiloxFold {
  uint32_t A;     // c1_0 ^ k1.x   -> c0_1 = hi1 ^ A
  uint32_t B;     // hi1_2 ^ k2.x  -> c0_2 = B ^ c1_1
  uint32_t D;     // c3_1 ^ k2.y   -> c2_2 = hi0 ^ D
  uint32_t c1_2;  // round-2 c1, constant for a fixed low word
  uint2 k3;       // key as of round 3
};

__device__ __forceinline__ PhiloxFold make_fold(const PhiloxState &p, uint32_t sub_lo) {
  uint32_t lo0, hi0, lo1, hi1;
  mulhilo(sub_lo, kM1, lo1, hi1);  // round 0: both products are constant
  const uint32_t c0_0 = hi1 ^ p.a, c1_0 = lo1;
  uint2 k = p.key;               // round-1 key
  mulhilo(c0_0, kM0, lo0, hi0);  // round 1: the .x product is constant
  PhiloxFold f;
  f.A = c1_0 ^ k.x;
  const uint32_t c2_1 = xor3(hi0, p.w, k.y), c3_1 = lo0;
  k.x += kW0;
  k.y += kW1;                    // round-2 key
  mulhilo(c2_1, kM1, lo1, hi1);  // round 2: the .z product is constant
  f.B = hi1 ^ k.x;
  f.c1_2 = lo1;
  f.D = c3_1 ^ k.y;
  k.x += kW0;
  k.y += kW1;  // round-3 key
  f.k3 = k;
  return f;
}

// N draws that share the fold's low word.  logical_hi[i] is the varying coordinate; the
// draw's subsequence high word is logical_hi[i] ^ p.b.
template <int N>
__device__ __forceinline__ void philox_fold_n(const uint32_t (&logical_hi)[N], const PhiloxFold &f,
                                              uint4 (&c)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) {  // the varying remnant of rounds 0..2
    uint32_t lo0, hi0, lo1, hi1;
    mulhilo(logical_hi[i], kM1, lo1, hi1);
    mulhilo(hi1 ^ f.A, kM0, lo0, hi0);
    c[i] = make_uint4(f.B ^ lo1, f.c1_2, hi0 ^ f.D, lo0);
  }
  uint2 k = f.k3;
#pragma unroll
  for (int r = 3; r < kRounds; ++r) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      uint32_t lo0, hi0, lo1, hi1;
      mulhilo(c[i].x, kM0, lo0, hi0);
      mulhilo(c[i].z, kM1, lo1, hi1);
      c[i] = make_uint4(xor3(hi1, c[i].y, k.x), lo1, xor3(hi0, c[i].w, k.y), lo0);
    }
    if (r < kRounds - 1) {
      k.x += kW0;
      k.y += kW1;
    }
  }
}

}  // namespace block_philox
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_NVFP4_BLOCK_PHILOX_CUH_
