/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_mla_rope_mxfp8_attn.cu
 *  \brief Multi-latent attention RoPE fused with MXFP8 quantization, writing the scaling factor
 *         layouts that MXFP8 attention and the up-projection GEMMs consume.
 *
 *  Forward, for MXFP8 attention (DeepSeek-V3 head dimensions: 128 + 64 query/key, 128 value):
 *   - query: RoPE of the query's last 64 dimensions and MXFP8 quantization;
 *   - key/value: split of the key/value up-projection output, RoPE of the shared key RoPE
 *     dimensions and MXFP8 quantization of key and value.
 *  Both write the data in SBHD and the scaling factors permuted to BHSD and swizzled for the GEMM,
 *  the bytes of mxfp8_quantize_only followed by mxfp8_transpose_swizzle (or
 *  fused_mla_kv_rope_mxfp8 followed by mxfp8_transpose_swizzle), so the scale permute and swizzle
 *  are not separate kernels.
 *
 *  Backward:
 *   - the query RoPE backward in place (BF16, Megatron-LM's roundings);
 *   - the query RoPE backward and the key/value split backward fused with the MXFP8 quantization
 *     of the result, which is the up-projections' grad_output: row-wise and column-wise data and
 *     GEMM-swizzled scaling factors, without writing the BF16 gradient.
 *
 *  Quantization is that of the MXFP8 quantizer (E4M3): amax of 32 BF16 values, E8M0 scale
 *  rounded up from amax / 448, data = x * 2^-scale in BF16 converted to E4M3 with saturation.
 *
 *  Work unit of the quantizing kernels: 32 tokens x 64 columns, one warp. Lane 8 * r + k holds
 *  the 16-byte chunk k (8 columns) of tokens 4 * i + r, i < 8. Row-wise amax: in-lane over 8
 *  values and 2 shuffles; column-wise: in-lane over 8 tokens and 2 shuffles. A 128-token tile's
 *  scaling factors are whole 512-byte blocks of the swizzled layout; they are assembled in shared
 *  memory and written with 16-byte stores.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "../util/ptx.cuh"
#include "../util/ptx_arch_spec.cuh"
#include "../utils.cuh"

namespace transformer_engine {
namespace {

constexpr int kNopeDim = 128;                // Query/key dimensions without RoPE.
constexpr int kRopeDim = 64;                 // RoPE dimensions of the query/key.
constexpr int kQKDim = kNopeDim + kRopeDim;  // 192
constexpr int kValueDim = 128;
constexpr int kKVDim = kNopeDim + kValueDim;  // 256, per head of the key/value up-projection
constexpr int kTileTokens = 128;              // Tokens of a swizzled scaling factor block.

// ---------------------------------------------------------------------------------------------
// Quantization of one unit (32 tokens x 64 columns) held by a warp.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t habs2(uint32_t w) {
  __nv_bfloat162 v = __habs2(*reinterpret_cast<__nv_bfloat162 *>(&w));
  return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) {
  __nv_bfloat162 v =
      __hmax2(*reinterpret_cast<__nv_bfloat162 *>(&a), *reinterpret_cast<__nv_bfloat162 *>(&b));
  return *reinterpret_cast<uint32_t *>(&v);
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }

// E8M0 scales of two FP32 values (rounded up, saturating): byte 0 <- lo, byte 1 <- hi.
__device__ __forceinline__ uint32_t e8m0x2(const float lo, const float hi) {
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    uint16_t out;
    asm volatile("{\n cvt.rp.satfinite.ue8m0x2.f32  %0, %1, %2;\n}" : "=h"(out) : "f"(hi), "f"(lo));
    return out;
  } else {
    return ptx::float_to_e8m0(lo) | (static_cast<uint32_t>(ptx::float_to_e8m0(hi)) << 8);
  }
}

// BF16 2^-e of an E8M0 exponent, as ptx::exp2f_rcp<bf16>.
__device__ __forceinline__ uint32_t exp2f_rcp_bf16_bits(const uint32_t e) {
  if (e == 255) return 0x7fffu;
  if (e == 254) return 0x0040u;
  return ((254 - e) << 7) & 0xFFFFu;
}

// BF16 reciprocal scales of the two E8M0 exponents in bytes 0 and 1 of e2, as a bf16x2
// (low half <- byte 0).
__device__ __forceinline__ uint32_t rcp_scales2(const uint32_t e2) {
  const uint32_t spread = __byte_perm(e2, 0, 0x7170);  // byte 0 -> bits 0..7, byte 1 -> 16..23
  uint32_t bits = 0x7F007F00u - spread * 0x80u;
  if (__builtin_expect((e2 & 0xFEu) == 0xFEu || (e2 & 0xFE00u) == 0xFE00u, 0)) {
    bits = exp2f_rcp_bf16_bits(e2 & 0xFF) | (exp2f_rcp_bf16_bits((e2 >> 8) & 0xFF) << 16);
  }
  return bits;
}

// Four BF16 values (two words) times one bf16x2 scale, converted to E4M3.
__device__ __forceinline__ uint32_t cvt_4x(const uint32_t x01, const uint32_t x23,
                                           const uint32_t scale) {
  const uint2 in = make_uint2(x01, x23);
  uint32_t out;
  ptx::mul_cvt_4x(reinterpret_cast<ptx::fp8e4m3x4 &>(out),
                  reinterpret_cast<const ptx::bf16x4 &>(in),
                  reinterpret_cast<const ptx::bf16x2 &>(scale));
  return out;
}

// Two BF16 pairs, each times its own bf16x2 scale, converted to E4M3.
__device__ __forceinline__ uint32_t cvt_2x2(const uint32_t in0, const uint32_t s0,
                                            const uint32_t in1, const uint32_t s1) {
  uint32_t out;
  ptx::mul_cvt_4x(
      reinterpret_cast<ptx::fp8e4m3x4 &>(out), reinterpret_cast<const ptx::bf16x2 &>(in0),
      reinterpret_cast<const ptx::bf16x2 &>(s0), reinterpret_cast<const ptx::bf16x2 &>(in1),
      reinterpret_cast<const ptx::bf16x2 &>(s1));
  return out;
}

// A quantized unit, for lane 8 * r + k: row-wise and column-wise E4M3 of tokens 4 * i + r,
// columns 8 * k .. 8 * k + 7.
struct QuantizedUnit {
  uint2 row[8], col[8];
  uint32_t rs[8];  // Lanes with k == 0: row-wise scales (blocks 0, 1 of the unit) of token 4i + r.
  uint2 cs;        // Column-wise scales of columns 8 * k .. 8 * k + 7.
};

__device__ __forceinline__ void quantize(const uint32_t (&v)[8][4], QuantizedUnit &o) {
  constexpr float kMaxNormRcp = Quantized_Limits<fp8e4m3>::max_norm_rcp;
  uint32_t a[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) a[i][j] = habs2(v[i][j]);
  }
  // Row-wise: amax per token over the lane's 8 values (two tokens per word), then over the
  // block's 4 lanes.
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    const uint32_t m0 = hmax2(hmax2(a[i][0], a[i][1]), hmax2(a[i][2], a[i][3]));
    const uint32_t m1 = hmax2(hmax2(a[i + 1][0], a[i + 1][1]), hmax2(a[i + 1][2], a[i + 1][3]));
    // Low half: token i, high half: token i + 1.
    uint32_t m = hmax2(__byte_perm(m0, m1, 0x5410), __byte_perm(m0, m1, 0x7632));
    m = hmax2(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = hmax2(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const uint32_t e2 = e8m0x2(bf16_lo(m) * kMaxNormRcp, bf16_hi(m) * kMaxNormRcp);
    const uint32_t sc2 = rcp_scales2(e2);
    const uint32_t s0 = __byte_perm(sc2, 0, 0x1010), s1 = __byte_perm(sc2, 0, 0x3232);
    o.row[i] = make_uint2(cvt_4x(v[i][0], v[i][1], s0), cvt_4x(v[i][2], v[i][3], s0));
    o.row[i + 1] =
        make_uint2(cvt_4x(v[i + 1][0], v[i + 1][1], s1), cvt_4x(v[i + 1][2], v[i + 1][3], s1));
    const uint32_t e2_block1 = __shfl_xor_sync(0xffffffffu, e2, 4);  // Lane k + 4.
    o.rs[i] = __byte_perm(e2, e2_block1, 0x0040);
    o.rs[i + 1] = __byte_perm(e2, e2_block1, 0x0051);
  }
  // Column-wise.
  uint32_t s[4], e[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    uint32_t m = hmax2(hmax2(hmax2(a[0][j], a[1][j]), hmax2(a[2][j], a[3][j])),
                       hmax2(hmax2(a[4][j], a[5][j]), hmax2(a[6][j], a[7][j])));
    m = hmax2(m, __shfl_xor_sync(0xffffffffu, m, 8));
    m = hmax2(m, __shfl_xor_sync(0xffffffffu, m, 16));
    e[j] = e8m0x2(bf16_lo(m) * kMaxNormRcp, bf16_hi(m) * kMaxNormRcp);
    s[j] = rcp_scales2(e[j]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    o.col[i] =
        make_uint2(cvt_2x2(v[i][0], s[0], v[i][1], s[1]), cvt_2x2(v[i][2], s[2], v[i][3], s[3]));
  }
  o.cs = make_uint2(e[0] | (e[1] << 16), e[2] | (e[3] << 16));
}

// Megatron-LM's MLA RoPE with BF16 cos / sin, 8 rotation pairs (x1, x2) -> 8 outputs (4 words)
// of the left or the right half: left = fma(x1, cos_l, -(x2 * sin_l)),
// right = fma(x1, sin_r, x2 * cos_r), each operation rounded to BF16.
__device__ __forceinline__ void rope8(const uint4 x_lo, const uint4 x_hi, const uint4 c,
                                      const uint4 sn, const bool right, uint32_t (&w)[4]) {
  const uint32_t xs[8] = {x_lo.x, x_lo.y, x_lo.z, x_lo.w, x_hi.x, x_hi.y, x_hi.z, x_hi.w};
  const uint32_t cw[4] = {c.x, c.y, c.z, c.w}, sw[4] = {sn.x, sn.y, sn.z, sn.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t x1u = __byte_perm(xs[2 * j], xs[2 * j + 1], 0x5410);
    const uint32_t x2u = __byte_perm(xs[2 * j], xs[2 * j + 1], 0x7632);
    const __nv_bfloat162 x1 = *reinterpret_cast<const __nv_bfloat162 *>(&x1u);
    const __nv_bfloat162 x2 = *reinterpret_cast<const __nv_bfloat162 *>(&x2u);
    const __nv_bfloat162 cj = *reinterpret_cast<const __nv_bfloat162 *>(&cw[j]);
    const __nv_bfloat162 sj = *reinterpret_cast<const __nv_bfloat162 *>(&sw[j]);
    const __nv_bfloat162 y =
        right ? __hfma2(x1, sj, __hmul2(x2, cj)) : __hfma2(x1, cj, __hneg2(__hmul2(x2, sj)));
    w[j] = *reinterpret_cast<const uint32_t *>(&y);
  }
}

// Megatron-LM's MLA query RoPE backward for 2 x 2 rotation pairs: left = dq[nope : nope + 32],
// right = dq[nope + 32 : nope + 64]; out[2p] = fma(left_p, cos_l_p, right_p * sin_r_p),
// out[2p + 1] = fma(right_p, cos_r_p, -(left_p * sin_l_p)), each operation rounded to BF16.
// Writes the four output words (interleaved pairs).
__device__ __forceinline__ void rope_bwd4(const uint2 left, const uint2 right, const uint2 cl,
                                          const uint2 sl, const uint2 cr, const uint2 sr,
                                          uint32_t (&o)[4]) {
  const uint32_t lw[2] = {left.x, left.y}, rw[2] = {right.x, right.y};
  const uint32_t clw[2] = {cl.x, cl.y}, slw[2] = {sl.x, sl.y};
  const uint32_t crw[2] = {cr.x, cr.y}, srw[2] = {sr.x, sr.y};
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    const __nv_bfloat162 xl = *reinterpret_cast<const __nv_bfloat162 *>(&lw[w]);
    const __nv_bfloat162 xr = *reinterpret_cast<const __nv_bfloat162 *>(&rw[w]);
    const __nv_bfloat162 c_l = *reinterpret_cast<const __nv_bfloat162 *>(&clw[w]);
    const __nv_bfloat162 s_l = *reinterpret_cast<const __nv_bfloat162 *>(&slw[w]);
    const __nv_bfloat162 c_r = *reinterpret_cast<const __nv_bfloat162 *>(&crw[w]);
    const __nv_bfloat162 s_r = *reinterpret_cast<const __nv_bfloat162 *>(&srw[w]);
    const __nv_bfloat162 x1 = __hfma2(xl, c_l, __hmul2(xr, s_r));
    const __nv_bfloat162 x2 = __hfma2(xr, c_r, __hneg2(__hmul2(xl, s_l)));
    const uint32_t a = *reinterpret_cast<const uint32_t *>(&x1);
    const uint32_t b = *reinterpret_cast<const uint32_t *>(&x2);
    o[2 * w] = __byte_perm(a, b, 0x5410);      // x1[lo], x2[lo]
    o[2 * w + 1] = __byte_perm(a, b, 0x7632);  // x1[hi], x2[hi]
  }
}

// ---------------------------------------------------------------------------------------------
// Forward: query / key-value RoPE + MXFP8, attention scale layout.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(const uint32_t smem_addr, const void *gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gptr)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// Stores the data of a quantized unit of head column `column` (= batch * heads + head) with head
// dimension D, starting at head dimension d0, tokens t0 .. t0 + 31.
__device__ __forceinline__ void store_data(const QuantizedUnit &o, uint8_t *rowwise_data,
                                           uint8_t *columnwise_data, const uint32_t D,
                                           const uint32_t t0, const uint32_t column,
                                           const uint32_t num_columns, const uint32_t d0,
                                           const int lane) {
  const uint32_t r = lane >> 3, k = lane & 7;
  const uint32_t row_bytes = num_columns * D;
  const uint32_t offset = (t0 + r) * row_bytes + column * D + d0 + 8 * k;
  uint8_t *rd = rowwise_data + offset;
  uint8_t *cd = columnwise_data + offset;
  const uint32_t step = 4 * row_bytes;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    *reinterpret_cast<uint2 *>(rd) = o.row[i];
    *reinterpret_cast<uint2 *>(cd) = o.col[i];
    rd += step;
    cd += step;
  }
}

// Places a unit's scales (token block j of the tile, head dimension d0) in the tile's swizzled
// scale blocks of one head, in shared memory:
//   row-wise: [D / 128 blocks][128 words]; word (s % 32) * 4 + s / 32 holds the scales of the
//   4 consecutive 32-element blocks of token s;
//   column-wise: [ceil(D / 128) blocks][128 words]; word (n % 32) * 4 + (n % 128) / 32 of block
//   n / 128 holds the scales of the 4 token blocks of column n.
__device__ __forceinline__ void stage_scales(const QuantizedUnit &o, uint8_t *rowwise_tile,
                                             uint8_t *columnwise_tile, const uint32_t j,
                                             const uint32_t d0, const int lane) {
  const uint32_t r = lane >> 3, k = lane & 7;
  const uint32_t block = d0 / 32;  // Even.
  if (k == 0) {
    uint8_t *base = rowwise_tile + (block / 4) * 512 + (block % 4) + 4 * j;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<uint16_t *>(base + 16 * (4 * i + r)) = static_cast<uint16_t>(o.rs[i]);
    }
  }
  if (r == 0) {
    const uint32_t bytes[2] = {o.cs.x, o.cs.y};
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const uint32_t n = d0 + 8 * k + jj;
      columnwise_tile[(n / 128) * 512 + ((n % 32) * 4 + (n % 128) / 32) * 4 + j] =
          static_cast<uint8_t>(bytes[jj / 4] >> (8 * (jj % 4)));
    }
  }
}

// One MXFP8 output in the attention layout: data [s, b * h * D] (SBHD), scaling factors permuted
// to BHSD and swizzled: row-wise [b * h * s, ceil(D / 32 / 4) * 4], column-wise
// [b * h * s / 32, ceil(D / 128) * 128].
struct AttnOutput {
  uint8_t *rowwise_data, *columnwise_data, *rowwise_scale_inv, *columnwise_scale_inv;
};

struct RopeParams {
  const uint8_t *src;                    // Query [s, b, h, 192] or key/value [s, b, h, 256], BF16.
  const uint8_t *k_rope;                 // Key/value: s * b rows of 64 BF16, row (t * b + batch).
  uint32_t k_rope_row_bytes;             // Bytes between consecutive k_rope rows.
  const uint16_t *cos, *sin;             // [>= s, 64] BF16.
  AttnOutput out0, out1;                 // Query; or key, value.
  uint32_t num_columns, heads, seq_len;  // b * h, h, s.
};

constexpr int kStages = 2;  // Load ring stages per warp.
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * THREADS_PER_WARP;
constexpr int kHeadsPerTile = 2;       // Heads of one batch entry per tile.
constexpr int kStageBytes = 32 * 128;  // One unit of BF16 input: 32 tokens x 64 columns.

// Shared memory of a CTA: the warps' load rings, then two sets of scale tiles (query: 1 + 1 KB per
// head; key/value: key 1 + 1 KB, value 0.5 + 0.5 KB per head).
template <bool kKV>
struct RopeLayout {
  static constexpr int kRing = kWarps * kStages * kStageBytes;
  static constexpr int kScales = kKV ? kHeadsPerTile * 3072 : kHeadsPerTile * 2048;
  static constexpr int kBytes = kRing + 2 * kScales;
  // Units per tile (4 token blocks): query 3 per head; key/value 4 per head + 1 RoPE unit
  // (the shared key RoPE dimensions) per token block.
  static constexpr int kUnits = kKV ? 4 * kHeadsPerTile * 4 + 4 : 4 * kHeadsPerTile * 3;
  static constexpr int kIters = (kUnits + kWarps - 1) / kWarps;
};

// Persistent: CTA x handles tiles x, x + gridDim.x, ... (tile = token tile * groups + head
// group); each warp streams its units of consecutive tiles through its cp.async ring, so the
// loads of the next tile overlap the end of the current one.
template <bool kKV>
__global__ void __launch_bounds__(kThreads) fused_mla_rope_mxfp8_attn_kernel(const RopeParams p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using L = RopeLayout<kKV>;
  extern __shared__ __align__(16) uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t r = lane >> 3, k = lane & 7;
  uint8_t *ring = smem + warp * kStages * kStageBytes;
  uint8_t *scales_base = smem + L::kRing;
  const uint32_t ring_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(ring)) + r * 128 + 16 * k;
  const uint32_t groups = p.num_columns / kHeadsPerTile;
  const uint32_t token_tiles = p.seq_len / kTileTokens;
  const uint32_t num_tiles = groups * token_tiles;
  const uint32_t my_tiles =
      blockIdx.x < num_tiles ? (num_tiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  const uint32_t num_steps = my_tiles * L::kIters;

  for (int i = threadIdx.x; i < 2 * L::kScales / 16; i += kThreads) {
    reinterpret_cast<uint4 *>(scales_base)[i] = make_uint4(0, 0, 0, 0);
  }

  // Unit u of a tile -> (token block j, head hh, part). Query parts: 64-column pieces of the
  // head; key/value parts 0, 1: key without RoPE, 2, 3: value, 4: the key RoPE dimensions.
  auto decode = [&](const uint32_t u, uint32_t &j, uint32_t &hh, uint32_t &part) {
    if (kKV) {
      if (u >= 16 * kHeadsPerTile) {
        j = u - 16 * kHeadsPerTile;
        hh = 0;
        part = 4;
      } else {
        j = u / (4 * kHeadsPerTile);
        const uint32_t rem = u - j * 4 * kHeadsPerTile;
        hh = rem >> 2;
        part = rem & 3;
      }
    } else {
      j = u / (3 * kHeadsPerTile);
      const uint32_t rem = u - j * 3 * kHeadsPerTile;
      hh = rem / 3;
      part = rem - hh * 3;
    }
  };
  // Issues the loads of this warp's step n into ring stage `stage`.
  auto issue = [&](const uint32_t n, const int stage) {
    const uint32_t ti = n / L::kIters;
    const uint32_t tile = blockIdx.x + ti * gridDim.x;
    const uint32_t u = warp + (n - ti * L::kIters) * kWarps;
    if (n < num_steps && u < L::kUnits) {
      uint32_t j, hh, part;
      decode(u, j, hh, part);
      const uint32_t tt = tile / groups, col0 = (tile - tt * groups) * kHeadsPerTile;
      const uint32_t t = (tt * 4 + j) * 32 + r, column = col0 + hh;
      const uint8_t *src;
      uint32_t stride;
      if (kKV && part == 4) {
        const uint32_t batches = p.num_columns / p.heads, batch = col0 / p.heads;
        stride = batches * p.k_rope_row_bytes;
        src = p.k_rope + static_cast<size_t>(t * batches + batch) * p.k_rope_row_bytes;
      } else {
        constexpr uint32_t kRowBytes = (kKV ? kKVDim : kQKDim) * 2;
        stride = p.num_columns * kRowBytes;
        src = p.src + static_cast<size_t>(t * p.num_columns + column) * kRowBytes + part * 128;
      }
      src += 16 * k;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        cp_async16(ring_addr + stage * kStageBytes + i * 512, src);
        src += 4 * stride;
      }
    }
    cp_async_commit();
  };

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) issue(s, s);
  __syncthreads();  // Scale tiles zeroed.
  int stage = 0;
  const uint32_t mc4 = p.num_columns * p.seq_len / kTileTokens;
#pragma unroll 1
  for (uint32_t ti = 0; ti < my_tiles; ++ti) {
    const uint32_t tile = blockIdx.x + ti * gridDim.x;
    const uint32_t tt = tile / groups, col0 = (tile - tt * groups) * kHeadsPerTile;
    uint8_t *scales = scales_base + (ti & 1) * L::kScales;
#pragma unroll 1
    for (int it = 0; it < L::kIters; ++it) {
      const uint32_t n = ti * L::kIters + it;
      const uint32_t u = warp + it * kWarps;
      issue(n + kStages - 1, (stage + kStages - 1) % kStages);
      cp_async_wait<kStages - 1>();
      __syncwarp();
      const bool valid = u < L::kUnits;
      uint32_t j, hh, part;
      decode(valid ? u : 0, j, hh, part);
      const bool rope = kKV ? part == 4 : part == 2;
      const uint32_t t0 = (tt * 4 + j) * 32;
      const uint8_t *staged = ring + stage * kStageBytes + r * 128;
      uint32_t v[8][4];
      if (!rope) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint4 x = *reinterpret_cast<const uint4 *>(staged + i * 512 + 16 * k);
          v[i][0] = x.x;
          v[i][1] = x.y;
          v[i][2] = x.z;
          v[i][3] = x.w;
        }
      } else {
        // Output chunk k: left (k < 4) or right half; rotation pairs 8 * (k % 4) .. + 7, which
        // are input chunks 2 * (k % 4) and 2 * (k % 4) + 1.
        const uint32_t g = k & 3;
        const bool right = k >= 4;
        const uint16_t *c = p.cos + (t0 + r) * kRopeDim + (right ? 32 : 0) + 8 * g;
        const uint16_t *sn = p.sin + (t0 + r) * kRopeDim + (right ? 32 : 0) + 8 * g;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const uint8_t *row = staged + i * 512 + 32 * g;
          const uint4 x_lo = *reinterpret_cast<const uint4 *>(row);
          const uint4 x_hi = *reinterpret_cast<const uint4 *>(row + 16);
          const uint4 ca = __ldg(reinterpret_cast<const uint4 *>(c + i * 4 * kRopeDim));
          const uint4 sa = __ldg(reinterpret_cast<const uint4 *>(sn + i * 4 * kRopeDim));
          rope8(x_lo, x_hi, ca, sa, right, v[i]);
        }
      }
      __syncwarp();
      QuantizedUnit o;
      quantize(v, o);
      if (valid) {
        if (kKV) {
          if (part == 4) {
            for (uint32_t h = 0; h < kHeadsPerTile; ++h) {
              store_data(o, p.out0.rowwise_data, p.out0.columnwise_data, kQKDim, t0, col0 + h,
                         p.num_columns, kNopeDim, lane);
              stage_scales(o, scales + h * 1024, scales + kHeadsPerTile * 1024 + h * 1024, j,
                           kNopeDim, lane);
            }
          } else if (part < 2) {
            store_data(o, p.out0.rowwise_data, p.out0.columnwise_data, kQKDim, t0, col0 + hh,
                       p.num_columns, 64 * part, lane);
            stage_scales(o, scales + hh * 1024, scales + kHeadsPerTile * 1024 + hh * 1024, j,
                         64 * part, lane);
          } else {
            store_data(o, p.out1.rowwise_data, p.out1.columnwise_data, kValueDim, t0, col0 + hh,
                       p.num_columns, 64 * (part - 2), lane);
            stage_scales(o, scales + kHeadsPerTile * 2048 + hh * 512,
                         scales + kHeadsPerTile * 2560 + hh * 512, j, 64 * (part - 2), lane);
          }
        } else {
          store_data(o, p.out0.rowwise_data, p.out0.columnwise_data, kQKDim, t0, col0 + hh,
                     p.num_columns, 64 * part, lane);
          stage_scales(o, scales + hh * 1024, scales + kHeadsPerTile * 1024 + hh * 1024, j,
                       64 * part, lane);
        }
      }
      stage = (stage + 1) % kStages;
    }
    __syncthreads();  // The tile's scales are staged.
    // Scale tiles -> global, then zeroed for tile ti + 2. Row-wise block b of column c:
    // ((c * s / 128 + tt) * (padded blocks / 4) + b) * 512 bytes. Column-wise block b of column
    // c: (b * c_total * s / 128 + c * s / 128 + tt) * 512 bytes.
    constexpr int kBlocksPerHead = kKV ? 6 : 4;  // Query/key: 2 row-wise + 2 column-wise;
                                                 // value: 1 + 1.
    for (int task = threadIdx.x; task < kHeadsPerTile * kBlocksPerHead * 32; task += kThreads) {
      const int vec = task & 31, blk = task >> 5;
      const int hh = blk / kBlocksPerHead, b = blk % kBlocksPerHead;
      const uint32_t c = col0 + hh;
      uint8_t *src_block;
      uint8_t *dst;
      if (b < 2) {
        src_block = scales + hh * 1024 + b * 512;
        dst = p.out0.rowwise_scale_inv + static_cast<size_t>((c * token_tiles + tt) * 2 + b) * 512;
      } else if (b < 4) {
        src_block = scales + kHeadsPerTile * 1024 + hh * 1024 + (b - 2) * 512;
        dst = p.out0.columnwise_scale_inv +
              static_cast<size_t>((b - 2) * mc4 + c * token_tiles + tt) * 512;
      } else if (b == 4) {
        src_block = scales + kHeadsPerTile * 2048 + hh * 512;
        dst = p.out1.rowwise_scale_inv + static_cast<size_t>(c * token_tiles + tt) * 512;
      } else {
        src_block = scales + kHeadsPerTile * 2560 + hh * 512;
        dst = p.out1.columnwise_scale_inv + static_cast<size_t>(c * token_tiles + tt) * 512;
      }
      uint4 *sv = reinterpret_cast<uint4 *>(src_block) + vec;
      reinterpret_cast<uint4 *>(dst)[vec] = *sv;
      *sv = make_uint4(0, 0, 0, 0);
    }
  }
  cp_async_wait<0>();
#else
  NVTE_DEVICE_THREAD0_ERROR("Fused MLA RoPE with MXFP8 output requires SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// ---------------------------------------------------------------------------------------------
// Backward.
// ---------------------------------------------------------------------------------------------

// Query RoPE backward in place on dq [rows, 192] (rows = s * b * h). Lane m of a row's 8 lanes
// reads rotation pairs 4m .. 4m + 3 (8 bytes of each half) and writes one 16-byte chunk; a warp
// covers 4 rows at a time, kRowsPerLane times.
constexpr int kQBwdRowsPerLane = 4;
constexpr int kQBwdThreads = 256;

__global__ void __launch_bounds__(kQBwdThreads)
    fused_mla_q_rope_bwd_kernel(uint16_t *__restrict__ dq, const uint16_t *__restrict__ cos,
                                const uint16_t *__restrict__ sin, const uint32_t rows,
                                const uint32_t rows_per_token) {
  constexpr int NR = kQBwdRowsPerLane;
  const uint32_t lane = threadIdx.x & 31, r = lane >> 3, m = lane & 7;
  const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const uint32_t row0 = warp * (4 * NR) + r;
  uint2 left[NR], right[NR], cl[NR], sl[NR], cr[NR], sr[NR];
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const uint32_t row = row0 + 4 * i;
    if (row < rows) {
      const uint16_t *base = dq + static_cast<size_t>(row) * kQKDim + kNopeDim;
      left[i] = *reinterpret_cast<const uint2 *>(base + 4 * m);
      right[i] = *reinterpret_cast<const uint2 *>(base + 32 + 4 * m);
      const uint32_t t = row / rows_per_token;
      cl[i] = __ldg(reinterpret_cast<const uint2 *>(cos + t * kRopeDim + 4 * m));
      sl[i] = __ldg(reinterpret_cast<const uint2 *>(sin + t * kRopeDim + 4 * m));
      cr[i] = __ldg(reinterpret_cast<const uint2 *>(cos + t * kRopeDim + 32 + 4 * m));
      sr[i] = __ldg(reinterpret_cast<const uint2 *>(sin + t * kRopeDim + 32 + 4 * m));
    }
  }
#pragma unroll
  for (int i = 0; i < NR; ++i) {
    const uint32_t row = row0 + 4 * i;
    if (row < rows) {
      uint32_t o[4];
      rope_bwd4(left[i], right[i], cl[i], sl[i], cr[i], sr[i], o);
      *reinterpret_cast<uint4 *>(dq + static_cast<size_t>(row) * kQKDim + kNopeDim + 8 * m) =
          make_uint4(o[0], o[1], o[2], o[3]);
    }
  }
}

__device__ __forceinline__ uint4 ld_stream(const void *ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint2 ld_stream2(const void *ptr) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];"
               : "=r"(v.x), "=r"(v.y)
               : "l"(ptr));
  return v;
}

// One MXFP8 output [T, N] with GEMM-swizzled scaling factors (row-wise [T, N / 32], column-wise
// [T / 32, N]), T and N multiples of 128.
struct GradOutput {
  uint8_t *rowwise_data, *columnwise_data, *rowwise_scale_inv, *columnwise_scale_inv;
};

struct GradParams {
  const uint8_t *a;           // Query: dq [T, h, 192]; key/value: dk [T, h, 192].
  const uint8_t *b;           // Key/value: dv [T, h, 128].
  const uint16_t *cos, *sin;  // Query: [>= s, 64] BF16.
  GradOutput out;
  uint32_t tokens, columns, heads, batch;  // T = s * b, N, h, b.
};

// CTA = 128 tokens x 128 columns, 8 warps; warp w = token block w / 2 (32 tokens) x column half
// w % 2 (64 columns). A CTA's row-wise and column-wise scaling factors are each exactly one
// 512-byte swizzled block. kKV = false: query (RoPE backward on head dimensions 128..191);
// kKV = true: key/value split backward (d_kv = [dk[:128] | dv] per head).
template <bool kKV>
__global__ void __launch_bounds__(kThreads) fused_mla_rope_bwd_mxfp8_kernel(const GradParams p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ __align__(16) uint8_t sc[1024];  // [0, 512): row-wise block, [512, 1024): column-wise.
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t r = lane >> 3, k = lane & 7;
  const uint32_t j = warp >> 1, half = warp & 1;
  const uint32_t t0 = blockIdx.y * kTileTokens + j * 32;  // First token of the unit.
  const uint32_t c0 = blockIdx.x * 128 + half * 64;       // First column of the unit.

  uint32_t v[8][4];
  if (!kKV) {
    const uint32_t d0 = c0 % kQKDim;
    const uint8_t *src = p.a + (static_cast<size_t>(t0 + r) * p.columns + c0) * 2;
    const size_t step = static_cast<size_t>(4) * p.columns * 2;
    if (d0 != kNopeDim) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint4 x = ld_stream(src + i * step + 16 * k);
        v[i][0] = x.x;
        v[i][1] = x.y;
        v[i][2] = x.z;
        v[i][3] = x.w;
      }
    } else {
      uint2 left[8], right[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        left[i] = ld_stream2(src + i * step + 8 * k);
        right[i] = ld_stream2(src + i * step + 64 + 8 * k);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const uint32_t pos = (t0 + 4 * i + r) / p.batch;
        const uint2 cl = __ldg(reinterpret_cast<const uint2 *>(p.cos + pos * kRopeDim + 4 * k));
        const uint2 sl = __ldg(reinterpret_cast<const uint2 *>(p.sin + pos * kRopeDim + 4 * k));
        const uint2 cr =
            __ldg(reinterpret_cast<const uint2 *>(p.cos + pos * kRopeDim + 32 + 4 * k));
        const uint2 sr =
            __ldg(reinterpret_cast<const uint2 *>(p.sin + pos * kRopeDim + 32 + 4 * k));
        rope_bwd4(left[i], right[i], cl, sl, cr, sr, v[i]);
      }
    }
  } else {
    const uint32_t h = c0 / kKVDim, d0 = c0 % kKVDim;
    const uint8_t *src;
    size_t step;
    if (d0 < kNopeDim) {
      src = p.a + ((static_cast<size_t>(t0 + r) * p.heads + h) * kQKDim + d0) * 2;
      step = static_cast<size_t>(4) * p.heads * kQKDim * 2;
    } else {
      src = p.b + ((static_cast<size_t>(t0 + r) * p.heads + h) * kValueDim + d0 - kNopeDim) * 2;
      step = static_cast<size_t>(4) * p.heads * kValueDim * 2;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 x = ld_stream(src + i * step + 16 * k);
      v[i][0] = x.x;
      v[i][1] = x.y;
      v[i][2] = x.z;
      v[i][3] = x.w;
    }
  }

  QuantizedUnit o;
  quantize(v, o);

  {
    const size_t offset = static_cast<size_t>(t0 + r) * p.columns + c0 + 8 * k;
    const size_t step = static_cast<size_t>(4) * p.columns;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<uint2 *>(p.out.rowwise_data + offset + i * step) = o.row[i];
      *reinterpret_cast<uint2 *>(p.out.columnwise_data + offset + i * step) = o.col[i];
    }
  }
  // Scales -> shared memory, swizzled 512-byte blocks:
  //   row-wise: token tl < 128, scale column c < 4: byte (tl % 32) * 16 + (tl / 32) * 4 + c;
  //   column-wise: column n < 128, token block jb < 4: byte (n % 32) * 16 + (n / 32) * 4 + jb.
  if (k == 0) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      *reinterpret_cast<uint16_t *>(sc + (4 * i + r) * 16 + j * 4 + half * 2) =
          static_cast<uint16_t>(o.rs[i]);
    }
  }
  if (r == 0) {
    const uint32_t bytes[2] = {o.cs.x, o.cs.y};
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      const uint32_t n = half * 64 + 8 * k + jj;
      sc[512 + (n % 32) * 16 + (n / 32) * 4 + j] =
          static_cast<uint8_t>(bytes[jj / 4] >> (8 * (jj % 4)));
    }
  }
  __syncthreads();
  if (threadIdx.x < 64) {
    const int vec = threadIdx.x & 31;
    const uint4 x = reinterpret_cast<const uint4 *>(sc + (threadIdx.x >> 5) * 512)[vec];
    uint8_t *dst =
        (threadIdx.x < 32)
            ? p.out.rowwise_scale_inv +
                  (static_cast<size_t>(blockIdx.y) * (p.columns / 128) + blockIdx.x) * 512
            : p.out.columnwise_scale_inv +
                  (static_cast<size_t>(blockIdx.x) * (p.tokens / kTileTokens) + blockIdx.y) * 512;
    reinterpret_cast<uint4 *>(dst)[vec] = x;
  }
#else
  NVTE_DEVICE_THREAD0_ERROR("Fused MLA RoPE backward with MXFP8 output requires SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// ---------------------------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------------------------

bool is_aligned_to(const void *ptr, const size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

void check_bf16_input(const Tensor &t, const char *name) {
  NVTE_CHECK(t.data.dtype == DType::kBFloat16, name, " must be BF16.");
  NVTE_CHECK(is_aligned_to(t.data.dptr, 16), name, " must be 16-byte aligned.");
}

void check_cos_sin(const Tensor &cos, const Tensor &sin, const size_t s) {
  check_bf16_input(cos, "cos");
  check_bf16_input(sin, "sin");
  NVTE_CHECK(cos.flat_last_dim() == kRopeDim && sin.flat_last_dim() == kRopeDim &&
                 cos.flat_first_dim() >= s && sin.flat_first_dim() >= s,
             "cos and sin must have at least ", s, " rows of ", kRopeDim, " elements, got shapes ",
             cos.data.shape, " and ", sin.data.shape, ".");
}

// An MXFP8 output with row-wise and column-wise E4M3 data and GEMM-swizzled scaling factors,
// whose flattened 2D shape is [rows, cols].
GradOutput check_swizzled_mxfp8_output(const Tensor &t, const char *name, const size_t rows,
                                       const size_t cols) {
  NVTE_CHECK(is_mxfp8_scaling(t.scaling_mode), name, " must be an MXFP8 tensor, got ",
             to_string(t.scaling_mode), ".");
  NVTE_CHECK(t.has_data() && t.has_columnwise_data(), name,
             " needs both row-wise and column-wise data.");
  NVTE_CHECK(t.data.dtype == DType::kFloat8E4M3 && t.columnwise_data.dtype == DType::kFloat8E4M3,
             name, " must have E4M3 data.");
  NVTE_CHECK(t.with_gemm_swizzled_scales, name, " must have GEMM-swizzled scaling factors.");
  const auto [first_dim, last_dim] = t.flat_2d_dims();
  NVTE_CHECK(first_dim == rows && last_dim == cols, name, " must have flattened shape [", rows,
             ", ", cols, "], got ", t.shape(), ".");
  CheckOutputTensor(t, name);
  GradOutput out{static_cast<uint8_t *>(t.data.dptr),
                 static_cast<uint8_t *>(t.columnwise_data.dptr),
                 static_cast<uint8_t *>(t.scale_inv.dptr),
                 static_cast<uint8_t *>(t.columnwise_scale_inv.dptr)};
  NVTE_CHECK(is_aligned_to(out.rowwise_data, 16) && is_aligned_to(out.columnwise_data, 16) &&
                 is_aligned_to(out.rowwise_scale_inv, 16) &&
                 is_aligned_to(out.columnwise_scale_inv, 16),
             name, " buffers must be 16-byte aligned.");
  return out;
}

AttnOutput to_attn_output(const GradOutput &o) {
  return AttnOutput{o.rowwise_data, o.columnwise_data, o.rowwise_scale_inv, o.columnwise_scale_inv};
}

template <bool kKV>
void launch_rope_mxfp8_attn(const RopeParams &p, cudaStream_t stream) {
  using L = RopeLayout<kKV>;
  const size_t num_tiles =
      static_cast<size_t>(p.num_columns / kHeadsPerTile) * (p.seq_len / kTileTokens);
  if (num_tiles == 0) return;
  auto kernel = fused_mla_rope_mxfp8_attn_kernel<kKV>;
  NVTE_CHECK_CUDA(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, L::kBytes));
  int ctas_per_sm = 0;
  NVTE_CHECK_CUDA(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas_per_sm, kernel, kThreads, L::kBytes));
  const size_t grid =
      std::min(static_cast<size_t>(cuda::sm_count()) * std::max(ctas_per_sm, 1), num_tiles);
  kernel<<<grid, kThreads, L::kBytes, stream>>>(p);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Checks of the forward inputs shared by query and key/value; returns (s, b, h).
std::array<size_t, 3> check_rope_input(const Tensor &x, const char *name, const size_t head_dim) {
  check_bf16_input(x, name);
  const auto &shape = x.data.shape;
  NVTE_CHECK(shape.size() == 4 && shape[3] == head_dim, name, " must have shape [s, b, h, ",
             head_dim, "], got ", shape, ".");
  const size_t s = shape[0], b = shape[1], h = shape[2];
  NVTE_CHECK(s % kTileTokens == 0, "The sequence length (", s, ") must be a multiple of ",
             kTileTokens, ".");
  NVTE_CHECK(h % kHeadsPerTile == 0, "The number of heads (", h, ") must be even.");
  // 32-bit offsets in the kernel.
  NVTE_CHECK(s * b * h * kKVDim * 2 < (static_cast<size_t>(1) << 31), name, " shape ", shape,
             " is too large.");
  return {s, b, h};
}

template <bool kKV>
void launch_rope_bwd_mxfp8(const GradParams &p, cudaStream_t stream) {
  if (p.tokens == 0) return;
  const dim3 grid(p.columns / 128, p.tokens / kTileTokens);
  fused_mla_rope_bwd_mxfp8_kernel<kKV><<<grid, kThreads, 0, stream>>>(p);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Checks of the backward gradients; returns (s, b, h).
std::array<size_t, 3> check_grad_input(const Tensor &x, const char *name, const size_t head_dim) {
  check_bf16_input(x, name);
  const auto &shape = x.data.shape;
  NVTE_CHECK(shape.size() == 4 && shape[3] == head_dim, name, " must have shape [s, b, h, ",
             head_dim, "], got ", shape, ".");
  const size_t s = shape[0], b = shape[1], h = shape[2];
  NVTE_CHECK((s * b) % kTileTokens == 0, "s * b (", s * b, ") must be a multiple of ", kTileTokens,
             ".");
  NVTE_CHECK(h % 2 == 0, "The number of heads (", h, ") must be even.");
  NVTE_CHECK(s * b * h * kKVDim < (static_cast<size_t>(1) << 31) && (s * b) / kTileTokens <= 65535,
             name, " shape ", shape, " is too large.");
  return {s, b, h};
}

}  // namespace

void fused_mla_q_rope_mxfp8_attn(const Tensor &q, const Tensor &cos, const Tensor &sin,
                                 Tensor *query, cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(), "Fused MLA query RoPE with MXFP8 output requires SM 10.0+.");
  const auto [s, b, h] = check_rope_input(q, "q", kQKDim);
  check_cos_sin(cos, sin, s);
  RopeParams p{};
  p.src = static_cast<const uint8_t *>(q.data.dptr);
  p.cos = static_cast<const uint16_t *>(cos.data.dptr);
  p.sin = static_cast<const uint16_t *>(sin.data.dptr);
  p.out0 = to_attn_output(check_swizzled_mxfp8_output(*query, "query", s * b * h, kQKDim));
  p.out1 = p.out0;
  p.num_columns = static_cast<uint32_t>(b * h);
  p.heads = static_cast<uint32_t>(h);
  p.seq_len = static_cast<uint32_t>(s);
  launch_rope_mxfp8_attn<false>(p, stream);
}

void fused_mla_kv_rope_mxfp8_attn(const Tensor &kv, const Tensor &k_rope, const int k_rope_stride,
                                  const Tensor &cos, const Tensor &sin, Tensor *key, Tensor *value,
                                  cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(),
             "Fused MLA key/value RoPE with MXFP8 output requires SM 10.0+.");
  const auto [s, b, h] = check_rope_input(kv, "kv", kKVDim);
  check_bf16_input(k_rope, "k_rope");
  NVTE_CHECK(k_rope.flat_first_dim() == s * b && k_rope.flat_last_dim() == kRopeDim,
             "k_rope must have s * b = ", s * b, " rows of ", kRopeDim, " elements, got shape ",
             k_rope.data.shape, ".");
  NVTE_CHECK(k_rope_stride >= kRopeDim && k_rope_stride % 8 == 0, "The k_rope row stride (",
             k_rope_stride, ") must be at least the row size and a multiple of 8 elements.");
  NVTE_CHECK(static_cast<size_t>(k_rope_stride) * 2 * s * b < (static_cast<size_t>(1) << 31),
             "k_rope is too large.");
  check_cos_sin(cos, sin, s);
  RopeParams p{};
  p.src = static_cast<const uint8_t *>(kv.data.dptr);
  p.k_rope = static_cast<const uint8_t *>(k_rope.data.dptr);
  p.k_rope_row_bytes = static_cast<uint32_t>(k_rope_stride) * 2;
  p.cos = static_cast<const uint16_t *>(cos.data.dptr);
  p.sin = static_cast<const uint16_t *>(sin.data.dptr);
  p.out0 = to_attn_output(check_swizzled_mxfp8_output(*key, "key", s * b * h, kQKDim));
  p.out1 = to_attn_output(check_swizzled_mxfp8_output(*value, "value", s * b * h, kValueDim));
  p.num_columns = static_cast<uint32_t>(b * h);
  p.heads = static_cast<uint32_t>(h);
  p.seq_len = static_cast<uint32_t>(s);
  launch_rope_mxfp8_attn<true>(p, stream);
}

void fused_mla_q_rope_backward(Tensor *dq, const Tensor &cos, const Tensor &sin,
                               cudaStream_t stream) {
  check_bf16_input(*dq, "dq");
  const auto &shape = dq->data.shape;
  NVTE_CHECK(shape.size() == 4 && shape[3] == kQKDim, "dq must have shape [s, b, h, ", kQKDim,
             "], got ", shape, ".");
  const size_t s = shape[0], b = shape[1], h = shape[2];
  NVTE_CHECK(s * b * h * kQKDim * 2 < (static_cast<size_t>(1) << 32), "dq shape ", shape,
             " is too large.");
  check_cos_sin(cos, sin, s);
  const size_t rows = s * b * h;
  if (rows == 0) return;
  constexpr size_t kRowsPerCTA = kQBwdThreads / THREADS_PER_WARP * 4 * kQBwdRowsPerLane;
  const size_t grid = DIVUP(rows, kRowsPerCTA);
  fused_mla_q_rope_bwd_kernel<<<grid, kQBwdThreads, 0, stream>>>(
      static_cast<uint16_t *>(dq->data.dptr), static_cast<const uint16_t *>(cos.data.dptr),
      static_cast<const uint16_t *>(sin.data.dptr), static_cast<uint32_t>(rows),
      static_cast<uint32_t>(b * h));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_mla_q_rope_backward_mxfp8(const Tensor &dq, const Tensor &cos, const Tensor &sin,
                                     Tensor *grad, cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(),
             "Fused MLA query RoPE backward with MXFP8 output requires SM 10.0+.");
  const auto [s, b, h] = check_grad_input(dq, "dq", kQKDim);
  check_cos_sin(cos, sin, s);
  GradParams p{};
  p.a = static_cast<const uint8_t *>(dq.data.dptr);
  p.cos = static_cast<const uint16_t *>(cos.data.dptr);
  p.sin = static_cast<const uint16_t *>(sin.data.dptr);
  p.out = check_swizzled_mxfp8_output(*grad, "grad", s * b, h * kQKDim);
  p.tokens = static_cast<uint32_t>(s * b);
  p.columns = static_cast<uint32_t>(h * kQKDim);
  p.heads = static_cast<uint32_t>(h);
  p.batch = static_cast<uint32_t>(b);
  launch_rope_bwd_mxfp8<false>(p, stream);
}

void fused_mla_kv_split_backward_mxfp8(const Tensor &dk, const Tensor &dv, Tensor *grad,
                                       cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(),
             "Fused MLA key/value split backward with MXFP8 output requires SM 10.0+.");
  const auto [s, b, h] = check_grad_input(dk, "dk", kQKDim);
  check_bf16_input(dv, "dv");
  NVTE_CHECK(dv.data.shape == std::vector<size_t>({s, b, h, static_cast<size_t>(kValueDim)}),
             "dv must have shape [", s, ", ", b, ", ", h, ", ", kValueDim, "], got ", dv.data.shape,
             ".");
  GradParams p{};
  p.a = static_cast<const uint8_t *>(dk.data.dptr);
  p.b = static_cast<const uint8_t *>(dv.data.dptr);
  p.out = check_swizzled_mxfp8_output(*grad, "grad", s * b, h * kKVDim);
  p.tokens = static_cast<uint32_t>(s * b);
  p.columns = static_cast<uint32_t>(h * kKVDim);
  p.heads = static_cast<uint32_t>(h);
  p.batch = static_cast<uint32_t>(b);
  launch_rope_bwd_mxfp8<true>(p, stream);
}

}  // namespace transformer_engine

void nvte_fused_mla_q_rope_mxfp8_attn(const NVTETensor q, const NVTETensor cos,
                                      const NVTETensor sin, NVTETensor query, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_q_rope_mxfp8_attn);
  using namespace transformer_engine;
  fused_mla_q_rope_mxfp8_attn(*convertNVTETensorCheck(q), *convertNVTETensorCheck(cos),
                              *convertNVTETensorCheck(sin), convertNVTETensorCheck(query), stream);
}

void nvte_fused_mla_kv_rope_mxfp8_attn(const NVTETensor kv, const NVTETensor k_rope,
                                       const int k_rope_stride, const NVTETensor cos,
                                       const NVTETensor sin, NVTETensor key, NVTETensor value,
                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_kv_rope_mxfp8_attn);
  using namespace transformer_engine;
  fused_mla_kv_rope_mxfp8_attn(*convertNVTETensorCheck(kv), *convertNVTETensorCheck(k_rope),
                               k_rope_stride, *convertNVTETensorCheck(cos),
                               *convertNVTETensorCheck(sin), convertNVTETensorCheck(key),
                               convertNVTETensorCheck(value), stream);
}

void nvte_fused_mla_q_rope_backward(NVTETensor dq, const NVTETensor cos, const NVTETensor sin,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_q_rope_backward);
  using namespace transformer_engine;
  fused_mla_q_rope_backward(convertNVTETensorCheck(dq), *convertNVTETensorCheck(cos),
                            *convertNVTETensorCheck(sin), stream);
}

void nvte_fused_mla_q_rope_backward_mxfp8(const NVTETensor dq, const NVTETensor cos,
                                          const NVTETensor sin, NVTETensor grad,
                                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_q_rope_backward_mxfp8);
  using namespace transformer_engine;
  fused_mla_q_rope_backward_mxfp8(*convertNVTETensorCheck(dq), *convertNVTETensorCheck(cos),
                                  *convertNVTETensorCheck(sin), convertNVTETensorCheck(grad),
                                  stream);
}

void nvte_fused_mla_kv_split_backward_mxfp8(const NVTETensor dk, const NVTETensor dv,
                                            NVTETensor grad, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_kv_split_backward_mxfp8);
  using namespace transformer_engine;
  fused_mla_kv_split_backward_mxfp8(*convertNVTETensorCheck(dk), *convertNVTETensorCheck(dv),
                                    convertNVTETensorCheck(grad), stream);
}
