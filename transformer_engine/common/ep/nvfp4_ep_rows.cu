/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_ep_rows.cu
 *  \brief NVFP4 token rows for the EP dispatch and combine backward.
 *
 *  The sender packs each token into one NVFP4 row (E2M1 codes, E4M3 block scales and a
 *  power-of-two FP32 token scale); the rows travel as an unscaled payload. The receiver either
 *  decodes them to BF16 or, with the direct receive, builds the grouped GEMM's NVFP4 operands
 *  from them in one pass:
 *
 *  - Rowwise: the sender's E2M1 codes are reused unchanged. With power-of-two token scales each
 *    block scale is only re-expressed against the per-expert scale G = max token scale, an exact
 *    power-of-two shift while it stays in E4M3's normal range.
 *  - Columnwise: the 16-point RHT along tokens is decoded and quantized in the same pass. Its
 *    second-level scale is the bound 4 * G: |RHT(x)| <= 4 max|x| and max|x| <= 6 * 448 * G, so
 *    no amax pass over the data is needed.
 *
 *  The same receive serves the combine backward (FC2's grad_output): the sender packs with
 *  stochastic rounding and the receiver quantizes the columnwise operand with stochastic
 *  rounding too, as the NVFP4 recipe does for gradients.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <transformer_engine/ep.h>

#include <algorithm>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/curanddx.hpp"
#include "../util/logging.h"
#include "../util/ptx.cuh"
#include "../utils.cuh"

namespace transformer_engine {
namespace ep_nvfp4 {
namespace {

constexpr int kTileM = 128;  // tokens per receive tile (8 RHT blocks of 16)
constexpr int kTileK = 256;  // hidden columns per receive tile (16 scale blocks of 16)
constexpr int kThreads = 256;
constexpr int kMaxGroups = 128;
constexpr float kNvfp4Max = 6.0f * 448.0f;

using Philox = curanddx::detail::philox4x32_native_state<NVTE_BUILD_NUM_PHILOX_ROUNDS>;

size_t min_row_bytes(size_t hidden) { return hidden / 2 + hidden / 16 + 4; }

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

__device__ __forceinline__ uint8_t float_to_e4m3_satfinite(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

#if FP4_TYPE_SUPPORTED

// Low nibble = lo; RN, satfinite.
__device__ __forceinline__ uint8_t float2_to_e2m1x2(float lo, float hi) {
  return static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(make_float2(lo, hi), __NV_E2M1, cudaRoundNearest));
}

// Low nibble = v.x; RN, satfinite.
__device__ __forceinline__ uint8_t half2_to_e2m1x2(__half2 v) {
  return static_cast<uint8_t>(
      __nv_cvt_halfraw2_to_fp4x2(static_cast<__half2_raw>(v), __NV_E2M1, cudaRoundNearest));
}

// Stochastic rounding of 4 floats to E2M1 (satfinite); x0 lands in the low nibble.
__device__ __forceinline__ uint16_t float4_to_e2m1x4_sr(float x0, float x1, float x2, float x3,
                                                        uint32_t rbits) {
  uint16_t out = 0;
  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
                 : "=h"(out)
                 : "f"(x3), "f"(x2), "f"(x1), "f"(x0), "r"(rbits));
  } else {
    NVTE_DEVICE_ERROR("Stochastic rounding to E2M1 requires SM 10.0, 10.3 or 10.7.");
  }
  return out;
}

#endif  // FP4_TYPE_SUPPORTED

__device__ __forceinline__ Philox make_rng(const int64_t *rng_state, uint64_t subsequence) {
  Philox rng;
  rng.init(static_cast<size_t>(rng_state[0]), subsequence, static_cast<size_t>(rng_state[1]));
  return rng;
}

template <bool kFastMath>
__device__ __forceinline__ float reciprocal(float x) {
  float r;
  if constexpr (kFastMath) {
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  } else {
    r = 1.0f / x;
  }
  return fminf(r, 3.4028234663852886e38f);
}

// Offset of scale (row r, scale col c) in the GEMM-swizzled 128x4 tile layout.
__device__ __forceinline__ int64_t swizzled_scale_offset(int64_t r, int64_t c,
                                                         int64_t num_scale_cols) {
  return ((r / 128) * ((num_scale_cols + 3) / 4) + c / 4) * 512 + (r % 32) * 16 +
         ((r / 32) % 4) * 4 + (c % 4);
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async4(void *smem, const void *gmem) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

struct Groups {
  int num;
  int64_t ends[kMaxGroups];  // inclusive prefix sums of the counts
};

__device__ __forceinline__ void load_groups(const int64_t *counts, int num_groups, Groups &s) {
  if (threadIdx.x == 0) {
    int64_t acc = 0;
    for (int g = 0; g < num_groups; ++g) {
      acc += counts[g];
      s.ends[g] = acc;
    }
    s.num = num_groups;
  }
  __syncthreads();
}

__device__ __forceinline__ int group_of_row(const Groups &s, int64_t r) {
  int g = 0;
  while (g < s.num - 1 && s.ends[g] <= r) ++g;
  return g;
}

__device__ __forceinline__ float group_scale(const int *gmax_bits, int g) {
  const float G = __int_as_float(gmax_bits[g]);
  return G > 0.0f ? G : 1.0f;
}

// Sender pack: one block per token row, one thread per 16-element scale block (blockDim = H/16).
template <bool kFastMath, bool kStochastic>
__global__ void pack_kernel(const __nv_bfloat16 *__restrict__ x, uint8_t *__restrict__ out, int H,
                            int64_t row_bytes, const int64_t *__restrict__ rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && FP4_TYPE_SUPPORTED
  __shared__ float s_warp_max[32];
  const int64_t row = blockIdx.x;
  const int b = threadIdx.x;
  const __nv_bfloat16 *xr = x + row * H + b * 16;
  uint8_t *orow = out + row * row_bytes;
  float v[16];
  {
    const uint4 a = *reinterpret_cast<const uint4 *>(xr);
    const uint4 c = *reinterpret_cast<const uint4 *>(xr + 8);
    const __nv_bfloat162 *pa = reinterpret_cast<const __nv_bfloat162 *>(&a);
    const __nv_bfloat162 *pc = reinterpret_cast<const __nv_bfloat162 *>(&c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 fa = __bfloat1622float2(pa[i]);
      const float2 fc = __bfloat1622float2(pc[i]);
      v[2 * i] = fa.x;
      v[2 * i + 1] = fa.y;
      v[8 + 2 * i] = fc.x;
      v[8 + 2 * i + 1] = fc.y;
    }
  }
  float bmax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
  float m = bmax;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) s_warp_max[warp] = m;
  __syncthreads();
  if (warp == 0) {
    const int nw = (blockDim.x + 31) >> 5;
    float mm = lane < nw ? s_warp_max[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, o));
    if (lane == 0) s_warp_max[0] = mm;
  }
  __syncthreads();
  const float row_amax = s_warp_max[0];
  // Power-of-two token scale: the smallest 2^e >= amax / (6 * 448).
  const uint32_t ratio_bits = __float_as_uint(__fdiv_rn(row_amax, kNvfp4Max));
  const uint32_t exp_bits = ratio_bits & 0x7F800000u;
  uint32_t dec_bits = (ratio_bits & 0x7FFFFFu) == 0 ? exp_bits : exp_bits + 0x800000u;
  dec_bits = max(dec_bits, 0x800000u);
  const float global_decode = row_amax == 0.0f ? 0.0f : __uint_as_float(dec_bits);
  const float global_encode = row_amax == 0.0f ? 1.0f : __fdiv_rn(1.0f, global_decode);
  // Same block-scale arithmetic as TE's NVFP4 cast (core_nvfp4.cuh).
  const float bs_f = fminf(bmax * (global_encode * (1.0f / 6.0f)), 448.0f);
  const uint8_t bs = float_to_e4m3_satfinite(bs_f);
  const float rcp = reciprocal<kFastMath>(e4m3_to_float(bs) * global_decode);
  uint32_t lo = 0, hi = 0;
  if constexpr (kStochastic) {
    Philox rng = make_rng(rng_state, static_cast<uint64_t>(row) * blockDim.x + b);
    const uint4 rb = rng.generate4();
    lo = static_cast<uint32_t>(
             float4_to_e2m1x4_sr(v[0] * rcp, v[1] * rcp, v[2] * rcp, v[3] * rcp, rb.x)) |
         (static_cast<uint32_t>(
              float4_to_e2m1x4_sr(v[4] * rcp, v[5] * rcp, v[6] * rcp, v[7] * rcp, rb.y))
          << 16);
    hi = static_cast<uint32_t>(
             float4_to_e2m1x4_sr(v[8] * rcp, v[9] * rcp, v[10] * rcp, v[11] * rcp, rb.z)) |
         (static_cast<uint32_t>(
              float4_to_e2m1x4_sr(v[12] * rcp, v[13] * rcp, v[14] * rcp, v[15] * rcp, rb.w))
          << 16);
  } else {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lo |= static_cast<uint32_t>(float2_to_e2m1x2(v[2 * p] * rcp, v[2 * p + 1] * rcp)) << (8 * p);
      hi |= static_cast<uint32_t>(float2_to_e2m1x2(v[8 + 2 * p] * rcp, v[8 + 2 * p + 1] * rcp))
            << (8 * p);
    }
  }
  *reinterpret_cast<uint2 *>(orow + b * 8) = make_uint2(lo, hi);
  orow[H / 2 + b] = bs;
  const int64_t tail = H / 2 + H / 16;
  if (b == 0) *reinterpret_cast<float *>(orow + tail) = global_decode;
  for (int64_t i = tail + 4 + b; i < row_bytes; i += blockDim.x) orow[i] = 0;
#else
  NVTE_DEVICE_ERROR("NVFP4 EP pack requires SM 10.0+.");
#endif
}

// Decode: one thread per 16-element block of a row, grid-stride, bounded by sum(counts).
__global__ void __launch_bounds__(kThreads)
    unpack_kernel(const uint8_t *__restrict__ packed, const int64_t *__restrict__ counts,
                  int num_counts, int64_t num_rows, int H, int64_t row_bytes,
                  __nv_bfloat16 *__restrict__ out) {
  constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  __shared__ int64_t s_bound;
  if (threadIdx.x == 0) {
    int64_t bound = num_rows;
    if (counts != nullptr) {
      int64_t total = 0;
      for (int g = 0; g < num_counts; ++g) total += counts[g];
      bound = min(total, num_rows);
    }
    s_bound = bound;
  }
  __syncthreads();
  const int blocks_per_row = H / 16;
  const int64_t num_items = s_bound * blocks_per_row;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < num_items;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t r = i / blocks_per_row;
    const int b = static_cast<int>(i % blocks_per_row);
    const uint8_t *row = packed + r * row_bytes;
    const uint2 codes = *reinterpret_cast<const uint2 *>(row + b * 8);
    const float token_scale = *reinterpret_cast<const float *>(row + H / 2 + H / 16);
    const float scale = e4m3_to_float(row[H / 2 + b]) * token_scale;
    const uint32_t words[2] = {codes.x, codes.y};
    __nv_bfloat162 vals[8];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const uint32_t byte = (words[w] >> (8 * p)) & 0xFFu;
        const uint32_t c0 = byte & 0xFu, c1 = byte >> 4;
        const float v0 = (c0 & 8u ? -kE2M1[c0 & 7u] : kE2M1[c0 & 7u]) * scale;
        const float v1 = (c1 & 8u ? -kE2M1[c1 & 7u] : kE2M1[c1 & 7u]) * scale;
        vals[w * 4 + p] = __floats2bfloat162_rn(v0, v1);
      }
    }
    uint4 *dst = reinterpret_cast<uint4 *>(out + r * H + b * 16);
    dst[0] = *reinterpret_cast<const uint4 *>(&vals[0]);
    dst[1] = *reinterpret_cast<const uint4 *>(&vals[4]);
  }
}

// Receive pass 1: per-group max of the (non-negative) token decode scales, as float bits.
__global__ void group_max_kernel(const uint8_t *__restrict__ packed, const int64_t *counts,
                                 int num_groups, int64_t capacity, int64_t row_bytes,
                                 int64_t scale_offset, int *__restrict__ gmax_bits) {
  __shared__ Groups s;
  __shared__ int local_max[kMaxGroups];
  load_groups(counts, num_groups, s);
  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) local_max[g] = 0;
  __syncthreads();
  const int64_t total = min(s.ends[num_groups - 1], capacity);
  for (int64_t r = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; r < total;
       r += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float ts = *reinterpret_cast<const float *>(packed + r * row_bytes + scale_offset);
    atomicMax(&local_max[group_of_row(s, r)], __float_as_int(fmaxf(ts, 0.0f)));
  }
  __syncthreads();
  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
    if (local_max[g] != 0) atomicMax(&gmax_bits[g], local_max[g]);
  }
}

// Receive pass 2: rowwise copy + scale re-expression, and columnwise RHT quantization, per
// 128x256 tile. Values are handled relative to the expert scale G (x / G): the codes times the
// re-expressed block scales. The columnwise math runs in half2 (two adjacent hidden columns share
// every E2M1 byte, block scale and token scale), with the RHT's 1/4 and random signs folded into
// the per-token scale. Relative values are <= 4 * 6 * 448, well inside FP16.
template <bool kFastMath, bool kStochastic>
__global__ void __launch_bounds__(kThreads)
    direct_receive_kernel(const uint8_t *__restrict__ packed, const int64_t *counts, int num_groups,
                          int64_t capacity, int H, int64_t row_bytes,
                          const int *__restrict__ gmax_bits, uint32_t sign_mask,
                          const int64_t *__restrict__ rng_state, uint8_t *__restrict__ row_data,
                          uint8_t *__restrict__ row_scale, uint8_t *__restrict__ col_data,
                          uint8_t *__restrict__ col_scale, float *__restrict__ row_amax,
                          float *__restrict__ col_amax) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && FP4_TYPE_SUPPORTED
  __shared__ Groups s;
  // Double-buffered tile inputs, filled with cp.async one tile ahead.
  __shared__ __align__(16) uint8_t s_codes_buf[2][kTileM][kTileK / 2];
  __shared__ __align__(16) uint8_t s_bscale_buf[2][kTileM][kTileK / 16];
  __shared__ __align__(16) float s_tscale_buf[2][kTileM];
  __shared__ __align__(16) __half2 s_hscale[kTileM][kTileK / 16];  // b * ts / G * sign / 4, x2
  load_groups(counts, num_groups, s);
  if (blockIdx.x == 0) {
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
      const float G = group_scale(gmax_bits, g);
      row_amax[g] = kNvfp4Max * G;
      col_amax[g] = 4.0f * kNvfp4Max * G;
    }
  }

  const int64_t total = min(s.ends[num_groups - 1], capacity);
  const int col_tiles = H / kTileK;
  const int64_t num_tiles = (total / kTileM) * col_tiles;
  const int64_t num_scale_cols_row = H / 16;
  const int64_t tscale_offset = H / 2 + H / 16;

  auto issue_loads = [&](int64_t tile, int buf) {
    const int64_t lr0 = (tile / col_tiles) * kTileM;
    const int lc0 = static_cast<int>(tile % col_tiles) * kTileK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * kThreads;  // 0..1023
      const int rr = idx >> 3;                     // 8 x 16 B per row
      const int cb = (idx & 7) * 16;
      cp_async16(&s_codes_buf[buf][rr][cb], packed + (lr0 + rr) * row_bytes + lc0 / 2 + cb);
    }
    if (threadIdx.x < kTileM) {
      const int rr = threadIdx.x;
      cp_async16(&s_bscale_buf[buf][rr][0], packed + (lr0 + rr) * row_bytes + H / 2 + lc0 / 16);
    } else {
      const int rr = threadIdx.x - kTileM;
      cp_async4(&s_tscale_buf[buf][rr], packed + (lr0 + rr) * row_bytes + tscale_offset);
    }
    cp_async_commit();
  };

  int buf = 0;
  if (blockIdx.x < num_tiles) issue_loads(blockIdx.x, 0);
  for (int64_t t = blockIdx.x; t < num_tiles; t += gridDim.x, buf ^= 1) {
    if (t + gridDim.x < num_tiles) {
      issue_loads(t + gridDim.x, buf ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    auto &s_codes = s_codes_buf[buf];
    auto &s_bscale = s_bscale_buf[buf];
    auto &s_tscale = s_tscale_buf[buf];
    const int64_t r0 = (t / col_tiles) * kTileM;
    const int c0 = static_cast<int>(t % col_tiles) * kTileK;
    const int g = group_of_row(s, r0);
    const int64_t g_end = s.ends[g];
    const int64_t g_rows = g == 0 ? g_end : g_end - s.ends[g - 1];
    const int64_t g_off = g_end - g_rows;
    const float inv_G = 1.0f / group_scale(gmax_bits, g);  // exact: G is a power of two

    // The rowwise data is the sender's codes, unchanged.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * kThreads;
      const int rr = idx >> 3;
      const int cb = (idx & 7) * 16;
      *reinterpret_cast<uint4 *>(row_data + (r0 + rr) * (H / 2) + c0 / 2 + cb) =
          *reinterpret_cast<const uint4 *>(&s_codes[rr][cb]);
    }

    // Re-expressed block scales: rowwise E4M3 (swizzled, one 4-byte word per item) and the FP16
    // columnwise factors.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = threadIdx.x + i * kThreads;  // 0..511: 128 rows x 4 words
      const int rr = idx & (kTileM - 1);
      const int m = idx >> 7;
      const float ratio = s_tscale[rr] * inv_G;
      const float col_factor = (((sign_mask >> (rr & 15)) & 1) ? -0.25f : 0.25f);
      uint32_t word = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float b = e4m3_to_float(s_bscale[rr][m * 4 + j]) * ratio;
        word |= static_cast<uint32_t>(float_to_e4m3_satfinite(b)) << (8 * j);
        s_hscale[rr][m * 4 + j] = __float2half2_rn(b * col_factor);
      }
      const int64_t off = swizzled_scale_offset(r0 + rr, c0 / 16 + m * 4, num_scale_cols_row);
      *reinterpret_cast<uint32_t *>(row_scale + off) = word;
    }
    __syncthreads();

    // Columnwise: thread = 2 adjacent hidden columns x 4 RHT blocks of 16 tokens.
    {
      const int cp = threadIdx.x & 127;
      const int th = threadIdx.x >> 7;
      const int sc = cp >> 3;
      uint32_t out_x[8], out_y[8];
      uint8_t scl_x[4], scl_y[4];
      Philox rng;
      if constexpr (kStochastic) {
        rng = make_rng(rng_state, static_cast<uint64_t>(t) * kThreads + threadIdx.x);
      }
#pragma unroll
      for (int b4 = 0; b4 < 4; ++b4) {
        const int rbase = (th * 4 + b4) * 16;
        __half2 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int rr = rbase + i;
          const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
              static_cast<__nv_fp4x2_storage_t>(s_codes[rr][cp]), __NV_E2M1);
          v[i] = __hmul2(__half2(raw), s_hscale[rr][sc]);
        }
        // In-place 16-point Walsh-Hadamard butterfly.
#pragma unroll
        for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            if ((i & h) == 0) {
              const __half2 a = v[i], b = v[i + h];
              v[i] = __hadd2(a, b);
              v[i + h] = __hsub2(a, b);
            }
          }
        }
        __half2 m2 = __habs2(v[0]);
#pragma unroll
        for (int i = 1; i < 16; ++i) m2 = __hmax2(m2, __habs2(v[i]));
        const float2 bmax = __half22float2(m2);
        // Relative to G: block scale = bmax_rel / 24, value = y_rel / (4 * block scale).
        const uint8_t sbx = float_to_e4m3_satfinite(fminf(bmax.x * (1.0f / 24.0f), 448.0f));
        const uint8_t sby = float_to_e4m3_satfinite(fminf(bmax.y * (1.0f / 24.0f), 448.0f));
        scl_x[b4] = sbx;
        scl_y[b4] = sby;
        const __half2 rx = __float2half2_rn(reciprocal<kFastMath>(4.0f * e4m3_to_float(sbx)));
        const __half2 ry = __float2half2_rn(reciprocal<kFastMath>(4.0f * e4m3_to_float(sby)));
        if constexpr (kStochastic) {
          // 4 random words per 8 tokens of both columns.
          const __half2 r2 = __halves2half2(__low2half(rx), __low2half(ry));
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            const uint4 rb = rng.generate4();
            float fx[8], fy[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const float2 f = __half22float2(__hmul2(v[w * 8 + j], r2));
              fx[j] = f.x;
              fy[j] = f.y;
            }
            out_x[b4 * 2 + w] =
                static_cast<uint32_t>(float4_to_e2m1x4_sr(fx[0], fx[1], fx[2], fx[3], rb.x)) |
                (static_cast<uint32_t>(float4_to_e2m1x4_sr(fx[4], fx[5], fx[6], fx[7], rb.y))
                 << 16);
            out_y[b4 * 2 + w] =
                static_cast<uint32_t>(float4_to_e2m1x4_sr(fy[0], fy[1], fy[2], fy[3], rb.z)) |
                (static_cast<uint32_t>(float4_to_e2m1x4_sr(fy[4], fy[5], fy[6], fy[7], rb.w))
                 << 16);
          }
        } else {
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            uint32_t wx = 0, wy = 0;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const int i = w * 8 + p * 2;
              // Regroup (col c, col c+1) pairs into (token i, token i+1) pairs per column.
              const __half2 qx = __hmul2(__lows2half2(v[i], v[i + 1]), rx);
              const __half2 qy = __hmul2(__highs2half2(v[i], v[i + 1]), ry);
              wx |= static_cast<uint32_t>(half2_to_e2m1x2(qx)) << (8 * p);
              wy |= static_cast<uint32_t>(half2_to_e2m1x2(qy)) << (8 * p);
            }
            out_x[b4 * 2 + w] = wx;
            out_y[b4 * 2 + w] = wy;
          }
        }
      }
      const int64_t local = r0 - g_off;
      const int64_t k = c0 + 2 * cp;
      uint8_t *dst = col_data + g_off * (H / 2) + k * (g_rows / 2) + local / 2 + th * 32;
      *reinterpret_cast<uint4 *>(dst) = make_uint4(out_x[0], out_x[1], out_x[2], out_x[3]);
      *reinterpret_cast<uint4 *>(dst + 16) = make_uint4(out_x[4], out_x[5], out_x[6], out_x[7]);
      dst += g_rows / 2;
      *reinterpret_cast<uint4 *>(dst) = make_uint4(out_y[0], out_y[1], out_y[2], out_y[3]);
      *reinterpret_cast<uint4 *>(dst + 16) = make_uint4(out_y[4], out_y[5], out_y[6], out_y[7]);
      const int64_t blk = local / 16 + th * 4;
      const int64_t ncols = g_rows / 16;
      const uint32_t sx =
          scl_x[0] | (scl_x[1] << 8) | (scl_x[2] << 16) | (static_cast<uint32_t>(scl_x[3]) << 24);
      const uint32_t sy =
          scl_y[0] | (scl_y[1] << 8) | (scl_y[2] << 16) | (static_cast<uint32_t>(scl_y[3]) << 24);
      *reinterpret_cast<uint32_t *>(col_scale + g_off * (H / 16) +
                                    swizzled_scale_offset(k, blk, ncols)) = sx;
      *reinterpret_cast<uint32_t *>(col_scale + g_off * (H / 16) +
                                    swizzled_scale_offset(k + 1, blk, ncols)) = sy;
    }
    __syncthreads();
  }
#else
  NVTE_DEVICE_ERROR("NVFP4 EP direct receive requires SM 10.0+.");
#endif
}

const int64_t *optional_rng_state(const Tensor *rng_state) {
  if (rng_state == nullptr || rng_state->data.dptr == nullptr) return nullptr;
  NVTE_CHECK(rng_state->dtype() == DType::kInt64 && rng_state->numel() == 2,
             "rng_state must be int64 [2] (seed, offset).");
  const int arch = cuda::sm_arch();
  NVTE_CHECK(arch == 100 || arch == 103 || arch == 107,
             "Stochastic rounding to E2M1 requires SM 10.0, 10.3 or 10.7 (got SM ", arch, ").");
  return reinterpret_cast<const int64_t *>(rng_state->data.dptr);
}

void check_blackwell() {
  NVTE_CHECK(cuda::sm_arch() >= 100, "NVFP4 EP token rows require SM 10.0+.");
}

}  // namespace

void pack(const Tensor &input, Tensor &packed, const Tensor *rng_state, bool use_fast_math,
          cudaStream_t stream) {
  check_blackwell();
  NVTE_CHECK(input.dtype() == DType::kBFloat16, "NVFP4 EP pack input must be BF16.");
  NVTE_CHECK(packed.dtype() == DType::kByte, "NVFP4 EP packed rows must be kByte.");
  NVTE_CHECK(input.dim() == 2 && packed.dim() == 2, "NVFP4 EP pack expects 2D tensors.");
  const size_t T = input.flat_first_dim();
  const size_t H = input.flat_last_dim();
  NVTE_CHECK(H % 512 == 0 && H <= 16384, "NVFP4 EP pack requires hidden % 512 == 0 and ",
             "hidden <= 16384 (got ", H, ").");
  const size_t row_bytes = packed.flat_last_dim();
  NVTE_CHECK(packed.flat_first_dim() == T && row_bytes % 16 == 0 && row_bytes >= min_row_bytes(H),
             "NVFP4 EP packed rows must be [", T, ", >= ", min_row_bytes(H),
             " (multiple of 16)], got ", packed.shape(), ".");
  const int64_t *rng = optional_rng_state(rng_state);
  if (T == 0) return;
  const auto *x = reinterpret_cast<const __nv_bfloat16 *>(input.data.dptr);
  auto *out = reinterpret_cast<uint8_t *>(packed.data.dptr);
  auto launch = [&](auto kernel) {
    kernel<<<static_cast<unsigned>(T), static_cast<unsigned>(H / 16), 0, stream>>>(
        x, out, static_cast<int>(H), static_cast<int64_t>(row_bytes), rng);
  };
  if (use_fast_math) {
    rng != nullptr ? launch(pack_kernel<true, true>) : launch(pack_kernel<true, false>);
  } else {
    rng != nullptr ? launch(pack_kernel<false, true>) : launch(pack_kernel<false, false>);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void unpack(const Tensor &packed, const Tensor *counts, Tensor &output, cudaStream_t stream) {
  NVTE_CHECK(packed.dtype() == DType::kByte, "NVFP4 EP packed rows must be kByte.");
  NVTE_CHECK(output.dtype() == DType::kBFloat16, "NVFP4 EP unpack output must be BF16.");
  const size_t R = packed.flat_first_dim();
  const size_t H = output.flat_last_dim();
  const size_t row_bytes = packed.flat_last_dim();
  NVTE_CHECK(H % 64 == 0, "NVFP4 EP unpack requires hidden % 64 == 0 (got ", H, ").");
  NVTE_CHECK(output.flat_first_dim() == R, "NVFP4 EP unpack output must have ", R, " rows.");
  NVTE_CHECK(row_bytes % 16 == 0 && row_bytes >= min_row_bytes(H),
             "NVFP4 EP packed rows are too short for hidden ", H, ".");
  const int64_t *counts_ptr = nullptr;
  int num_counts = 0;
  if (counts != nullptr && counts->data.dptr != nullptr) {
    NVTE_CHECK(counts->dtype() == DType::kInt64, "counts must be int64.");
    counts_ptr = reinterpret_cast<const int64_t *>(counts->data.dptr);
    num_counts = static_cast<int>(counts->numel());
  }
  if (R == 0) return;
  const size_t items = R * (H / 16);
  const size_t max_grid = 4 * static_cast<size_t>(cuda::sm_count());
  const int grid =
      static_cast<int>(std::min(DIVUP(items, static_cast<size_t>(kThreads)), max_grid));
  unpack_kernel<<<grid, kThreads, 0, stream>>>(reinterpret_cast<const uint8_t *>(packed.data.dptr),
                                               counts_ptr, num_counts, static_cast<int64_t>(R),
                                               static_cast<int>(H), static_cast<int64_t>(row_bytes),
                                               reinterpret_cast<__nv_bfloat16 *>(output.data.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void direct_receive(const Tensor &packed, const Tensor &counts, Tensor &workspace,
                    uint32_t sign_mask, Tensor &row_data, Tensor &row_scale, Tensor &col_data,
                    Tensor &col_scale, Tensor &row_amax, Tensor &col_amax, const Tensor *rng_state,
                    size_t hidden, bool use_fast_math, cudaStream_t stream) {
  check_blackwell();
  NVTE_CHECK(packed.dtype() == DType::kByte && packed.dim() == 2,
             "NVFP4 EP packed rows must be 2D kByte.");
  NVTE_CHECK(counts.dtype() == DType::kInt64, "counts must be int64.");
  const int num_groups = static_cast<int>(counts.numel());
  NVTE_CHECK(num_groups >= 1 && num_groups <= kMaxGroups, "NVFP4 EP direct receive supports 1..",
             kMaxGroups, " groups (got ", num_groups, ").");
  NVTE_CHECK(workspace.dtype() == DType::kInt32 && workspace.numel() >= num_groups,
             "workspace must be int32 with at least ", num_groups, " elements.");
  NVTE_CHECK(hidden % kTileK == 0, "NVFP4 EP direct receive requires hidden % ", kTileK,
             " == 0 (got ", hidden, ").");
  const size_t capacity = packed.flat_first_dim();
  NVTE_CHECK(capacity % kTileM == 0, "NVFP4 EP direct receive requires the row capacity to be a ",
             "multiple of ", kTileM, " (got ", capacity, ").");
  const size_t row_bytes = packed.flat_last_dim();
  NVTE_CHECK(row_bytes % 16 == 0 && row_bytes >= min_row_bytes(hidden),
             "NVFP4 EP packed rows are too short for hidden ", hidden, ".");
  const size_t data_bytes = capacity * hidden / 2, scale_bytes = capacity * hidden / 16;
  for (const Tensor *t : {&row_data, &col_data}) {
    NVTE_CHECK(t->dtype() == DType::kByte && t->numel() >= data_bytes,
               "NVFP4 EP direct receive data outputs must be kByte with ", data_bytes, " bytes.");
  }
  for (const Tensor *t : {&row_scale, &col_scale}) {
    NVTE_CHECK(t->dtype() == DType::kByte && t->numel() >= scale_bytes,
               "NVFP4 EP direct receive scale outputs must be kByte with ", scale_bytes, " bytes.");
  }
  for (const Tensor *t : {&row_amax, &col_amax}) {
    NVTE_CHECK(t->dtype() == DType::kFloat32 && t->numel() >= num_groups,
               "NVFP4 EP direct receive amax outputs must be float32 [", num_groups, "].");
  }
  const int64_t *rng = optional_rng_state(rng_state);

  const auto *packed_ptr = reinterpret_cast<const uint8_t *>(packed.data.dptr);
  const auto *counts_ptr = reinterpret_cast<const int64_t *>(counts.data.dptr);
  auto *gmax = reinterpret_cast<int *>(workspace.data.dptr);
  NVTE_CHECK_CUDA(cudaMemsetAsync(gmax, 0, num_groups * sizeof(int), stream));
  const int num_sms = cuda::sm_count();
  group_max_kernel<<<2 * num_sms, kThreads, 0, stream>>>(
      packed_ptr, counts_ptr, num_groups, static_cast<int64_t>(capacity),
      static_cast<int64_t>(row_bytes), static_cast<int64_t>(hidden / 2 + hidden / 16), gmax);
  NVTE_CHECK_CUDA(cudaGetLastError());
  auto launch = [&](auto kernel) {
    int per_sm = 1;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
    const int grid = std::max(1, per_sm) * num_sms;
    kernel<<<grid, kThreads, 0, stream>>>(packed_ptr, counts_ptr, num_groups,
                                          static_cast<int64_t>(capacity), static_cast<int>(hidden),
                                          static_cast<int64_t>(row_bytes), gmax, sign_mask, rng,
                                          reinterpret_cast<uint8_t *>(row_data.data.dptr),
                                          reinterpret_cast<uint8_t *>(row_scale.data.dptr),
                                          reinterpret_cast<uint8_t *>(col_data.data.dptr),
                                          reinterpret_cast<uint8_t *>(col_scale.data.dptr),
                                          reinterpret_cast<float *>(row_amax.data.dptr),
                                          reinterpret_cast<float *>(col_amax.data.dptr));
  };
  if (use_fast_math) {
    rng != nullptr ? launch(direct_receive_kernel<true, true>)
                   : launch(direct_receive_kernel<true, false>);
  } else {
    rng != nullptr ? launch(direct_receive_kernel<false, true>)
                   : launch(direct_receive_kernel<false, false>);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace ep_nvfp4
}  // namespace transformer_engine

void nvte_ep_nvfp4_pack(const NVTETensor input, NVTETensor packed, const NVTETensor rng_state,
                        int use_fast_math, cudaStream_t stream) {
  NVTE_API_CALL(nvte_ep_nvfp4_pack);
  using namespace transformer_engine;
  ep_nvfp4::pack(*convertNVTETensorCheck(input), *convertNVTETensorCheck(packed),
                 convertNVTETensor(rng_state), use_fast_math != 0, stream);
}

void nvte_ep_nvfp4_unpack(const NVTETensor packed, const NVTETensor counts, NVTETensor output,
                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_ep_nvfp4_unpack);
  using namespace transformer_engine;
  ep_nvfp4::unpack(*convertNVTETensorCheck(packed), convertNVTETensor(counts),
                   *convertNVTETensorCheck(output), stream);
}

void nvte_ep_nvfp4_direct_receive(const NVTETensor packed, const NVTETensor counts,
                                  NVTETensor workspace, uint32_t sign_mask, NVTETensor row_data,
                                  NVTETensor row_scale, NVTETensor col_data, NVTETensor col_scale,
                                  NVTETensor row_amax, NVTETensor col_amax,
                                  const NVTETensor rng_state, size_t hidden, int use_fast_math,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_ep_nvfp4_direct_receive);
  using namespace transformer_engine;
  ep_nvfp4::direct_receive(*convertNVTETensorCheck(packed), *convertNVTETensorCheck(counts),
                           *convertNVTETensorCheck(workspace), sign_mask,
                           *convertNVTETensorCheck(row_data), *convertNVTETensorCheck(row_scale),
                           *convertNVTETensorCheck(col_data), *convertNVTETensorCheck(col_scale),
                           *convertNVTETensorCheck(row_amax), *convertNVTETensorCheck(col_amax),
                           convertNVTETensor(rng_state), hidden, use_fast_math != 0, stream);
}
