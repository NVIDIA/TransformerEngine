/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cross_entropy_mxfp8.cu
 *  \brief Single-rank cross entropy backward that writes the gradient of the logits as MXFP8.
 *
 *  The gradient of row r and column c is
 *    bf16((exp(x[r, c] - m[r]) / d[r] - (c == y[r])) * grad_output[r])
 *  (zero for ignored rows), computed with the FP32 instruction sequence of the Triton
 *  cross_entropy_backward_kernel, so that its BF16 value is bit-identical:
 *    sub, mul by log2(e), ex2.approx.f32, div.full.f32, sub 1 at the target, mul by grad_output,
 *    select 0 for ignored rows, cvt.rn.bf16.
 *  div.full.f32(a, d) is computed as rcp.approx(d) * a with d one per row. That is how div.full
 *  is implemented for divisors in [2^-126, 2^126]; outside that range both operands are scaled
 *  first (by 1/4 or 2^24), as div.full does. The softmax denominator d is at least 1 for every
 *  row with a finite maximum.
 *
 *  The BF16 gradient is then quantized as TE's MXFP8 cast quantizes it: row-wise 1x32 and
 *  column-wise 32x1 blocks, E8M0 scale = amax * (1 / max_norm) in FP32 rounded up to a power of
 *  two, data = cvt.rn.satfinite(value * 2^-scale). Scales are written either compact or in the
 *  GEMM-swizzled layout (with_gemm_swizzled_scales), so the GEMMs need no swizzle kernels.
 *
 *  One CTA handles 128 tokens x 128 vocabulary columns with 256 threads. The logits tile is
 *  staged in shared memory with cp.async and overwritten in place by its BF16 gradient, which
 *  the column-wise pass reads back. The tile's row-wise scales (128 tokens x 4 blocks) and its
 *  column-wise scales (128 columns x 4 token blocks) are each exactly one 512-byte swizzled
 *  scale tile.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/cross_entropy.h>

#include <cstdint>
#include <limits>

#include "../common.h"
#include "../util/logging.h"
#include "../util/ptx.cuh"
#include "../utils.cuh"

namespace transformer_engine {
namespace {

constexpr int kRows = 128;
constexpr int kCols = 128;
constexpr int kThreads = 256;
// Thread layout: 16 lanes x 8 columns cover a 128-column row, so a warp covers 2 rows per step
// and 16 rows in 8 steps; 8 warps cover 128 rows. Thread (warp w, lane l) handles columns
// 8 * (l % 16) .. + 7 of rows 16 * w + 2 * i + l / 16, i = 0..7.
constexpr int kSteps = 8;
constexpr int kMinBlocksPerSM = 4;

// Single-precision helpers with a fixed instruction, so that the compiler cannot contract or
// reorder them (the results must match the Triton kernel bit for bit).
__device__ __forceinline__ float ex2_approx(const float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float ex2_approx_ftz(const float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float div_full(const float a, const float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float fsub(const float a, const float b) {
  float y;
  asm("sub.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float fmul(const float a, const float b) {
  float y;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float rcp_approx_ftz(const float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// Pairs of FP32 values. sub.rn.f32x2 / mul.rn.f32x2 round each half as the scalar instruction
// does; they are used where the target supports them.
struct f32x2 {
  float lo, hi;
};
__device__ __forceinline__ f32x2 sub2(const f32x2 a, const f32x2 b) {
  constexpr bool kPacked = NVTE_CUDA_ARCH_MATCHES(ptx::FamilySpecific<100>);
  if constexpr (kPacked) {
    uint64_t y;
    asm("{\n\t.reg .b64 pa, pb;\n\t"
        "mov.b64 pa, {%1, %2};\n\tmov.b64 pb, {%3, %4};\n\t"
        "sub.rn.f32x2 %0, pa, pb;\n\t}"
        : "=l"(y)
        : "f"(a.lo), "f"(a.hi), "f"(b.lo), "f"(b.hi));
    f32x2 r;
    asm("mov.b64 {%0, %1}, %2;" : "=f"(r.lo), "=f"(r.hi) : "l"(y));
    return r;
  } else {
    return {fsub(a.lo, b.lo), fsub(a.hi, b.hi)};
  }
}
__device__ __forceinline__ f32x2 mul2(const f32x2 a, const f32x2 b) {
  constexpr bool kPacked = NVTE_CUDA_ARCH_MATCHES(ptx::FamilySpecific<100>);
  if constexpr (kPacked) {
    uint64_t y;
    asm("{\n\t.reg .b64 pa, pb;\n\t"
        "mov.b64 pa, {%1, %2};\n\tmov.b64 pb, {%3, %4};\n\t"
        "mul.rn.f32x2 %0, pa, pb;\n\t}"
        : "=l"(y)
        : "f"(a.lo), "f"(a.hi), "f"(b.lo), "f"(b.hi));
    f32x2 r;
    asm("mov.b64 {%0, %1}, %2;" : "=f"(r.lo), "=f"(r.hi) : "l"(y));
    return r;
  } else {
    return {fmul(a.lo, b.lo), fmul(a.hi, b.hi)};
  }
}

// BF16 pair (lo = even column) <-> FP32 pair.
__device__ __forceinline__ uint32_t to_bf16x2(const float lo, const float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ f32x2 from_bf16x2(const uint32_t v) {
  return {__uint_as_float(v << 16), __uint_as_float(v & 0xFFFF0000u)};
}

template <bool kE5M2>
__device__ __forceinline__ uint16_t to_fp8x2(const float lo, const float hi) {
  uint16_t r;
  if constexpr (kE5M2) {
    asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  } else {
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  }
  return r;
}

// E8M0 scale of a block with maximum magnitude amax: amax * (1 / max_norm), rounded up to a
// power of two (TE's MXFP8 cast). Non-finite values are not special-cased.
template <bool kE5M2>
__device__ __forceinline__ uint32_t scale_exponent(const float amax) {
  constexpr float kMaxNormRcp = kE5M2 ? 1.0f / 57344.0f : 1.0f / 448.0f;
  const uint32_t bits = __float_as_uint(fmul(amax, kMaxNormRcp));
  const uint32_t exponent = (bits >> 23) & 0xFF;
  const uint32_t mantissa = bits & 0x7FFFFF;
  const bool round_up =
      (mantissa > 0) && (exponent != 0xFE) && !((exponent == 0) && (mantissa <= 0x400000));
  return round_up ? exponent + 1 : exponent;
}

// 2^-(exponent - 127) as FP32.
__device__ __forceinline__ float scale_reciprocal(const uint32_t exponent) {
  return __uint_as_float(exponent == 254 ? 0x00400000u : ((254u - exponent) << 23));
}

// Byte of scale (r, c) within its 128 x 4 GEMM-swizzled scale tile.
__device__ __forceinline__ int swizzled_offset(const int r, const int c) {
  return (r % 32) * 16 + ((r % 128) / 32) * 4 + (c % 4);
}

struct Params {
  const __nv_bfloat16 *logits;
  const int64_t *target;
  const float *stats;
  const int64_t *n_non_ignore;
  const float *grad_output;
  int64_t grad_output_stride;
  uint8_t *rowwise_data;
  uint8_t *rowwise_scale_inv;
  uint8_t *columnwise_data;
  uint8_t *columnwise_scale_inv;
  int64_t n_rows;
  int64_t n_cols;
  int64_t ignore_idx;
  bool rowwise;
  bool columnwise;
  bool swizzled_scales;
};

template <bool kReduceLoss, bool kE5M2>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSM)
    cross_entropy_backward_mxfp8_kernel(const Params p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ float s_col_amax[kThreads / THREADS_PER_WARP][kCols];  // per warp (16 rows)
  __shared__ __align__(16) uint8_t s_rowwise_scales[512];
  __shared__ __align__(16) uint8_t s_columnwise_scales[512];
  // Logits tile, overwritten in place by its BF16 gradient. Each thread only touches its own
  // 16-byte pieces until the column-wise pass.
  __shared__ __align__(16) uint32_t s_tile[kRows * kCols / 2];

  const int warp = threadIdx.x / THREADS_PER_WARP;
  const int lane = threadIdx.x % THREADS_PER_WARP;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kRows;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kCols;
  const int c0 = 8 * (lane % 16);
  const int64_t col = col_base + c0;
  const int64_t n_cols = p.n_cols;

#pragma unroll
  for (int i = 0; i < kSteps; ++i) {
    const int rl = 16 * warp + 2 * i + lane / 16;
    const uint32_t dst =
        static_cast<uint32_t>(__cvta_generic_to_shared(&s_tile[(rl * kCols + c0) / 2]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
                 "l"(p.logits + (row_base + rl) * n_cols + col));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;" ::: "memory");

  float n_non_ignore = 1.f;
  if constexpr (kReduceLoss) n_non_ignore = static_cast<float>(*p.n_non_ignore);

  float col_amax[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) col_amax[j] = 0.f;

#pragma unroll
  for (int i = 0; i < kSteps; ++i) {
    const int rl = 16 * warp + 2 * i + lane / 16;
    const int64_t r = row_base + rl;
    const int64_t y = p.target[r];
    const float m = p.stats[2 * r];
    const float d = p.stats[2 * r + 1];
    float go = p.grad_output[r * p.grad_output_stride];
    if constexpr (kReduceLoss) go = div_full(go, n_non_ignore);
    const bool ignored = (y == p.ignore_idx);
    // < 8 if the target is one of this thread's columns.
    const uint64_t target_offset = static_cast<uint64_t>(y - col);
    // div.full.f32 operand scaling for divisors outside [2^-126, 2^126].
    const float div_scale = fabsf(d) > 8.50705917302346158658e+37f
                                ? 0.25f
                                : (fabsf(d) < 1.175494350822287508e-38f ? 16777216.f : 1.f);
    const float d_rcp = rcp_approx_ftz(fmul(d, div_scale));
    const f32x2 m2{m, m};
    const f32x2 log2e2{1.4426950408889634f, 1.4426950408889634f};
    const f32x2 d_rcp2{d_rcp, d_rcp};
    const f32x2 go2{go, go};

    const uint4 raw = *reinterpret_cast<const uint4 *>(&s_tile[(rl * kCols + c0) / 2]);
    const uint32_t raw_words[4] = {raw.x, raw.y, raw.z, raw.w};
    float t[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const f32x2 v = mul2(sub2(from_bf16x2(raw_words[j]), m2), log2e2);
      t[2 * j] = v.lo;
      t[2 * j + 1] = v.hi;
    }
    float t_min = t[0];
#pragma unroll
    for (int j = 1; j < 8; ++j) t_min = fminf(t_min, t[j]);

    uint32_t g[4];  // BF16 gradient pairs
    // Warp-uniform fast path: no operand scaling, no ignored row, no target column, and every
    // exponent >= -126, where ex2.approx.f32 gives a normal number (= the ftz form).
    const bool fast = (div_scale == 1.f) && !ignored && (target_offset >= 8) && (t_min >= -126.f);
    if (__all_sync(0xffffffffu, fast)) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const f32x2 e{ex2_approx_ftz(t[2 * j]), ex2_approx_ftz(t[2 * j + 1])};
        const f32x2 q = mul2(mul2(d_rcp2, e), go2);
        g[j] = to_bf16x2(q.lo, q.hi);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        f32x2 e{ex2_approx(t[2 * j]), ex2_approx(t[2 * j + 1])};
        if (div_scale != 1.f) e = {fmul(e.lo, div_scale), fmul(e.hi, div_scale)};
        e = mul2(d_rcp2, e);
        // x - 0 == x, so only the target column changes.
        if (target_offset == static_cast<uint64_t>(2 * j)) e.lo = fsub(e.lo, 1.f);
        if (target_offset == static_cast<uint64_t>(2 * j + 1)) e.hi = fsub(e.hi, 1.f);
        e = mul2(e, go2);
        if (ignored) e = {0.f, 0.f};
        g[j] = to_bf16x2(e.lo, e.hi);
      }
    }

    float row_amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const f32x2 v = from_bf16x2(g[j]);
      const float a0 = fabsf(v.lo), a1 = fabsf(v.hi);
      row_amax = fmaxf(row_amax, fmaxf(a0, a1));
      col_amax[2 * j] = fmaxf(col_amax[2 * j], a0);
      col_amax[2 * j + 1] = fmaxf(col_amax[2 * j + 1], a1);
    }
    *reinterpret_cast<uint4 *>(&s_tile[(rl * kCols + c0) / 2]) = make_uint4(g[0], g[1], g[2], g[3]);

    if (p.rowwise) {
      // A row-wise block (32 columns) is 4 lanes.
      row_amax = fmaxf(row_amax, __shfl_xor_sync(0xffffffffu, row_amax, 1));
      row_amax = fmaxf(row_amax, __shfl_xor_sync(0xffffffffu, row_amax, 2));
      const uint32_t exponent = scale_exponent<kE5M2>(row_amax);
      const float rcp = scale_reciprocal(exponent);
      const f32x2 rcp2{rcp, rcp};
      uint16_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const f32x2 v = mul2(from_bf16x2(g[j]), rcp2);
        q[j] = to_fp8x2<kE5M2>(v.lo, v.hi);
      }
      uint2 out;
      out.x = q[0] | (static_cast<uint32_t>(q[1]) << 16);
      out.y = q[2] | (static_cast<uint32_t>(q[3]) << 16);
      *reinterpret_cast<uint2 *>(p.rowwise_data + r * n_cols + col) = out;
      if ((lane % 4) == 0) {
        const int block = (lane % 16) / 4;  // 32-column block within the tile
        if (p.swizzled_scales) {
          s_rowwise_scales[swizzled_offset(rl, block)] = static_cast<uint8_t>(exponent);
        } else {
          p.rowwise_scale_inv[r * (n_cols / 32) + col_base / 32 + block] =
              static_cast<uint8_t>(exponent);
        }
      }
    }
  }

  if (p.columnwise) {
    // Lanes l and l + 16 hold different rows of the same columns.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      col_amax[j] = fmaxf(col_amax[j], __shfl_xor_sync(0xffffffffu, col_amax[j], 16));
      if (lane < 16) s_col_amax[warp][c0 + j] = col_amax[j];
    }
  }
  __syncthreads();

  if (p.columnwise) {
    // The 32-token block of this warp is warp / 2, made of the 16 rows of two warps.
    const int token_block = warp / 2;
    float rcp[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float amax =
          fmaxf(s_col_amax[2 * token_block][c0 + j], s_col_amax[2 * token_block + 1][c0 + j]);
      const uint32_t exponent = scale_exponent<kE5M2>(amax);
      rcp[j] = scale_reciprocal(exponent);
      if ((warp % 2) == 0 && lane < 16) {
        if (p.swizzled_scales) {
          s_columnwise_scales[swizzled_offset(c0 + j, token_block)] =
              static_cast<uint8_t>(exponent);
        } else {
          p.columnwise_scale_inv[(row_base / 32 + token_block) * n_cols + col + j] =
              static_cast<uint8_t>(exponent);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kSteps; ++i) {
      const int rl = 16 * warp + 2 * i + lane / 16;
      const uint4 v = *reinterpret_cast<const uint4 *>(&s_tile[(rl * kCols + c0) / 2]);
      const uint32_t words[4] = {v.x, v.y, v.z, v.w};
      uint16_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const f32x2 s = mul2(from_bf16x2(words[j]), f32x2{rcp[2 * j], rcp[2 * j + 1]});
        q[j] = to_fp8x2<kE5M2>(s.lo, s.hi);
      }
      uint2 out;
      out.x = q[0] | (static_cast<uint32_t>(q[1]) << 16);
      out.y = q[2] | (static_cast<uint32_t>(q[3]) << 16);
      *reinterpret_cast<uint2 *>(p.columnwise_data + (row_base + rl) * n_cols + col) = out;
    }
  }

  if (p.swizzled_scales) {
    __syncthreads();
    // Row-wise: the tile of token band blockIdx.y and 32-column blocks 4 * blockIdx.x .. + 3.
    // Column-wise: the tile of column band blockIdx.x and token blocks 4 * blockIdx.y .. + 3.
    if (p.rowwise && threadIdx.x < 32) {
      const int64_t tile = static_cast<int64_t>(blockIdx.y) * (n_cols / 128) + blockIdx.x;
      reinterpret_cast<uint4 *>(p.rowwise_scale_inv + tile * 512)[threadIdx.x] =
          reinterpret_cast<const uint4 *>(s_rowwise_scales)[threadIdx.x];
    } else if (p.columnwise && threadIdx.x >= 32 && threadIdx.x < 64) {
      const int64_t tile = static_cast<int64_t>(blockIdx.x) * (p.n_rows / 128) + blockIdx.y;
      reinterpret_cast<uint4 *>(p.columnwise_scale_inv + tile * 512)[threadIdx.x - 32] =
          reinterpret_cast<const uint4 *>(s_columnwise_scales)[threadIdx.x - 32];
    }
  }
#else
  NVTE_DEVICE_THREAD0_ERROR("Cross entropy backward with MXFP8 output requires SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

bool is_aligned_to(const void *ptr, const size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

}  // namespace

void cross_entropy_backward_mxfp8(const Tensor &logits, const Tensor &target, const Tensor &stats,
                                  const Tensor *n_non_ignore, const Tensor &grad_output,
                                  Tensor *grad, const int64_t ignore_idx, const bool reduce_loss,
                                  cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(),
             "Cross entropy backward with MXFP8 output requires SM 10.0+.");
  NVTE_CHECK(logits.data.dtype == DType::kBFloat16, "logits must be BF16.");
  const size_t n_rows = logits.flat_first_dim();
  const size_t n_cols = logits.flat_last_dim();
  NVTE_CHECK(n_rows % kRows == 0 && n_cols % kCols == 0, "The number of rows (", n_rows,
             ") and the vocabulary size (", n_cols, ") must be multiples of 128.");
  NVTE_CHECK(n_rows / kRows <= 65535 && n_cols / kCols <= std::numeric_limits<int>::max(),
             "logits shape ", logits.data.shape, " is too large.");
  NVTE_CHECK(target.data.dtype == DType::kInt64 && target.numel() == n_rows,
             "target must hold one int64 per row.");
  NVTE_CHECK(stats.data.dtype == DType::kFloat32 && stats.numel() == 2 * n_rows,
             "stats must hold the FP32 (max, sum of exponentials) of every row.");
  NVTE_CHECK(grad_output.data.dtype == DType::kFloat32, "grad_output must be FP32.");
  if (reduce_loss) {
    NVTE_CHECK(grad_output.numel() == 1, "With reduce_loss, grad_output must be a scalar.");
    NVTE_CHECK(n_non_ignore != nullptr && n_non_ignore->data.dtype == DType::kInt64 &&
                   n_non_ignore->numel() == 1,
               "With reduce_loss, n_non_ignore must be one int64.");
  } else {
    NVTE_CHECK(grad_output.numel() == n_rows, "grad_output must hold one value per row.");
  }

  NVTE_CHECK(is_mxfp8_scaling(grad->scaling_mode), "grad must be an MXFP8 tensor, got ",
             to_string(grad->scaling_mode), ".");
  const bool rowwise = grad->has_data();
  const bool columnwise = grad->has_columnwise_data();
  NVTE_CHECK(rowwise || columnwise, "grad needs row-wise or column-wise data.");
  const DType fp8_dtype = rowwise ? grad->data.dtype : grad->columnwise_data.dtype;
  NVTE_CHECK(fp8_dtype == DType::kFloat8E4M3 || fp8_dtype == DType::kFloat8E5M2,
             "grad must have E4M3 or E5M2 data.");
  NVTE_CHECK(!(rowwise && columnwise) || grad->data.dtype == grad->columnwise_data.dtype,
             "grad must have the same FP8 type row-wise and column-wise.");
  const auto [grad_rows, grad_cols] = grad->flat_2d_dims();
  NVTE_CHECK(grad_rows == n_rows && grad_cols == n_cols, "grad must have the shape of logits, ",
             logits.data.shape, ", got ", grad->shape(), ".");
  CheckOutputTensor(*grad, "grad");

  Params p{};
  p.logits = static_cast<const __nv_bfloat16 *>(logits.data.dptr);
  p.target = static_cast<const int64_t *>(target.data.dptr);
  p.stats = static_cast<const float *>(stats.data.dptr);
  p.n_non_ignore = reduce_loss ? static_cast<const int64_t *>(n_non_ignore->data.dptr) : nullptr;
  p.grad_output = static_cast<const float *>(grad_output.data.dptr);
  p.grad_output_stride = reduce_loss ? 0 : 1;
  p.rowwise_data = rowwise ? static_cast<uint8_t *>(grad->data.dptr) : nullptr;
  p.rowwise_scale_inv = rowwise ? static_cast<uint8_t *>(grad->scale_inv.dptr) : nullptr;
  p.columnwise_data = columnwise ? static_cast<uint8_t *>(grad->columnwise_data.dptr) : nullptr;
  p.columnwise_scale_inv =
      columnwise ? static_cast<uint8_t *>(grad->columnwise_scale_inv.dptr) : nullptr;
  p.n_rows = static_cast<int64_t>(n_rows);
  p.n_cols = static_cast<int64_t>(n_cols);
  p.ignore_idx = ignore_idx;
  p.rowwise = rowwise;
  p.columnwise = columnwise;
  p.swizzled_scales = grad->with_gemm_swizzled_scales;
  NVTE_CHECK(is_aligned_to(p.logits, 16) && is_aligned_to(p.rowwise_data, 8) &&
                 is_aligned_to(p.columnwise_data, 8) && is_aligned_to(p.rowwise_scale_inv, 16) &&
                 is_aligned_to(p.columnwise_scale_inv, 16),
             "logits must be 16-byte aligned, and grad's data 8-byte and scales 16-byte aligned.");
  if (n_rows == 0 || n_cols == 0) {
    return;
  }

  const dim3 grid(static_cast<unsigned>(n_cols / kCols), static_cast<unsigned>(n_rows / kRows));
  TRANSFORMER_ENGINE_SWITCH_CONDITION(reduce_loss, kReduceLoss, {
    TRANSFORMER_ENGINE_SWITCH_CONDITION(fp8_dtype == DType::kFloat8E5M2, kE5M2, {
      cross_entropy_backward_mxfp8_kernel<kReduceLoss, kE5M2><<<grid, kThreads, 0, stream>>>(p);
    });  // NOLINT(*)
  });    // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_cross_entropy_backward_mxfp8(const NVTETensor logits, const NVTETensor target,
                                       const NVTETensor stats, const NVTETensor n_non_ignore,
                                       const NVTETensor grad_output, NVTETensor grad,
                                       const int64_t ignore_idx, const bool reduce_loss,
                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_cross_entropy_backward_mxfp8);
  using namespace transformer_engine;
  cross_entropy_backward_mxfp8(*convertNVTETensorCheck(logits), *convertNVTETensorCheck(target),
                               *convertNVTETensorCheck(stats), convertNVTETensor(n_non_ignore),
                               *convertNVTETensorCheck(grad_output), convertNVTETensorCheck(grad),
                               ignore_idx, reduce_loss, stream);
}
