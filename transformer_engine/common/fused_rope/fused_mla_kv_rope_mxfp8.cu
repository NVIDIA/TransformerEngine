/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_mla_kv_rope_mxfp8.cu
 *  \brief Multi-latent attention key/value split and key RoPE fused with MXFP8 quantization.
 *
 *  MLA's key/value up-projection gives, per head, kv = [k_nope | v], and the key is k_nope
 *  followed by k_rope, the key's RoPE dimensions that all heads share, with RoPE applied. For
 *  MXFP8 attention, key and value are quantized row-wise (32 elements of a head) and
 *  column-wise (32 tokens). Otherwise the key and value are written in BF16 and read again
 *  by the quantization; this kernel reads kv and k_rope once and writes the MXFP8 key and
 *  value directly, the same bytes as the MXFP8 quantizer applied to the [s, b * h * d] views
 *  of the RoPE'd key and of the value.
 *
 *  One CTA covers 32 tokens of two heads of one batch entry. It issues all of its global
 *  loads up front, stages the tile (RoPE applied) in shared memory, quantizes it with six
 *  warps row-wise and three warps column-wise, stages the results in shared memory in the
 *  layout of the CTA's part of the outputs and writes them with 16-byte stores.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_rope.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"
#include "../util/ptx.cuh"
#include "../util/ptx_arch_spec.cuh"
#include "../utils.cuh"

namespace transformer_engine {
namespace {

// Head dimensions: key without RoPE, RoPE part of the key, value (DeepSeek-V3's MLA).
constexpr int kNopeDim = 128;
constexpr int kRopeDim = 64;
constexpr int kHalfRopeDim = kRopeDim / 2;
constexpr int kValueDim = 128;
constexpr int kKeyDim = kNopeDim + kRopeDim;
constexpr int kKVDim = kNopeDim + kValueDim;

constexpr int kBlock = 32;  // MXFP8 scaling block, and tokens per CTA.
constexpr int kHeadsPerCTA = 2;
constexpr int kRowwiseThreads = 6 * THREADS_PER_WARP;
constexpr int kColwiseThreads = 3 * THREADS_PER_WARP;
constexpr int kThreads = kRowwiseThreads + kColwiseThreads;
constexpr int kPitch = kKeyDim + 8;  // BF16 elements per staged token; the padding staggers banks.

// Global buffers of one MXFP8 output.
struct MXFP8Output {
  uint8_t *rowwise_data;
  uint8_t *rowwise_scale_inv;
  uint8_t *columnwise_data;
  uint8_t *columnwise_scale_inv;
};

// A quantized tile of a tensor with head dimension D, laid out like its part of the global
// outputs: token r holds the kHeadsPerCTA * D bytes of the CTA's heads.
template <int D>
struct StagedOutput {
  alignas(16) uint8_t rowwise_data[kBlock * kHeadsPerCTA * D];
  alignas(16) uint8_t columnwise_data[kBlock * kHeadsPerCTA * D];
  alignas(16) uint8_t rowwise_scale_inv[kBlock * kHeadsPerCTA * (D / kBlock)];
  alignas(16) uint8_t columnwise_scale_inv[kHeadsPerCTA * D];
};

constexpr size_t kTileBytes = kHeadsPerCTA * kBlock * kPitch * sizeof(bf16);
constexpr size_t kSharedMemoryBytes = kTileBytes + sizeof(StagedOutput<kKeyDim>);

__device__ __forceinline__ bf16 bf16_from_bits(const uint32_t bits) {
  return __ushort_as_bfloat16(static_cast<uint16_t>(bits & 0xFFFFu));
}

__device__ __forceinline__ uint32_t bf16_bits(const bf16 x) { return __bfloat16_as_ushort(x); }

// Largest magnitude in a BF16 pair of maxima, as FP32 (NaNs are ignored, as by the quantizer).
__device__ __forceinline__ float bf16x2_max(const __nv_bfloat162 m) {
  return fmaxf(__bfloat162float(m.x), __bfloat162float(m.y));
}

__device__ __forceinline__ __nv_bfloat162 abs_max(const __nv_bfloat162 m, const uint32_t pair) {
  return __hmax2(m, __habs2(*reinterpret_cast<const __nv_bfloat162 *>(&pair)));
}

// kBlock tokens x W BF16 elements without RoPE, token rows `stride` elements apart, loaded as
// 16-byte vectors before any of them is used.
template <int W>
struct PlainLoads {
  static constexpr int kVectorsPerToken = W / 8;
  static constexpr int kTasks = kBlock * kVectorsPerToken;
  static constexpr int kIterations = (kTasks + kThreads - 1) / kThreads;
  uint4 vectors[kIterations];

  __device__ __forceinline__ void load(const bf16 *src, const int64_t stride) {
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int task = threadIdx.x + it * kThreads;
      if (task < kTasks) {
        const int token = task / kVectorsPerToken, vector = task % kVectorsPerToken;
        vectors[it] = __ldg(reinterpret_cast<const uint4 *>(src + token * stride) + vector);
      }
    }
  }

  __device__ __forceinline__ void store(bf16 *tile) const {
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int task = threadIdx.x + it * kThreads;
      if (task < kTasks) {
        const int token = task / kVectorsPerToken, vector = task % kVectorsPerToken;
        *reinterpret_cast<uint4 *>(tile + token * kPitch + 8 * vector) = vectors[it];
      }
    }
  }
};

// RoPE of k_rope for kBlock tokens, one task per (token, group of 4 rotation pairs).
template <bool kCosBf16>
struct RopeLoads {
  using CosType = std::conditional_t<kCosBf16, bf16, float>;
  using CosVector = std::conditional_t<kCosBf16, uint2, float4>;  // 4 values
  static constexpr int kGroups = kHalfRopeDim / 4;
  static constexpr int kTasks = kBlock * kGroups;
  static_assert(kTasks <= kThreads);
  uint4 pairs;  // 4 interleaved (x1, x2) pairs
  CosVector cos_left, cos_right, sin_left, sin_right;

  __device__ __forceinline__ bool active() const { return threadIdx.x < kTasks; }

  __device__ __forceinline__ void load(const bf16 *k_rope, const int64_t stride, const CosType *cos,
                                       const CosType *sin, const int64_t first_token) {
    if (!active()) return;
    const int token = threadIdx.x / kGroups, group = threadIdx.x % kGroups;
    pairs = __ldg(reinterpret_cast<const uint4 *>(k_rope + token * stride) + group);
    const CosType *c = cos + (first_token + token) * kRopeDim;
    const CosType *s = sin + (first_token + token) * kRopeDim;
    cos_left = __ldg(reinterpret_cast<const CosVector *>(c) + group);
    cos_right = __ldg(reinterpret_cast<const CosVector *>(c + kHalfRopeDim) + group);
    sin_left = __ldg(reinterpret_cast<const CosVector *>(s) + group);
    sin_right = __ldg(reinterpret_cast<const CosVector *>(s + kHalfRopeDim) + group);
  }

  // Writes x_left = x1 * cos_left - x2 * sin_left and x_right = x2 * cos_right + x1 * sin_right
  // to the tile's columns kNopeDim.. (all x_left, then all x_right), with the roundings of
  // Megatron-LM's MLA RoPE kernels: in BF16 for BF16 cos / sin (each product and FMA rounded),
  // in FP32 rounded once to BF16 for FP32 cos / sin.
  __device__ __forceinline__ void store(bf16 *tile) const {
    if (!active()) return;
    const int token = threadIdx.x / kGroups, group = threadIdx.x % kGroups;
    const uint32_t words[4] = {pairs.x, pairs.y, pairs.z, pairs.w};
    uint32_t left[4], right[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const bf16 x1 = bf16_from_bits(words[p]), x2 = bf16_from_bits(words[p] >> 16);
      if constexpr (kCosBf16) {
        const bf16 *cl = reinterpret_cast<const bf16 *>(&cos_left);
        const bf16 *cr = reinterpret_cast<const bf16 *>(&cos_right);
        const bf16 *sl = reinterpret_cast<const bf16 *>(&sin_left);
        const bf16 *sr = reinterpret_cast<const bf16 *>(&sin_right);
        left[p] = bf16_bits(__hfma(x1, cl[p], __hneg(__hmul(x2, sl[p]))));
        right[p] = bf16_bits(__hfma(x1, sr[p], __hmul(x2, cr[p])));
      } else {
        const float *cl = reinterpret_cast<const float *>(&cos_left);
        const float *cr = reinterpret_cast<const float *>(&cos_right);
        const float *sl = reinterpret_cast<const float *>(&sin_left);
        const float *sr = reinterpret_cast<const float *>(&sin_right);
        const float f1 = __bfloat162float(x1), f2 = __bfloat162float(x2);
        left[p] = bf16_bits(__float2bfloat16_rn(__fmaf_rn(f1, cl[p], -__fmul_rn(f2, sl[p]))));
        right[p] = bf16_bits(__float2bfloat16_rn(__fmaf_rn(f1, sr[p], __fmul_rn(f2, cr[p]))));
      }
    }
    bf16 *dst = tile + token * kPitch + kNopeDim + 4 * group;
    *reinterpret_cast<uint2 *>(dst) =
        make_uint2(left[0] | (left[1] << 16), left[2] | (left[3] << 16));
    *reinterpret_cast<uint2 *>(dst + kHalfRopeDim) =
        make_uint2(right[0] | (right[1] << 16), right[2] | (right[3] << 16));
  }
};

// Quantizes head h of the staged tile (kBlock tokens x D) into `out`, as the MXFP8 quantizer
// does: row-wise one (token, 32-element block) per task on the row-wise warps, column-wise two
// adjacent columns per task on the column-wise warps.
template <int D>
__device__ __forceinline__ void quantize_head(const bf16 *tile, StagedOutput<D> &out, const int h) {
  constexpr int kBlocksPerToken = D / kBlock;
  constexpr int kTokenBytes = kHeadsPerCTA * D;
  constexpr float kMaxNormRcp = Quantized_Limits<fp8e4m3>::max_norm_rcp;
  if (threadIdx.x < kRowwiseThreads) {
    for (int task = threadIdx.x; task < kBlock * kBlocksPerToken; task += kRowwiseThreads) {
      const int token = task % kBlock, block = task / kBlock;
      const uint4 *src = reinterpret_cast<const uint4 *>(tile + token * kPitch + kBlock * block);
      uint32_t words[16];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4 u = src[i];
        words[4 * i] = u.x;
        words[4 * i + 1] = u.y;
        words[4 * i + 2] = u.z;
        words[4 * i + 3] = u.w;
      }
      __nv_bfloat162 m = __habs2(*reinterpret_cast<const __nv_bfloat162 *>(&words[0]));
#pragma unroll
      for (int i = 1; i < 16; ++i) m = abs_max(m, words[i]);
      const e8m0_t biased_exponent = ptx::float_to_e8m0(bf16x2_max(m) * kMaxNormRcp);
      const ptx::bf16x2 scale = ptx::exp2f_rcp_2x(biased_exponent);
      uint32_t q[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        ptx::mul_cvt_4x(reinterpret_cast<ptx::fp8e4m3x4 &>(q[i]),
                        reinterpret_cast<const ptx::bf16x4 &>(words[2 * i]), scale);
      }
      uint4 *dst = reinterpret_cast<uint4 *>(out.rowwise_data + token * kTokenBytes + h * D +
                                             kBlock * block);
      dst[0] = make_uint4(q[0], q[1], q[2], q[3]);
      dst[1] = make_uint4(q[4], q[5], q[6], q[7]);
      out.rowwise_scale_inv[token * kHeadsPerCTA * kBlocksPerToken + h * kBlocksPerToken + block] =
          biased_exponent;
    }
  } else {
    for (int pair = threadIdx.x - kRowwiseThreads; pair < D / 2; pair += kColwiseThreads) {
      // Columns 2 * pair and 2 * pair + 1 of all kBlock tokens.
      const uint32_t *column = reinterpret_cast<const uint32_t *>(tile) + pair;
      uint32_t values[kBlock];
#pragma unroll
      for (int r = 0; r < kBlock; ++r) values[r] = column[r * (kPitch / 2)];
      __nv_bfloat162 m = __habs2(*reinterpret_cast<const __nv_bfloat162 *>(&values[0]));
#pragma unroll
      for (int r = 1; r < kBlock; ++r) m = abs_max(m, values[r]);
      const e8m0_t exponent0 = ptx::float_to_e8m0(__bfloat162float(m.x) * kMaxNormRcp);
      const e8m0_t exponent1 = ptx::float_to_e8m0(__bfloat162float(m.y) * kMaxNormRcp);
      const ptx::bf16x2 scale{ptx::exp2f_rcp<bf16>(exponent0), ptx::exp2f_rcp<bf16>(exponent1)};
#pragma unroll
      for (int r = 0; r < kBlock; r += 2) {
        uint32_t q;
        ptx::mul_cvt_4x(reinterpret_cast<ptx::fp8e4m3x4 &>(q),
                        reinterpret_cast<const ptx::bf16x2 &>(values[r]), scale,
                        reinterpret_cast<const ptx::bf16x2 &>(values[r + 1]), scale);
        uint8_t *dst = out.columnwise_data + r * kTokenBytes + h * D + 2 * pair;
        *reinterpret_cast<uint16_t *>(dst) = static_cast<uint16_t>(q);
        *reinterpret_cast<uint16_t *>(dst + kTokenBytes) = static_cast<uint16_t>(q >> 16);
      }
      *reinterpret_cast<uint16_t *>(out.columnwise_scale_inv + h * D + 2 * pair) =
          static_cast<uint16_t>(exponent0 | (exponent1 << 8));
    }
  }
}

// Writes the staged tile. first_row = first_token * num_columns + first_column is the tile's
// first (token, column) row of the [s * b * h, D] outputs; first_colwise_row is the matching
// row of the column-wise scales, [s / 32 * b * h, D].
template <int D>
__device__ __forceinline__ void write_out(const StagedOutput<D> &staged, const int64_t first_row,
                                          const int64_t num_columns,
                                          const int64_t first_colwise_row, const MXFP8Output &out) {
  constexpr int kTokenBytes = kHeadsPerCTA * D;
  constexpr int kVectorsPerToken = kTokenBytes / 16;
  constexpr int kScaleBytesPerToken = kHeadsPerCTA * (D / kBlock);
  static_assert(kTokenBytes % 16 == 0 && kScaleBytesPerToken % 4 == 0);
  constexpr int kScaleWordsPerToken = kScaleBytesPerToken / 4;
  for (int task = threadIdx.x; task < 2 * kBlock * kVectorsPerToken; task += kThreads) {
    const bool columnwise = task >= kBlock * kVectorsPerToken;
    const int t = task % (kBlock * kVectorsPerToken);
    const int token = t / kVectorsPerToken, vector = t % kVectorsPerToken;
    const uint8_t *src =
        (columnwise ? staged.columnwise_data : staged.rowwise_data) + token * kTokenBytes;
    uint8_t *dst = (columnwise ? out.columnwise_data : out.rowwise_data) +
                   (first_row + token * num_columns) * D;
    reinterpret_cast<uint4 *>(dst)[vector] = reinterpret_cast<const uint4 *>(src)[vector];
  }
  for (int task = threadIdx.x; task < kBlock * kScaleWordsPerToken; task += kThreads) {
    const int token = task / kScaleWordsPerToken, word = task % kScaleWordsPerToken;
    uint8_t *dst = out.rowwise_scale_inv + (first_row + token * num_columns) * (D / kBlock);
    reinterpret_cast<uint32_t *>(dst)[word] = reinterpret_cast<const uint32_t *>(
        staged.rowwise_scale_inv + token * kScaleBytesPerToken)[word];
  }
  for (int task = threadIdx.x; task < kTokenBytes / 16; task += kThreads) {
    reinterpret_cast<uint4 *>(out.columnwise_scale_inv + first_colwise_row * D)[task] =
        reinterpret_cast<const uint4 *>(staged.columnwise_scale_inv)[task];
  }
}

template <bool kCosBf16>
__global__ void __launch_bounds__(kThreads)
    fused_mla_kv_rope_mxfp8_kernel(const bf16 *__restrict__ kv, const bf16 *__restrict__ k_rope,
                                   const int64_t k_rope_stride,
                                   const typename RopeLoads<kCosBf16>::CosType *__restrict__ cos,
                                   const typename RopeLoads<kCosBf16>::CosType *__restrict__ sin,
                                   const MXFP8Output key, const MXFP8Output value,
                                   const int num_columns, const int heads) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ __align__(16) uint8_t shared_memory[];
  bf16 *tiles = reinterpret_cast<bf16 *>(shared_memory);  // [kHeadsPerCTA][kBlock][kPitch]
  auto &key_out = *reinterpret_cast<StagedOutput<kKeyDim> *>(shared_memory + kTileBytes);
  auto &value_out = *reinterpret_cast<StagedOutput<kValueDim> *>(shared_memory + kTileBytes);

  // Column c = batch * heads + head. blockIdx.x: kHeadsPerCTA columns of one batch entry
  // (kHeadsPerCTA divides heads), blockIdx.y: kBlock tokens.
  const int64_t columns = num_columns;
  const int64_t first_column = static_cast<int64_t>(blockIdx.x) * kHeadsPerCTA;
  const int64_t token_block = blockIdx.y;
  const int64_t first_token = token_block * kBlock;
  const int64_t batch_size = columns / heads;
  const int64_t batch = first_column / heads;

  PlainLoads<kNopeDim> k_nope[kHeadsPerCTA];
  PlainLoads<kValueDim> v[kHeadsPerCTA];
  RopeLoads<kCosBf16> rope;
#pragma unroll
  for (int h = 0; h < kHeadsPerCTA; ++h) {
    const bf16 *src = kv + (first_token * columns + first_column + h) * kKVDim;
    k_nope[h].load(src, columns * kKVDim);
    v[h].load(src + kNopeDim, columns * kKVDim);
  }
  rope.load(k_rope + (first_token * batch_size + batch) * k_rope_stride, batch_size * k_rope_stride,
            cos, sin, first_token);

  // Key: k_nope and the RoPE'd k_rope, which both heads share.
#pragma unroll
  for (int h = 0; h < kHeadsPerCTA; ++h) {
    k_nope[h].store(tiles + h * kBlock * kPitch);
    rope.store(tiles + h * kBlock * kPitch);
  }
  __syncthreads();
#pragma unroll 1
  for (int h = 0; h < kHeadsPerCTA; ++h) {
    quantize_head<kKeyDim>(tiles + h * kBlock * kPitch, key_out, h);
  }
  __syncthreads();
  write_out<kKeyDim>(key_out, first_token * columns + first_column, columns,
                     token_block * columns + first_column, key);
  __syncthreads();

  // Value.
#pragma unroll
  for (int h = 0; h < kHeadsPerCTA; ++h) v[h].store(tiles + h * kBlock * kPitch);
  __syncthreads();
#pragma unroll 1
  for (int h = 0; h < kHeadsPerCTA; ++h) {
    quantize_head<kValueDim>(tiles + h * kBlock * kPitch, value_out, h);
  }
  __syncthreads();
  write_out<kValueDim>(value_out, first_token * columns + first_column, columns,
                       token_block * columns + first_column, value);
#else
  NVTE_DEVICE_THREAD0_ERROR("Fused MLA key/value RoPE with MXFP8 output requires SM 10.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

bool is_aligned_to(const void *ptr, const size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

MXFP8Output check_mxfp8_output(const Tensor &t, const char *name, const size_t rows,
                               const size_t cols) {
  NVTE_CHECK(is_mxfp8_scaling(t.scaling_mode), name, " must be an MXFP8 tensor, got ",
             to_string(t.scaling_mode), ".");
  NVTE_CHECK(t.has_data() && t.has_columnwise_data(), name,
             " needs both row-wise and column-wise data.");
  NVTE_CHECK(t.data.dtype == DType::kFloat8E4M3 && t.columnwise_data.dtype == DType::kFloat8E4M3,
             name, " must have E4M3 data.");
  NVTE_CHECK(!t.with_gemm_swizzled_scales, name,
             " must have compact (not GEMM-swizzled) scaling factors.");
  const auto [first_dim, last_dim] = t.flat_2d_dims();
  NVTE_CHECK(first_dim == rows && last_dim == cols, name, " must have shape [", rows, ", ", cols,
             "], got ", t.shape(), ".");
  CheckOutputTensor(t, name);
  MXFP8Output out{static_cast<uint8_t *>(t.data.dptr), static_cast<uint8_t *>(t.scale_inv.dptr),
                  static_cast<uint8_t *>(t.columnwise_data.dptr),
                  static_cast<uint8_t *>(t.columnwise_scale_inv.dptr)};
  NVTE_CHECK(is_aligned_to(out.rowwise_data, 16) && is_aligned_to(out.columnwise_data, 16) &&
                 is_aligned_to(out.rowwise_scale_inv, 16) &&
                 is_aligned_to(out.columnwise_scale_inv, 16),
             name, " buffers must be 16-byte aligned.");
  return out;
}

}  // namespace

void fused_mla_kv_rope_mxfp8(const Tensor &kv, const Tensor &k_rope, const int k_rope_stride,
                             const Tensor &cos, const Tensor &sin, Tensor *key, Tensor *value,
                             cudaStream_t stream) {
  NVTE_CHECK(is_supported_by_CC_100(),
             "Fused MLA key/value RoPE with MXFP8 output requires SM 10.0+.");
  const auto &kv_shape = kv.data.shape;
  NVTE_CHECK(kv.data.dtype == DType::kBFloat16, "kv must be BF16.");
  NVTE_CHECK(kv_shape.size() == 4 && kv_shape[3] == kKVDim, "kv must have shape [s, b, h, ", kKVDim,
             "], got ", kv_shape, ".");
  const size_t s = kv_shape[0], b = kv_shape[1], h = kv_shape[2];
  NVTE_CHECK(s % kBlock == 0, "The sequence length (", s, ") must be a multiple of ", kBlock, ".");
  NVTE_CHECK(h % kHeadsPerCTA == 0, "The number of heads (", h, ") must be even.");
  NVTE_CHECK(s / kBlock <= 65535 && b * h <= static_cast<size_t>(std::numeric_limits<int>::max()),
             "kv shape ", kv_shape, " is too large.");

  NVTE_CHECK(k_rope.data.dtype == DType::kBFloat16, "k_rope must be BF16.");
  NVTE_CHECK(k_rope.flat_first_dim() == s * b && k_rope.flat_last_dim() == kRopeDim,
             "k_rope must have s * b = ", s * b, " rows of ", kRopeDim, " elements, got shape ",
             k_rope.data.shape, ".");
  NVTE_CHECK(k_rope_stride >= kRopeDim && k_rope_stride % 8 == 0, "The k_rope row stride (",
             k_rope_stride, ") must be at least the row size and a multiple of 8 elements.");

  NVTE_CHECK(cos.data.dtype == sin.data.dtype &&
                 (cos.data.dtype == DType::kBFloat16 || cos.data.dtype == DType::kFloat32),
             "cos and sin must both be BF16 or both FP32.");
  NVTE_CHECK(cos.flat_last_dim() == kRopeDim && sin.flat_last_dim() == kRopeDim &&
                 cos.flat_first_dim() >= s && sin.flat_first_dim() >= s,
             "cos and sin must have at least ", s, " rows of ", kRopeDim, " elements, got shapes ",
             cos.data.shape, " and ", sin.data.shape, ".");

  const MXFP8Output key_out = check_mxfp8_output(*key, "key", s, b * h * kKeyDim);
  const MXFP8Output value_out = check_mxfp8_output(*value, "value", s, b * h * kValueDim);
  NVTE_CHECK(is_aligned_to(kv.data.dptr, 16) && is_aligned_to(k_rope.data.dptr, 16) &&
                 is_aligned_to(cos.data.dptr, 16) && is_aligned_to(sin.data.dptr, 16),
             "kv, k_rope, cos and sin must be 16-byte aligned.");
  if (s == 0 || b == 0) {
    return;
  }

  const dim3 grid(b * h / kHeadsPerCTA, s / kBlock);
  TRANSFORMER_ENGINE_SWITCH_CONDITION(cos.data.dtype == DType::kBFloat16, kCosBf16, {
    using CosType = typename RopeLoads<kCosBf16>::CosType;
    auto kernel = fused_mla_kv_rope_mxfp8_kernel<kCosBf16>;
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         kSharedMemoryBytes));
    kernel<<<grid, kThreads, kSharedMemoryBytes, stream>>>(
        static_cast<const bf16 *>(kv.data.dptr), static_cast<const bf16 *>(k_rope.data.dptr),
        k_rope_stride, static_cast<const CosType *>(cos.data.dptr),
        static_cast<const CosType *>(sin.data.dptr), key_out, value_out, static_cast<int>(b * h),
        static_cast<int>(h));
  });  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_fused_mla_kv_rope_mxfp8(const NVTETensor kv, const NVTETensor k_rope,
                                  const int k_rope_stride, const NVTETensor cos,
                                  const NVTETensor sin, NVTETensor key, NVTETensor value,
                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_mla_kv_rope_mxfp8);
  using namespace transformer_engine;
  fused_mla_kv_rope_mxfp8(*convertNVTETensorCheck(kv), *convertNVTETensorCheck(k_rope),
                          k_rope_stride, *convertNVTETensorCheck(cos), *convertNVTETensorCheck(sin),
                          convertNVTETensorCheck(key), convertNVTETensorCheck(value), stream);
}
