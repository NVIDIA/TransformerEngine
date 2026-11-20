/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * ---------------------------------------------------------------------------
 * MULTI-TENSOR NVFP4 2D PARTIAL-SHARD KERNELS
 *
 * Multi-tensor versions of the NVFP4 2D partial amax / fused scale / partial cast helpers in
 * nvfp4.cu, for casting the master-weight shards of a distributed optimizer to NVFP4 model
 * weights. They produce the same bytes as those helpers, with these differences:
 *
 * - One launch covers up to kMaxTensorsPerLaunch tensors. Per-tensor metadata is passed as a
 *   kernel parameter, so the calls are CUDA-graph capturable and need no device buffers.
 * - One thread handles one 16x16 tile, with 16-byte loads of FP32 rows (8-byte loads of BF16
 *   rows) and 8-byte stores of the FP4 bytes of tiles that the shard covers fully. The existing
 *   kernels use one 256-thread block per tile and one element per thread.
 * - The input can be the FP32 master shard itself with `cast_dtype` = BF16. Each value is then
 *   rounded to BF16 in registers, which gives the values of master.to(torch.bfloat16) without
 *   materializing that copy. For the block amax, bf16(max |x|) == max |bf16(x)| because
 *   round-to-nearest-even is monotonic.
 * - The per-tensor global amax can be derived from the all-reduced block amaxes (the maximum of
 *   the per-tile maxima is the maximum of the tensor), instead of a separate amax pass over the
 *   shard and a second all-reduce.
 * ---------------------------------------------------------------------------
 */

#include <transformer_engine/recipe.h>

#include <algorithm>
#include <vector>

#include "../cast/nvfp4/core_nvfp4.cuh"
#include "../common.h"
#include "../util/ptx_arch_spec.cuh"
#include "../utils.cuh"

namespace transformer_engine {
namespace nvfp4_recipe {
namespace multi_tensor {

#if FP4_TYPE_SUPPORTED

constexpr int kTileDim = 16;
constexpr int kThreadsPerBlock = 256;
// Kernel parameters stay below 4 KB.
constexpr int kMaxTensorsPerLaunch = 24;

// One tensor's shard as seen by the amax and cast kernels.
struct ShardMeta {
  const void *input;          // shard of the flattened [h, w] tensor, `len` elements
  uint8_t *output;            // packed FP4 bytes; element idx is in output[(idx - start) / 2]
  float *block_amax;          // [tile_rows, tile_cols], contiguous
  const float *decode_scale;  // [tile_rows, tile_cols], contiguous
  const float *global_scale;  // [1]
  int64_t len;
  int64_t start;
  int64_t h;
  int64_t w;
  int64_t tile_cols;
  int64_t first_tile_row;  // first tile row the shard touches
  int32_t vec_load;        // whole-tile rows can be read with vector loads
  int32_t vec_store;       // whole-tile rows can be written with 8-byte stores
};

struct ShardArgs {
  ShardMeta meta[kMaxTensorsPerLaunch];
  int64_t tile_offset[kMaxTensorsPerLaunch + 1];  // prefix sum of the tiles each shard touches
  int num_tensors;
};

struct GlobalAmaxArgs {
  const float *block_amax[kMaxTensorsPerLaunch];
  float *global_amax[kMaxTensorsPerLaunch];
  int64_t num_blocks[kMaxTensorsPerLaunch];
};

struct ScaleMeta {
  const float *block_amax;   // [tile_rows, tile_cols]
  const float *global_amax;  // [1]
  float *per_block_scale;    // [tile_rows, tile_cols]
  uint8_t *target_scale;     // [rows_padded, tile_cols]
  float *target_amax;        // [1]
  int64_t tile_rows;
  int64_t tile_cols;
  int64_t rows_padded;
};

struct ScaleArgs {
  ScaleMeta meta[kMaxTensorsPerLaunch];
  // Prefix sum of (row groups of 16 rows of target_scale) x tile_cols.
  int64_t work_offset[kMaxTensorsPerLaunch + 1];
  int num_tensors;
};

__device__ __forceinline__ int find_tensor(const int64_t *offsets, int n, int64_t t) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// Value as the existing kernels see it: the input converted to `cast_dtype` (input.to(dtype)),
// then to FP32.
template <bool kRoundToBF16>
__device__ __forceinline__ float cast_value(const float x) {
  if constexpr (kRoundToBF16) {
    return __bfloat162float(__float2bfloat16_rn(x));
  } else {
    return x;
  }
}

template <typename IType>
__device__ __forceinline__ void load4(const IType *p, float (&v)[4]);

template <>
__device__ __forceinline__ void load4<float>(const float *p, float (&v)[4]) {
  const float4 t = __ldg(reinterpret_cast<const float4 *>(p));
  v[0] = t.x;
  v[1] = t.y;
  v[2] = t.z;
  v[3] = t.w;
}

template <>
__device__ __forceinline__ void load4<bf16>(const bf16 *p, float (&v)[4]) {
  const uint2 t = __ldg(reinterpret_cast<const uint2 *>(p));
  v[0] = __uint_as_float(t.x << 16);
  v[1] = __uint_as_float(t.x & 0xFFFF0000u);
  v[2] = __uint_as_float(t.y << 16);
  v[3] = __uint_as_float(t.y & 0xFFFF0000u);
}

// Tile (tile_row, tile_col) of shard `m` for thread t of the launch.
__device__ __forceinline__ void tile_of(const ShardMeta &m, const int64_t local, int64_t *r0,
                                        int64_t *c0) {
  *r0 = (m.first_tile_row + local / m.tile_cols) * kTileDim;
  *c0 = (local % m.tile_cols) * kTileDim;
}

__device__ __forceinline__ bool tile_in_shard(const ShardMeta &m, const int64_t r0,
                                              const int64_t c0) {
  return (r0 + kTileDim <= m.h) && (c0 + kTileDim <= m.w) && (r0 * m.w + c0 >= m.start) &&
         ((r0 + kTileDim - 1) * m.w + c0 + kTileDim - 1 < m.start + m.len);
}

template <typename IType, bool kRoundToBF16>
__global__ void __launch_bounds__(kThreadsPerBlock)
    partial_amax_kernel(const ShardArgs args, const int64_t total_tiles) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total_tiles) return;
  const int ti = find_tensor(args.tile_offset, args.num_tensors, t);
  const ShardMeta &m = args.meta[ti];
  int64_t r0, c0;
  tile_of(m, t - args.tile_offset[ti], &r0, &c0);
  const int64_t end = m.start + m.len;
  const IType *in = reinterpret_cast<const IType *>(m.input) - m.start;

  float amax = 0.f;
  if (m.vec_load && tile_in_shard(m, r0, c0)) {
#pragma unroll 4
    for (int r = 0; r < kTileDim; ++r) {
      const IType *row = in + (r0 + r) * m.w + c0;
#pragma unroll
      for (int k = 0; k < kTileDim / 4; ++k) {
        float v[4];
        load4(row + 4 * k, v);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
      }
    }
  } else {
    bool any = false;
    for (int r = 0; r < kTileDim && r0 + r < m.h; ++r) {
      for (int c = 0; c < kTileDim && c0 + c < m.w; ++c) {
        const int64_t idx = (r0 + r) * m.w + c0 + c;
        if (idx >= m.start && idx < end) {
          amax = fmaxf(amax, fabsf(static_cast<float>(in[idx])));
          any = true;
        }
      }
    }
    // Tiles of other ranks are not written; the caller zero-initializes the amax buffer.
    if (!any) return;
  }
  m.block_amax[(r0 / kTileDim) * m.tile_cols + c0 / kTileDim] = cast_value<kRoundToBF16>(amax);
}

// One block per tensor: global amax = maximum of the (all-reduced) block amaxes.
__global__ void __launch_bounds__(kThreadsPerBlock) global_amax_kernel(const GlobalAmaxArgs args) {
  const int ti = blockIdx.x;
  const float *block_amax = args.block_amax[ti];
  const int64_t n = args.num_blocks[ti];
  float a = 0.f;
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) a = fmaxf(a, block_amax[i]);
  for (int d = 16; d > 0; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, d));
  __shared__ float smem[kThreadsPerBlock / 32];
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kThreadsPerBlock / 32; ++i) a = fmaxf(a, smem[i]);
    *args.global_amax[ti] = a;
  }
}

// Same arithmetic as nvfp4_fused_scale_kernel; one thread per 16-row group and tile column.
template <typename ScaleType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    fused_scale_kernel(const ScaleArgs args, const int64_t total) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int ti = find_tensor(args.work_offset, args.num_tensors, t);
  const ScaleMeta &m = args.meta[ti];
  const int64_t local = t - args.work_offset[ti];
  const int64_t tile_row = local / m.tile_cols;
  const int64_t col = local % m.tile_cols;

  const float g_amax = *m.global_amax;
  if (local == 0) *m.target_amax = g_amax;

  constexpr float fp4_max = transformer_engine::TypeInfo<fp4e2m1>::max_finite_value;
  constexpr float scale_max = transformer_engine::TypeInfo<ScaleType>::max_finite_value;
  constexpr float flt_max = 3.402823466e+38f;
  constexpr float tiny = 1.17549435e-38f;

  float scale_val = 0.0f;
  if (tile_row < m.tile_rows) {
    const float safe_global_amax = fmaxf(g_amax, tiny);
    const float global_scale =
        (g_amax > 0.0f) ? fminf((scale_max * fp4_max) / safe_global_amax, flt_max) : 1.0f;
    constexpr float fp4_max_inv = 1.0f / fp4_max;
    const float global_scale_multiplier = global_scale * fp4_max_inv;
    const float amax_val = m.block_amax[tile_row * m.tile_cols + col];
    scale_val = fminf(amax_val * global_scale_multiplier, flt_max);
    m.per_block_scale[tile_row * m.tile_cols + col] = scale_val;
  }
  const ScaleType q = static_cast<ScaleType>(scale_val);
  const uint8_t q_bits = reinterpret_cast<const uint8_t &>(q);
  const int64_t row_end = min((tile_row + 1) * kTileDim, m.rows_padded);
  for (int64_t r = tile_row * kTileDim; r < row_end; ++r) {
    m.target_scale[r * m.tile_cols + col] = q_bits;
  }
}

__device__ __forceinline__ uint16_t cvt_4x(const float (&v)[4], const float s) {
  const auto packed = ptx::mul_cvt_fp32_to_fp4_4x<false>(
      make_float2(v[0], v[1]), make_float2(v[2], v[3]), make_float2(s, s), 0);
  return reinterpret_cast<const uint16_t &>(packed);
}

__device__ __forceinline__ void store_nibble(uint8_t *byte, const int64_t elem_idx,
                                             const uint8_t nibble) {
  *byte = ((elem_idx & 1) == 0) ? static_cast<uint8_t>((*byte & 0xF0u) | nibble)
                                : static_cast<uint8_t>((*byte & 0x0Fu) | (nibble << 4));
}

// Same arithmetic as nvfp4_2d_partial_cast_kernel; one thread per 16x16 tile.
template <typename IType, typename ScaleType, bool kRoundToBF16>
__global__ void __launch_bounds__(kThreadsPerBlock)
    partial_cast_kernel(const ShardArgs args, const int64_t total_tiles) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= total_tiles) return;
  const int ti = find_tensor(args.tile_offset, args.num_tensors, t);
  const ShardMeta &m = args.meta[ti];
  int64_t r0, c0;
  tile_of(m, t - args.tile_offset[ti], &r0, &c0);
  const int64_t end = m.start + m.len;
  const IType *in = reinterpret_cast<const IType *>(m.input) - m.start;

  float global_encode_scale = *m.global_scale;
  if (global_encode_scale <= 0.f) {
    global_encode_scale = 1.f;
  }
  const float global_decode_scale = 1.0f / global_encode_scale;
  float tile_decode_scale = m.decode_scale[(r0 / kTileDim) * m.tile_cols + c0 / kTileDim];
  tile_decode_scale = static_cast<float>(static_cast<ScaleType>(tile_decode_scale));
  constexpr float kFp32Max = 3.402823466e+38F;
  float s = (tile_decode_scale > 0.f) ? 1.0f / (tile_decode_scale * global_decode_scale) : kFp32Max;
  s = fminf(s, kFp32Max);

  if (m.vec_load && tile_in_shard(m, r0, c0)) {
#pragma unroll 2
    for (int r = 0; r < kTileDim; ++r) {
      const int64_t idx0 = (r0 + r) * m.w + c0;
      uint64_t bits = 0;
#pragma unroll
      for (int k = 0; k < kTileDim / 4; ++k) {
        float v[4];
        load4(in + idx0 + 4 * k, v);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = cast_value<kRoundToBF16>(v[j]);
        bits |= static_cast<uint64_t>(cvt_4x(v, s)) << (16 * k);
      }
      uint8_t *out = m.output + ((idx0 - m.start) >> 1);
      if (m.vec_store) {
        *reinterpret_cast<uint2 *>(out) =
            make_uint2(static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
      } else {
#pragma unroll
        for (int b = 0; b < 8; ++b) out[b] = static_cast<uint8_t>(bits >> (8 * b));
      }
    }
    return;
  }

  // Tile cut by the shard or by the tensor edge: the nibble read-modify-write of
  // nvfp4_2d_partial_cast_kernel, four elements at a time.
  for (int r = 0; r < kTileDim && r0 + r < m.h; ++r) {
    for (int g = 0; g < kTileDim / 4; ++g) {
      float v[4];
      bool mask[4];
      int64_t elem_idx[4];
      bool any = false;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int64_t col = c0 + 4 * g + j;
        const int64_t idx = (r0 + r) * m.w + col;
        elem_idx[j] = idx;
        mask[j] = col < m.w && idx >= m.start && idx < end;
        v[j] = mask[j] ? cast_value<kRoundToBF16>(static_cast<float>(in[idx])) : 0.0f;
        any |= mask[j];
      }
      if (!any) continue;
      const uint16_t packed_bits = cvt_4x(v, s);
      for (int pair = 0; pair < 2; ++pair) {
        const int first = 2 * pair, second = first + 1;
        if (!mask[first] && !mask[second]) continue;
        const int64_t ref_idx = mask[first] ? elem_idx[first] : elem_idx[second];
        uint8_t byte = m.output[(ref_idx - m.start) >> 1];
        if (mask[first]) {
          store_nibble(&byte, elem_idx[first],
                       static_cast<uint8_t>((packed_bits >> (4 * first)) & 0xF));
        }
        if (mask[second]) {
          store_nibble(&byte, elem_idx[second],
                       static_cast<uint8_t>((packed_bits >> (4 * second)) & 0xF));
        }
        m.output[(ref_idx - m.start) >> 1] = byte;
      }
    }
  }
}

// Fills one launch's ShardArgs from tensors [begin, end); returns the number of tiles.
int64_t make_shard_args(ShardArgs *args, const Tensor *const *inp, Tensor *const *out,
                        Tensor *const *amax, const Tensor *const *scale,
                        const Tensor *const *global_scale, const size_t *h, const size_t *w,
                        const size_t *start_offset, size_t begin, size_t end, size_t elem_size) {
  int64_t tiles = 0;
  args->num_tensors = static_cast<int>(end - begin);
  for (size_t i = begin; i < end; ++i) {
    ShardMeta &m = args->meta[i - begin];
    const Tensor &input = *inp[i];
    m.input = input.data.dptr;
    m.output = out != nullptr ? reinterpret_cast<uint8_t *>(out[i]->data.dptr) : nullptr;
    m.block_amax = amax != nullptr ? reinterpret_cast<float *>(amax[i]->data.dptr) : nullptr;
    m.decode_scale =
        scale != nullptr ? reinterpret_cast<const float *>(scale[i]->data.dptr) : nullptr;
    m.global_scale = global_scale != nullptr
                         ? reinterpret_cast<const float *>(global_scale[i]->data.dptr)
                         : nullptr;
    m.len = static_cast<int64_t>(input.numel());
    m.start = static_cast<int64_t>(start_offset[i]);
    m.h = static_cast<int64_t>(h[i]);
    m.w = static_cast<int64_t>(w[i]);
    NVTE_CHECK(m.h > 0 && m.w > 0, "NVFP4 2D tensors must not be empty.");
    NVTE_CHECK(m.start % 2 == 0, "NVFP4 2D shards must start at an even element offset.");
    NVTE_CHECK(m.start + m.len <= m.h * m.w, "NVFP4 2D shard exceeds its tensor.");
    m.tile_cols = (m.w + kTileDim - 1) / kTileDim;
    args->tile_offset[i - begin] = tiles;
    if (m.len > 0) {
      const int64_t first = (m.start / m.w) / kTileDim;
      const int64_t last = ((m.start + m.len - 1) / m.w) / kTileDim;
      m.first_tile_row = first;
      tiles += (last - first + 1) * m.tile_cols;
    } else {
      m.first_tile_row = 0;
    }
    const uintptr_t in_ptr = reinterpret_cast<uintptr_t>(m.input);
    m.vec_load = (m.w % 4 == 0) && (m.start % 4 == 0) && (in_ptr % (4 * elem_size) == 0);
    const uintptr_t out_ptr = reinterpret_cast<uintptr_t>(m.output);
    m.vec_store = (m.w % kTileDim == 0) && (m.start % kTileDim == 0) && (out_ptr % 8 == 0);
  }
  args->tile_offset[end - begin] = tiles;
  return tiles;
}

void check_shard_input(const Tensor &inp, DType input_dtype) {
  NVTE_CHECK(inp.dtype() == input_dtype, "All NVFP4 2D shards of one call must have one dtype.");
}

void partial_amax(const Tensor *const *inp, Tensor *const *amax, const size_t *h, const size_t *w,
                  const size_t *start_offset, size_t num_tensors, size_t block_len,
                  DType cast_dtype, cudaStream_t stream) {
  NVTE_CHECK(block_len == kTileDim, "NVFP4 2D supports 16x16 tiles only (block_len = 16).");
  if (num_tensors == 0) return;
  const DType input_dtype = inp[0]->dtype();
  NVTE_CHECK(input_dtype == DType::kFloat32 || input_dtype == DType::kBFloat16,
             "NVFP4 2D shards must be FP32 or BF16.");
  NVTE_CHECK(cast_dtype == DType::kFloat32 || cast_dtype == DType::kBFloat16,
             "cast_dtype must be FP32 or BF16.");
  for (size_t i = 0; i < num_tensors; ++i) {
    check_shard_input(*inp[i], input_dtype);
    NVTE_CHECK(amax[i]->dtype() == DType::kFloat32, "Block amax must be float32.");
  }
  const bool round_to_bf16 = cast_dtype == DType::kBFloat16 && input_dtype == DType::kFloat32;
  for (size_t begin = 0; begin < num_tensors; begin += kMaxTensorsPerLaunch) {
    const size_t end = std::min(num_tensors, begin + kMaxTensorsPerLaunch);
    ShardArgs args;
    const int64_t tiles = make_shard_args(&args, inp, nullptr, amax, nullptr, nullptr, h, w,
                                          start_offset, begin, end, typeToSize(input_dtype));
    if (tiles == 0) continue;
    const dim3 grid(static_cast<unsigned>((tiles + kThreadsPerBlock - 1) / kThreadsPerBlock));
    TRANSFORMER_ENGINE_SWITCH_CONDITION(
        input_dtype == DType::kFloat32, kFP32Input,
        TRANSFORMER_ENGINE_SWITCH_CONDITION(
            round_to_bf16, kRoundToBF16, using IType = std::conditional_t<kFP32Input, float, bf16>;
            partial_amax_kernel<IType, kRoundToBF16>
            <<<grid, kThreadsPerBlock, 0, stream>>>(args, tiles);););
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void global_amax_from_block_amax(const Tensor *const *block_amax, Tensor *const *global_amax,
                                 size_t num_tensors, cudaStream_t stream) {
  for (size_t begin = 0; begin < num_tensors; begin += kMaxTensorsPerLaunch) {
    const size_t end = std::min(num_tensors, begin + kMaxTensorsPerLaunch);
    GlobalAmaxArgs args;
    for (size_t i = begin; i < end; ++i) {
      NVTE_CHECK(block_amax[i]->dtype() == DType::kFloat32, "Block amax must be float32.");
      NVTE_CHECK(global_amax[i]->dtype() == DType::kFloat32, "Global amax must be float32.");
      NVTE_CHECK(global_amax[i]->numel() == 1, "Global amax must be a single element tensor.");
      args.block_amax[i - begin] = reinterpret_cast<const float *>(block_amax[i]->data.dptr);
      args.global_amax[i - begin] = reinterpret_cast<float *>(global_amax[i]->data.dptr);
      args.num_blocks[i - begin] = static_cast<int64_t>(block_amax[i]->numel());
    }
    global_amax_kernel<<<static_cast<unsigned>(end - begin), kThreadsPerBlock, 0, stream>>>(args);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void fused_scale(const Tensor *const *block_amax, const Tensor *const *global_amax,
                 Tensor *const *per_block_scale, Tensor *const *target_scale,
                 Tensor *const *target_amax, const size_t *tile_rows, const size_t *tile_cols,
                 const size_t *rows_padded, size_t num_tensors, size_t block_len, DType scale_dtype,
                 cudaStream_t stream) {
  NVTE_CHECK(block_len == kTileDim, "NVFP4 2D supports 16x16 tiles only (block_len = 16).");
  for (size_t begin = 0; begin < num_tensors; begin += kMaxTensorsPerLaunch) {
    const size_t end = std::min(num_tensors, begin + kMaxTensorsPerLaunch);
    ScaleArgs args;
    args.num_tensors = static_cast<int>(end - begin);
    int64_t work = 0;
    for (size_t i = begin; i < end; ++i) {
      NVTE_CHECK(block_amax[i]->dtype() == DType::kFloat32, "Block amax must be float32.");
      NVTE_CHECK(global_amax[i]->dtype() == DType::kFloat32, "Global amax must be float32.");
      NVTE_CHECK(per_block_scale[i]->dtype() == DType::kFloat32,
                 "Per-block scale must be float32.");
      NVTE_CHECK(
          target_scale[i]->dtype() == DType::kByte || target_scale[i]->dtype() == scale_dtype,
          "Target scale must be byte storage or have the selected NVFP4 scale dtype.");
      NVTE_CHECK(target_amax[i]->dtype() == DType::kFloat32, "Target amax must be float32.");
      NVTE_CHECK(global_amax[i]->numel() == 1 && target_amax[i]->numel() == 1,
                 "Global and target amax must be single element tensors.");
      ScaleMeta &m = args.meta[i - begin];
      m.block_amax = reinterpret_cast<const float *>(block_amax[i]->data.dptr);
      m.global_amax = reinterpret_cast<const float *>(global_amax[i]->data.dptr);
      m.per_block_scale = reinterpret_cast<float *>(per_block_scale[i]->data.dptr);
      m.target_scale = reinterpret_cast<uint8_t *>(target_scale[i]->data.dptr);
      m.target_amax = reinterpret_cast<float *>(target_amax[i]->data.dptr);
      m.tile_rows = static_cast<int64_t>(tile_rows[i]);
      m.tile_cols = static_cast<int64_t>(tile_cols[i]);
      m.rows_padded = static_cast<int64_t>(rows_padded[i]);
      args.work_offset[i - begin] = work;
      work += ((m.rows_padded + kTileDim - 1) / kTileDim) * m.tile_cols;
    }
    args.work_offset[end - begin] = work;
    if (work == 0) continue;
    const dim3 grid(static_cast<unsigned>((work + kThreadsPerBlock - 1) / kThreadsPerBlock));
    TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(
        scale_dtype, ScaleType,
        fused_scale_kernel<ScaleType><<<grid, kThreadsPerBlock, 0, stream>>>(args, work););
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void partial_cast(const Tensor *const *inp, Tensor *const *out, const Tensor *const *scale,
                  const Tensor *const *global_scale, const size_t *h, const size_t *w,
                  const size_t *start_offset, size_t num_tensors, size_t block_len,
                  DType cast_dtype, DType scale_dtype, cudaStream_t stream) {
  NVTE_CHECK(block_len == kTileDim, "NVFP4 2D supports 16x16 tiles only (block_len = 16).");
  if (num_tensors == 0) return;
  const DType input_dtype = inp[0]->dtype();
  NVTE_CHECK(input_dtype == DType::kFloat32 || input_dtype == DType::kBFloat16,
             "NVFP4 2D shards must be FP32 or BF16.");
  NVTE_CHECK(cast_dtype == DType::kFloat32 || cast_dtype == DType::kBFloat16,
             "cast_dtype must be FP32 or BF16.");
  for (size_t i = 0; i < num_tensors; ++i) {
    check_shard_input(*inp[i], input_dtype);
    NVTE_CHECK(out[i]->dtype() == DType::kByte, "NVFP4 rowwise data must be uint8.");
    NVTE_CHECK(scale[i]->dtype() == DType::kFloat32, "Scale must be float32.");
    NVTE_CHECK(global_scale[i]->dtype() == DType::kFloat32 && global_scale[i]->numel() == 1,
               "Global scale must be a single float32 element.");
  }
  const bool round_to_bf16 = cast_dtype == DType::kBFloat16 && input_dtype == DType::kFloat32;
  for (size_t begin = 0; begin < num_tensors; begin += kMaxTensorsPerLaunch) {
    const size_t end = std::min(num_tensors, begin + kMaxTensorsPerLaunch);
    ShardArgs args;
    const int64_t tiles = make_shard_args(&args, inp, out, nullptr, scale, global_scale, h, w,
                                          start_offset, begin, end, typeToSize(input_dtype));
    if (tiles == 0) continue;
    const dim3 grid(static_cast<unsigned>((tiles + kThreadsPerBlock - 1) / kThreadsPerBlock));
    TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(
        scale_dtype, ScaleType,
        TRANSFORMER_ENGINE_SWITCH_CONDITION(
            input_dtype == DType::kFloat32, kFP32Input,
            TRANSFORMER_ENGINE_SWITCH_CONDITION(
                round_to_bf16, kRoundToBF16,
                using IType = std::conditional_t<kFP32Input, float, bf16>;
                partial_cast_kernel<IType, ScaleType, kRoundToBF16>
                <<<grid, kThreadsPerBlock, 0, stream>>>(args, tiles);)););
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace multi_tensor
}  // namespace nvfp4_recipe
}  // namespace transformer_engine

namespace {

std::vector<transformer_engine::Tensor *> convert_list(const NVTETensor *tensors, size_t n) {
  std::vector<transformer_engine::Tensor *> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = transformer_engine::convertNVTETensorCheck(tensors[i]);
  return out;
}

}  // namespace

void nvte_nvfp4_2d_multi_tensor_partial_amax(const NVTETensor *inp, NVTETensor *amax,
                                             const size_t *h, const size_t *w,
                                             const size_t *start_offset, size_t num_tensors,
                                             size_t block_len, NVTEDType cast_dtype,
                                             cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_2d_multi_tensor_partial_amax);
  using namespace transformer_engine;
  const auto inp_t = convert_list(inp, num_tensors);
  const auto amax_t = convert_list(amax, num_tensors);
  nvfp4_recipe::multi_tensor::partial_amax(inp_t.data(), amax_t.data(), h, w, start_offset,
                                           num_tensors, block_len, static_cast<DType>(cast_dtype),
                                           stream);
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

void nvte_nvfp4_2d_multi_tensor_global_amax(const NVTETensor *block_amax, NVTETensor *global_amax,
                                            size_t num_tensors, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_2d_multi_tensor_global_amax);
  using namespace transformer_engine;
  const auto block_amax_t = convert_list(block_amax, num_tensors);
  const auto global_amax_t = convert_list(global_amax, num_tensors);
  nvfp4_recipe::multi_tensor::global_amax_from_block_amax(block_amax_t.data(), global_amax_t.data(),
                                                          num_tensors, stream);
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

void nvte_nvfp4_2d_multi_tensor_fused_scale(const NVTETensor *block_amax,
                                            const NVTETensor *global_amax,
                                            NVTETensor *per_block_scale, NVTETensor *target_scale,
                                            NVTETensor *target_amax, const size_t *tile_rows,
                                            const size_t *tile_cols, const size_t *rows_padded,
                                            size_t num_tensors, size_t block_len,
                                            NVTEDType scale_dtype, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_2d_multi_tensor_fused_scale);
  using namespace transformer_engine;
  const auto block_amax_t = convert_list(block_amax, num_tensors);
  const auto global_amax_t = convert_list(global_amax, num_tensors);
  const auto per_block_scale_t = convert_list(per_block_scale, num_tensors);
  const auto target_scale_t = convert_list(target_scale, num_tensors);
  const auto target_amax_t = convert_list(target_amax, num_tensors);
  nvfp4_recipe::multi_tensor::fused_scale(
      block_amax_t.data(), global_amax_t.data(), per_block_scale_t.data(), target_scale_t.data(),
      target_amax_t.data(), tile_rows, tile_cols, rows_padded, num_tensors, block_len,
      static_cast<DType>(scale_dtype), stream);
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

void nvte_nvfp4_2d_multi_tensor_partial_cast(
    const NVTETensor *inp, NVTETensor *out, const NVTETensor *scale, const NVTETensor *global_scale,
    const size_t *h, const size_t *w, const size_t *start_offset, size_t num_tensors,
    size_t block_len, NVTEDType cast_dtype, NVTEDType scale_dtype, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_2d_multi_tensor_partial_cast);
  using namespace transformer_engine;
  const auto inp_t = convert_list(inp, num_tensors);
  const auto out_t = convert_list(out, num_tensors);
  const auto scale_t = convert_list(scale, num_tensors);
  const auto global_scale_t = convert_list(global_scale, num_tensors);
  nvfp4_recipe::multi_tensor::partial_cast(inp_t.data(), out_t.data(), scale_t.data(),
                                           global_scale_t.data(), h, w, start_offset, num_tensors,
                                           block_len, static_cast<DType>(cast_dtype),
                                           static_cast<DType>(scale_dtype), stream);
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}
