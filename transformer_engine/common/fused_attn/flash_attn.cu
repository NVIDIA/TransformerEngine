/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>

#include "../common.h"
#include "../util/cuda_driver.h"
#include "../util/cuda_runtime.h"
#include "../util/ptx.cuh"
#include "../utils.cuh"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {

// ============================================================================
// prepare_flash_attn: repack interleaved QKV for FlashAttention backend
// ============================================================================

namespace flash_attention {

/// Packed vector of N elements of T; alignment matches a single wide load/store of N * sizeof(T) bytes.
template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
};

constexpr int warp_size = 32;
constexpr int type_size = 2;
constexpr int nvec = sizeof(uint64_t) / type_size;
constexpr int nvec128 = sizeof(uint4) / type_size;
constexpr int load_size = warp_size * nvec;
constexpr int block_size = 512;

template <typename T>
__launch_bounds__(block_size) __global__
    void prepare_kernel_fwd(const T *qkvi, T *qkv, const size_t B, const size_t S, const size_t Z,
                            const size_t W) {
  const int warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  const int id_in_warp = threadIdx.x % warp_size;
  const size_t offset_input = blockIdx.y * W + warpid * 3 * W * Z + id_in_warp * nvec;
  const T *my_input = qkvi + offset_input;

  const size_t s = warpid / B;
  if (s >= S) return;

  const size_t b = warpid % B;

  const size_t offset_output = blockIdx.y * B * S * Z * W + (s + b * S) * W * Z + id_in_warp * nvec;

  T *my_output = qkv + offset_output;

  for (int i = 0; i < Z; ++i) {
    Vec<T, nvec> *const out = reinterpret_cast<Vec<T, nvec> *>(my_output + i * load_size);
    *out = *reinterpret_cast<const Vec<T, nvec> *>(my_input + i * load_size * 3);
  }
}

template <typename T>
__launch_bounds__(block_size) __global__
    void prepare_kernel_bwd(const T *q, const T *k, const T *v, T *qkv, const size_t B,
                            const size_t S, const size_t Z, const size_t W) {
  const T *input = blockIdx.y == 0 ? q : (blockIdx.y == 1 ? k : v);

  const int warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  const int id_in_warp = threadIdx.x % warp_size;
  const size_t offset_input = warpid * W * Z + id_in_warp * nvec;
  const T *my_input = input + offset_input;

  const size_t b = warpid / S;
  if (b >= B) return;

  const size_t s = warpid % S;

  const size_t offset_output = (b + s * B) * 3 * W * Z + id_in_warp * nvec + blockIdx.y * W;

  T *my_output = qkv + offset_output;

  for (int i = 0; i < Z; ++i) {
    Vec<T, nvec> *const out = reinterpret_cast<Vec<T, nvec> *>(my_output + i * load_size * 3);
    *out = *reinterpret_cast<const Vec<T, nvec> *>(my_input + i * load_size);
  }
}

void prepare_flash_attn_fwd(Tensor qkvi, Tensor qkv, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(qkvi.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(qkvi.dtype() == DType::kFloat16 || qkvi.dtype() == DType::kBFloat16);

  auto qkvi_shape = qkvi.shape();

  NVTE_CHECK(qkvi_shape[3] % load_size == 0);
  NVTE_CHECK(qkvi_shape[3] == load_size);

  // [s, b, n, h * 3] -> [3, b, s, n, h]
  std::vector<uint64_t> shape = {3, qkvi_shape[1], qkvi_shape[0], qkvi_shape[2], qkvi_shape[3]};

  size_t warps = qkvi_shape[0] * qkvi_shape[1];
  size_t warps_per_block = block_size / warp_size;
  size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(blocks, 3);
  int threads = block_size;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      qkvi.dtype(), dtype,
      prepare_kernel_fwd<dtype><<<grid, threads, 0, stream>>>(
          reinterpret_cast<dtype *>(qkvi.data.dptr), reinterpret_cast<dtype *>(qkv.data.dptr),
          shape[1], shape[2], shape[3], shape[4]););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void prepare_flash_attn_bwd(Tensor q, Tensor k, Tensor v, Tensor qkv, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(q.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(k.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(v.dim() == 4, "Expected 4-dim tensor.");
  NVTE_CHECK(q.dtype() == DType::kFloat16 || q.dtype() == DType::kBFloat16);
  NVTE_CHECK(k.dtype() == q.dtype());
  NVTE_CHECK(v.dtype() == q.dtype());

  auto q_shape = q.shape();
  auto k_shape = k.shape();
  auto v_shape = v.shape();

  NVTE_CHECK(q_shape[3] % load_size == 0);
  NVTE_CHECK(q_shape[3] == load_size);
  NVTE_CHECK(k_shape[3] % load_size == 0);
  NVTE_CHECK(k_shape[3] == load_size);
  NVTE_CHECK(v_shape[3] % load_size == 0);
  NVTE_CHECK(v_shape[3] == load_size);

  // 3 x [s, b, n, h] -> [b, s, n, 3 * h]
  std::vector<uint64_t> shape = {q_shape[1], q_shape[0], q_shape[2], 3 * q_shape[3]};

  size_t warps = q_shape[0] * q_shape[1];
  size_t warps_per_block = block_size / warp_size;
  size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(blocks, 3);
  int threads = block_size;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      q.dtype(), dtype,
      prepare_kernel_bwd<dtype><<<grid, threads, 0, stream>>>(
          reinterpret_cast<dtype *>(q.data.dptr), reinterpret_cast<dtype *>(k.data.dptr),
          reinterpret_cast<dtype *>(v.data.dptr), reinterpret_cast<dtype *>(qkv.data.dptr),
          q_shape[0], q_shape[1], q_shape[2], q_shape[3]););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace flash_attention

// ============================================================================
// permute_to_grouped_tensor: BSHD/SBHD ↔ BHSD permutation
// ============================================================================

namespace permute_to_grouped_tensor {

using flash_attention::Vec;

// ---------- fallback_not_vec_aligned: row-copy helper (D is small / misaligned) ----------

__device__ __forceinline__ void copy_row_bytes(const char *__restrict__ src,
                                               char *__restrict__ dst, size_t D_bytes) {
  size_t off = 0;
  for (; off + 16 <= D_bytes; off += 16) {
    uint4 tmp;
    memcpy(&tmp, src + off, 16);
    memcpy(dst + off, &tmp, 16);
  }
  for (; off + 8 <= D_bytes; off += 8) {
    uint2 tmp;
    memcpy(&tmp, src + off, 8);
    memcpy(dst + off, &tmp, 8);
  }
  for (; off + 4 <= D_bytes; off += 4) {
    unsigned int tmp;
    memcpy(&tmp, src + off, 4);
    memcpy(dst + off, &tmp, 4);
  }
  for (; off + 2 <= D_bytes; off += 2) {
    unsigned short tmp;
    memcpy(&tmp, src + off, 2);
    memcpy(dst + off, &tmp, 2);
  }
  for (; off < D_bytes; ++off) dst[off] = src[off];
}

__device__ __forceinline__ void copy_and_pad_row_bytes(const char *__restrict__ src,
                                                       char *__restrict__ dst,
                                                       size_t D_bytes, size_t D_out_bytes) {
  copy_row_bytes(src, dst, D_bytes);
  for (size_t off = D_bytes; off < D_out_bytes; ++off) dst[off] = 0;
}


// ---------- fallback_not_vec_aligned: tiled-transpose kernels ----------

constexpr int TRANSPOSE_TILE  = 32;
constexpr int TRANSPOSE_BLOCK = 256;
constexpr int TRANSPOSE_WARPS = TRANSPOSE_BLOCK / 32;   // 8

template <typename T, bool kIsSbhd>
__launch_bounds__(TRANSPOSE_BLOCK) __global__
    void permute_to_grouped_tensor_fwd_fallback_not_vec_aligned_kernel(
        const T *__restrict__ q_in, const T *__restrict__ k_in, const T *__restrict__ v_in,
        T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out,
        size_t b, size_t s_q, size_t h_q, size_t d_qk, size_t d_qk_out,
        size_t s_kv, size_t h_kv, size_t d_v, size_t d_v_out,
        unsigned int s_tiles) {
  const int which = blockIdx.z;
  const T *__restrict__ in = which == 0 ? q_in : (which == 1 ? k_in : v_in);
  T *__restrict__ out           = which == 0 ? q_out : (which == 1 ? k_out : v_out);
  const size_t S     = which == 0 ? s_q  : s_kv;
  const size_t H     = which == 0 ? h_q  : h_kv;
  const size_t D     = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);
  const size_t D_out = which == 0 ? d_qk_out : (which == 1 ? d_qk_out : d_v_out);
  const size_t D_bytes = D * sizeof(T);
  const size_t D_out_bytes = D_out * sizeof(T);
  const size_t D_smem_pad = (D_bytes + 3u) & ~size_t(3);

  const size_t tile_s = static_cast<size_t>(blockIdx.x) % static_cast<size_t>(s_tiles);
  const size_t b_i    = static_cast<size_t>(blockIdx.x) / static_cast<size_t>(s_tiles);
  if (b_i >= b) return;
  const size_t tile_h = static_cast<size_t>(blockIdx.y);

  const size_t s_base = tile_s * TRANSPOSE_TILE;
  const size_t h_base = tile_h * TRANSPOSE_TILE;

  extern __shared__ char smem[];
  const size_t smem_row = static_cast<size_t>(TRANSPOSE_TILE) * D_smem_pad + 4;

  // ---- Phase 1: global → smem (sweep consecutive H → coalesced reads) ----
  for (unsigned int warp_off = threadIdx.x >> 5;
       warp_off < TRANSPOSE_TILE; warp_off += TRANSPOSE_WARPS) {
    const size_t local_s = warp_off;
    const size_t local_h = threadIdx.x & 31u;
    const size_t s_i = s_base + local_s;
    const size_t h_i = h_base + local_h;
    if (s_i < S && h_i < H) {
      const char *__restrict__ src;
      if constexpr (kIsSbhd)
        src = reinterpret_cast<const char *>(in + s_i * b * H * D + b_i * H * D + h_i * D);
      else
        src = reinterpret_cast<const char *>(in + b_i * S * H * D + s_i * H * D + h_i * D);
      copy_row_bytes(src, smem + local_s * smem_row + local_h * D_smem_pad, D_bytes);
    }
  }

  __syncthreads();

  // ---- Phase 2: smem → global (sweep consecutive S → coalesced writes, with padding) ----
  for (unsigned int warp_off = threadIdx.x >> 5;
       warp_off < TRANSPOSE_TILE; warp_off += TRANSPOSE_WARPS) {
    const size_t local_h = warp_off;
    const size_t local_s = threadIdx.x & 31u;
    const size_t s_i = s_base + local_s;
    const size_t h_i = h_base + local_h;
    if (s_i < S && h_i < H) {
      copy_and_pad_row_bytes(
          smem + local_s * smem_row + local_h * D_smem_pad,
          reinterpret_cast<char *>(out + b_i * H * S * D_out + h_i * S * D_out + s_i * D_out),
          D_bytes, D_out_bytes);
    }
  }
}

template <typename T, bool kIsSbhd>
__launch_bounds__(TRANSPOSE_BLOCK) __global__
    void permute_to_grouped_tensor_bwd_fallback_not_vec_aligned_kernel(
        const T *__restrict__ grad_q, const T *__restrict__ grad_k, const T *__restrict__ grad_v,
        T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out,
        size_t b, size_t s_q, size_t h_q, size_t d_qk, size_t d_qk_in,
        size_t s_kv, size_t h_kv, size_t d_v, size_t d_v_in,
        unsigned int s_tiles) {
  const int which = blockIdx.z;
  const T *__restrict__ in = which == 0 ? grad_q : (which == 1 ? grad_k : grad_v);
  T *__restrict__ out           = which == 0 ? q_out : (which == 1 ? k_out : v_out);
  const size_t S     = which == 0 ? s_q  : s_kv;
  const size_t H     = which == 0 ? h_q  : h_kv;
  const size_t D_write = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);
  const size_t D_read  = which == 0 ? d_qk_in : (which == 1 ? d_qk_in : d_v_in);
  const size_t D_data_bytes = D_write * sizeof(T);
  const size_t D_smem_pad = (D_data_bytes + 3u) & ~size_t(3);

  const size_t tile_s = static_cast<size_t>(blockIdx.x) % static_cast<size_t>(s_tiles);
  const size_t b_i    = static_cast<size_t>(blockIdx.x) / static_cast<size_t>(s_tiles);
  if (b_i >= b) return;
  const size_t tile_h = static_cast<size_t>(blockIdx.y);

  const size_t s_base = tile_s * TRANSPOSE_TILE;
  const size_t h_base = tile_h * TRANSPOSE_TILE;

  extern __shared__ char smem[];
  const size_t smem_row = static_cast<size_t>(TRANSPOSE_TILE) * D_smem_pad + 4;

  // ---- Phase 1: global → smem (BHSD input with D_read stride, copy D_write bytes) ----
  for (unsigned int warp_off = threadIdx.x >> 5;
       warp_off < TRANSPOSE_TILE; warp_off += TRANSPOSE_WARPS) {
    const size_t local_h = warp_off;
    const size_t local_s = threadIdx.x & 31u;
    const size_t s_i = s_base + local_s;
    const size_t h_i = h_base + local_h;
    if (s_i < S && h_i < H) {
      copy_row_bytes(
          reinterpret_cast<const char *>(in + b_i * H * S * D_read + h_i * S * D_read + s_i * D_read),
          smem + local_s * smem_row + local_h * D_smem_pad, D_data_bytes);
    }
  }

  __syncthreads();

  // ---- Phase 2: smem → global (SBHD/BSHD output with D_write stride) ----
  for (unsigned int warp_off = threadIdx.x >> 5;
       warp_off < TRANSPOSE_TILE; warp_off += TRANSPOSE_WARPS) {
    const size_t local_s = warp_off;
    const size_t local_h = threadIdx.x & 31u;
    const size_t s_i = s_base + local_s;
    const size_t h_i = h_base + local_h;
    if (s_i < S && h_i < H) {
      char *__restrict__ dst;
      if constexpr (kIsSbhd)
        dst = reinterpret_cast<char *>(out + s_i * b * H * D_write + b_i * H * D_write + h_i * D_write);
      else
        dst = reinterpret_cast<char *>(out + b_i * S * H * D_write + s_i * H * D_write + h_i * D_write);
      copy_row_bytes(smem + local_s * smem_row + local_h * D_smem_pad, dst, D_data_bytes);
    }
  }
}

// ---------- fallback_vec_aligned: ----------

constexpr int fallback_permute_threads = 1024;

template <typename T, bool kIsSbhd, int N>
__device__ __forceinline__ void permute_fwd_vec_loop(
    const T *__restrict__ in, T *__restrict__ out, size_t b, size_t S, size_t H,
    size_t D, size_t D_out,
    size_t b_i, size_t h_i, size_t s_begin, size_t S_chunk) {
  const size_t out_base = b_i * H * S * D_out + h_i * S * D_out;
  const size_t d_vec = D / static_cast<size_t>(N);
  const size_t total_work = S_chunk * d_vec;
  for (size_t w = static_cast<size_t>(threadIdx.x); w < total_work;
       w += static_cast<size_t>(blockDim.x)) {
    const size_t s_local = w / d_vec;
    const size_t s_i = s_begin + s_local;
    const size_t d_off = (w % d_vec) * static_cast<size_t>(N);
    const T *__restrict__ in_ptr;
    if constexpr (kIsSbhd) {
      in_ptr = in + s_i * (b * H * D) + b_i * (H * D) + h_i * D + d_off;
    } else {
      in_ptr = in + b_i * (S * H * D) + s_i * (H * D) + h_i * D + d_off;
    }
    T *__restrict__ out_ptr = out + out_base + s_i * D_out + d_off;
    *reinterpret_cast<Vec<T, N> *>(out_ptr) = *reinterpret_cast<const Vec<T, N> *>(in_ptr);
  }
  if (D_out > D) {
    const size_t pad_elems = D_out - D;
    const size_t total_pad = S_chunk * pad_elems;
    for (size_t w = static_cast<size_t>(threadIdx.x); w < total_pad;
         w += static_cast<size_t>(blockDim.x)) {
      const size_t s_local = w / pad_elems;
      const size_t s_i = s_begin + s_local;
      const size_t d_off = D + (w % pad_elems);
      out[out_base + s_i * D_out + d_off] = static_cast<T>(0);
    }
  }
}

template <typename T, bool kIsSbhd, int N>
__device__ __forceinline__ void permute_bwd_vec_loop(
    const T *__restrict__ in, T *__restrict__ out, size_t b, size_t S, size_t H,
    size_t D_read, size_t D_write,
    size_t b_i, size_t h_i, size_t s_begin, size_t S_chunk) {
  const size_t in_base = b_i * H * S * D_read + h_i * S * D_read;
  const size_t D_copy = (D_write < D_read) ? D_write : D_read;
  const size_t d_vec = D_copy / static_cast<size_t>(N);
  const size_t total_work = S_chunk * d_vec;
  for (size_t w = static_cast<size_t>(threadIdx.x); w < total_work;
       w += static_cast<size_t>(blockDim.x)) {
    const size_t s_local = w / d_vec;
    const size_t s_i = s_begin + s_local;
    const size_t d_off = (w % d_vec) * static_cast<size_t>(N);
    const T *__restrict__ in_ptr = in + in_base + s_i * D_read + d_off;
    T *__restrict__ out_ptr;
    if constexpr (kIsSbhd) {
      out_ptr = out + s_i * (b * H * D_write) + b_i * (H * D_write) + h_i * D_write + d_off;
    } else {
      out_ptr = out + b_i * (S * H * D_write) + s_i * (H * D_write) + h_i * D_write + d_off;
    }
    *reinterpret_cast<Vec<T, N> *>(out_ptr) = *reinterpret_cast<const Vec<T, N> *>(in_ptr);
  }
}

template <typename T, bool kIsSbhd>
__launch_bounds__(fallback_permute_threads) __global__
    void permute_to_grouped_tensor_fwd_fallback_vec_aligned_kernel(
        const T *__restrict__ q_in, const T *__restrict__ k_in, const T *__restrict__ v_in,
        T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out, size_t b, size_t s_q,
        size_t h_q, size_t d_qk, size_t d_qk_out, size_t s_kv, size_t h_kv, size_t d_v,
        size_t d_v_out, unsigned int permute_s_splits) {
  const int which = blockIdx.z;
  const T *__restrict__ in = which == 0 ? q_in : (which == 1 ? k_in : v_in);
  T *__restrict__ out = which == 0 ? q_out : (which == 1 ? k_out : v_out);
  const size_t S = which == 0 ? s_q : s_kv;
  const size_t H = which == 0 ? h_q : h_kv;
  const size_t D = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);
  const size_t D_out = which == 0 ? d_qk_out : (which == 1 ? d_qk_out : d_v_out);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;
  if (b_i >= b) return;
  if (which == 0) {
    if (h_i >= h_q) return;
  } else {
    if (h_i >= h_kv) return;
  }

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (S * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (S * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  if (s_begin >= s_end) return;
  const size_t S_chunk = s_end - s_begin;

  const size_t D_bytes = D * sizeof(T);

  if (D_bytes % 16 == 0) {
    constexpr size_t N = 16 / sizeof(T);
    permute_fwd_vec_loop<T, kIsSbhd, N>(in, out, b, S, H, D, D_out, b_i, h_i, s_begin, S_chunk);
    return;
  }
  if (D_bytes % 8 == 0) {
    constexpr size_t N = 8 / sizeof(T);
    permute_fwd_vec_loop<T, kIsSbhd, N>(in, out, b, S, H, D, D_out, b_i, h_i, s_begin, S_chunk);
    return;
  }
  if constexpr (sizeof(T) <= 4) {
    if (D_bytes % 4 == 0) {
      constexpr size_t N = 4 / sizeof(T);
      permute_fwd_vec_loop<T, kIsSbhd, N>(in, out, b, S, H, D, D_out, b_i, h_i, s_begin, S_chunk);
      return;
    }
  }
}

template <typename T, bool kIsSbhd>
__launch_bounds__(fallback_permute_threads) __global__
    void permute_to_grouped_tensor_bwd_fallback_vec_aligned_kernel(
        const T *__restrict__ grad_q, const T *__restrict__ grad_k, const T *__restrict__ grad_v,
        T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out, size_t b, size_t s_q,
        size_t h_q, size_t d_qk, size_t d_qk_in, size_t s_kv, size_t h_kv, size_t d_v,
        size_t d_v_in, unsigned int permute_s_splits) {
  const int which = blockIdx.z;
  const T *__restrict__ in = which == 0 ? grad_q : (which == 1 ? grad_k : grad_v);
  T *__restrict__ out = which == 0 ? q_out : (which == 1 ? k_out : v_out);
  const size_t S = which == 0 ? s_q : s_kv;
  const size_t H = which == 0 ? h_q : h_kv;
  const size_t D_write = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);
  const size_t D_read  = which == 0 ? d_qk_in : (which == 1 ? d_qk_in : d_v_in);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;
  if (b_i >= b) return;
  if (which == 0) {
    if (h_i >= h_q) return;
  } else {
    if (h_i >= h_kv) return;
  }

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (S * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (S * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  if (s_begin >= s_end) return;
  const size_t S_chunk = s_end - s_begin;

  const size_t D_bytes = D_write * sizeof(T);

  if (D_bytes % 16 == 0) {
    constexpr size_t N = 16 / sizeof(T);
    permute_bwd_vec_loop<T, kIsSbhd, N>(in, out, b, S, H, D_read, D_write, b_i, h_i, s_begin, S_chunk);
    return;
  }
  if (D_bytes % 8 == 0) {
    constexpr size_t N = 8 / sizeof(T);
    permute_bwd_vec_loop<T, kIsSbhd, N>(in, out, b, S, H, D_read, D_write, b_i, h_i, s_begin, S_chunk);
    return;
  }
  if constexpr (sizeof(T) <= 4) {
    if (D_bytes % 4 == 0) {
      constexpr size_t N = 4 / sizeof(T);
      permute_bwd_vec_loop<T, kIsSbhd, N>(in, out, b, S, H, D_read, D_write, b_i, h_i, s_begin, S_chunk);
      return;
    }
  }
}


// ---------- main path: TMA ----------

constexpr int tma_permute_threads = 128;
constexpr int tma_permute_s_tile_default = 32;

// ---- 4D TMA PTX wrappers ----

__device__ __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
    void *dst_shmem, const CUtensorMap *tensor_map, uint32_t c0, uint32_t c1, uint32_t c2,
    uint32_t c3, uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t dst = __cvta_generic_to_shared(dst_shmem);
  uint32_t bar = __cvta_generic_to_shared(mbar);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
      "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(bar)
      : "memory");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_4d_global_to_shared requires SM 10.0+.");
#endif
}

__device__ __forceinline__ void cp_async_bulk_tensor_4d_shared_to_global(
    const CUtensorMap *tensor_map, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    void *src_shmem) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t src = __cvta_generic_to_shared(src_shmem);
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
      " [%0, {%1, %2, %3, %4}], [%5];" ::"l"(tensor_map),
      "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(src)
      : "memory");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_4d_shared_to_global requires SM 9.0+.");
#endif
}

// ---- Host-side 4D tensor map creation ----

static void create_4D_tensor_map(CUtensorMap &tensorMap, void *dataPtr, DType dtype, uint64_t dim0,
                                 uint64_t dim1, uint64_t dim2, uint64_t dim3, uint32_t box0,
                                 uint32_t box1, uint32_t box2, uint32_t box3) {
  cuda_driver::ensure_context_exists();
  static PFN_cuTensorMapEncodeTiled_v12000 cuDriverTensorMapEncodeTiled = []() {
    void *ptr = cuda_driver::get_symbol("cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
  }();

  CUtensorMapDataType tma_dtype;
  size_t elem_bytes;
  switch (dtype) {
    case DType::kFloat16:
      tma_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      elem_bytes = 2;
      break;
    case DType::kBFloat16:
      tma_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
      elem_bytes = 2;
      break;
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
    case DType::kFloat8E8M0:
    case DType::kByte:
      tma_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      elem_bytes = 1;
      break;
    default:
      NVTE_ERROR("create_4D_tensor_map: unsupported dtype ",
                 to_string(static_cast<DType>(dtype)));
  }

  constexpr uint32_t rank = 4;
  uint64_t size[rank] = {dim0, dim1, dim2, dim3};
  uint64_t stride[rank - 1] = {
      dim0 * elem_bytes,
      dim0 * dim1 * elem_bytes,
      dim0 * dim1 * dim2 * elem_bytes,
  };
  uint32_t boxSize[rank] = {box0, box1, box2, box3};
  uint32_t elemStride[rank] = {1, 1, 1, 1};

  const auto oob_fill = (tma_dtype == CU_TENSOR_MAP_DATA_TYPE_UINT8)
      ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
      : CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;

  NVTE_CHECK_CUDA_DRIVER(cuDriverTensorMapEncodeTiled(
      &tensorMap, tma_dtype, rank, dataPtr, size, stride, boxSize, elemStride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      oob_fill));
}

// ---- TMA helpers ----
// Strided BSHD: TMA dims [D, H, S, B], coords [0, h, s, b]
// Strided SBHD: TMA dims [D, H, B, S], coords [0, h, b, s]

template <typename T, bool kIsBshdBshdBshd>
__device__ __forceinline__ void issue_tma_load_strided(T *smem_buf, const CUtensorMap *tma,
                                                       size_t h_i, size_t s_tile, size_t b_i,
                                                       uint64_t *mbar, size_t tile_bytes) {
  ptx::mbarrier_arrive_expect_tx(mbar, static_cast<uint32_t>(tile_bytes));
  if constexpr (kIsBshdBshdBshd) {
    cp_async_bulk_tensor_4d_global_to_shared(smem_buf, tma, 0, static_cast<uint32_t>(h_i),
                                             static_cast<uint32_t>(s_tile),
                                             static_cast<uint32_t>(b_i), mbar);
  } else {
    cp_async_bulk_tensor_4d_global_to_shared(smem_buf, tma, 0, static_cast<uint32_t>(h_i),
                                             static_cast<uint32_t>(b_i),
                                             static_cast<uint32_t>(s_tile), mbar);
  }
}

template <typename T, bool kIsBshdBshdBshd>
__device__ __forceinline__ void issue_tma_store_strided(const CUtensorMap *tma, T *smem_buf,
                                                        size_t h_i, size_t s_tile, size_t b_i) {
  if constexpr (kIsBshdBshdBshd) {
    cp_async_bulk_tensor_4d_shared_to_global(tma, 0, static_cast<uint32_t>(h_i),
                                             static_cast<uint32_t>(s_tile),
                                             static_cast<uint32_t>(b_i), smem_buf);
  } else {
    cp_async_bulk_tensor_4d_shared_to_global(tma, 0, static_cast<uint32_t>(h_i),
                                             static_cast<uint32_t>(b_i),
                                             static_cast<uint32_t>(s_tile), smem_buf);
  }
  ptx::cp_async_bulk_commit_group();
}

__device__ __forceinline__ void st_global_cs_uint4(uint4 *ptr, uint4 val) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
               "r"(val.z), "r"(val.w)
               : "memory");
}

// ---- forward: BSHD/SBHD → BHSD ----
// TMA load from strided input → smem → non-temporal stores to contiguous output.

template <typename T, bool kIsBshdBshdBshd>
__launch_bounds__(tma_permute_threads) __global__
    void permute_to_grouped_tensor_fwd_kernel(const __grid_constant__ CUtensorMap tma_q_in,
                                              const __grid_constant__ CUtensorMap tma_k_in,
                                              const __grid_constant__ CUtensorMap tma_v_in,
                                              T *__restrict__ q_out, T *__restrict__ k_out,
                                              T *__restrict__ v_out, size_t b, size_t s_q,
                                              size_t h_q, size_t d_qk, size_t d_qk_out,
                                              size_t s_kv, size_t h_kv,
                                              size_t d_v, size_t d_v_out,
                                              unsigned int permute_s_splits,
                                              size_t s_tile_size) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int which = blockIdx.z;
  const CUtensorMap *tma_in = which == 0 ? &tma_q_in : (which == 1 ? &tma_k_in : &tma_v_in);
  T *__restrict__ tensor_out = which == 0 ? q_out : (which == 1 ? k_out : v_out);
  const size_t Sdim = which == 0 ? s_q : s_kv;
  const size_t Hdim = which == 0 ? h_q : h_kv;
  const size_t Ddim = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);
  const size_t Ddim_out = which == 0 ? d_qk_out : (which == 1 ? d_qk_out : d_v_out);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;

  if (b_i >= b) return;
  if (which == 0) {
    if (h_i >= h_q) return;
  } else {
    if (h_i >= h_kv) return;
  }

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (Sdim * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (Sdim * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  if (s_begin >= s_end) return;

  const size_t out_base = b_i * Hdim * Sdim * Ddim_out + h_i * Sdim * Ddim_out;

  extern __shared__ __align__(128) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);

  __shared__ __align__(8) uint64_t mbar;
  const bool is_leader = (threadIdx.x == 0);

  if (is_leader) {
    ptx::mbarrier_init(&mbar, static_cast<uint32_t>(blockDim.x));
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const size_t S_TILE = s_tile_size;
  const uint32_t tile_bytes = static_cast<uint32_t>(S_TILE * Ddim * sizeof(T));
  int parity = 0;

  for (size_t s_tile = s_begin; s_tile < s_end; s_tile += S_TILE) {
    const size_t tile_rows = min(S_TILE, s_end - s_tile);

    if (is_leader) {
      issue_tma_load_strided<T, kIsBshdBshdBshd>(smem, tma_in, h_i, s_tile, b_i, &mbar, tile_bytes);
    } else {
      ptx::mbarrier_arrive(&mbar);
    }

    ptx::mbarrier_wait_parity(&mbar, parity);
    parity ^= 1;

    T *__restrict__ out_ptr = tensor_out + out_base + s_tile * Ddim_out;
    constexpr size_t vec_elems = sizeof(uint4) / sizeof(T);

    if (Ddim_out == Ddim) {
      const size_t total_elems = tile_rows * Ddim;
      for (size_t i = threadIdx.x * vec_elems; i < total_elems;
           i += static_cast<size_t>(blockDim.x) * vec_elems) {
        uint4 v = *reinterpret_cast<const uint4 *>(smem + i);
        st_global_cs_uint4(reinterpret_cast<uint4 *>(out_ptr + i), v);
      }
    } else {
      const size_t total_out_elems = tile_rows * Ddim_out;
      for (size_t i = threadIdx.x * vec_elems; i < total_out_elems;
           i += static_cast<size_t>(blockDim.x) * vec_elems) {
        const size_t row = i / Ddim_out;
        const size_t col = i % Ddim_out;
        uint4 v;
        if (col + vec_elems <= Ddim) {
          v = *reinterpret_cast<const uint4 *>(smem + row * Ddim + col);
        } else {
          memset(&v, 0, sizeof(v));
          const size_t smem_off = row * Ddim + col;
          size_t copy_elems = (col < Ddim) ? (Ddim - col) : 0;
          if (copy_elems > 0) memcpy(&v, smem + smem_off, copy_elems * sizeof(T));
        }
        st_global_cs_uint4(reinterpret_cast<uint4 *>(out_ptr + i), v);
      }
    }

    __syncthreads();
  }

  if (is_leader) {
    ptx::mbarrier_invalid(&mbar);
  }
#endif
}

// ---- backward: BHSD → BSHD/SBHD ----
// Vectorized loads from contiguous input → smem → TMA store to strided output.

template <typename T, bool kIsBshdBshdBshd>
__launch_bounds__(tma_permute_threads) __global__ void permute_to_grouped_tensor_bwd_kernel(
    const T *__restrict__ grad_q, const T *__restrict__ grad_k, const T *__restrict__ grad_v,
    const __grid_constant__ CUtensorMap tma_q_out, const __grid_constant__ CUtensorMap tma_k_out,
    const __grid_constant__ CUtensorMap tma_v_out, size_t b, size_t s_q, size_t h_q,
    size_t d_qk, size_t d_qk_in,
    size_t s_kv, size_t h_kv, size_t d_v, size_t d_v_in,
    unsigned int permute_s_splits, size_t s_tile_size) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int which = blockIdx.z;
  const T *__restrict__ tensor_in = which == 0 ? grad_q : (which == 1 ? grad_k : grad_v);
  const CUtensorMap *tma_out = which == 0 ? &tma_q_out : (which == 1 ? &tma_k_out : &tma_v_out);
  const size_t Sdim = which == 0 ? s_q : s_kv;
  const size_t Hdim = which == 0 ? h_q : h_kv;
  const size_t Ddim = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);
  const size_t Ddim_in = which == 0 ? d_qk_in : (which == 1 ? d_qk_in : d_v_in);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;

  if (b_i >= b) return;
  if (which == 0) {
    if (h_i >= h_q) return;
  } else {
    if (h_i >= h_kv) return;
  }

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (Sdim * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (Sdim * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  if (s_begin >= s_end) return;

  const size_t in_base = b_i * Hdim * Sdim * Ddim_in + h_i * Sdim * Ddim_in;

  extern __shared__ __align__(128) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);

  const size_t S_TILE = s_tile_size;
  constexpr size_t vec_elems = sizeof(uint4) / sizeof(T);

  for (size_t s_tile = s_begin; s_tile < s_end; s_tile += S_TILE) {
    const size_t tile_rows = min(S_TILE, s_end - s_tile);
    const size_t smem_elems = tile_rows * Ddim;

    if (Ddim_in == Ddim) {
      const T *__restrict__ in_ptr = tensor_in + in_base + s_tile * Ddim;
      for (size_t i = threadIdx.x * vec_elems; i < smem_elems;
           i += static_cast<size_t>(blockDim.x) * vec_elems) {
        *reinterpret_cast<uint4 *>(smem + i) = *reinterpret_cast<const uint4 *>(in_ptr + i);
      }
    } else {
      const T *__restrict__ in_ptr = tensor_in + in_base + s_tile * Ddim_in;
      const size_t total_in_elems = tile_rows * Ddim_in;
      for (size_t i = threadIdx.x; i < smem_elems;
           i += static_cast<size_t>(blockDim.x)) {
        const size_t row = i / Ddim;
        const size_t col = i % Ddim;
        smem[i] = in_ptr[row * Ddim_in + col];
      }
    }

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0) {
      issue_tma_store_strided<T, kIsBshdBshdBshd>(tma_out, smem, h_i, s_tile, b_i);
    }

    ptx::cp_async_bulk_wait_group();
    __syncthreads();
  }
#endif
}


// ---- create a 4D TMA descriptor ----
// For BSHD [B, S, H, D]: TMA dims [D, H, S, B], box [D, 1, S_TILE, 1]
// For SBHD [S, B, H, D]: TMA dims [D, H, B, S], box [D, 1, 1, S_TILE]

static void create_strided_tensor_map(CUtensorMap &map, void *ptr, DType dtype, size_t b, size_t s,
                                      size_t h, size_t d, size_t s_tile, bool is_bshd) {
  if (is_bshd) {
    create_4D_tensor_map(map, ptr, dtype, static_cast<uint64_t>(d), static_cast<uint64_t>(h),
                         static_cast<uint64_t>(s), static_cast<uint64_t>(b),
                         static_cast<uint32_t>(d), 1, static_cast<uint32_t>(s_tile), 1);
  } else {
    create_4D_tensor_map(map, ptr, dtype, static_cast<uint64_t>(d), static_cast<uint64_t>(h),
                         static_cast<uint64_t>(b), static_cast<uint64_t>(s),
                         static_cast<uint32_t>(d), 1, 1, static_cast<uint32_t>(s_tile));
  }
}

// ---- check if TMA path is feasible ----

static bool can_use_tma_permute(DType dtype, size_t d_qk, size_t d_v) {
  const int sm = cuda::sm_arch(cuda::current_device());
  if (sm < 90) return false;

  switch (dtype) {
    case DType::kFloat16:
    case DType::kBFloat16:
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
    case DType::kFloat8E8M0:
    case DType::kByte:
      break;
    default:
      return false;
  }
  const size_t elem_size = typeToSize(dtype);
  const size_t inner_qk = d_qk * elem_size;
  const size_t inner_v = d_v * elem_size;
  // hardware requirements for TMA
  if (inner_qk < 32 || inner_v < 32) return false;
  if (inner_qk % 16 != 0 || inner_v % 16 != 0) return false;
  return true;
}


void permute_to_grouped_tensor_fwd(Tensor q, Tensor k, Tensor v, Tensor q_out, Tensor k_out,
                                   Tensor v_out, NVTE_QKV_Format original_format,
                                   size_t num_tensors, cudaStream_t stream) {
  using namespace transformer_engine;
  const size_t b = q_out.shape()[0];
  const size_t h_q = q_out.shape()[1];
  const size_t s_q = q_out.shape()[2];
  const size_t h_kv = k_out.shape()[1];
  const size_t s_kv = k_out.shape()[2];

  const size_t d_qk = q.shape()[3];
  const size_t d_v  = v.shape()[3];
  const size_t d_qk_out = q_out.shape()[3];
  const size_t d_v_out  = v_out.shape()[3];

  const bool is_bshd = (original_format == NVTE_QKV_Format::NVTE_BSHD);

  if (!can_use_tma_permute(q.dtype(), d_qk, d_v)) {
    const size_t elem_size = typeToSize(q.dtype());
    const size_t d_qk_bytes = d_qk * elem_size;
    const size_t d_v_bytes  = d_v  * elem_size;
    const bool needs_transpose = (d_qk_bytes % 4 != 0) || (d_v_bytes % 4 != 0);

    if (needs_transpose) {
      const size_t s_max = std::max(s_q, s_kv);
      const size_t h_max = std::max(h_q, h_kv);
      const unsigned int st = static_cast<unsigned int>(
          (s_max + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
      const unsigned int ht = static_cast<unsigned int>(
          (h_max + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
      dim3 grid(static_cast<unsigned int>(b) * st, ht,
                static_cast<unsigned int>(num_tensors));
      const size_t d_max = std::max(d_qk, d_v);
      const size_t D_pad = (d_max * elem_size + 3u) & ~size_t(3);
      const size_t smem_bytes =
          static_cast<size_t>(TRANSPOSE_TILE) *
          (static_cast<size_t>(TRANSPOSE_TILE) * D_pad + 4);

      if (is_bshd) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
            q.dtype(), dtype,
            permute_to_grouped_tensor_fwd_fallback_not_vec_aligned_kernel<dtype, false>
                <<<grid, TRANSPOSE_BLOCK, smem_bytes, stream>>>(
                    reinterpret_cast<const dtype *>(q.data.dptr),
                    reinterpret_cast<const dtype *>(k.data.dptr),
                    reinterpret_cast<const dtype *>(v.data.dptr),
                    reinterpret_cast<dtype *>(q_out.data.dptr),
                    reinterpret_cast<dtype *>(k_out.data.dptr),
                    reinterpret_cast<dtype *>(v_out.data.dptr),
                    b, s_q, h_q, d_qk, d_qk_out, s_kv, h_kv, d_v, d_v_out, st););
      } else {
        TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
            q.dtype(), dtype,
            permute_to_grouped_tensor_fwd_fallback_not_vec_aligned_kernel<dtype, true>
                <<<grid, TRANSPOSE_BLOCK, smem_bytes, stream>>>(
                    reinterpret_cast<const dtype *>(q.data.dptr),
                    reinterpret_cast<const dtype *>(k.data.dptr),
                    reinterpret_cast<const dtype *>(v.data.dptr),
                    reinterpret_cast<dtype *>(q_out.data.dptr),
                    reinterpret_cast<dtype *>(k_out.data.dptr),
                    reinterpret_cast<dtype *>(v_out.data.dptr),
                    b, s_q, h_q, d_qk, d_qk_out, s_kv, h_kv, d_v, d_v_out, st););
      }
      NVTE_CHECK_CUDA(cudaGetLastError());
      return;
    }

    const size_t s_min = std::min(s_q, s_kv);
    const unsigned int permute_s_splits = std::max(
        1u, static_cast<unsigned int>(s_min / static_cast<size_t>(fallback_permute_threads)));
    const size_t h_grid = std::max(h_q, h_kv);
    dim3 grid(static_cast<unsigned int>(b * h_grid), permute_s_splits,
              static_cast<unsigned int>(num_tensors));

    if (is_bshd) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          q.dtype(), dtype,
          permute_to_grouped_tensor_fwd_fallback_vec_aligned_kernel<dtype, false>
              <<<grid, fallback_permute_threads, 0, stream>>>(
                  reinterpret_cast<const dtype *>(q.data.dptr),
                  reinterpret_cast<const dtype *>(k.data.dptr),
                  reinterpret_cast<const dtype *>(v.data.dptr),
                  reinterpret_cast<dtype *>(q_out.data.dptr),
                  reinterpret_cast<dtype *>(k_out.data.dptr),
                  reinterpret_cast<dtype *>(v_out.data.dptr),
                  b, s_q, h_q, d_qk, d_qk_out, s_kv, h_kv, d_v, d_v_out,
                  permute_s_splits););
    } else {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          q.dtype(), dtype,
          permute_to_grouped_tensor_fwd_fallback_vec_aligned_kernel<dtype, true>
              <<<grid, fallback_permute_threads, 0, stream>>>(
                  reinterpret_cast<const dtype *>(q.data.dptr),
                  reinterpret_cast<const dtype *>(k.data.dptr),
                  reinterpret_cast<const dtype *>(v.data.dptr),
                  reinterpret_cast<dtype *>(q_out.data.dptr),
                  reinterpret_cast<dtype *>(k_out.data.dptr),
                  reinterpret_cast<dtype *>(v_out.data.dptr),
                  b, s_q, h_q, d_qk, d_qk_out, s_kv, h_kv, d_v, d_v_out,
                  permute_s_splits););
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
    return;
  }

  const size_t elem_size = typeToSize(q.dtype());
  const size_t s_min = std::min(s_q, s_kv);
  const size_t s_tile = std::min(static_cast<size_t>(tma_permute_s_tile_default), s_min);
  NVTE_CHECK((s_tile * d_qk * elem_size) % sizeof(uint4) == 0 &&
             (s_tile * d_v  * elem_size) % sizeof(uint4) == 0,
             "permute_to_grouped_tensor_fwd: S_TILE(", s_tile, ") * D * elem_size must "
             "be divisible by ", sizeof(uint4), ". d_qk=", d_qk, ", d_v=", d_v,
             ", elem_size=", elem_size, ".");

  alignas(64) CUtensorMap tma_q_in{}, tma_k_in{}, tma_v_in{};
  create_strided_tensor_map(tma_q_in, q.data.dptr, q.dtype(), b, s_q, h_q, d_qk, s_tile, is_bshd);
  create_strided_tensor_map(tma_k_in, k.data.dptr, k.dtype(), b, s_kv, h_kv, d_qk, s_tile, is_bshd);
  create_strided_tensor_map(tma_v_in, v.data.dptr, v.dtype(), b, s_kv, h_kv, d_v, s_tile, is_bshd);

  const unsigned int permute_s_splits =
      std::max(1u, static_cast<unsigned int>(s_min / s_tile));
  const size_t h_grid = std::max(h_q, h_kv);
  dim3 grid(static_cast<unsigned int>(b * h_grid), permute_s_splits,
            static_cast<unsigned int>(num_tensors));

  const size_t d_max = std::max(d_qk, d_v);
  const size_t smem_bytes = s_tile * d_max * elem_size;

  if (is_bshd) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        q.dtype(), dtype, auto kernel = permute_to_grouped_tensor_fwd_kernel<dtype, true>;
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            tma_q_in, tma_k_in, tma_v_in, reinterpret_cast<dtype *>(q_out.data.dptr),
            reinterpret_cast<dtype *>(k_out.data.dptr), reinterpret_cast<dtype *>(v_out.data.dptr),
            b, s_q, h_q, d_qk, d_qk_out, s_kv, h_kv, d_v, d_v_out,
            permute_s_splits, s_tile););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        q.dtype(), dtype, auto kernel = permute_to_grouped_tensor_fwd_kernel<dtype, false>;
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            tma_q_in, tma_k_in, tma_v_in, reinterpret_cast<dtype *>(q_out.data.dptr),
            reinterpret_cast<dtype *>(k_out.data.dptr), reinterpret_cast<dtype *>(v_out.data.dptr),
            b, s_q, h_q, d_qk, d_qk_out, s_kv, h_kv, d_v, d_v_out,
            permute_s_splits, s_tile););
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void permute_to_grouped_tensor_bwd(Tensor grad_q, Tensor grad_k, Tensor grad_v, Tensor q, Tensor k,
                                   Tensor v, NVTE_QKV_Format original_format,
                                   size_t num_tensors, cudaStream_t stream) {
  using namespace transformer_engine;
  const size_t b = grad_q.shape()[0];
  const size_t h_q = grad_q.shape()[1];
  const size_t s_q = grad_q.shape()[2];
  const size_t h_kv = grad_k.shape()[1];
  const size_t s_kv = grad_k.shape()[2];

  const size_t d_qk = q.shape()[q.shape().size() - 1];
  const size_t d_v  = v.shape()[v.shape().size() - 1];
  const size_t d_qk_in = grad_q.shape()[3];
  const size_t d_v_in  = grad_v.shape()[3];

  const bool is_bshd = (original_format == NVTE_QKV_Format::NVTE_BSHD);

  if (!can_use_tma_permute(grad_q.dtype(), d_qk, d_v)) {
    const size_t elem_size = typeToSize(grad_q.dtype());
    const size_t d_qk_bytes = d_qk * elem_size;
    const size_t d_v_bytes  = d_v  * elem_size;
    const bool needs_transpose = (d_qk_bytes % 4 != 0) || (d_v_bytes % 4 != 0);

    if (needs_transpose) {
      const size_t s_max = std::max(s_q, s_kv);
      const size_t h_max = std::max(h_q, h_kv);
      const unsigned int st = static_cast<unsigned int>(
          (s_max + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
      const unsigned int ht = static_cast<unsigned int>(
          (h_max + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE);
      dim3 grid(static_cast<unsigned int>(b) * st, ht,
                static_cast<unsigned int>(num_tensors));
      const size_t d_max = std::max(d_qk, d_v);
      const size_t D_pad = (d_max * elem_size + 3u) & ~size_t(3);
      const size_t smem_bytes =
          static_cast<size_t>(TRANSPOSE_TILE) *
          (static_cast<size_t>(TRANSPOSE_TILE) * D_pad + 4);

      if (is_bshd) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
            grad_q.dtype(), dtype,
            permute_to_grouped_tensor_bwd_fallback_not_vec_aligned_kernel<dtype, false>
                <<<grid, TRANSPOSE_BLOCK, smem_bytes, stream>>>(
                    reinterpret_cast<const dtype *>(grad_q.data.dptr),
                    reinterpret_cast<const dtype *>(grad_k.data.dptr),
                    reinterpret_cast<const dtype *>(grad_v.data.dptr),
                    reinterpret_cast<dtype *>(q.data.dptr),
                    reinterpret_cast<dtype *>(k.data.dptr),
                    reinterpret_cast<dtype *>(v.data.dptr),
                    b, s_q, h_q, d_qk, d_qk_in, s_kv, h_kv, d_v, d_v_in, st););
      } else {
        TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
            grad_q.dtype(), dtype,
            permute_to_grouped_tensor_bwd_fallback_not_vec_aligned_kernel<dtype, true>
                <<<grid, TRANSPOSE_BLOCK, smem_bytes, stream>>>(
                    reinterpret_cast<const dtype *>(grad_q.data.dptr),
                    reinterpret_cast<const dtype *>(grad_k.data.dptr),
                    reinterpret_cast<const dtype *>(grad_v.data.dptr),
                    reinterpret_cast<dtype *>(q.data.dptr),
                    reinterpret_cast<dtype *>(k.data.dptr),
                    reinterpret_cast<dtype *>(v.data.dptr),
                    b, s_q, h_q, d_qk, d_qk_in, s_kv, h_kv, d_v, d_v_in, st););
      }
      NVTE_CHECK_CUDA(cudaGetLastError());
      return;
    }

    const size_t s_min = std::min(s_q, s_kv);
    const unsigned int permute_s_splits = std::max(
        1u, static_cast<unsigned int>(s_min / static_cast<size_t>(fallback_permute_threads)));
    const size_t h_grid = std::max(h_q, h_kv);
    dim3 grid(static_cast<unsigned int>(b * h_grid), permute_s_splits,
              static_cast<unsigned int>(num_tensors));

    if (is_bshd) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          grad_q.dtype(), dtype,
          permute_to_grouped_tensor_bwd_fallback_vec_aligned_kernel<dtype, false>
              <<<grid, fallback_permute_threads, 0, stream>>>(
                  reinterpret_cast<const dtype *>(grad_q.data.dptr),
                  reinterpret_cast<const dtype *>(grad_k.data.dptr),
                  reinterpret_cast<const dtype *>(grad_v.data.dptr),
                  reinterpret_cast<dtype *>(q.data.dptr),
                  reinterpret_cast<dtype *>(k.data.dptr),
                  reinterpret_cast<dtype *>(v.data.dptr),
                  b, s_q, h_q, d_qk, d_qk_in, s_kv, h_kv, d_v, d_v_in,
                  permute_s_splits););
    } else {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          grad_q.dtype(), dtype,
          permute_to_grouped_tensor_bwd_fallback_vec_aligned_kernel<dtype, true>
              <<<grid, fallback_permute_threads, 0, stream>>>(
                  reinterpret_cast<const dtype *>(grad_q.data.dptr),
                  reinterpret_cast<const dtype *>(grad_k.data.dptr),
                  reinterpret_cast<const dtype *>(grad_v.data.dptr),
                  reinterpret_cast<dtype *>(q.data.dptr),
                  reinterpret_cast<dtype *>(k.data.dptr),
                  reinterpret_cast<dtype *>(v.data.dptr),
                  b, s_q, h_q, d_qk, d_qk_in, s_kv, h_kv, d_v, d_v_in,
                  permute_s_splits););
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
    return;
  }

  const size_t elem_size = typeToSize(grad_q.dtype());
  const size_t s_min = std::min(s_q, s_kv);
  const size_t s_tile = std::min(static_cast<size_t>(tma_permute_s_tile_default), s_min);
  NVTE_CHECK((s_tile * d_qk * elem_size) % sizeof(uint4) == 0 &&
             (s_tile * d_v  * elem_size) % sizeof(uint4) == 0,
             "permute_to_grouped_tensor_bwd: S_TILE(", s_tile, ") * D * elem_size must "
             "be divisible by ", sizeof(uint4), ". d_qk=", d_qk, ", d_v=", d_v,
             ", elem_size=", elem_size, ".");

  alignas(64) CUtensorMap tma_q_out{}, tma_k_out{}, tma_v_out{};
  create_strided_tensor_map(tma_q_out, q.data.dptr, q.dtype(), b, s_q, h_q, d_qk, s_tile, is_bshd);
  create_strided_tensor_map(tma_k_out, k.data.dptr, k.dtype(), b, s_kv, h_kv, d_qk, s_tile, is_bshd);
  create_strided_tensor_map(tma_v_out, v.data.dptr, v.dtype(), b, s_kv, h_kv, d_v, s_tile, is_bshd);

  const unsigned int permute_s_splits =
      std::max(1u, static_cast<unsigned int>(s_min / s_tile));
  const size_t h_grid = std::max(h_q, h_kv);
  dim3 grid(static_cast<unsigned int>(b * h_grid), permute_s_splits,
            static_cast<unsigned int>(num_tensors));

  const size_t d_max = std::max(d_qk, d_v);
  const size_t smem_bytes = s_tile * d_max * elem_size;

  if (is_bshd) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        grad_q.dtype(), dtype, auto kernel = permute_to_grouped_tensor_bwd_kernel<dtype, true>;
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            reinterpret_cast<const dtype *>(grad_q.data.dptr),
            reinterpret_cast<const dtype *>(grad_k.data.dptr),
            reinterpret_cast<const dtype *>(grad_v.data.dptr), tma_q_out, tma_k_out, tma_v_out, b,
            s_q, h_q, d_qk, d_qk_in, s_kv, h_kv, d_v, d_v_in,
            permute_s_splits, s_tile););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        grad_q.dtype(), dtype, auto kernel = permute_to_grouped_tensor_bwd_kernel<dtype, false>;
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            reinterpret_cast<const dtype *>(grad_q.data.dptr),
            reinterpret_cast<const dtype *>(grad_k.data.dptr),
            reinterpret_cast<const dtype *>(grad_v.data.dptr), tma_q_out, tma_k_out, tma_v_out, b,
            s_q, h_q, d_qk, d_qk_in, s_kv, h_kv, d_v, d_v_in,
            permute_s_splits, s_tile););
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace permute_to_grouped_tensor

// ===================================================================================
// multi_tensor_pad_last_dim: pad the last dimension of multiple tensors to a certain alignment
// ===================================================================================

namespace multi_tensor_pad_last_dim {

constexpr int pad_threads_per_block = 256;
constexpr int kMaxPadTensors = 16;

struct PadLastDimArgs {
  const uint8_t *input;
  uint32_t *output;
  size_t n_uint32;
  uint32_t in_row_bytes;
  uint32_t out_row_uint32;
};

struct MultiPadParams {
  PadLastDimArgs tensors[kMaxPadTensors];
};

__launch_bounds__(pad_threads_per_block) __global__
    void multi_tensor_pad_last_dim_kernel(MultiPadParams params) {
  const auto &a = params.tensors[blockIdx.y];

  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < a.n_uint32;
       idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint32_t col_byte = (idx % a.out_row_uint32) * 4;
    const size_t row = idx / a.out_row_uint32;
    const uint8_t *__restrict__ src = a.input + row * static_cast<size_t>(a.in_row_bytes);

    uint32_t val;
    if (col_byte + 4 <= a.in_row_bytes) {
      memcpy(&val, src + col_byte, 4);
    } else if (col_byte >= a.in_row_bytes) {
      val = 0;
    } else {
      val = 0;
      memcpy(&val, src + col_byte, a.in_row_bytes - col_byte);
    }
    a.output[idx] = val;
  }
}

void multi_tensor_pad_last_dim(Tensor *inputs, Tensor *outputs, size_t num_tensors,
                        cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(num_tensors > 0 && num_tensors <= kMaxPadTensors,
             "num_tensors must be in [1, ", kMaxPadTensors, "], got ", num_tensors, ".");

  MultiPadParams params{};
  size_t max_n_uint32 = 0;
  int kernel_count = 0;

  for (size_t i = 0; i < num_tensors; ++i) {
    auto &inp = inputs[i];
    auto &out = outputs[i];

    NVTE_CHECK(inp.data.shape.size() == 2, "Expected 2D input tensor at index ", i, ".");
    NVTE_CHECK(out.data.shape.size() == 2, "Expected 2D output tensor at index ", i, ".");
    NVTE_CHECK(inp.data.dtype == out.data.dtype, "Dtype mismatch at index ", i, ".");

    const size_t rows = inp.data.shape[0];
    const size_t in_cols = inp.data.shape[1];
    const size_t out_cols = out.data.shape[1];

    NVTE_CHECK(out.data.shape[0] == rows, "Row count mismatch at index ", i, ".");
    NVTE_CHECK(out_cols >= in_cols, "out_cols < in_cols at index ", i, ".");

    if (rows == 0) continue;

    if (in_cols == out_cols) {
      const size_t total_bytes = rows * in_cols * typeToSize(inp.data.dtype);
      NVTE_CHECK_CUDA(cudaMemcpyAsync(out.data.dptr, inp.data.dptr, total_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
      continue;
    }

    const size_t elem_size = typeToSize(inp.data.dtype);
    const auto in_row_bytes = static_cast<uint32_t>(in_cols * elem_size);
    const auto out_row_bytes = static_cast<uint32_t>(out_cols * elem_size);
    NVTE_CHECK(out_row_bytes % 4 == 0,
               "Padded row size in bytes (", out_row_bytes, ") must be a multiple of 4.");

    const uint32_t out_row_uint32 = out_row_bytes / 4;
    const size_t n_uint32 = rows * out_row_uint32;

    params.tensors[kernel_count] = {reinterpret_cast<const uint8_t *>(inp.data.dptr),
                                    reinterpret_cast<uint32_t *>(out.data.dptr), n_uint32,
                                    in_row_bytes, out_row_uint32};
    max_n_uint32 = std::max(max_n_uint32, n_uint32);
    ++kernel_count;
  }

  if (kernel_count == 0) return;

  constexpr int threads = pad_threads_per_block;
  const int blocks_x =
      static_cast<int>(std::min(DIVUP(max_n_uint32, static_cast<size_t>(threads)),
                                static_cast<size_t>(65535)));
  dim3 grid(blocks_x, kernel_count);

  multi_tensor_pad_last_dim_kernel<<<grid, threads, 0, stream>>>(params);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_pad_last_dim
}  // namespace transformer_engine

void nvte_prepare_flash_attn_fwd(NVTETensor qkvi, NVTETensor qkv, cudaStream_t stream) {
  NVTE_API_CALL(nvte_prepare_flash_attn_fwd);
  using namespace transformer_engine;

  flash_attention::prepare_flash_attn_fwd(*convertNVTETensorCheck(qkvi),
                                          *convertNVTETensorCheck(qkv), stream);
}

void nvte_prepare_flash_attn_bwd(NVTETensor q, NVTETensor k, NVTETensor v, NVTETensor qkv,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_prepare_flash_attn_bwd);
  using namespace transformer_engine;

  flash_attention::prepare_flash_attn_bwd(*convertNVTETensorCheck(q), *convertNVTETensorCheck(k),
                                          *convertNVTETensorCheck(v), *convertNVTETensorCheck(qkv),
                                          stream);
}

void nvte_permute_to_grouped_tensor_fwd(NVTETensor q, NVTETensor k, NVTETensor v, NVTETensor q_out,
                                        NVTETensor k_out, NVTETensor v_out,
                                        NVTE_QKV_Format original_format, size_t num_tensors,
                                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_permute_to_grouped_tensor_fwd);
  using namespace transformer_engine;

  permute_to_grouped_tensor::permute_to_grouped_tensor_fwd(
      *convertNVTETensorCheck(q), *convertNVTETensorCheck(k), *convertNVTETensorCheck(v),
      *convertNVTETensorCheck(q_out), *convertNVTETensorCheck(k_out),
      *convertNVTETensorCheck(v_out), original_format, num_tensors, stream);
}

void nvte_permute_to_grouped_tensor_bwd(NVTETensor grad_q, NVTETensor grad_k, NVTETensor grad_v,
                                        NVTETensor q, NVTETensor k, NVTETensor v,
                                        NVTE_QKV_Format original_format, size_t num_tensors,
                                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_permute_to_grouped_tensor_bwd);
  using namespace transformer_engine;

  permute_to_grouped_tensor::permute_to_grouped_tensor_bwd(
      *convertNVTETensorCheck(grad_q), *convertNVTETensorCheck(grad_k),
      *convertNVTETensorCheck(grad_v), *convertNVTETensorCheck(q), *convertNVTETensorCheck(k),
      *convertNVTETensorCheck(v), original_format, num_tensors, stream);
}

void nvte_multi_tensor_pad_last_dim(NVTETensor *inputs, NVTETensor *outputs, size_t num_tensors,
                             cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_pad_last_dim);
  using namespace transformer_engine;

  std::vector<Tensor> in_vec(num_tensors), out_vec(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    in_vec[i] = *convertNVTETensorCheck(inputs[i]);
    out_vec[i] = *convertNVTETensorCheck(outputs[i]);
  }
  multi_tensor_pad_last_dim::multi_tensor_pad_last_dim(in_vec.data(), out_vec.data(), num_tensors, stream);
}
