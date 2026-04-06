/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>

#include "../common.h"
#include "../util/cuda_driver.h"
#include "../util/ptx.cuh"
#include "../utils.cuh"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace flash_attention {

/// Packed vector of N elements of T; alignment matches a single wide load/store of N * sizeof(T) bytes.
template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
};

constexpr int warp_size = 32;
constexpr int type_size = 2;  // FP16 or BF16
constexpr int nvec = sizeof(uint64_t) / type_size;
constexpr int nvec128 = sizeof(uint4) / type_size;
constexpr int load_size = warp_size * nvec;
constexpr int block_size = 512;

// TMA permute kernel configuration
constexpr int tma_permute_threads = 32;
constexpr int tma_permute_s_tile = 32;

// ---- 4D TMA PTX wrappers ----

__device__ __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
    void *dst_shmem, const CUtensorMap *tensor_map,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t dst = __cvta_generic_to_shared(dst_shmem);
  uint32_t bar = __cvta_generic_to_shared(mbar);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6];"
      ::"r"(dst), "l"(tensor_map),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3),
        "r"(bar)
      : "memory");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_4d_global_to_shared requires SM 10.0+.");
#endif
}

__device__ __forceinline__ void cp_async_bulk_tensor_4d_shared_to_global(
    const CUtensorMap *tensor_map,
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    void *src_shmem) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t src = __cvta_generic_to_shared(src_shmem);
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
      " [%0, {%1, %2, %3, %4}], [%5];"
      ::"l"(tensor_map),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3),
        "r"(src)
      : "memory");
#else
  NVTE_DEVICE_ERROR("cp_async_bulk_tensor_4d_shared_to_global requires SM 9.0+.");
#endif
}

// ---- Host-side 4D tensor map creation ----
//
// Creates a 4D TMA descriptor for a densely-packed tensor whose logical
// dimensions (innermost-first) are [dim0, dim1, dim2, dim3].
//
// The box (tile) copied per TMA instruction is [box0, box1, box2, box3].

static void create_4D_tensor_map(CUtensorMap &tensorMap, void *dataPtr, DType dtype,
                                 uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                 uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3) {
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
    default:
      NVTE_ERROR("create_4D_tensor_map: unsupported dtype");
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

  NVTE_CHECK_CUDA_DRIVER(cuDriverTensorMapEncodeTiled(
      &tensorMap, tma_dtype, rank, dataPtr, size, stride, boxSize, elemStride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
}

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

// ---- TMA helpers for strided (BSHD/SBHD) tensors ----
//
// Strided BSHD: TMA dims [D, H, S, B], coords [0, h, s, b]
// Strided SBHD: TMA dims [D, H, B, S], coords [0, h, b, s]

template <typename T, bool kIsBshdBshdBshd>
__device__ __forceinline__ void issue_tma_load_strided(
    T *smem_buf, const CUtensorMap *tma,
    size_t h_i, size_t s_tile, size_t b_i,
    uint64_t *mbar, size_t tile_bytes) {
  ptx::mbarrier_arrive_expect_tx(mbar, static_cast<uint32_t>(tile_bytes));
  if constexpr (kIsBshdBshdBshd) {
    cp_async_bulk_tensor_4d_global_to_shared(
        smem_buf, tma,
        0, static_cast<uint32_t>(h_i),
        static_cast<uint32_t>(s_tile), static_cast<uint32_t>(b_i),
        mbar);
  } else {
    cp_async_bulk_tensor_4d_global_to_shared(
        smem_buf, tma,
        0, static_cast<uint32_t>(h_i),
        static_cast<uint32_t>(b_i), static_cast<uint32_t>(s_tile),
        mbar);
  }
}

template <typename T, bool kIsBshdBshdBshd>
__device__ __forceinline__ void issue_tma_store_strided(
    const CUtensorMap *tma, T *smem_buf,
    size_t h_i, size_t s_tile, size_t b_i) {
  if constexpr (kIsBshdBshdBshd) {
    cp_async_bulk_tensor_4d_shared_to_global(
        tma,
        0, static_cast<uint32_t>(h_i),
        static_cast<uint32_t>(s_tile), static_cast<uint32_t>(b_i),
        smem_buf);
  } else {
    cp_async_bulk_tensor_4d_shared_to_global(
        tma,
        0, static_cast<uint32_t>(h_i),
        static_cast<uint32_t>(b_i), static_cast<uint32_t>(s_tile),
        smem_buf);
  }
  ptx::cp_async_bulk_commit_group();
}

__device__ __forceinline__ void st_global_cs_uint4(uint4 *ptr, uint4 val) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1, %2, %3, %4};"
      :: "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
      : "memory");
}

// ---- Forward: BSHD/SBHD → BHSD ----
//
// TMA load from strided input → smem → non-temporal stores to contiguous output.

template <typename T, bool kIsBshdBshdBshd>
__launch_bounds__(tma_permute_threads) __global__
    void permute_to_grouped_tensor_fwd_kernel(
        const __grid_constant__ CUtensorMap tma_q_in,
        const __grid_constant__ CUtensorMap tma_k_in,
        const __grid_constant__ CUtensorMap tma_v_in,
        T *__restrict__ q_out, T *__restrict__ k_out, T *__restrict__ v_out,
        size_t b, size_t s_q, size_t h_q, size_t d_qk,
        size_t s_kv, size_t h_kv, size_t d_v,
        unsigned int permute_s_splits) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int which = blockIdx.z;
  const CUtensorMap *tma_in = which == 0 ? &tma_q_in : (which == 1 ? &tma_k_in : &tma_v_in);
  T *__restrict__ tensor_out = which == 0 ? q_out : (which == 1 ? k_out : v_out);
  const size_t Sdim = which == 0 ? s_q : s_kv;
  const size_t Hdim = which == 0 ? h_q : h_kv;
  const size_t Ddim = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;

  if (b_i >= b) return;
  if (which == 0) { if (h_i >= h_q) return; }
  else            { if (h_i >= h_kv) return; }

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (Sdim * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (Sdim * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  if (s_begin >= s_end) return;

  const size_t out_base = b_i * Hdim * Sdim * Ddim + h_i * Sdim * Ddim;

  extern __shared__ __align__(128) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);

  __shared__ __align__(8) uint64_t mbar;
  const bool is_leader = (threadIdx.x == 0);

  if (is_leader) {
    ptx::mbarrier_init(&mbar, static_cast<uint32_t>(blockDim.x));
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  constexpr size_t S_TILE = tma_permute_s_tile;
  const uint32_t tile_bytes = static_cast<uint32_t>(S_TILE * Ddim * sizeof(T));
  int parity = 0;

  for (size_t s_tile = s_begin; s_tile < s_end; s_tile += S_TILE) {
    const size_t tile_rows = min(S_TILE, s_end - s_tile);

    if (is_leader) {
      issue_tma_load_strided<T, kIsBshdBshdBshd>(
          smem, tma_in, h_i, s_tile, b_i, &mbar, tile_bytes);
    } else {
      ptx::mbarrier_arrive(&mbar);
    }

    ptx::mbarrier_wait_parity(&mbar, parity);
    parity ^= 1;

    T *__restrict__ out_ptr = tensor_out + out_base + s_tile * Ddim;
    const size_t total_elems = tile_rows * Ddim;
    constexpr size_t vec_elems = sizeof(uint4) / sizeof(T);

    for (size_t i = threadIdx.x * vec_elems; i < total_elems;
         i += static_cast<size_t>(blockDim.x) * vec_elems) {
      uint4 v = *reinterpret_cast<const uint4 *>(smem + i);
      st_global_cs_uint4(reinterpret_cast<uint4 *>(out_ptr + i), v);
    }

    __syncthreads();
  }

  if (is_leader) {
    ptx::mbarrier_invalid(&mbar);
  }
#endif
}

// ---- Backward: BHSD → BSHD/SBHD ----
//
// Vectorized loads from contiguous input → smem → TMA store to strided output.

template <typename T, bool kIsBshdBshdBshd>
__launch_bounds__(tma_permute_threads) __global__
    void permute_to_grouped_tensor_bwd_kernel(
        const T *__restrict__ grad_q, const T *__restrict__ grad_k, const T *__restrict__ grad_v,
        const __grid_constant__ CUtensorMap tma_q_out,
        const __grid_constant__ CUtensorMap tma_k_out,
        const __grid_constant__ CUtensorMap tma_v_out,
        size_t b, size_t s_q, size_t h_q, size_t d_qk,
        size_t s_kv, size_t h_kv, size_t d_v,
        unsigned int permute_s_splits) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int which = blockIdx.z;
  const T *__restrict__ tensor_in =
      which == 0 ? grad_q : (which == 1 ? grad_k : grad_v);
  const CUtensorMap *tma_out = which == 0 ? &tma_q_out : (which == 1 ? &tma_k_out : &tma_v_out);
  const size_t Sdim = which == 0 ? s_q : s_kv;
  const size_t Hdim = which == 0 ? h_q : h_kv;
  const size_t Ddim = which == 0 ? d_qk : (which == 1 ? d_qk : d_v);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;

  if (b_i >= b) return;
  if (which == 0) { if (h_i >= h_q) return; }
  else            { if (h_i >= h_kv) return; }

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (Sdim * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (Sdim * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  if (s_begin >= s_end) return;

  const size_t in_base = b_i * Hdim * Sdim * Ddim + h_i * Sdim * Ddim;

  extern __shared__ __align__(128) char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);

  constexpr size_t S_TILE = tma_permute_s_tile;
  constexpr size_t vec_elems = sizeof(uint4) / sizeof(T);

  for (size_t s_tile = s_begin; s_tile < s_end; s_tile += S_TILE) {
    const size_t tile_rows = min(S_TILE, s_end - s_tile);

    const T *__restrict__ in_ptr = tensor_in + in_base + s_tile * Ddim;
    const size_t total_elems = tile_rows * Ddim;

    for (size_t i = threadIdx.x * vec_elems; i < total_elems;
         i += static_cast<size_t>(blockDim.x) * vec_elems) {
      *reinterpret_cast<uint4 *>(smem + i) =
          *reinterpret_cast<const uint4 *>(in_ptr + i);
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

// Helper: create a 4D TMA descriptor for the strided (BSHD or SBHD) tensor.
//
// For BSHD [B, S, H, D]: TMA dims [D, H, S, B], box [D, 1, S_TILE, 1]
// For SBHD [S, B, H, D]: TMA dims [D, H, B, S], box [D, 1, 1, S_TILE]
static void create_strided_tensor_map(CUtensorMap &map, void *ptr, DType dtype,
                                      size_t b, size_t s, size_t h, size_t d,
                                      bool is_bshd) {
  if (is_bshd) {
    create_4D_tensor_map(map, ptr, dtype,
                         static_cast<uint64_t>(d), static_cast<uint64_t>(h),
                         static_cast<uint64_t>(s), static_cast<uint64_t>(b),
                         static_cast<uint32_t>(d), 1,
                         static_cast<uint32_t>(tma_permute_s_tile), 1);
  } else {
    create_4D_tensor_map(map, ptr, dtype,
                         static_cast<uint64_t>(d), static_cast<uint64_t>(h),
                         static_cast<uint64_t>(b), static_cast<uint64_t>(s),
                         static_cast<uint32_t>(d), 1, 1,
                         static_cast<uint32_t>(tma_permute_s_tile));
  }
}

void permute_to_grouped_tensor_fwd(Tensor q, Tensor k, Tensor v, Tensor q_out, Tensor k_out,
                                   Tensor v_out, NVTE_QKV_Layout original_layout,
                                   cudaStream_t stream) {
  using namespace transformer_engine;
  const size_t b    = q_out.shape()[0];
  const size_t h_q  = q_out.shape()[1];
  const size_t s_q  = q_out.shape()[2];
  const size_t d_qk = q_out.shape()[3];
  const size_t h_kv = k_out.shape()[1];
  const size_t s_kv = k_out.shape()[2];
  const size_t d_v  = v_out.shape()[3];

  NVTE_CHECK(d_qk % nvec128 == 0 && d_v % nvec128 == 0,
             "permute_to_grouped_tensor_fwd: head dim must be divisible by ", nvec128, ".");

  const bool is_bshd = (original_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD);

  alignas(64) CUtensorMap tma_q_in{}, tma_k_in{}, tma_v_in{};
  create_strided_tensor_map(tma_q_in, q.data.dptr, q.dtype(), b, s_q, h_q, d_qk, is_bshd);
  create_strided_tensor_map(tma_k_in, k.data.dptr, k.dtype(), b, s_kv, h_kv, d_qk, is_bshd);
  create_strided_tensor_map(tma_v_in, v.data.dptr, v.dtype(), b, s_kv, h_kv, d_v, is_bshd);

  const size_t s_min = std::min(s_q, s_kv);
  const unsigned int permute_s_splits =
      std::max(1u, static_cast<unsigned int>(s_min / static_cast<size_t>(tma_permute_threads)));
  const size_t h_grid = std::max(h_q, h_kv);
  dim3 grid(static_cast<unsigned int>(b * h_grid), permute_s_splits, 3);

  const size_t d_max = std::max(d_qk, d_v);
  const size_t smem_bytes = tma_permute_s_tile * d_max * sizeof(uint16_t);

  if (is_bshd) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        q.dtype(), dtype,
        auto kernel = permute_to_grouped_tensor_fwd_kernel<dtype, true>;
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            tma_q_in, tma_k_in, tma_v_in,
            reinterpret_cast<dtype *>(q_out.data.dptr),
            reinterpret_cast<dtype *>(k_out.data.dptr),
            reinterpret_cast<dtype *>(v_out.data.dptr),
            b, s_q, h_q, d_qk, s_kv, h_kv, d_v, permute_s_splits););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        q.dtype(), dtype,
        auto kernel = permute_to_grouped_tensor_fwd_kernel<dtype, false>;
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            tma_q_in, tma_k_in, tma_v_in,
            reinterpret_cast<dtype *>(q_out.data.dptr),
            reinterpret_cast<dtype *>(k_out.data.dptr),
            reinterpret_cast<dtype *>(v_out.data.dptr),
            b, s_q, h_q, d_qk, s_kv, h_kv, d_v, permute_s_splits););
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void permute_to_grouped_tensor_bwd(Tensor grad_q, Tensor grad_k, Tensor grad_v, Tensor q, Tensor k,
                                   Tensor v, NVTE_QKV_Layout original_layout, cudaStream_t stream) {
  using namespace transformer_engine;
  const size_t b    = grad_q.shape()[0];
  const size_t h_q  = grad_q.shape()[1];
  const size_t s_q  = grad_q.shape()[2];
  const size_t d_qk = grad_q.shape()[3];
  const size_t h_kv = grad_k.shape()[1];
  const size_t s_kv = grad_k.shape()[2];
  const size_t d_v  = grad_v.shape()[3];

  NVTE_CHECK(d_qk % nvec128 == 0 && d_v % nvec128 == 0,
             "permute_to_grouped_tensor_bwd: head dim must be divisible by ", nvec128, ".");

  const bool is_bshd = (original_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD);

  alignas(64) CUtensorMap tma_q_out{}, tma_k_out{}, tma_v_out{};
  create_strided_tensor_map(tma_q_out, q.data.dptr, q.dtype(), b, s_q, h_q, d_qk, is_bshd);
  create_strided_tensor_map(tma_k_out, k.data.dptr, k.dtype(), b, s_kv, h_kv, d_qk, is_bshd);
  create_strided_tensor_map(tma_v_out, v.data.dptr, v.dtype(), b, s_kv, h_kv, d_v, is_bshd);

  const size_t s_min = std::min(s_q, s_kv);
  const unsigned int permute_s_splits =
      std::max(1u, static_cast<unsigned int>(s_min / static_cast<size_t>(tma_permute_threads)));
  const size_t h_grid = std::max(h_q, h_kv);
  dim3 grid(static_cast<unsigned int>(b * h_grid), permute_s_splits, 3);

  const size_t d_max = std::max(d_qk, d_v);
  const size_t smem_bytes = tma_permute_s_tile * d_max * sizeof(uint16_t);

  if (is_bshd) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        grad_q.dtype(), dtype,
        auto kernel = permute_to_grouped_tensor_bwd_kernel<dtype, true>;
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            reinterpret_cast<const dtype *>(grad_q.data.dptr),
            reinterpret_cast<const dtype *>(grad_k.data.dptr),
            reinterpret_cast<const dtype *>(grad_v.data.dptr),
            tma_q_out, tma_k_out, tma_v_out,
            b, s_q, h_q, d_qk, s_kv, h_kv, d_v, permute_s_splits););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        grad_q.dtype(), dtype,
        auto kernel = permute_to_grouped_tensor_bwd_kernel<dtype, false>;
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        kernel<<<grid, tma_permute_threads, smem_bytes, stream>>>(
            reinterpret_cast<const dtype *>(grad_q.data.dptr),
            reinterpret_cast<const dtype *>(grad_k.data.dptr),
            reinterpret_cast<const dtype *>(grad_v.data.dptr),
            tma_q_out, tma_k_out, tma_v_out,
            b, s_q, h_q, d_qk, s_kv, h_kv, d_v, permute_s_splits););
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}
}  // namespace flash_attention
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
                                        NVTE_QKV_Layout original_layout, cudaStream_t stream) {
  NVTE_API_CALL(nvte_permute_to_grouped_tensor_fwd);
  using namespace transformer_engine;

  flash_attention::permute_to_grouped_tensor_fwd(
      *convertNVTETensorCheck(q), *convertNVTETensorCheck(k), *convertNVTETensorCheck(v),
      *convertNVTETensorCheck(q_out), *convertNVTETensorCheck(k_out),
      *convertNVTETensorCheck(v_out), original_layout, stream);
}

void nvte_permute_to_grouped_tensor_bwd(NVTETensor grad_q, NVTETensor grad_k, NVTETensor grad_v,
                                        NVTETensor q, NVTETensor k, NVTETensor v,
                                        NVTE_QKV_Layout original_layout, cudaStream_t stream) {
  NVTE_API_CALL(nvte_permute_to_grouped_tensor_bwd);
  using namespace transformer_engine;

  flash_attention::permute_to_grouped_tensor_bwd(
      *convertNVTETensorCheck(grad_q), *convertNVTETensorCheck(grad_k),
      *convertNVTETensorCheck(grad_v), *convertNVTETensorCheck(q), *convertNVTETensorCheck(k),
      *convertNVTETensorCheck(v), original_layout, stream);
}
