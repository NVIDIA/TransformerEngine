/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
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

template <typename T, bool kIsBshdBshdBshd>
__launch_bounds__(1024) __global__
    void permute_to_grouped_tensor_fwd_kernel(const T *__restrict__ q, const T *__restrict__ k,
                                              const T *__restrict__ v, T *__restrict__ q_out,
                                              T *__restrict__ k_out, T *__restrict__ v_out,
                                              size_t b, size_t s_q, size_t h_q, size_t d_qk,
                                              size_t s_kv, size_t h_kv, size_t d_v,
                                              unsigned int permute_s_splits) {
  const int which_tensor = blockIdx.z;
  const T *__restrict__ tensor_in = which_tensor == 0 ? q : (which_tensor == 1 ? k : v);
  T *__restrict__ tensor_out = which_tensor == 0 ? q_out : (which_tensor == 1 ? k_out : v_out);
  const size_t Sdim = which_tensor == 0 ? s_q : s_kv;
  const size_t Hdim = which_tensor == 0 ? h_q : h_kv;
  const size_t Ddim = which_tensor == 0 ? d_qk : (which_tensor == 1 ? d_qk : d_v);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;

  if (b_i >= b) return;
  if (which_tensor == 0) {
    if (h_i >= h_q) return;
  } else {
    if (h_i >= h_kv) return;
  }
  if (Ddim % static_cast<size_t>(nvec) != 0) return;

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (Sdim * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (Sdim * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  const size_t S_chunk = s_end - s_begin;

  const size_t in_base = kIsBshdBshdBshd ? b_i * Sdim * Hdim * Ddim : b_i * Hdim * Ddim;
  const size_t out_base = b_i * Hdim * Sdim * Ddim + h_i * Sdim * Ddim;
  const bool use_vec128 =
      (Ddim % static_cast<size_t>(nvec128) == 0) &&
      ((reinterpret_cast<uintptr_t>(tensor_in) % alignof(Vec<T, nvec128>)) == 0) &&
      ((reinterpret_cast<uintptr_t>(tensor_out) % alignof(Vec<T, nvec128>)) == 0);

  if (use_vec128) {
    const size_t d_vec = Ddim / static_cast<size_t>(nvec128);
    const size_t total_work = S_chunk * d_vec;
    for (size_t w = static_cast<size_t>(threadIdx.x); w < total_work;
         w += static_cast<size_t>(blockDim.x)) {
      const size_t s_local = w / d_vec;
      const size_t s_i = s_begin + s_local;
      const size_t v = w % d_vec;
      const size_t d_off = v * static_cast<size_t>(nvec128);

      const T *__restrict__ in_ptr;
      if constexpr (kIsBshdBshdBshd) {
        in_ptr = tensor_in + in_base + s_i * Hdim * Ddim + h_i * Ddim + d_off;
      } else {
        in_ptr = tensor_in + s_i * b * Hdim * Ddim + in_base + h_i * Ddim + d_off;
      }
      T *__restrict__ out_ptr = tensor_out + out_base + s_i * Ddim + d_off;
      *reinterpret_cast<Vec<T, nvec128> *>(out_ptr) =
          *reinterpret_cast<const Vec<T, nvec128> *>(in_ptr);
    }
  } else {
    const size_t d_vec = Ddim / static_cast<size_t>(nvec);
    const size_t total_work = S_chunk * d_vec;
    for (size_t w = static_cast<size_t>(threadIdx.x); w < total_work;
         w += static_cast<size_t>(blockDim.x)) {
      const size_t s_local = w / d_vec;
      const size_t s_i = s_begin + s_local;
      const size_t v = w % d_vec;
      const size_t d_off = v * static_cast<size_t>(nvec);

      const T *__restrict__ in_ptr;
      if constexpr (kIsBshdBshdBshd) {
        in_ptr = tensor_in + in_base + s_i * Hdim * Ddim + h_i * Ddim + d_off;
      } else {
        in_ptr = tensor_in + s_i * b * Hdim * Ddim + in_base + h_i * Ddim + d_off;
      }
      T *__restrict__ out_ptr = tensor_out + out_base + s_i * Ddim + d_off;
      *reinterpret_cast<Vec<T, nvec> *>(out_ptr) = *reinterpret_cast<const Vec<T, nvec> *>(in_ptr);
    }
  }
}

template <typename T, bool kIsBshdBshdBshd>
__launch_bounds__(1024) __global__ void permute_to_grouped_tensor_bwd_kernel(
    const T *__restrict__ grad_q, const T *__restrict__ grad_k, const T *__restrict__ grad_v,
    T *__restrict__ q, T *__restrict__ k, T *__restrict__ v, size_t b, size_t s_q, size_t h_q,
    size_t d_qk, size_t s_kv, size_t h_kv, size_t d_v, unsigned int permute_s_splits) {
  const int which_tensor = blockIdx.z;
  const T *__restrict__ tensor_in =
      which_tensor == 0 ? grad_q : (which_tensor == 1 ? grad_k : grad_v);
  T *__restrict__ tensor_out = which_tensor == 0 ? q : (which_tensor == 1 ? k : v);
  const size_t Sdim = which_tensor == 0 ? s_q : s_kv;
  const size_t Hdim = which_tensor == 0 ? h_q : h_kv;
  const size_t Ddim = which_tensor == 0 ? d_qk : (which_tensor == 1 ? d_qk : d_v);

  const size_t h_grid = h_q > h_kv ? h_q : h_kv;
  const size_t b_i = static_cast<size_t>(blockIdx.x) / h_grid;
  const size_t h_i = static_cast<size_t>(blockIdx.x) % h_grid;

  if (b_i >= b) return;
  if (which_tensor == 0) {
    if (h_i >= h_q) return;
  } else {
    if (h_i >= h_kv) return;
  }
  if (Ddim % static_cast<size_t>(nvec) != 0) return;

  const unsigned int s_part = blockIdx.y;
  const size_t s_begin =
      (Sdim * static_cast<size_t>(s_part)) / static_cast<size_t>(permute_s_splits);
  const size_t s_end =
      (Sdim * static_cast<size_t>(s_part + 1)) / static_cast<size_t>(permute_s_splits);
  const size_t S_chunk = s_end - s_begin;

  const size_t in_base = b_i * Hdim * Sdim * Ddim + h_i * Sdim * Ddim;
  const size_t out_base =
      kIsBshdBshdBshd ? b_i * Sdim * Hdim * Ddim + h_i * Ddim : b_i * Hdim * Ddim + h_i * Ddim;
  const bool use_vec128 =
      (Ddim % static_cast<size_t>(nvec128) == 0) &&
      ((reinterpret_cast<uintptr_t>(tensor_in) % alignof(Vec<T, nvec128>)) == 0) &&
      ((reinterpret_cast<uintptr_t>(tensor_out) % alignof(Vec<T, nvec128>)) == 0);

  if (use_vec128) {
    const size_t d_vec = Ddim / static_cast<size_t>(nvec128);
    const size_t total_work = S_chunk * d_vec;
    for (size_t w = static_cast<size_t>(threadIdx.x); w < total_work;
         w += static_cast<size_t>(blockDim.x)) {
      const size_t s_local = w / d_vec;
      const size_t s_i = s_begin + s_local;
      const size_t v = w % d_vec;
      const size_t d_off = v * static_cast<size_t>(nvec128);

      const T *__restrict__ in_ptr = tensor_in + in_base + s_i * Ddim + d_off;
      T *__restrict__ out_ptr;
      if constexpr (kIsBshdBshdBshd) {
        out_ptr = tensor_out + out_base + s_i * Hdim * Ddim + d_off;
      } else {
        out_ptr = tensor_out + s_i * b * Hdim * Ddim + out_base + d_off;
      }
      *reinterpret_cast<Vec<T, nvec128> *>(out_ptr) =
          *reinterpret_cast<const Vec<T, nvec128> *>(in_ptr);
    }
  } else {
    const size_t d_vec = Ddim / static_cast<size_t>(nvec);
    const size_t total_work = S_chunk * d_vec;
    for (size_t w = static_cast<size_t>(threadIdx.x); w < total_work;
         w += static_cast<size_t>(blockDim.x)) {
      const size_t s_local = w / d_vec;
      const size_t s_i = s_begin + s_local;
      const size_t v = w % d_vec;
      const size_t d_off = v * static_cast<size_t>(nvec);

      const T *__restrict__ in_ptr = tensor_in + in_base + s_i * Ddim + d_off;
      T *__restrict__ out_ptr;
      if constexpr (kIsBshdBshdBshd) {
        out_ptr = tensor_out + out_base + s_i * Hdim * Ddim + d_off;
      } else {
        out_ptr = tensor_out + s_i * b * Hdim * Ddim + out_base + d_off;
      }
      *reinterpret_cast<Vec<T, nvec> *>(out_ptr) = *reinterpret_cast<const Vec<T, nvec> *>(in_ptr);
    }
  }
}

void permute_to_grouped_tensor_fwd(Tensor q, Tensor k, Tensor v, Tensor q_out, Tensor k_out,
                                   Tensor v_out, NVTE_QKV_Layout original_layout,
                                   cudaStream_t stream) {
  using namespace transformer_engine;
  size_t b = 0, s_q = 0, s_kv = 0, h_q = 0, h_kv = 0, d_qk = 0, d_v = 0;
  b = q_out.shape()[0];
  h_q = q_out.shape()[1];
  s_q = q_out.shape()[2];
  d_qk = q_out.shape()[3];
  h_kv = k_out.shape()[1];
  s_kv = k_out.shape()[2];
  d_v = v_out.shape()[3];

  NVTE_CHECK(d_qk % nvec == 0 && d_v % nvec == 0,
             "permute_to_grouped_tensor_fwd: head dim must be divisible by vector width.");
  // Split S across grid.y; work out permute_s_splits so S_chunk >= threads
  const int threads = 1024;
  const size_t s_min = std::min(s_q, s_kv);
  const unsigned int permute_s_splits =
      std::max(1u, static_cast<unsigned int>(s_min / static_cast<size_t>(threads)));
  const size_t h_grid = std::max(h_q, h_kv);
  dim3 grid(b * h_grid, permute_s_splits, 3);

  if (original_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        q.dtype(), dtype,
        permute_to_grouped_tensor_fwd_kernel<dtype, true><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const dtype *>(q.data.dptr),
            reinterpret_cast<const dtype *>(k.data.dptr),
            reinterpret_cast<const dtype *>(v.data.dptr),
            reinterpret_cast<dtype *>(q_out.data.dptr), reinterpret_cast<dtype *>(k_out.data.dptr),
            reinterpret_cast<dtype *>(v_out.data.dptr), b, s_q, h_q, d_qk, s_kv, h_kv, d_v,
            permute_s_splits););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        q.dtype(), dtype,
        permute_to_grouped_tensor_fwd_kernel<dtype, false><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const dtype *>(q.data.dptr),
            reinterpret_cast<const dtype *>(k.data.dptr),
            reinterpret_cast<const dtype *>(v.data.dptr),
            reinterpret_cast<dtype *>(q_out.data.dptr), reinterpret_cast<dtype *>(k_out.data.dptr),
            reinterpret_cast<dtype *>(v_out.data.dptr), b, s_q, h_q, d_qk, s_kv, h_kv, d_v,
            permute_s_splits););
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void permute_to_grouped_tensor_bwd(Tensor grad_q, Tensor grad_k, Tensor grad_v, Tensor q, Tensor k,
                                   Tensor v, NVTE_QKV_Layout original_layout, cudaStream_t stream) {
  using namespace transformer_engine;
  size_t b = 0, s_q = 0, s_kv = 0, h_q = 0, h_kv = 0, d_qk = 0, d_v = 0;
  b = grad_q.shape()[0];
  h_q = grad_q.shape()[1];
  s_q = grad_q.shape()[2];
  d_qk = grad_q.shape()[3];
  h_kv = grad_k.shape()[1];
  s_kv = grad_k.shape()[2];
  d_v = grad_v.shape()[3];

  NVTE_CHECK(d_qk % nvec == 0 && d_v % nvec == 0,
             "permute_to_grouped_tensor_bwd: head dim must be divisible by vector width.");
  const int threads = 1024;
  const size_t s_min = std::min(s_q, s_kv);
  const unsigned int permute_s_splits =
      std::max(1u, static_cast<unsigned int>(s_min / static_cast<size_t>(threads)));
  const size_t h_grid = std::max(h_q, h_kv);
  dim3 grid(b * h_grid, permute_s_splits, 3);

  if (original_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        grad_q.dtype(), dtype,
        permute_to_grouped_tensor_bwd_kernel<dtype, true><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const dtype *>(grad_q.data.dptr),
            reinterpret_cast<const dtype *>(grad_k.data.dptr),
            reinterpret_cast<const dtype *>(grad_v.data.dptr),
            reinterpret_cast<dtype *>(q.data.dptr), reinterpret_cast<dtype *>(k.data.dptr),
            reinterpret_cast<dtype *>(v.data.dptr), b, s_q, h_q, d_qk, s_kv, h_kv, d_v,
            permute_s_splits););
  } else {
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
        grad_q.dtype(), dtype,
        permute_to_grouped_tensor_bwd_kernel<dtype, false><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const dtype *>(grad_q.data.dptr),
            reinterpret_cast<const dtype *>(grad_k.data.dptr),
            reinterpret_cast<const dtype *>(grad_v.data.dptr),
            reinterpret_cast<dtype *>(q.data.dptr), reinterpret_cast<dtype *>(k.data.dptr),
            reinterpret_cast<dtype *>(v.data.dptr), b, s_q, h_q, d_qk, s_kv, h_kv, d_v,
            permute_s_splits););
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
