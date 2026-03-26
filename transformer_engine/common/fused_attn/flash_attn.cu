/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>

#include "../common.h"
#include "transformer_engine/fused_attn.h"

namespace transformer_engine {
namespace flash_attention {

constexpr int warp_size = 32;
constexpr int type_size = 2;  // FP16 or BF16
constexpr int nvec = sizeof(uint64_t) / type_size;
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
    uint64_t *out = reinterpret_cast<uint64_t *>(my_output + i * load_size);
    *out = *reinterpret_cast<const uint64_t *>(my_input + i * load_size * 3);
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
    uint64_t *out = reinterpret_cast<uint64_t *>(my_output + i * load_size * 3);
    *out = *reinterpret_cast<const uint64_t *>(my_input + i * load_size);
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

template <typename T>
__launch_bounds__(block_size) __global__ void permute_to_grouped_tensor_fwd_kernel(
    T *q, T *k, T *v, T *q_out, T *k_out, T *v_out,
    size_t b, size_t s_q, size_t h_q, size_t d_qk, size_t s_kv, size_t h_kv, size_t d_v,
    NVTE_QKV_Layout original_layout) {
  const int which_tensor = blockIdx.y;
  T *tensor_in = which_tensor == 0 ? q : (which_tensor == 1 ? k : v);
  T *tensor_out = which_tensor == 0 ? q_out : (which_tensor == 1 ? k_out : v_out);
  const size_t s = which_tensor == 0 ? s_q : s_kv;
  const size_t h = which_tensor == 0 ? h_q : h_kv;
  const size_t d = which_tensor == 0 ? d_qk : (which_tensor == 1 ? d_qk : d_v);

  const size_t warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  size_t s_i, b_i;
  if (original_layout == NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD) {
    s_i = warpid / b;
    b_i = warpid % b;
  } else if (original_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    b_i = warpid / s;
    s_i = warpid % s;
  } else {
    return;
  }
  if (s_i >= s) return;
  if (b_i >= b) return;

  const T *input_base;
  if (original_layout == NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD) {
    input_base = tensor_in + (s_i * b + b_i) * h * d;
  } else {
    input_base = tensor_in + (b_i * s + s_i) * h * d;
  }

  const size_t id_in_warp = threadIdx.x % warp_size;
  // SBHD/BSHD [..,H,D] -> BHSD out[b,h,s,d] = in[...,h,d] with (s_i,b_i) fixed per warp.
  for (int jj = 0; jj < static_cast<int>(d); jj += load_size) {
    const size_t d_off = static_cast<size_t>(jj) + id_in_warp * nvec;
    if (d_off + nvec > d) continue;
    for (int i = 0; i < static_cast<int>(h); ++i) {
      const T *input_ptr = input_base + static_cast<size_t>(i) * d + d_off;
      T *output_ptr = tensor_out + b_i * h * s * d + static_cast<size_t>(i) * s * d + s_i * d + d_off;
      *reinterpret_cast<uint64_t *>(output_ptr) = *reinterpret_cast<const uint64_t *>(input_ptr);
    }
  }
}

template <typename T>
__launch_bounds__(block_size) __global__ void permute_to_grouped_tensor_bwd_kernel(
    T *grad_q, T *grad_k, T *grad_v, T *q, T *k, T *v,
    size_t b, size_t s_q, size_t h_q, size_t d_qk, size_t s_kv, size_t h_kv, size_t d_v,
    NVTE_QKV_Layout original_layout) {
  const int which_tensor = blockIdx.y;
  T *tensor_in = which_tensor == 0 ? grad_q : (which_tensor == 1 ? grad_k : grad_v);
  T *tensor_out = which_tensor == 0 ? q : (which_tensor == 1 ? k : v);
  const size_t s = which_tensor == 0 ? s_q : s_kv;
  const size_t h = which_tensor == 0 ? h_q : h_kv;
  const size_t d = which_tensor == 0 ? d_qk : (which_tensor == 1 ? d_qk : d_v);

  const size_t warpid = (blockDim.x * blockIdx.x + threadIdx.x) / warp_size;
  size_t s_i, b_i;
  if (original_layout == NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD) {
    s_i = warpid / b;
    b_i = warpid % b;
  } else if (original_layout == NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD) {
    b_i = warpid / s;
    s_i = warpid % s;
  } else {
    return;
  }
  if (s_i >= s) return;
  if (b_i >= b) return;

  T *output_base;
  if (original_layout == NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD) {
    output_base = tensor_out + (s_i * b + b_i) * h * d;
  } else {
    output_base = tensor_out + (b_i * s + s_i) * h * d;
  }

  const size_t id_in_warp = threadIdx.x % warp_size;
  for (int jj = 0; jj < static_cast<int>(d); jj += load_size) {
    const size_t d_off = static_cast<size_t>(jj) + id_in_warp * nvec;
    if (d_off + nvec > d) continue;
    for (int i = 0; i < static_cast<int>(h); ++i) {
      const T *input_ptr =
          tensor_in + b_i * h * s * d + static_cast<size_t>(i) * s * d + s_i * d + d_off;
      T *output_ptr = output_base + static_cast<size_t>(i) * d + d_off;
      *reinterpret_cast<uint64_t *>(output_ptr) = *reinterpret_cast<const uint64_t *>(input_ptr);
    }
  }
}

void permute_to_grouped_tensor_fwd(Tensor q, Tensor k, Tensor v, Tensor q_out, Tensor k_out,
                                   Tensor v_out, NVTE_QKV_Layout original_layout, cudaStream_t stream) {
  using namespace transformer_engine;
  size_t b=0, s_q=0, s_kv=0, h_q = 0, h_kv = 0, d_qk = 0, d_v = 0;
  b = q_out.shape()[0];
  h_q = q_out.shape()[1];
  s_q = q_out.shape()[2];
  d_qk = q_out.shape()[3];
  h_kv = k_out.shape()[1];
  s_kv = k_out.shape()[2];
  d_v = v_out.shape()[3];

  size_t warps = b * std::max(s_q, s_kv);
  size_t warps_per_block = block_size / warp_size;
  size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(blocks, 3);
  int threads = block_size;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      q.dtype(), dtype,
      permute_to_grouped_tensor_fwd_kernel<dtype><<<grid, threads, 0, stream>>>(
          reinterpret_cast<dtype *>(q.data.dptr), reinterpret_cast<dtype *>(k.data.dptr),
          reinterpret_cast<dtype *>(v.data.dptr), reinterpret_cast<dtype *>(q_out.data.dptr),
          reinterpret_cast<dtype *>(k_out.data.dptr), reinterpret_cast<dtype *>(v_out.data.dptr),
          b, s_q, h_q, d_qk, s_kv, h_kv, d_v, original_layout);
  );
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void permute_to_grouped_tensor_bwd(Tensor grad_q, Tensor grad_k, Tensor grad_v,
                                   Tensor q, Tensor k, Tensor v,
                                   NVTE_QKV_Layout original_layout, cudaStream_t stream) {
  using namespace transformer_engine;
  size_t b=0, s_q=0, s_kv=0, h_q = 0, h_kv = 0, d_qk = 0, d_v = 0;
  b = grad_q.shape()[0];
  h_q = grad_q.shape()[1];
  s_q = grad_q.shape()[2];
  d_qk = grad_q.shape()[3];
  h_kv = grad_k.shape()[1];
  s_kv = grad_k.shape()[2];
  d_v = grad_v.shape()[3];

  size_t warps = b * std::max(s_q, s_kv);
  size_t warps_per_block = block_size / warp_size;
  size_t blocks = (warps + warps_per_block - 1) / warps_per_block;
  dim3 grid(blocks, 3);
  int threads = block_size;

  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(
      grad_q.dtype(), dtype,
      permute_to_grouped_tensor_bwd_kernel<dtype><<<grid, threads, 0, stream>>>(
          reinterpret_cast<dtype *>(grad_q.data.dptr), reinterpret_cast<dtype *>(grad_k.data.dptr),
          reinterpret_cast<dtype *>(grad_v.data.dptr), reinterpret_cast<dtype *>(q.data.dptr),
          reinterpret_cast<dtype *>(k.data.dptr), reinterpret_cast<dtype *>(v.data.dptr),
          b, s_q, h_q, d_qk, s_kv, h_kv, d_v, original_layout);
  );
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
                                        NVTETensor k_out, NVTETensor v_out, NVTE_QKV_Layout original_layout,
                                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_permute_to_grouped_tensor_fwd);
  using namespace transformer_engine;

  flash_attention::permute_to_grouped_tensor_fwd(
      *convertNVTETensorCheck(q), *convertNVTETensorCheck(k), *convertNVTETensorCheck(v),
      *convertNVTETensorCheck(q_out), *convertNVTETensorCheck(k_out),
      *convertNVTETensorCheck(v_out), original_layout, stream);
}

void nvte_permute_to_grouped_tensor_bwd(NVTETensor grad_q, NVTETensor grad_k, NVTETensor grad_v,
                                        NVTETensor q, NVTETensor k, NVTETensor v, NVTE_QKV_Layout original_layout,
                                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_permute_to_grouped_tensor_bwd);
  using namespace transformer_engine;

  flash_attention::permute_to_grouped_tensor_bwd(
      *convertNVTETensorCheck(grad_q), *convertNVTETensorCheck(grad_k), *convertNVTETensorCheck(grad_v),
      *convertNVTETensorCheck(q), *convertNVTETensorCheck(k), *convertNVTETensorCheck(v), original_layout, stream);
}