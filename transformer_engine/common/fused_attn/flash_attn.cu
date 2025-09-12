/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

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
