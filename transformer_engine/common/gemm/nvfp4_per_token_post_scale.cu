/*************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_per_token_post_scale.cu
 *  \brief NVFP4 per-token GEMM-output post-scale: d[i,j] *= r_A[i] * r_B[j].
 *
 *  Standalone bf16 epilogue applied after cuBLAS LT NVFP4 GEMM with the
 *  operand amaxes pinned to 1.0. See nvfp4_per_token.h for the math chain.
 */

#include <transformer_engine/nvfp4_per_token.h>

#include "../common.h"
#include "../util/logging.h"
#include "../util/ptx.cuh"

namespace transformer_engine {
namespace nvfp4_per_token {

namespace {

// Each block tiles 16 rows x 256 cols of the output: amaxes are loaded
// once into SMEM, then each thread handles 8 cols via a 16-byte int4 LD/ST
// for peak HBM coalescing on SM100. Wrapper enforces M, N % 128 alignment.
constexpr int kTileCols = 256;
constexpr int kTileRows = 16;
constexpr int kElemsPerThread = 8;  // bf16x8 = 16-byte vector
constexpr int kThreadsX = kTileCols / kElemsPerThread;
constexpr int kThreadsY = kTileRows;
constexpr int kThreadsPerBlock = kThreadsX * kThreadsY;
static_assert(kTileCols % kElemsPerThread == 0, "kTileCols must be a multiple of kElemsPerThread");
static_assert(kElemsPerThread * sizeof(__nv_bfloat16) == sizeof(int4),
              "kElemsPerThread bf16 must pack into a single int4 (16 bytes)");

__global__ void __launch_bounds__(kThreadsPerBlock)
    per_token_post_scale_kernel(__nv_bfloat16* __restrict__ d, const float* __restrict__ row_amax_a,
                                const float* __restrict__ row_amax_b, const int M, const int N) {
  __shared__ float s_row_amax[kTileRows];
  __shared__ float s_col_amax[kTileCols];

  const int row_tile = blockIdx.y * kTileRows;
  const int col_tile = blockIdx.x * kTileCols;

  // Cooperatively load row + col amaxes into SMEM (272 floats / 512 threads).
  const int tid = threadIdx.y * kThreadsX + threadIdx.x;
  if (tid < kTileRows) {
    const int gi = row_tile + tid;
    s_row_amax[tid] = (gi < M) ? row_amax_a[gi] : 0.0f;
  }
  if (tid < kTileCols) {
    const int gj = col_tile + tid;
    s_col_amax[tid] = (gj < N) ? row_amax_b[gj] : 0.0f;
  }
  __syncthreads();

  const int i = row_tile + threadIdx.y;
  const int j0 = col_tile + threadIdx.x * kElemsPerThread;
  if (i >= M || j0 >= N) return;

  const float a = s_row_amax[threadIdx.y];
  const size_t base = static_cast<size_t>(i) * N + j0;

  // Fast path = 16-byte aligned LD/ST; slow path = boundary tile fallback.
  if (j0 + kElemsPerThread <= N) {
    // __align__(16) is required for the int4 reinterpret_cast to be defined.
    __nv_bfloat16 __align__(16) chunk[kElemsPerThread];
    *reinterpret_cast<int4*>(chunk) = *reinterpret_cast<const int4*>(&d[base]);
#pragma unroll
    for (int e = 0; e < kElemsPerThread; ++e) {
      const float b = s_col_amax[threadIdx.x * kElemsPerThread + e];
      const float current = static_cast<float>(chunk[e]);
      chunk[e] = static_cast<__nv_bfloat16>(current * a * b);
    }
    *reinterpret_cast<int4*>(&d[base]) = *reinterpret_cast<const int4*>(chunk);
  } else {
#pragma unroll
    for (int e = 0; e < kElemsPerThread; ++e) {
      const int j = j0 + e;
      if (j >= N) break;
      const float b = s_col_amax[threadIdx.x * kElemsPerThread + e];
      const size_t idx = base + e;
      const float current = static_cast<float>(d[idx]);
      d[idx] = static_cast<__nv_bfloat16>(current * a * b);
    }
  }
}

}  // namespace

void per_token_post_scale(Tensor* d, const Tensor& row_amax_a, const Tensor& row_amax_b,
                          cudaStream_t stream) {
  NVTE_CHECK(d->has_data(), "NVFP4 per-token post-scale: d has no data.");
  NVTE_CHECK(d->data.dtype == DType::kBFloat16,
             "NVFP4 per-token post-scale: d must be BF16 (got non-BF16 dtype).");
  NVTE_CHECK(row_amax_a.data.dtype == DType::kFloat32,
             "NVFP4 per-token post-scale: row_amax_a must be FP32.");
  NVTE_CHECK(row_amax_b.data.dtype == DType::kFloat32,
             "NVFP4 per-token post-scale: row_amax_b must be FP32.");

  const auto& d_shape = d->data.shape;
  NVTE_CHECK(d_shape.size() == 2,
             "NVFP4 per-token post-scale: d must be 2D, got rank=", d_shape.size());
  const int M = static_cast<int>(d_shape[0]);
  const int N = static_cast<int>(d_shape[1]);
  NVTE_CHECK(row_amax_a.data.numel() == static_cast<size_t>(M),
             "NVFP4 per-token post-scale: row_amax_a numel must equal M=", M, ", got ",
             row_amax_a.data.numel());
  NVTE_CHECK(row_amax_b.data.numel() == static_cast<size_t>(N),
             "NVFP4 per-token post-scale: row_amax_b numel must equal N=", N, ", got ",
             row_amax_b.data.numel());

  if (M == 0 || N == 0) {
    return;
  }

  // 32 x 16 threads = 512/block; covers 256 cols x 16 rows = 4096 elems/block.
  dim3 block(kThreadsX, kThreadsY, 1);
  dim3 grid((N + kTileCols - 1) / kTileCols, (M + kTileRows - 1) / kTileRows, 1);
  per_token_post_scale_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16*>(d->data.dptr),
      reinterpret_cast<const float*>(row_amax_a.data.dptr),
      reinterpret_cast<const float*>(row_amax_b.data.dptr), M, N);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace nvfp4_per_token
}  // namespace transformer_engine

void nvte_nvfp4_per_token_post_scale(NVTETensor d, const NVTETensor row_amax_a,
                                     const NVTETensor row_amax_b, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_per_token_post_scale);
  using namespace transformer_engine;

  transformer_engine::nvfp4_per_token::per_token_post_scale(
      convertNVTETensorCheck(d), *convertNVTETensorCheck(row_amax_a),
      *convertNVTETensorCheck(row_amax_b), stream);
}
