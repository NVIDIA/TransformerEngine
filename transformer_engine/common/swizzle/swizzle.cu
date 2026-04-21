/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/swizzle.h>

#include <cassert>
#include <numeric>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace {

constexpr int MXFP8_BLOCK_SIZE = 32;
constexpr int NVFP4_BLOCK_SIZE = 16;

int get_max_dynamic_smem() {
  static int max_smem = -1;
  if (max_smem < 0) {
    int device;
    NVTE_CHECK_CUDA(cudaGetDevice(&device));
    NVTE_CHECK_CUDA(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  }
  return max_smem;
}

constexpr __device__ __host__ int TB_DIM = 32;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K = 16;
constexpr __device__ __host__ int N_SF_PER_TD_PER_TILE = 4;

// output is in ~K-major interleaved blocks
constexpr __device__ __host__ int NEW_SF_TILE_DIM_K_I32 = NEW_SF_TILE_DIM_K / 4;
constexpr __device__ __host__ int NEW_SF_TILE_DIM_M_I32 = 32;

template <typename LType>
__device__ inline void regs_shuffle_with_bit_shifts(LType* regs_vec) {
  // inp, 4-byte chunks [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
  // out, swapping byte to form new 4-byte chunks [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t new_regs[kVectorSize];
  int32_t* regs = reinterpret_cast<int32_t*>(regs_vec);

#pragma unroll
  for (int i = 0; i < N_TILE_PER_TD; i++) {
#pragma unroll
    for (int j = 0; j < N_SF_PER_TD_PER_TILE; j++) {
      new_regs[i * N_SF_PER_TD_PER_TILE + j] =
          (((regs[i + 0 * N_TILE_PER_TD] >> 8 * j) & 0xFF)) |
          (((regs[i + 1 * N_TILE_PER_TD] >> 8 * j) & 0xFF) << 8) |
          (((regs[i + 2 * N_TILE_PER_TD] >> 8 * j) & 0xFF) << 16) |
          (((regs[i + 3 * N_TILE_PER_TD] >> 8 * j) & 0xFF) << 24);
    }
  }
#pragma unroll
  for (int i = 0; i < kVectorSize; i++) regs[i] = new_regs[i];
}

template <typename LType>
__device__ inline void regs_unshuffle_with_bit_shifts(LType* regs_vec) {
  // Inverse of regs_shuffle_with_bit_shifts
  // inp, 4-byte chunks [0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15]
  // out, swapping byte to form new 4-byte chunks [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t new_regs[kVectorSize];
  int32_t* regs = reinterpret_cast<int32_t*>(regs_vec);

#pragma unroll
  for (int i = 0; i < N_TILE_PER_TD; i++) {
#pragma unroll
    for (int j = 0; j < N_SF_PER_TD_PER_TILE; j++) {
      new_regs[i + j * N_TILE_PER_TD] =
          ((regs[i * N_SF_PER_TD_PER_TILE + 0] >> 8 * j) & 0xFF) |
          (((regs[i * N_SF_PER_TD_PER_TILE + 1] >> 8 * j) & 0xFF) << 8) |
          (((regs[i * N_SF_PER_TD_PER_TILE + 2] >> 8 * j) & 0xFF) << 16) |
          (((regs[i * N_SF_PER_TD_PER_TILE + 3] >> 8 * j) & 0xFF) << 24);
    }
  }
#pragma unroll
  for (int i = 0; i < kVectorSize; i++) regs[i] = new_regs[i];
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_col_scaling_kernel_impl(const void* input, void* output, const int M,
                                                const int K, const int original_M,
                                                const int original_K, const int bid_x,
                                                const int bid_y, const int grid_dim_x,
                                                const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_SF_PER_TD = N_TILE_PER_TD * N_SF_PER_TD_PER_TILE;
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;

  // input is in M-major
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M / 4;
  constexpr int SF_TILE_DIM_K_I32 = SF_TILE_DIM_K;

  const int M_i32 = M / 4;
  const int K_i32 = K;

  int m_tiles_in_tb = N_TILE_PER_TD;
  int k_tiles_in_tb = TB_DIM;
  if (bid_x == grid_dim_x - 1) {
    k_tiles_in_tb = (K_i32 / SF_TILE_DIM_K_I32 - 1) % k_tiles_in_tb + 1;
  }
  if (bid_y == grid_dim_y - 1) {
    m_tiles_in_tb = (M_i32 / SF_TILE_DIM_M_I32 - 1) % m_tiles_in_tb + 1;
  }

  bool padding_m = (bid_y == grid_dim_y - 1) && (original_M < M);
  bool padding_k = (bid_x == grid_dim_x - 1) && (original_K < K);

  const int input_offset =
      bid_x * TB_DIM * SF_TILE_DIM_K_I32 * M_i32 + bid_y * N_TILE_PER_TD * SF_TILE_DIM_M_I32;
  const int32_t* input_i32 = reinterpret_cast<const int32_t*>(input) + input_offset;
  int32_t* output_i32[N_TILE_PER_TD];
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    output_i32[i] = reinterpret_cast<int32_t*>(output) + bid_x * TB_DIM * SF_TILE_SIZE_I32 +
                    (bid_y * N_TILE_PER_TD + i) * SF_TILE_DIM_M_I32 * K_i32;
  }
  extern __shared__ int slm[];

  // load, global -> regs
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < m_tiles_in_tb * SF_TILE_DIM_M_I32 &&
      threadIdx.y < k_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset =
          (threadIdx.y * SF_TILE_DIM_K_I32 + i) * M_i32 + threadIdx.x * N_TILE_PER_TD;
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(input_i32 + thread_offset));
      // Pad zeros
      if (padding_m || padding_k) {
        for (int j = 0; j < N_TILE_PER_TD * sizeof(int); j++) {
          const int index = (input_offset + thread_offset) * sizeof(int) + j;
          if (index / M >= original_K || index % M >= original_M) {
            reinterpret_cast<uint8_t*>(regs_vec + i)[j] = 0;
          }
        }
      }
    }

    // local shuffle
    regs_shuffle_with_bit_shifts(regs_vec);

    // store, regs -> shared
    int tM = threadIdx.x * N_SF_PER_TD;
    int* slm_tile = slm + (threadIdx.y * SF_TILE_SIZE_I32 +
                           tM / SF_TILE_DIM_M * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD; i++) {
      /* TODO rotate_i */
      slm_tile[(tM % SF_TILE_DIM_M) / NEW_SF_TILE_DIM_M_I32 +
               ((tM + i) % NEW_SF_TILE_DIM_M_I32) * NEW_SF_TILE_DIM_K_I32] =
          reinterpret_cast<int*>(regs_vec)[i];
    }
  }
  __syncthreads();

  // store, shared -> global
  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    __align__(16) int4* output_v4i = reinterpret_cast<int4*>(output_i32[i]);
    __align__(16) int4* slm_v4i =
        reinterpret_cast<int4*>(slm + i * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int j = linear_id; j < SF_TILE_SIZE_I32 * k_tiles_in_tb / 4;
         j += blockDim.x * blockDim.y) {
      output_v4i[j] = slm_v4i[j];
    }
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_col_scaling_kernel(const void* input, void* output, const int M, const int K,
                               const int original_M, const int original_K) {
  swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
}

template <typename LType>
__device__ inline void regs_shuffle(LType* regs_vec) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  if constexpr (N_TILE_PER_TD == 1) return;

  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t tmp[kVectorSize];
  int32_t* ptr = reinterpret_cast<int32_t*>(regs_vec);
#pragma unroll
  for (int i = 0; i < kVectorSize; i++)
    tmp[i % N_TILE_PER_TD * N_SF_PER_TD_PER_TILE + i / N_TILE_PER_TD] = ptr[i];

#pragma unroll
  for (int i = 0; i < kVectorSize; i++) ptr[i] = tmp[i];
}

// Inverse of regs_shuffle.
template <typename LType>
__device__ inline void regs_unshuffle(LType* regs_vec) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  if constexpr (N_TILE_PER_TD == 1) return;

  constexpr int kVectorSize = N_SF_PER_TD_PER_TILE * N_TILE_PER_TD;
  int32_t tmp[kVectorSize];
  int32_t* ptr = reinterpret_cast<int32_t*>(regs_vec);
#pragma unroll
  for (int i = 0; i < kVectorSize; i++)
    tmp[i % N_SF_PER_TD_PER_TILE * N_TILE_PER_TD + i / N_SF_PER_TD_PER_TILE] = ptr[i];

#pragma unroll
  for (int i = 0; i < kVectorSize; i++) ptr[i] = tmp[i];
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_kernel_impl(const void* input, void* output, const int M,
                                                const int K, const int original_M,
                                                const int original_K, const int bid_x,
                                                const int bid_y, const int grid_dim_x,
                                                const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  // input is in K-major
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M;

  int n_tiles_in_tb = N_TILES_IN_TB;
  const int K_i32 = K / 4;
  if (bid_x == grid_dim_x - 1) {
    n_tiles_in_tb = (K_i32 - 1) % N_TILES_IN_TB + 1;
  }

  bool padding_m = (bid_y == grid_dim_y - 1) && (original_M < M);
  bool padding_k = (bid_x == grid_dim_x - 1) && (original_K < K);

  const int input_offset = bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB;
  const int* input_i32 = reinterpret_cast<const int*>(input) + input_offset;
  int* output_i32 = reinterpret_cast<int*>(output) + bid_y * SF_TILE_DIM_M_I32 * K_i32 +
                    bid_x * N_TILES_IN_TB * SF_TILE_SIZE_I32;

  extern __shared__ int4 slm_v4i[];

  // load, global -> regs
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < n_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset = (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD;
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(input_i32 + thread_offset));
      if (padding_m || padding_k) {
        // Pad zeros
        for (int j = 0; j < N_TILE_PER_TD * sizeof(int); j++) {
          const int index = (input_offset + thread_offset) * sizeof(int) + j;
          if (index / K >= original_M || index % K >= original_K) {
            reinterpret_cast<uint8_t*>(regs_vec + i)[j] = 0;
          }
        }
      }
    }

    // shuffle regs
    regs_shuffle<LType>(regs_vec);

// store, regs -> shared
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      /* TODO rotate i */
      slm_v4i[(threadIdx.x * N_TILE_PER_TD + i) * SF_TILE_SIZE_I32 / 4 + threadIdx.y] =
          reinterpret_cast<int4*>(regs_vec)[i];
    }
  }
  __syncthreads();

  // store, shared -> global
  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  __align__(16) int4* output_v4i = reinterpret_cast<int4*>(output_i32);
#pragma unroll
  for (int i = linear_id; i < SF_TILE_SIZE_I32 * n_tiles_in_tb / 4; i += blockDim.x * blockDim.y) {
    output_v4i[i] = slm_v4i[i];
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_row_scaling_kernel(const void* input, void* output, const int M, const int K,
                               const int original_M, const int original_K) {
  swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
}

// Narrow-K specialization for row scaling swizzle.
// When K is small (num_tiles_k < TB_DIM), the standard kernel wastes threadIdx.x
// because there aren't enough K-tiles to distribute across threads.
// This kernel repurposes the thread dimensions: threadIdx.x iterates rows within
// an M-tile, threadIdx.y indexes M-tiles within the block, processing TB_DIM
// M-tiles per block with full thread utilization.
template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_row_scaling_narrow_k_kernel_impl(const void* input, void* output,
                                                         const int M, const int K,
                                                         const int original_M, const int original_K,
                                                         const int bid, const int grid_dim) {
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  const int K_i32 = K / 4;
  const int num_tiles_m = M / SF_TILE_DIM_M;

  const int m_tile = bid * blockDim.y + threadIdx.y;
  const bool active = (m_tile < num_tiles_m);

  extern __shared__ int4 slm_v4i[];
  const int slm_tile_v4i = K_i32 * (SF_TILE_SIZE_I32 / 4);

  if (active) {
    const bool padding_m = (m_tile == num_tiles_m - 1) && (original_M < M);
    const bool padding_k = (original_K < K);

    int4* my_slm = slm_v4i + threadIdx.y * slm_tile_v4i;

    for (int k = 0; k < K_i32; k++) {
      const int input_base = m_tile * SF_TILE_DIM_M * K_i32 + k;
      const int* input_i32 = reinterpret_cast<const int*>(input) + input_base;

      int regs[N_SF_PER_TD_PER_TILE];
#pragma unroll
      for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
        const int row = i * TB_DIM + threadIdx.x;
        regs[i] = __ldg(input_i32 + row * K_i32);
        if (padding_m || padding_k) {
          for (int j = 0; j < 4; j++) {
            const int byte_row = m_tile * SF_TILE_DIM_M + row;
            const int byte_col = k * 4 + j;
            if (byte_row >= original_M || byte_col >= original_K) {
              reinterpret_cast<uint8_t*>(&regs[i])[j] = 0;
            }
          }
        }
      }

      my_slm[k * (SF_TILE_SIZE_I32 / 4) + threadIdx.x] = *reinterpret_cast<int4*>(regs);
    }
  }

  __syncthreads();

  if (active) {
    int4* my_slm = slm_v4i + threadIdx.y * slm_tile_v4i;
    int4* out_v4i =
        reinterpret_cast<int4*>(reinterpret_cast<int*>(output) + m_tile * SF_TILE_DIM_M * K_i32);

    for (int i = threadIdx.x; i < slm_tile_v4i; i += blockDim.x) {
      out_v4i[i] = my_slm[i];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_row_scaling_narrow_k_kernel(const void* input, void* output, const int M, const int K,
                                        const int original_M, const int original_K) {
  swizzle_row_scaling_narrow_k_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, gridDim.x);
}

// Narrow-M variant of the column scaling swizzle kernel, for when num_tiles_m < TB_DIM.
// Analogous to the narrow-K row kernel: when the M dimension is small, the normal
// col kernel underutilizes threads in the load phase because threadIdx.x covers M
// positions with vectorized loads, leaving many threads idle. This kernel repurposes
// thread dimensions: threadIdx.y indexes K-tiles within the block, threadIdx.x covers
// one int32 column of an M-tile, and M-tiles are iterated serially.
template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void swizzle_col_scaling_narrow_m_kernel_impl(const void* input, void* output,
                                                         const int M, const int K,
                                                         const int original_M, const int original_K,
                                                         const int bid, const int grid_dim) {
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M / 4;
  constexpr int SF_TILE_DIM_K_I32 = SF_TILE_DIM_K;

  const int M_i32 = M / 4;
  const int K_i32 = K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int num_tiles_k = K / SF_TILE_DIM_K;

  const int k_tile = bid * blockDim.y + threadIdx.y;
  const bool active = (k_tile < num_tiles_k);
  const int remaining = num_tiles_k - bid * static_cast<int>(blockDim.y);
  const int k_tiles_in_block = remaining <= 0 ? 0 : (remaining < TB_DIM ? remaining : TB_DIM);

  extern __shared__ int slm_narrow_m[];

  if (active) {
    const bool padding_k = (k_tile == num_tiles_k - 1) && (original_K < K);
    const int32_t* input_i32 = reinterpret_cast<const int32_t*>(input);

    for (int m_tile = 0; m_tile < num_tiles_m; m_tile++) {
      const bool padding_m = (m_tile == num_tiles_m - 1) && (original_M < M);

      int regs[N_SF_PER_TD_PER_TILE];
#pragma unroll
      for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
        const int k_row = k_tile * SF_TILE_DIM_K_I32 + i;
        const int m_col = m_tile * SF_TILE_DIM_M_I32 + threadIdx.x;
        regs[i] = __ldg(input_i32 + k_row * M_i32 + m_col);
        if (padding_m || padding_k) {
          for (int j = 0; j < 4; j++) {
            if (m_col * 4 + j >= original_M || k_row >= original_K) {
              reinterpret_cast<uint8_t*>(&regs[i])[j] = 0;
            }
          }
        }
      }

      regs_shuffle_with_bit_shifts<int>(regs);

      int tM = threadIdx.x * N_SF_PER_TD_PER_TILE;
      int* slm_tile =
          slm_narrow_m + m_tile * TB_DIM * SF_TILE_SIZE_I32 + threadIdx.y * SF_TILE_SIZE_I32;
#pragma unroll
      for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
        slm_tile[(tM % SF_TILE_DIM_M) / NEW_SF_TILE_DIM_M_I32 +
                 ((tM + i) % NEW_SF_TILE_DIM_M_I32) * NEW_SF_TILE_DIM_K_I32] = regs[i];
      }
    }
  }

  __syncthreads();

  const int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  for (int m_tile = 0; m_tile < num_tiles_m; m_tile++) {
    int4* out_v4i = reinterpret_cast<int4*>(reinterpret_cast<int*>(output) +
                                            m_tile * SF_TILE_DIM_M_I32 * K_i32 +
                                            bid * TB_DIM * SF_TILE_SIZE_I32);
    int4* slm_v4i = reinterpret_cast<int4*>(slm_narrow_m + m_tile * TB_DIM * SF_TILE_SIZE_I32);
    const int n_v4i = k_tiles_in_block * SF_TILE_SIZE_I32 / 4;
    for (int j = linear_id; j < n_v4i; j += blockDim.x * blockDim.y) {
      out_v4i[j] = slm_v4i[j];
    }
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    swizzle_col_scaling_narrow_m_kernel(const void* input, void* output, const int M, const int K,
                                        const int original_M, const int original_K) {
  swizzle_col_scaling_narrow_m_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, blockIdx.x, gridDim.x);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void unswizzle_row_scaling_kernel_impl(const void* input, void* output, const int M,
                                                  const int K, const int bid_x, const int bid_y,
                                                  const int grid_dim_x, const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M;

  int n_tiles_in_tb = N_TILES_IN_TB;
  const int K_i32 = K / 4;
  if (bid_x == grid_dim_x - 1) {
    n_tiles_in_tb = (K_i32 - 1) % N_TILES_IN_TB + 1;
  }

  const int input_offset =
      bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB * SF_TILE_SIZE_I32;
  const int* input_i32 = reinterpret_cast<const int*>(input) + input_offset;
  const int output_offset = bid_y * SF_TILE_DIM_M_I32 * K_i32 + bid_x * N_TILES_IN_TB;
  int* output_i32 = reinterpret_cast<int*>(output) + output_offset;

  extern __shared__ int4 slm_v4i[];

  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
  const int4* input_v4i = reinterpret_cast<const int4*>(input_i32);
#pragma unroll
  for (int i = linear_id; i < SF_TILE_SIZE_I32 * n_tiles_in_tb / 4; i += blockDim.x * blockDim.y) {
    slm_v4i[i] = input_v4i[i];
  }
  __syncthreads();

  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < n_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_TILE_PER_TD; i++) {
      reinterpret_cast<int4*>(regs_vec)[i] =
          slm_v4i[(threadIdx.x * N_TILE_PER_TD + i) * SF_TILE_SIZE_I32 / 4 + threadIdx.y];
    }

    regs_unshuffle<LType>(regs_vec);

#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset = (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD;
      reinterpret_cast<LType*>(output_i32 + thread_offset)[0] = regs_vec[i];
    }
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__device__ void unswizzle_col_scaling_kernel_impl(const void* input, void* output, const int M,
                                                  const int K, const int bid_x, const int bid_y,
                                                  const int grid_dim_x, const int grid_dim_y) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_SF_PER_TD = N_TILE_PER_TD * N_SF_PER_TD_PER_TILE;
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;

  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M / 4;
  constexpr int SF_TILE_DIM_K_I32 = SF_TILE_DIM_K;

  const int M_i32 = M / 4;
  const int K_i32 = K;

  int m_tiles_in_tb = N_TILE_PER_TD;
  int k_tiles_in_tb = TB_DIM;
  if (bid_x == grid_dim_x - 1) {
    k_tiles_in_tb = (K_i32 / SF_TILE_DIM_K_I32 - 1) % k_tiles_in_tb + 1;
  }
  if (bid_y == grid_dim_y - 1) {
    m_tiles_in_tb = (M_i32 / SF_TILE_DIM_M_I32 - 1) % m_tiles_in_tb + 1;
  }

  const int32_t* input_i32[N_TILE_PER_TD];
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    input_i32[i] = reinterpret_cast<const int32_t*>(input) + bid_x * TB_DIM * SF_TILE_SIZE_I32 +
                   (bid_y * N_TILE_PER_TD + i) * SF_TILE_DIM_M_I32 * K_i32;
  }
  const int output_offset =
      bid_x * TB_DIM * SF_TILE_DIM_K_I32 * M_i32 + bid_y * N_TILE_PER_TD * SF_TILE_DIM_M_I32;
  int* output_i32 = reinterpret_cast<int*>(output) + output_offset;

  extern __shared__ int slm[];

  int linear_id = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    __align__(16) const int4* input_v4i = reinterpret_cast<const int4*>(input_i32[i]);
    __align__(16) int4* slm_v4i =
        reinterpret_cast<int4*>(slm + i * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int j = linear_id; j < SF_TILE_SIZE_I32 * k_tiles_in_tb / 4;
         j += blockDim.x * blockDim.y) {
      slm_v4i[j] = input_v4i[j];
    }
  }
  __syncthreads();

  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < m_tiles_in_tb * SF_TILE_DIM_M_I32 &&
      threadIdx.y < k_tiles_in_tb) {
    int tM = threadIdx.x * N_SF_PER_TD;
    int* slm_tile = slm + (threadIdx.y * SF_TILE_SIZE_I32 +
                           tM / SF_TILE_DIM_M * k_tiles_in_tb * SF_TILE_SIZE_I32);
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD; i++) {
      reinterpret_cast<int*>(regs_vec)[i] =
          slm_tile[(tM % SF_TILE_DIM_M) / NEW_SF_TILE_DIM_M_I32 +
                   ((tM + i) % NEW_SF_TILE_DIM_M_I32) * NEW_SF_TILE_DIM_K_I32];
    }

    regs_unshuffle_with_bit_shifts(regs_vec);

#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      const int thread_offset =
          (threadIdx.y * SF_TILE_DIM_K_I32 + i) * M_i32 + threadIdx.x * N_TILE_PER_TD;
      reinterpret_cast<LType*>(output_i32 + thread_offset)[0] = regs_vec[i];
    }
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    unswizzle_scaling_kernel(const void* input, void* output, const int M, const int K,
                             const bool row_scaling) {
  const int bid_x = blockIdx.x;
  const int bid_y = blockIdx.y;
  const int grid_dim_x = gridDim.x;
  const int grid_dim_y = gridDim.y;
  if (row_scaling) {
    unswizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  } else {
    unswizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
  }
}

constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB
struct MultiSwizzleArgs {
  // (input) Data buffers for input scaling factors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for swizzled scaling factors
  void* output_list[kMaxTensorsPerKernel];
  // Input scaling factor m
  int m_list[kMaxTensorsPerKernel];
  // Input scaling factor k
  int k_list[kMaxTensorsPerKernel];
  // Input scaling factor m before padding
  int original_m_list[kMaxTensorsPerKernel];
  // Input scaling factor k before padding
  int original_k_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

constexpr size_t round_up_to_multiple(size_t value, size_t multiple) {
  return DIVUP(value, multiple) * multiple;
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_row_scaling_uniform_shape_kernel(const void* input, void* output, const int M,
                                                     const int K, const int original_M,
                                                     const int original_K,
                                                     const size_t scale_stride_bytes) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * scale_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * scale_stride_bytes;
  swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x,
      gridDim.y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_col_scaling_uniform_shape_kernel(const void* input, void* output, const int M,
                                                     const int K, const int original_M,
                                                     const int original_K,
                                                     const size_t scale_stride_bytes) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * scale_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * scale_stride_bytes;
  swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input_base, output_base, M, K, original_M, original_K, blockIdx.x, blockIdx.y, gridDim.x,
      gridDim.y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_unswizzle_scaling_uniform_shape_kernel(const void* input, void* output, const int M,
                                                   const int K, const size_t scale_stride_bytes,
                                                   const bool row_scaling) {
  const int tensor_id = blockIdx.z;
  const uint8_t* input_base =
      reinterpret_cast<const uint8_t*>(input) + tensor_id * scale_stride_bytes;
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + tensor_id * scale_stride_bytes;
  if (row_scaling) {
    unswizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, M, K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
  } else {
    unswizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
        input_base, output_base, M, K, blockIdx.x, blockIdx.y, gridDim.x, gridDim.y);
  }
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_unswizzle_row_scaling_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int flat_offset = bid - kernel_args.block_range[tensor_id];
  const int grid_dim_x = DIVUP(num_tiles_k, N_TILES_IN_TB);
  const int grid_dim_y = num_tiles_m;
  const int bid_x = flat_offset / grid_dim_y;
  const int bid_y = flat_offset % grid_dim_y;

  unswizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_unswizzle_col_scaling_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);

  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int flat_offset = bid - kernel_args.block_range[tensor_id];
  const int grid_dim_x = DIVUP(num_tiles_k, TB_DIM);
  const int grid_dim_y = DIVUP(num_tiles_m, N_TILE_PER_TD);
  const int bid_x = flat_offset / grid_dim_y;
  const int bid_y = flat_offset % grid_dim_y;

  unswizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, bid_x, bid_y, grid_dim_x, grid_dim_y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_swizzle_row_scaling_kernel(MultiSwizzleArgs kernel_args) {
  // Find tensor corresponding to block
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  // Get args corresponding to block
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  // Get block index in grid. Emulate 2D grid.
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int grid_dim_x = DIVUP(num_tiles_k, N_TILES_IN_TB);
  const int grid_dim_y = num_tiles_m;
  const int bid_x = (bid - kernel_args.block_range[tensor_id]) / grid_dim_y;
  const int bid_y = (bid - kernel_args.block_range[tensor_id]) % grid_dim_y;

  swizzle_row_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
}

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void multi_tensor_swizzle_col_scaling_kernel(MultiSwizzleArgs kernel_args) {
  // Find tensor corresponding to block
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  // Get args corresponding to block
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];

  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);

  // Get block index in grid. Emulate 2D grid.
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int grid_dim_x = DIVUP(num_tiles_k, TB_DIM);
  const int grid_dim_y = DIVUP(num_tiles_m, N_TILE_PER_TD);
  const int bid_x = (bid - kernel_args.block_range[tensor_id]) / grid_dim_y;
  const int bid_y = (bid - kernel_args.block_range[tensor_id]) % grid_dim_y;

  swizzle_col_scaling_kernel_impl<LType, SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, bid_x, bid_y, grid_dim_x, grid_dim_y);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    multi_tensor_swizzle_row_scaling_narrow_k_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];
  const int flat_bid = bid - kernel_args.block_range[tensor_id];
  const int num_tiles_m = M / SF_TILE_DIM_M;
  const int grid_dim = DIVUP(num_tiles_m, TB_DIM);

  swizzle_row_scaling_narrow_k_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, flat_bid, grid_dim);
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    multi_tensor_swizzle_col_scaling_narrow_m_kernel(MultiSwizzleArgs kernel_args) {
  const int bid = blockIdx.x;
  int tensor_id = 0;
  while (kernel_args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const void* input = kernel_args.input_list[tensor_id];
  void* output = kernel_args.output_list[tensor_id];
  const int M = kernel_args.m_list[tensor_id];
  const int K = kernel_args.k_list[tensor_id];
  const int original_M = kernel_args.original_m_list[tensor_id];
  const int original_K = kernel_args.original_k_list[tensor_id];
  const int flat_bid = bid - kernel_args.block_range[tensor_id];
  const int num_tiles_k = K / SF_TILE_DIM_K;
  const int grid_dim = DIVUP(num_tiles_k, TB_DIM);

  swizzle_col_scaling_narrow_m_kernel_impl<SF_TILE_DIM_M, SF_TILE_DIM_K>(
      input, output, M, K, original_M, original_K, flat_bid, grid_dim);
}

}  // namespace

void swizzle_scaling_factors(const Tensor* input, Tensor* output, cudaStream_t stream) {
  // Check scaling mode
  const auto& scaling_mode = input->scaling_mode;
  NVTE_CHECK(scaling_mode == NVTE_MXFP8_1D_SCALING || scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Input tensor has invalid scaling mode (", to_string(input->scaling_mode), ").");

  // Check tensors
  CheckInputTensor(*input, "scaling_factor_input");
  CheckInputTensor(*output, "scaling_factor_output");
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Expected input tensor with scales in compact format.");
  NVTE_CHECK(output->with_gemm_swizzled_scales,
             "Expected output tensor with scales in GEMM swizzled format.");
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING:
      NVTE_CHECK(is_fp8_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP8, got ",
                 to_string(input->dtype()), ").");
      break;
    case NVTE_NVFP4_1D_SCALING:
      NVTE_CHECK(is_fp4_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP4, got ",
                 to_string(input->dtype()), ").");
      break;
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  // Check if scaling factors are non-trivial
  const bool has_rowwise_scale_inv = input->scale_inv.has_data();
  const bool has_columnwise_scale_inv = input->columnwise_scale_inv.has_data();
  NVTE_CHECK(!has_rowwise_scale_inv || !has_columnwise_scale_inv,
             "Input tensor has both row-wise and column-wise scaling factors");
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }

  // Deduce tensor dims
  int m{0}, k{0};
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(input->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->scale_inv.shape, ".");
        m = input->scale_inv.shape[0];
        k = input->scale_inv.shape[1];
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(input->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->columnwise_scale_inv.shape,
                   ".");
        m = input->columnwise_scale_inv.shape[1];
        k = input->columnwise_scale_inv.shape[0];
      }
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(input->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->scale_inv.shape, ".");
        m = input->scale_inv.shape[0];
        k = input->scale_inv.shape[1];
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(input->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", input->columnwise_scale_inv.shape,
                   ".");
        m = input->columnwise_scale_inv.shape[0];
        k = input->columnwise_scale_inv.shape[1];
      }
      break;
    }
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  // Check dims
  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
  NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");

  // Check that output tensor matches input tensor
  if (has_rowwise_scale_inv) {
    NVTE_CHECK(output->scale_inv.has_data(),
               "Output tensor does not have row-wise scaling factors.");
    NVTE_CHECK(m * k == output->scale_inv.numel(), "Expected output tensor to have ", m * k,
               " row-wise scaling factors, but got shape=", output->scale_inv.shape, ".");
  }
  if (has_columnwise_scale_inv) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(),
               "Output tensor does not have column-wise scaling factors.");
    NVTE_CHECK(
        m * k == output->columnwise_scale_inv.numel(), "Expected output tensor to have ", m * k,
        " column-wise scaling factors, but got shape=", output->columnwise_scale_inv.shape, ".");
  }

  // Choose swizzle implementation
  bool rowwise_swizzle{false}, columnwise_swizzle{false};
  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      rowwise_swizzle = has_rowwise_scale_inv;
      columnwise_swizzle = has_columnwise_scale_inv;
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      // NVFP4 column-wise data is transposed, so row-wise and
      // column-wise scales have same swizzling format
      rowwise_swizzle = true;
      columnwise_swizzle = false;
      break;
    }
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  const dim3 block_size(TB_DIM, TB_DIM);
  const int num_tiles_m = m / SF_TILE_DIM_M;
  const int num_tiles_k = k / SF_TILE_DIM_K;

  // Perform row-wise swizzle
  if (rowwise_swizzle) {
    int original_M{0}, original_K{0};
    void *input_scale_inv_ptr{nullptr}, *output_scale_inv_ptr{nullptr};
    switch (scaling_mode) {
      case NVTE_MXFP8_1D_SCALING: {
        original_M = input->flat_first_dim();
        original_K = input->flat_last_dim() / MXFP8_BLOCK_SIZE;
        input_scale_inv_ptr = input->scale_inv.dptr;
        output_scale_inv_ptr = output->scale_inv.dptr;
        break;
      }
      case NVTE_NVFP4_1D_SCALING: {
        if (has_rowwise_scale_inv) {
          original_M = input->flat_first_dim();
          original_K = input->flat_last_dim() / NVFP4_BLOCK_SIZE;
          input_scale_inv_ptr = input->scale_inv.dptr;
          output_scale_inv_ptr = output->scale_inv.dptr;
        } else if (has_columnwise_scale_inv) {
          original_M = input->flat_last_dim();
          original_K = input->flat_first_dim() / NVFP4_BLOCK_SIZE;
          input_scale_inv_ptr = input->columnwise_scale_inv.dptr;
          output_scale_inv_ptr = output->columnwise_scale_inv.dptr;
        }
        break;
      }
      default:
        NVTE_ERROR("Invalid scaling mode");
    }

    const int narrow_k_slm_size =
        TB_DIM * num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
    if (num_tiles_k < TB_DIM && narrow_k_slm_size <= get_max_dynamic_smem()) {
      // Narrow-K: batch TB_DIM M-tiles per block, fully utilizing all threads.
      dim3 num_blocks_narrow(DIVUP(num_tiles_m, TB_DIM));
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, narrow_k_slm_size));
      swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks_narrow, block_size, narrow_k_slm_size, stream>>>(
              input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
    } else {
      int vec_load_size = (num_tiles_k - 1) % 4 + 1;
      /* there is no int3 and misaligned if using int4/int2 */
      if (vec_load_size == 3) vec_load_size = 1;
      int n_tiles_in_tb = TB_DIM * vec_load_size;
      dim3 num_blocks(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m);
      int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
          break;
        case 2:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
          break;
        case 1:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input_scale_inv_ptr, output_scale_inv_ptr, m, k, original_M, original_K);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  // Perform column-wise swizzle
  if (columnwise_swizzle) {
    const int original_M = input->flat_last_dim();
    const int original_K = input->flat_first_dim() / MXFP8_BLOCK_SIZE;

    const int narrow_m_slm_size =
        TB_DIM * num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
    if (num_tiles_m < TB_DIM && narrow_m_slm_size <= get_max_dynamic_smem()) {
      // Narrow-M: batch TB_DIM K-tiles per block, fully utilizing all threads.
      dim3 num_blocks_narrow(DIVUP(num_tiles_k, TB_DIM));
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, narrow_m_slm_size));
      swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks_narrow, block_size, narrow_m_slm_size, stream>>>(
              input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m, k, original_M,
              original_K);
    } else {
      int vec_load_size = (num_tiles_m - 1) % 4 + 1;
      if (vec_load_size == 3) vec_load_size = 1; /* no int3 and misaligned if using int4/int2 */
      int n_tiles_in_tb = TB_DIM * vec_load_size;
      dim3 num_blocks(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size));
      int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->columnwise_scale_inv.dptr,
                                                             output->columnwise_scale_inv.dptr, m,
                                                             k, original_M, original_K);
          break;
        case 2:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->columnwise_scale_inv.dptr,
                                                             output->columnwise_scale_inv.dptr, m,
                                                             k, original_M, original_K);
          break;
        case 1:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->columnwise_scale_inv.dptr,
                                                             output->columnwise_scale_inv.dptr, m,
                                                             k, original_M, original_K);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
void launch_multi_tensor_swizzle_scaling_factors(MultiSwizzleArgs& kernel_args,
                                                 const int vec_load_size, const bool is_rowwise,
                                                 const bool use_narrow_k, const bool use_narrow_m,
                                                 cudaStream_t stream) {
  // cudaFuncSetAttribute is a host-synchronous driver call; cache the max shared memory
  // setting per kernel variant so we only pay the cost when slm_size actually increases.
  auto set_smem_if_needed = [](auto kernel_fn, int slm, int& cached) {
    if (cached < slm) {
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, slm));
      cached = slm;
    }
  };

  dim3 block_size(TB_DIM, TB_DIM);

  if (is_rowwise && use_narrow_k) {
    // Narrow-K path: each block handles TB_DIM M-tiles with full thread utilization.
    // slm_size depends on num_tiles_k, which can vary per tensor — use the max.
    int max_num_tiles_k = 0;
    for (size_t j = 0; j < kernel_args.num_tensors; j++) {
      const int num_tiles_m = kernel_args.m_list[j] / SF_TILE_DIM_M;
      const int num_tiles_k = kernel_args.k_list[j] / SF_TILE_DIM_K;
      max_num_tiles_k = std::max(max_num_tiles_k, num_tiles_k);
      kernel_args.block_range[j + 1] = kernel_args.block_range[j] + DIVUP(num_tiles_m, TB_DIM);
    }
    int slm_size = TB_DIM * max_num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
    const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];

    static int cached_narrow_k = -1;
    set_smem_if_needed(
        multi_tensor_swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
        cached_narrow_k);
    multi_tensor_swizzle_row_scaling_narrow_k_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
        <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
  } else if (!is_rowwise && use_narrow_m) {
    // Narrow-M path: each block handles TB_DIM K-tiles with full thread utilization.
    // slm_size depends on num_tiles_m, which can vary per tensor — use the max.
    int max_num_tiles_m = 0;
    for (size_t j = 0; j < kernel_args.num_tensors; j++) {
      const int num_tiles_m = kernel_args.m_list[j] / SF_TILE_DIM_M;
      const int num_tiles_k = kernel_args.k_list[j] / SF_TILE_DIM_K;
      max_num_tiles_m = std::max(max_num_tiles_m, num_tiles_m);
      kernel_args.block_range[j + 1] = kernel_args.block_range[j] + DIVUP(num_tiles_k, TB_DIM);
    }
    int slm_size = TB_DIM * max_num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
    const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];

    static int cached_narrow_m = -1;
    set_smem_if_needed(
        multi_tensor_swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
        cached_narrow_m);
    multi_tensor_swizzle_col_scaling_narrow_m_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
        <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
  } else {
    int n_tiles_in_tb = TB_DIM * vec_load_size;
    int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
    /* Calculate number of CUDA blocks needed for each tensor.
    * We have to do it here because we have to iterate over all tensors in this batch to
    * get the minimum vec_load_size.
    */
    for (size_t j = 0; j < kernel_args.num_tensors; j++) {
      const int m = kernel_args.m_list[j];
      const int k = kernel_args.k_list[j];
      int num_tiles_m = m / SF_TILE_DIM_M;
      int num_tiles_k = k / SF_TILE_DIM_K;
      if (is_rowwise) {
        kernel_args.block_range[j + 1] =
            kernel_args.block_range[j] + DIVUP(num_tiles_k, n_tiles_in_tb) * num_tiles_m;
      } else {
        kernel_args.block_range[j + 1] =
            kernel_args.block_range[j] +
            DIVUP(num_tiles_k, TB_DIM) * DIVUP(num_tiles_m, vec_load_size);
      }
    }
    const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];

    static int cached_row_int4 = -1, cached_row_int2 = -1, cached_row_int1 = -1;
    static int cached_col_int4 = -1, cached_col_int2 = -1, cached_col_int1 = -1;

    if (is_rowwise) {
      switch (vec_load_size) {
        case 4:
          set_smem_if_needed(
              multi_tensor_swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_row_int4);
          multi_tensor_swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          set_smem_if_needed(
              multi_tensor_swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_row_int2);
          multi_tensor_swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          set_smem_if_needed(
              multi_tensor_swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_row_int1);
          multi_tensor_swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    } else {
      switch (vec_load_size) {
        case 4:
          set_smem_if_needed(
              multi_tensor_swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_col_int4);
          multi_tensor_swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          set_smem_if_needed(
              multi_tensor_swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_col_int2);
          multi_tensor_swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          set_smem_if_needed(
              multi_tensor_swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>, slm_size,
              cached_col_int1);
          multi_tensor_swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
void launch_multi_tensor_unswizzle_scaling_factors(MultiSwizzleArgs& kernel_args,
                                                   const int vec_load_size, const bool is_rowwise,
                                                   cudaStream_t stream) {
  int n_tiles_in_tb = TB_DIM * vec_load_size;
  int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
  for (size_t j = 0; j < kernel_args.num_tensors; j++) {
    const int m = kernel_args.m_list[j];
    const int k = kernel_args.k_list[j];
    int num_tiles_m = m / SF_TILE_DIM_M;
    int num_tiles_k = k / SF_TILE_DIM_K;
    if (is_rowwise) {
      kernel_args.block_range[j + 1] =
          kernel_args.block_range[j] + DIVUP(num_tiles_k, n_tiles_in_tb) * num_tiles_m;
    } else {
      kernel_args.block_range[j + 1] =
          kernel_args.block_range[j] +
          DIVUP(num_tiles_k, TB_DIM) * DIVUP(num_tiles_m, vec_load_size);
    }
  }

  int num_blocks = kernel_args.block_range[kernel_args.num_tensors];
  if (num_blocks > 0) {
    dim3 block_size(TB_DIM, TB_DIM);
    if (is_rowwise) {
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    } else {
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 2:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        case 1:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              multi_tensor_unswizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          multi_tensor_unswizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void multi_tensor_swizzle_scaling_factors(const std::vector<Tensor*>& input,
                                          std::vector<Tensor*>& output, cudaStream_t stream,
                                          bool check_scale_inv_shapes) {
  auto num_tensors = input.size();
  bool all_has_data = true;
  bool all_has_columnwise_data = true;
  bool all_nvfp4 = true;
  for (size_t i = 0; i < num_tensors; i++) {
    auto scaling_mode = input[i]->scaling_mode;
    auto is_fp8 = is_fp8_dtype(input[i]->dtype());
    auto is_fp4 = is_fp4_dtype(input[i]->dtype());
    NVTE_CHECK(
        (is_fp8 && is_mxfp8_scaling(scaling_mode)) || (is_fp4 && is_nvfp4_scaling(scaling_mode)),
        "Not implemented scaling mode " + to_string(scaling_mode) + ".");
    NVTE_CHECK(!input[i]->with_gemm_swizzled_scales,
               "Expected input tensors with scales in compact format.");
    NVTE_CHECK(output[i]->with_gemm_swizzled_scales,
               "Expected output tensors with scales in GEMM swizzled format.");

    // We don't allow empty tensors. They should be filtered out before calling this function.
    NVTE_CHECK(input[i]->numel() != 0, "Tensor input[", i, "] is empty.");
    CheckInputTensor(*input[i], "scaling_factor_input[" + std::to_string(i) + "]",
                     check_scale_inv_shapes);
    CheckInputTensor(*output[i], "scaling_factor_output[" + std::to_string(i) + "]",
                     check_scale_inv_shapes);
    all_has_data = all_has_data && input[i]->scale_inv.has_data();
    all_has_columnwise_data =
        (all_has_columnwise_data && input[i]->columnwise_scale_inv.has_data());
    all_nvfp4 = all_nvfp4 && is_nvfp4_scaling(scaling_mode);
  }
  NVTE_CHECK(all_has_data || all_has_columnwise_data,
             "All tensors should have data or columnwise data.");
  NVTE_CHECK(!all_has_data || !all_has_columnwise_data,
             "All tensors have both data and columnwise data.");

  const bool rowwise_swizzle = all_has_data || all_nvfp4;
  const bool columnwise_swizzle = all_has_columnwise_data && !all_nvfp4;

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  if (rowwise_swizzle) {
    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    bool all_narrow_k = true;
    for (size_t i = 0; i < num_tensors; i++) {
      //Launch kernel if argument struct is full
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        // There is no int3 and misaligned if using int4/int2.
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, true, all_narrow_k, false, stream);
        // Reset the argument struct and vec_load_size
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
        all_narrow_k = true;
      }

      int m, k;

      if (all_has_data) {
        m = input[i]->scale_inv.shape[0];
        k = input[i]->scale_inv.shape[1];
      } else {
        NVTE_CHECK(all_nvfp4, "When doing rowwise swizzle with rowwise data, it has to be NVFP4");
        m = input[i]->columnwise_scale_inv.shape[0];
        k = input[i]->columnwise_scale_inv.shape[1];
      }

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Input scale inverse should be 2D!");

      if (all_has_data) {
        NVTE_CHECK(output[i]->scale_inv.has_data(), "Output tensor ", i,
                   " does not have row-wise scaling factors.");
        NVTE_CHECK(m * k == output[i]->scale_inv.numel(), "Expected output tensor ", i, " to have ",
                   m * k, " row-wise scaling factors, but got shape=", output[i]->scale_inv.shape,
                   ".");
      }
      if (all_has_columnwise_data) {
        NVTE_CHECK(output[i]->columnwise_scale_inv.has_data(), "Output tensor ", i,
                   " does not have column-wise scaling factors.");
        NVTE_CHECK(m * k == output[i]->columnwise_scale_inv.numel(), "Expected output tensor ", i,
                   " to have ", m * k, " column-wise scaling factors, but got shape=",
                   output[i]->columnwise_scale_inv.shape, ".");
      }

      int num_tiles_k = k / SF_TILE_DIM_K;
      const int narrow_k_slm =
          TB_DIM * num_tiles_k * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
      all_narrow_k =
          all_narrow_k && (num_tiles_k < TB_DIM) && (narrow_k_slm <= get_max_dynamic_smem());
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      // We use the minimum vec_load_size across all tensors.
      // TODO(zhongbo): fix vec_load_size for NVFP4
      // Current unit test won't capture this issue, but in E2E
      // using vec_load_size = 1 other than 1 will lead to mis-aligned
      // address error in MOE training
      vec_load_size = all_nvfp4 ? 1 : std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      if (!all_nvfp4 || all_has_data) {
        int block_scale_size = all_nvfp4 ? NVFP4_BLOCK_SIZE : MXFP8_BLOCK_SIZE;
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->scale_inv.dptr;
        kernel_args.original_m_list[pos] = input[i]->flat_first_dim();
        kernel_args.original_k_list[pos] = input[i]->flat_last_dim() / block_scale_size;
      } else {
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
        kernel_args.original_m_list[pos] = input[i]->flat_last_dim();
        kernel_args.original_k_list[pos] = input[i]->flat_first_dim() / NVFP4_BLOCK_SIZE;
      }
      kernel_args.num_tensors++;
    }
    // Launch the remaining tensors
    // There is no int3 and misaligned if using int4/int2.
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, true, all_narrow_k, false, stream);
  }

  if (columnwise_swizzle) {
    // NVFP4 shouldn't end up here because it only needs rowwise swizzle
    NVTE_CHECK(!all_nvfp4, "NVFP4 shouldn't end up here because it only needs rowwise swizzle");

    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    bool all_narrow_m = true;
    for (size_t i = 0; i < num_tensors; i++) {
      //Launch kernel if argument struct is full
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        // There is no int3 and misaligned if using int4/int2.
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, false, false, all_narrow_m, stream);
        // Reset the argument struct and vec_load_size
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
        all_narrow_m = true;
      }
      const int m = input[i]->columnwise_scale_inv.shape[1];
      const int k = input[i]->columnwise_scale_inv.shape[0];

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Input scale inverse should be 2D!");
      NVTE_CHECK(m * k == std::accumulate(output[i]->columnwise_scale_inv.shape.begin(),
                                          output[i]->columnwise_scale_inv.shape.end(), 1,
                                          std::multiplies<int>()),
                 "Input.columnwise_scale_inv size is not equal to "
                 "Output.columnwise_scale_inv size!");

      int num_tiles_m = m / SF_TILE_DIM_M;
      int num_tiles_k = k / SF_TILE_DIM_K;
      const int narrow_m_slm =
          TB_DIM * num_tiles_m * SF_TILE_DIM_M * SF_TILE_DIM_K * static_cast<int>(sizeof(int8_t));
      all_narrow_m =
          all_narrow_m && (num_tiles_m < TB_DIM) && (narrow_m_slm <= get_max_dynamic_smem());
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      // We use the minimum vec_load_size across all tensors.
      vec_load_size = std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
      kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      kernel_args.original_m_list[pos] = input[i]->flat_last_dim();
      kernel_args.original_k_list[pos] = input[i]->flat_first_dim() / MXFP8_BLOCK_SIZE;
      kernel_args.num_tensors++;
    }
    // Launch the remaining tensors
    // There is no int3 and misaligned if using int4/int2.
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, false, false, all_narrow_m, stream);
  }
}

void unswizzle_scaling_factors(const Tensor* input, Tensor* output, cudaStream_t stream) {
  const auto& scaling_mode = output->scaling_mode;
  NVTE_CHECK(scaling_mode == NVTE_MXFP8_1D_SCALING || scaling_mode == NVTE_NVFP4_1D_SCALING,
             "Output tensor has invalid scaling mode (", to_string(output->scaling_mode), ").");

  CheckInputTensor(*input, "scaling_factor_input");
  CheckInputTensor(*output, "scaling_factor_output");
  NVTE_CHECK(input->with_gemm_swizzled_scales, "Expected input tensor with swizzled scales.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Expected output tensor in row-major compact format.");
  NVTE_CHECK(input->scaling_mode == scaling_mode,
             "Input and output tensors must have matching scaling modes, but got ",
             to_string(input->scaling_mode), " and ", to_string(output->scaling_mode), ".");

  const bool has_rowwise_scale_inv = output->scale_inv.has_data();
  const bool has_columnwise_scale_inv = output->columnwise_scale_inv.has_data();
  NVTE_CHECK(!has_rowwise_scale_inv || !has_columnwise_scale_inv,
             "Output tensor has both row-wise and column-wise scaling factors");
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }
  if (has_rowwise_scale_inv) {
    NVTE_CHECK(input->scale_inv.has_data(),
               "Output tensor requests row-wise scaling factors, but input tensor does not "
               "provide them.");
  } else if (has_columnwise_scale_inv) {
    NVTE_CHECK(input->columnwise_scale_inv.has_data(),
               "Output tensor requests column-wise scaling factors, but input tensor does not "
               "provide them.");
  }

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  const dim3 block_size(TB_DIM, TB_DIM);

  int m{0}, k{0};
  void* input_ptr{nullptr};
  void* output_ptr{nullptr};
  bool rowwise{false};

  switch (scaling_mode) {
    case NVTE_MXFP8_1D_SCALING: {
      NVTE_CHECK(is_fp8_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP8, got ",
                 to_string(input->dtype()), ").");
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(output->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->scale_inv.shape, ".");
        m = output->scale_inv.shape[0];
        k = output->scale_inv.shape[1];
        NVTE_CHECK(static_cast<size_t>(m) * k == input->scale_inv.numel(),
                   "Expected input tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", input->scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", output->scale_inv.shape, ".");
        input_ptr = input->scale_inv.dptr;
        output_ptr = output->scale_inv.dptr;
        rowwise = true;
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(output->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->columnwise_scale_inv.shape,
                   ".");
        m = output->columnwise_scale_inv.shape[1];
        k = output->columnwise_scale_inv.shape[0];
        NVTE_CHECK(
            static_cast<size_t>(m) * k == input->columnwise_scale_inv.numel(),
            "Expected input tensor to have ", static_cast<size_t>(m) * k,
            " column-wise scaling factors, but got shape=", input->columnwise_scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->columnwise_scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " column-wise scaling factors, but got shape=",
                   output->columnwise_scale_inv.shape, ".");
        input_ptr = input->columnwise_scale_inv.dptr;
        output_ptr = output->columnwise_scale_inv.dptr;
        rowwise = false;
      }
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      NVTE_CHECK(is_fp4_dtype(input->dtype()), "Input tensor has invalid dtype (expected FP4, got ",
                 to_string(input->dtype()), ").");
      // NVFP4: always unswizzle rowwise regardless of which scale buffer holds the data
      if (has_rowwise_scale_inv) {
        NVTE_CHECK(output->scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->scale_inv.shape, ".");
        m = output->scale_inv.shape[0];
        k = output->scale_inv.shape[1];
        // Example for NVFP4 rowwise path:
        NVTE_CHECK(static_cast<size_t>(m) * k == input->scale_inv.numel(),
                   "Expected input tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", input->scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " row-wise scaling factors, but got shape=", output->scale_inv.shape, ".");
        input_ptr = input->scale_inv.dptr;
        output_ptr = output->scale_inv.dptr;
      } else if (has_columnwise_scale_inv) {
        NVTE_CHECK(output->columnwise_scale_inv.shape.size() == 2,
                   "Expected 2D scaling factors, got shape=", output->columnwise_scale_inv.shape,
                   ".");
        m = output->columnwise_scale_inv.shape[0];
        k = output->columnwise_scale_inv.shape[1];
        NVTE_CHECK(
            static_cast<size_t>(m) * k == input->columnwise_scale_inv.numel(),
            "Expected input tensor to have ", static_cast<size_t>(m) * k,
            " column-wise scaling factors, but got shape=", input->columnwise_scale_inv.shape, ".");
        NVTE_CHECK(static_cast<size_t>(m) * k == output->columnwise_scale_inv.numel(),
                   "Expected output tensor to have ", static_cast<size_t>(m) * k,
                   " column-wise scaling factors, but got shape=",
                   output->columnwise_scale_inv.shape, ".");
        input_ptr = input->columnwise_scale_inv.dptr;
        output_ptr = output->columnwise_scale_inv.dptr;
      }
      rowwise = true;
      break;
    }
    default:
      NVTE_ERROR("Invalid scaling mode");
  }

  NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Output should be padded in M/N dimension!");
  NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Output should be padded in K dimension!");

  const int num_tiles_m = m / SF_TILE_DIM_M;
  const int num_tiles_k = k / SF_TILE_DIM_K;

  auto launch_unswizzle = [&](int vec_load_size, const dim3& num_blocks, int slm_size) {
    switch (vec_load_size) {
      case 4:
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(unswizzle_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        unswizzle_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, m, k, rowwise);
        break;
      case 2:
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(unswizzle_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        unswizzle_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, m, k, rowwise);
        break;
      case 1:
        NVTE_CHECK_CUDA(
            cudaFuncSetAttribute(unswizzle_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        unswizzle_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, m, k, rowwise);
        break;
      default:
        NVTE_ERROR("Not valid vec_load_size.");
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  int vec_load_size = rowwise ? (num_tiles_k - 1) % 4 + 1 : (num_tiles_m - 1) % 4 + 1;
  if (vec_load_size == 3) vec_load_size = 1;
  int n_tiles_in_tb = TB_DIM * vec_load_size;
  dim3 num_blocks = rowwise ? dim3(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m)
                            : dim3(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size));
  int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
  launch_unswizzle(vec_load_size, num_blocks, slm_size);
}

void multi_tensor_unswizzle_scaling_factors(const std::vector<Tensor*>& input,
                                            std::vector<Tensor*>& output, cudaStream_t stream) {
  size_t num_tensors = output.size();
  const auto& first_scaling_mode = output[0]->scaling_mode;

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;

  bool all_has_data = true;
  bool all_has_columnwise_data = true;
  bool all_nvfp4 = true;
  for (size_t i = 0; i < num_tensors; i++) {
    const auto scaling_mode = output[i]->scaling_mode;
    const auto is_fp8 = is_fp8_dtype(input[i]->dtype());
    const auto is_fp4 = is_fp4_dtype(input[i]->dtype());

    NVTE_CHECK(scaling_mode == first_scaling_mode,
               "All tensors should have the same scaling mode in multi-tensor unswizzle.");
    NVTE_CHECK(
        (is_fp8 && is_mxfp8_scaling(scaling_mode)) || (is_fp4 && is_nvfp4_scaling(scaling_mode)),
        "Not implemented scaling mode " + to_string(scaling_mode) + ".");
    NVTE_CHECK(input[i]->with_gemm_swizzled_scales,
               "Expected input tensors with scales in GEMM swizzled format.");
    NVTE_CHECK(!output[i]->with_gemm_swizzled_scales,
               "Expected output tensors with scales in compact format.");
    NVTE_CHECK(input[i]->numel() != 0, "Tensor input[", i, "] is empty.");
    CheckInputTensor(*input[i], "scaling_factor_input[" + std::to_string(i) + "]");
    CheckInputTensor(*output[i], "scaling_factor_output[" + std::to_string(i) + "]");

    all_has_data = all_has_data && output[i]->scale_inv.has_data();
    all_has_columnwise_data =
        (all_has_columnwise_data && output[i]->columnwise_scale_inv.has_data());
    all_nvfp4 = all_nvfp4 && is_nvfp4_scaling(scaling_mode);
  }
  NVTE_CHECK(all_has_data || all_has_columnwise_data,
             "All tensors should have data or columnwise data.");
  NVTE_CHECK(!all_has_data || !all_has_columnwise_data,
             "All tensors have both data and columnwise data.");

  const bool rowwise_unswizzle = all_has_data || all_nvfp4;
  const bool columnwise_unswizzle = all_has_columnwise_data && !all_nvfp4;

  if (rowwise_unswizzle) {
    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    for (size_t i = 0; i < num_tensors; i++) {
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, true, stream);
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
      }
      int m, k;
      if (all_has_data) {
        NVTE_CHECK(input[i]->scale_inv.has_data(), "Input tensor ", i,
                   " does not have row-wise scaling factors.");
        NVTE_CHECK(output[i]->scale_inv.shape.size() == 2, "Expected output tensor ", i,
                   " to have ", "2D scaling factors, got shape=", output[i]->scale_inv.shape, ".");
        m = output[i]->scale_inv.shape[0];
        k = output[i]->scale_inv.shape[1];
        NVTE_CHECK(m * k == input[i]->scale_inv.numel(), "Expected input tensor ", i, " to have ",
                   m * k, " row-wise scaling factors, but got shape=", input[i]->scale_inv.shape,
                   ".");
      }

      if (all_has_columnwise_data) {
        NVTE_CHECK(all_nvfp4,
                   "When doing rowwise unswizzle with columnwise data, it has to be NVFP4");
        NVTE_CHECK(input[i]->columnwise_scale_inv.has_data(), "Input tensor ", i,
                   " does not have column-wise scaling factors.");
        NVTE_CHECK(output[i]->columnwise_scale_inv.shape.size() == 2, "Expected output tensor ", i,
                   " to have ",
                   "2D scaling factors, got shape=", output[i]->columnwise_scale_inv.shape, ".");
        m = output[i]->columnwise_scale_inv.shape[0];
        k = output[i]->columnwise_scale_inv.shape[1];
        NVTE_CHECK(m * k == input[i]->columnwise_scale_inv.numel(), "Expected input tensor ", i,
                   " to have ", m * k, " column-wise scaling factors, but got shape=",
                   input[i]->columnwise_scale_inv.shape, ".");
      }

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Output should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Output should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Output scale inverse should be 2D!");

      int num_tiles_k = k / SF_TILE_DIM_K;
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      vec_load_size = all_nvfp4 ? 1 : std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      if (!all_nvfp4 || all_has_data) {
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->scale_inv.dptr;
      } else {
        kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
        kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
      }
      kernel_args.num_tensors++;
    }
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, true, stream);
  }

  if (columnwise_unswizzle) {
    NVTE_CHECK(!all_nvfp4, "NVFP4 shouldn't end up here because it only needs rowwise unswizzle");

    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    for (size_t i = 0; i < num_tensors; i++) {
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, false, stream);
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
      }
      NVTE_CHECK(output[i]->columnwise_scale_inv.shape.size() == 2, "Expected output tensor ", i,
                 " to have ",
                 "2D scaling factors, got shape=", output[i]->columnwise_scale_inv.shape, ".");
      const int m = output[i]->columnwise_scale_inv.shape[1];
      const int k = output[i]->columnwise_scale_inv.shape[0];

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Output should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Output should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Output scale inverse should be 2D!");
      NVTE_CHECK(m * k == std::accumulate(input[i]->columnwise_scale_inv.shape.begin(),
                                          input[i]->columnwise_scale_inv.shape.end(), 1,
                                          std::multiplies<int>()),
                 "Input.columnwise_scale_inv size is not equal to "
                 "Output.columnwise_scale_inv size!");

      int num_tiles_k = k / SF_TILE_DIM_K;
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      vec_load_size = std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.input_list[pos] = const_cast<void*>(input[i]->columnwise_scale_inv.dptr);
      kernel_args.output_list[pos] = output[i]->columnwise_scale_inv.dptr;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      kernel_args.num_tensors++;
    }
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_unswizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, false, stream);
  }
}
}  // namespace transformer_engine

/*
* WIP (Phuong):
*   - Opt for bank conflicts
*   - Adding swizzle for 2d-block scaling.
*/
void nvte_swizzle_scaling_factors(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_scaling_factors);
  using namespace transformer_engine;
  swizzle_scaling_factors(convertNVTETensorCheck(input), convertNVTETensorCheck(output), stream);
}

void nvte_multi_tensor_swizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                               const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_swizzle_scaling_factors);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");
  std::vector<Tensor*> input_list, output_list;
  for (size_t i = 0; i < num_tensors; i++) {
    input_list.push_back(convertNVTETensorCheck(inputs[i]));
    output_list.push_back(convertNVTETensorCheck(outputs[i]));
  }
  multi_tensor_swizzle_scaling_factors(input_list, output_list, stream,
                                       /*check_scale_inv_shapes=*/true);
}

void nvte_multi_tensor_swizzle_scaling_factors_unchecked(const NVTETensor* inputs,
                                                         NVTETensor* outputs,
                                                         const size_t num_tensors,
                                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_swizzle_scaling_factors_unchecked);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");
  std::vector<Tensor*> input_list, output_list;
  for (size_t i = 0; i < num_tensors; i++) {
    input_list.push_back(convertNVTETensorCheck(inputs[i]));
    output_list.push_back(convertNVTETensorCheck(outputs[i]));
  }
  multi_tensor_swizzle_scaling_factors(input_list, output_list, stream,
                                       /*check_scale_inv_shapes=*/false);
}

void nvte_unswizzle_scaling_factors(const NVTETensor input, NVTETensor output,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_unswizzle_scaling_factors);
  using namespace transformer_engine;
  unswizzle_scaling_factors(convertNVTETensorCheck(input), convertNVTETensorCheck(output), stream);
}

void nvte_multi_tensor_unswizzle_scaling_factors(const NVTETensor* inputs, NVTETensor* outputs,
                                                 const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_unswizzle_scaling_factors);
  using namespace transformer_engine;
  NVTE_CHECK(num_tensors > 0, "Number of tensors should be greater than 0.");
  std::vector<Tensor*> input_list, output_list;
  for (size_t i = 0; i < num_tensors; i++) {
    input_list.push_back(convertNVTETensorCheck(inputs[i]));
    output_list.push_back(convertNVTETensorCheck(outputs[i]));
  }
  multi_tensor_unswizzle_scaling_factors(input_list, output_list, stream);
}

namespace transformer_engine {

void swizzle_grouped_scaling_factors(const GroupedTensor* input, GroupedTensor* output,
                                     cudaStream_t stream) {
  // Check scaling mode
  NVTE_CHECK(input->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Grouped swizzle supports only MXFP8 scaling.");

  // Check tensors
  CheckInputGroupedTensor(*input, "input");
  CheckOutputGroupedTensor(*output, "output", false);
  NVTE_CHECK(!input->with_gemm_swizzled_scales,
             "Expected input grouped tensor with scales in compact format.");
  NVTE_CHECK(output->with_gemm_swizzled_scales,
             "Expected output grouped tensor with scales in GEMM swizzled format.");

  // Check scaling factors availability
  const bool has_rowwise_scale_inv = input->scale_inv.has_data();
  const bool has_columnwise_scale_inv = input->columnwise_scale_inv.has_data();
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }

  // Only support uniform shapes for graph-safe grouped swizzle
  NVTE_CHECK(input->all_same_shape(), "Grouped swizzle requires uniform tensor shapes.");
  NVTE_CHECK(input->all_same_last_dim() && input->all_same_first_dim(),
             "Grouped swizzle requires uniform tensor shapes.");

  // Assumption is that all the tensors share the same shapes and are contgiuous.
  // And so we dont need to pass array of input/output pointers(due to conttiguity)
  // as well as array of shapes(due to uniform shapes).
  const size_t first_dim = input->get_common_first_dim();
  const size_t last_dim = input->get_common_last_dim();

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  const dim3 block_size(TB_DIM, TB_DIM);

  auto launch_grouped_swizzle = [&](bool rowwise) {
    const size_t m = rowwise ? first_dim : last_dim;
    const size_t k = rowwise ? last_dim : first_dim;
    const size_t padded_m = round_up_to_multiple(m, 128);
    const size_t padded_k =
        round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);
    const size_t scale_elems = padded_m * padded_k;

    const size_t scale_elem_size = rowwise ? typeToSize(input->scale_inv.dtype)
                                           : typeToSize(input->columnwise_scale_inv.dtype);
    const size_t scale_stride_bytes = scale_elems * scale_elem_size;

    if (rowwise) {
      NVTE_CHECK(input->scale_inv.numel() == input->num_tensors * scale_elems,
                 "Grouped input scale_inv size does not match expected packed size.");
      NVTE_CHECK(output->scale_inv.numel() == output->num_tensors * scale_elems,
                 "Grouped output scale_inv size does not match expected packed size.");
    } else {
      NVTE_CHECK(input->columnwise_scale_inv.numel() == input->num_tensors * scale_elems,
                 "Grouped input columnwise_scale_inv size does not match expected packed size.");
      NVTE_CHECK(output->columnwise_scale_inv.numel() == output->num_tensors * scale_elems,
                 "Grouped output columnwise_scale_inv size does not match expected packed size.");
    }

    const int num_tiles_m = padded_m / SF_TILE_DIM_M;
    const int num_tiles_k = padded_k / SF_TILE_DIM_K;
    int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
    if (vec_load_size == 3) vec_load_size = 1;
    const int n_tiles_in_tb = TB_DIM * vec_load_size;

    dim3 num_blocks;
    if (rowwise) {
      num_blocks = dim3(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m, input->num_tensors);
    } else {
      num_blocks =
          dim3(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size), input->num_tensors);
    }
    const int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

    const int original_M = static_cast<int>(rowwise ? first_dim : last_dim);
    const int original_K = static_cast<int>(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)));
    const void* input_ptr = rowwise ? input->scale_inv.dptr : input->columnwise_scale_inv.dptr;
    void* output_ptr = rowwise ? output->scale_inv.dptr : output->columnwise_scale_inv.dptr;

    if (rowwise) {
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              grouped_swizzle_row_scaling_uniform_shape_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          grouped_swizzle_row_scaling_uniform_shape_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                             padded_k, original_M, original_K,
                                                             scale_stride_bytes);
          break;
        case 2:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              grouped_swizzle_row_scaling_uniform_shape_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          grouped_swizzle_row_scaling_uniform_shape_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                             padded_k, original_M, original_K,
                                                             scale_stride_bytes);
          break;
        case 1:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              grouped_swizzle_row_scaling_uniform_shape_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          grouped_swizzle_row_scaling_uniform_shape_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                             padded_k, original_M, original_K,
                                                             scale_stride_bytes);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
      }
    } else {
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              grouped_swizzle_col_scaling_uniform_shape_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          grouped_swizzle_col_scaling_uniform_shape_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                             padded_k, original_M, original_K,
                                                             scale_stride_bytes);
          break;
        case 2:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              grouped_swizzle_col_scaling_uniform_shape_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          grouped_swizzle_col_scaling_uniform_shape_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                             padded_k, original_M, original_K,
                                                             scale_stride_bytes);
          break;
        case 1:
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(
              grouped_swizzle_col_scaling_uniform_shape_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          grouped_swizzle_col_scaling_uniform_shape_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                             padded_k, original_M, original_K,
                                                             scale_stride_bytes);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
      }
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  if (has_rowwise_scale_inv) {
    launch_grouped_swizzle(true);
  }
  if (has_columnwise_scale_inv) {
    launch_grouped_swizzle(false);
  }
}

void unswizzle_grouped_scaling_factors(const GroupedTensor* input, GroupedTensor* output,
                                       cudaStream_t stream) {
  NVTE_CHECK(output->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Grouped unswizzle supports only MXFP8 scaling.");

  CheckInputGroupedTensor(*input, "input");
  CheckOutputGroupedTensor(*output, "output", false);
  NVTE_CHECK(input->with_gemm_swizzled_scales,
             "Expected input grouped tensor with scales in GEMM swizzled format.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Expected output grouped tensor with scales in compact format.");
  NVTE_CHECK(input->scaling_mode == output->scaling_mode,
             "Input and output grouped tensors must have matching scaling modes.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Input and output grouped tensors must have the same number of tensors.");

  const bool has_rowwise_scale_inv = output->scale_inv.has_data();
  const bool has_columnwise_scale_inv = output->columnwise_scale_inv.has_data();
  if (!has_rowwise_scale_inv && !has_columnwise_scale_inv) {
    return;
  }

  NVTE_CHECK(input->all_same_shape() && output->all_same_shape(),
             "Grouped unswizzle requires uniform tensor shapes.");

  const size_t first_dim = output->get_common_first_dim();
  const size_t last_dim = output->get_common_last_dim();

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  const dim3 block_size(TB_DIM, TB_DIM);

  auto launch_grouped_unswizzle = [&](bool rowwise) {
    const size_t m = rowwise ? first_dim : last_dim;
    const size_t k = rowwise ? last_dim : first_dim;
    const size_t padded_m = round_up_to_multiple(m, 128);
    const size_t padded_k =
        round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);
    const size_t scale_elems = padded_m * padded_k;

    const size_t scale_elem_size = rowwise ? typeToSize(output->scale_inv.dtype)
                                           : typeToSize(output->columnwise_scale_inv.dtype);
    const size_t scale_stride_bytes = scale_elems * scale_elem_size;

    if (rowwise) {
      NVTE_CHECK(input->scale_inv.numel() == input->num_tensors * scale_elems,
                 "Grouped input scale_inv size does not match expected packed size.");
      NVTE_CHECK(output->scale_inv.numel() == output->num_tensors * scale_elems,
                 "Grouped output scale_inv size does not match expected packed size.");
    } else {
      NVTE_CHECK(input->columnwise_scale_inv.numel() == input->num_tensors * scale_elems,
                 "Grouped input columnwise_scale_inv size does not match expected packed size.");
      NVTE_CHECK(output->columnwise_scale_inv.numel() == output->num_tensors * scale_elems,
                 "Grouped output columnwise_scale_inv size does not match expected packed size.");
    }

    const int num_tiles_m = padded_m / SF_TILE_DIM_M;
    const int num_tiles_k = padded_k / SF_TILE_DIM_K;
    int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
    if (vec_load_size == 3) vec_load_size = 1;
    const int n_tiles_in_tb = TB_DIM * vec_load_size;

    dim3 num_blocks;
    if (rowwise) {
      num_blocks = dim3(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m, output->num_tensors);
    } else {
      num_blocks =
          dim3(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size), output->num_tensors);
    }
    const int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

    const void* input_ptr = rowwise ? input->scale_inv.dptr : input->columnwise_scale_inv.dptr;
    void* output_ptr = rowwise ? output->scale_inv.dptr : output->columnwise_scale_inv.dptr;

    using kernel_t = void (*)(const void*, void*, const int, const int, const size_t, const bool);
    kernel_t kernel_fn = nullptr;
    switch (vec_load_size) {
      case 4:
        kernel_fn =
            grouped_unswizzle_scaling_uniform_shape_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>;
        break;
      case 2:
        kernel_fn =
            grouped_unswizzle_scaling_uniform_shape_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>;
        break;
      case 1:
        kernel_fn =
            grouped_unswizzle_scaling_uniform_shape_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>;
        break;
      default:
        NVTE_ERROR("Not valid vec_load_size.");
    }
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
    kernel_fn<<<num_blocks, block_size, slm_size, stream>>>(input_ptr, output_ptr, padded_m,
                                                            padded_k, scale_stride_bytes, rowwise);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  if (has_rowwise_scale_inv) {
    launch_grouped_unswizzle(true);
  }
  if (has_columnwise_scale_inv) {
    launch_grouped_unswizzle(false);
  }
}

}  // namespace transformer_engine

void nvte_swizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_grouped_scaling_factors);
  using namespace transformer_engine;
  swizzle_grouped_scaling_factors(convertNVTEGroupedTensorCheck(input),
                                  convertNVTEGroupedTensorCheck(output), stream);
}

void nvte_unswizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                            cudaStream_t stream) {
  NVTE_API_CALL(nvte_unswizzle_grouped_scaling_factors);
  using namespace transformer_engine;
  unswizzle_grouped_scaling_factors(convertNVTEGroupedTensorCheck(input),
                                    convertNVTEGroupedTensorCheck(output), stream);
}
