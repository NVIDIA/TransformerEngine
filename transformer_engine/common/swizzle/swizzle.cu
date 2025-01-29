/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace {

constexpr int TB_DIM = 32;
constexpr int NEW_SF_TILE_DIM_K = 16;
constexpr int N_SF_PER_TD_PER_TILE = 4;

// output is in ~K-major interleaved blocks
constexpr int NEW_SF_TILE_DIM_K_I32 = NEW_SF_TILE_DIM_K / 4;
constexpr int NEW_SF_TILE_DIM_M_I32 = 32;

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

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void swizzle_col_scaling_kernel(const void* input, void* output, const int M,
                                           const int K) {
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
  if (blockIdx.x == gridDim.x - 1) {
    k_tiles_in_tb = (K_i32 / SF_TILE_DIM_K_I32 - 1) % k_tiles_in_tb + 1;
  }
  if (blockIdx.y == gridDim.y - 1) {
    m_tiles_in_tb = (M_i32 / SF_TILE_DIM_M_I32 - 1) % m_tiles_in_tb + 1;
  }

  const int32_t* input_i32 = reinterpret_cast<const int32_t*>(input) +
                             blockIdx.x * TB_DIM * SF_TILE_DIM_K_I32 * M_i32 +
                             blockIdx.y * N_TILE_PER_TD * SF_TILE_DIM_M_I32;
  int32_t* output_i32[N_TILE_PER_TD];
#pragma unroll
  for (int i = 0; i < m_tiles_in_tb; i++) {
    output_i32[i] = reinterpret_cast<int32_t*>(output) + blockIdx.x * TB_DIM * SF_TILE_SIZE_I32 +
                    (blockIdx.y * N_TILE_PER_TD + i) * SF_TILE_DIM_M_I32 * K_i32;
  }
  extern __shared__ int slm[];

  // load, global -> regs
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < m_tiles_in_tb * SF_TILE_DIM_M_I32 &&
      threadIdx.y < k_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(
          input_i32 + (threadIdx.y * SF_TILE_DIM_K_I32 + i) * M_i32 + threadIdx.x * N_TILE_PER_TD));
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

template <typename LType, int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void swizzle_row_scaling_kernel(const void* input, void* output, const int M,
                                           const int K) {
  constexpr int N_TILE_PER_TD = sizeof(LType) / sizeof(int);
  constexpr int N_TILES_IN_TB = TB_DIM * N_TILE_PER_TD;

  // input is in K-major
  constexpr int SF_TILE_SIZE_I32 = SF_TILE_DIM_M * SF_TILE_DIM_K / 4;
  constexpr int SF_TILE_DIM_M_I32 = SF_TILE_DIM_M;

  int n_tiles_in_tb = N_TILES_IN_TB;
  const int K_i32 = K / 4;
  if (blockIdx.x == gridDim.x - 1) {
    n_tiles_in_tb = (K_i32 - 1) % N_TILES_IN_TB + 1;
  }

  const int* input_i32 = reinterpret_cast<const int*>(input) +
                         blockIdx.y * SF_TILE_DIM_M_I32 * K_i32 + blockIdx.x * N_TILES_IN_TB;
  int* output_i32 = reinterpret_cast<int*>(output) + blockIdx.y * SF_TILE_DIM_M_I32 * K_i32 +
                    blockIdx.x * N_TILES_IN_TB * SF_TILE_SIZE_I32;

  extern __shared__ int4 slm_v4i[];

  // load, global -> regs
  LType regs_vec[N_SF_PER_TD_PER_TILE];
  if (threadIdx.x * N_TILE_PER_TD < n_tiles_in_tb) {
#pragma unroll
    for (int i = 0; i < N_SF_PER_TD_PER_TILE; i++) {
      regs_vec[i] = __ldg(reinterpret_cast<const LType*>(
          input_i32 + (i * TB_DIM + threadIdx.y) * K_i32 + threadIdx.x * N_TILE_PER_TD));
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

}  // namespace

namespace transformer_engine {

void swizzle_scaling_factors(const Tensor* input, Tensor* output, cudaStream_t stream) {
  if (!is_fp8_dtype(input->dtype()) || is_delayed_tensor_scaling(input->scaling_mode)) {
    NVTE_ERROR("Not implemented caling mode " + to_string(input->scaling_mode) + ".");
  }

  // Do nothing if tensor is empty
  if (input->data.numel() == 0) {
    return;
  }

  CheckInputTensor(*input, "scaling_factor_input");
  CheckInputTensor(*output, "scaling_factor_output");

  auto& scaling_mode = input->scaling_mode;

  // 1D block scaling, row-wise or colum-wise
  if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
    const int m =
        input->has_data() ? input->scale_inv.shape[0] : input->columnwise_scale_inv.shape[1];
    const int k =
        input->has_data() ? input->scale_inv.shape[1] : input->columnwise_scale_inv.shape[0];

    constexpr int SF_TILE_DIM_M = 128;
    constexpr int SF_TILE_DIM_K = 4;

    NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
    NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");
    NVTE_CHECK(k > 0, "Input scale inverse should be 2D!");
    if (output->has_data()) {
      NVTE_CHECK(m * k == std::accumulate(output->scale_inv.shape.begin(),
                                          output->scale_inv.shape.end(), 1, std::multiplies<int>()),
                 "Input.scale_inv size is not equal to Output.scale_inv size!");
    }
    if (output->has_columnwise_data()) {
      NVTE_CHECK(m * k == std::accumulate(output->columnwise_scale_inv.shape.begin(),
                                          output->columnwise_scale_inv.shape.end(), 1,
                                          std::multiplies<int>()),
                 "Input.columnwise_scale_inv size is not equal to "
                 "Output.columnwise_scale_inv size!");
    }

    int num_tiles_m = m / SF_TILE_DIM_M;
    int num_tiles_k = k / SF_TILE_DIM_K;

    dim3 block_size(TB_DIM, TB_DIM);
    if (input->has_data()) {
      int vec_load_size = (num_tiles_k - 1) % 4 + 1;
      /* there is no int3 and misaligned if using int4/int2 */
      if (vec_load_size == 3) vec_load_size = 1;
      int n_tiles_in_tb = TB_DIM * vec_load_size;
      dim3 num_blocks(DIVUP(num_tiles_k, n_tiles_in_tb), num_tiles_m);
      int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
      switch (vec_load_size) {
        case 4:
          cudaFuncSetAttribute(swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size);
          swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->scale_inv.dptr,
                                                             output->scale_inv.dptr, m, k);
          break;
        case 2:
          cudaFuncSetAttribute(swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size);
          swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->scale_inv.dptr,
                                                             output->scale_inv.dptr, m, k);
          break;
        case 1:
          cudaFuncSetAttribute(swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size);
          swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(input->scale_inv.dptr,
                                                             output->scale_inv.dptr, m, k);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }
    if (input->has_columnwise_data()) {
      int vec_load_size = (num_tiles_m - 1) % 4 + 1;
      if (vec_load_size == 3) vec_load_size = 1; /* no int3 and misaligned if using int4/int2 */
      int n_tiles_in_tb = TB_DIM * vec_load_size;
      dim3 num_blocks(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size));
      int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
      switch (vec_load_size) {
        case 4:
          cudaFuncSetAttribute(swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size);
          swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m, k);
          break;
        case 2:
          cudaFuncSetAttribute(swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size);
          swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m, k);
          break;
        case 1:
          cudaFuncSetAttribute(swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size);
          swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input->columnwise_scale_inv.dptr, output->columnwise_scale_inv.dptr, m, k);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
    }

    // 2D block scaling
  } else {
    NVTE_ERROR("Not implemented for scaling_mode " + to_string(input->scaling_mode) + ", trans.");
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
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
  swizzle_scaling_factors(reinterpret_cast<const Tensor*>(input), reinterpret_cast<Tensor*>(output),
                          stream);
}
