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

namespace transformer_engine {
namespace {

constexpr __device__ __host__ int MXFP8_BLOCK_SIZE = 32;
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

}  // namespace

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
      const int original_M = input->flat_first_dim();
      const int original_K = input->flat_last_dim() / MXFP8_BLOCK_SIZE;
      switch (vec_load_size) {
        case 4:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input->scale_inv.dptr, output->scale_inv.dptr, m, k, original_M, original_K);
          break;
        case 2:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input->scale_inv.dptr, output->scale_inv.dptr, m, k, original_M, original_K);
          break;
        case 1:
          NVTE_CHECK_CUDA(
              cudaFuncSetAttribute(swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
          swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
              <<<num_blocks, block_size, slm_size, stream>>>(
                  input->scale_inv.dptr, output->scale_inv.dptr, m, k, original_M, original_K);
          break;
        default:
          NVTE_ERROR("Not valid vec_load_size.");
          break;
      }
      NVTE_CHECK_CUDA(cudaGetLastError());
    }
    if (input->has_columnwise_data()) {
      int vec_load_size = (num_tiles_m - 1) % 4 + 1;
      if (vec_load_size == 3) vec_load_size = 1; /* no int3 and misaligned if using int4/int2 */
      int n_tiles_in_tb = TB_DIM * vec_load_size;
      dim3 num_blocks(DIVUP(num_tiles_k, TB_DIM), DIVUP(num_tiles_m, vec_load_size));
      int slm_size = n_tiles_in_tb * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);
      const int original_M = input->flat_last_dim();
      const int original_K = input->flat_first_dim() / MXFP8_BLOCK_SIZE;
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
      NVTE_CHECK_CUDA(cudaGetLastError());
    }

    // 2D block scaling
  } else {
    NVTE_ERROR("Not implemented for scaling_mode " + to_string(input->scaling_mode) + ", trans.");
  }

  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
void launch_multi_tensor_swizzle_scaling_factors(MultiSwizzleArgs& kernel_args,
                                                 const int vec_load_size, const bool is_rowwise,
                                                 cudaStream_t stream) {
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
  // Launch kernel
  const int num_blocks = kernel_args.block_range[kernel_args.num_tensors];
  dim3 block_size(TB_DIM, TB_DIM);
  if (is_rowwise) {
    switch (vec_load_size) {
      case 4:
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            multi_tensor_swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        multi_tensor_swizzle_row_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
        break;
      case 2:
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            multi_tensor_swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        multi_tensor_swizzle_row_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
        break;
      case 1:
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            multi_tensor_swizzle_row_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
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
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            multi_tensor_swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        multi_tensor_swizzle_col_scaling_kernel<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
        break;
      case 2:
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            multi_tensor_swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        multi_tensor_swizzle_col_scaling_kernel<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
        break;
      case 1:
        NVTE_CHECK_CUDA(cudaFuncSetAttribute(
            multi_tensor_swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, slm_size));
        multi_tensor_swizzle_col_scaling_kernel<int, SF_TILE_DIM_M, SF_TILE_DIM_K>
            <<<num_blocks, block_size, slm_size, stream>>>(kernel_args);
        break;
      default:
        NVTE_ERROR("Not valid vec_load_size.");
        break;
    }
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}
void multi_tensor_swizzle_scaling_factors(const std::vector<Tensor*>& input,
                                          std::vector<Tensor*>& output, cudaStream_t stream) {
  auto num_tensors = input.size();
  bool all_has_data = true;
  bool all_has_columnwise_data = true;
  for (size_t i = 0; i < num_tensors; i++) {
    if (!is_fp8_dtype(input[i]->dtype()) || !is_mxfp_scaling(input[i]->scaling_mode)) {
      NVTE_ERROR("Not implemented caling mode " + to_string(input[i]->scaling_mode) + ".");
    }
    // We don't allow empty tensors. They should be filtered out before calling this function.
    if (input[i]->data.numel() == 0) {
      NVTE_ERROR("Tensor input[" + std::to_string(i) + "] is empty.");
    }
    CheckInputTensor(*input[i], "scaling_factor_input[" + std::to_string(i) + "]");
    CheckInputTensor(*output[i], "scaling_factor_output[" + std::to_string(i) + "]");
    all_has_data &= input[i]->has_data();
    all_has_columnwise_data &= input[i]->has_columnwise_data();
  }
  NVTE_CHECK(all_has_data || all_has_columnwise_data,
             "All tensors should have data or columnwise data.");

  constexpr int SF_TILE_DIM_M = 128;
  constexpr int SF_TILE_DIM_K = 4;
  if (all_has_data) {
    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    for (size_t i = 0; i < num_tensors; i++) {
      //Launch kernel if argument struct is full
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        // There is no int3 and misaligned if using int4/int2.
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, true, stream);
        // Reset the argument struct and vec_load_size
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
      }
      const int m = input[i]->scale_inv.shape[0];
      const int k = input[i]->scale_inv.shape[1];

      NVTE_CHECK(m % SF_TILE_DIM_M == 0, "Input should be padded in M/N dimension!");
      NVTE_CHECK(k % SF_TILE_DIM_K == 0, "Input should be padded in K dimension!");
      NVTE_CHECK(k > 0, "Input scale inverse should be 2D!");
      NVTE_CHECK(
          m * k == std::accumulate(output[i]->scale_inv.shape.begin(),
                                   output[i]->scale_inv.shape.end(), 1, std::multiplies<int>()),
          "Input.scale_inv size is not equal to Output.scale_inv size!");

      int num_tiles_k = k / SF_TILE_DIM_K;
      int vec_load_size_i = (num_tiles_k - 1) % 4 + 1;
      // We use the minimum vec_load_size across all tensors.
      vec_load_size = std::min(vec_load_size, vec_load_size_i);

      const int pos = kernel_args.num_tensors;
      kernel_args.input_list[pos] = const_cast<void*>(input[i]->scale_inv.dptr);
      kernel_args.output_list[pos] = output[i]->scale_inv.dptr;
      kernel_args.m_list[pos] = m;
      kernel_args.k_list[pos] = k;
      kernel_args.original_m_list[pos] = input[i]->flat_first_dim();
      kernel_args.original_k_list[pos] = input[i]->flat_last_dim() / MXFP8_BLOCK_SIZE;
      kernel_args.num_tensors++;
    }
    // Launch the remaining tensors
    // There is no int3 and misaligned if using int4/int2.
    if (vec_load_size == 3) vec_load_size = 1;
    launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
        kernel_args, vec_load_size, true, stream);
  }

  if (all_has_columnwise_data) {
    MultiSwizzleArgs kernel_args;
    kernel_args.num_tensors = 0;
    kernel_args.block_range[0] = 0;
    int vec_load_size = 4;
    for (size_t i = 0; i < num_tensors; i++) {
      //Launch kernel if argument struct is full
      if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
        // There is no int3 and misaligned if using int4/int2.
        if (vec_load_size == 3) vec_load_size = 1;
        launch_multi_tensor_swizzle_scaling_factors<SF_TILE_DIM_M, SF_TILE_DIM_K>(
            kernel_args, vec_load_size, false, stream);
        // Reset the argument struct and vec_load_size
        kernel_args.num_tensors = 0;
        vec_load_size = 4;
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

      int num_tiles_k = k / SF_TILE_DIM_K;
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
  multi_tensor_swizzle_scaling_factors(input_list, output_list, stream);
}
