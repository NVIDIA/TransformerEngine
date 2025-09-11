/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cuda/barrier>

#include "common/common.h"
#include "common/recipe/recipe_common.cuh"
#include "common/util/cuda_runtime.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

#if (!defined(__CUDA_MINIMUM_ARCH__) && __CUDA_ARCH__ >= 900) || \
    (defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ >= 900)
#define TMA_HW_SUPPORTED
#endif

namespace transformer_engine {
namespace {

// const values configuration

constexpr size_t kThreadsPerWarp = 32;
#ifdef TMA_HW_SUPPORTED
constexpr size_t BLOCK_TILE_DIM = 128;
constexpr size_t WARP_TILE_DIM_X = 32;
constexpr size_t WARP_TILE_DIM_Y = 64;
constexpr size_t THREAD_TILE_DIM_X = 16;
constexpr size_t THREAD_TILE_DIM_Y = 4;
#else
constexpr size_t BLOCK_TILE_DIM = 128;
constexpr size_t WARP_TILE_DIM_X = 64;
constexpr size_t WARP_TILE_DIM_Y = 32;
constexpr size_t THREAD_TILE_DIM_X = 8;
constexpr size_t THREAD_TILE_DIM_Y = 8;
#endif

#ifdef TMA_HW_SUPPORTED
constexpr size_t NUM_BYTES_PER_BANK = 4;
constexpr size_t NUM_BANKS_PER_SHARED_ELEM = THREAD_TILE_DIM_Y / NUM_BYTES_PER_BANK;
constexpr size_t SHARED_BLOCK_TILE_DIM_Y = BLOCK_TILE_DIM;
constexpr size_t SHARED_BLOCK_TILE_DIM_X_BANKS =
    BLOCK_TILE_DIM / (NUM_BYTES_PER_BANK * NUM_BANKS_PER_SHARED_ELEM);
constexpr size_t NUM_BANKS_Y_IN_WARP = WARP_TILE_DIM_Y / NUM_BYTES_PER_BANK;
#endif
constexpr size_t ELE_PER_THREAD = THREAD_TILE_DIM_X * THREAD_TILE_DIM_Y;
constexpr size_t THREADS_PER_BLOCK = BLOCK_TILE_DIM * BLOCK_TILE_DIM / ELE_PER_THREAD;
constexpr size_t NUM_WARPS_X_IN_BLOCK = BLOCK_TILE_DIM / WARP_TILE_DIM_X;
constexpr size_t NUM_WARPS_Y_IN_BLOCK = BLOCK_TILE_DIM / WARP_TILE_DIM_Y;
constexpr size_t NUM_WARPS_IN_BLOCK = NUM_WARPS_X_IN_BLOCK * NUM_WARPS_Y_IN_BLOCK;

constexpr size_t NUM_THREADS_X_IN_WARP = WARP_TILE_DIM_X / THREAD_TILE_DIM_X;
constexpr size_t NUM_THREADS_Y_IN_WARP = kThreadsPerWarp / NUM_THREADS_X_IN_WARP;

#define MIN(a, b) (a < b ? a : b)

template <bool kReturnTranspose, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    block_scaled_cast_transpose_kernel(const IType* const input, OType* const output_c,
                                       OType* const output_t, CType* const tile_scales_inv_c,
                                       CType* const tile_scales_inv_t, const size_t row_length,
                                       const size_t num_rows, const size_t scale_stride_x,
                                       const size_t scale_stride_y, const size_t scale_t_stride_x,
                                       const size_t scale_t_stride_y, const float epsilon,
                                       const __grid_constant__ CUtensorMap tensor_map_output_t,
                                       const bool pow_2_scaling, const float* noop_ptr,
                                       const bool pdl_sync, const bool pdl_trigger) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = kReturnTranspose ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  // This is ONLY true if the input is a full tile
  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_idx =
      tile_id_y * BLOCK_TILE_DIM * row_length + tile_id_x * BLOCK_TILE_DIM;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

// Block until the previous kernel has completed and flushed results to global memory.
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (pdl_sync) {
    cudaGridDependencySynchronize();
  }
#endif

// Step 1: Load a block tile of input data into thread tiles on registers
#pragma unroll
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
    thrd_tile_input[i].load_from(input + thread_tile_start_idx + i * row_length);
  }

  // Step 2: calculate block tile amax and scale
  // Calculate thread_tile amax
  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
    }
  }
  // Reduce amax in the warp (32x32 tile)
  warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  // broadcast the amax to all threads in a warp from the lane 0
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  // reduce warp_tile_amax across multiple warps in a thread block using shared mem
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  // only 8 elements needs reduction, if using reduction tree, multiple _syncthreads will be needed,
  // instead we just let thread 0 do the job
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (kReturnTranspose) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

// Trigger the next kernel here so that it's load from global memory can overlap with this kernel's
// store to global memory.
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (pdl_trigger) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  OVecCast tmp_output_c;

  for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      // Step 3: Store cast output
      CType scale_data = block_tile_scale;

      OType scaled_elt =
          static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) * scale_data);
      tmp_output_c.data.elt[j] = scaled_elt;
      // Step 4: do transpose within thread tile
      if constexpr (kReturnTranspose) {
        thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
      }
    }
    tmp_output_c.store_to(output_c + thread_tile_start_idx + i * row_length);
  }

  // Step 4: store transpose into shared memory
  if constexpr (kReturnTranspose) {
#ifdef TMA_HW_SUPPORTED
    __shared__ alignas(128)
        OVecTrans block_tile_trans_shared[SHARED_BLOCK_TILE_DIM_Y][SHARED_BLOCK_TILE_DIM_X_BANKS];
    OType(*block_tile_trans_shared_otype_ptr)[BLOCK_TILE_DIM] =
        reinterpret_cast<OType(*)[BLOCK_TILE_DIM]>(block_tile_trans_shared);

#pragma unroll
    for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
      auto warp_id_in_block_x_ = warp_id_in_block_y;
      auto warp_id_in_block_y_ = warp_id_in_block_x;
      int row_idx = warp_id_in_block_y_ * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
                    tid_in_warp_x * THREAD_TILE_DIM_X + i;
      int col_idx =
          warp_id_in_block_x_ * (NUM_BANKS_Y_IN_WARP / NUM_BANKS_PER_SHARED_ELEM) + tid_in_warp_y;
      block_tile_trans_shared[row_idx][col_idx] = thrd_tile_out_trans[i];
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Step 5: store transpose output
    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t*>(&tensor_map_output_t), tile_id_y * BLOCK_TILE_DIM,
          tile_id_x * BLOCK_TILE_DIM,
          reinterpret_cast<uint64_t*>(block_tile_trans_shared_otype_ptr));
      // Wait for TMA transfer to have finished reading shared memory.
      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
      // Wait for the group to have completed reading from shared memory.
      ptx::cp_async_bulk_wait_group_read<0>();
    }
#else
    // Step 4 Alternative (when TMA is not available, skip writing to shared memory)
    const size_t block_tile_t_start_idx =
        tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
    const size_t warp_tile_t_start_idx =
        block_tile_t_start_idx +
        warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
        warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
    const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                           tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                           tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
    for (int i = 0; i < THREAD_TILE_DIM_X; i++) {
      thrd_tile_out_trans[i].store_to(output_t + thread_tile_t_start_idx + i * num_rows);
    }
#endif
  }
}

template <bool kReturnTranspose, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK) block_scaled_cast_transpose_kernel_notaligned(
    const IType* const input, OType* const output_c, OType* const output_t,
    CType* const tile_scales_inv_c, CType* const tile_scales_inv_t, const size_t row_length,
    const size_t num_rows, const size_t scale_stride_x, const size_t scale_stride_y,
    const size_t scale_t_stride_x, const size_t scale_t_stride_y, const float epsilon,
    const bool pow_2_scaling, const float* noop_ptr, const bool pdl_sync, const bool pdl_trigger) {
  using IVec = Vec<IType, THREAD_TILE_DIM_X>;
  using OVecCast = Vec<OType, THREAD_TILE_DIM_X>;
  using OVecTrans = Vec<OType, THREAD_TILE_DIM_Y>;

  if (noop_ptr != nullptr && noop_ptr[0] == 1.0f) {
    return;
  }

  // shared mem for amax reduction in entire block, each warp produces one amax, there are
  // NUM_WARPS_IN_BLOCK amax to reduce
  __shared__ CType block_tile_amax_shared[NUM_WARPS_IN_BLOCK];

  IVec thrd_tile_input[THREAD_TILE_DIM_Y];
  constexpr int THREAD_TILE_DIM_X_ = kReturnTranspose ? THREAD_TILE_DIM_X : 1;
  OVecTrans thrd_tile_out_trans[THREAD_TILE_DIM_X_];

  const int tid_in_warp = threadIdx.x % kThreadsPerWarp;
  const int tid_in_warp_x = tid_in_warp % NUM_THREADS_X_IN_WARP;
  const int tid_in_warp_y = tid_in_warp / NUM_THREADS_X_IN_WARP;
  const int warp_id_in_block = threadIdx.x / kThreadsPerWarp;
  const int warp_id_in_block_x = warp_id_in_block % NUM_WARPS_X_IN_BLOCK;
  const int warp_id_in_block_y = warp_id_in_block / NUM_WARPS_X_IN_BLOCK;

  const int tile_id_x = blockIdx.x;
  const int tile_id_y = blockIdx.y;

  const size_t block_tile_start_row_idx = tile_id_y * BLOCK_TILE_DIM;
  const size_t block_tile_start_col_idx = tile_id_x * BLOCK_TILE_DIM;
  const size_t block_tile_start_idx =
      block_tile_start_row_idx * row_length + block_tile_start_col_idx;
  const size_t warp_tile_start_idx =
      block_tile_start_idx +
      warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP * row_length +
      warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP;
  const size_t thread_tile_start_idx = warp_tile_start_idx +
                                       tid_in_warp_y * THREAD_TILE_DIM_Y * row_length +
                                       tid_in_warp_x * THREAD_TILE_DIM_X;

  // handle non-full tile
  // check for three cases: full thread tile, nonfull thread tile, empty thread tile
  // for empty thread tile, directly write zero to the transposed shared mem buffer
  // for nonfull thread tile, fill zero to thread tile and act as if it's full
  const size_t thread_tile_start_row_idx =
      tile_id_y * BLOCK_TILE_DIM + warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP +
      tid_in_warp_y * THREAD_TILE_DIM_Y;
  const size_t thread_tile_start_col_idx =
      tile_id_x * BLOCK_TILE_DIM + warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP +
      tid_in_warp_x * THREAD_TILE_DIM_X;

  const size_t thread_tile_end_row_idx = thread_tile_start_row_idx + THREAD_TILE_DIM_Y - 1;
  const size_t thread_tile_end_col_idx = thread_tile_start_col_idx + THREAD_TILE_DIM_X - 1;

  bool full_thrd_tile =
      (thread_tile_end_row_idx < num_rows) && (thread_tile_end_col_idx < row_length);
  bool empty_thrd_tile =
      (thread_tile_start_row_idx >= num_rows) || (thread_tile_start_col_idx >= row_length);
  bool nonfull_thrd_tile = (!full_thrd_tile) && (!empty_thrd_tile);

  const size_t thread_tile_ncols =
      MIN(THREAD_TILE_DIM_X,
          (MIN(thread_tile_end_col_idx, row_length - 1) - thread_tile_start_col_idx + 1));
  const size_t thread_tile_nrows =
      MIN(THREAD_TILE_DIM_Y,
          (MIN(thread_tile_end_row_idx, num_rows - 1) - thread_tile_start_row_idx + 1));

  CType warp_tile_amax;
  CType block_tile_amax;
  CType block_tile_scale;
  CType amax = 0;

// Block until the previous kernel has completed and flushed results to global memory.
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (pdl_sync) {
    cudaGridDependencySynchronize();
  }
#endif

  if (!empty_thrd_tile) {
    // Step 1: Load a block tile of input data into thread tiles on registers
    // Edge case: nonfull thread tile case, will use the partial load function here
    if (nonfull_thrd_tile) {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        if (i >= thread_tile_nrows) {
          thrd_tile_input[i].clear();
        } else {
          thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * row_length, 0,
                                            thread_tile_ncols);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
        thrd_tile_input[i].load_from_elts(input + thread_tile_start_idx + i * row_length, 0,
                                          THREAD_TILE_DIM_X);
      }
    }

    // Step 2: calculate block tile amax and scale
    // Calculate thread_tile amax
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(static_cast<CType>(thrd_tile_input[i].data.elt[j])));
      }
    }
  }
  // Reduce amax in the warp (32x32 tile)
  warp_tile_amax = warp_reduce_max<kThreadsPerWarp>(amax);
  // broadcast the amax to all threads in a warp from the lane 0
  constexpr int lane_zero = 0;
  warp_tile_amax = __shfl_sync(0xFFFFFFFF, warp_tile_amax, lane_zero);

  // reduce warp_tile_amax across multiple warps in a thread block using shared mem
  if (tid_in_warp == 0) {
    block_tile_amax_shared[warp_id_in_block_y * NUM_WARPS_X_IN_BLOCK + warp_id_in_block_x] =
        warp_tile_amax;
  }
  __syncthreads();
  // only 8 elements needs reduction, if using reduction tree, multiple _syncthreads will be needed,
  // instead we just let thread 0 do the job
  if (threadIdx.x == 0) {
    CType blk_amax = block_tile_amax_shared[0];
#pragma unroll
    for (int idx = 1; idx < NUM_WARPS_IN_BLOCK; idx++) {
      blk_amax = fmaxf(blk_amax, block_tile_amax_shared[idx]);
    }
    block_tile_amax_shared[0] = blk_amax;
  }
  __syncthreads();
  block_tile_amax = block_tile_amax_shared[0];

  block_tile_scale =
      compute_scale_from_types<IType, OType>(block_tile_amax, epsilon, pow_2_scaling);

  if (threadIdx.x == 0) {
    static_assert(std::is_same<CType, float>::value);
    const CType scale_inv = 1.0f / block_tile_scale;

    size_t row_idx = tile_id_y;
    size_t col_idx = tile_id_x;
    tile_scales_inv_c[row_idx * scale_stride_y + col_idx * scale_stride_x] = scale_inv;

    if constexpr (kReturnTranspose) {
      row_idx = tile_id_x;
      col_idx = tile_id_y;
      tile_scales_inv_t[row_idx * scale_t_stride_y + col_idx * scale_t_stride_x] = scale_inv;
    }
  }

// Trigger the next kernel here so that it's load from global memory can overlap with this kernel's
// store to global memory.
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (pdl_trigger) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  // Step 3: Store cast output, Step 4: do transpose within thread tile
  // Edge case: in the non-full tile case, there are three subcases
  // for full thread tile, it's the same thing here
  // for nonfull thread tile, pay attention when saving tmp_output_c to global
  // memory, cannot vec store_to, but need to elt store to for empty tile,
  // it should not enter this step, skip to Step 4

  // set thrd_tile_out_trans to all zero
  if constexpr (kReturnTranspose) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
      thrd_tile_out_trans[j].clear();
    }
  }

  if (!empty_thrd_tile) {
    OVecCast tmp_output_c;
    for (int i = 0; i < THREAD_TILE_DIM_Y; i++) {
      if (i >= thread_tile_nrows) {
        continue;
      }
#pragma unroll
      for (int j = 0; j < THREAD_TILE_DIM_X; j++) {
        // Step 3: Store cast output
        CType scale_data = block_tile_scale;

        OType scaled_elt =
            static_cast<OType>(static_cast<CType>(thrd_tile_input[i].data.elt[j]) * scale_data);
        tmp_output_c.data.elt[j] = scaled_elt;
        // Step 4: do transpose within thread tile
        if constexpr (kReturnTranspose) {
          thrd_tile_out_trans[j].data.elt[i] = scaled_elt;
        }
      }
      tmp_output_c.store_to_elts(output_c + thread_tile_start_idx + i * row_length, 0,
                                 thread_tile_ncols);
    }

    if constexpr (kReturnTranspose) {
      const size_t block_tile_t_start_idx =
          tile_id_x * BLOCK_TILE_DIM * num_rows + tile_id_y * BLOCK_TILE_DIM;
      const size_t warp_tile_t_start_idx =
          block_tile_t_start_idx +
          warp_id_in_block_x * THREAD_TILE_DIM_X * NUM_THREADS_X_IN_WARP * num_rows +
          warp_id_in_block_y * THREAD_TILE_DIM_Y * NUM_THREADS_Y_IN_WARP;
      const size_t thread_tile_t_start_idx = warp_tile_t_start_idx +
                                             tid_in_warp_x * THREAD_TILE_DIM_X * num_rows +
                                             tid_in_warp_y * THREAD_TILE_DIM_Y;
#pragma unroll
      for (int i = 0; i < thread_tile_ncols; i++) {
        thrd_tile_out_trans[i].store_to_elts(output_t + thread_tile_t_start_idx + i * num_rows, 0,
                                             thread_tile_nrows);
      }
    }
  }
}

template <typename OutputType>
CUtensorMap get_tensor_map(const SimpleTensor& tensor, size_t global_dim_x, size_t global_dim_y) {
  CUtensorMapDataType dataType;
  if constexpr (std::is_same_v<OutputType, __nv_fp8_e4m3> ||
                std::is_same_v<OutputType, __nv_fp8_e5m2>) {
    dataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else {
    NVTE_CHECK(false, "Invalid Output type (must be FP8).");
  }

  CUtensorMap tensor_map_output_trans{};
  create_2D_tensor_map(tensor_map_output_trans, tensor, global_dim_y, global_dim_x,
                       /*shmemY=*/BLOCK_TILE_DIM, /*shmemX=*/BLOCK_TILE_DIM,
                       /*stride_elems=*/global_dim_x, /*offset_elems=*/0, sizeof(OutputType) * 8);
  return tensor_map_output_trans;
}

}  // namespace
}  // namespace transformer_engine

namespace transformer_engine::detail {

void quantize_transpose_square_blockwise(const SimpleTensor& input, SimpleTensor& scale_inv,
                                         SimpleTensor& scale_inv_t, SimpleTensor& output,
                                         SimpleTensor& output_t, const float epsilon,
                                         const bool return_transpose, const bool pow_2_scale,
                                         const SimpleTensor& noop_tensor, const bool pdl_sync,
                                         const bool pdl_trigger, cudaStream_t stream) {
  NVTE_API_CALL(quantize_transpose_square_blockwise);
  checkCuDriverContext(stream);

  NVTE_CHECK(input.shape == output.shape, "Input and output must have the same shape.");
  const size_t row_length = input.shape.size() > 0 ? input.shape.at(input.shape.size() - 1) : 1u;
  size_t num_rows = 1;
  for (size_t i = 0; (i < input.shape.size() - 1) && (input.shape.size() > 0); ++i) {
    num_rows *= input.shape.at(i);
  }

  NVTE_CHECK(scale_inv.shape.size() == 2, "scale_inv must have 2 dimensions.");

  size_t scale_k = scale_inv.shape[1];

  const size_t scale_stride_x = 1;
  const size_t scale_stride_y = scale_k;

  size_t scale_t_stride_x = 0;
  size_t scale_t_stride_y = 0;

  const float* noop_ptr = reinterpret_cast<const float*>(noop_tensor.dptr);

  if (return_transpose) {
    NVTE_CHECK(output_t.shape.size() == input.shape.size(),
               "output_t must have same number of dimensions as input.");
    if (output_t.shape.size() > 0) {
      NVTE_CHECK(output_t.shape[0] == row_length, "Wrong dimension 0 of output_t.");
      for (size_t i = 1; i < output_t.shape.size(); ++i) {
        NVTE_CHECK(output_t.shape.at(i) == input.shape.at(i - 1), "Wrong dimension in output_t");
      }
    }
    NVTE_CHECK(output.dtype == output_t.dtype, "output and output_t need to have the same type.");

    NVTE_CHECK(scale_inv_t.shape.size() == 2, "scale_inv_t must have 2 dimensions.");

    scale_t_stride_x = 1;
    scale_t_stride_y = scale_inv_t.shape[1];
  }

  const size_t num_blocks_x = DIVUP(row_length, BLOCK_TILE_DIM);
  const size_t num_blocks_y = DIVUP(num_rows, BLOCK_TILE_DIM);
  dim3 grid(num_blocks_x, num_blocks_y, 1);
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {grid, THREADS_PER_BLOCK, 0, stream, NULL, 0};
  if (transformer_engine::cuda::sm_arch(transformer_engine::cuda::current_device()) >= 90) {
    cfg.attrs = attribute;
    cfg.numAttrs = 1;
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype, InputType,

      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output.dtype, OutputType,

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_transpose, kReturnTranspose,

              const bool full_tile =
                  row_length % BLOCK_TILE_DIM == 0 && num_rows % BLOCK_TILE_DIM == 0;

              if (full_tile) {
                CUtensorMap tensor_map_output_trans;
                if (return_transpose) {
                  tensor_map_output_trans =
                      get_tensor_map<OutputType>(output_t, num_rows, row_length);
                }
                NVTE_CHECK_CUDA(cudaLaunchKernelEx(
                    &cfg,
                    block_scaled_cast_transpose_kernel<kReturnTranspose, float, InputType,
                                                       OutputType>,
                    reinterpret_cast<const InputType*>(input.dptr),
                    reinterpret_cast<OutputType*>(output.dptr),
                    reinterpret_cast<OutputType*>(output_t.dptr),
                    reinterpret_cast<float*>(scale_inv.dptr),
                    reinterpret_cast<float*>(scale_inv_t.dptr), row_length, num_rows,
                    scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y, epsilon,
                    tensor_map_output_trans, pow_2_scale, noop_ptr, pdl_sync, pdl_trigger));
              } else {
                NVTE_CHECK_CUDA(cudaLaunchKernelEx(
                    &cfg,
                    block_scaled_cast_transpose_kernel_notaligned<kReturnTranspose, float,
                                                                  InputType, OutputType>,
                    reinterpret_cast<const InputType*>(input.dptr),
                    reinterpret_cast<OutputType*>(output.dptr),
                    reinterpret_cast<OutputType*>(output_t.dptr),
                    reinterpret_cast<float*>(scale_inv.dptr),
                    reinterpret_cast<float*>(scale_inv_t.dptr), row_length, num_rows,
                    scale_stride_x, scale_stride_y, scale_t_stride_x, scale_t_stride_y, epsilon,
                    pow_2_scale, noop_ptr, pdl_sync, pdl_trigger));
              }  // full-tile
              )  // return_transpose
          )      // OutputType
      )          // InputType
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine::detail
