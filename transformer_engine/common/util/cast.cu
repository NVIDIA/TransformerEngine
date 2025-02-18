/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>
#include <iostream>
#include <limits>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "cuda_driver.h"
#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {

constexpr size_t MXFP8_CHUNK_DIM_Y = 64;
constexpr size_t MXFP8_CHUNK_DIM_X = 64;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK_X = 1;
constexpr size_t MXFP8_CHUNKS_PER_BLOCK = MXFP8_CHUNKS_PER_BLOCK_Y * MXFP8_CHUNKS_PER_BLOCK_X;
constexpr size_t MXFP8_THREADS_PER_CHUNK = 64;
constexpr size_t MXFP8_BUFFERS_NUM = 2;
constexpr size_t MXFP8_PREFETCH_BUFFERS_NUM = 1;
static_assert(MXFP8_PREFETCH_BUFFERS_NUM < MXFP8_BUFFERS_NUM);

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t MXFP8_BUFFER_DIM_Y = 32;                 // only 32 is supported
constexpr size_t MXFP8_BUFFER_DIM_X = MXFP8_CHUNK_DIM_X;  // 64
constexpr size_t MXFP8_SHMEM_DIM_Y = MXFP8_BUFFER_DIM_Y;  // 32
constexpr size_t MXFP8_SHMEM_DIM_X = MXFP8_BUFFER_DIM_X;  // 64

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE =
    MXFP8_CHUNK_DIM_X / ELEMS_PER_THREAD;  //   4 = 64 / 16
constexpr size_t THREADS_PER_CHUNK_Y_ROWWISE =
    MXFP8_THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_ROWWISE;         //  16 = 64 / 4
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = MXFP8_CHUNK_DIM_X;  //  64
constexpr size_t MXFP8_BUFF_STAGES_NUM =
    MXFP8_BUFFER_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE;                        //   2 = 32 / 16
constexpr size_t MXFP8_ITERATIONS = MXFP8_CHUNK_DIM_Y / MXFP8_BUFFER_DIM_Y;  //   2 = 64 / 32
static_assert(MXFP8_ITERATIONS >= MXFP8_PREFETCH_BUFFERS_NUM);

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(MXFP8_THREADS_PER_CHUNK)
    cast_mxfp8_2D_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                         const __grid_constant__ CUtensorMap tensor_map_act_input,
                         const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                         const __grid_constant__ CUtensorMap tensor_map_output_colwise,
                         e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                         float *const dbias_workspace, float *const amax_ptr_rowwise,
                         float *const amax_ptr_colwise, const size_t rows, const size_t cols,
                         const size_t scale_stride_rowwise, const size_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_DBIAS_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = MXFP8_CHUNK_DIM_Y;                //   2 = 64 / 32
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = MXFP8_CHUNK_DIM_X / SCALE_DIM_X;  //  64 = 64 / 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_Y =
      SCALES_ROWWISE_PER_CHUNK_Y * MXFP8_CHUNKS_PER_BLOCK_Y;  //   2 = 2 * 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_X =
      SCALES_ROWWISE_PER_CHUNK_X * MXFP8_CHUNKS_PER_BLOCK_X;  //  64 = 64 * 1

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = MXFP8_CHUNK_DIM_Y / SCALE_DIM_Y;  //   2 = 64 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = MXFP8_CHUNK_DIM_X;                //  64 = 64 / 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_Y =
      SCALES_COLWISE_PER_CHUNK_Y * MXFP8_CHUNKS_PER_BLOCK_Y;  //   2 = 2 * 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_X =
      SCALES_COLWISE_PER_CHUNK_X * MXFP8_CHUNKS_PER_BLOCK_X;  //  64 = 64 * 1

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                      //   2 = 32 / 16
  constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X_ROWWISE;  //   2

  const int block_offset_Y = blockIdx.y * MXFP8_CHUNKS_PER_BLOCK_Y * MXFP8_CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X;
  const int scales_rowwise_block_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_BLOCK_Y;
  const int scales_rowwise_block_offset_X = blockIdx.x * SCALES_ROWWISE_PER_BLOCK_X;
  const int scales_colwise_block_offset_Y = blockIdx.y * SCALES_COLWISE_PER_BLOCK_Y;
  const int scales_colwise_block_offset_X = blockIdx.x * SCALES_COLWISE_PER_BLOCK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  // const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  const int dbias_rowwise_offset_Y = blockIdx.y * MXFP8_CHUNKS_PER_BLOCK_Y + tid_rowwise_Y;
  const int dbias_rowwise_block_offset_X =
      blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X + thread_offset_X_rowwise;
  const int dbias_colwise_offset_Y = blockIdx.y;
  const int dbias_colwise_block_offset_X =
      blockIdx.x * MXFP8_CHUNKS_PER_BLOCK_X * MXFP8_CHUNK_DIM_X + tid_colwise_X;
  const int dbias_stride = cols;

  Vec<float, ELEMS_PER_THREAD> partial_dbias_rowwise[MXFP8_CHUNKS_PER_BLOCK_X];
  float partial_dbias_colwise[MXFP8_CHUNKS_PER_BLOCK_X];
  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
#pragma unroll
      for (int i = 0; i < MXFP8_CHUNKS_PER_BLOCK_X; ++i) {
        partial_dbias_rowwise[i].clear();
      }
    } else {
#pragma unroll
      for (int i = 0; i < MXFP8_CHUNKS_PER_BLOCK_X; ++i) {
        partial_dbias_colwise[i] = 0;
      }
    }
  }

  // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
  __shared__ alignas(128) IType in_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128) IType act_in_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128)
      OType out_rowwise_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];
  __shared__ alignas(128)
      OType out_colwise_sh[MXFP8_BUFFERS_NUM][MXFP8_SHMEM_DIM_Y][MXFP8_SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / MXFP8_BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size * (IS_DACT ? 2 : 1);

  const bool is_master_thread = (threadIdx.x == 0);

  float block_amax = 0;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[MXFP8_ITERATIONS];

  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < MXFP8_ITERATIONS; ++iter) {
      ptx::mbarrier_init(&mbar[iter], MXFP8_THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
#pragma unroll
  for (int chunk = 0; chunk < MXFP8_CHUNKS_PER_BLOCK; ++chunk) {
    const int chunk_Y = chunk / MXFP8_CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % MXFP8_CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * MXFP8_CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * MXFP8_CHUNK_DIM_X;

    const int scales_rowwise_chunk_offset_Y =
        scales_rowwise_block_offset_Y + chunk_Y * SCALES_ROWWISE_PER_CHUNK_Y;
    const int scales_rowwise_chunk_offset_X =
        scales_rowwise_block_offset_X + chunk_X * SCALES_ROWWISE_PER_CHUNK_X;
    const int scales_colwise_chunk_offset_Y =
        scales_colwise_block_offset_Y + chunk_Y * SCALES_COLWISE_PER_CHUNK_Y;
    const int scales_colwise_chunk_offset_X =
        scales_colwise_block_offset_X + chunk_X * SCALES_COLWISE_PER_CHUNK_X;

    if (is_master_thread) {
#pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < MXFP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        const int chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * MXFP8_BUFFER_DIM_Y;
        const int chunk_stage_offset_X = chunk_offset_X;
        // Initiate bulk tensor copy
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[prefetch_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
            chunk_stage_offset_Y, &mbar[prefetch_buff]);

        if constexpr (IS_DACT) {
          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&act_in_sh[prefetch_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_stage_offset_X,
              chunk_stage_offset_Y, &mbar[prefetch_buff]);
        }

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[prefetch_buff], transaction_size);
      }
    } else {
// Other threads just arrive
#pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < MXFP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        ptx::mbarrier_arrive(&mbar[prefetch_buff]);
      }
    }

#pragma unroll
    for (int iter = 0; iter < MXFP8_ITERATIONS; ++iter) {
      const int buff = iter % MXFP8_BUFFERS_NUM;
      const int next_iter = iter + MXFP8_PREFETCH_BUFFERS_NUM;
      if (next_iter < MXFP8_ITERATIONS) {
        if (is_master_thread) {
          const int next_buff = next_iter % MXFP8_BUFFERS_NUM;
          const int chunk_it_offset_y = chunk_offset_Y + next_iter * MXFP8_BUFFER_DIM_Y;
          const int chunk_it_offset_x = chunk_offset_X;
          // Initiate bulk tensor copy
          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
              chunk_it_offset_y, &mbar[next_iter]);

          if constexpr (IS_DACT) {
            ptx::cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t *>(&act_in_sh[next_buff]),
                reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_it_offset_x,
                chunk_it_offset_y, &mbar[next_iter]);
          }

          // Arrive on the barrier and tell how many bytes are expected to come in.
          ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size);
        } else {
          // Other threads just arrive
          ptx::mbarrier_arrive(&mbar[next_iter]);
        }
      }

      ptx::fence_proxy_async_shared_cta();

      // Wait for the data to have arrived
      ptx::mbarrier_wait_parity(&mbar[iter], parity);

      if constexpr (USE_ROWWISE_SCALING) {
        Vec<IType, ELEMS_PER_THREAD> in;
        Vec<IType, ELEMS_PER_THREAD> act_in;
        Vec<OType, ELEMS_PER_THREAD> out_c;

        const int iteration_scale_rowwise_offset_Y =
            scales_rowwise_chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;

#pragma unroll
        for (int stage = 0; stage < MXFP8_BUFF_STAGES_NUM; ++stage) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y_ROWWISE;
          const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
          const int shmem_offset_x = thread_offset_X_rowwise;
          in.load_from(&in_sh[buff][shmem_offset_y][shmem_offset_x]);
          if constexpr (IS_DACT) {
            act_in.load_from(&act_in_sh[buff][shmem_offset_y][shmem_offset_x]);
          }

          float thread_amax = 0;
          float in_compute[ELEMS_PER_THREAD];

#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
            float elt = static_cast<float>(in.data.elt[j]);
            if constexpr (IS_DACT) {
              elt *= OP(static_cast<float>(act_in.data.elt[j]), {});
            }
            if constexpr (IS_DBIAS && COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
              partial_dbias_rowwise[chunk_X].data.elt[j] += elt;
            }
            in_compute[j] = elt;
            if (isfinite(elt)) {
              thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
          }

          __builtin_assume(block_amax >= 0);
          __builtin_assume(thread_amax >= 0);
          block_amax = fmaxf(block_amax, thread_amax);

          const float subwarp_amax = subwarp_reduce_max_broadcast<SUBWARP_WIDTH>(thread_amax);
          const e8m0_t biased_exponent =
              float_to_e8m0(subwarp_amax * Quantized_Limits<OType>::max_norm_rcp);

          // Only single thread writes the computed scaling factor
          if (tid_rowwise_X % THREADS_PER_SCALE_X_ROWWISE == 0) {
            const int global_scales_offset_Y =
                iteration_scale_rowwise_offset_Y + stage_offset_Y + tid_rowwise_Y;
            const int global_scales_offset_X =
                scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE;
            const int scale_idx =
                global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
            scales_rowwise[scale_idx] = biased_exponent;
          }

          const float block_scale_inverse = exp2f_rcp(biased_exponent);

#pragma unroll
          for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
            out_c.data.elt[j] = static_cast<OType>(in_compute[j] * block_scale_inverse);
          }
          out_c.store_to(&out_rowwise_sh[buff][shmem_offset_y][shmem_offset_x]);
        }
      }

      if constexpr (USE_COLWISE_SCALING) {
        float in_compute[SCALE_DIM_Y];

        float amax = 0;
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          float elt = static_cast<float>(in_sh[buff][i][tid_colwise_X]);
          if constexpr (IS_DACT) {
            elt *= OP(static_cast<float>(act_in_sh[buff][i][tid_colwise_X]), {});
          }
          if constexpr (IS_DBIAS) {
            partial_dbias_colwise[chunk_X] += elt;
          }
          in_compute[i] = elt;
          if (isfinite(elt)) {
            amax = fmaxf(amax, fabsf(elt));
          }
        }

        __builtin_assume(block_amax >= 0);
        __builtin_assume(amax >= 0);
        block_amax = fmaxf(block_amax, amax);

        const e8m0_t biased_exponent = float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);

        const int global_scales_offset_Y = scales_colwise_chunk_offset_Y + iter;
        const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X;
        const int scale_idx =
            global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
        scales_colwise[scale_idx] = biased_exponent;

        const float block_scale_inverse = exp2f_rcp(biased_exponent);
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          out_colwise_sh[buff][i][tid_colwise_X] =
              static_cast<OType>(in_compute[i] * block_scale_inverse);
        }
      }

      // Wait for shared memory writes to be visible to TMA engine.
      ptx::fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (is_master_thread) {
        const int chunk_it_offset_y = chunk_offset_Y + iter * MXFP8_BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        if constexpr (USE_ROWWISE_SCALING) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), chunk_it_offset_x,
              chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_rowwise_sh[buff]));
        }
        if constexpr (USE_COLWISE_SCALING) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), chunk_it_offset_x,
              chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_colwise_sh[buff]));
        }
        // Create a "bulk async-group" out of the previous bulk copy operation.
        ptx::cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        ptx::cp_async_bulk_wait_group_read<MXFP8_PREFETCH_BUFFERS_NUM>();
      }
    }
    ptx::cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    parity ^= 1;
  }

  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
      constexpr size_t CZ = MXFP8_CHUNKS_PER_BLOCK_X;
      constexpr size_t Y = THREADS_PER_CHUNK_Y_ROWWISE - 1;
      constexpr size_t X = THREADS_PER_CHUNK_X_ROWWISE;
      __shared__ float shmem_partial_dbias_rowwise[CZ][Y][X][ELEMS_PER_THREAD];

      if (tid_rowwise_Y > 0) {
#pragma unroll
        for (int c = 0; c < MXFP8_CHUNKS_PER_BLOCK_X; ++c) {
          partial_dbias_rowwise[c].store_to(
              &shmem_partial_dbias_rowwise[c][tid_rowwise_Y - 1][tid_rowwise_X]);
        }
      }
      __syncthreads();

      if (tid_rowwise_Y == 0) {
#pragma unroll
        for (int c = 0; c < MXFP8_CHUNKS_PER_BLOCK_X; ++c) {
          Vec<float, ELEMS_PER_THREAD> other_row_dbias;
#pragma unroll
          for (int i = 0; i < Y; ++i) {
            other_row_dbias.load_from(&shmem_partial_dbias_rowwise[c][i][tid_rowwise_X]);
#pragma unroll
            for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
              partial_dbias_rowwise[c].data.elt[j] += other_row_dbias.data.elt[j];
            }
          }
          const int dbias_rowwise_offset_X = dbias_rowwise_block_offset_X + c * MXFP8_CHUNK_DIM_X;
          const int dbias_offset = dbias_rowwise_offset_Y * dbias_stride + dbias_rowwise_offset_X;
          partial_dbias_rowwise[c].store_to(&dbias_workspace[dbias_offset]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < MXFP8_CHUNKS_PER_BLOCK_X; ++i) {
        const int dbias_colwise_offset_X = dbias_colwise_block_offset_X + i * MXFP8_CHUNK_DIM_X;
        const int dbias_offset = dbias_colwise_offset_Y * dbias_stride + dbias_colwise_offset_X;
        dbias_workspace[dbias_offset] = partial_dbias_colwise[i];
      }
    }
  }

  if (amax_ptr_rowwise != nullptr || amax_ptr_colwise != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    block_amax = reduce_max<MXFP8_THREADS_PER_CHUNK / THREADS_PER_WARP>(block_amax, warp_id);
  }

  if (is_master_thread && amax_ptr_rowwise != nullptr) {
    atomicMaxFloat(amax_ptr_rowwise, block_amax);
  }

  if (is_master_thread && amax_ptr_colwise != nullptr) {
    atomicMaxFloat(amax_ptr_colwise, block_amax);
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < MXFP8_ITERATIONS; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

constexpr size_t FP8_CHUNK_DIM_Y = 128;
constexpr size_t FP8_CHUNK_DIM_X = 128;
constexpr size_t FP8_THREADS_PER_CHUNK = 128;
constexpr size_t FP8_BUFFERS_NUM = 2;
constexpr size_t FP8_PREFETCH_BUFFERS_NUM = 1;
static_assert(FP8_PREFETCH_BUFFERS_NUM < FP8_BUFFERS_NUM);

constexpr size_t FP8_BUFFER_DIM_Y = 16;
constexpr size_t FP8_BUFFER_DIM_X = FP8_CHUNK_DIM_X;  // 128
constexpr size_t FP8_SHMEM_DIM_Y = FP8_BUFFER_DIM_Y;  // 16
constexpr size_t FP8_SHMEM_DIM_X = FP8_BUFFER_DIM_X;  // 128

constexpr size_t FP8_BUFF_STAGES_NUM = FP8_BUFFER_DIM_Y;               //  16
constexpr size_t FP8_ITERATIONS = FP8_CHUNK_DIM_Y / FP8_BUFFER_DIM_Y;  //   8 = 128 / 16
static_assert(FP8_ITERATIONS >= FP8_PREFETCH_BUFFERS_NUM);

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType>
__global__ void __launch_bounds__(FP8_THREADS_PER_CHUNK)
    cast_fp8_2D_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                       const __grid_constant__ CUtensorMap tensor_map_act_input,
                       const __grid_constant__ CUtensorMap tensor_map_output,
                       float *const dbias_workspace, float *const amax_ptr,
                       float *const scale_inv_ptr, const float *const scale_ptr, const size_t rows,
                       const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int block_offset_Y = blockIdx.y * FP8_CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * FP8_CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / FP8_THREADS_PER_CHUNK;
  const int tid_X = threadIdx.x % FP8_THREADS_PER_CHUNK;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  const int dbias_offset_Y = blockIdx.y + tid_Y;
  const int dbias_block_offset_X = blockIdx.x * FP8_CHUNK_DIM_X + thread_offset_X;
  const int dbias_stride = cols;

  float partial_dbias = 0.f;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(128) IType in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(128) IType act_in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(128) OType out_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / FP8_BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size * (IS_DACT ? 2 : 1);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[FP8_ITERATIONS];

  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < FP8_ITERATIONS; ++iter) {
      ptx::mbarrier_init(&mbar[iter], FP8_THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;

  const int chunk_offset_Y = block_offset_Y;
  const int chunk_offset_X = block_offset_X;

  if (is_master_thread) {
#pragma unroll
    for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
      const int chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * FP8_BUFFER_DIM_Y;
      const int chunk_stage_offset_X = chunk_offset_X;
      // Initiate bulk tensor copy
      ptx::cp_async_bulk_tensor_2d_global_to_shared(
          reinterpret_cast<uint64_t *>(&in_sh[prefetch_buff]),
          reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
          chunk_stage_offset_Y, &mbar[prefetch_buff]);

      if constexpr (IS_DACT) {
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&act_in_sh[prefetch_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_stage_offset_X,
            chunk_stage_offset_Y, &mbar[prefetch_buff]);
      }

      // Arrive on the barrier and tell how many bytes are expected to come in.
      ptx::mbarrier_arrive_expect_tx(&mbar[prefetch_buff], transaction_size);
    }
  } else {
// Other threads just arrive
#pragma unroll
    for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
      ptx::mbarrier_arrive(&mbar[prefetch_buff]);
    }
  }

#pragma unroll
  for (int iter = 0; iter < FP8_ITERATIONS; ++iter) {
    const int buff = iter % FP8_BUFFERS_NUM;
    const int next_iter = iter + FP8_PREFETCH_BUFFERS_NUM;
    if (next_iter < FP8_ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_iter % FP8_BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_iter * FP8_BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
            chunk_it_offset_y, &mbar[next_iter]);

        if constexpr (IS_DACT) {
          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&act_in_sh[next_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_it_offset_x,
              chunk_it_offset_y, &mbar[next_iter]);
        }

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_iter]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

#pragma unroll
    for (int stage = 0; stage < FP8_BUFF_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;

      float elt = static_cast<float>(in_sh[buff][shmem_offset_y][shmem_offset_x]);
      if constexpr (IS_DACT) {
        elt *= OP(static_cast<float>(act_in_sh[buff][shmem_offset_y][shmem_offset_x]), {});
      }
      if constexpr (IS_DBIAS) {
        partial_dbias += elt;
      }
      if (isfinite(elt)) {
        amax = fmaxf(amax, fabsf(elt));
      }
      out_sh[buff][shmem_offset_y][shmem_offset_x] = static_cast<OType>(elt * scale);
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * FP8_BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
          chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<FP8_PREFETCH_BUFFERS_NUM>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  parity ^= 1;

  if constexpr (IS_DBIAS) {
    const int dbias_offset_X = dbias_block_offset_X;
    const int dbias_offset = dbias_offset_Y * dbias_stride + dbias_offset_X;
    dbias_workspace[dbias_offset] = partial_dbias;
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<FP8_THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < FP8_ITERATIONS; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

constexpr size_t CHUNKS_PER_BLOCK = 128;
constexpr size_t THREADS_PER_BLOCK = FP8_THREADS_PER_CHUNK;
constexpr size_t CHUNK_SIZE = THREADS_PER_BLOCK;
constexpr size_t ELEMS_PER_BLOCK = CHUNKS_PER_BLOCK * CHUNK_SIZE;
constexpr size_t CHUNKS_PER_ITERATION = 32;
constexpr size_t SHMEM_DIM = CHUNKS_PER_ITERATION * CHUNK_SIZE;
constexpr size_t ITERATIONS = CHUNKS_PER_BLOCK / CHUNKS_PER_ITERATION;
constexpr size_t SHMEM_BUFFERS = 2;
static_assert(CHUNKS_PER_BLOCK % CHUNKS_PER_ITERATION == 0);

template <typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    cast_fp8_1D_kernel(const IType *input_ptr, OType *output_ptr, float *const amax_ptr,
                       float *const scale_inv_ptr, const float *const scale_ptr, const size_t N) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int block_offset = blockIdx.x * ELEMS_PER_BLOCK;
  const IType *input = input_ptr + block_offset;
  OType *output = output_ptr + block_offset;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(128) IType in_sh[SHMEM_BUFFERS][SHMEM_DIM];
  __shared__ alignas(128) OType out_sh[SHMEM_BUFFERS][SHMEM_DIM];

  constexpr int transaction_size_IN = sizeof(in_sh) / SHMEM_BUFFERS;
  constexpr int transaction_size_OUT = sizeof(out_sh) / SHMEM_BUFFERS;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_BLOCK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;

  constexpr int iter_zero = 0;
  constexpr int buff_zero = 0;

  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_1d_global_to_shared(reinterpret_cast<uint64_t *>(&in_sh[buff_zero]),
                                                  reinterpret_cast<const uint64_t *>(input),
                                                  transaction_size_IN, &mbar[iter_zero]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[iter_zero], transaction_size_IN);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[iter_zero]);
  }

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const int buff = iter % SHMEM_BUFFERS;
    const int it_offset = iter * SHMEM_DIM;

    const int next_iter = iter + 1;
    const int next_buff = next_iter % SHMEM_BUFFERS;
    const int next_iter_offset = next_iter * SHMEM_DIM;

    if (next_iter < ITERATIONS) {
      if (is_master_thread) {
        // Initiate bulk tensor copy
        ptx::cp_async_bulk_tensor_1d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
            reinterpret_cast<const uint64_t *>(input + next_iter_offset), transaction_size_IN,
            &mbar[next_iter]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size_IN);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_iter]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_ITERATION; ++chunk) {
      const int shmem_offset = chunk * CHUNK_SIZE + threadIdx.x;
      float elt = static_cast<float>(in_sh[buff][shmem_offset]);
      if (isfinite(elt)) {
        amax = fmaxf(amax, fabsf(elt));
      }
      out_sh[buff][shmem_offset] = static_cast<OType>(elt * scale);
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      ptx::cp_async_bulk_tensor_1d_shared_to_global(
          reinterpret_cast<uint64_t *>(output + it_offset),
          reinterpret_cast<uint64_t *>(&out_sh[buff]), transaction_size_OUT);

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

constexpr size_t DBIAS_THREADS_PER_BLOCK = 256;
template <int nvec, typename OType>
__global__ void __launch_bounds__(DBIAS_THREADS_PER_BLOCK)
    reduce_dbias_kernel(OType *const dbias_output, const float *const dbias_partial, const int rows,
                        const int cols) {
  using ComputeVec = Vec<float, nvec>;
  using OutputVec = Vec<OType, nvec>;

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id * nvec >= cols) {
    return;
  }

  const float *const thread_in_base = dbias_partial + thread_id * nvec;
  OType *const thread_out_base = dbias_output + thread_id * nvec;

  ComputeVec ldg_vec;
  ComputeVec acc_vec;
  acc_vec.clear();
  for (int i = 0; i < rows; ++i) {
    ldg_vec.load_from(thread_in_base + i * cols);
#pragma unroll
    for (int e = 0; e < nvec; ++e) {
      acc_vec.data.elt[e] += ldg_vec.data.elt[e];
    }
  }

  OutputVec stg_vec;
#pragma unroll
  for (int e = 0; e < nvec; ++e) {
    stg_vec.data.elt[e] = static_cast<OType>(acc_vec.data.elt[e]);
  }
  stg_vec.store_to(thread_out_base);
}

template <typename IType>
void reduce_dbias(const float *workspace_ptr, Tensor *dbias, const size_t rows, const size_t cols,
                  cudaStream_t stream) {
  constexpr int reduce_dbias_store_bytes = 8;  // stg.64
  constexpr int reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(IType);

  NVTE_CHECK(cols % reduce_dbias_nvec == 0, "Unsupported shape.");
  const size_t reduce_dbias_num_blocks = DIVUP(cols, DBIAS_THREADS_PER_BLOCK * reduce_dbias_nvec);

  reduce_dbias_kernel<reduce_dbias_nvec, IType>
      <<<reduce_dbias_num_blocks, DBIAS_THREADS_PER_BLOCK, 0, stream>>>(
          reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, rows, cols);
}

void cast_fp8_1D(const Tensor &input, Tensor *output, cudaStream_t stream) {
  const size_t N = product(input.data.shape);

  const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  const size_t chunks = DIVUP(N, CHUNK_SIZE);
  const size_t blocks = DIVUP(chunks, CHUNKS_PER_BLOCK);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  const float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(blocks);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          const IType *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
          OType *output_ptr = reinterpret_cast<OType *>(output->data.dptr);

          cast_fp8_1D_kernel<IType, OType><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, amax_ptr, scale_inv_ptr, scale_ptr, N););  // NOLINT(*)
  );                                                                            // NOLINT(*)
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void cast_fp8_2D(const Tensor &input, const Tensor &act_input, Tensor *output, Tensor *dbias,
                 Tensor *workspace, cudaStream_t stream) {
  checkCuDriverContext(stream);
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");

  const size_t rows = input.data.shape[0];
  const size_t cols = input.data.shape[1];
  const size_t chunks_Y = DIVUP(rows, FP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, FP8_CHUNK_DIM_X);
  const size_t blocks_Y = chunks_Y;
  const size_t blocks_X = chunks_X;

  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  const bool isFullTile = (rows % FP8_CHUNK_DIM_Y == 0) && (cols % FP8_CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.data.dtype, "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(FP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          alignas(64) CUtensorMap tensor_map_input{};
          alignas(64) CUtensorMap tensor_map_act_input{};
          alignas(64) CUtensorMap tensor_map_output{};

          create_2D_tensor_map<IType>(tensor_map_input, &input, rows, cols, FP8_SHMEM_DIM_Y,
                                      FP8_SHMEM_DIM_X);

          if constexpr (IS_DACT) {
            create_2D_tensor_map<IType>(tensor_map_act_input, &act_input, rows, cols,
                                        FP8_SHMEM_DIM_Y, FP8_SHMEM_DIM_X);
          }

          create_2D_tensor_map<OType>(tensor_map_output, output, rows, cols, FP8_SHMEM_DIM_Y,
                                      FP8_SHMEM_DIM_X);

          cast_fp8_2D_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType>
          <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_act_input, tensor_map_output,
                                       workspace_ptr, amax_ptr, scale_inv_ptr, scale_ptr, rows,
                                       cols);

          if constexpr (IS_DBIAS) {
            reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <bool IS_DBIAS, bool IS_DACT, ScalingType SCALING, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void cast_mxfp8(const Tensor &input, const Tensor &act_input, Tensor *output_rowwise,
                Tensor *output_colwise, Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  constexpr bool USE_ROWWISE_SCALING =
      (SCALING == ScalingType::BIDIMENTIONAL || SCALING == ScalingType::ROWWISE);
  constexpr bool USE_COLWISE_SCALING =
      (SCALING == ScalingType::BIDIMENTIONAL || SCALING == ScalingType::COLWISE);
  checkCuDriverContext(stream);
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");

  const size_t scale_dim_X_rowwise = USE_ROWWISE_SCALING ? output_rowwise->scaling_mode.y : 1;
  const size_t scale_dim_Y_colwise = USE_COLWISE_SCALING ? output_colwise->scaling_mode.x : 1;

  const size_t rows = input.data.shape[0];
  const size_t cols = input.data.shape[1];
  const size_t chunks_Y = DIVUP(rows, MXFP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, MXFP8_CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, MXFP8_CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, MXFP8_CHUNKS_PER_BLOCK_X);
  const size_t scale_stride_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t scale_stride_colwise = cols;

  const bool isFullTile = (rows % MXFP8_CHUNK_DIM_Y == 0) && (cols % MXFP8_CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  e8m0_t *const scales_rowwise_ptr =
      USE_ROWWISE_SCALING ? reinterpret_cast<e8m0_t *>(output_rowwise->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      USE_COLWISE_SCALING ? reinterpret_cast<e8m0_t *>(output_colwise->scale_inv.dptr) : nullptr;
  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.data.dtype, "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }

  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr_rowwise =
      USE_ROWWISE_SCALING ? reinterpret_cast<float *>(output_rowwise->amax.dptr) : nullptr;
  float *const amax_ptr_colwise =
      USE_COLWISE_SCALING ? reinterpret_cast<float *>(output_colwise->amax.dptr) : nullptr;

  const dim3 block(MXFP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  DType OutputType = USE_ROWWISE_SCALING ? output_rowwise->data.dtype : output_colwise->data.dtype;

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
              input.data.dtype, IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
                  OutputType, OType,

                  alignas(64) CUtensorMap tensor_map_input{};
                  alignas(64) CUtensorMap tensor_map_act_input{};
                  alignas(64) CUtensorMap tensor_map_output_rowwise{};
                  alignas(64) CUtensorMap tensor_map_output_colwise{};

                  create_2D_tensor_map<IType>(tensor_map_input, &input, rows, cols,
                                              MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X);

                  if constexpr (IS_DACT) {
                    create_2D_tensor_map<IType>(tensor_map_act_input, &act_input, rows, cols,
                                                MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X);
                  } if constexpr (USE_ROWWISE_SCALING) {
                    create_2D_tensor_map<OType>(tensor_map_output_rowwise, output_rowwise, rows,
                                                cols, MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X);
                  } if constexpr (USE_COLWISE_SCALING) {
                    create_2D_tensor_map<OType>(tensor_map_output_colwise, output_colwise, rows,
                                                cols, MXFP8_SHMEM_DIM_Y, MXFP8_SHMEM_DIM_X);
                  }

                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType, SCALE_DIM_Y,
                                       SCALE_DIM_X><<<grid, block, 0, stream>>>(
                      tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                      tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr,
                      workspace_ptr, amax_ptr_rowwise, amax_ptr_colwise, rows, cols,
                      scale_stride_rowwise, scale_stride_colwise);

                  if constexpr (IS_DBIAS) {
                    reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
                  });  // NOLINT(*)
          );           // NOLINT(*)
      );               // NOLINT(*)
  );                   // NOLINT(*)
}

namespace detail {

struct Empty {};

__device__ inline float identity(float value, const Empty &) { return value; }

struct DequantizeParam {
  const float *scale_inv;
};

__device__ inline float dequantize_func(float value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

// Supported by the Arch < 10.0
void CastVectorizedUnaryKernelLauncher(const Tensor &input, const Tensor &act_input, Tensor *output,
                                       Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          constexpr int nvec = 32 / sizeof(IType);
          VectorizedUnaryKernelLauncher<nvec, detail::Empty, detail::identity>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr),
              reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {},
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

// Supported by the Arch >= 10.0
template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize_arch_ge_100(const Tensor &input, const Tensor &act_input, Tensor *output,
                              Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  // MXFP8 Scaling
  if (is_mxfp8_cast_supported_shape(output)) {
    const bool is_colwise_scaling = (output->scaling_mode.x > 1);
    if (is_colwise_scaling) {
      cast_mxfp8<IS_DBIAS, IS_DACT, ScalingType::COLWISE, ParamOP, OP>(
          input, act_input, nullptr, output, dbias, workspace, stream);
    } else {
      cast_mxfp8<IS_DBIAS, IS_DACT, ScalingType::ROWWISE, ParamOP, OP>(
          input, act_input, output, nullptr, dbias, workspace, stream);
    }
    // Regular FP8 Scaling
  } else if (is_fp8_cast_supported_shape(output)) {
    if (!IS_DBIAS && !IS_DACT) {
      const size_t N = product(input.data.shape);
      const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
      // Aligned
      if (isFullTile) {
        cast_fp8_1D(input, output, stream);
      } else {
        // Unaligned
        CastVectorizedUnaryKernelLauncher(input, act_input, output, dbias, workspace, stream);
      }
    } else {
      cast_fp8_2D<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias, workspace,
                                                  stream);
    }
  } else {
    NVTE_ERROR("Not implemented on Arch >= 10.0: " + to_string(output->scaling_mode) + ".");
  }
}

// Supported by the Arch < 10.0
template <bool IS_DBIAS, bool IS_DACT>
void fp8_quantize_arch_l_100(const Tensor &input, const Tensor &act_input, Tensor *output,
                             Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  if (is_delayed_tensor_scaling(output->scaling_mode) && !IS_DBIAS && !IS_DACT) {
    CastVectorizedUnaryKernelLauncher(input, act_input, output, dbias, workspace, stream);
  } else {
    NVTE_ERROR("Not implemented on Arch < 10.0: " + to_string(output->scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize(const Tensor &input, const Tensor &act_input, Tensor *output, Tensor *dbias,
                  Tensor *workspace, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  if constexpr (IS_DBIAS) {
    CheckOutputTensor(*dbias, "dbias");
  }
  if constexpr (IS_DACT) {
    CheckInputTensor(act_input, "activation_input");
    NVTE_CHECK(input.data.dtype == act_input.data.dtype, "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input.data.shape, "Shapes of both inputs must match.");
  }

  NVTE_CHECK(!is_fp8_dtype(input.data.dtype), "Input must be in higher precision.");
  NVTE_CHECK(is_fp8_dtype(output->data.dtype), "Output must have FP8 type.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  // Supported by the Arch >= 10.0
  if (is_supported_by_CC_100()) {
    fp8_quantize_arch_ge_100<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias,
                                                             workspace, stream);
  } else {
    // Supported by the Arch < 10.0
    fp8_quantize_arch_l_100<IS_DBIAS, IS_DACT>(input, act_input, output, dbias, workspace, stream);
  }
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize_x2(const Tensor &input, const Tensor &act_input, Tensor *output_rowwise,
                     Tensor *output_colwise, Tensor *dbias, Tensor *workspace,
                     cudaStream_t stream) {
  if (!is_supported_by_CC_100()) {
    NVTE_ERROR("Not supported by Arch < 10.0");
  }

  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output_rowwise, "cast_output_rowwise");
  CheckOutputTensor(*output_colwise, "cast_output_colwise");

  if constexpr (IS_DBIAS) {
    CheckOutputTensor(*dbias, "dbias");
  }
  if constexpr (IS_DACT) {
    CheckInputTensor(act_input, "activation_input");
    NVTE_CHECK(input.data.dtype == act_input.data.dtype, "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input.data.shape, "Shapes of both inputs must match.");
  }

  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(!is_fp8_dtype(input.data.dtype), "Input must be in higher precision.");

  NVTE_CHECK(output_rowwise->data.dtype == output_colwise->data.dtype,
             "Rowwise and colwise outputs must have the same output type.");
  NVTE_CHECK(is_fp8_dtype(output_rowwise->data.dtype), "Rowwise output must have FP8 type.");
  NVTE_CHECK(is_fp8_dtype(output_colwise->data.dtype), "Colwise output must have FP8 type.");
  NVTE_CHECK(output_rowwise->data.shape == input.data.shape,
             "Input and rowwise output shapes need to match.");
  NVTE_CHECK(output_colwise->data.shape == input.data.shape,
             "Input and colwise output shapes need to match.");

  NVTE_CHECK(output_rowwise->data.shape.size() == 2, "Cast output rowwise must have 2 dimensions.");
  NVTE_CHECK(output_colwise->data.shape.size() == 2, "Cast output colwise must have 2 dimensions.");

  NVTE_CHECK(output_rowwise->scale_inv.dptr != nullptr, "Rowwise scaling tensor must be allocated");
  NVTE_CHECK(output_colwise->scale_inv.dptr != nullptr, "Colwise scaling tensor must be allocated");

  if (is_mxfp8_cast_supported_shape(output_rowwise) &&
      is_mxfp8_cast_supported_shape(output_colwise)) {
    cast_mxfp8<IS_DBIAS, IS_DACT, ScalingType::BIDIMENTIONAL, ParamOP, OP>(
        input, act_input, output_rowwise, output_colwise, dbias, workspace, stream);
  } else {
    NVTE_ERROR("Not implemented 2x scaling mode: " + to_string(output_rowwise->scaling_mode) +
               to_string(output_colwise->scaling_mode) + ".");
  }
}

void fp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  if (is_tensor_scaling(input.scaling_mode)) {
    const size_t N = product(input.data.shape);
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        input.data.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
            output->data.dtype, OType, constexpr int nvec = 32 / sizeof(OType);
            detail::DequantizeParam p;
            p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
            VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, nullptr, N, p,
                stream););  // NOLINT(*)
    );                      // NOLINT(*)
  } else {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

}  // namespace transformer_engine

void nvte_fp8_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;
  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                             NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dgelu(const NVTETensor input, const NVTETensor activation_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dgelu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dsilu(const NVTETensor input, const NVTETensor activation_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dsilu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dsilu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_drelu(const NVTETensor input, const NVTETensor activation_input,
                                   NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_drelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &drelu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dqgelu(const NVTETensor input, const NVTETensor activation_input,
                                    NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dqgelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dqgelu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dsrelu(const NVTETensor input, const NVTETensor activation_input,
                                    NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dsrelu);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dsrelu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(dbias),
      reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_x2(const NVTETensor input, NVTETensor output_rowwise,
                          NVTETensor output_colwise, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;

  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;
  constexpr const NVTETensor activation_input = nullptr;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_x2(const NVTETensor input, NVTETensor output_rowwise,
                                NVTETensor output_colwise, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr const NVTETensor activation_input = nullptr;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dgelu_x2(const NVTETensor input, const NVTETensor activation_input,
                                      NVTETensor output_rowwise, NVTETensor output_colwise,
                                      NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dgelu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dgelu<fp32, fp32>;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dsilu_x2(const NVTETensor input, const NVTETensor activation_input,
                                      NVTETensor output_rowwise, NVTETensor output_colwise,
                                      NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dsilu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dsilu<fp32, fp32>;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_drelu_x2(const NVTETensor input, const NVTETensor activation_input,
                                      NVTETensor output_rowwise, NVTETensor output_colwise,
                                      NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_drelu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &drelu<fp32, fp32>;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dqgelu_x2(const NVTETensor input, const NVTETensor activation_input,
                                       NVTETensor output_rowwise, NVTETensor output_colwise,
                                       NVTETensor dbias, NVTETensor workspace,
                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dqgelu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dqgelu<fp32, fp32>;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dsrelu_x2(const NVTETensor input, const NVTETensor activation_input,
                                       NVTETensor output_rowwise, NVTETensor output_colwise,
                                       NVTETensor dbias, NVTETensor workspace,
                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dsrelu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr auto dActivation = &dsrelu<fp32, fp32>;

  fp8_quantize_x2<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_dequantize);
  using namespace transformer_engine;
  fp8_dequantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
                 stream);
}
