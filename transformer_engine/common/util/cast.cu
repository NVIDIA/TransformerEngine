/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_map>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {

using namespace ptx;

constexpr uint32_t FP32_EXPONENT_BIAS = 127;
constexpr uint32_t FP32_EXPONENT_BITS = 8;
constexpr uint32_t FP32_MANTISSA_BITS = 23;  // FP32 = [S1] [E8] [M23]
constexpr uint32_t SIGN_MASK =
    1U << (FP32_MANTISSA_BITS + FP32_EXPONENT_BITS);  // most significant bit mask
constexpr uint32_t NUMBER_MASK = ~SIGN_MASK;
constexpr uint32_t MANTISSA_MASK = (1U << FP32_MANTISSA_BITS) - 1;
constexpr uint32_t EXPONENT_MASK = NUMBER_MASK & (~MANTISSA_MASK);

template <typename T>
struct Numeric_Traits;

template <>
struct Numeric_Traits<fp8e4m3> {
  static constexpr int maxUnbiasedExponent = 8;
};

template <>
struct Numeric_Traits<fp8e5m2> {
  static constexpr int maxUnbiasedExponent = 15;
};

template <typename T>
struct Quantized_Limits {
  static constexpr inline int max_norm_unbiased_exponent() {
    return Numeric_Traits<T>::maxUnbiasedExponent;
  }
};

constexpr size_t CHUNK_DIM_Y = 64;
constexpr size_t CHUNK_DIM_X = 64;
constexpr size_t CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t CHUNKS_PER_BLOCK_X = 1;
constexpr size_t CHUNKS_PER_BLOCK = CHUNKS_PER_BLOCK_Y * CHUNKS_PER_BLOCK_X;
constexpr size_t THREADS_PER_CHUNK = 64;
constexpr size_t DBIAS_THREADS_PER_BLOCK = 256;
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t PREFETCH_BUFFERS_NUM = 1;
static_assert(PREFETCH_BUFFERS_NUM < BUFFERS_NUM);

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t BUFFER_DIM_Y = 32;           // only 32 is supported
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 64
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 64

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  //   4 = 64 / 16
constexpr size_t THREADS_PER_CHUNK_Y_ROWWISE =
    THREADS_PER_CHUNK / THREADS_PER_CHUNK_X_ROWWISE;                            //  16 = 64 / 4
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;                     //  64
constexpr size_t BUFF_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y_ROWWISE;  //   2 = 32 / 16
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                       //   2 = 64 / 32
static_assert(ITERATIONS >= PREFETCH_BUFFERS_NUM);

using e8m0_t = uint8_t;

__device__ __forceinline__ int32_t extract_biased_exponent(const float val) {
  return (__float_as_int(val) & EXPONENT_MASK) >> FP32_MANTISSA_BITS;
}

__device__ __forceinline__ int lower_biased_exp_limit_fp32(const int biased_exponent) {
  return (biased_exponent < 0) ? 0 : biased_exponent;
}

template <typename OType>
__device__ __forceinline__ e8m0_t compute_shared_biased_exponent(float amax) {
  __builtin_assume(amax >= 0);
  if (amax == 0) {
    constexpr int exponent_ = 0 + FP32_EXPONENT_BIAS;
    return static_cast<e8m0_t>(exponent_);
  }
  int exponent =
      extract_biased_exponent(amax) - Quantized_Limits<OType>::max_norm_unbiased_exponent();

  // Clamp the shared unbiased exponent between the representable numbers of uint8_t
  // i.e., between [0, 255]
  return static_cast<e8m0_t>(lower_biased_exp_limit_fp32(exponent));
}

/**
 * Max reduction in subwarps
 * E.g., if nvec=4, each warp processes 128 elements (32 x 4), that covers four MXFP8 scaling factors.
 * To compute an actual scaling factor for 32 consequentive elements, only 8 threads need to participate,
 * thus splitting the warp into 4x smaller subwarps 8-thread width. 'Butterfly' reduction is implemented
 * inside those subwarps.
 */
template <int subwarp_width>
__forceinline__ __device__ float subwarp_max_reduction(const float val) {
  float val_tmp = val;
#pragma unroll
  for (int offset = subwarp_width / 2; offset > 0; offset /= 2) {
    const float val_other = __shfl_down_sync(0xFFFFFFFF, val_tmp, offset, subwarp_width);
    __builtin_assume(val_tmp >= 0);
    __builtin_assume(val_other >= 0);
    val_tmp = fmaxf(val_tmp, val_other);
  }
  // Broadcast the amax to other threads of the subwarp from the zero subwarp lane_id
  constexpr int subwarp_lane_zero = 0;
  val_tmp = __shfl_sync(0xFFFFFFFF, val_tmp, subwarp_lane_zero, subwarp_width);
  return val_tmp;
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_mxfp8_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                      const __grid_constant__ CUtensorMap tensor_map_act_input,
                      const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                      const __grid_constant__ CUtensorMap tensor_map_output_colwise,
                      e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                      float *const dbias_workspace, const size_t rows, const size_t cols,
                      const size_t scale_stride_rowwise, const size_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_DBIAS_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //   2 = 64 / 32
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //  64 = 64 / 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_Y =
      SCALES_ROWWISE_PER_CHUNK_Y * CHUNKS_PER_BLOCK_Y;  //   2 = 2 * 1
  constexpr size_t SCALES_ROWWISE_PER_BLOCK_X =
      SCALES_ROWWISE_PER_CHUNK_X * CHUNKS_PER_BLOCK_X;  //  64 = 64 * 1

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //   2 = 64 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  64 = 64 / 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_Y =
      SCALES_COLWISE_PER_CHUNK_Y * CHUNKS_PER_BLOCK_Y;  //   2 = 2 * 1
  constexpr size_t SCALES_COLWISE_PER_BLOCK_X =
      SCALES_COLWISE_PER_CHUNK_X * CHUNKS_PER_BLOCK_X;  //  64 = 64 * 1

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);                      //   2 = 32 / 16
  constexpr size_t SUBWARP_WIDTH = THREADS_PER_SCALE_X_ROWWISE;  //   2

  const int block_offset_Y = blockIdx.y * CHUNKS_PER_BLOCK_Y * CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * CHUNKS_PER_BLOCK_X * CHUNK_DIM_X;
  const int scales_rowwise_block_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_BLOCK_Y;
  const int scales_rowwise_block_offset_X = blockIdx.x * SCALES_ROWWISE_PER_BLOCK_X;
  const int scales_colwise_block_offset_Y = blockIdx.y * SCALES_COLWISE_PER_BLOCK_Y;
  const int scales_colwise_block_offset_X = blockIdx.x * SCALES_COLWISE_PER_BLOCK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  const int dbias_rowwise_offset_Y =
      blockIdx.y * CHUNKS_PER_BLOCK_Y * THREADS_PER_CHUNK_Y_ROWWISE + tid_rowwise_Y;
  const int dbias_rowwise_block_offset_X =
      blockIdx.x * CHUNKS_PER_BLOCK_X * CHUNK_DIM_X + thread_offset_X_rowwise;
  const int dbias_colwise_offset_Y = blockIdx.y;
  const int dbias_colwise_block_offset_X =
      blockIdx.x * CHUNKS_PER_BLOCK_X * CHUNK_DIM_X + tid_colwise_X;
  const int dbias_stride = cols;

  Vec<float, ELEMS_PER_THREAD> partial_dbias_rowwise[CHUNKS_PER_BLOCK_X];
  float partial_dbias_colwise[CHUNKS_PER_BLOCK_X];
  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
#pragma unroll
      for (int i = 0; i < CHUNKS_PER_BLOCK_X; ++i) {
        partial_dbias_rowwise[i].clear();
      }
    } else {
#pragma unroll
      for (int i = 0; i < CHUNKS_PER_BLOCK_X; ++i) {
        partial_dbias_colwise[i] = 0;
      }
    }
  }

  // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
  __shared__ alignas(16) IType in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(16) IType act_in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(16) OType out_rowwise_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(16) OType out_colwise_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size * (IS_DACT ? 2 : 1);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      mbarrier_init(&mbar[it], THREADS_PER_CHUNK);
    }
    fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
#pragma unroll
  for (int chunk = 0; chunk < CHUNKS_PER_BLOCK; ++chunk) {
    const int chunk_Y = chunk / CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * CHUNK_DIM_X;

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
      for (int prefetch_buff = 0; prefetch_buff < PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        const int chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * BUFFER_DIM_Y;
        const int chunk_stage_offset_X = chunk_offset_X;
        // Initiate bulk tensor copy
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[prefetch_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
            chunk_stage_offset_Y, &mbar[prefetch_buff]);

        if constexpr (IS_DACT) {
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&act_in_sh[prefetch_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_stage_offset_X,
              chunk_stage_offset_Y, &mbar[prefetch_buff]);
        }

        // Arrive on the barrier and tell how many bytes are expected to come in.
        mbarrier_arrive_expect_tx(&mbar[prefetch_buff], transaction_size);
      }
    } else {
// Other threads just arrive
#pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        mbarrier_arrive(&mbar[prefetch_buff]);
      }
    }

#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      const int buff = it % BUFFERS_NUM;
      const int next_it = it + PREFETCH_BUFFERS_NUM;
      if (next_it < ITERATIONS) {
        if (is_master_thread) {
          const int next_buff = next_it % BUFFERS_NUM;
          const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
          const int chunk_it_offset_x = chunk_offset_X;
          // Initiate bulk tensor copy
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
              chunk_it_offset_y, &mbar[next_it]);

          if constexpr (IS_DACT) {
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t *>(&act_in_sh[next_buff]),
                reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_it_offset_x,
                chunk_it_offset_y, &mbar[next_it]);
          }

          // Arrive on the barrier and tell how many bytes are expected to come in.
          mbarrier_arrive_expect_tx(&mbar[next_it], transaction_size);
        } else {
          // Other threads just arrive
          mbarrier_arrive(&mbar[next_it]);
        }
      }

      // Wait for the data to have arrived
      mbarrier_wait_parity(&mbar[it], parity);

      if constexpr (USE_ROWWISE_SCALING) {
        Vec<IType, ELEMS_PER_THREAD> in;
        Vec<IType, ELEMS_PER_THREAD> act_in;
        Vec<OType, ELEMS_PER_THREAD> out_c;

        const int iteration_scale_rowwise_offset_Y =
            scales_rowwise_chunk_offset_Y + it * BUFFER_DIM_Y;

#pragma unroll
        for (int stage = 0; stage < BUFF_STAGES_NUM; ++stage) {
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

          const float subwarp_amax = subwarp_max_reduction<SUBWARP_WIDTH>(thread_amax);
          const e8m0_t biased_exponent = compute_shared_biased_exponent<OType>(subwarp_amax);

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

          const float block_scale_inverse =
              exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exponent));
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
        const e8m0_t biased_exponent = compute_shared_biased_exponent<OType>(amax);

        const int global_scales_offset_Y = scales_colwise_chunk_offset_Y + it;
        const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_colwise_X;
        const int scale_idx =
            global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
        scales_colwise[scale_idx] = biased_exponent;

        const float block_scale_inverse =
            exp2f(FP32_EXPONENT_BIAS - static_cast<float>(biased_exponent));
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          out_colwise_sh[buff][i][tid_colwise_X] =
              static_cast<OType>(in_compute[i] * block_scale_inverse);
        }
      }

      // Wait for shared memory writes to be visible to TMA engine.
      fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (is_master_thread) {
        const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        if constexpr (USE_ROWWISE_SCALING) {
          cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), chunk_it_offset_x,
              chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_rowwise_sh[buff]));
        }
        if constexpr (USE_COLWISE_SCALING) {
          cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), chunk_it_offset_x,
              chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_colwise_sh[buff]));
        }
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        cp_async_bulk_wait_group_read<PREFETCH_BUFFERS_NUM>();
      }
    }
    cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    parity ^= 1;
  }

  if constexpr (IS_DBIAS) {
    if constexpr (COMPUTE_DBIAS_IN_ROWWISE_SECTION) {
#pragma unroll
      for (int i = 0; i < CHUNKS_PER_BLOCK_X; ++i) {
        const int dbias_rowwise_offset_X = dbias_rowwise_block_offset_X + i * CHUNK_DIM_X;
        const int dbias_offset = dbias_rowwise_offset_Y * dbias_stride + dbias_rowwise_offset_X;
        partial_dbias_rowwise[i].store_to(&dbias_workspace[dbias_offset]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < CHUNKS_PER_BLOCK_X; ++i) {
        const int dbias_colwise_offset_X = dbias_colwise_block_offset_X + i * CHUNK_DIM_X;
        const int dbias_offset = dbias_colwise_offset_Y * dbias_stride + dbias_colwise_offset_X;
        dbias_workspace[dbias_offset] = partial_dbias_colwise[i];
      }
    }
  }

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

constexpr size_t FP8_CHUNK_DIM_Y = 128;
constexpr size_t FP8_CHUNK_DIM_X = 128;
constexpr size_t FP8_CHUNKS_PER_BLOCK_Y = 1;
constexpr size_t FP8_CHUNKS_PER_BLOCK_X = 1;
constexpr size_t FP8_CHUNKS_PER_BLOCK = FP8_CHUNKS_PER_BLOCK_Y * FP8_CHUNKS_PER_BLOCK_X;
constexpr size_t FP8_THREADS_PER_CHUNK = 128;
constexpr size_t FP8_BUFFERS_NUM = 2;
constexpr size_t FP8_PREFETCH_BUFFERS_NUM = 1;
static_assert(FP8_PREFETCH_BUFFERS_NUM < FP8_BUFFERS_NUM);

constexpr size_t FP8_BUFFER_DIM_Y = 16;
constexpr size_t FP8_BUFFER_DIM_X = FP8_CHUNK_DIM_X;  // 128
constexpr size_t FP8_SHMEM_DIM_Y = FP8_BUFFER_DIM_Y;  // 16
constexpr size_t FP8_SHMEM_DIM_X = FP8_BUFFER_DIM_X;  // 128

constexpr size_t FP8_BUFF_STAGES_NUM = FP8_BUFFER_DIM_Y;                //  16
constexpr size_t FP8_ITERATIONS = FP8_CHUNK_DIM_Y / FP8_BUFFER_DIM_Y;   //   8 = 128 / 16
static_assert(FP8_ITERATIONS >= FP8_PREFETCH_BUFFERS_NUM);

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType>
__global__ void __launch_bounds__(FP8_THREADS_PER_CHUNK)
    cast_fp8_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                    const __grid_constant__ CUtensorMap tensor_map_act_input,
                    const __grid_constant__ CUtensorMap tensor_map_output, const fp32 *noop,
                    float *const dbias_workspace, float *const amax_ptr, float *const scale_inv_ptr,
                    const float *const scale_ptr, const size_t rows, const size_t cols) {
  if (noop != nullptr && noop[0] == 1) return;
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int block_offset_Y = blockIdx.y * FP8_CHUNKS_PER_BLOCK_Y * FP8_CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * FP8_CHUNKS_PER_BLOCK_X * FP8_CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / FP8_THREADS_PER_CHUNK;
  const int tid_X = threadIdx.x % FP8_THREADS_PER_CHUNK;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  const int dbias_offset_Y = blockIdx.y * FP8_CHUNKS_PER_BLOCK_Y + tid_Y;
  const int dbias_block_offset_X =
      blockIdx.x * FP8_CHUNKS_PER_BLOCK_X * FP8_CHUNK_DIM_X + thread_offset_X;
  const int dbias_stride = cols;

  float partial_dbias[FP8_CHUNKS_PER_BLOCK_X];

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(16) IType in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(16) IType act_in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(16) OType out_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / FP8_BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size * (IS_DACT ? 2 : 1);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[FP8_ITERATIONS];

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int it = 0; it < FP8_ITERATIONS; ++it) {
      mbarrier_init(&mbar[it], FP8_THREADS_PER_CHUNK);
    }
    fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
#pragma unroll
  for (int chunk = 0; chunk < FP8_CHUNKS_PER_BLOCK; ++chunk) {
    const int chunk_Y = chunk / FP8_CHUNKS_PER_BLOCK_X;
    const int chunk_X = chunk % FP8_CHUNKS_PER_BLOCK_X;

    const int chunk_offset_Y = block_offset_Y + chunk_Y * FP8_CHUNK_DIM_Y;
    const int chunk_offset_X = block_offset_X + chunk_X * FP8_CHUNK_DIM_X;

    if (is_master_thread) {
#pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        const int chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * FP8_BUFFER_DIM_Y;
        const int chunk_stage_offset_X = chunk_offset_X;
        // Initiate bulk tensor copy
        cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[prefetch_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
            chunk_stage_offset_Y, &mbar[prefetch_buff]);

        if constexpr (IS_DACT) {
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&act_in_sh[prefetch_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_stage_offset_X,
              chunk_stage_offset_Y, &mbar[prefetch_buff]);
        }

        // Arrive on the barrier and tell how many bytes are expected to come in.
        mbarrier_arrive_expect_tx(&mbar[prefetch_buff], transaction_size);
      }
    } else {
// Other threads just arrive
#pragma unroll
      for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
        mbarrier_arrive(&mbar[prefetch_buff]);
      }
    }

#pragma unroll 2
    for (int it = 0; it < FP8_ITERATIONS; ++it) {
      const int buff = it % FP8_BUFFERS_NUM;
      const int next_it = it + FP8_PREFETCH_BUFFERS_NUM;
      if (next_it < FP8_ITERATIONS) {
        if (is_master_thread) {
          const int next_buff = next_it % FP8_BUFFERS_NUM;
          const int chunk_it_offset_y = chunk_offset_Y + next_it * FP8_BUFFER_DIM_Y;
          const int chunk_it_offset_x = chunk_offset_X;
          // Initiate bulk tensor copy
          cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
              reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
              chunk_it_offset_y, &mbar[next_it]);

          if constexpr (IS_DACT) {
            cp_async_bulk_tensor_2d_global_to_shared(
                reinterpret_cast<uint64_t *>(&act_in_sh[next_buff]),
                reinterpret_cast<const uint64_t *>(&tensor_map_act_input), chunk_it_offset_x,
                chunk_it_offset_y, &mbar[next_it]);
          }

          // Arrive on the barrier and tell how many bytes are expected to come in.
          mbarrier_arrive_expect_tx(&mbar[next_it], transaction_size);
        } else {
          // Other threads just arrive
          mbarrier_arrive(&mbar[next_it]);
        }
      }

      // Wait for the data to have arrived
      mbarrier_wait_parity(&mbar[it], parity);

#pragma unroll 4
      for (int stage = 0; stage < FP8_BUFF_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;

        float elt = static_cast<float>(in_sh[buff][shmem_offset_y][shmem_offset_x]);
        if constexpr (IS_DACT) {
          elt *= OP(static_cast<float>(act_in_sh[buff][shmem_offset_y][shmem_offset_x]), {});
        }
        if constexpr (IS_DBIAS) {
          partial_dbias[chunk_X] += elt;
        }
        if (isfinite(elt)) {
          amax = fmaxf(amax, fabsf(elt));
        }
        out_sh[buff][shmem_offset_y][shmem_offset_x] = static_cast<OType>(elt * scale);
      }

      // Wait for shared memory writes to be visible to TMA engine.
      fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.

      // Initiate TMA transfer to copy shared memory to global memory
      if (is_master_thread) {
        const int chunk_it_offset_y = chunk_offset_Y + it * FP8_BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
            chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

        // Create a "bulk async-group" out of the previous bulk copy operation.
        cp_async_bulk_commit_group();

        // Wait for TMA transfer to have finished reading shared memory.
        cp_async_bulk_wait_group_read<FP8_PREFETCH_BUFFERS_NUM>();
      }
    }
    cp_async_bulk_wait_group_read<0>();
    __syncthreads();

    parity ^= 1;
  }

  if constexpr (IS_DBIAS) {
#pragma unroll
    for (int i = 0; i < FP8_CHUNKS_PER_BLOCK_X; ++i) {
      const int dbias_offset_X = dbias_block_offset_X + i * FP8_CHUNK_DIM_X;
      const int dbias_offset = dbias_offset_Y * dbias_stride + dbias_offset_X;
      dbias_workspace[dbias_offset] = partial_dbias[i];
    }
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
    for (int it = 0; it < FP8_ITERATIONS; ++it) {
      mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

static PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled() {
  void *driver_ptr = nullptr;
  cudaDriverEntryPointQueryResult driver_status;
  NVTE_CHECK_CUDA(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr, cudaEnableDefault,
                                          &driver_status));
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}

CUtensorMapDataType get_CUtensorMapDataType(DType dtype) {
  static const std::unordered_map<DType, CUtensorMapDataType> dtypeMapping = {
      {DType::kByte, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat32, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
      {DType::kFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
      {DType::kBFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
      {DType::kFloat8E4M3, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat8E5M2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}};
  return dtypeMapping.at(dtype);
}

// Set up parameters to create TMA descriptor.
template <typename T>
void create_tensor_map(CUtensorMap &tensorMap, const Tensor *tensor_ptr, const uint64_t globalY,
                       const uint64_t globalX, const uint32_t shmemY, const uint32_t shmemX) {
  const Tensor &tensor = *tensor_ptr;
  // rank is the number of dimensions of the array
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {globalX, globalY};

  // The stride is the number of bytes to traverse from the first element of one row to the next
  uint64_t stride[rank - 1] = {globalX * sizeof(T)};

  // The boxSize is the size of the shared memory buffer that is used as the
  // source/destination of a TMA transfer
  uint32_t boxSize[rank] = {shmemX, shmemY};

  // The distance between elements in units of sizeof(element)
  uint32_t elemStride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  const CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.data.dtype);
  void *dataPtr = reinterpret_cast<void *>(tensor.data.dptr);

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
      &tensorMap,  // CUtensorMap *tensorMap,
      tensorDataType,
      rank,        // cuuint32_t tensorRank,
      dataPtr,     // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      boxSize,     // const cuuint32_t *boxDim,
      elemStride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,

      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,

      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

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

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void cast_fp8(const Tensor &input, const Tensor &act_input, Tensor *output, Tensor *dbias,
              Tensor *workspace, const Tensor *noop, cudaStream_t stream) {
  const size_t rows = input.data.shape[0];
  const size_t cols = input.data.shape[1];
  const size_t chunks_Y = DIVUP(rows, FP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, FP8_CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, FP8_CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, FP8_CHUNKS_PER_BLOCK_X);

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
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(FP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          CUtensorMap tensor_map_input{};
          CUtensorMap tensor_map_act_input{}; CUtensorMap tensor_map_output{};

          create_tensor_map<IType>(tensor_map_input, &input, rows, cols, FP8_SHMEM_DIM_Y,
                                   FP8_SHMEM_DIM_X);

          if constexpr (IS_DACT) {
            create_tensor_map<IType>(tensor_map_act_input, &act_input, rows, cols, FP8_SHMEM_DIM_Y,
                                     FP8_SHMEM_DIM_X);
          } create_tensor_map<OType>(tensor_map_output, output, rows, cols, FP8_SHMEM_DIM_Y,
                                     FP8_SHMEM_DIM_X);

          fp32 *noop_flag = noop != nullptr ? reinterpret_cast<fp32 *>(noop->data.dptr) : nullptr;
          cast_fp8_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType><<<grid, block, 0, stream>>>(
              tensor_map_input, tensor_map_act_input, tensor_map_output, noop_flag, workspace_ptr,
              amax_ptr, scale_inv_ptr, scale_ptr, rows, cols);

          if constexpr (IS_DBIAS) {
            reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

enum ScalingType { ROWWISE = 0, COLWISE = 1, BIDIMENTIONAL = 2 };

template <bool IS_DBIAS, bool IS_DACT, ScalingType SCALING, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void cast_mxfp8(const Tensor &input, const Tensor &act_input, Tensor *output_rowwise,
                Tensor *output_colwise, const Tensor *noop, Tensor *dbias, Tensor *workspace,
                cudaStream_t stream) {
  NVTE_CHECK(noop == nullptr, "TODO: Noop is not supported yet here!");
  constexpr bool USE_ROWWISE_SCALING =
      (SCALING == ScalingType::BIDIMENTIONAL || SCALING == ScalingType::ROWWISE);
  constexpr bool USE_COLWISE_SCALING =
      (SCALING == ScalingType::BIDIMENTIONAL || SCALING == ScalingType::COLWISE);

  const size_t scale_dim_X_rowwise = USE_ROWWISE_SCALING ? output_rowwise->scaling_mode.y : 1;
  const size_t scale_dim_Y_colwise = USE_COLWISE_SCALING ? output_colwise->scaling_mode.x : 1;

  const size_t rows = input.data.shape[0];
  const size_t cols = input.data.shape[1];
  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);
  const size_t blocks_Y = DIVUP(chunks_Y, CHUNKS_PER_BLOCK_Y);
  const size_t blocks_X = DIVUP(chunks_X, CHUNKS_PER_BLOCK_X);
  const size_t scale_stride_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t scale_stride_colwise = cols;

  const bool isFullTile = (rows % CHUNK_DIM_Y == 0) && (cols % CHUNK_DIM_X == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");

  e8m0_t *const scales_rowwise_ptr =
      USE_ROWWISE_SCALING ? reinterpret_cast<e8m0_t *>(output_rowwise->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      USE_COLWISE_SCALING ? reinterpret_cast<e8m0_t *>(output_colwise->scale_inv.dptr) : nullptr;
  const size_t dbias_rows =
      (scale_dim_Y_colwise > 1) ? blocks_Y : blocks_Y * THREADS_PER_CHUNK_Y_ROWWISE;
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

  const dim3 block(THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  DType OutputType = USE_ROWWISE_SCALING ? output_rowwise->data.dtype : output_colwise->data.dtype;

#define SCALE_DIM_SWITCH(SCALE_DIM, DIM, ...)               \
  switch (SCALE_DIM) {                                      \
    case 1: {                                               \
      constexpr size_t DIM = 1;                             \
      { __VA_ARGS__ }                                       \
    } break;                                                \
    case 32: {                                              \
      constexpr size_t DIM = 32;                            \
      { __VA_ARGS__ }                                       \
    } break;                                                \
    default: {                                              \
      NVTE_ERROR("Invalid size of the MX scaling factor."); \
    }                                                       \
  }

  SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
              input.data.dtype, IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
                  OutputType, OType,

                  CUtensorMap tensor_map_input{};
                  CUtensorMap tensor_map_act_input{}; CUtensorMap tensor_map_output_rowwise{};
                  CUtensorMap tensor_map_output_colwise{};

                  create_tensor_map<IType>(tensor_map_input, &input, rows, cols, SHMEM_DIM_Y,
                                           SHMEM_DIM_X);

                  if constexpr (IS_DACT) {
                    create_tensor_map<IType>(tensor_map_act_input, &act_input, rows, cols,
                                             SHMEM_DIM_Y, SHMEM_DIM_X);
                  } if constexpr (USE_ROWWISE_SCALING) {
                    create_tensor_map<OType>(tensor_map_output_rowwise, output_rowwise, rows, cols,
                                             SHMEM_DIM_Y, SHMEM_DIM_X);
                  } if constexpr (USE_COLWISE_SCALING) {
                    create_tensor_map<OType>(tensor_map_output_colwise, output_colwise, rows, cols,
                                             SHMEM_DIM_Y, SHMEM_DIM_X);
                  }

                  cast_mxfp8_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType, SCALE_DIM_Y,
                                    SCALE_DIM_X><<<grid, block, 0, stream>>>(
                      tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                      tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr,
                      workspace_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise);

                  if constexpr (IS_DBIAS) {
                    reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
                  });  // NOLINT(*)
          );           // NOLINT(*)
      );               // NOLINT(*)
  );                   // NOLINT(*)

#undef SCALE_DIM_SWITCH
}

namespace detail {

struct Empty {};

__device__ inline fp32 identity(fp32 value, const Empty &) { return value; }

struct DequantizeParam {
  const fp32 *scale_inv;
};

__device__ inline fp32 dequantize_func(fp32 value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

static const int32_t deviceComputeCapability = []() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return 10 * deviceProp.major + deviceProp.minor;
}();

static bool is_supported_by_CC_100() { return deviceComputeCapability >= 100; }

static bool is_supported_shape(const Tensor *output) {
  const NVTEScalingMode &scaling_mode = output->scaling_mode;
  const bool is_shape_supported = (scaling_mode.delayed_scaling == 0) &&
                                  (scaling_mode.x == 1 || scaling_mode.x == 32) &&
                                  (scaling_mode.y == 1 || scaling_mode.y == 32);
  return is_shape_supported;
}

static bool is_supported(const Tensor *output) {
  return is_supported_by_CC_100() && is_supported_shape(output);
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize(const Tensor &input, const Tensor &act_input, Tensor *output, const Tensor *noop,
                  Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
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
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Colwise scaling tensor must be allocated");

  if (is_supported_by_CC_100() && is_supported_shape(output)) {
    const bool is_colwise_scaling = (output->scaling_mode.x > 1);
    if (is_colwise_scaling) {
      cast_mxfp8<IS_DBIAS, IS_DACT, ScalingType::COLWISE, ParamOP, OP>(
          input, act_input, nullptr, output, noop, dbias, workspace, stream);
    } else {
      cast_mxfp8<IS_DBIAS, IS_DACT, ScalingType::ROWWISE, ParamOP, OP>(
          input, act_input, output, nullptr, noop, dbias, workspace, stream);
    }
  } else if (is_delayed_tensor_scaling(output->scaling_mode) && !IS_DBIAS && !IS_DACT) {
    const size_t N = product(input.data.shape);
    fp32 *noop_flag = noop != nullptr ? reinterpret_cast<fp32 *>(noop->data.dptr) : nullptr;
    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
        input.data.dtype, IType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            output->data.dtype, OType, constexpr int nvec = 32 / sizeof(IType);
            VectorizedUnaryKernelLauncher<nvec, detail::Empty, detail::identity>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {},
                noop_flag, stream););  // NOLINT(*)
    );                                 // NOLINT(*)
  }  else if (is_delayed_tensor_scaling(output->scaling_mode)) {
      if (output->scaling_mode.x == -1 && output->scaling_mode.y == -1) {
        cast_fp8<IS_DBIAS, IS_DACT, ParamOP, OP>(
          input, act_input, output, dbias, workspace, noop, stream);
      } else {
        NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
      }
      return;
  } else {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void fp8_quantize(const Tensor &input, const Tensor &act_input, Tensor *output_rowwise,
                  Tensor *output_colwise, const Tensor *noop, Tensor *dbias, Tensor *workspace,
                  cudaStream_t stream) {
  if (!is_supported_by_CC_100()) {
    NVTE_ERROR("Not supported by CC <10.0");
    return;
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

  if (is_supported(output_rowwise) && is_supported(output_colwise)) {
    cast_mxfp8<IS_DBIAS, IS_DACT, ScalingType::BIDIMENTIONAL, ParamOP, OP>(
        input, act_input, output_rowwise, output_colwise, noop, dbias, workspace, stream);
  } else {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(output_rowwise->scaling_mode) +
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
                nullptr, stream););  // NOLINT(*)
    );                               // NOLINT(*)
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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
      reinterpret_cast<Tensor *>(output), nullptr, reinterpret_cast<Tensor *>(dbias),
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

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_x2(const NVTETensor input, NVTETensor output_rowwise,
                                NVTETensor output_colwise, NVTETensor dbias, NVTETensor workspace,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;

  constexpr const NVTETensor activation_input = nullptr;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dgelu_x2(const NVTETensor input, const NVTETensor activation_input,
                                      NVTETensor output_rowwise, NVTETensor output_colwise,
                                      NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dgelu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &dgelu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_dsilu_x2(const NVTETensor input, const NVTETensor activation_input,
                                      NVTETensor output_rowwise, NVTETensor output_colwise,
                                      NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_dsilu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &dsilu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_dbias_drelu_x2(const NVTETensor input, const NVTETensor activation_input,
                                      NVTETensor output_rowwise, NVTETensor output_colwise,
                                      NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize_dbias_drelu_x2);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;

  constexpr auto dActivation = &drelu<fp32, fp32>;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
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

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
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

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, dActivation>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output_rowwise), reinterpret_cast<Tensor *>(output_colwise),
      nullptr, reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_quantize_with_noop(const NVTETensor input, NVTETensor output, const NVTETensor noop,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_quantize);
  using namespace transformer_engine;

  constexpr bool IS_DBIAS = false;
  constexpr bool IS_DACT = false;

  constexpr NVTETensor dbias = nullptr;
  constexpr NVTETensor workspace = nullptr;

  constexpr const NVTETensor activation_input = nullptr;

  fp8_quantize<IS_DBIAS, IS_DACT, Empty, nullptr>(
      *reinterpret_cast<const Tensor *>(input), *reinterpret_cast<const Tensor *>(activation_input),
      reinterpret_cast<Tensor *>(output), reinterpret_cast<Tensor *>(noop),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}

void nvte_fp8_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_dequantize);
  using namespace transformer_engine;
  fp8_dequantize(*reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(output),
                 stream);
}
