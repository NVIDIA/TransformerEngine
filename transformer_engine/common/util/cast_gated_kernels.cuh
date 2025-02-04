/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_gated_kernels.cuh
 *  \brief CUDA gated activations kernels to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_GATED_KERNELS_CUH_
#define TRANSFORMER_ENGINE_CAST_GATED_KERNELS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include <cfloat>

#include "../common.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"

namespace transformer_engine {

template <typename T1, typename T2>
__device__ __host__ __forceinline__ uint64_t DIVUP_TO_MULTIPLE(T1 N, T2 M) {
  return DIVUP(static_cast<uint64_t>(N), static_cast<uint64_t>(M)) * M;
}

namespace gated_kernels {

constexpr size_t ALIGNMENT_SIZE = 128;
constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 512;
constexpr size_t THREADS_PER_CHUNK_X = CHUNK_DIM_X;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;  // 4 = 512 / 128
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFFER_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                 //   4 = 128 / 32
static_assert(ITERATIONS >= 1);

__device__ inline float sigmoidf(const float x) { return __frcp_rn(1.0f + __expf(-x)); }

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_fp8_gated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                          const __grid_constant__ CUtensorMap tensor_map_input_act,
                          const __grid_constant__ CUtensorMap tensor_map_input_gate,
                          const __grid_constant__ CUtensorMap tensor_map_output_act,
                          const __grid_constant__ CUtensorMap tensor_map_output_gate,
                          float *const amax_ptr, float *const scale_inv_ptr,
                          const float *const scale_ptr, const size_t rows, const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  constexpr size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  constexpr size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  constexpr size_t buff_size_aligned_out =
      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

  constexpr size_t grad_mem = IS_DGATED ? buff_size_aligned_in : 0;

  constexpr size_t in_act_mem = buff_size_aligned_in;
  constexpr size_t in_gate_mem = buff_size_aligned_in;
  constexpr size_t in_mem = in_act_mem + in_gate_mem;

  constexpr size_t out_act_mem = buff_size_aligned_out;
  constexpr size_t out_gate_mem = buff_size_aligned_out;
  constexpr size_t out_mem = out_act_mem + out_gate_mem;

  // const size_t in_transaction_size = grad_mem + in_mem;
  constexpr size_t in_transaction_size = buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);
  OType *out_act_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);
  // uint64_t *mbar = reinterpret_cast<uint64_t *>(dshmem + grad_mem + in_mem + out_mem);

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_in_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
  const uint64_t *TMAP_in_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);
  const uint64_t *TMAP_output_act = reinterpret_cast<const uint64_t *>(&tensor_map_output_act);
  const uint64_t *TMAP_output_gate = reinterpret_cast<const uint64_t *>(&tensor_map_output_gate);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  // Prefetch data of the first stage

  if constexpr (IS_DGATED) {
    copy_2d_to_sharedx3(in_grad_sh, TMAP_grad_in, chunk_offset_X, chunk_offset_Y, in_act_sh,
                        TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh, TMAP_in_gate,
                        chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  } else {
    copy_2d_to_sharedx2(in_act_sh, TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh,
                        TMAP_in_gate, chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const int buff = it % BUFFERS_NUM;
    const int next_it = it + 1;
    if (next_it < ITERATIONS) {
      const int next_buff = next_it % BUFFERS_NUM;
      const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      if constexpr (IS_DGATED) {
        copy_2d_to_sharedx3(
            &in_grad_sh[next_buff * buff_elems], TMAP_grad_in, chunk_it_offset_x, chunk_it_offset_y,
            &in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x, chunk_it_offset_y,
            &in_gate_sh[next_buff * buff_elems], TMAP_in_gate, chunk_it_offset_x, chunk_it_offset_y,
            in_transaction_size, &mbar[next_it], is_master_thread);
      } else {
        copy_2d_to_sharedx2(&in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x,
                            chunk_it_offset_y, &in_gate_sh[next_buff * buff_elems], TMAP_in_gate,
                            chunk_it_offset_x, chunk_it_offset_y, in_transaction_size,
                            &mbar[next_it], is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_sh_curr = out_act_sh + buff * buff_elems;
    OType *out_gate_sh_curr = out_gate_sh + buff * buff_elems;

#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);

      if constexpr (IS_DGATED) {
        float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);

        const float x = act_elt;
        float act_x;
        float dact_x;

        if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
          const float s = sigmoidf(x);
          act_x = x * s;
          dact_x = x * s * (1 - s) + s;
        } else {
          act_x = ActOP(x, {});
          dact_x = DActOP(x, {});
        }

        float after_dact = dact_x * grad_elt * gate_elt;
        float after_dgate = act_x * grad_elt;

        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dact);
        out_gate_sh_curr[shmem_idx] = static_cast<OType>(scale * after_dgate);

        amax = fmaxf(amax, fabsf(after_dact));
        amax = fmaxf(amax, fabsf(after_dgate));
      } else {
        const float after_act = ActOP(act_elt, {}) * gate_elt;
        out_act_sh_curr[shmem_idx] = static_cast<OType>(scale * after_act);
        amax = fmaxf(amax, fabsf(after_act));
      }
    }

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      ptx::cp_async_bulk_tensor_2d_shared_to_global(TMAP_output_act, chunk_it_offset_x,
                                                    chunk_it_offset_y,
                                                    reinterpret_cast<uint64_t *>(out_act_sh_curr));

      if constexpr (IS_DGATED) {
        // dGate
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_gate, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_gate_sh_curr));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
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
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType,
          size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_mxfp8_gated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                            const __grid_constant__ CUtensorMap tensor_map_input_act,
                            const __grid_constant__ CUtensorMap tensor_map_input_gate,
                            const __grid_constant__ CUtensorMap tensor_map_output_act_rowwise,
                            const __grid_constant__ CUtensorMap tensor_map_output_gate_rowwise,
                            const __grid_constant__ CUtensorMap tensor_map_output_act_colwise,
                            const __grid_constant__ CUtensorMap tensor_map_output_gate_colwise,
                            e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                            const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
                            const size_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;
  constexpr bool USE_COLWISE_SCALING = SCALE_DIM_Y > 1;
  constexpr bool COMPUTE_IN_ROWWISE_SECTION = !USE_COLWISE_SCALING;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const int tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const int thread_offset_Y = tid_Y;
  const int thread_offset_X = tid_X;

  const bool col_out_of_bounds = (chunk_offset_X + thread_offset_X >= cols);

  extern __shared__ char dshmem_unaligned[];
  const uint64_t dshmem_unaligned_as_uint = reinterpret_cast<uint64_t>(dshmem_unaligned);
  const uint64_t dshmem_aligned_as_uint =
      DIVUP(dshmem_unaligned_as_uint, static_cast<uint64_t>(ALIGNMENT_SIZE)) * ALIGNMENT_SIZE;
  char *dshmem = reinterpret_cast<char *>(dshmem_aligned_as_uint);

  const size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  const size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  const size_t buff_size_aligned_in =
      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
  const size_t buff_size_aligned_out =
      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

  const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);

  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t in_mem = in_act_mem + in_gate_mem;

  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = buff_size_aligned_out;
  const size_t out_mem = out_act_mem + out_gate_mem;

  // const size_t in_transaction_size = grad_mem + in_mem;
  const size_t in_transaction_size = (IS_DGATED ? 3 : 2) * buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);

  OType *out_act_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_rowwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  OType *out_act_colwise_sh = out_act_rowwise_sh;
  OType *out_gate_colwise_sh = out_gate_rowwise_sh;

  if constexpr (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) {
    out_act_colwise_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem);
    out_gate_colwise_sh =
        reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_mem + out_act_mem);
  }

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_in_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
  const uint64_t *TMAP_in_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);
  const uint64_t *TMAP_output_act_rowwise =
      reinterpret_cast<const uint64_t *>(&tensor_map_output_act_rowwise);
  const uint64_t *TMAP_output_gate_rowwise =
      reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_rowwise);
  const uint64_t *TMAP_output_act_colwise =
      reinterpret_cast<const uint64_t *>(&tensor_map_output_act_colwise);
  const uint64_t *TMAP_output_gate_colwise =
      reinterpret_cast<const uint64_t *>(&tensor_map_output_gate_colwise);

  __shared__ float stage_amax_sh[THREADS_PER_CHUNK_Y][CHUNK_DIM_X];

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  const bool is_master_thread = (threadIdx.x == 0);

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_init(&mbar[it], THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;

  // Prefetch data of the first stage
  if (is_master_thread) {
    // Initiate bulk tensor copy
    // Grad
    if constexpr (IS_DGATED) {
      ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_grad_sh[0]),
                                                    TMAP_grad_in, chunk_offset_X, chunk_offset_Y,
                                                    &mbar[0]);
    }

    // Act
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_act_sh[0]),
                                                  TMAP_in_act, chunk_offset_X, chunk_offset_Y,
                                                  &mbar[0]);

    // Gate
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(&in_gate_sh[0]),
                                                  TMAP_in_gate, chunk_offset_X, chunk_offset_Y,
                                                  &mbar[0]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[0], in_transaction_size);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[0]);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const int buff = it % BUFFERS_NUM;
    const int next_it = it + 1;
    const size_t row_base = chunk_offset_Y + it * BUFFER_DIM_Y;
    if (next_it < ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_it % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        if constexpr (IS_DGATED) {
          // Grad
          ptx::cp_async_bulk_tensor_2d_global_to_shared(
              reinterpret_cast<uint64_t *>(&in_grad_sh[next_buff * buff_elems]), TMAP_grad_in,
              chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
        }
        // Act
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_act_sh[next_buff * buff_elems]), TMAP_in_act,
            chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);
        // Gate
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_gate_sh[next_buff * buff_elems]), TMAP_in_gate,
            chunk_it_offset_x, chunk_it_offset_y, &mbar[next_it]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_it], in_transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_it]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_rowwise_sh_curr = out_act_rowwise_sh + buff * buff_elems;
    OType *out_gate_rowwise_sh_curr = out_gate_rowwise_sh + buff * buff_elems;
    OType *out_act_colwise_sh_curr = out_act_colwise_sh + buff * buff_elems;
    OType *out_gate_colwise_sh_curr = out_gate_colwise_sh + buff * buff_elems;

    // Assuming one iteration covers exactly 32 rows
    const int iteration_scale_colwise_offset_Y = scales_colwise_chunk_offset_Y + it;
    const int iteration_scale_rowwise_offset_Y = scales_rowwise_chunk_offset_Y + it * BUFFER_DIM_Y;

    float after_dact_reg[BUFFER_STAGES_NUM];
    float after_dgate_reg[BUFFER_STAGES_NUM];
    float thread_Y_mx_block_amax = 0.0f;
    float thread_Y_mx_block_amax_gate = 0.0f;

#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
      const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
      const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const int shmem_offset_x = thread_offset_X;
      const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

      const size_t row = row_base + shmem_offset_y;
      const bool row_out_of_bounds = (row >= rows);
      const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

      float act_elt = static_cast<float>(in_act_sh_curr[shmem_idx]);
      float gate_elt = static_cast<float>(in_gate_sh_curr[shmem_idx]);

      if constexpr (IS_DGATED) {
        float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx]);
        const float x = act_elt;
        float act_x;
        float dact_x;

        if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
          const float s = sigmoidf(x);
          act_x = x * s;
          dact_x = x * s * (1 - s) + s;
        } else {
          act_x = ActOP(x, {});
          dact_x = DActOP(x, {});
        }
        after_dact_reg[stage] = dact_x * grad_elt * gate_elt;
        after_dgate_reg[stage] = act_x * grad_elt;
      } else {
        after_dact_reg[stage] = ActOP(act_elt, {}) * gate_elt;
      }

      if constexpr (USE_ROWWISE_SCALING) {
        if constexpr (IS_DGATED) {
          // dgate
          float amax = fabsf(after_dgate_reg[stage]);
          const float mx_block_X_amax = warp_reduce_max_broadcast(amax);
          const e8m0_t biased_exponent_X =
              float_to_e8m0(mx_block_X_amax * Quantized_Limits<OType>::max_norm_rcp);
          const float scale_reciprocal_X = exp2f_rcp(biased_exponent_X);

          out_gate_rowwise_sh_curr[shmem_idx] =
              static_cast<OType>(scale_reciprocal_X * after_dgate_reg[stage]);

          // Only single thread writes the computed scaling factor
          if ((tid_X % SCALE_DIM_X == 0) && !out_of_bounds) {
            const int global_scales_offset_Y =
                iteration_scale_rowwise_offset_Y + stage_offset_Y + thread_offset_Y;
            const int global_scales_offset_X =
                scales_rowwise_chunk_offset_X + (tid_X + cols) / SCALE_DIM_X;
            const int scale_idx =
                global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
            scales_rowwise[scale_idx] = biased_exponent_X;
          }
        }
        float amax = fabsf(after_dact_reg[stage]);
        const float mx_block_X_amax = warp_reduce_max_broadcast(amax);
        const e8m0_t biased_exponent_X =
            float_to_e8m0(mx_block_X_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal_X = exp2f_rcp(biased_exponent_X);

        out_act_rowwise_sh_curr[shmem_idx] =
            static_cast<OType>(scale_reciprocal_X * after_dact_reg[stage]);

        // Only single thread writes the computed scaling factor
        if ((tid_X % SCALE_DIM_X == 0) && !out_of_bounds) {
          const int global_scales_offset_Y =
              iteration_scale_rowwise_offset_Y + stage_offset_Y + thread_offset_Y;
          const int global_scales_offset_X = scales_rowwise_chunk_offset_X + tid_X / SCALE_DIM_X;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_rowwise + global_scales_offset_X;
          scales_rowwise[scale_idx] = biased_exponent_X;
        }
      }

      if constexpr (USE_COLWISE_SCALING) {
        __builtin_assume(thread_Y_mx_block_amax >= 0);
        __builtin_assume(thread_Y_mx_block_amax_gate >= 0);
        thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, fabsf(after_dact_reg[stage]));
        if constexpr (IS_DGATED) {
          thread_Y_mx_block_amax_gate =
              fmaxf(thread_Y_mx_block_amax_gate, fabsf(after_dgate_reg[stage]));
        }
      }
    }

    if constexpr (USE_COLWISE_SCALING) {
      const bool row_out_of_bounds = (row_base >= rows);
      const bool out_of_bounds = (col_out_of_bounds || row_out_of_bounds);

      if constexpr (IS_DGATED) {
        // Colwise max reduction of the amax element
        if (tid_Y > 0) {
          stage_amax_sh[tid_Y][tid_X] = thread_Y_mx_block_amax_gate;
        }
        __syncthreads();
        if (tid_Y == 0) {
#pragma unroll
          for (int y = 1; y < THREADS_PER_CHUNK_Y; ++y) {
            thread_Y_mx_block_amax_gate =
                fmaxf(thread_Y_mx_block_amax_gate, stage_amax_sh[y][tid_X]);
          }
          stage_amax_sh[0][tid_X] = thread_Y_mx_block_amax_gate;  // write mx column-block amax
        }
        __syncthreads();

        const float mx_block_Y_amax = stage_amax_sh[0][tid_X];  // read the mx column-block amax

        // For the scaling along both dimensions, the thread amax is already computed in ROWWISE section
        if constexpr (!USE_ROWWISE_SCALING) {
          __builtin_assume(mx_block_Y_amax >= 0);
        }

        const e8m0_t biased_exponent =
            float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
        const float scale_reciprocal = exp2f_rcp(biased_exponent);

        // Only single thread writes the computed scaling factor
        // Also assuming one iteration covers exactly 32 rows
        if ((tid_Y == 0) && !out_of_bounds) {
          const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
          const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_X + cols;
          const int scale_idx =
              global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
          scales_colwise[scale_idx] = biased_exponent;
        }

#pragma unroll
        for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
          const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
          const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
          const int shmem_offset_x = thread_offset_X;
          const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

          out_gate_colwise_sh_curr[shmem_idx] =
              static_cast<OType>(scale_reciprocal * after_dgate_reg[stage]);
        }
      }
      // Colwise max reduction of the amax element
      if (tid_Y > 0) {
        stage_amax_sh[tid_Y][tid_X] = thread_Y_mx_block_amax;
      }
      __syncthreads();
      if (tid_Y == 0) {
#pragma unroll
        for (int y = 1; y < THREADS_PER_CHUNK_Y; ++y) {
          thread_Y_mx_block_amax = fmaxf(thread_Y_mx_block_amax, stage_amax_sh[y][tid_X]);
        }
        stage_amax_sh[0][tid_X] = thread_Y_mx_block_amax;  // write mx column-block amax
      }
      __syncthreads();

      const float mx_block_Y_amax = stage_amax_sh[0][tid_X];  // read the mx column-block amax

      // For the scaling along both dimensions, the thread amax is already computed in ROWWISE section
      if constexpr (!USE_ROWWISE_SCALING) {
        __builtin_assume(mx_block_Y_amax >= 0);
      }

      const e8m0_t biased_exponent =
          float_to_e8m0(mx_block_Y_amax * Quantized_Limits<OType>::max_norm_rcp);
      const float scale_reciprocal = exp2f_rcp(biased_exponent);

      // Only single thread writes the computed scaling factor
      // Also assuming one iteration covers exactly 32 rows
      if ((tid_Y == 0) && !out_of_bounds) {
        const int global_scales_offset_Y = iteration_scale_colwise_offset_Y;
        const int global_scales_offset_X = scales_colwise_chunk_offset_X + tid_X;
        const int scale_idx =
            global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
        scales_colwise[scale_idx] = biased_exponent;
      }

#pragma unroll
      for (int stage = 0; stage < BUFFER_STAGES_NUM; ++stage) {
        const int stage_offset_Y = stage * THREADS_PER_CHUNK_Y;
        const int shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const int shmem_offset_x = thread_offset_X;
        const int shmem_idx = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

        out_act_colwise_sh_curr[shmem_idx] =
            static_cast<OType>(scale_reciprocal * after_dact_reg[stage]);
      }
    }  // endif USE_COLWISE_SCALING

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      if constexpr (USE_ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_act_rowwise, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_act_rowwise_sh_curr));

        if constexpr (IS_DGATED) {
          // dGate
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              TMAP_output_gate_rowwise, chunk_it_offset_x, chunk_it_offset_y,
              reinterpret_cast<uint64_t *>(out_gate_rowwise_sh_curr));
        }
      }

      // dGeLU
      if constexpr (USE_COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_act_colwise, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_act_colwise_sh_curr));

        if constexpr (IS_DGATED) {
          // dGate
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              TMAP_output_gate_colwise, chunk_it_offset_x, chunk_it_offset_y,
              reinterpret_cast<uint64_t *>(out_gate_colwise_sh_curr));
        }
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_fp8_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                    cudaStream_t stream) {
  if (output->has_data()) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (output->has_columnwise_data()) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }

  NVTE_CHECK(!output->has_columnwise_data(), "Only rowwise cast supported in this function.");
  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_input_act{};
          alignas(64) CUtensorMap tensor_map_input_gate{};
          alignas(64) CUtensorMap tensor_map_output_act{};
          alignas(64) CUtensorMap tensor_map_output_gate{};

          if constexpr (IS_DGATED) {
            create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X,
                                 cols, 0, sizeof(IType));
          }

          const uint32_t tensor_stride_elems = output_cols;

          create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, 0, sizeof(IType));
          create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, cols, sizeof(IType));
          create_2D_tensor_map(tensor_map_output_act, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, 0, sizeof(OType));
          create_2D_tensor_map(tensor_map_output_gate, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, cols, sizeof(OType));

          const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
          const size_t buff_size_aligned_in =
              DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
          const size_t buff_size_aligned_out =
              DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
          const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);
          const size_t in_act_mem = buff_size_aligned_in;
          const size_t in_gate_mem = buff_size_aligned_in;
          const size_t out_act_mem = buff_size_aligned_out;
          const size_t out_gate_mem = buff_size_aligned_out;
          // const size_t mbar_mem = ITERATIONS * sizeof(uint64_t);
          const size_t shmem_size = ALIGNMENT_SIZE + grad_mem + (in_act_mem + in_gate_mem) +
                                    (out_act_mem + out_gate_mem);  // + mbar_mem;

          cudaFuncSetAttribute(
              cast_fp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

          cast_fp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType>
          <<<grid_dim, block_dim, shmem_size, stream>>>(
              tensor_map_grad, tensor_map_input_act, tensor_map_input_gate, tensor_map_output_act,
              tensor_map_output_gate, amax_ptr, scale_inv_ptr, scale_ptr, rows,
              cols););  // NOLINT(*)
  );                    // NOLINT(*)
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_mxfp8_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                      cudaStream_t stream) {
  const bool USE_ROWWISE_SCALING = output->has_data();
  const bool USE_COLWISE_SCALING = output->has_columnwise_data();

  if (USE_ROWWISE_SCALING) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }
  if (USE_COLWISE_SCALING) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  }

  // TODO: Make more general
  const size_t scale_dim_X_rowwise = USE_ROWWISE_SCALING ? 32 : 1;
  const size_t scale_dim_Y_colwise = USE_COLWISE_SCALING ? 32 : 1;

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  size_t scale_stride_rowwise = USE_ROWWISE_SCALING ? output->scale_inv.shape[1] : 1;
  size_t scale_stride_colwise = USE_COLWISE_SCALING ? output->columnwise_scale_inv.shape[1] : 1;

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);

  e8m0_t *const scales_rowwise_ptr =
      USE_ROWWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      USE_COLWISE_SCALING ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
              gated_input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
                  output->dtype(), OType,

                  alignas(64) CUtensorMap tensor_map_grad{};
                  alignas(64) CUtensorMap tensor_map_input_act{};
                  alignas(64) CUtensorMap tensor_map_input_gate{};
                  alignas(64) CUtensorMap tensor_map_output_act_rowwise{};
                  alignas(64) CUtensorMap tensor_map_output_gate_rowwise{};
                  alignas(64) CUtensorMap tensor_map_output_act_colwise{};
                  alignas(64) CUtensorMap tensor_map_output_gate_colwise{};

                  if constexpr (IS_DGATED) {
                    create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, SHMEM_DIM_Y,
                                         SHMEM_DIM_X, cols, 0, sizeof(IType));
                  }

                  const uint32_t tensor_stride_elems = output_cols;
                  create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols,
                                       SHMEM_DIM_Y, SHMEM_DIM_X, cols * 2, 0, sizeof(IType));
                  create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols,
                                       SHMEM_DIM_Y, SHMEM_DIM_X, cols * 2, cols, sizeof(IType));

                  if (USE_ROWWISE_SCALING) {
                    create_2D_tensor_map(tensor_map_output_act_rowwise, output->data, rows, cols,
                                         SHMEM_DIM_Y, SHMEM_DIM_X, tensor_stride_elems, 0,
                                         sizeof(OType));
                    create_2D_tensor_map(tensor_map_output_gate_rowwise, output->data, rows, cols,
                                         SHMEM_DIM_Y, SHMEM_DIM_X, tensor_stride_elems, cols,
                                         sizeof(OType));
                  }

                  if (USE_COLWISE_SCALING) {
                    create_2D_tensor_map(tensor_map_output_act_colwise, output->columnwise_data,
                                         rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X, tensor_stride_elems,
                                         0, sizeof(OType));
                    create_2D_tensor_map(tensor_map_output_gate_colwise, output->columnwise_data,
                                         rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X, tensor_stride_elems,
                                         cols, sizeof(OType));
                  }

                  const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
                  const size_t buff_size_aligned_in =
                      DIVUP(buff_elems_total * sizeof(IType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;
                  const size_t buff_size_aligned_out =
                      DIVUP(buff_elems_total * sizeof(OType), ALIGNMENT_SIZE) * ALIGNMENT_SIZE;

                  const size_t grad_mem = (IS_DGATED ? buff_size_aligned_in : 0);
                  const size_t in_act_mem = buff_size_aligned_in;
                  const size_t in_gate_mem = buff_size_aligned_in;
                  const size_t in_mem = grad_mem + in_act_mem + in_gate_mem;

                  const size_t out_act_mem = buff_size_aligned_out;
                  const size_t out_gate_mem = buff_size_aligned_out;
                  size_t out_mem = out_act_mem + out_gate_mem;
                  if (USE_ROWWISE_SCALING && USE_COLWISE_SCALING) { out_mem *= 2; }

                  // const size_t mbar_mem = ITERATIONS * sizeof(uint64_t);
                  // const size_t shmem_size = ALIGNMENT_SIZE + in_mem + out_mem + mbar_mem;

                  const size_t shmem_size = ALIGNMENT_SIZE + in_mem + out_mem;

                  cudaFuncSetAttribute(
                      cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType,
                                              SCALE_DIM_Y, SCALE_DIM_X>,
                      cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

                  cast_mxfp8_gated_kernel<IS_DGATED, ParamOP, ActOP, DActOP, IType, OType,
                                          SCALE_DIM_Y, SCALE_DIM_X>
                  <<<grid_dim, block_dim, shmem_size, stream>>>(
                      tensor_map_grad, tensor_map_input_act, tensor_map_input_gate,
                      tensor_map_output_act_rowwise, tensor_map_output_gate_rowwise,
                      tensor_map_output_act_colwise, tensor_map_output_gate_colwise,
                      scales_rowwise_ptr, scales_colwise_ptr, rows, cols, scale_stride_rowwise,
                      scale_stride_colwise););  // NOLINT(*)
          );                                    // NOLINT(*)
      );                                        // NOLINT(*)
  );                                            // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void cast_gated(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "gated_act_input");
  CheckOutputTensor(*output, "gated_act_output");
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->data.shape.size() == 2, "Output must have 2 dimensions.");
  NVTE_CHECK(input.data.shape[0] == output->data.shape[0],
             "Input shape[0] must be equal to output shape[0].");
  NVTE_CHECK(input.data.shape[1] == output->data.shape[1] * 2,
             "Input shape[1] must be 2x larger than output shape[1].");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,

          if (!is_fp8_dtype(output->data.dtype) ||
              is_delayed_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            GatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), output->data.shape[0],
                output->data.shape[1], {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_dgated(const Tensor &grad, const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(grad, "dgated_act_grad");
  CheckInputTensor(input, "dgated_act_input");
  CheckOutputTensor(*output, "dgated_act_output");
  NVTE_CHECK(output->flat_first_dim() == grad.flat_first_dim(),
             "Wrong output shape. Expected (after flattening) [", grad.flat_first_dim(),
             ", *], got [", output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(output->flat_last_dim() == grad.flat_last_dim() * 2,
             "Wrong output shape. Expected (after flattening) [*, ", grad.flat_last_dim() * 2,
             "], got [", output->flat_first_dim(), ", ", output->flat_last_dim(), "].");
  NVTE_CHECK(input.data.shape == output->data.shape,
             "Input and output shapes must match. Input shape: ", input.data.shape,
             ", output shape: ", output->data.shape, ".");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          if (!is_fp8_dtype(output->data.dtype) ||
              is_delayed_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            DGatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP, DActOP>(
                reinterpret_cast<const IType *>(grad.data.dptr),
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), grad.flat_first_dim(),
                grad.flat_last_dim(), {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated(const Tensor &grad, const Tensor &gated_input, Tensor *output,
                    cudaStream_t stream) {
  checkCuDriverContext(stream);
  constexpr bool allow_empty = false;
  CheckInputTensor(gated_input, "gated_input");
  CheckOutputTensor(*output, "output", allow_empty);

  NVTE_CHECK(gated_input.flat_last_dim() % 2 == 0, "Number of columns must be even.");

  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_DGATED ? 2 : 1) * cols;

  if constexpr (IS_DGATED) {
    CheckInputTensor(grad, "grad");
    NVTE_CHECK(!is_fp8_dtype(grad.data.dtype), "Grad input must be in higher precision.");
    NVTE_CHECK(grad.data.dtype == gated_input.data.dtype, "Types of both inputs must match.");
    NVTE_CHECK(grad.flat_first_dim() == rows, "Wrong dimension of the grad input.");
    NVTE_CHECK(grad.flat_last_dim() == cols, "Wrong dimension of the grad input.");
  }

  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Either rowwise or columnwise output data need to be allocated.");

  bool is_fp8_rowwise_output = true;
  bool is_fp8_colwise_output = true;
  if (output->has_data()) {
    is_fp8_rowwise_output = is_fp8_dtype(output->data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == output_cols, "Wrong dimension of the output.");
  }
  if (output->has_columnwise_data()) {
    is_fp8_colwise_output = is_fp8_dtype(output->columnwise_data.dtype);
    NVTE_CHECK(output->flat_first_dim() == rows, "Wrong dimension of the output.");
    NVTE_CHECK(output->flat_last_dim() == output_cols, "Wrong dimension of the output.");
  }

  const bool use_tma_kernels = is_fp8_rowwise_output && is_fp8_colwise_output && cols % 32 == 0;

  if (is_delayed_tensor_scaling(output->scaling_mode)) {
    if (use_tma_kernels) {
      cast_fp8_gated<IS_DGATED, ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
    } else {
      if constexpr (IS_DGATED) {
        cast_dgated<ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
      } else {
        cast_gated<ParamOP, ActOP>(gated_input, output, stream);
      }
    }
  } else if (is_mxfp_scaling(output->scaling_mode)) {
    if (use_tma_kernels) {
      cast_mxfp8_gated<IS_DGATED, ParamOP, ActOP, DActOP>(grad, gated_input, output, stream);
    } else {
      NVTE_ERROR("Invalid input shape. Expected the last dimension to be divisible ",
                 "by 32, got input of shape ", gated_input.data.shape);
    }
  } else {
    NVTE_ERROR("Not supported scaling mode");
  }
}
}  // namespace gated_kernels

namespace detail {

template <bool IS_DGATED, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void quantize_gated_helper(const NVTETensor grad, const NVTETensor gated_input, NVTETensor output,
                           cudaStream_t stream) {
  using namespace gated_kernels;
  Tensor grad_empty_tensor;
  const Tensor &grad_tensor =
      IS_DGATED ? *(reinterpret_cast<const Tensor *>(grad)) : grad_empty_tensor;
  const Tensor gated_input_tensor = *reinterpret_cast<const Tensor *>(gated_input);
  Tensor *output_tensor = reinterpret_cast<Tensor *>(output);

  if (is_supported_by_CC_100()) {
    quantize_gated<IS_DGATED, ParamOP, ActOP, DActOP>(grad_tensor, gated_input_tensor,
                                                      output_tensor, stream);
  } else {
    if (is_delayed_tensor_scaling(output_tensor->scaling_mode)) {
      if constexpr (IS_DGATED) {
        cast_dgated<ParamOP, ActOP, DActOP>(grad_tensor, gated_input_tensor, output_tensor, stream);
      } else {
        cast_gated<ParamOP, ActOP>(gated_input_tensor, output_tensor, stream);
      }
    } else {
      // MX scaling
      NVTE_ERROR("Not supported by the Arch < 10.0");
    }
  }
}
}  // namespace detail

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_GATED_KERNELS_CUH_
