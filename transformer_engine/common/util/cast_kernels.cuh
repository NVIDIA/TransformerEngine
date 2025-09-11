/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_kernels.cuh
 *  \brief CUDA kernels to cast to/from FP8/MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_CAST_KERNELS_CUH_
#define TRANSFORMER_ENGINE_CAST_KERNELS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

namespace mxfp8_kernel {

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t PACK_SIZE = 4;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

// Number of 1-byte elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4) / 1;  // 128

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM_X;  // 4 = 128 / 32

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, bool ROWWISE_SCALING,
          bool COLWISE_SCALING, size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X, size_t THREADS_PER_CHUNK>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_mxfp8_2D_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                         const __grid_constant__ CUtensorMap tensor_map_act_input,
                         const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                         const __grid_constant__ CUtensorMap tensor_map_output_colwise,
                         e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                         const float *noop, float *const dbias_workspace, float *const amax_ptr,
                         const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
                         const size_t scale_stride_colwise, const bool pdl_sync,
                         const bool pdl_trigger) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool COMPUTE_ACTIVATIONS = IS_DACT || IS_ACT;
  constexpr bool NO_ACTIVATIONS = !COMPUTE_ACTIVATIONS;

  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;

  if constexpr (NO_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }
  constexpr size_t THREADS_X = CHUNK_DIM_X / SCALE_DIM_X;
  constexpr size_t THREADS_Y = THREADS_PER_CHUNK / THREADS_X;

  constexpr size_t BUFF_DIM_Y = THREADS_Y;
  constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;
  constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
  static_assert(BUFF_DIM_Y == 32);

  constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
  static_assert(STAGES >= 1);

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && ROWWISE_SCALING && COLWISE_SCALING;

  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;
  const size_t scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = blockIdx.x * CHUNK_DIM_X / SCALE_DIM_X;
  const size_t scales_block_offset_Y_colwise = blockIdx.y * CHUNK_DIM_Y / SCALE_DIM_Y;
  const size_t scales_block_offset_X_colwise = blockIdx.x * CHUNK_DIM_X;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X;
  const size_t tid_Y_colwise = 0;
  const size_t tid_X_colwise = threadIdx.x;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;
  const size_t thread_offset_Y_colwise = tid_Y_colwise;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t row_base_colwise = block_offset_Y + thread_offset_Y_colwise;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const size_t scales_offset_Y_colwise = scales_block_offset_Y_colwise + tid_Y_colwise;
  const size_t scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

  // helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  constexpr size_t elt_input_mem = buff_size_aligned_in;
  constexpr size_t act_input_mem = (IS_DACT ? buff_size_aligned_in : 0);
  constexpr size_t in_mem = elt_input_mem + act_input_mem;

  constexpr size_t out_mem_rowwise = (ROWWISE_SCALING ? buff_size_aligned_out : 0);

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  IType *act_in_sh = reinterpret_cast<IType *>(dshmem + elt_input_mem);
  OType *out_rowwise_sh = reinterpret_cast<OType *>(dshmem + in_mem);
  OType *out_colwise_sh = reinterpret_cast<OType *>(dshmem + in_mem + out_mem_rowwise);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  float partial_dbias_colwise = 0.0f;
  float thread_dbias_rowwise[SCALE_DIM_X];
  if constexpr (IS_DBIAS) {
#pragma unroll
    for (int j = 0; j < SCALE_DIM_X; ++j) {
      thread_dbias_rowwise[j] = 0.0f;
    }
  }

  float block_amax = 0.0f;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

// Block until the previous kernel has completed and flushed results to global memory.
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (pdl_sync) {
    cudaGridDependencySynchronize();
  }
#endif

  if constexpr (IS_DACT) {
    copy_2d_to_sharedx2(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, &act_in_sh[0],
                        &tensor_map_act_input, block_offset_X, block_offset_Y, shmem_buff_size,
                        &mbar[0], is_master_thread);
  } else {
    copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                      &mbar[0], is_master_thread);
  }

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t next_buff_offset = next_buff * BUFF_DIM;
      if constexpr (IS_DACT) {
        copy_2d_to_sharedx2(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                            global_offset_Y, &act_in_sh[next_buff_offset], &tensor_map_act_input,
                            global_offset_X, global_offset_Y, shmem_buff_size, &mbar[next_stage],
                            is_master_thread);
      } else {
        copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                          global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], parity);

    // Trigger the next kernel, so its TMA load can be overlapped with the current kernel
    if (pdl_trigger && stage == STAGES - 1) {
      cudaTriggerProgrammaticLaunchCompletion();
    }

    float thread_amax = 0.0f;
    if constexpr (COLWISE_SCALING) {
      const size_t shmem_offset_base_colwise = buff * BUFF_DIM + tid_X_colwise;
      thread_amax = 0.0f;
      float in_compute_colwise[BUFF_DIM_Y];
      IType in_colwise_IType[BUFF_DIM_Y];

      // 1. Read/Compute elements. Find MXFP8-block AMAX
      if constexpr (NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>)) {
        IType thread_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
        for (int i = 0; i < BUFF_DIM_Y; ++i) {
          const size_t shmem_offset_colwise = shmem_offset_base_colwise + i * BUFF_DIM_X;
          in_colwise_IType[i] = in_sh[shmem_offset_colwise];
          thread_amax_f16 = __hmax(thread_amax_f16, __habs(in_colwise_IType[i]));
        }
        thread_amax = static_cast<float>(thread_amax_f16);
      } else {
#pragma unroll
        for (int i = 0; i < BUFF_DIM_Y; ++i) {
          const size_t shmem_offset_colwise = shmem_offset_base_colwise + i * BUFF_DIM_X;

          float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
          if constexpr (IS_ACT) {
            elt = OP(elt, {});
          }
          if constexpr (IS_DACT) {
            float act_in_elt = static_cast<float>(act_in_sh[shmem_offset_colwise]);
            elt *= OP(act_in_elt, {});
          }
          if constexpr (IS_DBIAS) {
            partial_dbias_colwise += elt;
          }
          // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
          if constexpr (!std::is_same_v<IType, float>) {
            elt = static_cast<float>(static_cast<IType>(elt));
          }
          // Cache computed activations to avoid computing them again in the 2nd pass along another dimension
          if constexpr (IS_CACHED_ACT_OP) {
            cached_act_sh[shmem_offset_colwise] = static_cast<IType>(elt);
          }

          if constexpr (COMPUTE_ACTIVATIONS) {
            const bool row_out_of_bounds_colwise = (row_base_colwise + stage_offset_Y + i >= rows);
            const bool out_of_bounds = (col_out_of_bounds_colwise || row_out_of_bounds_colwise);
            if (!out_of_bounds) {
              thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
          } else {
            // If no activation, elt is 0 so we can safely do this
            thread_amax = fmaxf(thread_amax, fabsf(elt));
          }
          in_compute_colwise[i] = elt;
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);

      const size_t global_scales_offset_Y = scales_offset_Y_colwise + stage;
      const size_t global_scales_offset_X = scales_offset_X_colwise;
      const size_t scale_idx =
          global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
      scales_colwise[scale_idx] = biased_exponent;

      const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      const ptx::floatx2 block_scale_inverse_2x = {block_scale_inverse, block_scale_inverse};

// 3. Scale elements
#pragma unroll
      for (int i = 0; i < SCALE_DIM_Y; ++i) {
        float in;
        if constexpr (NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>)) {
          in = static_cast<float>(in_colwise_IType[i]);
        } else {
          in = in_compute_colwise[i];
        }
        const float scaled_out = in * block_scale_inverse;

        const size_t shmem_offset_elt = shmem_offset_base_colwise + i * BUFF_DIM_X;
        out_colwise_sh[shmem_offset_elt] = static_cast<OType>(scaled_out);
      }
    }

    if constexpr (ROWWISE_SCALING) {
      const size_t shmem_offset_base_rowwise =
          buff * BUFF_DIM + thread_offset_Y_rowwise * BUFF_DIM_X;
      thread_amax = 0.0f;
      float in_compute_rowwise[SCALE_DIM_X];
      Vec<IType, PACK_SIZE> in_cached[WAVES];

      // used as an IType container for BF16/FP16 --> MXFP8 CAST ONLY
      Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

      // 1. Read/Compute elements. Find MXFP8-block AMAX
      if constexpr (NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>)) {
        IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;
          // Load elements
          in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 2; ++e) {
            ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
          }
        }
        thread_amax =
            static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
      } else if constexpr (IS_CACHED_ACT_OP) {
        // ensures that all writes to cache made in the section above are visible to all threads
        __syncthreads();
        IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;

          const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
          const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
          const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);

          // Load cached elements
          in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
          // Since TMA requirement for the data alignment is 16B (i.e. cols % 8 == 0, in case of BF16 elements)
          // only single check (w.r.t. column direction) is sufficient to be sure the entire wave is inside the boundaries
          if (!out_of_bounds) {
            if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
              for (int e = 0; e < PACK_SIZE; ++e) {
                thread_amax = fmaxf(thread_amax, fabsf(in_cached[w].data.elt[e]));
              }
            } else {
#pragma unroll
              for (int e = 0; e < PACK_SIZE; e += 2) {
                const IType2 in_cached_2x = {in_cached[w].data.elt[e],
                                             in_cached[w].data.elt[e + 1]};
                ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_cached_2x);
              }
            }
          }
        }
        if constexpr (!std::is_same_v<IType, float>) {
          thread_amax =
              static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        }
      } else {
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;

          Vec<IType, PACK_SIZE> in;
          Vec<IType, PACK_SIZE> act_in;

          in.load_from(&in_sh[shmem_offset_rowwise]);
          if constexpr (IS_DACT) {
            act_in.load_from(&act_in_sh[shmem_offset_rowwise]);
          }
#pragma unroll
          for (int e = 0; e < PACK_SIZE; ++e) {
            const int j = w * PACK_SIZE + e;
            // Compute element
            float elt = static_cast<float>(in.data.elt[e]);
            if constexpr (IS_ACT) {
              elt = OP(elt, {});
            }
            if constexpr (IS_DACT) {
              float act_in_elt = static_cast<float>(act_in.data.elt[e]);
              elt *= OP(act_in_elt, {});
            }

            // If DBIAS was computed in the 1st pass (COLWISE) then no need to compute it again
            if constexpr (IS_DBIAS && (!COLWISE_SCALING)) {
              thread_dbias_rowwise[j] += elt;
            }
            // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
            if constexpr (!std::is_same_v<IType, float>) {
              elt = static_cast<float>(static_cast<IType>(elt));
            }
            if constexpr (COMPUTE_ACTIVATIONS) {
              const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
              const bool swizzled_col_out_of_bounds =
                  (block_offset_X + swizzled_thread_idx >= cols);
              const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
              if (!out_of_bounds) {
                thread_amax = fmaxf(thread_amax, fabsf(elt));
              }
            } else {
              // If no activation, elt is 0 so we can safely do this
              thread_amax = fmaxf(thread_amax, fabsf(elt));
            }
            in_compute_rowwise[j] = elt;
          }
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);
      const size_t stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
      const size_t stage_scales_offset_X = scales_offset_X_rowwise;
      const size_t scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
      scales_rowwise[scale_idx] = biased_exponent;

      const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
      const ptx::floatx2 block_scale_inverse_2x = {block_scale_inverse, block_scale_inverse};

      // 3. Scale elements
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        Vec<OType2, PACK_SIZE / 2> out;
#pragma unroll
        for (int e = 0; e < PACK_SIZE / 2; ++e) {
          IType2 in;
          OType2 &out_pair = reinterpret_cast<OType2 &>(out.data.elt[e]);
          if constexpr (NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>)) {
            in = in_IType[w].data.elt[e];
          } else if constexpr (IS_CACHED_ACT_OP) {
            in.x = in_cached[w].data.elt[2 * e];
            in.y = in_cached[w].data.elt[2 * e + 1];
          } else {
            const int j = w * PACK_SIZE + 2 * e;
            in.x = in_compute_rowwise[j];
            in.y = in_compute_rowwise[j + 1];
          }
          ptx::mul_cvt_2x(out_pair, in, block_scale_inverse_2x);
        }
        const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
        const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
        const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_idx;
        out.store_to(&out_rowwise_sh[shmem_offset_rowwise]);
      }
    }

    __builtin_assume(block_amax >= 0);
    __builtin_assume(thread_amax >= 0);
    block_amax = fmaxf(block_amax, thread_amax);

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t buff_offset = buff * BUFF_DIM;

      if constexpr (ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_rowwise_sh[buff_offset]));
      }
      if constexpr (COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_colwise_sh[buff_offset]));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }

  parity ^= 1;

  if constexpr (IS_DBIAS) {
    float thread_partial_dbias = 0.0f;
    if constexpr (COLWISE_SCALING) {
      thread_partial_dbias = partial_dbias_colwise;
    } else {
      // Reusing dshmem (in_sh) as dbias buffer [HEIGHT x WIDTH]
      // HEIGHT = THREADS_Y
      // WIDTH = THREADS_X * (SCALE_DIM_X + 1)
      // Added extra 1-element padding per thread_X to reduce bank conflicts
      float *partial_dbias_rowwise = reinterpret_cast<float *>(dshmem);

      constexpr size_t DBIAS_BUFF_WIDTH = THREADS_X * (SCALE_DIM_X + 1);

      const size_t shmem_thread_offset =
          tid_Y_rowwise * DBIAS_BUFF_WIDTH + tid_X_rowwise * (SCALE_DIM_X + 1);
#pragma unroll
      for (int w = 0; w < WAVES; ++w) {
        const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
        const size_t swizzled_group_offset = shmem_thread_offset + swizzled_group_idx;
#pragma unroll
        for (int e = 0; e < PACK_SIZE; ++e) {
          const int j = w * PACK_SIZE + e;
          const size_t shmem_elt_idx = swizzled_group_offset + e;
          partial_dbias_rowwise[shmem_elt_idx] = thread_dbias_rowwise[j];
        }
      }
      __syncthreads();
#pragma unroll
      for (int i = 0; i < THREADS_Y; ++i) {
        // Add extra element offset per MXFP8 scaling block [1x32]
        const size_t scaling_block = threadIdx.x / SCALE_DIM_X;
        thread_partial_dbias +=
            partial_dbias_rowwise[i * DBIAS_BUFF_WIDTH + threadIdx.x + scaling_block];
      }
    }
    const size_t dbias_stride = cols;
    const size_t dbias_offset_Y = blockIdx.y;
    const size_t dbias_offset_X = blockIdx.x * CHUNK_DIM_X + threadIdx.x;
    const size_t dbias_idx = dbias_offset_Y * dbias_stride + dbias_offset_X;
    const bool col_out_of_bounds_dbias = (dbias_offset_X >= cols);
    if (!col_out_of_bounds_dbias) {
      dbias_workspace[dbias_idx] = thread_partial_dbias;
    }
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    block_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(block_amax, warp_id);
  }

  if (is_master_thread && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, block_amax);
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace mxfp8_kernel

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

  const size_t block_offset_Y = blockIdx.y * FP8_CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * FP8_CHUNK_DIM_X;

  const size_t tid_Y = threadIdx.x / FP8_THREADS_PER_CHUNK;
  const size_t tid_X = threadIdx.x % FP8_THREADS_PER_CHUNK;

  const size_t thread_offset_Y = tid_Y;
  const size_t thread_offset_X = tid_X;

  const size_t dbias_offset_Y = blockIdx.y + tid_Y;
  const size_t my_column = blockIdx.x * FP8_CHUNK_DIM_X + thread_offset_X;
  const bool col_out_of_bounds = my_column >= cols;
  const size_t dbias_stride = cols;

  float partial_dbias = 0.f;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(TMA_SHMEM_ALIGNMENT)
      IType in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT)
      IType act_in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT)
      OType out_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];

  constexpr size_t shmem_buff_size = sizeof(in_sh) / FP8_BUFFERS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[FP8_ITERATIONS];

  initialize_barriers<FP8_ITERATIONS, FP8_THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  const size_t chunk_offset_Y = block_offset_Y;
  const size_t chunk_offset_X = block_offset_X;

#pragma unroll
  for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
    const size_t chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * FP8_BUFFER_DIM_Y;
    const size_t chunk_stage_offset_X = chunk_offset_X;
    if constexpr (IS_DACT) {
      copy_2d_to_sharedx2(&in_sh[prefetch_buff], &tensor_map_input, chunk_stage_offset_X,
                          chunk_stage_offset_Y, &act_in_sh[prefetch_buff], &tensor_map_act_input,
                          chunk_stage_offset_X, chunk_stage_offset_Y, shmem_buff_size,
                          &mbar[prefetch_buff], is_master_thread);
    } else {
      copy_2d_to_shared(&in_sh[prefetch_buff], &tensor_map_input, chunk_stage_offset_X,
                        chunk_stage_offset_Y, shmem_buff_size, &mbar[prefetch_buff],
                        is_master_thread);
    }
  }

#pragma unroll
  for (int iter = 0; iter < FP8_ITERATIONS; ++iter) {
    const size_t buff = iter % FP8_BUFFERS_NUM;
    const size_t next_iter = iter + FP8_PREFETCH_BUFFERS_NUM;
    const size_t row_base = block_offset_Y + iter * FP8_BUFFER_DIM_Y;
    if (next_iter < FP8_ITERATIONS) {
      const size_t next_buff = next_iter % FP8_BUFFERS_NUM;
      const size_t chunk_it_offset_y = chunk_offset_Y + next_iter * FP8_BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;
      if constexpr (IS_DACT) {
        copy_2d_to_sharedx2(&in_sh[next_buff], &tensor_map_input, chunk_it_offset_x,
                            chunk_it_offset_y, &act_in_sh[next_buff], &tensor_map_act_input,
                            chunk_it_offset_x, chunk_it_offset_y, shmem_buff_size, &mbar[next_iter],
                            is_master_thread);
      } else {
        copy_2d_to_shared(&in_sh[next_buff], &tensor_map_input, chunk_it_offset_x,
                          chunk_it_offset_y, shmem_buff_size, &mbar[next_iter], is_master_thread);
      }
    }

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

#pragma unroll
    for (int stage = 0; stage < FP8_BUFF_STAGES_NUM; ++stage) {
      const size_t stage_offset_Y = stage;
      const size_t shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const size_t shmem_offset_x = thread_offset_X;
      const size_t row = row_base + shmem_offset_y;
      const bool row_out_of_bounds = row >= rows;
      const bool out_of_bounds = col_out_of_bounds || row_out_of_bounds;

      float elt = static_cast<float>(in_sh[buff][shmem_offset_y][shmem_offset_x]);
      if constexpr (IS_DACT) {
        float act_in_elt = static_cast<float>(act_in_sh[buff][shmem_offset_y][shmem_offset_x]);
        elt *= OP(act_in_elt, {});
      }
      if constexpr (IS_DBIAS) {
        if constexpr (IS_DACT) {
          if (!out_of_bounds) {
            partial_dbias += elt;
          }
        } else {
          // If no activation, elt is 0 so we can safely do this
          partial_dbias += elt;
        }
      }
      __builtin_assume(amax >= 0);
      if (IS_DACT) {
        if (!out_of_bounds) {
          amax = fmaxf(amax, fabsf(elt));
        }
      } else {
        // If no activation, elt is 0 so we can safely do this
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
      const size_t chunk_it_offset_y = chunk_offset_Y + iter * FP8_BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;
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
    const size_t dbias_offset_X = my_column;
    const size_t dbias_offset = dbias_offset_Y * dbias_stride + dbias_offset_X;
    if (!col_out_of_bounds) {
      dbias_workspace[dbias_offset] = partial_dbias;
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

  destroy_barriers<FP8_ITERATIONS>(mbar, is_master_thread);
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

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    cast_fp8_1D_kernel(const IType *input_ptr, OType *output_ptr, float *const amax_ptr,
                       float *const scale_inv_ptr, const float *const scale_ptr, const size_t N) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const size_t block_offset = blockIdx.x * ELEMS_PER_BLOCK;
  const IType *input = input_ptr + block_offset;
  OType *output = output_ptr + block_offset;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) IType in_sh[SHMEM_BUFFERS][SHMEM_DIM];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) OType out_sh[SHMEM_BUFFERS][SHMEM_DIM];

  constexpr size_t transaction_size_IN = sizeof(in_sh) / SHMEM_BUFFERS;
  constexpr size_t transaction_size_OUT = sizeof(out_sh) / SHMEM_BUFFERS;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_BLOCK>(mbar, is_master_thread);

  int parity = 0;

  copy_1d_to_shared(&(in_sh[0]), input, transaction_size_IN, &(mbar[0]), is_master_thread);

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const size_t buff = iter % SHMEM_BUFFERS;
    const size_t it_offset = iter * SHMEM_DIM;

    const size_t next_iter = iter + 1;
    const size_t next_buff = next_iter % SHMEM_BUFFERS;
    const size_t next_iter_offset = next_iter * SHMEM_DIM;

    if (next_iter < ITERATIONS) {
      copy_1d_to_shared(&(in_sh[next_buff]), input + next_iter_offset, transaction_size_IN,
                        &(mbar[next_iter]), is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_ITERATION; ++chunk) {
      const size_t shmem_offset = chunk * CHUNK_SIZE + threadIdx.x;
      float elt = static_cast<float>(in_sh[buff][shmem_offset]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(elt));
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

  destroy_barriers<ITERATIONS>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

constexpr size_t DBIAS_THREADS_PER_BLOCK = 256;
template <int nvec, typename OType>
__global__ void __launch_bounds__(DBIAS_THREADS_PER_BLOCK)
    reduce_dbias_kernel(OType *const dbias_output, const float *const dbias_partial,
                        const size_t rows, const size_t cols) {
  using ComputeVec = Vec<float, nvec>;
  using OutputVec = Vec<OType, nvec>;

  const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

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
  constexpr size_t reduce_dbias_store_bytes = 8;  // stg.64
  constexpr size_t reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(IType);

  NVTE_CHECK(cols % reduce_dbias_nvec == 0, "Unsupported shape.");
  const size_t reduce_dbias_num_blocks = DIVUP(cols, DBIAS_THREADS_PER_BLOCK * reduce_dbias_nvec);

  reduce_dbias_kernel<reduce_dbias_nvec, IType>
      <<<reduce_dbias_num_blocks, DBIAS_THREADS_PER_BLOCK, 0, stream>>>(
          reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, rows, cols);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
static void cast_fp8_1D(const Tensor &input, Tensor *output, cudaStream_t stream) {
  const size_t N = product(input.data.shape);

  const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  const size_t chunks = DIVUP(N, CHUNK_SIZE);
  const size_t blocks = DIVUP(chunks, CHUNKS_PER_BLOCK);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  const float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(blocks);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const IType *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
          OType *output_ptr = reinterpret_cast<OType *>(output->data.dptr);

          cast_fp8_1D_kernel<IS_ACT, ParamOP, OP, IType, OType><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, amax_ptr, scale_inv_ptr, scale_ptr, N););  // NOLINT(*)
  );                                                                            // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void cast_fp8_2D(const Tensor &input, const Tensor *act_input, Tensor *output, Tensor *dbias,
                 Tensor *workspace, cudaStream_t stream) {
  checkCuDriverContext(stream);

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, FP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, FP8_CHUNK_DIM_X);
  const size_t blocks_Y = chunks_Y;
  const size_t blocks_X = chunks_X;

  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

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

          create_2D_tensor_map(tensor_map_input, input.data, rows, cols, FP8_SHMEM_DIM_Y,
                               FP8_SHMEM_DIM_X, cols, 0, typeToNumBits(input.data.dtype));

          if constexpr (IS_DACT) {
            create_2D_tensor_map(tensor_map_act_input, act_input->data, rows, cols, FP8_SHMEM_DIM_Y,
                                 FP8_SHMEM_DIM_X, cols, 0, typeToNumBits(input.data.dtype));
          }

          create_2D_tensor_map(tensor_map_output, output->data, rows, cols, FP8_SHMEM_DIM_Y,
                               FP8_SHMEM_DIM_X, cols, 0, typeToNumBits(output->data.dtype));

          cast_fp8_2D_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType>
          <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_act_input, tensor_map_output,
                                       workspace_ptr, amax_ptr, scale_inv_ptr, scale_ptr, rows,
                                       cols);
          NVTE_CHECK_CUDA(cudaGetLastError());

          if constexpr (IS_DBIAS) {
            reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void mxfp8_quantize(const Tensor &input, const Tensor *act_input,
                    const Tensor *noop,  // TODO (ksivamani)
                    Tensor *output, Tensor *dbias, Tensor *workspace, const bool pdl_sync,
                    const bool pdl_trigger, cudaStream_t stream) {
  using namespace mxfp8_kernel;
  checkCuDriverContext(stream);

  bool use_rowwise_scaling = output->has_data();
  bool use_colwise_scaling = output->has_columnwise_data();
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");

  if (use_rowwise_scaling) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");
  }
  if (use_colwise_scaling) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scaling tensor must be allocated");
  }
  CheckNoopTensor(*noop, "cast_noop");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  constexpr bool CAST_DBIAS_ONLY = IS_DBIAS && (!IS_DACT) && (!IS_ACT);

  constexpr size_t CHUNK_DIM_Y = CAST_DBIAS_ONLY ? 128 : 64;
  constexpr size_t CHUNK_DIM_X = CAST_DBIAS_ONLY ? 128 : 64;
  constexpr size_t THREADS_PER_CHUNK = CAST_DBIAS_ONLY ? 128 : 64;

  constexpr size_t THREADS_X = CHUNK_DIM_X / SCALE_DIM_X;
  constexpr size_t THREADS_Y = THREADS_PER_CHUNK / THREADS_X;
  constexpr size_t BUFF_DIM_Y = THREADS_Y;
  constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_PER_CHUNK;

  const size_t scale_stride_rowwise = use_rowwise_scaling ? output->scale_inv.shape[1] : 1;
  const size_t scale_stride_colwise =
      use_colwise_scaling ? output->columnwise_scale_inv.shape[1] : 1;

  e8m0_t *const scales_rowwise_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(output->scale_inv.dptr) : nullptr;
  e8m0_t *const scales_colwise_ptr =
      use_colwise_scaling ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;
  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  ScalingType scaling_type;
  if (use_rowwise_scaling && (!use_colwise_scaling)) {
    scaling_type = ScalingType::ROWWISE;
  } else if ((!use_rowwise_scaling) && use_colwise_scaling) {
    scaling_type = ScalingType::COLWISE;
  } else if (use_rowwise_scaling && use_colwise_scaling) {
    scaling_type = ScalingType::BIDIMENSIONAL;
  }

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.dtype(), "DBias must have the same type as input.");
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
  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_input{};
          alignas(64) CUtensorMap tensor_map_act_input{};
          alignas(64) CUtensorMap tensor_map_output_rowwise{};
          alignas(64) CUtensorMap tensor_map_output_colwise{};

          constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
          constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

          create_2D_tensor_map(tensor_map_input, input.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                               cols, 0, input_type_bit_size);

          if constexpr (IS_DACT) {
            create_2D_tensor_map(tensor_map_act_input, act_input->data, rows, cols, BUFF_DIM_Y,
                                 BUFF_DIM_X, cols, 0, input_type_bit_size);
          }

          if (use_rowwise_scaling) {
            create_2D_tensor_map(tensor_map_output_rowwise, output->data, rows, cols, BUFF_DIM_Y,
                                 BUFF_DIM_X, cols, 0, output_type_bit_size);
          }

          if (use_colwise_scaling) {
            create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, cols, 0, output_type_bit_size);
          }

          constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
          constexpr size_t buff_elems_total = mxfp8_kernel::BUFFS_NUM * buff_elems;
          constexpr size_t input_buff_size = (buff_elems_total * input_type_bit_size) / 8;
          constexpr size_t output_buff_size = (buff_elems_total * output_type_bit_size) / 8;
          constexpr size_t buff_size_aligned_in =
              DIVUP_TO_MULTIPLE(input_buff_size, TMA_SHMEM_ALIGNMENT);
          constexpr size_t buff_size_aligned_out =
              DIVUP_TO_MULTIPLE(output_buff_size, TMA_SHMEM_ALIGNMENT);

          constexpr size_t elt_input_mem = buff_size_aligned_in;
          constexpr size_t act_input_mem = (IS_DACT ? buff_size_aligned_in : 0);
          constexpr size_t in_mem = elt_input_mem + act_input_mem;

          const size_t out_rowwise_mem = (use_rowwise_scaling ? buff_size_aligned_out : 0);
          const size_t out_colwise_mem = (use_colwise_scaling ? buff_size_aligned_out : 0);
          const size_t out_mem = out_rowwise_mem + out_colwise_mem;

          const size_t dshmem_size = in_mem + out_mem + TMA_SHMEM_ALIGNMENT;

          cudaLaunchConfig_t cfg = {grid, block_size, dshmem_size, stream, NULL, 0};
          cudaLaunchAttribute attribute[1];
          attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
          attribute[0].val.programmaticStreamSerializationAllowed = 1;
          // This kernel will only be called on sm100+, so no need to check sm_arch
          cfg.attrs = attribute;
          cfg.numAttrs = 1;

          switch (scaling_type) {
            case ScalingType::ROWWISE:
              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, true,
                                       false, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              NVTE_CHECK_CUDA(cudaLaunchKernelEx(
                  &cfg,
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, true,
                                       false, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>,
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr, noop_ptr,
                  workspace_ptr, amax_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise,
                  pdl_sync, pdl_trigger));
              break;
            case ScalingType::COLWISE:
              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, false,
                                       true, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              NVTE_CHECK_CUDA(cudaLaunchKernelEx(
                  &cfg,
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, false,
                                       true, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>,
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr, noop_ptr,
                  workspace_ptr, amax_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise,
                  pdl_sync, pdl_trigger));
              break;
            case ScalingType::BIDIMENSIONAL:
              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, true,
                                       true, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              NVTE_CHECK_CUDA(cudaLaunchKernelEx(
                  &cfg,
                  cast_mxfp8_2D_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, true,
                                       true, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>,
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr, noop_ptr,
                  workspace_ptr, amax_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise,
                  pdl_sync, pdl_trigger));
              break;
          }

          if constexpr (IS_DBIAS) {
            reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

namespace detail {

using Empty = transformer_engine::Empty;

__device__ inline float identity(float value, const Empty &) { return value; }

struct DequantizeParam {
  const float *scale_inv;
};

__device__ inline float dequantize_func(float value, const DequantizeParam &param) {
  return value * (*(param.scale_inv));
}

}  // namespace detail

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
void CastVectorizedUnaryKernelLauncher(const Tensor &input, const Tensor *noop, Tensor *output,
                                       cudaStream_t stream) {
  constexpr float (*UnaryOP)(float, const ParamOP &) = (OP == nullptr) ? detail::identity : OP;
  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,
          if (!is_fp8_dtype(output->data.dtype) || is_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            VectorizedUnaryKernelLauncher<nvec, ParamOP, UnaryOP>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<const fp32 *>(noop->data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
void CastVectorizedUnaryGradKernelLauncher(const Tensor &grad, const Tensor *input, Tensor *output,
                                           cudaStream_t stream) {
  constexpr float (*UnaryOP)(float, const ParamOP &) = (OP == nullptr) ? detail::identity : OP;
  const size_t N = product(input->data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,
          if (!is_fp8_dtype(output->data.dtype) || is_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            VectorizedUnaryGradKernelLauncher<nvec, ParamOP, UnaryOP>(
                reinterpret_cast<const IType *>(grad.data.dptr),
                reinterpret_cast<const IType *>(input->data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

namespace {

static bool is_full_tile_1D_tensor(const Tensor *const t) {
  const size_t N = product(t->data.shape);
  const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
  return isFullTile;
}

bool dimensions_supported_by_TMA(const Tensor *const t) {
  const size_t cols = t->flat_last_dim();
  constexpr size_t TMA_bytes = 16;
  const size_t alignment_requirement = (TMA_bytes * 8) / typeToNumBits(t->dtype());
  return cols % alignment_requirement == 0;
}

}  // namespace

// Supported by the Arch >= 10.0
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void fp8_quantize_arch_ge_100(const Tensor &input, const Tensor *act_input, const Tensor *noop,
                              Tensor *output, Tensor *dbias, Tensor *workspace,
                              cudaStream_t stream) {
  switch (output->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (!IS_DBIAS && !IS_DACT) {
        if (is_full_tile_1D_tensor(output) && is_fp8_dtype(output->dtype()) &&
            is_aligned_tensor_data(input, TMA_GMEM_ALIGNMENT) &&
            is_aligned_tensor_data(*output, TMA_GMEM_ALIGNMENT)) {
          // Aligned AND FP8
          cast_fp8_1D<IS_ACT, ParamOP, OP>(input, output, stream);
        } else {
          // Unaligned
          CastVectorizedUnaryKernelLauncher<ParamOP, OP>(input, noop, output, stream);
        }
      } else if (!IS_DBIAS && IS_DACT) {
        if (dimensions_supported_by_TMA(output) && is_fp8_dtype(output->dtype()) &&
            is_aligned_tensor_data(input, TMA_GMEM_ALIGNMENT) &&
            is_aligned_tensor_data(*output, TMA_GMEM_ALIGNMENT) &&
            is_aligned_tensor_data(*act_input, TMA_GMEM_ALIGNMENT)) {
          // Aligned AND FP8 (+dAct)
          cast_fp8_2D<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias, workspace,
                                                      stream);
        } else {
          // Unaligned
          CastVectorizedUnaryGradKernelLauncher<ParamOP, OP>(input, act_input, output, stream);
        }
      } else {
        cast_fp8_2D<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias, workspace,
                                                    stream);
      }
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

// Supported by the Arch < 10.0
template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void fp8_quantize_arch_l_100(const Tensor &input, const Tensor *act_input, const Tensor *noop,
                             Tensor *output, Tensor *dbias, Tensor *workspace,
                             cudaStream_t stream) {
  if (!is_tensor_scaling(output->scaling_mode) || IS_DBIAS) {
    // zhongboz: should we just ignore IS_ACT here?
    NVTE_ERROR("Not implemented scaling mode or fusion: " + to_string(output->scaling_mode) +
               " or IS_DBIAS=true" + " on GPU with compute capability < 10.0.");
  }
  switch (output->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (!IS_DACT) {
        CastVectorizedUnaryKernelLauncher<ParamOP, OP>(input, noop, output, stream);
      } else {
        CastVectorizedUnaryGradKernelLauncher<ParamOP, OP>(input, act_input, output, stream);
      }
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
  }
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void fp8_quantize(const Tensor &input, const Tensor *act_input, const Tensor *noop, Tensor *output,
                  Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias != nullptr);
    CheckOutputTensor(*dbias, "dbias");
  }
  if constexpr (IS_DACT) {
    NVTE_CHECK(act_input != nullptr);
    CheckInputTensor(*act_input, "activation_input");
    NVTE_CHECK(input.dtype() == act_input->dtype(), "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input->data.shape, "Shapes of both inputs must match.");
  }

  NVTE_CHECK(!is_fp8_dtype(input.dtype()), "Input must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  // Supported by the Arch >= 10.0
  if (is_supported_by_CC_100()) {
    fp8_quantize_arch_ge_100<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(input, act_input, noop, output,
                                                                     dbias, workspace, stream);
  } else {
    // Supported by the Arch < 10.0
    fp8_quantize_arch_l_100<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(input, act_input, noop, output,
                                                                    dbias, workspace, stream);
  }
}

namespace detail {

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void quantize_helper(const NVTETensor input, const NVTETensor grad, NVTETensor output,
                     NVTETensor dbias, NVTETensor workspace,
                     const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  const Tensor *input_tensor;
  const Tensor *activation_input_tensor;
  if constexpr (IS_DBIAS || IS_DACT) {
    // backward - input is incoming gradient
    input_tensor = convertNVTETensorCheck(grad);
    activation_input_tensor = convertNVTETensor(input);
  } else {
    // forward = input is activation input
    input_tensor = convertNVTETensorCheck(input);
    activation_input_tensor = nullptr;
  }
  auto output_tensor = convertNVTETensorCheck(output);
  auto dbias_tensor = convertNVTETensor(dbias);
  auto workspace_tensor = convertNVTETensor(workspace);

  const QuantizationConfig *quant_config_cpp =
      reinterpret_cast<const QuantizationConfig *>(quant_config);

  // extract noop tensor from quant_config_cpp if it's not null
  const NVTETensor noop = quant_config_cpp ? quant_config_cpp->noop_tensor : nullptr;
  const auto noop_tensor = noop != nullptr ? *(convertNVTETensorCheck(noop)) : Tensor();
  const bool pdl_sync = quant_config_cpp ? quant_config_cpp->pdl_sync : false;
  const bool pdl_trigger = quant_config_cpp ? quant_config_cpp->pdl_trigger : false;

  switch (output_tensor->scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      if (output_tensor->has_columnwise_data()) {
        NVTE_CHECK(output_tensor->has_data(),
                   "Quantizing in only the columnwise direction not supported yet!");
        if constexpr (!IS_DBIAS && !IS_DACT && !IS_ACT) {
          cast_transpose(*input_tensor, noop_tensor, output_tensor, stream);
        } else {
          cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, float, ParamOP, OP>(
              *input_tensor, activation_input_tensor, output_tensor, dbias_tensor, workspace_tensor,
              stream);
        }
      } else if (output_tensor->has_data()) {
        fp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
            *input_tensor, activation_input_tensor, &noop_tensor, output_tensor, dbias_tensor,
            workspace_tensor, stream);
      }
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      mxfp8_quantize<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
          *input_tensor, activation_input_tensor, &noop_tensor, output_tensor, dbias_tensor,
          workspace_tensor, pdl_sync, pdl_trigger, stream);
      break;
    }
    case NVTE_BLOCK_SCALING_2D: {
      // TODO(kwyss): IS_BIAS, IS_DACT, IS_ACT, ParamOP, OP parameters support.
      NVTE_CHECK((!IS_DBIAS && !IS_DACT && !IS_ACT),
                 "IS_DBIAS, IS_DACT, and IS_ACT not implemented for NVTE_BLOCK_SCALING_2D");
      bool force_pow_2_scales = quant_config_cpp ? quant_config_cpp->force_pow_2_scales : true;
      float epsilon = quant_config_cpp ? quant_config_cpp->amax_epsilon : 0.0f;
      quantize_transpose_square_blockwise(
          input_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon,
          /*return_transpose=*/output_tensor->has_columnwise_data(), force_pow_2_scales,
          /*noop_tensor=*/noop_tensor.data, pdl_sync, pdl_trigger, stream);
      break;
    }
    case NVTE_BLOCK_SCALING_1D: {
      // TODO(kwyss): IS_BIAS, IS_DACT, IS_ACT, ParamOP, OP parameters support.
      NVTE_CHECK((!IS_DBIAS && !IS_DACT && !IS_ACT),
                 "IS_DBIAS, IS_DACT, and IS_ACT not implemented for NVTE_BLOCK_SCALING_1D");
      bool force_pow_2_scales = quant_config_cpp ? quant_config_cpp->force_pow_2_scales : false;
      float epsilon = quant_config_cpp ? quant_config_cpp->amax_epsilon : 0.0f;
      FP8BlockwiseRowwiseOption rowwise_option = FP8BlockwiseRowwiseOption::NONE;
      FP8BlockwiseColumnwiseOption columnwise_option = FP8BlockwiseColumnwiseOption::NONE;
      if (output_tensor->has_data()) {
        bool rowwise_compact = quant_config_cpp
                                   ? quant_config_cpp->float8_block_scale_tensor_format ==
                                         Float8BlockScaleTensorFormat::COMPACT
                                   : false;
        rowwise_option = rowwise_compact ? FP8BlockwiseRowwiseOption::ROWWISE_COMPACT
                                         : FP8BlockwiseRowwiseOption::ROWWISE_GEMM_READY;
      }
      if (output_tensor->has_columnwise_data()) {
        bool columnwise_compact = quant_config_cpp
                                      ? quant_config_cpp->float8_block_scale_tensor_format ==
                                            Float8BlockScaleTensorFormat::COMPACT
                                      : false;
        columnwise_option = columnwise_compact
                                ? FP8BlockwiseColumnwiseOption::COLUMNWISE_COMPACT
                                : FP8BlockwiseColumnwiseOption::COLUMNWISE_GEMM_READY;
      }
      quantize_transpose_vector_blockwise(
          input_tensor->data, output_tensor->scale_inv, output_tensor->columnwise_scale_inv,
          output_tensor->data, output_tensor->columnwise_data, epsilon, rowwise_option,
          columnwise_option, force_pow_2_scales, noop_tensor.data, pdl_sync, pdl_trigger, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(output_tensor->scaling_mode) + ".");
  }
}

}  // namespace detail
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_KERNELS_CUH_
