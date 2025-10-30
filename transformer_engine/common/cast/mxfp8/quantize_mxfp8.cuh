/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_mxfp8.cuh
 *  \brief CUDA kernels to quantize to MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_MXFP8_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_MXFP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {

__device__ __forceinline__ void
mbarrier_arrive_expect_tx_cta_relaxed_shared_cta(uint64_t* mbar_ptr) {
  uint32_t mbar = __cvta_generic_to_shared(mbar_ptr);
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], 16;" :: "r"(mbar));
}

__device__ __forceinline__ void
mbarrier_arrive_relaxed_cta_shared_cta(uint64_t* mbar_ptr) {
  uint32_t mbar = __cvta_generic_to_shared(mbar_ptr);
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0], 1;" :: "r"(mbar));
}

__device__ __forceinline__ void
mbarrier_arrive_release_cta_shared_cta(uint64_t* mbar_ptr) {
  uint32_t mbar = __cvta_generic_to_shared(mbar_ptr);
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;" :: "r"(mbar));
}

__device__ __forceinline__ void
clusterlaunchcontrol_try_cancel_async_shared_cta_mbarrier_complete_tx_bytes(
  uint64_t* mbar_ptr, __uint128_t* response_data_ptr) {
  uint32_t mbar = __cvta_generic_to_shared(mbar_ptr);
  uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
  asm volatile("clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 "
               "[%0], [%1];" :: "r"(workID_response), "r"(mbar));
}

__device__ __forceinline__ void
mbarrier_wait_parity_acquire_cta_shared_cta(uint64_t* mbar_ptr, uint32_t phase_parity) {
  uint32_t mbar = __cvta_generic_to_shared(mbar_ptr);
  asm volatile(
    "{\n\t"
      ".reg .b64 r1; \n\t"
      ".reg .pred waitComplete; \n\t"         // predicate representing if barrier condition is met
      "WAIT: \n\t"                            // loop around barrier wait
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [%0], %1; \n\t"
      "@waitComplete bra DONE; \n\t"          // mbarrier conditions are met
      "bra WAIT; \n\t"                        // just a time-out, try again
      "DONE: \n\t"
    "}\n\t"
    :
    : "r"(mbar),
      "r"(phase_parity)
    : "memory"
  );
}

__device__ __forceinline__ void
mbarrier_wait_parity_relaxed_cta_shared_cta(uint64_t* mbar_ptr, uint32_t phase_parity) {
  uint32_t mbar = __cvta_generic_to_shared(mbar_ptr);
  asm volatile(
    "{\n\t"
      ".reg .b64 r1; \n\t"
      ".reg .pred waitComplete; \n\t"         // predicate representing if barrier condition is met
      "WAIT: \n\t"                            // loop around barrier wait
      "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64  waitComplete, [%0], %1; \n\t"
      "@waitComplete bra DONE; \n\t"          // mbarrier conditions are met
      "bra WAIT; \n\t"                        // just a time-out, try again
      "DONE: \n\t"
    "}\n\t"
    :
    : "r"(mbar),
      "r"(phase_parity)
    : "memory"
  );
}

__device__ __forceinline__ void
get_cancelled_cta_2D_id(__uint128_t* response_data_ptr, int32_t& ctaid_X, int32_t& ctaid_Y) {
  uint32_t workID_response = __cvta_generic_to_shared(response_data_ptr);
  asm volatile(
    "{\n\t"
      ".reg .s32 x_ctaid; \n\t"
      ".reg .s32 y_ctaid; \n\t"
      "mov .s32 x_ctaid, -1; \n\t"
      "mov .s32 y_ctaid, -1; \n\t"
      ".reg.b128 try_cancel_response; \n\t"
      "ld.shared.b128 try_cancel_response, [%2]; \n\t"
      ".reg .pred P1; \n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P1, try_cancel_response; \n\t"
      "@P1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x_ctaid, y_ctaid, _, _}, try_cancel_response; \n\t"
      "mov .s32 %0, x_ctaid; \n\t"
      "mov .s32 %1, y_ctaid; \n\t"
    "}\n\t"
      : "=r"(ctaid_X),
        "=r"(ctaid_Y)
      : "r"(workID_response)
      : "memory"
  );
}

__device__ __forceinline__ unsigned char*
align_ptr_per_TMA_requirements(unsigned char* p) {
  size_t addr = reinterpret_cast<size_t>(p);
  addr = (addr + TMA_SHMEM_ALIGNMENT - 1) & ~(TMA_SHMEM_ALIGNMENT - 1);
  return reinterpret_cast<unsigned char*>(addr);
}

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t PREFETCH_STAGES = 1;
constexpr size_t BUFFS_NUM = PREFETCH_STAGES + 1;
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
    quantize_mxfp8_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                          const __grid_constant__ CUtensorMap tensor_map_act_input,
                          const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                          const __grid_constant__ CUtensorMap tensor_map_output_colwise,
                          e8m0_t *const scales_rowwise, e8m0_t *const scales_colwise,
                          const float *noop, float *const dbias_workspace, float *const amax_ptr,
                          const size_t rows, const size_t cols, const size_t scale_stride_rowwise,
                          const size_t scale_stride_colwise) {
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
  static_assert(BUFF_DIM_Y == 32);

  constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
  static_assert(STAGES >= 1);

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && ROWWISE_SCALING && COLWISE_SCALING;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X;
  const size_t tid_Y_colwise = 0;
  const size_t tid_X_colwise = threadIdx.x;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;
  const size_t thread_offset_Y_colwise = tid_Y_colwise;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  // helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out = DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  constexpr size_t elt_input_mem = buff_size_aligned_in;
  constexpr size_t act_input_mem = (IS_DACT ? buff_size_aligned_in : 0);
  constexpr size_t in_mem = elt_input_mem + act_input_mem;

  constexpr size_t out_mem_rowwise = (ROWWISE_SCALING ? buff_size_aligned_out : 0);

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char* dshmem = align_ptr_per_TMA_requirements(dynamic_shmem);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh_ptr = reinterpret_cast<IType *>(dshmem);
  IType *act_in_sh_ptr = reinterpret_cast<IType *>(dshmem + elt_input_mem);

  OType *out_rowwise_data_sh_ptr = reinterpret_cast<OType *>(dshmem + in_mem);
  OType *out_colwise_data_sh_ptr = reinterpret_cast<OType *>(dshmem + in_mem + out_mem_rowwise);
  IType *cached_act_sh_ptr = in_sh_ptr;  // in_sh_ptr is used as a cache buffer

  // IN Buff
  IType (&in_sh)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X] =
      *reinterpret_cast<IType (*)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X]>(in_sh_ptr);

  // Activations IN Buff
  IType (&act_in_sh)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X] =
      *reinterpret_cast<IType (*)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X]>(act_in_sh_ptr);

  // Rowwise OUT Buff
  OType (&out_rowwise_data_sh)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X] =
      *reinterpret_cast<OType (*)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X]>(out_rowwise_data_sh_ptr);

  // Colwise OUT Buff
  OType (&out_colwise_data_sh)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X] =
      *reinterpret_cast<OType (*)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X]>(out_colwise_data_sh_ptr);

  // Cached activations Buff
  IType (&cached_act_sh)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X] =
      *reinterpret_cast<IType (*)[BUFFS_NUM][BUFF_DIM_Y][BUFF_DIM_X]>(cached_act_sh_ptr);

  constexpr size_t shmem_buff_size = (IS_DACT ? 2 : 1) * buff_size_aligned_in / BUFFS_NUM;

  const bool leading_thread = (threadIdx.x == 0);

  __shared__ uint64_t workID_mbar;
  __shared__ __uint128_t workID_response;

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];
  int IN_buff_readable_parity[BUFFS_NUM];
  int ctaid_parity = 0;

  // Initialize shared memory barriers with the number of threads participating in them.
  if (leading_thread) {
    #pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
      IN_buff_readable_parity[buff] = 0;
    }
    ptx::mbarrier_init(&workID_mbar, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();


  float CTA_amax = 0.0f;

  // Coordinates of the first chunk (CTA) to process
  int32_t ctaid_X = blockIdx.x;
  int32_t ctaid_Y = blockIdx.y;
  
  bool job_finished = false;
  int buff_in = 0;
  int buff_out = 0;

  // Prefetch input data only when processing the first chunk,
  // which enables the one-iteration overlap throughout the entire kernel life
  #pragma unroll
  for (size_t stage = 0; stage < PREFETCH_STAGES; ++stage) {
    const int buff = stage;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    // Offsets change, because coordinates of the next "to-be-prefetched" CTA do also chage
    const size_t block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
    const size_t block_offset_X = ctaid_X * CHUNK_DIM_X;

    const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
    const size_t global_offset_X = block_offset_X;

    uint64_t* barrier = &IN_buff_readable_mbar[buff];
    if (leading_thread) {
      // Arrive on the barrier and tell how many bytes are expected to come in.
      ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);
      // Initiate bulk tensor copy
      {
        uint64_t* dst = reinterpret_cast<uint64_t *>(&in_sh[buff]);
        const uint64_t* src = reinterpret_cast<const uint64_t *>(&tensor_map_input);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y, barrier);
      }
      if constexpr (IS_DACT) {
        uint64_t* dst = reinterpret_cast<uint64_t *>(&act_in_sh[buff]);
        const uint64_t* src = reinterpret_cast<const uint64_t *>(&tensor_map_act_input);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y, barrier);
      }
    }
  }

  while (!job_finished) {

    float partial_dbias_colwise = 0.0f;
    float thread_dbias_rowwise[SCALE_DIM_X];
    if constexpr (IS_DBIAS) {
      #pragma unroll
      for (int j = 0; j < SCALE_DIM_X; ++j) {
        thread_dbias_rowwise[j] = 0.0f;
      }
    }

    const size_t block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
    const size_t block_offset_X = ctaid_X * CHUNK_DIM_X;
    const size_t scales_block_offset_Y_rowwise = ctaid_Y * CHUNK_DIM_Y;
    const size_t scales_block_offset_X_rowwise = ctaid_X * CHUNK_DIM_X / SCALE_DIM_X;
    const size_t scales_block_offset_Y_colwise = ctaid_Y * CHUNK_DIM_Y / SCALE_DIM_Y;
    const size_t scales_block_offset_X_colwise = ctaid_X * CHUNK_DIM_X;

    const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
    const size_t row_base_colwise = block_offset_Y + thread_offset_Y_colwise;
    const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

    const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

    const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
    const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
    const size_t scales_offset_Y_colwise = scales_block_offset_Y_colwise + tid_Y_colwise;
    const size_t scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

    const bool rowwise_scale_is_within_bounds = scales_offset_X_rowwise < cols;

    const size_t dbias_offset_Y = ctaid_Y;
    const size_t dbias_offset_X = ctaid_X * CHUNK_DIM_X + threadIdx.x;

    if (leading_thread) {
      mbarrier_arrive_expect_tx_cta_relaxed_shared_cta(&workID_mbar);
      clusterlaunchcontrol_try_cancel_async_shared_cta_mbarrier_complete_tx_bytes(&workID_mbar, &workID_response);
    }

    #pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      const size_t stage_offset_Y = stage * BUFF_DIM_Y;
  
      if (stage == STAGES - PREFETCH_STAGES) {
        mbarrier_wait_parity_acquire_cta_shared_cta(&workID_mbar, ctaid_parity);
        get_cancelled_cta_2D_id(&workID_response, ctaid_X, ctaid_Y);
        if (ctaid_X == -1 && ctaid_Y == -1) {
          job_finished = true;
        }
        ctaid_parity ^= 1;
      }

      // Prefetch next stage Input data
      if (!job_finished || (stage < STAGES - PREFETCH_STAGES)) {
        const size_t next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM; 
        const size_t next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES; 
        const size_t next_prefetch_stage_offset_Y = next_prefetch_stage * BUFF_DIM_Y;

        // Offsets change, because coordinates of the next "to-be-prefetched" CTA do also chage
        const size_t block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
        const size_t block_offset_X = ctaid_X * CHUNK_DIM_X;

        const size_t global_offset_Y = block_offset_Y + next_prefetch_stage_offset_Y;
        const size_t global_offset_X = block_offset_X;

        uint64_t* barrier = &IN_buff_readable_mbar[next_prefetch_buff];
        if (leading_thread) {
          // Arrive on the barrier and tell how many bytes are expected to come in.
          ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);
          // Initiate bulk tensor copy
          {
            uint64_t* dst = reinterpret_cast<uint64_t *>(&in_sh[next_prefetch_buff]);
            const uint64_t* src = reinterpret_cast<const uint64_t *>(&tensor_map_input);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y, barrier);  
          }
          if constexpr (IS_DACT) {
            uint64_t* dst = reinterpret_cast<uint64_t *>(&act_in_sh[next_prefetch_buff]);
            const uint64_t* src = reinterpret_cast<const uint64_t *>(&tensor_map_act_input);
            ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y, barrier);  
          }
        }
        ptx::fence_proxy_async_shared_cta();
      }
  
      // Wait for the data to have arrived
      mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in], IN_buff_readable_parity[buff_in]);
      IN_buff_readable_parity[buff_in] ^= 1;

      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      // ptx::cp_async_bulk_wait_group_read<PREFETCH_STAGES>();
      ptx::cp_async_bulk_wait_group_read<1>();
  
      float thread_amax = 0.0f;
      if constexpr (COLWISE_SCALING) {
        thread_amax = 0.0f;
        float in_compute_colwise[BUFF_DIM_Y];
        IType in_colwise_IType[BUFF_DIM_Y];
  
        // 1. Read/Compute elements. Find MXFP8-block AMAX
        if constexpr (NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>)) {
          IType thread_amax_f16 = static_cast<IType>(0.0f);
          #pragma unroll
          for (int i = 0; i < BUFF_DIM_Y; ++i) {
            in_colwise_IType[i] = in_sh[buff_in][i][tid_X_colwise];
            thread_amax_f16 = __hmax(thread_amax_f16, __habs(in_colwise_IType[i]));
          }
          thread_amax = static_cast<float>(thread_amax_f16);
        } else {
          #pragma unroll
          for (int i = 0; i < BUFF_DIM_Y; ++i) {
            float elt = static_cast<float>(in_sh[buff_in][i][tid_X_colwise]);
            if constexpr (IS_ACT) {
              elt = OP(elt, {});
            }
            if constexpr (IS_DACT) {
              float act_in_elt = static_cast<float>(act_in_sh[buff_in][i][tid_X_colwise]);
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
              cached_act_sh[buff_in][i][tid_X_colwise] = static_cast<IType>(elt);
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
  
          out_colwise_data_sh[buff_out][i][tid_X_colwise] = static_cast<OType>(scaled_out);
        }
      }
  
      if constexpr (ROWWISE_SCALING) {
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
            // Load elements
            in_IType[w].load_from(&in_sh[buff_in][thread_offset_Y_rowwise][swizzled_thread_idx]);
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
  
            const bool row_out_of_bounds_rowwise = (row_base_rowwise + stage_offset_Y >= rows);
            const bool swizzled_col_out_of_bounds = (block_offset_X + swizzled_thread_idx >= cols);
            const bool out_of_bounds = (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
  
            // Load cached elements
            in_cached[w].load_from(&cached_act_sh[buff_in][thread_offset_Y_rowwise][swizzled_thread_idx]);
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
  
            Vec<IType, PACK_SIZE> in;
            Vec<IType, PACK_SIZE> act_in;
  
            in.load_from(&in_sh[buff_in][thread_offset_Y_rowwise][swizzled_thread_idx]);
            if constexpr (IS_DACT) {
              act_in.load_from(&act_in_sh[buff_in][thread_offset_Y_rowwise][swizzled_thread_idx]);
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
        const int stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
        const int stage_scales_offset_X = scales_offset_X_rowwise;
        const int scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
        if (rowwise_scale_is_within_bounds) {
          scales_rowwise[scale_idx] = biased_exponent;
        }
  
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
          out.store_to(&out_rowwise_data_sh[buff_out][thread_offset_Y_rowwise][swizzled_idx]);
        }
      }
  
      __builtin_assume(CTA_amax >= 0);
      __builtin_assume(thread_amax >= 0);
      CTA_amax = fmaxf(CTA_amax, thread_amax);
  
      // Wait for shared memory writes to be visible to TMA engine.
      ptx::fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine.
  
      // Initiate TMA transfer to copy shared memory to global memory
      if (leading_thread) {
        const int global_offset_Y = block_offset_Y + stage_offset_Y;
        const int global_offset_X = block_offset_X;
  
        if constexpr (ROWWISE_SCALING) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), global_offset_X,
              global_offset_Y, reinterpret_cast<uint64_t *>(&out_rowwise_data_sh[buff_out]));
        }
        if constexpr (COLWISE_SCALING) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), global_offset_X,
              global_offset_Y, reinterpret_cast<uint64_t *>(&out_colwise_data_sh[buff_out]));
        }
  
        // Create a "bulk async-group" out of the previous bulk copy operation.
        ptx::cp_async_bulk_commit_group();
      }
      buff_in = (buff_in + 1) % BUFFS_NUM;
      buff_out = (buff_out + 1) % BUFFS_NUM;
    }
  
    if constexpr (IS_DBIAS) {
      float thread_partial_dbias = 0.0f;
      if constexpr (COLWISE_SCALING) {
        thread_partial_dbias = partial_dbias_colwise;
      } else {
        // Reusing dshmem (in_sh_ptr) as dbias buffer [HEIGHT x WIDTH]
        // HEIGHT = THREADS_Y
        // WIDTH = THREADS_X * (SCALE_DIM_X + 1)
        // Added extra 1-element padding per thread_X to reduce bank conflicts
        float *partial_dbias_rowwise = reinterpret_cast<float *>(dshmem);
  
        constexpr int DBIAS_BUFF_WIDTH = THREADS_X * (SCALE_DIM_X + 1);
  
        const int shmem_thread_offset =
            tid_Y_rowwise * DBIAS_BUFF_WIDTH + tid_X_rowwise * (SCALE_DIM_X + 1);
        #pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const int swizzled_group_offset = shmem_thread_offset + swizzled_group_idx;
          #pragma unroll
          for (int e = 0; e < PACK_SIZE; ++e) {
            const int j = w * PACK_SIZE + e;
            const int shmem_elt_idx = swizzled_group_offset + e;
            partial_dbias_rowwise[shmem_elt_idx] = thread_dbias_rowwise[j];
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < THREADS_Y; ++i) {
          // Add extra element offset per MXFP8 scaling block [1x32]
          const int scaling_block = threadIdx.x / SCALE_DIM_X;
          thread_partial_dbias += partial_dbias_rowwise[i * DBIAS_BUFF_WIDTH + threadIdx.x + scaling_block];
        }
      }
      const int dbias_stride = cols;
      const int dbias_idx = dbias_offset_Y * dbias_stride + dbias_offset_X;
      const bool col_out_of_bounds_dbias = (dbias_offset_X >= cols);
      if (!col_out_of_bounds_dbias) {
        dbias_workspace[dbias_idx] = thread_partial_dbias;
      }
    }
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    CTA_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(CTA_amax, warp_id);
    if (leading_thread) {
      atomicMaxFloat(amax_ptr, CTA_amax);
    }
  }

  if (leading_thread) {
    #pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
    ptx::mbarrier_invalid(&workID_mbar);
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace quantize_kernel

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void quantize(const Tensor &input, const Tensor *act_input, const Tensor *noop,  // TODO (ksivamani)
              Tensor *output, Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  using namespace quantize_kernel;
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
          constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
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

          switch (scaling_type) {
            case ScalingType::ROWWISE: {
              auto kernel =
                  quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, true,
                                        false, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>;
              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr, noop_ptr,
                  workspace_ptr, amax_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise);
              NVTE_CHECK_CUDA(cudaGetLastError());
              break;
            }
            case ScalingType::COLWISE: {
              auto kernel =
                  quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, false,
                                        true, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>;
              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr, noop_ptr,
                  workspace_ptr, amax_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise);
              NVTE_CHECK_CUDA(cudaGetLastError());
              break;
            }
            case ScalingType::BIDIMENSIONAL: {
              auto kernel =
                  quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType, true,
                                        true, CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>;
              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, scales_rowwise_ptr, scales_colwise_ptr, noop_ptr,
                  workspace_ptr, amax_ptr, rows, cols, scale_stride_rowwise, scale_stride_colwise);
              NVTE_CHECK_CUDA(cudaGetLastError());
              break;
            }
          }

          if constexpr (IS_DBIAS) {
            common::reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_MXFP8_CUH_
