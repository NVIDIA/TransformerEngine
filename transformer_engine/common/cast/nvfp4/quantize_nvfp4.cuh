/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_nvfp4.cuh
 *  \brief CUDA kernels to cast to NVFP4.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace quantize_kernel {

using namespace ptx;
using namespace quantization_SF;
using namespace core;

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 16;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t BUFF_DIM_Y = 32;

constexpr size_t PACK_SIZE = 8;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

// Number of 4-bit elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM_X;  // 8 = 128 / 16

#define DIRECT_SCALING_FACTORS_STORE 1

template <bool COMPUTE_ACTIVATIONS, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType, bool COLWISE_SCALING, size_t CHUNK_DIM_Y,
          size_t CHUNK_DIM_X, size_t THREADS_PER_CHUNK>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    quantize_nvfp4_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                          const __grid_constant__ CUtensorMap tensor_map_output_rowwise,
                          const __grid_constant__ CUtensorMap tensor_map_output_colwise,
                          fp8e4m3 *const scales_rowwise_e4m3, e8m0_t *const scales_colwise_e8m0,
                          const float *noop, float *const amax_ptr,
                          const float *const nvfp4_second_stage_scale_ptr, const size_t rows,
                          const size_t cols, const size_t scale_stride_rowwise,
                          const size_t scale_stride_colwise) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool ROWWISE_SCALING = true;
  constexpr bool NO_ACTIVATIONS_NOT_FP32_INPUT =
      (!COMPUTE_ACTIVATIONS) && (!std::is_same_v<IType, float>);

  using IType2 = typename ptx::FPx2<IType>;

  if constexpr (!COMPUTE_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }
  constexpr size_t NVFP4_SCALING_FACTORS_PER_CHUNK_ROW = CHUNK_DIM_X / SCALE_DIM_X;
  constexpr size_t THREADS_X_ROWWISE = NVFP4_SCALING_FACTORS_PER_CHUNK_ROW;
  constexpr size_t THREADS_Y_ROWWISE = THREADS_PER_CHUNK / THREADS_X_ROWWISE;

  static_assert(BUFF_DIM_Y >= SCALE_DIM_Y &&
                "Number of buffer rows must be greater or equal to the size of the columwise "
                "scaling block\0");
  static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
  static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE &&
                "Number of buffer rows must be greater or equal to the number of rowwise "
                "processing threads in Y dimension\0");

  constexpr size_t BUFF_IN_DIM_X = CHUNK_DIM_X;
  constexpr size_t BUFF_OUT_DIM_X = (CHUNK_DIM_X * 4) / 8;  // Holds 2 elements of 4-bit size
  constexpr size_t BUFF_IN_DIM = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t BUFF_OUT_DIM = BUFF_DIM_Y * BUFF_OUT_DIM_X;

  constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;

  constexpr size_t ITERATIONS_ROWWISE = BUFF_DIM_Y / THREADS_Y_ROWWISE;
  // static_assert(THREADS_PER_CHUNK >= CHUNK_DIM_X);    // there should be a sufficient number of
  //                                                     // threads to process one row in a single iteration

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && ROWWISE_SCALING && COLWISE_SCALING;

  const int block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int block_offset_X = blockIdx.x * CHUNK_DIM_X;
  const int scales_block_offset_Y_rowwise = blockIdx.y * CHUNK_DIM_Y;
  const int scales_block_offset_X_rowwise = blockIdx.x * CHUNK_DIM_X / SCALE_DIM_X;
  const int scales_block_offset_Y_colwise = blockIdx.y * CHUNK_DIM_Y / SCALE_DIM_Y;
  const int scales_block_offset_X_colwise = blockIdx.x * CHUNK_DIM_X;

  const int tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const int tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const int tid_Y_colwise = 0;
  const int tid_X_colwise = threadIdx.x;

  const int thread_offset_Y_rowwise = tid_Y_rowwise;
  const int thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;
  const int thread_offset_Y_colwise = tid_Y_colwise;
  const int thread_offset_X_colwise = tid_X_colwise;  // Each thread processes two adjacent elements

  const int row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const int row_base_colwise = block_offset_Y + thread_offset_Y_colwise;
  const int col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const int scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
  const int scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
  const int scales_offset_Y_colwise = scales_block_offset_Y_colwise + tid_Y_colwise;
  const int scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

  const bool rowwise_scale_is_within_bounds = scales_offset_X_rowwise < cols;
  const bool colwise_scale_is_within_bounds = scales_offset_X_colwise < cols;

  // helps resolving bank conflicts in shmem
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out_nvfp4 =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out_mxfp8 =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  constexpr size_t buff_size_nvfp4_scales =
      CHUNK_DIM_Y * (CHUNK_DIM_X / SCALE_DIM_X) * sizeof(fp8e4m3);
  constexpr size_t buff_size_mxfp8_scales =
      (CHUNK_DIM_Y / SCALE_DIM_Y) * CHUNK_DIM_X * sizeof(fp8e8m0);

  constexpr size_t in_mem = buff_size_aligned_in;

  constexpr size_t out_mem_rowwise_data = (ROWWISE_SCALING ? buff_size_aligned_out_nvfp4 : 0);
  constexpr size_t out_mem_colwise_data = (COLWISE_SCALING ? buff_size_aligned_out_mxfp8 : 0);
  constexpr size_t out_mem_rowwise_scales = (ROWWISE_SCALING ? buff_size_nvfp4_scales : 0);
  constexpr size_t out_mem_colwise_scales = (COLWISE_SCALING ? buff_size_mxfp8_scales : 0);

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *out_rowwise_data_sh = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  OType *out_colwise_data_sh = reinterpret_cast<OType *>(dshmem + in_mem + out_mem_rowwise_data);
  fp8e4m3 *out_rowwise_scales_sh =
      reinterpret_cast<fp8e4m3 *>(dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  e8m0_t *out_colwise_scales_sh = reinterpret_cast<e8m0_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

  // Compute a global encoding/decoding scaling factor for all S_dec_b
  const float S_enc =
      (nvfp4_second_stage_scale_ptr == nullptr) ? 1.0f : 1.0f / (*nvfp4_second_stage_scale_ptr);

  float thread_amax = 0.0f;

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];

  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, is_master_thread);

  copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                    &mbar[0], is_master_thread);

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const int buff = stage % BUFFS_NUM;
    const int next_stage = stage + 1;
    const int stage_offset_Y = stage * BUFF_DIM_Y;

    const int buff_offset_in = buff * BUFF_IN_DIM;
    const int buff_offset_out = buff * BUFF_OUT_DIM;

    if (next_stage < STAGES) {
      // Wait for TMA transfer to have finished reading shared memory.
      // I.e. the buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<1>();

      const int next_buff = next_stage % BUFFS_NUM;
      const int next_stage_offset_Y = next_stage * BUFF_DIM_Y;
      const int global_offset_Y = block_offset_Y + next_stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int next_buff_offset = next_buff * BUFF_IN_DIM;

      copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                        global_offset_Y, shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;
    if constexpr (COLWISE_SCALING) {
      const int shmem_offset_base_colwise = buff_offset_in + tid_X_colwise;

      block_amax = 0.0f;
      float in_compute_colwise[SCALE_DIM_Y];
      IType in_colwise_IType[SCALE_DIM_Y];

      // 1. Read/Compute elements. Find MXFP8-block AMAX
      if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
        IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          const int shmem_offset_colwise = shmem_offset_base_colwise + i * BUFF_IN_DIM_X;
          in_colwise_IType[i] = in_sh[shmem_offset_colwise];
          block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
        }
        block_amax = static_cast<float>(block_amax_f16);
      } else {
#pragma unroll
        for (int i = 0; i < SCALE_DIM_Y; ++i) {
          const int shmem_offset_colwise = shmem_offset_base_colwise + i * BUFF_IN_DIM_X;

          float elt = static_cast<float>(in_sh[shmem_offset_colwise]);
          if constexpr (COMPUTE_ACTIVATIONS) {
            elt = OP(elt, {});
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
              block_amax = fmaxf(block_amax, fabsf(elt));
            }
          } else {
            // If no activation, elt is 0 so we can safely do this
            block_amax = fmaxf(block_amax, fabsf(elt));
          }
          in_compute_colwise[i] = elt;
        }
      }
      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(block_amax * Quantized_Limits<OType>::max_norm_rcp);

      const int global_scales_offset_Y = scales_offset_Y_colwise + stage;
      const int global_scales_offset_X = scales_offset_X_colwise;
      const int scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
      if (colwise_scale_is_within_bounds) {
        scales_colwise_e8m0[scale_idx] = biased_exponent;
      }
      const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);

// 3. Scale elements
#pragma unroll
      for (int i = 0; i < SCALE_DIM_Y; ++i) {
        float in;
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          in = static_cast<float>(in_colwise_IType[i]);
        } else {
          in = in_compute_colwise[i];
        }
        const float scaled_out = in * block_scale_inverse;

        const int shmem_offset_elt = shmem_offset_base_colwise + i * BUFF_IN_DIM_X;
        out_colwise_data_sh[shmem_offset_elt] = static_cast<OType>(scaled_out);
      }
    }

    if constexpr (ROWWISE_SCALING) {
      const int stage_rowwise_scales_offset_Y = stage * BUFF_DIM_Y;
#pragma unroll
      for (int it = 0; it < ITERATIONS_ROWWISE; ++it) {
        const int it_thread_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

        const int shmem_offset_base_rowwise_in =
            buff_offset_in + it_thread_offset_Y_rowwise * BUFF_IN_DIM_X;
        const int shmem_offset_base_rowwise_out =
            buff_offset_out + it_thread_offset_Y_rowwise * BUFF_OUT_DIM_X;

        const int it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;
        float in_compute_rowwise[SCALE_DIM_X];
        Vec<IType, PACK_SIZE> in_cached[WAVES];

        // used as an IType container for BF16/FP16 --> NVFP4 CAST ONLY
        Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];

        // 1. Read/Compute elements. Find NVFP4-block AMAX
        if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
            const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const int shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;
            // Load elements
            in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE / 2; ++e) {
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
            }
          }
          block_amax =
              static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
        } else if constexpr (IS_CACHED_ACT_OP) {
          // ensures that all writes to cache made in the section above are visible to all threads
          __syncthreads();
          IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
            const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const int shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            const bool row_out_of_bounds_rowwise = (row_base_rowwise + it_offset_Y >= rows);
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
                  block_amax = fmaxf(block_amax, fabsf(in_cached[w].data.elt[e]));
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
            block_amax =
                static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
          }
        } else {
#pragma unroll
          for (int w = 0; w < WAVES; ++w) {
            const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
            const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
            const int shmem_offset_rowwise = shmem_offset_base_rowwise_in + swizzled_thread_idx;

            Vec<IType, PACK_SIZE> in;
            Vec<IType, PACK_SIZE> act_in;

            in.load_from(&in_sh[shmem_offset_rowwise]);
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              const int j = w * PACK_SIZE + e;
              // Compute element
              float elt = static_cast<float>(in.data.elt[e]);
              if constexpr (COMPUTE_ACTIVATIONS) {
                elt = OP(elt, {});
              }
              // Numerical truncation: Downcast to IType (BF16/FP16), then upcast it back to FP32
              if constexpr (!std::is_same_v<IType, float>) {
                elt = static_cast<float>(static_cast<IType>(elt));
              }
              if constexpr (COMPUTE_ACTIVATIONS) {
                const bool row_out_of_bounds_rowwise = (row_base_rowwise + it_offset_Y >= rows);
                const bool swizzled_col_out_of_bounds =
                    (block_offset_X + swizzled_thread_idx >= cols);
                const bool out_of_bounds =
                    (row_out_of_bounds_rowwise || swizzled_col_out_of_bounds);
                if (!out_of_bounds) {
                  block_amax = fmaxf(block_amax, fabsf(elt));
                }
              } else {
                // If no activation, elt is 0 so we can safely do this
                block_amax = fmaxf(block_amax, fabsf(elt));
              }
              in_compute_rowwise[j] = elt;
            }
          }
        }

        // 2. Compute E4M3 scaling factor
        const fp8e4m3 S_dec_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc);

#if DIRECT_SCALING_FACTORS_STORE
        // Check boundaries
        if (rowwise_scale_is_within_bounds) {
          const int scales_offset_Y =
              scales_offset_Y_rowwise + stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE;
          const int scales_offset_X = scales_offset_X_rowwise;
          const int scale_idx_global = scales_offset_Y * scale_stride_rowwise + scales_offset_X;
          scales_rowwise_e4m3[scale_idx_global] = S_dec_b_fp8;
        }
#else
        const int shmem_scales_offset_Y =
            stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE + tid_Y_rowwise;
        const int shmem_scales_offset_X = tid_X_rowwise;
        const int scale_idx =
            shmem_scales_offset_Y * NVFP4_SCALING_FACTORS_PER_CHUNK_ROW + shmem_scales_offset_X;
        out_rowwise_scales_sh[scale_idx] = S_dec_b_fp8;
#endif
        // Compute "correct" per-block encoding scaling factor
        const float block_scale_inverse =
            __fdiv_rn(S_enc, static_cast<float>(S_dec_b_fp8));  // S_enc_b_fp8

// 3. Scale elements
#pragma unroll
        for (int w = 0; w < WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;  // Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < PACK_SIZE / 4; ++e) {
            IType2 in01;
            IType2 in23;
            if constexpr (NO_ACTIVATIONS_NOT_FP32_INPUT) {
              in01 = in_IType[w].data.elt[2 * e];
              in23 = in_IType[w].data.elt[2 * e + 1];
            } else if constexpr (IS_CACHED_ACT_OP) {
              in01.x = in_cached[w].data.elt[4 * e];
              in01.y = in_cached[w].data.elt[4 * e + 1];
              in23.x = in_cached[w].data.elt[4 * e + 2];
              in23.y = in_cached[w].data.elt[4 * e + 3];
            } else {
              const int j = w * PACK_SIZE + 4 * e;
              in01.x = in_compute_rowwise[j];
              in01.y = in_compute_rowwise[j + 1];
              in23.x = in_compute_rowwise[j + 2];
              in23.y = in_compute_rowwise[j + 3];
            }
            fp4e2m1x4 &out_quad = reinterpret_cast<fp4e2m1x4 &>(out.data.elt[e]);
            ptx::mul_cvt_4x(out_quad, in01, in23, block_scale_inverse);
          }
          const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
          const int swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          const int shmem_offset_rowwise = shmem_offset_base_rowwise_out + swizzled_idx / 2;
          out.store_to(&out_rowwise_data_sh[shmem_offset_rowwise]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    __builtin_assume(block_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int global_offset_Y = block_offset_Y + stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int buff_offset_nvfp4 = buff * BUFF_OUT_DIM;
      const int buff_offset_mxfp8 = buff * BUFF_IN_DIM;

      if constexpr (ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_rowwise_data_sh[buff_offset_nvfp4]));
      }
      if constexpr (COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_colwise_data_sh[buff_offset_mxfp8]));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }

#if !DIRECT_SCALING_FACTORS_STORE
  // Vectorized store of scaling factors.
  // Each thread stores multiple scaling factors in one store instruction.
  if constexpr (ROWWISE_SCALING) {
    // Number of scaling factors = CHUNK_DIM_X / SCALE_DIM_X
    const int scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + threadIdx.x;
    const int scales_offset_X_rowwise = scales_block_offset_X_rowwise;
    const int scale_idx_global =
        scales_offset_Y_rowwise * scale_stride_rowwise + scales_offset_X_rowwise;
    const int scale_idx_shmem = threadIdx.x * NVFP4_SCALING_FACTORS_PER_CHUNK_ROW;

    if ((threadIdx.x < CHUNK_DIM_Y) && (scales_offset_Y_rowwise < rows) &&
        (scales_offset_X_rowwise < (cols / SCALE_DIM_X))) {
      using ScalesVec_t = Vec<fp8e4m3, NVFP4_SCALING_FACTORS_PER_CHUNK_ROW>;
      const ScalesVec_t &scales =
          *reinterpret_cast<ScalesVec_t *>(&out_rowwise_scales_sh[scale_idx_shmem]);
      scales.store_to(&scales_rowwise_e4m3[scale_idx_global]);
    }
  }
#endif

  float chunk_amax = 0.0f;
  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    chunk_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(thread_amax, warp_id);
  }

  if (is_master_thread && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, chunk_amax);
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace quantize_kernel

// This kernel supports only two scaling cases:
// 1. r16c0  - Rowwise NVFP4
// 2. r16c32 - Rowwise NVFP4 AND Colwise MXFP8
inline void quantize(const Tensor &input, const Tensor *noop, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_kernel;
  using namespace ptx;
  checkCuDriverContext(stream);

  constexpr bool COMPUTE_ACTIVATIONS = false;
  using ParamOP = Empty;
  constexpr float (*OP)(float, const ParamOP &) = nullptr;

  NVTE_CHECK(output->has_data(), "NVFP4 Output tensor must be allocated.");
  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");

  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  bool use_colwise_scaling = output->has_columnwise_data();
  if (use_colwise_scaling) {
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise scaling tensor must be allocated");
  }
  CheckNoopTensor(*noop, "cast_noop");

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  constexpr size_t CHUNK_DIM_Y = 128;
  constexpr size_t CHUNK_DIM_X = 128;
  constexpr size_t THREADS_PER_CHUNK = 128;

  constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);
  const dim3 grid(blocks_X, blocks_Y);
  const size_t block_size = THREADS_PER_CHUNK;

  const size_t scale_stride_rowwise = output->scale_inv.shape[1];
  const size_t scale_stride_colwise =
      use_colwise_scaling ? output->columnwise_scale_inv.shape[1] : 1;

  fp8e4m3 *const scales_rowwise_e4m3_ptr = reinterpret_cast<fp8e4m3 *>(output->scale_inv.dptr);
  e8m0_t *const scales_colwise_e8m0_ptr =
      use_colwise_scaling ? reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr) : nullptr;

  const ScalingType scaling_type =
      use_colwise_scaling ? ScalingType::BIDIMENSIONAL : ScalingType::ROWWISE;

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float *const nvfp4_second_stage_scale_ptr =
      reinterpret_cast<const float *>(output->scale.dptr);

  // Output data type is only required for the column-wise MXFP8 scaling.
  // It has no effect for the row-wise NVFP4 scaling, but is set to the default E4M3 for the macros to work
  const DType output_data_type =
      use_colwise_scaling ? output->columnwise_data.dtype : DType::kFloat8E4M3;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output_data_type, OType, alignas(64) CUtensorMap tensor_map_input{};
          alignas(64) CUtensorMap tensor_map_output_rowwise{};
          alignas(64) CUtensorMap tensor_map_output_colwise{};

          create_2D_tensor_map(tensor_map_input, input.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X,
                               cols, 0, sizeof(IType) * 8);

          create_2D_tensor_map(tensor_map_output_rowwise, output->data, rows, cols, BUFF_DIM_Y,
                               BUFF_DIM_X, cols, 0, 4);

          if (use_colwise_scaling) {
            create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data, rows, cols,
                                 BUFF_DIM_Y, BUFF_DIM_X, cols, 0, sizeof(OType) * 8);
          }

          constexpr size_t buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
          constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;
          constexpr size_t buff_size_aligned_in =
              DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
          constexpr size_t buff_size_aligned_out_nvfp4 =
              DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);
          constexpr size_t buff_size_aligned_out_mxfp8 =
              DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);
          constexpr size_t buff_size_nvfp4_scales =
              (CHUNK_DIM_Y * CHUNK_DIM_X) / 16 * sizeof(fp8e4m3);
          constexpr size_t buff_size_mxfp8_scales =
              (CHUNK_DIM_Y * CHUNK_DIM_X) / 32 * sizeof(e8m0_t);

          constexpr size_t in_mem = buff_size_aligned_in;

          const size_t out_rowwise_data_mem = buff_size_aligned_out_nvfp4;
          const size_t out_colwise_data_mem = use_colwise_scaling ? buff_size_aligned_out_mxfp8 : 0;

          const size_t out_rowwise_scales_mem = buff_size_nvfp4_scales;
          const size_t out_colwise_scales_mem = use_colwise_scaling ? buff_size_mxfp8_scales : 0;

          const size_t out_mem = out_rowwise_data_mem + out_colwise_data_mem +
                                 out_rowwise_scales_mem + out_colwise_scales_mem +
                                 TMA_SHMEM_ALIGNMENT;

          const size_t dshmem_size = in_mem + out_mem;

          switch (scaling_type) {
            case ScalingType::ROWWISE: {
              auto kernel =
                  quantize_nvfp4_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType, OType, false,
                                        CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>;
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   dshmem_size);

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  tensor_map_input, tensor_map_output_rowwise, tensor_map_output_colwise,
                  scales_rowwise_e4m3_ptr, scales_colwise_e8m0_ptr, noop_ptr, amax_ptr,
                  nvfp4_second_stage_scale_ptr, rows, cols, scale_stride_rowwise,
                  scale_stride_colwise);
              break;
            }
            case ScalingType::BIDIMENSIONAL: {
              auto kernel =
                  quantize_nvfp4_kernel<COMPUTE_ACTIVATIONS, ParamOP, OP, IType, OType, true,
                                        CHUNK_DIM_Y, CHUNK_DIM_X, THREADS_PER_CHUNK>;
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   dshmem_size);

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  tensor_map_input, tensor_map_output_rowwise, tensor_map_output_colwise,
                  scales_rowwise_e4m3_ptr, scales_colwise_e8m0_ptr, noop_ptr, amax_ptr,
                  nvfp4_second_stage_scale_ptr, rows, cols, scale_stride_rowwise,
                  scale_stride_colwise);
              break;
            }
          } NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                                // NOLINT(*)
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_NVFP4_CUH_
