/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_mxfp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_MXFP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_MXFP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_quantize_kernel {

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;
__device__ alignas(128) CUtensorMap g_tensor_maps_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_act_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output_rowwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output_colwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];

enum ShapeRepresentation {
  SAME_BOTH_DIMS = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM = 2,
  VARYING_BOTH_DIMS = 3
};

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t BUFFS_NUM = 2;
constexpr size_t PACK_SIZE = 4;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;

constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;

constexpr size_t THREADS_X = CHUNK_DIM_X / SCALE_DIM_X;
constexpr size_t THREADS_Y = THREADS_PER_CHUNK / THREADS_X;

constexpr size_t BUFF_DIM_Y = THREADS_Y;
constexpr size_t BUFF_DIM_X = CHUNK_DIM_X;
constexpr size_t BUFF_DIM = BUFF_DIM_Y * BUFF_DIM_X;
static_assert(BUFF_DIM_Y == 32);

constexpr size_t STAGES = CHUNK_DIM_Y / BUFF_DIM_Y;
static_assert(STAGES >= 1);

// Number of 1-byte elements that span 32 banks (4-byte each) of shared memory
constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4) / 1;  // 128

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr size_t THREADS_PER_BANK = TOTAL_BANKS_WIDTH / SCALE_DIM_X;  // 4 = 128 / 32

__device__ __forceinline__ size_t get_current_tensor_id(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t current_offset,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t current_row = current_offset / last_logical_dim;
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    return current_row / rows_per_tensor;
  } else {
    size_t low = 1;
    size_t hi = num_tensors;  // [low, hi]

    while (low < hi) {
      const size_t mid = low + (hi - low) / 2;
      const size_t mid_offset = static_cast<size_t>(offsets_ptr[mid]);

      if (mid_offset <= current_offset) {
        low = mid + 1;
      } else {
        hi = mid;
      }
    }
    return low - 1;
  }
}

__device__ __forceinline__ size_t get_tensor_rows_num(
    const size_t tensor_id, const ShapeRepresentation shape_rep, const size_t first_logical_dim,
    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  size_t rows_num = 0;
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
    case ShapeRepresentation::VARYING_LAST_DIM:
      rows_num = first_logical_dim;
      break;
    case ShapeRepresentation::VARYING_FIRST_DIM:
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      rows_num = static_cast<size_t>(first_dims_ptr[tensor_id]);
      break;
  }
  return rows_num;
}

__device__ __forceinline__ size_t get_tensor_cols_num(
    const size_t tensor_id, const ShapeRepresentation shape_rep, const size_t last_logical_dim,
    const int64_t *const __restrict__ last_dims_ptr) {
  size_t cols_num = 0;
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
    case ShapeRepresentation::VARYING_FIRST_DIM:
      cols_num = last_logical_dim;
      break;
    case ShapeRepresentation::VARYING_LAST_DIM:
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      cols_num = static_cast<size_t>(last_dims_ptr[tensor_id]);
      break;
  }
  return cols_num;
}

// Copies the base tensor map to shmem, modifies the copy, stores the modified tensor map at index
__device__ __forceinline__ void modify_base_tensor_map(const CUtensorMap base_tensor_map,
                                                       CUtensorMap *global_tensor_map,
                                                       const uintptr_t global_data_ptr,
                                                       const size_t global_dim_Y,
                                                       const size_t global_dim_X,
                                                       const size_t data_type_size_bytes) {
  __shared__ CUtensorMap shared_tensor_map;
  shared_tensor_map = base_tensor_map;  // Copy the base tensor map into shmem
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    const size_t global_stride_bytes = global_dim_X * data_type_size_bytes;
    if (global_stride_bytes % TMA_GMEM_ALIGNMENT != 0) {
      NVTE_DEVICE_ERROR("Shape not supported, as data stride must be 16B aligned.");
    }
    if (global_data_ptr % TMA_GMEM_ALIGNMENT != 0) {
      NVTE_DEVICE_ERROR("Tensor data pointer must be 16B aligned");
    }

    asm volatile(
        "{\n\t"
        ".reg.b64 tensor_map_ptr; \n\t"
        "mov.b64 tensor_map_ptr, %0; \n\t"
        "tensormap.replace.tile.global_address.b1024.b64  [tensor_map_ptr], %1; \n\t"
        "tensormap.replace.tile.global_dim.b1024.b32  [tensor_map_ptr], 1, %2; \n\t"  // DIM Y
        "tensormap.replace.tile.global_dim.b1024.b32  [tensor_map_ptr], 0, %3; \n\t"  // DIM X
        "tensormap.replace.tile.global_stride.b1024.b64  [tensor_map_ptr], 0, %4; \n"
        "}\n" ::"l"(reinterpret_cast<uintptr_t>(&shared_tensor_map)),
        "l"(global_data_ptr), "r"(static_cast<uint32_t>(global_dim_Y)),
        "r"(static_cast<uint32_t>(global_dim_X)), "l"(static_cast<uint64_t>(global_stride_bytes))
        : "memory");
    *global_tensor_map = shared_tensor_map;
  } else {
    NVTE_DEVICE_ERROR(
        "tensormap.replace is architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

template <typename IType, typename OType>
__global__ void update_tma_descriptors(
    const __grid_constant__ CUtensorMap base_tensor_map_input,
    const __grid_constant__ CUtensorMap base_tensor_map_act_input,
    const __grid_constant__ CUtensorMap base_tensor_map_output_rowwise,
    const __grid_constant__ CUtensorMap base_tensor_map_output_colwise,
    const IType *const __restrict__ input_data_ptr,
    const IType *const __restrict__ act_input_data_ptr,
    const OType *const __restrict__ output_rowwise_data_ptr,
    const OType *const __restrict__ output_colwise_data_ptr, const ShapeRepresentation shape_rep,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, const bool rowwise, const bool colwise,
    const bool compute_dactivations) {
  const bool leading_thread = (threadIdx.x == 0);
  const size_t tensor_id = blockIdx.x;

  const size_t rows =
      get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);

  const size_t offset_elts = offsets_ptr[tensor_id];

  if (leading_thread && (tensor_id < num_tensors)) {
    {
      const uintptr_t global_data_ptr = reinterpret_cast<uintptr_t>(input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_input, &g_tensor_maps_input[tensor_id],
                             global_data_ptr, rows, cols, sizeof(IType));
    }
    if (compute_dactivations) {
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(act_input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_act_input, &g_tensor_maps_act_input[tensor_id],
                             global_data_ptr, rows, cols, sizeof(IType));
    }
    if (rowwise) {
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(output_rowwise_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output_rowwise,
                             &g_tensor_maps_output_rowwise[tensor_id], global_data_ptr, rows, cols,
                             sizeof(OType));
    }
    if (colwise) {
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(output_colwise_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output_colwise,
                             &g_tensor_maps_output_colwise[tensor_id], global_data_ptr, rows, cols,
                             sizeof(OType));
    }
  }
}

__device__ __forceinline__ void fence_acquire_tensormap(const CUtensorMap *tensor_map) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.tensormap::generic.acquire.cta [%0], 128;" ::"l"(tensor_map));
#else
  NVTE_DEVICE_ERROR("fence_acquire_tensormap is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, bool ROWWISE_SCALING,
          bool COLWISE_SCALING, bool WITH_GEMM_SWIZZLED_SCALES>
__global__ void __launch_bounds__(THREADS_PER_CHUNK) group_quantize_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input_static,
    const __grid_constant__ CUtensorMap tensor_map_act_input_static,
    const __grid_constant__ CUtensorMap tensor_map_output_rowwise_static,
    const __grid_constant__ CUtensorMap tensor_map_output_colwise_static,
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, e8m0_t *const __restrict__ scales_rowwise_ptr,
    e8m0_t *const __restrict__ scales_colwise_ptr, const float *__restrict__ noop,
    float *const __restrict__ dbias_workspace, float *const __restrict__ amax_ptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool COMPUTE_ACTIVATIONS = IS_DACT || IS_ACT;
  constexpr bool NO_ACTIVATIONS = !COMPUTE_ACTIVATIONS;

  using IType2 = typename ptx::FPx2<IType>;
  using OType2 = typename ptx::FPx2<OType>;

  using transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx;

  if constexpr (NO_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }

  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && ROWWISE_SCALING && COLWISE_SCALING;

  const size_t block_global_offset = blockIdx.x * ELTS_PER_CHUNK;

  const size_t tensor_id = get_current_tensor_id(shape_rep, num_tensors, block_global_offset,
                                                 first_logical_dim, last_logical_dim, offsets_ptr);

  const size_t rows =
      get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);

  const size_t scale_stride_rowwise = DIVUP_TO_MULTIPLE(DIVUP(cols, static_cast<size_t>(32)), 4);
  const size_t scale_stride_colwise = DIVUP_TO_MULTIPLE(cols, 128);

  const bool is_single_tensor = (shape_rep == SAME_BOTH_DIMS || shape_rep == VARYING_FIRST_DIM);

  // grouped tensor can be treated as continuous tensor for MXFP8
  const size_t tensor_base = is_single_tensor ? 0 : static_cast<size_t>(offsets_ptr[tensor_id]);

  const CUtensorMap &tensor_map_input =
      is_single_tensor ? tensor_map_input_static : g_tensor_maps_input[tensor_id];
  const CUtensorMap &tensor_map_act_input =
      is_single_tensor ? tensor_map_act_input_static : g_tensor_maps_act_input[tensor_id];
  const CUtensorMap &tensor_map_output_rowwise =
      is_single_tensor ? tensor_map_output_rowwise_static : g_tensor_maps_output_rowwise[tensor_id];
  const CUtensorMap &tensor_map_output_colwise =
      is_single_tensor ? tensor_map_output_colwise_static : g_tensor_maps_output_colwise[tensor_id];

  const bool leading_thread = (threadIdx.x == 0);

  if (leading_thread && (!is_single_tensor)) {
    fence_acquire_tensormap(&tensor_map_input);
    if constexpr (COMPUTE_ACTIVATIONS) {
      fence_acquire_tensormap(&tensor_map_act_input);
    }
    if constexpr (ROWWISE_SCALING) {
      fence_acquire_tensormap(&tensor_map_output_rowwise);
    }
    if constexpr (COLWISE_SCALING) {
      fence_acquire_tensormap(&tensor_map_output_colwise);
    }
  }

  const size_t blocks_X_num_in_current_tensor = DIVUP(cols, static_cast<size_t>(128));
  const size_t block_id_in_current_tensor =
      is_single_tensor ? blockIdx.x : (blockIdx.x - tensor_base / ELTS_PER_CHUNK);

  const size_t block_id_Y = block_id_in_current_tensor / blocks_X_num_in_current_tensor;
  const size_t block_id_X = block_id_in_current_tensor % blocks_X_num_in_current_tensor;

  const size_t block_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const size_t block_offset_X = block_id_X * CHUNK_DIM_X;

  e8m0_t *const scales_rowwise =
      scales_rowwise_ptr + (is_single_tensor ? 0 : tensor_base / SCALE_DIM_X);
  e8m0_t *const scales_colwise =
      scales_colwise_ptr + (is_single_tensor ? 0 : tensor_base / SCALE_DIM_Y);

  const size_t scales_block_offset_Y_rowwise = block_id_Y * CHUNK_DIM_Y;
  const size_t scales_block_offset_X_rowwise = block_id_X * CHUNK_DIM_X / SCALE_DIM_X;
  const size_t scales_block_offset_Y_colwise = block_id_Y * CHUNK_DIM_Y / SCALE_DIM_Y;
  const size_t scales_block_offset_X_colwise = block_id_X * CHUNK_DIM_X;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X;
  const size_t tid_Y_colwise = 0;
  const size_t tid_X_colwise = threadIdx.x;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;

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

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char *dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_sh = reinterpret_cast<IType *>(dshmem);
  IType *act_in_sh = reinterpret_cast<IType *>(dshmem + elt_input_mem);

  OType *out_rowwise_data_sh = reinterpret_cast<OType *>(dshmem + in_mem);
  OType *out_colwise_data_sh = reinterpret_cast<OType *>(dshmem + in_mem + out_mem_rowwise);
  IType *cached_act_sh = in_sh;  // in_sh is used as a cache buffer

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

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

  initialize_barriers<STAGES, THREADS_PER_CHUNK>(mbar, leading_thread);

  int parity = 0;

  if constexpr (IS_DACT) {
    copy_2d_to_sharedx2(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, &act_in_sh[0],
                        &tensor_map_act_input, block_offset_X, block_offset_Y, shmem_buff_size,
                        &mbar[0], leading_thread);
  } else {
    copy_2d_to_shared(&in_sh[0], &tensor_map_input, block_offset_X, block_offset_Y, shmem_buff_size,
                      &mbar[0], leading_thread);
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
                            leading_thread);
      } else {
        copy_2d_to_shared(&in_sh[next_buff_offset], &tensor_map_input, global_offset_X,
                          global_offset_Y, shmem_buff_size, &mbar[next_stage], leading_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[stage], parity);

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
          thread_amax = fmaxf(thread_amax, fabsf(elt));
          in_compute_colwise[i] = elt;
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);

      const size_t global_scales_offset_Y = scales_offset_Y_colwise + stage;
      const size_t global_scales_offset_X = scales_offset_X_colwise;

      size_t scale_idx = 0;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx = gemm_swizzled_scale_idx(global_scales_offset_X, global_scales_offset_Y,
                                            DIVUP(rows, static_cast<size_t>(128)));
      } else {
        scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
      }
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
        out_colwise_data_sh[shmem_offset_elt] = static_cast<OType>(scaled_out);
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

          // Load cached elements
          in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
          // Since TMA requirement for the data alignment is 16B (i.e. cols % 8 == 0, in case of BF16 elements)
          // only single check (w.r.t. column direction) is sufficient to be sure the entire wave is inside the boundaries
          if constexpr (std::is_same_v<IType, float>) {
#pragma unroll
            for (int e = 0; e < PACK_SIZE; ++e) {
              thread_amax = fmaxf(thread_amax, fabsf(in_cached[w].data.elt[e]));
            }
          } else {
#pragma unroll
            for (int e = 0; e < PACK_SIZE; e += 2) {
              const IType2 in_cached_2x = {in_cached[w].data.elt[e], in_cached[w].data.elt[e + 1]};
              ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_cached_2x);
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
            thread_amax = fmaxf(thread_amax, fabsf(elt));
            in_compute_rowwise[j] = elt;
          }
        }
      }

      // 2. Compute E8M0 scaling factor
      const e8m0_t biased_exponent =
          ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);
      const int stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
      const int stage_scales_offset_X = scales_offset_X_rowwise;

      size_t scale_idx = 0;
      if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
        scale_idx = gemm_swizzled_scale_idx(stage_scales_offset_Y, stage_scales_offset_X,
                                            DIVUP(cols, static_cast<size_t>(128)));
      } else {
        scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
      }
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
        out.store_to(&out_rowwise_data_sh[shmem_offset_rowwise]);
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
    if (leading_thread) {
      const int global_offset_Y = block_offset_Y + stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int buff_offset = buff * BUFF_DIM;

      if constexpr (ROWWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_rowwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_rowwise_data_sh[buff_offset]));
      }
      if constexpr (COLWISE_SCALING) {
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output_colwise), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&out_colwise_data_sh[buff_offset]));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();
    }
  }

  parity ^= 1;

  if constexpr (IS_DBIAS) {
    if (is_single_tensor) {
      float thread_partial_dbias = 0.0f;
      if constexpr (COLWISE_SCALING) {
        thread_partial_dbias = partial_dbias_colwise;
      } else {
        // Reusing dshmem (in_sh) as dbias buffer [HEIGHT x WIDTH]
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
          thread_partial_dbias +=
              partial_dbias_rowwise[i * DBIAS_BUFF_WIDTH + threadIdx.x + scaling_block];
        }
      }
      const int dbias_stride = cols;
      const int dbias_offset_Y = block_id_Y;
      const int dbias_offset_X = block_id_X * CHUNK_DIM_X + threadIdx.x;
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
    block_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(block_amax, warp_id);
  }

  if (leading_thread && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, block_amax);
  }

  destroy_barriers<STAGES>(mbar, leading_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace group_quantize_kernel

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const GroupedTensor *activations,
                    const Tensor *noop, GroupedTensor *output, Tensor *dbias, Tensor *workspace,
                    cudaStream_t stream) {
  using namespace group_quantize_kernel;

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  const bool use_rowwise_scaling = output->has_data();
  const bool use_colwise_scaling = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise_scaling || use_colwise_scaling,
             "Either rowwise or columnwise output data need to be allocated.");

  ScalingType scaling_type = ScalingType::BIDIMENSIONAL;
  if (!use_colwise_scaling) {
    scaling_type = ScalingType::ROWWISE;
  } else if (!use_rowwise_scaling) {
    scaling_type = ScalingType::COLWISE;
  }

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (output->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output->all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else if (output->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (output->varying_both_dims()) {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }

  // Treat a grouped tensor with const last dims as a single tensor
  const bool is_single_tensor = (shape_rep == SAME_BOTH_DIMS || shape_rep == VARYING_FIRST_DIM);

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");

  if (IS_DACT) {
    NVTE_CHECK(activations->has_data(), "Activations tensor must have data.");
    NVTE_CHECK(input->num_tensors == activations->num_tensors,
               "Number of grad and activations tensors must be same.");
    NVTE_CHECK(input->dtype() == activations->dtype(),
               "Grad and activations tensors must have the same type.");
  }

  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t elts_total = first_logical_dim * last_logical_dim;

  const size_t num_tensors = input->num_tensors;

  size_t blocks = 0;

  if (is_single_tensor) {
    const size_t blocks_Y = DIVUP(first_logical_dim, CHUNK_DIM_Y);
    const size_t blocks_X = DIVUP(last_logical_dim, CHUNK_DIM_X);
    blocks = blocks_Y * blocks_X;
  } else {
    NVTE_CHECK(num_tensors < MAX_SUPPORTED_TENSOR_DESCRIPTORS,
               "Number of tensors in a group is larger than "
               "the MAX number of supported descriptors (64).");
    // Only full tiles supported
    NVTE_CHECK(last_logical_dim % CHUNK_DIM_X == 0,
               "Last dimension of a grouped tensor should be divisible by 128.");
    blocks = DIVUP(elts_total, CHUNK_DIM_Y * CHUNK_DIM_X);
  }
  const dim3 grid(blocks);
  const size_t block_size = THREADS_PER_CHUNK;

  const bool with_gemm_swizzled_scales = output->with_gemm_swizzled_scales;

  // Logical shape of a tensor with varying all dims is [1, M*K]
  if (shape_rep != ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(first_logical_dim % 128 == 0,
               "First dimension of a grouped tensor should be divisible by 128.");
  }

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);

  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  e8m0_t *const scales_rowwise_ptr = reinterpret_cast<e8m0_t *>(output->scale_inv.dptr);
  e8m0_t *const scales_colwise_ptr = reinterpret_cast<e8m0_t *>(output->columnwise_scale_inv.dptr);

  if (use_rowwise_scaling) {
    NVTE_CHECK(scales_rowwise_ptr != nullptr, "Scaling tensor must be allocated");
  }
  if (use_colwise_scaling) {
    NVTE_CHECK(scales_colwise_ptr != nullptr, "Columnwise scaling tensor must be allocated");
  }

  const size_t dbias_rows = DIVUP(first_logical_dim, CHUNK_DIM_Y);
  const size_t dbias_cols = last_logical_dim;
  if constexpr (IS_DBIAS) {
    NVTE_CHECK(is_single_tensor,
               "DBias is only supported for tensors with the const last dimension.");
    NVTE_CHECK(dbias->data.dtype == input->dtype(),
               "DBias must have the same type as input_tensor.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{last_logical_dim}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,

              alignas(64) CUtensorMap tensor_map_input{};
              alignas(64) CUtensorMap tensor_map_act_input{};
              alignas(64) CUtensorMap tensor_map_output_rowwise{};
              alignas(64) CUtensorMap tensor_map_output_colwise{};

              constexpr size_t input_type_bit_size = TypeInfo<IType>::size;
              constexpr size_t output_type_bit_size = TypeInfo<OType>::size;

              create_2D_tensor_map(tensor_map_input, input->data, first_logical_dim,
                                   last_logical_dim, BUFF_DIM_Y, BUFF_DIM_X, last_logical_dim, 0,
                                   input_type_bit_size);

              if constexpr (IS_DACT) {
                create_2D_tensor_map(tensor_map_act_input, activations->data, first_logical_dim,
                                     last_logical_dim, BUFF_DIM_Y, BUFF_DIM_X, last_logical_dim, 0,
                                     input_type_bit_size);
              }

              if (use_rowwise_scaling) {
                create_2D_tensor_map(tensor_map_output_rowwise, output->data, first_logical_dim,
                                     last_logical_dim, BUFF_DIM_Y, BUFF_DIM_X, last_logical_dim, 0,
                                     output_type_bit_size);
              }

              if (use_colwise_scaling) {
                create_2D_tensor_map(tensor_map_output_colwise, output->columnwise_data,
                                     first_logical_dim, last_logical_dim, BUFF_DIM_Y, BUFF_DIM_X,
                                     last_logical_dim, 0, output_type_bit_size);
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

              auto kernel =
                  group_quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType,
                                              true, true, WITH_GEMM_SWIZZLED_SCALES>;
              switch (scaling_type) {
                case ScalingType::ROWWISE: {
                  kernel =
                      group_quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType,
                                                  OType, true, false, WITH_GEMM_SWIZZLED_SCALES>;
                  break;
                }
                case ScalingType::COLWISE: {
                  kernel =
                      group_quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType,
                                                  OType, false, true, WITH_GEMM_SWIZZLED_SCALES>;
                  break;
                }
                case ScalingType::BIDIMENSIONAL: {
                  kernel =
                      group_quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType,
                                                  OType, true, true, WITH_GEMM_SWIZZLED_SCALES>;
                  break;
                }
              }

              // Update tensor descriptors before launching the kernel
              if (!is_single_tensor) {
                const IType *const input_dptr = reinterpret_cast<const IType *>(input->data.dptr);

                const IType *const act_input_dptr =
                    IS_DACT ? reinterpret_cast<const IType *>(activations->data.dptr) : nullptr;

                OType *const output_rowwise_dptr =
                    use_rowwise_scaling ? reinterpret_cast<OType *>(output->data.dptr) : nullptr;

                OType *const output_colwise_dptr =
                    use_colwise_scaling ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                        : nullptr;
                update_tma_descriptors<IType, OType><<<num_tensors, 32, 0, stream>>>(
                    tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                    tensor_map_output_colwise, input_dptr, act_input_dptr, output_rowwise_dptr,
                    output_colwise_dptr, shape_rep, num_tensors, first_logical_dim,
                    last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr,
                    use_rowwise_scaling, use_colwise_scaling, IS_DACT);
              }

              NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                  kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

              kernel<<<grid, block_size, dshmem_size, stream>>>(
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, shape_rep, num_tensors, first_logical_dim,
                  last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr, scales_rowwise_ptr,
                  scales_colwise_ptr, noop_ptr, workspace_ptr, amax_ptr);

              if constexpr (IS_DBIAS) {
                common::reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
              }

              NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
      );                                              // NOLINT(*)
  );                                                  // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_MXFP8_CUH_
