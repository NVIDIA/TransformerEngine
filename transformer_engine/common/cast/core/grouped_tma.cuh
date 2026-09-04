/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file grouped_tma.cuh
 *  \brief Architecture-specific TMA helpers for grouped quantization.
 *
 *  TMA descriptor storage/update and bulk-copy staging built on Blackwell
 *  family/arch-specific PTX (via ptx.cuh). Translation units that include this
 *  header must be compiled for the smXXXa/smXXXf targets (i.e. listed in
 *  transformer_engine_cuda_arch_specific_sources in CMakeLists.txt). The
 *  arch-neutral work-decomposition helpers live in grouped_layout.cuh.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_CORE_GROUPED_TMA_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_CORE_GROUPED_TMA_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "grouped_layout.cuh"

namespace transformer_engine {
namespace dispatch {
namespace common {

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;

struct alignas(128) TensorMapStorage {
  alignas(128) CUtensorMap input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  alignas(128) CUtensorMap act_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  alignas(128) CUtensorMap output_rowwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  alignas(128) CUtensorMap output_colwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
};

// Internal linkage avoids device-link ODR issues when this header is included by multiple .cu TUs.
static __device__ TensorMapStorage g_tensor_maps;

inline bool dimensions_supported_by_TMA(const Tensor *const t) {
  const size_t cols = t->flat_last_dim();
  constexpr size_t TMA_bytes = 16;
  const size_t alignment_requirement = (TMA_bytes * 8) / typeToNumBits(t->dtype());
  return cols % alignment_requirement == 0;
}

__device__ __forceinline__ unsigned char *align_smem_ptr_per_TMA_requirements(unsigned char *p) {
  size_t addr = reinterpret_cast<size_t>(p);
  addr = (addr + TMA_SHMEM_ALIGNMENT - 1) & ~(TMA_SHMEM_ALIGNMENT - 1);
  return reinterpret_cast<unsigned char *>(addr);
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
      NVTE_DEVICE_ERROR("Shape not supported. Data stride must be 16B aligned.");
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
    NVTE_DEVICE_ERROR("tensormap.replace is architecture-specific. ");
  }
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(1)
    update_tma_descriptors(const __grid_constant__ CUtensorMap base_tensor_map_input,
                           const __grid_constant__ CUtensorMap base_tensor_map_act_input,
                           const __grid_constant__ CUtensorMap base_tensor_map_output_rowwise,
                           const __grid_constant__ CUtensorMap base_tensor_map_output_colwise,
                           const IType *const __restrict__ input_data_ptr,
                           const IType *const __restrict__ act_input_data_ptr,
                           const OType *const __restrict__ output_rowwise_data_ptr,
                           const OType *const __restrict__ output_colwise_data_ptr,
                           const ShapeRepresentation shape_rep, const size_t num_tensors,
                           const size_t first_logical_dim, const size_t last_logical_dim,
                           const int64_t *const __restrict__ offsets_ptr,
                           const int64_t *const __restrict__ first_dims_ptr,
                           const int64_t *const __restrict__ last_dims_ptr, const bool rowwise,
                           const bool colwise, const bool compute_dactivations) {
  const size_t tensor_id = blockIdx.x;
  const size_t rows =
      get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);

  const size_t offset_elts = offsets_ptr[tensor_id];

  // Zero-sized groups: skip TMA descriptor update. The main kernel already returns
  // early for rows==0 or cols==0, but creating a TMA descriptor with a zero dimension
  // is invalid and causes CUDA_ERROR_ILLEGAL_ADDRESS.
  if (rows == 0 || cols == 0) {
    return;
  }

  if (tensor_id < num_tensors) {
    {
      CUtensorMap *modified_tensor_map_input = &g_tensor_maps.input[tensor_id];
      const uintptr_t global_data_ptr = reinterpret_cast<uintptr_t>(input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_input, modified_tensor_map_input, global_data_ptr,
                             rows, cols, sizeof(IType));
    }
    if (compute_dactivations) {
      CUtensorMap *modified_tensor_map_act_input = &g_tensor_maps.act_input[tensor_id];
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(act_input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_act_input, modified_tensor_map_act_input,
                             global_data_ptr, rows, cols, sizeof(IType));
    }
    if (rowwise) {
      CUtensorMap *modified_tensor_map_output_rowwise = &g_tensor_maps.output_rowwise[tensor_id];
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(output_rowwise_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output_rowwise, modified_tensor_map_output_rowwise,
                             global_data_ptr, rows, cols, sizeof(OType));
    }
    if (colwise) {
      CUtensorMap *modified_tensor_map_output_colwise = &g_tensor_maps.output_colwise[tensor_id];
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(output_colwise_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output_colwise, modified_tensor_map_output_colwise,
                             global_data_ptr, rows, cols, sizeof(OType));
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

// Issue TMA global->shared transfer for one stage of input (and optional activation input).
template <typename IType, bool IS_DACT>
__device__ __forceinline__ void prefetch_input_stage(
    IType *in_sh, IType *act_in_sh, const CUtensorMap &tensor_map_input,
    const CUtensorMap &tensor_map_act_input, const size_t global_offset_X,
    const size_t global_offset_Y, const size_t buff_offset, const size_t shmem_buff_size,
    uint64_t *barrier, const bool leading_thread) {
  if (leading_thread) {
    ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(&in_sh[buff_offset]),
        reinterpret_cast<const uint64_t *>(&tensor_map_input), global_offset_X, global_offset_Y,
        barrier);
    if constexpr (IS_DACT) {
      ptx::cp_async_bulk_tensor_2d_global_to_shared(
          reinterpret_cast<uint64_t *>(&act_in_sh[buff_offset]),
          reinterpret_cast<const uint64_t *>(&tensor_map_act_input), global_offset_X,
          global_offset_Y, barrier);
    }
  }
}

// Issue TMA shared->global transfer for one stage of outputs.
template <typename OType, bool ROWWISE_SCALING, bool COLWISE_SCALING>
__device__ __forceinline__ void store_output_stage(
    OType *out_rowwise_data_sh, OType *out_colwise_data_sh,
    const CUtensorMap &tensor_map_output_rowwise, const CUtensorMap &tensor_map_output_colwise,
    const size_t global_offset_X, const size_t global_offset_Y, const size_t buff_offset,
    const bool leading_thread) {
  if (!leading_thread) {
    return;
  }

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
  if constexpr (ROWWISE_SCALING || COLWISE_SCALING) {
    ptx::cp_async_bulk_commit_group();
  }
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_CORE_GROUPED_TMA_CUH_
