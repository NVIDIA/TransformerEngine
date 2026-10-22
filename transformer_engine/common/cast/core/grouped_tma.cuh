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

#include <cstdint>
#include <limits>

#include "../../common.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "grouped_layout.cuh"

namespace transformer_engine {
namespace dispatch {
namespace common {

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;

// Non-owning workspace view passed by value to the updater and consumer kernels.
// Null pointers select the original per-translation-unit device symbols.
struct GroupedTmaStorageView {
  GroupedTmaStorageView() = default;

  GroupedTmaStorageView(char *base, size_t map_bytes, size_t metadata_bytes)
      : input(reinterpret_cast<CUtensorMap *>(base)),
        act_input(reinterpret_cast<CUtensorMap *>(base + map_bytes)),
        output_rowwise(reinterpret_cast<CUtensorMap *>(base + 2 * map_bytes)),
        output_colwise(reinterpret_cast<CUtensorMap *>(base + 3 * map_bytes)),
        rows(reinterpret_cast<size_t *>(base + 4 * map_bytes)),
        cols(reinterpret_cast<size_t *>(base + 4 * map_bytes + metadata_bytes)),
        offsets(reinterpret_cast<size_t *>(base + 4 * map_bytes + 2 * metadata_bytes)) {}

  CUtensorMap *input = nullptr;
  CUtensorMap *act_input = nullptr;
  CUtensorMap *output_rowwise = nullptr;
  CUtensorMap *output_colwise = nullptr;
  size_t *rows = nullptr;
  size_t *cols = nullptr;
  size_t *offsets = nullptr;
};

inline size_t get_grouped_tma_workspace_size(size_t num_tensors) {
  // Tensor maps occupy 128 bytes, so every map region remains 128-byte aligned.
  static_assert(sizeof(CUtensorMap) == 128);
  constexpr size_t bytes_per_group = 4 * sizeof(CUtensorMap) + 3 * sizeof(size_t);
  NVTE_CHECK(num_tensors <= std::numeric_limits<size_t>::max() / bytes_per_group,
             "Grouped TMA descriptor count is too large.");
  // A nonempty allocation distinguishes execution from a workspace size query,
  // including layouts that do not use descriptors.
  return num_tensors == 0 ? 1 : num_tensors * bytes_per_group;
}

inline GroupedTmaStorageView get_grouped_tma_storage(Tensor *workspace, size_t num_tensors) {
  const size_t required_bytes = get_grouped_tma_workspace_size(num_tensors);
  NVTE_CHECK(workspace->data.dtype == DType::kByte && workspace->data.shape.size() == 1,
             "Workspace must be a one-dimensional byte tensor.");
  NVTE_CHECK(workspace->data.shape[0] >= required_bytes,
             "Workspace is too small: expected at least ", required_bytes, " bytes.");
  if (num_tensors == 0) {
    return {};
  }
  NVTE_CHECK(is_aligned_tensor_data(*workspace, 128),
             "Grouped TMA workspace must be 128-byte aligned.");
  return {static_cast<char *>(workspace->data.dptr), num_tensors * sizeof(CUtensorMap),
          num_tensors * sizeof(size_t)};
}

struct alignas(128) TensorMapStorage {
  alignas(128) CUtensorMap input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  alignas(128) CUtensorMap act_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  alignas(128) CUtensorMap output_rowwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  alignas(128) CUtensorMap output_colwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  size_t rows[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  size_t cols[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
  size_t offsets[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
};

// Internal linkage avoids device-link ODR issues when this header is included by multiple .cu TUs.
static __device__ TensorMapStorage g_tensor_maps;

inline bool dimensions_supported_by_TMA(const Tensor *const t) {
  const size_t cols = t->flat_last_dim();
  constexpr size_t TMA_bytes = 16;
  const size_t alignment_requirement = (TMA_bytes * 8) / typeToNumBits(t->dtype());
  return cols % alignment_requirement == 0;
}

// Copy the base tensor map to shared memory, modify it, and publish it to global memory. This
// function must be called convergently by every thread in a one-warp CTA.
__device__ __forceinline__ void modify_base_tensor_map(const CUtensorMap base_tensor_map,
                                                       CUtensorMap *global_tensor_map,
                                                       const uintptr_t global_data_ptr,
                                                       const size_t global_dim_Y,
                                                       const size_t global_dim_X,
                                                       const size_t data_type_size_bytes) {
  __shared__ alignas(128) CUtensorMap shared_tensor_map;
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    const uint32_t shared_tensor_map_ptr = __cvta_generic_to_shared(&shared_tensor_map);
    if (threadIdx.x == 0) {
      shared_tensor_map = base_tensor_map;

      const size_t global_stride_bytes = global_dim_X * data_type_size_bytes;
      if (global_stride_bytes % TMA_GMEM_ALIGNMENT != 0) {
        NVTE_DEVICE_ERROR("Shape not supported. Data stride must be 16B aligned.");
      }
      if (global_data_ptr % TMA_GMEM_ALIGNMENT != 0) {
        NVTE_DEVICE_ERROR("Tensor data pointer must be 16B aligned");
      }

      asm volatile(
          "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;\n\t"
          "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %2;\n\t"
          "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 0, %3;\n\t"
          "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %4;\n"
          :
          : "r"(shared_tensor_map_ptr), "l"(global_data_ptr),
            "r"(static_cast<uint32_t>(global_dim_Y)), "r"(static_cast<uint32_t>(global_dim_X)),
            "l"(static_cast<uint64_t>(global_stride_bytes))
          : "memory");
    }

    // tensormap.cp_fenceproxy is a warp-collective instruction. Besides copying the complete
    // 128-byte descriptor, its GPU-scope release makes the update visible to tensor-map proxy
    // accesses that perform a matching acquire in a consumer CTA.
    __syncwarp();
    const uintptr_t global_tensor_map_ptr = reinterpret_cast<uintptr_t>(global_tensor_map);
    asm volatile(
        "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned "
        "[%0], [%1], 128;"
        :
        : "l"(global_tensor_map_ptr), "r"(shared_tensor_map_ptr)
        : "memory");
  } else {
    NVTE_DEVICE_ERROR("tensormap.replace is architecture-specific. ");
  }
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_WARP)
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
                           const bool colwise, const bool compute_dactivations,
                           const GroupedTmaStorageView storage) {
  const size_t tensor_id = blockIdx.x;
  size_t *const rows_map = storage.rows != nullptr ? storage.rows : g_tensor_maps.rows;
  size_t *const cols_map = storage.cols != nullptr ? storage.cols : g_tensor_maps.cols;
  size_t *const offsets_map = storage.offsets != nullptr ? storage.offsets : g_tensor_maps.offsets;
  CUtensorMap *const input_maps = storage.input != nullptr ? storage.input : g_tensor_maps.input;
  CUtensorMap *const act_input_maps =
      storage.act_input != nullptr ? storage.act_input : g_tensor_maps.act_input;
  CUtensorMap *const rowwise_maps =
      storage.output_rowwise != nullptr ? storage.output_rowwise : g_tensor_maps.output_rowwise;
  CUtensorMap *const colwise_maps =
      storage.output_colwise != nullptr ? storage.output_colwise : g_tensor_maps.output_colwise;
  const size_t rows =
      get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);

  const size_t offset_elts = offsets_ptr[tensor_id];
  if (threadIdx.x == 0) {
    rows_map[tensor_id] = rows;
    cols_map[tensor_id] = cols;
    offsets_map[tensor_id] = offset_elts;
  }

  // Zero-sized groups: skip TMA descriptor update. The main kernel already returns
  // early for rows==0 or cols==0, but creating a TMA descriptor with a zero dimension
  // is invalid and causes CUDA_ERROR_ILLEGAL_ADDRESS.
  if (rows == 0 || cols == 0) {
    return;
  }

  if (tensor_id < num_tensors) {
    {
      CUtensorMap *modified_tensor_map_input = &input_maps[tensor_id];
      const uintptr_t global_data_ptr = reinterpret_cast<uintptr_t>(input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_input, modified_tensor_map_input, global_data_ptr,
                             rows, cols, sizeof(IType));
    }
    if (compute_dactivations) {
      CUtensorMap *modified_tensor_map_act_input = &act_input_maps[tensor_id];
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(act_input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_act_input, modified_tensor_map_act_input,
                             global_data_ptr, rows, cols, sizeof(IType));
    }
    if (rowwise) {
      CUtensorMap *modified_tensor_map_output_rowwise = &rowwise_maps[tensor_id];
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(output_rowwise_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output_rowwise, modified_tensor_map_output_rowwise,
                             global_data_ptr, rows, cols, sizeof(OType));
    }
    if (colwise) {
      CUtensorMap *modified_tensor_map_output_colwise = &colwise_maps[tensor_id];
      const uintptr_t global_data_ptr =
          reinterpret_cast<uintptr_t>(output_colwise_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output_colwise, modified_tensor_map_output_colwise,
                             global_data_ptr, rows, cols, sizeof(OType));
    }
  }
}

__device__ __forceinline__ void fence_acquire_tensormap(const CUtensorMap *tensor_map) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // The descriptor updater and consumer execute in different CTAs, so CTA scope is insufficient.
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(tensor_map)
               : "memory");
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
