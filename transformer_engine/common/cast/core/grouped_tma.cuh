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
#include "../../util/cuda_runtime.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "grouped_layout.cuh"

namespace transformer_engine {
namespace dispatch {
namespace common {

inline bool use_tma_descriptor_override() {
#if CUDA_VERSION >= 13040
  return transformer_engine::cuda::sm_arch() == 107;
#else
  return false;
#endif
}

inline void create_grouped_tma_map(CUtensorMap &map, const SimpleTensor &tensor,
                                   const bool use_override, const uint64_t rows,
                                   const uint64_t cols, const uint32_t tile_rows,
                                   const uint32_t tile_cols, const size_t type_bits,
                                   const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE) {
  const uint64_t map_rows = use_override ? 1 : rows;
  const uint64_t map_cols = use_override ? tile_cols : cols;
  create_2D_tensor_map(map, tensor, map_rows, map_cols, tile_rows, tile_cols, map_cols, 0,
                       type_bits, swizzle);
}

struct TmaDescriptorOverride2D {
  uint64_t address;
  uint16_t size_x;
  uint16_t size_y;
  uint32_t lower_stride;
  uint16_t upper_stride;
};

template <size_t TILE_DIM_Y, size_t TILE_DIM_X, typename T>
__device__ __forceinline__ TmaDescriptorOverride2D make_tma_descriptor_override_2d(
    const T *data, const size_t rows, const size_t cols, const size_t x, const size_t y) {
  static_assert((TILE_DIM_X > 0) && (TILE_DIM_X < 256) && (TILE_DIM_Y > 0) && (TILE_DIM_Y < 256));
  const uint64_t stride_bytes = static_cast<uint64_t>(cols) * sizeof(T);
  const uint64_t address = reinterpret_cast<uint64_t>(data + y * cols + x);
  if (stride_bytes % TMA_GMEM_ALIGNMENT != 0 || address % TMA_GMEM_ALIGNMENT != 0) {
    NVTE_DEVICE_ERROR("TMA override address and stride must be 16B aligned.");
  }
  const uint64_t stride = stride_bytes >> 4;
  if ((stride >> 36) != 0) {
    NVTE_DEVICE_ERROR("TMA override stride exceeds 36 bits.");
  }
  return {address, static_cast<uint16_t>(min(TILE_DIM_X, cols - x)),
          static_cast<uint16_t>(min(TILE_DIM_Y, rows - y)), static_cast<uint32_t>(stride),
          static_cast<uint16_t>(stride >> 32)};
}

template <size_t TILE_DIM_Y, size_t TILE_DIM_X, typename T>
__device__ __forceinline__ void load_grouped_tma_2d(T *dst, const CUtensorMap &map, const size_t x,
                                                    const size_t y, uint64_t *barrier,
                                                    const T *data, const size_t rows,
                                                    const size_t cols) {
  constexpr bool has_override = ARCH_HAS_TMA_DESCRIPTOR_OVERRIDE;
  if constexpr (has_override) {
    if (data != nullptr) {
      const auto info =
          make_tma_descriptor_override_2d<TILE_DIM_Y, TILE_DIM_X>(data, rows, cols, x, y);
      ptx::cp_async_bulk_tensor_2d_global_to_shared_rubin(
          reinterpret_cast<uint64_t *>(dst), reinterpret_cast<const uint64_t *>(&map), info.address,
          info.size_x, info.size_y, info.lower_stride, info.upper_stride, barrier);
      return;
    }
  }
  ptx::cp_async_bulk_tensor_2d_global_to_shared(
      reinterpret_cast<uint64_t *>(dst), reinterpret_cast<const uint64_t *>(&map), x, y, barrier);
}

template <size_t TILE_DIM_Y, size_t TILE_DIM_X, typename T>
__device__ __forceinline__ void store_grouped_tma_2d(T *src, const CUtensorMap &map, const size_t x,
                                                     const size_t y, const T *data,
                                                     const size_t rows, const size_t cols) {
  constexpr bool has_override = ARCH_HAS_TMA_DESCRIPTOR_OVERRIDE;
  if constexpr (has_override) {
    if (data != nullptr) {
      if (x >= cols || y >= rows) {
        return;
      }

      const auto info =
          make_tma_descriptor_override_2d<TILE_DIM_Y, TILE_DIM_X>(data, rows, cols, x, y);
      ptx::cp_async_bulk_tensor_2d_shared_to_global_rubin(
          reinterpret_cast<const uint64_t *>(&map), info.address, info.size_x, info.size_y,
          info.lower_stride, info.upper_stride, reinterpret_cast<uint64_t *>(src));
      return;
    }
  }
  ptx::cp_async_bulk_tensor_2d_shared_to_global(reinterpret_cast<const uint64_t *>(&map), x, y,
                                                reinterpret_cast<uint64_t *>(src));
}

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;

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

// Must have internal linkage like g_tensor_maps. Otherwise, before CUDA 13.0 (nvcc
// -static-global-template-stub=false), the host stubs from every TU are merged, so the launch
// writes another TU's g_tensor_maps and the consumer kernel reads an empty table.
template <typename IType, typename OType>
static __global__ void __launch_bounds__(THREADS_PER_WARP)
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
  if (threadIdx.x == 0) {
    g_tensor_maps.rows[tensor_id] = rows;
    g_tensor_maps.cols[tensor_id] = cols;
    g_tensor_maps.offsets[tensor_id] = offset_elts;
  }

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
  // The descriptor updater and consumer execute in different CTAs, so CTA scope is insufficient.
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(tensor_map)
               : "memory");
#else
  NVTE_DEVICE_ERROR("fence_acquire_tensormap is only supported on SM 9.0+.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// Issue TMA global->shared transfer for one stage of input (and optional activation input).
template <typename IType, bool IS_DACT, size_t TILE_DIM_Y, size_t TILE_DIM_X>
__device__ __forceinline__ void prefetch_input_stage(
    IType *in_sh, IType *act_in_sh, const CUtensorMap &tensor_map_input,
    const CUtensorMap &tensor_map_act_input, const size_t global_offset_X,
    const size_t global_offset_Y, const size_t buff_offset, const size_t shmem_buff_size,
    uint64_t *barrier, const bool leading_thread, const IType *input_data = nullptr,
    const IType *act_data = nullptr, const size_t rows = 0, const size_t cols = 0) {
  constexpr bool has_override = ARCH_HAS_TMA_DESCRIPTOR_OVERRIDE;
  if constexpr (has_override) {
    if ((input_data != nullptr) && (global_offset_X >= cols || global_offset_Y >= rows)) {
      for (size_t i = threadIdx.x; i < TILE_DIM_Y * TILE_DIM_X; i += blockDim.x) {
        in_sh[buff_offset + i] = static_cast<IType>(0.0f);
        if constexpr (IS_DACT) {
          act_in_sh[buff_offset + i] = static_cast<IType>(0.0f);
        }
      }
      __syncthreads();
      if (leading_thread) {
        ptx::mbarrier_arrive(barrier);
      }

      return;
    }
  }
  if (leading_thread) {
    ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);
    load_grouped_tma_2d<TILE_DIM_Y, TILE_DIM_X>(&in_sh[buff_offset], tensor_map_input,
                                                global_offset_X, global_offset_Y, barrier,
                                                input_data, rows, cols);
    if constexpr (IS_DACT) {
      load_grouped_tma_2d<TILE_DIM_Y, TILE_DIM_X>(&act_in_sh[buff_offset], tensor_map_act_input,
                                                  global_offset_X, global_offset_Y, barrier,
                                                  act_data, rows, cols);
    }
  }
}

// Issue TMA shared->global transfer for one stage of outputs.
template <typename OType, bool ROWWISE_SCALING, bool COLWISE_SCALING, size_t TILE_DIM_Y = 32,
          size_t TILE_DIM_X = 128>
__device__ __forceinline__ void store_output_stage(
    OType *out_rowwise_data_sh, OType *out_colwise_data_sh,
    const CUtensorMap &tensor_map_output_rowwise, const CUtensorMap &tensor_map_output_colwise,
    const size_t global_offset_X, const size_t global_offset_Y, const size_t buff_offset,
    const bool leading_thread, const OType *rowwise_data = nullptr,
    const OType *colwise_data = nullptr, const size_t rows = 0, const size_t cols = 0) {
  if (!leading_thread) {
    return;
  }

  if constexpr (ROWWISE_SCALING) {
    store_grouped_tma_2d<TILE_DIM_Y, TILE_DIM_X>(&out_rowwise_data_sh[buff_offset],
                                                 tensor_map_output_rowwise, global_offset_X,
                                                 global_offset_Y, rowwise_data, rows, cols);
  }
  if constexpr (COLWISE_SCALING) {
    store_grouped_tma_2d<TILE_DIM_Y, TILE_DIM_X>(&out_colwise_data_sh[buff_offset],
                                                 tensor_map_output_colwise, global_offset_X,
                                                 global_offset_Y, colwise_data, rows, cols);
  }
  if constexpr (ROWWISE_SCALING || COLWISE_SCALING) {
    ptx::cp_async_bulk_commit_group();
  }
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_CORE_GROUPED_TMA_CUH_
