/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file common.cuh
 *  \brief Common functions in quantize.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../utils.cuh"

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

inline bool full_tile_1D_tensor(const Tensor *const t, const size_t elems_per_block) {
  const size_t N = product(t->data.shape);
  const bool isFullTile = (N % elems_per_block == 0);
  return isFullTile;
}

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

namespace kernel {

constexpr size_t THREADS_PER_BLOCK = 256;
template <int nvec, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
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

template <int nvec, typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    group_reduce_dbias_kernel(const ShapeRepresentation shape_rep, const size_t num_tensors,
                              const size_t first_logical_dim, const size_t last_logical_dim,
                              const int64_t *const offsets_ptr, const int64_t *const first_dims_ptr,
                              const int64_t *const last_dims_ptr, OType *const dbias_output,
                              const float *dbias_partial, const size_t chunk_dim_Y) {
  using ComputeVec = Vec<float, nvec>;
  using OutputVec = Vec<OType, nvec>;

  const size_t tensor_id = blockIdx.y;
  const size_t tensor_rows = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS)
                                 ? (first_logical_dim / num_tensors)
                                 : static_cast<size_t>(first_dims_ptr[tensor_id]);

  const size_t rows = tensor_rows / chunk_dim_Y;
  const size_t cols = last_logical_dim;

  const size_t dbias_in_offset_Y =
      (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS)
          ? (tensor_id * (tensor_rows / chunk_dim_Y))
          : (static_cast<size_t>(offsets_ptr[tensor_id]) / cols / chunk_dim_Y);

  const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id * nvec >= cols) {
    return;
  }

  const float *const thread_in_base = dbias_partial + dbias_in_offset_Y * cols + thread_id * nvec;
  OType *const thread_out_base = dbias_output + tensor_id * cols + thread_id * nvec;

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
}  // namespace kernel

template <typename IType>
void reduce_dbias(const float *workspace_ptr, Tensor *dbias, const size_t rows, const size_t cols,
                  cudaStream_t stream) {
  using namespace kernel;
  constexpr size_t reduce_dbias_store_bytes = 8;  // stg.64
  constexpr size_t reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(IType);

  NVTE_CHECK(cols % reduce_dbias_nvec == 0, "Unsupported shape.");
  const size_t reduce_dbias_num_blocks = DIVUP(cols, THREADS_PER_BLOCK * reduce_dbias_nvec);

  reduce_dbias_kernel<reduce_dbias_nvec, IType>
      <<<reduce_dbias_num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
          reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, rows, cols);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename IType>
void grouped_reduce_dbias(const ShapeRepresentation shape_rep, const size_t num_tensors,
                          const size_t first_logical_dim, const size_t last_logical_dim,
                          const int64_t *const data_tensor_offsets_ptr,
                          const int64_t *const data_tensor_first_dims_ptr,
                          const int64_t *const data_tensor_last_dims_ptr, GroupedTensor *dbias,
                          const float *workspace_ptr, const size_t chunk_dim_Y,
                          cudaStream_t stream) {
  using namespace kernel;
  constexpr size_t reduce_dbias_store_bytes = 8;  // stg.64
  constexpr size_t reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(IType);

  NVTE_CHECK(last_logical_dim % reduce_dbias_nvec == 0, "Unsupported shape.");

  const size_t blocks_X = DIVUP(last_logical_dim, THREADS_PER_BLOCK * reduce_dbias_nvec);
  const size_t blocks_Y = num_tensors;
  const dim3 grid(blocks_X, blocks_Y);

  group_reduce_dbias_kernel<reduce_dbias_nvec, IType><<<grid, THREADS_PER_BLOCK, 0, stream>>>(
      shape_rep, num_tensors, first_logical_dim, last_logical_dim, data_tensor_offsets_ptr,
      data_tensor_first_dims_ptr, data_tensor_last_dims_ptr,
      reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, chunk_dim_Y);

  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <ShapeRepresentation SHAPE_REP, size_t CHUNK_DIM_Y>
__device__ __forceinline__ size_t
get_current_tensor_id(const size_t num_tensors, const size_t current_offset, const size_t block_Y,
                      const size_t first_logical_dim, const size_t last_logical_dim,
                      const int64_t *const __restrict__ offsets_ptr) {
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t current_row = block_Y * CHUNK_DIM_Y;
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

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t
get_tensor_rows_num(const size_t tensor_id, const size_t first_logical_dim,
                    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  size_t rows_num = 0;
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                SHAPE_REP == ShapeRepresentation::VARYING_LAST_DIM) {
    rows_num = first_logical_dim;
  } else {
    rows_num = static_cast<size_t>(first_dims_ptr[tensor_id]);
  }
  if (rows_num % 128 != 0) {
    NVTE_DEVICE_ERROR("First dimension of each tensor in a group must be divisible by 128.");
  }
  return rows_num;
}

__device__ __forceinline__ size_t get_tensor_rows_num(
    const size_t tensor_id, const ShapeRepresentation shape_rep, const size_t first_logical_dim,
    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      return get_tensor_rows_num<ShapeRepresentation::SAME_BOTH_DIMS>(tensor_id, first_logical_dim,
                                                                      first_dims_ptr, num_tensors);
    case ShapeRepresentation::VARYING_FIRST_DIM:
      return get_tensor_rows_num<ShapeRepresentation::VARYING_FIRST_DIM>(
          tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
    case ShapeRepresentation::VARYING_LAST_DIM:
      return get_tensor_rows_num<ShapeRepresentation::VARYING_LAST_DIM>(
          tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      return get_tensor_rows_num<ShapeRepresentation::VARYING_BOTH_DIMS>(
          tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
  }
  return 0;
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t
get_tensor_cols_num(const size_t tensor_id, const size_t last_logical_dim,
                    const int64_t *const __restrict__ last_dims_ptr) {
  size_t cols_num = 0;
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    cols_num = last_logical_dim;
  } else {
    cols_num = static_cast<size_t>(last_dims_ptr[tensor_id]);
    if (cols_num % 128 != 0) {
      NVTE_DEVICE_ERROR(
          "For varying last dimensions support, the last dimension of each tensor in a group "
          "must be divisible by 128.");
    }
  }
  return cols_num;
}

__device__ __forceinline__ size_t get_tensor_cols_num(
    const size_t tensor_id, const ShapeRepresentation shape_rep, const size_t last_logical_dim,
    const int64_t *const __restrict__ last_dims_ptr) {
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      return get_tensor_cols_num<ShapeRepresentation::SAME_BOTH_DIMS>(tensor_id, last_logical_dim,
                                                                      last_dims_ptr);
    case ShapeRepresentation::VARYING_FIRST_DIM:
      return get_tensor_cols_num<ShapeRepresentation::VARYING_FIRST_DIM>(
          tensor_id, last_logical_dim, last_dims_ptr);
    case ShapeRepresentation::VARYING_LAST_DIM:
      return get_tensor_cols_num<ShapeRepresentation::VARYING_LAST_DIM>(tensor_id, last_logical_dim,
                                                                        last_dims_ptr);
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      return get_tensor_cols_num<ShapeRepresentation::VARYING_BOTH_DIMS>(
          tensor_id, last_logical_dim, last_dims_ptr);
  }
  return 0;
}

// Logical work-item decoded from CTA coordinates.
struct JobDescriptor {
  size_t block_id = 0;
  size_t block_global_offset = 0;
  size_t tensor_id = 0;
  size_t rows = 0;
  size_t cols = 0;

  __host__ __device__ __forceinline__ constexpr JobDescriptor() = default;

  __host__ __device__ __forceinline__ constexpr JobDescriptor(const size_t block_id_,
                                                              const size_t block_global_offset_,
                                                              const size_t tensor_id_,
                                                              const size_t rows_,
                                                              const size_t cols_)
      : block_id(block_id_),
        block_global_offset(block_global_offset_),
        tensor_id(tensor_id_),
        rows(rows_),
        cols(cols_) {}
};

// Tensor-local coordinates for a work-item.
struct BlockDescriptor {
  size_t tensor_base = 0;
  size_t block_id_in_current_tensor = 0;
  size_t block_id_Y = 0;
  size_t block_id_X = 0;
  size_t block_offset_Y = 0;
  size_t block_offset_X = 0;

  __host__ __device__ __forceinline__ constexpr BlockDescriptor() = default;

  __host__ __device__ __forceinline__ constexpr BlockDescriptor(
      const size_t tensor_base_, const size_t block_id_in_current_tensor_, const size_t block_id_Y_,
      const size_t block_id_X_, const size_t block_offset_Y_, const size_t block_offset_X_)
      : tensor_base(tensor_base_),
        block_id_in_current_tensor(block_id_in_current_tensor_),
        block_id_Y(block_id_Y_),
        block_id_X(block_id_X_),
        block_offset_Y(block_offset_Y_),
        block_offset_X(block_offset_X_) {}
};

template <ShapeRepresentation SHAPE_REP, size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X>
__device__ __forceinline__ JobDescriptor decode_job(
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const size_t work_blocks_X, const int32_t ctaid_X, const int32_t ctaid_Y,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr) {
  constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;
  constexpr bool is_single_tensor = (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                                     SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM);
  const size_t block_id = ctaid_Y * work_blocks_X + ctaid_X;
  const size_t block_global_offset =
      is_single_tensor ? (ctaid_Y * CHUNK_DIM_Y * last_logical_dim + ctaid_X * CHUNK_DIM_X)
                       : (block_id * ELTS_PER_CHUNK);
  const size_t tensor_id = get_current_tensor_id<SHAPE_REP, CHUNK_DIM_Y>(
      num_tensors, block_global_offset, ctaid_Y, first_logical_dim, last_logical_dim, offsets_ptr);
  const size_t rows =
      get_tensor_rows_num<SHAPE_REP>(tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num<SHAPE_REP>(tensor_id, last_logical_dim, last_dims_ptr);
  return JobDescriptor(block_id, block_global_offset, tensor_id, rows, cols);
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ bool is_job_valid(const JobDescriptor &job,
                                             const size_t total_work_blocks,
                                             const int64_t *const __restrict__ offsets_ptr) {
  const bool is_valid = (job.block_id < total_work_blocks);
  if (!is_valid) {
    return false;
  }
  if (job.rows == 0 || job.cols == 0) {
    return true;
  }
  if constexpr (SHAPE_REP == SAME_BOTH_DIMS) {
    return true;
  }

  const size_t tensor_start_offset = static_cast<size_t>(offsets_ptr[job.tensor_id]);
  const size_t tensor_end_offset = static_cast<size_t>(offsets_ptr[job.tensor_id + 1]);
  if (job.block_global_offset >= tensor_end_offset) {
    return false;
  }

  const size_t tensor_offset_from_start = job.block_global_offset - tensor_start_offset;
  const size_t block_offset_Y_in_tensor = tensor_offset_from_start / job.cols;
  if (block_offset_Y_in_tensor >= job.rows) {
    return false;
  }

  return true;
}

__device__ __forceinline__ bool job_has_work(const JobDescriptor &job) {
  return job.rows != 0 && job.cols != 0;
}

__device__ __forceinline__ void advance_to_next_job(bool &job_finished, int32_t &ctaid_X,
                                                    int32_t &ctaid_Y, size_t &static_next_block_id,
                                                    const size_t static_block_stride,
                                                    const size_t total_work_blocks,
                                                    const size_t work_blocks_X) {
  if (static_next_block_id < total_work_blocks) {
    ctaid_X = static_cast<int32_t>(static_next_block_id % work_blocks_X);
    ctaid_Y = static_cast<int32_t>(static_next_block_id / work_blocks_X);
    static_next_block_id += static_block_stride;
  } else {
    job_finished = true;
  }
}

template <ShapeRepresentation SHAPE_REP, size_t CHUNK_DIM_Y, size_t CHUNK_DIM_X>
__device__ __forceinline__ BlockDescriptor
decode_block(const JobDescriptor &job, const int64_t *const __restrict__ offsets_ptr) {
  constexpr bool is_single_tensor = (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                                     SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM);
  constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;
  const size_t blocks_X_num_in_current_tensor = DIVUP(job.cols, CHUNK_DIM_X);
  const size_t tensor_base = is_single_tensor ? 0 : static_cast<size_t>(offsets_ptr[job.tensor_id]);
  const size_t block_id_in_current_tensor =
      is_single_tensor ? job.block_id : (job.block_id - tensor_base / ELTS_PER_CHUNK);
  const size_t block_id_Y = block_id_in_current_tensor / blocks_X_num_in_current_tensor;
  const size_t block_id_X = block_id_in_current_tensor % blocks_X_num_in_current_tensor;
  const size_t block_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const size_t block_offset_X = block_id_X * CHUNK_DIM_X;
  return BlockDescriptor(tensor_base, block_id_in_current_tensor, block_id_Y, block_id_X,
                         block_offset_Y, block_offset_X);
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

#endif  // TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_
