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

enum ShapeRepresentation {
  SAME_BOTH_DIMS = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM = 2,
  VARYING_BOTH_DIMS = 3
};

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
    group_reduce_dbias_kernel(const ShapeRepresentation shape_rep,
                              const size_t num_tensors,
                              const size_t first_logical_dim,
                              const size_t last_logical_dim,
                              const int64_t *const offsets_ptr,
                              const int64_t *const first_dims_ptr,
                              const int64_t *const last_dims_ptr,
                              OType *const dbias_output,
                              const float *dbias_partial,
                              const size_t chunk_dim_Y) {
  using ComputeVec = Vec<float, nvec>;
  using OutputVec = Vec<OType, nvec>;

  const size_t tensor_id = blockIdx.y;
  const size_t tensor_rows = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS)
                             ? (first_logical_dim / num_tensors)
                             : first_dims_ptr[tensor_id];
  
  const size_t rows = tensor_rows / chunk_dim_Y;
  const size_t cols = last_logical_dim;

  const size_t dbias_in_offset_Y = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS)
                                   ? (tensor_id * (tensor_rows / chunk_dim_Y))
                                   : (offsets_ptr[tensor_id] / cols / chunk_dim_Y);

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
void grouped_reduce_dbias(const ShapeRepresentation shape_rep,
                          const size_t num_tensors,
                          const size_t first_logical_dim,
                          const size_t last_logical_dim,
                          const int64_t *const data_tensor_offsets_ptr,
                          const int64_t *const data_tensor_first_dims_ptr,
                          const int64_t *const data_tensor_last_dims_ptr,
                          GroupedTensor *dbias,
                          const float *workspace_ptr,
                          const size_t chunk_dim_Y,
                          cudaStream_t stream) {
  using namespace kernel;
  constexpr size_t reduce_dbias_store_bytes = 8;  // stg.64
  constexpr size_t reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(IType);

  NVTE_CHECK(last_logical_dim % reduce_dbias_nvec == 0, "Unsupported shape.");

  const size_t blocks_X = DIVUP(last_logical_dim, THREADS_PER_BLOCK * reduce_dbias_nvec);
  const size_t blocks_Y = num_tensors;
  const dim3 grid(blocks_X, blocks_Y);

  group_reduce_dbias_kernel<reduce_dbias_nvec, IType>
      <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
         shape_rep, num_tensors, first_logical_dim, last_logical_dim,
         data_tensor_offsets_ptr, data_tensor_first_dims_ptr, data_tensor_last_dims_ptr,
         reinterpret_cast<IType *>(dbias->data.dptr), workspace_ptr, chunk_dim_Y);

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_CORE_COMMON_CUH_
