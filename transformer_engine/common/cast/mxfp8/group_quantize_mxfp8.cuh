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
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"
#include "swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_quantize_kernel {

using namespace dispatch::common;

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;
__device__ alignas(128) CUtensorMap g_tensor_maps_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_act_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output_rowwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output_colwise[MAX_SUPPORTED_TENSOR_DESCRIPTORS];

struct TunableConfig {
  static constexpr size_t CHUNK_DIM_Y = 128;
  static constexpr size_t CHUNK_DIM_X = 128;
  static constexpr size_t THREADS_PER_CHUNK = 128;
  static constexpr size_t PREFETCH_STAGES = 1;
  // true  -> static persistent grid-stride scheduler
  // false -> non-persistent one-job-per-CTA execution
  static constexpr bool PERSISTENT = true;
  // Launch static persistent grid as (SM_count * STATIC_PERSISTENT_BLOCKS_PER_SM, 1, 1).
  static constexpr size_t STATIC_PERSISTENT_BLOCKS_PER_SM = 24;
};

constexpr bool PERSISTENT = TunableConfig::PERSISTENT;
static_assert(!PERSISTENT || (TunableConfig::STATIC_PERSISTENT_BLOCKS_PER_SM > 0),
              "STATIC_PERSISTENT_BLOCKS_PER_SM must be greater than zero in persistent mode.");

constexpr size_t SCALE_DIM_Y = 32;
constexpr size_t SCALE_DIM_X = 32;

constexpr size_t PREFETCH_STAGES = TunableConfig::PREFETCH_STAGES;
constexpr size_t BUFFS_NUM = PREFETCH_STAGES + 1;
constexpr size_t PACK_SIZE = 4;
constexpr size_t WAVES = SCALE_DIM_X / PACK_SIZE;

constexpr size_t CHUNK_DIM_Y = TunableConfig::CHUNK_DIM_Y;
constexpr size_t CHUNK_DIM_X = TunableConfig::CHUNK_DIM_X;
constexpr size_t THREADS_PER_CHUNK = TunableConfig::THREADS_PER_CHUNK;

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

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ size_t get_current_tensor_id(
    const size_t num_tensors, const size_t current_offset, const size_t block_Y,
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
__device__ __forceinline__ size_t get_tensor_rows_num(
    const size_t tensor_id, const size_t first_logical_dim,
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
      return get_tensor_rows_num<ShapeRepresentation::SAME_BOTH_DIMS>(
          tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
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
__device__ __forceinline__ size_t get_tensor_cols_num(
    const size_t tensor_id, const size_t last_logical_dim,
    const int64_t *const __restrict__ last_dims_ptr) {
  size_t cols_num = 0;
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    cols_num = last_logical_dim;
  } else {
    cols_num = static_cast<size_t>(last_dims_ptr[tensor_id]);
    if (cols_num % 128 != 0) {
      NVTE_DEVICE_ERROR(
          "For non-single tensors, the last dimension of each tensor in a group "
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
      return get_tensor_cols_num<ShapeRepresentation::SAME_BOTH_DIMS>(
          tensor_id, last_logical_dim, last_dims_ptr);
    case ShapeRepresentation::VARYING_FIRST_DIM:
      return get_tensor_cols_num<ShapeRepresentation::VARYING_FIRST_DIM>(
          tensor_id, last_logical_dim, last_dims_ptr);
    case ShapeRepresentation::VARYING_LAST_DIM:
      return get_tensor_cols_num<ShapeRepresentation::VARYING_LAST_DIM>(
          tensor_id, last_logical_dim, last_dims_ptr);
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

  __host__ __device__ __forceinline__ constexpr JobDescriptor(
      const size_t block_id_, const size_t block_global_offset_, const size_t tensor_id_,
      const size_t rows_, const size_t cols_)
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
      const size_t tensor_base_, const size_t block_id_in_current_tensor_,
      const size_t block_id_Y_, const size_t block_id_X_, const size_t block_offset_Y_,
      const size_t block_offset_X_)
      : tensor_base(tensor_base_),
        block_id_in_current_tensor(block_id_in_current_tensor_),
        block_id_Y(block_id_Y_),
        block_id_X(block_id_X_),
        block_offset_Y(block_offset_Y_),
        block_offset_X(block_offset_X_) {}
};

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ JobDescriptor decode_job(
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const size_t work_blocks_X, const int32_t ctaid_X, const int32_t ctaid_Y,
    const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr) {
  constexpr bool is_single_tensor =
      (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
       SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM);
  const size_t block_id = ctaid_Y * work_blocks_X + ctaid_X;
  const size_t block_global_offset =
      is_single_tensor ? (ctaid_Y * CHUNK_DIM_Y * last_logical_dim + ctaid_X * CHUNK_DIM_X)
                       : (block_id * ELTS_PER_CHUNK);
  const size_t tensor_id = get_current_tensor_id<SHAPE_REP>(
      num_tensors, block_global_offset, ctaid_Y, first_logical_dim, last_logical_dim, offsets_ptr);
  const size_t rows =
      get_tensor_rows_num<SHAPE_REP>(tensor_id, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num<SHAPE_REP>(tensor_id, last_logical_dim, last_dims_ptr);
  return JobDescriptor(block_id, block_global_offset, tensor_id, rows, cols);
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ bool is_job_valid(const JobDescriptor &job, const size_t total_work_blocks,
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
  const size_t block_offset_X_in_tensor = tensor_offset_from_start % job.cols;
  if (block_offset_Y_in_tensor >= job.rows || block_offset_X_in_tensor >= job.cols) {
    return false;
  }

  return true;
}

__device__ __forceinline__ bool job_has_work(const JobDescriptor &job) {
  return job.rows != 0 && job.cols != 0;
}

__device__ __forceinline__ void advance_to_next_job(
    bool &job_finished, int32_t &ctaid_X, int32_t &ctaid_Y, size_t &static_next_block_id,
    const size_t static_block_stride, const size_t total_work_blocks, const size_t work_blocks_X) {
  if constexpr (PERSISTENT) {
    if (static_next_block_id < total_work_blocks) {
      ctaid_X = static_cast<int32_t>(static_next_block_id % work_blocks_X);
      ctaid_Y = static_cast<int32_t>(static_next_block_id / work_blocks_X);
      static_next_block_id += static_block_stride;
    } else {
      job_finished = true;
    }
  } else {
    job_finished = true;
  }
}

template <ShapeRepresentation SHAPE_REP>
__device__ __forceinline__ BlockDescriptor decode_block(
    const JobDescriptor &job, const int64_t *const __restrict__ offsets_ptr) {
  constexpr bool is_single_tensor =
      (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
       SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM);
  const size_t CHUNK_DIM_X_ = CHUNK_DIM_X;
  const size_t blocks_X_num_in_current_tensor = DIVUP(job.cols, CHUNK_DIM_X_);
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
    NVTE_DEVICE_ERROR(
        "tensormap.replace is architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

template <typename IType, typename OType>
__global__ void __launch_bounds__(1) update_tma_descriptors(
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
  const size_t tensor_id = blockIdx.x;
  const size_t rows = get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
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
__device__ __forceinline__ void store_output_stage(OType *out_rowwise_data_sh,
                                                   OType *out_colwise_data_sh,
                                                   const CUtensorMap &tensor_map_output_rowwise,
                                                   const CUtensorMap &tensor_map_output_colwise,
                                                   const int global_offset_X,
                                                   const int global_offset_Y, const int buff_offset,
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
  ptx::cp_async_bulk_commit_group();
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, bool ROWWISE_SCALING,
          bool WITH_GEMM_SWIZZLED_SCALES, bool USE_FAST_MATH>
__device__ __forceinline__ float process_colwise_stage(
    const size_t buff, const int stage, const size_t tid_X_colwise,
    const size_t scales_offset_Y_colwise, const size_t scales_offset_X_colwise,
    const size_t scale_stride_colwise, const size_t tensor_base_for_scales, const size_t rows,
    const size_t cols, IType *in_sh, IType *act_in_sh, IType *cached_act_sh,
    OType *out_colwise_data_sh, e8m0_t *scales_colwise, float &partial_dbias_colwise) {
  using IType2 = typename ptx::FPx2<IType>;
  using IType4 = typename ptx::FPx4<IType>;
  using OType4 = typename ptx::FPx4<OType>;

  constexpr uint32_t IN_SHMEM_STRIDE = static_cast<uint32_t>(BUFF_DIM_X * sizeof(IType));
  constexpr uint32_t OUT_SHMEM_STRIDE = static_cast<uint32_t>(BUFF_DIM_X * sizeof(OType));

  constexpr bool COMPUTE_ACTIVATIONS = IS_DACT || IS_ACT;
  constexpr bool NO_ACTIVATIONS = !COMPUTE_ACTIVATIONS;
  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && ROWWISE_SCALING;
  constexpr bool NON_FP32_CAST_ONLY = NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>);

  const size_t shmem_offset_base_colwise = buff * BUFF_DIM + tid_X_colwise;
  float thread_amax = 0.0f;
  float in_compute_colwise[BUFF_DIM_Y];
  IType in_colwise_IType[BUFF_DIM_Y];
  IType4 in_colwise_IType4[BUFF_DIM_Y/4];

  if constexpr (USE_FAST_MATH && NON_FP32_CAST_ONLY) {
    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
    #pragma unroll
    for (int i = 0; i < BUFF_DIM_Y; i += 4) {
      const size_t shmem_offset_colwise = shmem_offset_base_colwise + i * BUFF_DIM_X;
      const uint32_t src_smem_ptr = __cvta_generic_to_shared(&in_sh[shmem_offset_colwise]);

      // Load 4x elts S2R and find amax
      if constexpr (std::is_same_v<IType, bf16>) {
        asm volatile(
          "{\n"
          ".reg.u32 base_offset, stride; \n\t"
          "mov.u32 base_offset, %2; \n\t"
          "mov.u32 stride, %3; \n\t"
          ".reg.u32 ptr0,ptr1,ptr2,ptr3; \n\t"
          "mad.lo.u32 ptr0, 0, stride, base_offset; \n\t"
          "mad.lo.u32 ptr1, 1, stride, base_offset; \n\t"
          "mad.lo.u32 ptr2, 2, stride, base_offset; \n\t"
          "mad.lo.u32 ptr3, 3, stride, base_offset; \n\t"
          ".reg.b16 x0,x1,x2,x3; \n\t"
          "ld.shared.b16 x0, [ptr0]; \n\t"
          "ld.shared.b16 x1, [ptr1]; \n\t"
          "ld.shared.b16 x2, [ptr2]; \n\t"
          "ld.shared.b16 x3, [ptr3]; \n\t"
          "mov.b64 %0, {x0,x1,x2,x3}; \n\t"
          ".reg.b32 x01,x23; \n\t"
          "mov.b32 x01, {x0,x1}; \n\t"
          "mov.b32 x23, {x2,x3}; \n\t"
          "max.xorsign.abs.bf16x2 x01, x01, x23; \n\t"
          "max.xorsign.abs.bf16x2 %1, %1, x01; \n"
          "}\n"
          : "=l"(reinterpret_cast<uint64_t&>(in_colwise_IType4[i/4])),
            "+r"(reinterpret_cast<uint32_t&>(thread_amax_2x))
          : "r"(src_smem_ptr), "r"(IN_SHMEM_STRIDE)
        );
      } else {
        asm volatile(
          "{\n"
          ".reg.u32 base_offset, stride; \n\t"
          "mov.u32 base_offset, %2; \n\t"
          "mov.u32 stride, %3; \n\t"
          ".reg.u32 ptr0,ptr1,ptr2,ptr3; \n\t"
          "mad.lo.u32 ptr0, 0, stride, base_offset; \n\t"
          "mad.lo.u32 ptr1, 1, stride, base_offset; \n\t"
          "mad.lo.u32 ptr2, 2, stride, base_offset; \n\t"
          "mad.lo.u32 ptr3, 3, stride, base_offset; \n\t"
          ".reg.b16 x0,x1,x2,x3; \n\t"
          "ld.shared.b16 x0, [ptr0]; \n\t"
          "ld.shared.b16 x1, [ptr1]; \n\t"
          "ld.shared.b16 x2, [ptr2]; \n\t"
          "ld.shared.b16 x3, [ptr3]; \n\t"
          "mov.b64 %0, {x0,x1,x2,x3}; \n\t"
          ".reg.b32 x01,x23; \n\t"
          "mov.b32 x01, {x0,x1}; \n\t"
          "mov.b32 x23, {x2,x3}; \n\t"
          "max.xorsign.abs.f16x2 x01, x01, x23; \n\t"
          "max.xorsign.abs.f16x2 %1, %1, x01; \n"
          "}\n"
          : "=l"(reinterpret_cast<uint64_t&>(in_colwise_IType4[i/4])),
            "+r"(reinterpret_cast<uint32_t&>(thread_amax_2x))
          : "r"(src_smem_ptr), "r"(IN_SHMEM_STRIDE)
        );
      }
    }
    thread_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));

  } else if constexpr (NON_FP32_CAST_ONLY) {
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
      if constexpr (!std::is_same_v<IType, float>) {
        elt = static_cast<float>(static_cast<IType>(elt));
      }
      if constexpr (IS_CACHED_ACT_OP) {
        cached_act_sh[shmem_offset_colwise] = static_cast<IType>(elt);
      }
      thread_amax = fmaxf(thread_amax, fabsf(elt));
      in_compute_colwise[i] = elt;
    }
  }

  const e8m0_t biased_exponent =
      ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);

  const size_t global_scales_offset_Y = scales_offset_Y_colwise + stage;
  const size_t global_scales_offset_X = scales_offset_X_colwise;

  size_t scale_idx = 0;
  if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
    const size_t tensor_base_row = tensor_base_for_scales / cols;
    const size_t tensor_scales_offset_Y_base = tensor_base_row / SCALE_DIM_Y;
    const size_t tensor_scales_offset_colwise_base = tensor_base_for_scales / SCALE_DIM_Y;
    const size_t local_scales_offset_Y = global_scales_offset_Y - tensor_scales_offset_Y_base;
    scale_idx =
        tensor_scales_offset_colwise_base +
        transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx(
            global_scales_offset_X, local_scales_offset_Y, DIVUP(rows, static_cast<size_t>(128)));
  } else {
    scale_idx = global_scales_offset_Y * scale_stride_colwise + global_scales_offset_X;
  }
  scales_colwise[scale_idx] = biased_exponent;

  const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
  const IType block_scale_inverse_f16 = static_cast<IType>(block_scale_inverse);

  if constexpr (USE_FAST_MATH && NON_FP32_CAST_ONLY) {
    #pragma unroll
    for (int i = 0; i < SCALE_DIM_Y; i += 4) {
      OType4 out;
      ptx::mul_cvt_4x(out, in_colwise_IType4[i/4], block_scale_inverse_f16);

      const size_t shmem_offset_elt = shmem_offset_base_colwise + i * BUFF_DIM_X;
      const uint32_t dst_smem_ptr = __cvta_generic_to_shared(&out_colwise_data_sh[shmem_offset_elt]);

      asm volatile(
        "{\n"
        ".reg.u32 base_offset, stride; \n\t"
        "mov.u32 base_offset, %0; \n\t"
        "mov.u32 stride, %1; \n\t"
        ".reg.u32 ptr0,ptr1,ptr2,ptr3; \n\t"
        "mad.lo.u32 ptr0, 0, stride, base_offset; \n\t"
        "mad.lo.u32 ptr1, 1, stride, base_offset; \n\t"
        "mad.lo.u32 ptr2, 2, stride, base_offset; \n\t"
        "mad.lo.u32 ptr3, 3, stride, base_offset; \n\t"
        ".reg.b8 x0,x1,x2,x3; \n\t"
        "mov.b32 {x0,x1,x2,x3}, %2; \n\t"
        "st.shared.b8 [ptr0], x0; \n\t"
        "st.shared.b8 [ptr1], x1; \n\t"
        "st.shared.b8 [ptr2], x2; \n\t"
        "st.shared.b8 [ptr3], x3; \n"
        "}\n"
        :: "r"(dst_smem_ptr), "r"(OUT_SHMEM_STRIDE), "r"(reinterpret_cast<const uint32_t&>(out))
      );
    }
  } else {
    #pragma unroll
    for (int i = 0; i < SCALE_DIM_Y; ++i) {
      float in;
      if constexpr (NON_FP32_CAST_ONLY) {
        in = static_cast<float>(in_colwise_IType[i]);
      } else {
        in = in_compute_colwise[i];
      }
      const float scaled_out = in * block_scale_inverse;
  
      const size_t shmem_offset_elt = shmem_offset_base_colwise + i * BUFF_DIM_X;
      out_colwise_data_sh[shmem_offset_elt] = static_cast<OType>(scaled_out);
    }
  }
  return thread_amax;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType, bool COLWISE_SCALING,
          bool WITH_GEMM_SWIZZLED_SCALES, bool USE_FAST_MATH>
__device__ __forceinline__ float process_rowwise_stage(
    const size_t buff, const size_t stage_offset_Y, const size_t thread_offset_Y_rowwise,
    const size_t thread_offset_X_rowwise, const int bank_group,
    const size_t scales_offset_Y_rowwise, const size_t scales_offset_X_rowwise,
    const size_t scale_stride_rowwise, const bool rowwise_scale_is_within_bounds, const size_t cols,
    IType *in_sh, IType *act_in_sh, IType *cached_act_sh, OType *out_rowwise_data_sh,
    e8m0_t *scales_rowwise, float *thread_dbias_rowwise) {
  using IType2 = typename ptx::FPx2<IType>;
  using IType4 = typename ptx::FPx4<IType>;
  using OType2 = typename ptx::FPx2<OType>;
  using OType4 = typename ptx::FPx4<OType>;
  constexpr bool COMPUTE_ACTIVATIONS = IS_DACT || IS_ACT;
  constexpr bool NO_ACTIVATIONS = !COMPUTE_ACTIVATIONS;
  constexpr bool IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && COLWISE_SCALING;
  constexpr bool NON_FP32_CAST_ONLY = NO_ACTIVATIONS && (!IS_DBIAS) && (!std::is_same_v<IType, float>);

  const size_t shmem_offset_base_rowwise = buff * BUFF_DIM + thread_offset_Y_rowwise * BUFF_DIM_X;
  float thread_amax = 0.0f;
  float in_compute_rowwise[SCALE_DIM_X];
  Vec<IType, PACK_SIZE> in_cached[WAVES];
  Vec<IType2, PACK_SIZE / 2> in_IType[WAVES];
  IType4 in_IType4[WAVES];

  if constexpr (NON_FP32_CAST_ONLY) {
    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
    #pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
      const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
      const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;
      if constexpr (USE_FAST_MATH) {
        const uint32_t src_smem_ptr = __cvta_generic_to_shared(&in_sh[shmem_offset_rowwise]);
        // Load 4x elts S2R and find amax
        asm volatile(
          "{\n"
          "ld.shared.b64 %0, [%2]; \n\t"
          ".reg.b32 x01,x23; \n\t"
          "mov.b64 {x01, x23}, %0; \n\t"
          "max.xorsign.abs.bf16x2 x01, x01, x23; \n\t"
          "max.xorsign.abs.bf16x2 %1, %1, x01; \n"
          "}\n"
          : "+l"(reinterpret_cast<uint64_t&>(in_IType4[w])),
            "+r"(reinterpret_cast<uint32_t&>(thread_amax_2x))
          : "r"(src_smem_ptr)
        );
      } else {
        in_IType[w].load_from(&in_sh[shmem_offset_rowwise]);
        #pragma unroll
        for (int e = 0; e < PACK_SIZE / 2; ++e) {
          ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
        }
      }
    }
    thread_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
  } else if constexpr (IS_CACHED_ACT_OP) {
    __syncthreads();
    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
      const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
      const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_thread_idx;
      in_cached[w].load_from(&cached_act_sh[shmem_offset_rowwise]);
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
      thread_amax = static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));
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
        float elt = static_cast<float>(in.data.elt[e]);
        if constexpr (IS_ACT) {
          elt = OP(elt, {});
        }
        if constexpr (IS_DACT) {
          float act_in_elt = static_cast<float>(act_in.data.elt[e]);
          elt *= OP(act_in_elt, {});
        }

        if constexpr (IS_DBIAS && (!COLWISE_SCALING)) {
          thread_dbias_rowwise[j] += elt;
        }
        if constexpr (!std::is_same_v<IType, float>) {
          elt = static_cast<float>(static_cast<IType>(elt));
        }
        thread_amax = fmaxf(thread_amax, fabsf(elt));
        in_compute_rowwise[j] = elt;
      }
    }
  }

  const e8m0_t biased_exponent =
      ptx::float_to_e8m0(thread_amax * Quantized_Limits<OType>::max_norm_rcp);
  const int stage_scales_offset_Y = scales_offset_Y_rowwise + stage_offset_Y;
  const int stage_scales_offset_X = scales_offset_X_rowwise;

  size_t scale_idx = 0;
  if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
    scale_idx = transformer_engine::dispatch::mxfp8::swizzle::gemm_swizzled_scale_idx(
        stage_scales_offset_Y, stage_scales_offset_X, DIVUP(cols, static_cast<size_t>(128)));
  } else {
    scale_idx = stage_scales_offset_Y * scale_stride_rowwise + stage_scales_offset_X;
  }
  if (rowwise_scale_is_within_bounds) {
    scales_rowwise[scale_idx] = biased_exponent;
  }

  const float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
  const IType block_scale_inverse_f16 = static_cast<IType>(block_scale_inverse);
  const ptx::floatx2 block_scale_inverse_2x = {block_scale_inverse, block_scale_inverse};

  #pragma unroll
  for (int w = 0; w < WAVES; ++w) {
    const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM_X;
    const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
    const size_t shmem_offset_rowwise = shmem_offset_base_rowwise + swizzled_idx;

    if constexpr (USE_FAST_MATH && NON_FP32_CAST_ONLY) {
      uint32_t out_4x = 0;
      OType4& out = *reinterpret_cast<OType4*>(&out_4x);
      ptx::mul_cvt_4x(out, in_IType4[w], block_scale_inverse_f16);

      const uint32_t dst_smem_ptr = __cvta_generic_to_shared(&out_rowwise_data_sh[shmem_offset_rowwise]);
      asm volatile("st.shared.b32 [%0], %1;" : : "r"(dst_smem_ptr), "r"(out_4x));
    } else {
      Vec<OType2, PACK_SIZE / 2> out;
      #pragma unroll
      for (int e = 0; e < PACK_SIZE / 2; ++e) {
        IType2 in;
        OType2 &out_pair = reinterpret_cast<OType2 &>(out.data.elt[e]);
        if constexpr (NON_FP32_CAST_ONLY) {
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
      out.store_to(&out_rowwise_data_sh[shmem_offset_rowwise]);
    }
  }

  return thread_amax;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &), typename IType, typename OType,
          ScalingType SCALING_TYPE, bool WITH_GEMM_SWIZZLED_SCALES,
          ShapeRepresentation SHAPE_REP, bool USE_FAST_MATH>
__global__ void __launch_bounds__(THREADS_PER_CHUNK) group_quantize_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input_static,
    const __grid_constant__ CUtensorMap tensor_map_act_input_static,
    const __grid_constant__ CUtensorMap tensor_map_output_rowwise_static,
    const __grid_constant__ CUtensorMap tensor_map_output_colwise_static,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, e8m0_t *const __restrict__ scales_rowwise_ptr,
    e8m0_t *const __restrict__ scales_colwise_ptr, const float *__restrict__ noop,
    float *const __restrict__ dbias_workspace, float *const __restrict__ amax_ptr,
    const size_t work_blocks_X, const size_t work_blocks_Y) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool COMPUTE_ACTIVATIONS = IS_DACT || IS_ACT;
  constexpr bool NO_ACTIVATIONS = !COMPUTE_ACTIVATIONS;

  if constexpr (NO_ACTIVATIONS) {
    if (noop != nullptr && noop[0] == 1.0f) {
      return;
    }
  }

  constexpr bool ROWWISE_SCALING = (SCALING_TYPE == ScalingType::ROWWISE)
                                   || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  constexpr bool COLWISE_SCALING = (SCALING_TYPE == ScalingType::COLWISE)
                                   || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);

  constexpr ShapeRepresentation shape_rep = SHAPE_REP;
  constexpr bool is_single_tensor = (shape_rep == SAME_BOTH_DIMS || shape_rep == VARYING_FIRST_DIM);

  const bool leading_thread = (threadIdx.x == 0);

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X;
  const size_t tid_Y_colwise = 0;
  const size_t tid_X_colwise = threadIdx.x;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM_X;

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

  constexpr size_t shmem_buff_size = (IS_DACT ? 2 : 1) * buff_size_aligned_in / BUFFS_NUM;

  float block_amax = 0.0f;

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];

  // Initialize barriers shared by the entire CTA:
  // - IN_buff_readable_mbar tracks per-buffer TMA global->shared completion.
  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const size_t total_work_blocks = work_blocks_X * work_blocks_Y;
  const size_t launch_block_id = blockIdx.y * gridDim.x + blockIdx.x;

  int IN_buff_readable_parity[BUFFS_NUM] = {0};
  int32_t ctaid_X = static_cast<int32_t>(blockIdx.x);
  int32_t ctaid_Y = static_cast<int32_t>(blockIdx.y);
  size_t static_next_block_id = 0;
  size_t static_block_stride = 0;
  // In persistent mode, physical CTAs iterate over a virtual work grid via grid-stride.
  if constexpr (PERSISTENT) {
    if (launch_block_id >= total_work_blocks) {
      return;
    }
    ctaid_X = static_cast<int32_t>(launch_block_id % work_blocks_X);
    ctaid_Y = static_cast<int32_t>(launch_block_id / work_blocks_X);
    static_block_stride = gridDim.x * gridDim.y;
    static_next_block_id = launch_block_id + static_block_stride;
  }
  bool job_finished = false;
  size_t last_acquired_tensor_id = num_tensors;

  // Main work loop: decode current job, prime its pipeline, then process all 32-row stages.
  while (!job_finished) {
    // Decode CTA assignment into logical tensor coordinates and validate bounds.
    const JobDescriptor current_job = decode_job<SHAPE_REP>(
        num_tensors, first_logical_dim, last_logical_dim, work_blocks_X, ctaid_X, ctaid_Y,
        offsets_ptr, first_dims_ptr, last_dims_ptr);
    const bool current_job_is_valid =
        is_job_valid<SHAPE_REP>(current_job, total_work_blocks, offsets_ptr);
    if (!current_job_is_valid) {
      break;
    }
    if (!job_has_work(current_job)) {
      // Zero-sized tensors are valid grouped-tensor entries; skip them and keep scheduling work.
      advance_to_next_job(job_finished, ctaid_X, ctaid_Y, static_next_block_id,
                          static_block_stride, total_work_blocks, work_blocks_X);
      continue;
    }

    const size_t tensor_id = current_job.tensor_id;
    const size_t rows = current_job.rows;
    const size_t cols = current_job.cols;
    const BlockDescriptor current_block = decode_block<SHAPE_REP>(current_job, offsets_ptr);

    const size_t scale_stride_rowwise = DIVUP_TO_MULTIPLE(DIVUP(cols, static_cast<size_t>(32)), 4);
    const size_t scale_stride_colwise = DIVUP_TO_MULTIPLE(cols, 128);

    const size_t tensor_base = current_block.tensor_base;
    const size_t tensor_base_for_scales = (is_single_tensor && num_tensors > 1)
                                              ? static_cast<size_t>(offsets_ptr[tensor_id])
                                              : tensor_base;
    const size_t block_id_Y = current_block.block_id_Y;
    const size_t block_id_X = current_block.block_id_X;
    const size_t block_offset_Y = current_block.block_offset_Y;
    const size_t block_offset_X = current_block.block_offset_X;

    e8m0_t *const scales_rowwise =
        scales_rowwise_ptr + (is_single_tensor ? 0 : tensor_base / SCALE_DIM_X);
    e8m0_t *const scales_colwise =
        scales_colwise_ptr + (is_single_tensor ? 0 : tensor_base / SCALE_DIM_Y);

    const size_t scales_block_offset_Y_rowwise = block_id_Y * CHUNK_DIM_Y;
    const size_t scales_block_offset_X_rowwise = block_id_X * CHUNK_DIM_X / SCALE_DIM_X;
    const size_t scales_block_offset_Y_colwise = block_id_Y * CHUNK_DIM_Y / SCALE_DIM_Y;
    const size_t scales_block_offset_X_colwise = block_id_X * CHUNK_DIM_X;

    const size_t scales_offset_Y_rowwise = scales_block_offset_Y_rowwise + tid_Y_rowwise;
    const size_t scales_offset_X_rowwise = scales_block_offset_X_rowwise + tid_X_rowwise;
    const size_t scales_offset_Y_colwise = scales_block_offset_Y_colwise + tid_Y_colwise;
    const size_t scales_offset_X_colwise = scales_block_offset_X_colwise + tid_X_colwise;

    const bool rowwise_scale_is_within_bounds = scales_offset_X_rowwise < cols;

    const int dbias_offset_Y = block_id_Y;
    const int dbias_offset_X = block_id_X * CHUNK_DIM_X + threadIdx.x;

    const CUtensorMap &tensor_map_input =
        is_single_tensor ? tensor_map_input_static : g_tensor_maps_input[tensor_id];
    const CUtensorMap &tensor_map_act_input =
        is_single_tensor ? tensor_map_act_input_static : g_tensor_maps_act_input[tensor_id];
    const CUtensorMap &tensor_map_output_rowwise = is_single_tensor
                                                       ? tensor_map_output_rowwise_static
                                                       : g_tensor_maps_output_rowwise[tensor_id];
    const CUtensorMap &tensor_map_output_colwise = is_single_tensor
                                                       ? tensor_map_output_colwise_static
                                                       : g_tensor_maps_output_colwise[tensor_id];

    if (leading_thread && (!is_single_tensor) && (last_acquired_tensor_id != tensor_id)) {
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
      last_acquired_tensor_id = tensor_id;
    }

    int buff_in = 0;

    // Prime the pipeline with the first PREFETCH_STAGES slices of the current block.
#pragma unroll
    for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
      const size_t buff = stage;
      const size_t stage_offset_Y = stage * BUFF_DIM_Y;
      const size_t global_offset_Y = block_offset_Y + stage_offset_Y;
      const size_t global_offset_X = block_offset_X;
      const size_t buff_offset = buff * BUFF_DIM;
      uint64_t *barrier = &IN_buff_readable_mbar[buff];
      prefetch_input_stage<IType, IS_DACT>(in_sh, act_in_sh, tensor_map_input, tensor_map_act_input,
                                           global_offset_X, global_offset_Y, buff_offset,
                                           shmem_buff_size, barrier, leading_thread);
    }

    float partial_dbias_colwise = 0.0f;
    float thread_dbias_rowwise[SCALE_DIM_X];
    if constexpr (IS_DBIAS) {
#pragma unroll
      for (int j = 0; j < SCALE_DIM_X; ++j) {
        thread_dbias_rowwise[j] = 0.0f;
      }
    }

    // Process one [CHUNK_DIM_Y x CHUNK_DIM_X] block in STAGES slices (32 rows each).
#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      const size_t stage_offset_Y = stage * BUFF_DIM_Y;
      if (stage < STAGES - PREFETCH_STAGES) {
        const size_t next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
        const size_t next_prefetch_stage = stage + PREFETCH_STAGES;
        const size_t next_prefetch_stage_offset_Y = next_prefetch_stage * BUFF_DIM_Y;

        const size_t global_offset_Y = block_offset_Y + next_prefetch_stage_offset_Y;
        const size_t global_offset_X = block_offset_X;
        const size_t next_prefetch_buff_offset = next_prefetch_buff * BUFF_DIM;

        uint64_t *barrier = &IN_buff_readable_mbar[next_prefetch_buff];
        prefetch_input_stage<IType, IS_DACT>(in_sh, act_in_sh, tensor_map_input, tensor_map_act_input,
                                             global_offset_X, global_offset_Y,
                                             next_prefetch_buff_offset, shmem_buff_size, barrier,
                                             leading_thread);
      }

      ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                       IN_buff_readable_parity[buff_in]);
      IN_buff_readable_parity[buff_in] ^= 1;
      ptx::cp_async_bulk_wait_group_read<PREFETCH_STAGES>();

      const size_t buff = buff_in;
      float thread_amax = 0.0f;
      if constexpr (COLWISE_SCALING) {
        thread_amax = process_colwise_stage<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType,
                                            ROWWISE_SCALING, WITH_GEMM_SWIZZLED_SCALES, USE_FAST_MATH>(
            buff, stage, tid_X_colwise, scales_offset_Y_colwise, scales_offset_X_colwise,
            scale_stride_colwise, tensor_base_for_scales, rows, cols, in_sh, act_in_sh,
            cached_act_sh, out_colwise_data_sh, scales_colwise, partial_dbias_colwise);
      }

      if constexpr (ROWWISE_SCALING) {
        thread_amax = process_rowwise_stage<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType,
                                            COLWISE_SCALING, WITH_GEMM_SWIZZLED_SCALES, USE_FAST_MATH>(
            buff, stage_offset_Y, thread_offset_Y_rowwise, thread_offset_X_rowwise, bank_group,
            scales_offset_Y_rowwise, scales_offset_X_rowwise, scale_stride_rowwise,
            rowwise_scale_is_within_bounds, cols, in_sh, act_in_sh, cached_act_sh,
            out_rowwise_data_sh, scales_rowwise, thread_dbias_rowwise);
      }

      __builtin_assume(block_amax >= 0);
      __builtin_assume(thread_amax >= 0);
      block_amax = fmaxf(block_amax, thread_amax);

      ptx::fence_proxy_async_shared_cta();
      __syncthreads();

      // Publish the stage from shared memory into global outputs via TMA.
      const int global_offset_Y = block_offset_Y + stage_offset_Y;
      const int global_offset_X = block_offset_X;
      const int buff_offset = buff * BUFF_DIM;
      store_output_stage<OType, ROWWISE_SCALING, COLWISE_SCALING>(
          out_rowwise_data_sh, out_colwise_data_sh, tensor_map_output_rowwise,
          tensor_map_output_colwise, global_offset_X, global_offset_Y, buff_offset, leading_thread);

      buff_in = (buff_in + 1) % BUFFS_NUM;
    }

    if constexpr (IS_DBIAS) {
      if (is_single_tensor) {
        float thread_partial_dbias = 0.0f;
        if constexpr (COLWISE_SCALING) {
          thread_partial_dbias = partial_dbias_colwise;
        } else {
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
            const int scaling_block = threadIdx.x / SCALE_DIM_X;
            thread_partial_dbias +=
                partial_dbias_rowwise[i * DBIAS_BUFF_WIDTH + threadIdx.x + scaling_block];
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

    advance_to_next_job(job_finished, ctaid_X, ctaid_Y, static_next_block_id,
                        static_block_stride, total_work_blocks, work_blocks_X);
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    block_amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(block_amax, warp_id);
  }

  if (leading_thread && amax_ptr != nullptr) {
    atomicMaxFloat(amax_ptr, block_amax);
  }

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace group_quantize_kernel

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const GroupedTensor *activations,
                    const Tensor *noop, GroupedTensor *output, GroupedTensor *dbias,
                    Tensor *workspace, const QuantizationConfig *quant_config,
                    cudaStream_t stream) {
  using namespace group_quantize_kernel;

  const bool use_fast_math = quant_config ? quant_config->use_fast_math : false;
  if (use_fast_math) {
    NVTE_CHECK(input->dtype() == DType::kBFloat16 || input->dtype() == DType::kFloat16,
               "Fast math supports only BF16 and FP16 input types.");
    NVTE_CHECK(!IS_DBIAS && !IS_DACT && !IS_ACT,
               "Fast math does not support fused casts.");
  }

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

  size_t work_blocks_X = 0;
  size_t work_blocks_Y = 0;

  if (is_single_tensor) {
    work_blocks_Y = DIVUP(first_logical_dim, CHUNK_DIM_Y);
    work_blocks_X = DIVUP(last_logical_dim, CHUNK_DIM_X);
  } else {
    NVTE_CHECK(num_tensors <= MAX_SUPPORTED_TENSOR_DESCRIPTORS,
               "Number of tensors in a group is larger than "
               "the MAX number of supported descriptors (64).");
    work_blocks_Y = 1;
    work_blocks_X = DIVUP(elts_total, CHUNK_DIM_Y * CHUNK_DIM_X);
  }

  size_t launch_blocks_X = work_blocks_X;
  size_t launch_blocks_Y = work_blocks_Y;
  if constexpr (PERSISTENT) {
    const size_t sm_num = static_cast<size_t>(transformer_engine::cuda::sm_count());
    const size_t static_grid_size = sm_num * TunableConfig::STATIC_PERSISTENT_BLOCKS_PER_SM;
    NVTE_CHECK(static_grid_size > 0, "Static persistent grid size must be greater than zero.");
    launch_blocks_X = static_grid_size;
    launch_blocks_Y = 1;
  }
  const dim3 grid(launch_blocks_X, launch_blocks_Y);
  const size_t block_size = THREADS_PER_CHUNK;

  const bool with_gemm_swizzled_scales = output->with_gemm_swizzled_scales;

  // Logical shape of a tensor with varying all dims is [1, M*K]
  if (shape_rep != ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(first_logical_dim % 128 == 0,
               "First logical dimension of a grouped tensor must be divisible by 128.");
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

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(is_single_tensor,
               "DBias is only supported for tensors with the const last dimension.");
    NVTE_CHECK(dbias->data.dtype == input->dtype(),
               "DBias must have the same type as input_tensor.");

    std::vector<size_t> expected_shape_dbias_tensor = {num_tensors, last_logical_dim};
    NVTE_CHECK(dbias->data.shape == expected_shape_dbias_tensor, "Wrong shape of DBias.");

    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");
    const size_t dbias_workspace_rows = DIVUP(first_logical_dim, CHUNK_DIM_Y);
    const size_t dbias_workspace_cols = last_logical_dim;
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_workspace_rows, dbias_workspace_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->dtype(), IType,
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(output->dtype(), OType,
      TRANSFORMER_ENGINE_SCALING_TYPE_SWITCH(scaling_type, SCALING_TYPE,
        TRANSFORMER_ENGINE_SWITCH_CONDITION(with_gemm_swizzled_scales, WITH_GEMM_SWIZZLED_SCALES,
          TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(shape_rep, SHAPE_REP,
            TRANSFORMER_ENGINE_SWITCH_CONDITION(use_fast_math, USE_FAST_MATH,
            {
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
              update_tma_descriptors<IType, OType><<<num_tensors, 1, 0, stream>>>(
                  tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                  tensor_map_output_colwise, input_dptr, act_input_dptr, output_rowwise_dptr,
                  output_colwise_dptr, shape_rep, num_tensors, first_logical_dim,
                  last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr,
                  use_rowwise_scaling, use_colwise_scaling, IS_DACT);
            }

            auto kernel =
                group_quantize_mxfp8_kernel<IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP, IType, OType,
                                            SCALING_TYPE, WITH_GEMM_SWIZZLED_SCALES, SHAPE_REP,
                                            USE_FAST_MATH>;

            NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

            kernel<<<grid, block_size, dshmem_size, stream>>>(
                tensor_map_input, tensor_map_act_input, tensor_map_output_rowwise,
                tensor_map_output_colwise, num_tensors, first_logical_dim, last_logical_dim,
                offsets_ptr, first_dims_ptr, last_dims_ptr, scales_rowwise_ptr,
                scales_colwise_ptr, noop_ptr, workspace_ptr, amax_ptr, work_blocks_X,
                work_blocks_Y);

            if constexpr (IS_DBIAS) {
              common::grouped_reduce_dbias<IType>(
                  shape_rep, num_tensors, first_logical_dim, last_logical_dim, offsets_ptr,
                  first_dims_ptr, last_dims_ptr, dbias, workspace_ptr, CHUNK_DIM_Y, stream);
            }

            NVTE_CHECK_CUDA(cudaGetLastError());
            }
            );  // NOLINT(*)
          );  // NOLINT(*)
        );  // NOLINT(*)
      );  // NOLINT(*)
    );  // NOLINT(*)
  );  // NOLINT(*)
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_MXFP8_CUH_
