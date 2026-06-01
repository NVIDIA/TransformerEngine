/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_dequantize_mxfp8.cuh
 *  \brief CUDA kernels to dequantize grouped tensors from MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_MXFP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_MXFP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "group_quantize_mxfp8.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace group_dequantize_kernel {

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;
__device__ alignas(128) CUtensorMap g_tensor_maps_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output[MAX_SUPPORTED_TENSOR_DESCRIPTORS];

// Reuse helper types and functions from common namespace
using common::fence_acquire_tensormap;
using common::get_tensor_cols_num;
using common::get_tensor_rows_num;
using common::modify_base_tensor_map;

// Runtime dispatch wrapper for get_current_tensor_id (common only has template version)
template <size_t CHUNK_DIM_Y_>
__device__ __forceinline__ size_t get_current_tensor_id(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t current_offset,
    const size_t block_Y, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr) {
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      return common::get_current_tensor_id<ShapeRepresentation::SAME_BOTH_DIMS, CHUNK_DIM_Y_>(
          num_tensors, current_offset, block_Y, first_logical_dim, last_logical_dim, offsets_ptr);
    case ShapeRepresentation::VARYING_FIRST_DIM:
      return common::get_current_tensor_id<ShapeRepresentation::VARYING_FIRST_DIM, CHUNK_DIM_Y_>(
          num_tensors, current_offset, block_Y, first_logical_dim, last_logical_dim, offsets_ptr);
    case ShapeRepresentation::VARYING_LAST_DIM:
      return common::get_current_tensor_id<ShapeRepresentation::VARYING_LAST_DIM, CHUNK_DIM_Y_>(
          num_tensors, current_offset, block_Y, first_logical_dim, last_logical_dim, offsets_ptr);
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      return common::get_current_tensor_id<ShapeRepresentation::VARYING_BOTH_DIMS, CHUNK_DIM_Y_>(
          num_tensors, current_offset, block_Y, first_logical_dim, last_logical_dim, offsets_ptr);
  }
  return 0;
}

// Shared constexpr parameters used by both the kernel and the launch function.
// Defined in a struct so they are visible in both host and device code.
struct DequantizeConfig {
  static constexpr size_t CHUNK_DIM_Y = 128;
  static constexpr size_t CHUNK_DIM_X = 128;
  static constexpr size_t THREADS_PER_CHUNK = 128;
  static constexpr size_t BUFFERS_NUM = 2;
  static constexpr size_t ELEMS_PER_THREAD = 16;
  static constexpr size_t BUFFER_DIM_Y = 16;
  static constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;
  static constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;
  static constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;
  static constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;
  static constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;
  static constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;
  static constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;
};

template <typename IType, typename OType>
__global__ void update_tma_descriptors(const __grid_constant__ CUtensorMap base_tensor_map_input,
                                       const __grid_constant__ CUtensorMap base_tensor_map_output,
                                       const IType *const __restrict__ input_data_ptr,
                                       const OType *const __restrict__ output_data_ptr,
                                       const ShapeRepresentation shape_rep,
                                       const size_t num_tensors, const size_t first_logical_dim,
                                       const size_t last_logical_dim,
                                       const int64_t *const __restrict__ offsets_ptr,
                                       const int64_t *const __restrict__ first_dims_ptr,
                                       const int64_t *const __restrict__ last_dims_ptr) {
  const bool leading_thread = (threadIdx.x == 0);
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

  if (leading_thread && (tensor_id < num_tensors)) {
    {
      const uintptr_t global_data_ptr = reinterpret_cast<uintptr_t>(input_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_input, &g_tensor_maps_input[tensor_id],
                             global_data_ptr, rows, cols, sizeof(IType));
    }
    {
      const uintptr_t global_data_ptr = reinterpret_cast<uintptr_t>(output_data_ptr + offset_elts);
      modify_base_tensor_map(base_tensor_map_output, &g_tensor_maps_output[tensor_id],
                             global_data_ptr, rows, cols, sizeof(OType));
    }
  }
}

template <typename IType, typename OType, bool ROWWISE>
__global__ void __launch_bounds__(128)
    group_dequantize_mxfp8_kernel(const __grid_constant__ CUtensorMap tensor_map_input_static,
                                  const __grid_constant__ CUtensorMap tensor_map_output_static,
                                  const ShapeRepresentation shape_rep, const size_t num_tensors,
                                  const size_t first_logical_dim, const size_t last_logical_dim,
                                  const int64_t *const __restrict__ offsets_ptr,
                                  const int64_t *const __restrict__ first_dims_ptr,
                                  const int64_t *const __restrict__ last_dims_ptr,
                                  const e8m0_t *const __restrict__ scales_ptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr size_t CHUNK_DIM_Y = DequantizeConfig::CHUNK_DIM_Y;
  constexpr size_t CHUNK_DIM_X = DequantizeConfig::CHUNK_DIM_X;
  constexpr size_t THREADS_PER_CHUNK = DequantizeConfig::THREADS_PER_CHUNK;
  constexpr size_t BUFFERS_NUM = DequantizeConfig::BUFFERS_NUM;
  constexpr size_t ELEMS_PER_THREAD = DequantizeConfig::ELEMS_PER_THREAD;
  constexpr size_t BUFFER_DIM_Y = DequantizeConfig::BUFFER_DIM_Y;
  constexpr size_t SHMEM_DIM_Y = DequantizeConfig::SHMEM_DIM_Y;
  constexpr size_t SHMEM_DIM_X = DequantizeConfig::SHMEM_DIM_X;
  constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = DequantizeConfig::THREADS_PER_CHUNK_X_ROWWISE;
  constexpr size_t THREADS_PER_CHUNK_X_COLWISE = DequantizeConfig::THREADS_PER_CHUNK_X_COLWISE;
  constexpr size_t ITERATIONS = DequantizeConfig::ITERATIONS;
  constexpr size_t ELTS_PER_CHUNK = DequantizeConfig::ELTS_PER_CHUNK;

  constexpr bool USE_ROWWISE_SCALING = ROWWISE;
  constexpr size_t SCALE_DIM_Y = ROWWISE ? 1 : 32;
  constexpr size_t SCALE_DIM_X = ROWWISE ? 32 : 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE = DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);

  // Group-awareness: determine which tensor this block belongs to
  const bool is_single_tensor = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM);

  size_t tensor_id;
  size_t block_id_Y, block_id_X;

  if (is_single_tensor) {
    // SAME_BOTH_DIMS or VARYING_FIRST_DIM: simple 2D tiling over single logical tensor
    const size_t chunks_X = DIVUP(last_logical_dim, CHUNK_DIM_X);
    block_id_Y = blockIdx.x / chunks_X;
    block_id_X = blockIdx.x % chunks_X;
    const size_t block_global_offset = blockIdx.x * ELTS_PER_CHUNK;
    tensor_id =
        get_current_tensor_id<CHUNK_DIM_Y>(shape_rep, num_tensors, block_global_offset, block_id_Y,
                                           first_logical_dim, last_logical_dim, offsets_ptr);
  } else if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM) {
    // Virtual 2D grid: DIVUP(R,128) row-tiles x (total_cols/128) col-tiles
    const size_t chunks_X_total = last_logical_dim / CHUNK_DIM_X;
    const size_t col_chunk_global = blockIdx.x % chunks_X_total;
    block_id_Y = blockIdx.x / chunks_X_total;
    // Search using column-based element offset (works with existing binary search)
    const size_t search_offset = col_chunk_global * CHUNK_DIM_X * first_logical_dim;
    tensor_id =
        get_current_tensor_id<CHUNK_DIM_Y>(shape_rep, num_tensors, search_offset, block_id_Y,
                                           first_logical_dim, last_logical_dim, offsets_ptr);
    const size_t tensor_col_start = static_cast<size_t>(offsets_ptr[tensor_id]) / first_logical_dim;
    block_id_X = col_chunk_global - tensor_col_start / CHUNK_DIM_X;
  } else {
    // VARYING_BOTH_DIMS: 1D grid, element-offset-based (both dims 128-aligned)
    const size_t block_global_offset = blockIdx.x * ELTS_PER_CHUNK;
    const size_t chunks_X_for_id = DIVUP(last_logical_dim, CHUNK_DIM_X);
    tensor_id = get_current_tensor_id<CHUNK_DIM_Y>(shape_rep, num_tensors, block_global_offset,
                                                   blockIdx.x / chunks_X_for_id, first_logical_dim,
                                                   last_logical_dim, offsets_ptr);
    const size_t vb_tensor_base = static_cast<size_t>(offsets_ptr[tensor_id]);
    const size_t vb_cols =
        get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);
    const size_t chunks_X = DIVUP(vb_cols, CHUNK_DIM_X);
    const size_t block_id_in_tensor = blockIdx.x - vb_tensor_base / ELTS_PER_CHUNK;
    block_id_Y = block_id_in_tensor / chunks_X;
    block_id_X = block_id_in_tensor % chunks_X;
  }

  const size_t rows =
      get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);

  // Compute per-tensor scale stride from cols (matches group_quantize kernel)
  const size_t scale_stride = USE_ROWWISE_SCALING
                                  ? DIVUP_TO_MULTIPLE(DIVUP(cols, static_cast<size_t>(32)), 4)
                                  : DIVUP_TO_MULTIPLE(cols, 128);

  const size_t tensor_base = is_single_tensor ? 0 : static_cast<size_t>(offsets_ptr[tensor_id]);

  // Select TMA descriptors (static for single tensor, per-tensor for multi-tensor)
  const CUtensorMap &tensor_map_input =
      is_single_tensor ? tensor_map_input_static : g_tensor_maps_input[tensor_id];
  const CUtensorMap &tensor_map_output =
      is_single_tensor ? tensor_map_output_static : g_tensor_maps_output[tensor_id];

  if (!is_single_tensor) {
    fence_acquire_tensormap(&tensor_map_input);
    fence_acquire_tensormap(&tensor_map_output);
  }

  const int chunk_offset_Y = block_id_Y * CHUNK_DIM_Y;
  const int chunk_offset_X = block_id_X * CHUNK_DIM_X;

  // Per-tensor scale offset
  constexpr size_t SCALE_DIVISOR = USE_ROWWISE_SCALING ? SCALE_DIM_X : SCALE_DIM_Y;
  size_t scales_base_offset;
  if (is_single_tensor) {
    scales_base_offset = 0;
  } else if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM) {
    const size_t sum_prev_cols = tensor_base / first_logical_dim;
    if constexpr (USE_ROWWISE_SCALING) {
      // Scale layout: DIVUP_TO_MULTIPLE(R, 128) rows x (Ki/32) cols per tensor
      const size_t padded_rows = DIVUP_TO_MULTIPLE(first_logical_dim, static_cast<size_t>(128));
      scales_base_offset = (padded_rows / SCALE_DIM_X) * sum_prev_cols;
    } else {
      // Scale layout: DIVUP_TO_MULTIPLE(ceil(R/32), 4) rows x Ki cols per tensor
      const size_t padded_scale_rows = DIVUP_TO_MULTIPLE(
          DIVUP(first_logical_dim, static_cast<size_t>(SCALE_DIM_Y)), static_cast<size_t>(4));
      scales_base_offset = padded_scale_rows * sum_prev_cols;
    }
  } else {
    // VARYING_BOTH_DIMS: both dims 128-padded, original formula is exact
    scales_base_offset = tensor_base / SCALE_DIVISOR;
  }
  const e8m0_t *const tensor_scales_ptr = scales_ptr + scales_base_offset;

  const int scales_rowwise_chunk_offset_Y = block_id_Y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = block_id_X * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = block_id_Y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = block_id_X * SCALES_COLWISE_PER_CHUNK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;

  // Static shared memory (matching single-tensor dequantize)
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) IType in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) OType out_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;
  constexpr int iteration_zero = 0;
  constexpr int buffer_zero = 0;
  if (is_master_thread) {
    const int chunk_stage_offset_Y = chunk_offset_Y;
    const int chunk_stage_offset_X = chunk_offset_X;
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(&in_sh[buffer_zero]),
        reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
        chunk_stage_offset_Y, &mbar[iteration_zero]);

    ptx::mbarrier_arrive_expect_tx(&mbar[iteration_zero], transaction_size);
  } else {
    ptx::mbarrier_arrive(&mbar[iteration_zero]);
  }

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const int buff = iter % BUFFERS_NUM;
    const int next_iter = iter + 1;
    if (next_iter < ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_iter % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_iter * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
            chunk_it_offset_y, &mbar[next_iter]);

        ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size);
      } else {
        ptx::mbarrier_arrive(&mbar[next_iter]);
      }
    }

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

    const int scale_offset_Y =
        USE_ROWWISE_SCALING ? (scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y + tid_rowwise_Y)
                            : (scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y);

    const int scale_offset_X =
        USE_ROWWISE_SCALING
            ? (scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE)
            : (scales_colwise_chunk_offset_X + tid_colwise_X);

    const int scale_idx = scale_offset_Y * scale_stride + scale_offset_X;
    const e8m0_t biased_exponent = tensor_scales_ptr[scale_idx];
    const float block_scale = ptx::exp2f(biased_exponent);

    if constexpr (USE_ROWWISE_SCALING) {
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;

      const int shmem_offset_y = thread_offset_Y;
      const int shmem_offset_x = thread_offset_X_rowwise;
      in.load_from(&in_sh[buff][shmem_offset_y][shmem_offset_x]);

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
        out.data.elt[j] = static_cast<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[buff][shmem_offset_y][shmem_offset_x]);
    } else {
#pragma unroll
      for (int i = 0; i < BUFFER_DIM_Y; ++i) {
        const float elt = static_cast<float>(in_sh[buff][i][tid_colwise_X]);
        out_sh[buff][i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
      }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
          chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read<1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  destroy_barriers<ITERATIONS>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace group_dequantize_kernel

inline void group_dequantize(const GroupedTensor *input, GroupedTensor *output,
                             cudaStream_t stream) {
  using namespace group_dequantize_kernel;

  checkCuDriverContext(stream);

  const bool use_rowwise_scaling = input->has_data();
  const bool use_colwise_scaling = input->has_columnwise_data();
  NVTE_CHECK(use_rowwise_scaling || use_colwise_scaling,
             "Input tensor must have either rowwise or columnwise data.");
  NVTE_CHECK(!(use_rowwise_scaling && use_colwise_scaling),
             "Dequantize only supports rowwise or columnwise scaling, not both simultaneously.");

  NVTE_CHECK(!input->with_gemm_swizzled_scales, "Input must have scales in compact format.");
  NVTE_CHECK(!is_fp8_dtype(output->dtype()), "Output must be in higher precision.");
  NVTE_CHECK(!is_fp4_dtype(output->dtype()), "Output must not be FP4.");
  NVTE_CHECK(is_fp8_dtype(input->dtype()), "Input must have FP8 type.");

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (input->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (input->all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else if (input->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (input->varying_both_dims()) {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }

  const bool is_single_tensor = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM);

  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t elts_total = first_logical_dim * last_logical_dim;

  const size_t num_tensors = input->num_tensors;

  constexpr size_t CHUNK_DIM_Y = DequantizeConfig::CHUNK_DIM_Y;
  constexpr size_t CHUNK_DIM_X = DequantizeConfig::CHUNK_DIM_X;
  constexpr size_t THREADS_PER_CHUNK = DequantizeConfig::THREADS_PER_CHUNK;
  constexpr size_t SHMEM_DIM_Y = DequantizeConfig::SHMEM_DIM_Y;
  constexpr size_t SHMEM_DIM_X = DequantizeConfig::SHMEM_DIM_X;

  size_t blocks = 0;
  if (is_single_tensor) {
    const size_t blocks_Y = DIVUP(first_logical_dim, CHUNK_DIM_Y);
    const size_t blocks_X = DIVUP(last_logical_dim, CHUNK_DIM_X);
    blocks = blocks_Y * blocks_X;
  } else {
    NVTE_CHECK(num_tensors <= MAX_SUPPORTED_TENSOR_DESCRIPTORS,
               "Number of tensors in a group is larger than "
               "the MAX number of supported descriptors (64).");
    NVTE_CHECK(last_logical_dim % CHUNK_DIM_X == 0,
               "Last dimension of a grouped tensor should be divisible by 128.");
    if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM) {
      blocks = DIVUP(first_logical_dim, CHUNK_DIM_Y) * (last_logical_dim / CHUNK_DIM_X);
    } else {
      blocks = DIVUP(elts_total, CHUNK_DIM_Y * CHUNK_DIM_X);
    }
  }

  const dim3 grid(blocks);
  const dim3 block(THREADS_PER_CHUNK);

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(input->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(input->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(input->last_dims.dptr);

  const e8m0_t *const scales_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(input->scale_inv.dptr)
                          : reinterpret_cast<e8m0_t *>(input->columnwise_scale_inv.dptr);

  const SimpleTensor &input_data = use_rowwise_scaling ? input->data : input->columnwise_data;

  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_input{};
          alignas(64) CUtensorMap tensor_map_output{};

          create_2D_tensor_map(tensor_map_input, input_data, first_logical_dim, last_logical_dim,
                               SHMEM_DIM_Y, SHMEM_DIM_X, last_logical_dim, 0,
                               typeToNumBits(input->dtype()));
          create_2D_tensor_map(tensor_map_output, output->data, first_logical_dim, last_logical_dim,
                               SHMEM_DIM_Y, SHMEM_DIM_X, last_logical_dim, 0,
                               typeToNumBits(output->dtype()));

          // Update tensor descriptors before launching the kernel
          if (!is_single_tensor) {
            const IType *const input_dptr = reinterpret_cast<const IType *>(input_data.dptr);
            OType *const output_dptr = reinterpret_cast<OType *>(output->data.dptr);

            update_tma_descriptors<IType, OType><<<num_tensors, 32, 0, stream>>>(
                tensor_map_input, tensor_map_output, input_dptr, output_dptr, shape_rep,
                num_tensors, first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr,
                last_dims_ptr);
          }

          if (use_rowwise_scaling) {
            group_dequantize_mxfp8_kernel<IType, OType, true><<<grid, block, 0, stream>>>(
                tensor_map_input, tensor_map_output, shape_rep, num_tensors, first_logical_dim,
                last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr, scales_ptr);
          } else {
            group_dequantize_mxfp8_kernel<IType, OType, false><<<grid, block, 0, stream>>>(
                tensor_map_input, tensor_map_output, shape_rep, num_tensors, first_logical_dim,
                last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr, scales_ptr);
          });  // NOLINT(*)
  );           // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_DEQUANTIZE_MXFP8_CUH_
