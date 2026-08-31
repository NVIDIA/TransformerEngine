/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_transpose_nvfp4_tuned_1D.cuh
 *  \brief Tuned grouped kernel to cast to NVFP4 and transpose.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../../common.h"
#include "../../../util/cuda_runtime.h"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"
#include "../../../utils.cuh"
#include "../../core/common.cuh"
#include "../core_nvfp4.cuh"
#include "scaling_nvfp4_tuned_1D.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace group_quantize_transpose_tuned_kernel {

using namespace quantization_and_transposition_SF;
using namespace core;
using namespace ptx;
using namespace dispatch::common;

#if FP4_TYPE_SUPPORTED

using tuned_1D_scaling_common::colwise_scaling;
using tuned_1D_scaling_common::rowwise_scaling;

template <ShapeRepresentation SHAPE_REP>
struct TunableConfig {
  static constexpr bool PERSISTENT = true;
  static constexpr int BLOCKS_PER_SM = 128;
  static_assert(BLOCKS_PER_SM > 0,
                "STATIC_PERSISTENT_BLOCKS_PER_SM must be greater than zero in persistent mode.");
};

template <>
struct TunableConfig<ShapeRepresentation::SAME_BOTH_DIMS> {
  static constexpr bool PERSISTENT = false;
  static constexpr int BLOCKS_PER_SM = 1;
};

using RNG_t = typename transformer_engine::curanddx::detail::philox4x32_native_state<10>;

using ScalingTraits = tuned_1D_scaling_common::GroupedKernelTraits;
using IType = typename ScalingTraits::IType;
using IType3D = typename ScalingTraits::IType3D;
using OType2x3D = typename ScalingTraits::OType2x3D;
using OType2xt3D = typename ScalingTraits::OType2xt3D;
using ScalesType2D = typename ScalingTraits::ScalesType2D;
using ScalesTypeTr2D = typename ScalingTraits::ScalesTypeTr2D;

template <ShapeRepresentation SHAPE_REP>
struct WorkProvider {
  static constexpr bool FIXED_X_DIM = SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                                      SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM;
  static constexpr int CHUNK_DIM_Y = ScalingTraits::CHUNK_DIM_Y;
  static constexpr int CHUNK_DIM_X = ScalingTraits::CHUNK_DIM_X;

  TensorMetadata metadata_;
  int tensor_id_;
  int rows_;
  int cols_;
  int scale_stride_;
  int scale_stride_t_;
  int block_id_X_;
  size_t blocks_X_per_tensor_;
  size_t blocks_Y_per_tensor_;
  size_t tiles_per_tensor_;
  size_t current_block_id_Y_;
  size_t next_block_id_Y_;
  size_t current_tile_id_;
  size_t next_tile_id_;
  size_t work_stride_;
  size_t rowwise_scale_base_;
  size_t colwise_scale_base_;
  size_t launch_block_id_;
  bool valid_;

  __device__ __forceinline__ WorkProvider(const size_t num_tensors, const int common_rows,
                                          const int common_cols, const int common_scale_stride,
                                          const int common_scale_stride_t,
                                          const size_t common_blocks_Y_per_tensor)
      : valid_(false) {
    if constexpr (FIXED_X_DIM) {
      tensor_id_ = static_cast<int>(blockIdx.z);
      rows_ = common_rows;
      cols_ = common_cols;
      scale_stride_ = common_scale_stride;
      scale_stride_t_ = common_scale_stride_t;
      blocks_X_per_tensor_ = 0;
      blocks_Y_per_tensor_ = common_blocks_Y_per_tensor;
      tiles_per_tensor_ = 0;
      current_tile_id_ = 0;
      next_tile_id_ = 0;

      if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
        metadata_ = g_tensor_metadata[tensor_id_];
        if (metadata_.rows == 0 || metadata_.cols == 0) {
          return;
        }
        rows_ = static_cast<int>(metadata_.rows);
        scale_stride_t_ = DIVUP_TO_MULTIPLE(DIVUP(rows_, static_cast<int>(NVFP4_SCALE_DIM)), 4);
        blocks_Y_per_tensor_ = DIVUP(rows_, static_cast<int>(CHUNK_DIM_Y));
      }

      block_id_X_ = static_cast<int>(blockIdx.x);
      current_block_id_Y_ = static_cast<size_t>(blockIdx.y);
      if (current_block_id_Y_ >= blocks_Y_per_tensor_) {
        return;
      }

      launch_block_id_ = (static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
      work_stride_ = static_cast<size_t>(gridDim.y);

      const size_t tensor_id_u = static_cast<size_t>(tensor_id_);
      rowwise_scale_base_ = tensor_id_u * rows_ * scale_stride_;
      colwise_scale_base_ = tensor_id_u * cols_ * scale_stride_t_;
      if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
        rowwise_scale_base_ = metadata_.rowwise_scale_base;
        colwise_scale_base_ = metadata_.colwise_scale_base;
      }
    } else {
      const size_t tensor_id_u = static_cast<size_t>(blockIdx.y);
      if (tensor_id_u >= num_tensors) {
        return;
      }

      tensor_id_ = static_cast<int>(tensor_id_u);
      metadata_ = g_tensor_metadata[tensor_id_];
      const size_t rows_u = metadata_.rows;
      const size_t cols_u = metadata_.cols;
      if (rows_u == 0 || cols_u == 0) {
        return;
      }

      rows_ = static_cast<int>(rows_u);
      cols_ = static_cast<int>(cols_u);
      scale_stride_ = DIVUP_TO_MULTIPLE(DIVUP(cols_u, static_cast<size_t>(NVFP4_SCALE_DIM)), 4);
      scale_stride_t_ = DIVUP_TO_MULTIPLE(DIVUP(rows_u, static_cast<size_t>(NVFP4_SCALE_DIM)), 4);
      block_id_X_ = 0;
      blocks_X_per_tensor_ = DIVUP(cols_u, static_cast<size_t>(CHUNK_DIM_X));
      blocks_Y_per_tensor_ = DIVUP(rows_u, static_cast<size_t>(CHUNK_DIM_Y));
      tiles_per_tensor_ = blocks_X_per_tensor_ * blocks_Y_per_tensor_;
      current_block_id_Y_ = 0;
      next_block_id_Y_ = 0;
      current_tile_id_ = static_cast<size_t>(blockIdx.x);
      if (current_tile_id_ >= tiles_per_tensor_) {
        return;
      }

      launch_block_id_ = tensor_id_u * gridDim.x + blockIdx.x;
      work_stride_ = static_cast<size_t>(gridDim.x);
      rowwise_scale_base_ = metadata_.rowwise_scale_base;
      colwise_scale_base_ = metadata_.colwise_scale_base;
    }

    valid_ = true;
  }

  __device__ __forceinline__ bool is_valid() const { return valid_; }
  __device__ __forceinline__ int tensor_id() const { return tensor_id_; }
  __device__ __forceinline__ int rows() const { return rows_; }
  __device__ __forceinline__ int cols() const { return cols_; }
  __device__ __forceinline__ int scale_stride() const { return scale_stride_; }
  __device__ __forceinline__ int scale_stride_t() const { return scale_stride_t_; }
  __device__ __forceinline__ size_t rowwise_scale_base() const { return rowwise_scale_base_; }
  __device__ __forceinline__ size_t colwise_scale_base() const { return colwise_scale_base_; }
  __device__ __forceinline__ size_t launch_block_id() const { return launch_block_id_; }

  __device__ __forceinline__ void current_block_ids(int &block_id_Y, int &block_id_X) const {
    if constexpr (FIXED_X_DIM) {
      block_id_Y = static_cast<int>(current_block_id_Y_);
      block_id_X = block_id_X_;
    } else {
      const size_t block_id_Y_u = current_tile_id_ / blocks_X_per_tensor_;
      const size_t block_id_X_u = current_tile_id_ - block_id_Y_u * blocks_X_per_tensor_;
      block_id_Y = static_cast<int>(block_id_Y_u);
      block_id_X = static_cast<int>(block_id_X_u);
    }
  }

  __device__ __forceinline__ void prepare_next(bool &job_finished,
                                               int &prefetch_block_offset_Y,
                                               int &prefetch_block_offset_X) {
    if constexpr (FIXED_X_DIM) {
      next_block_id_Y_ = current_block_id_Y_ + work_stride_;
      job_finished = (next_block_id_Y_ >= blocks_Y_per_tensor_);
      if (!job_finished) {
        prefetch_block_offset_Y = static_cast<int>(next_block_id_Y_ * CHUNK_DIM_Y);
        prefetch_block_offset_X = block_id_X_ * CHUNK_DIM_X;
      }
    } else {
      next_tile_id_ = current_tile_id_ + work_stride_;
      job_finished = (next_tile_id_ >= tiles_per_tensor_);
      if (!job_finished) {
        const size_t prefetch_block_id_Y = next_tile_id_ / blocks_X_per_tensor_;
        const size_t prefetch_block_id_X = next_tile_id_ - prefetch_block_id_Y * blocks_X_per_tensor_;
        prefetch_block_offset_Y = static_cast<int>(prefetch_block_id_Y * CHUNK_DIM_Y);
        prefetch_block_offset_X = static_cast<int>(prefetch_block_id_X * CHUNK_DIM_X);
      }
    }
  }

  __device__ __forceinline__ void commit_next() {
    if constexpr (FIXED_X_DIM) {
      current_block_id_Y_ = next_block_id_Y_;
    } else {
      current_tile_id_ = next_tile_id_;
    }
  }
};

template <ShapeRepresentation SHAPE_REP, bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH,
          bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(ScalingTraits::THREADS_NUM)
group_quantize_transpose_nvfp4_tuned_1D_kernel(
    const size_t num_tensors,
    nvfp4_scale_t *const scales_ptr,
    nvfp4_scale_t *const scales_t_ptr,
    const float *noop,
    const float *const amax_rowwise_ptr,
    const float *const amax_colwise_ptr,
    const size_t amax_rowwise_numel,
    const size_t amax_colwise_numel,
    const int common_rows,
    const int common_cols,
    const int common_scale_stride,
    const int common_scale_stride_t,
    const size_t common_blocks_Y_per_tensor,
    const size_t *rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  WorkProvider<SHAPE_REP> work(num_tensors, common_rows, common_cols, common_scale_stride,
                               common_scale_stride_t, common_blocks_Y_per_tensor);
  if (!work.is_valid()) {
    return;
  }

  extern __shared__ char dynamic_shmem[];
  __shared__ uint64_t IN_buff_readable_mbar[ScalingTraits::BUFFS_NUM];

  constexpr int CHUNK_DIM_Y = ScalingTraits::CHUNK_DIM_Y;
  constexpr int CHUNK_DIM_X = ScalingTraits::CHUNK_DIM_X;
  constexpr int PREFETCH_STAGES = ScalingTraits::PREFETCH_STAGES;
  constexpr int THREADS_NUM = ScalingTraits::THREADS_NUM;
  constexpr int TILE_DIM_Y = ScalingTraits::TILE_DIM_Y;
  constexpr int TILE_DIM_X = ScalingTraits::TILE_DIM_X;
  constexpr int STAGES_X = ScalingTraits::STAGES_X;
  constexpr int STAGES = ScalingTraits::STAGES;
  constexpr int BUFFS_NUM = ScalingTraits::BUFFS_NUM;
  constexpr int BUFFS_NUM_IN = ScalingTraits::BUFFS_NUM_IN;
  constexpr int BUFFS_NUM_OUT = ScalingTraits::BUFFS_NUM_OUT;
  constexpr int BUFFS_NUM_OUT_TR = ScalingTraits::BUFFS_NUM_OUT_TR;
  constexpr int BUFF_SIZE_ALIGNED_IN = ScalingTraits::BUFF_SIZE_ALIGNED_IN;
  constexpr int BUFF_SIZE_ALIGNED_OUT = ScalingTraits::BUFF_SIZE_ALIGNED_OUT;
  constexpr int BUFF_SIZE_ALIGNED_OUT_TR = ScalingTraits::BUFF_SIZE_ALIGNED_OUT_TR;
  constexpr int BUFF_SIZE_ROWWISE_SCALES = ScalingTraits::BUFF_SIZE_ROWWISE_SCALES;
  constexpr int SCALES_PER_CHUNK_X = ScalingTraits::SCALES_PER_CHUNK_X;
  constexpr int SCALES_PER_CHUNK_Y = ScalingTraits::SCALES_PER_CHUNK_Y;

  const int tensor_id = work.tensor_id();
  const int rows = work.rows();
  const int cols = work.cols();
  const int scale_stride = work.scale_stride();
  const int scale_stride_t = work.scale_stride_t();

  const size_t rng_sequence = threadIdx.x + work.launch_block_id() * THREADS_NUM;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG_t rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0, 0, 0, 0};
  int rnd_idx = 0;

  const bool leading_thread = (threadIdx.x == 0);

  const int amax_rowwise_idx = (amax_rowwise_numel > 1) ? tensor_id : 0;
  const float S_enc_rowwise = (amax_rowwise_ptr == nullptr || amax_rowwise_numel == 0)
                              ? 1.0f
                              : core::compute_global_encode_scaling_factor_FP4(amax_rowwise_ptr[amax_rowwise_idx]);
  const int amax_colwise_idx = (amax_colwise_numel > 1) ? tensor_id : 0;
  const float S_enc_colwise = (amax_colwise_ptr == nullptr || amax_colwise_numel == 0)
                              ? S_enc_rowwise
                              : core::compute_global_encode_scaling_factor_FP4(amax_colwise_ptr[amax_colwise_idx]);

  nvfp4_scale_t *const scales_rowwise = scales_ptr + work.rowwise_scale_base();
  nvfp4_scale_t *const scales_colwise = RETURN_TRANSPOSE
                                        ? (scales_t_ptr + work.colwise_scale_base())
                                        : nullptr;

  const CUtensorMap &tensor_map_input = g_tensor_maps.input[tensor_id];
  const CUtensorMap &tensor_map_output = g_tensor_maps.output_rowwise[tensor_id];
  const CUtensorMap &tensor_map_output_t = g_tensor_maps.output_colwise[tensor_id];

  constexpr int in_mem = BUFF_SIZE_ALIGNED_IN;
  constexpr int out_mem_rowwise_data = BUFF_SIZE_ALIGNED_OUT;
  constexpr int out_mem_colwise_data = RETURN_TRANSPOSE ? BUFF_SIZE_ALIGNED_OUT_TR : 0;
  constexpr int out_mem_rowwise_scales = BUFF_SIZE_ROWWISE_SCALES;

  char *dshmem = align_up(dynamic_shmem, TMA_SHMEM_ALIGNMENT);

  IType *sIn_ptr = reinterpret_cast<IType *>(dshmem);
  fp4e2m1x2 *sOut_ptr = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem);
  fp4e2m1x2 *sOut_tr_ptr = reinterpret_cast<fp4e2m1x2 *>(dshmem + in_mem + out_mem_rowwise_data);

  auto &sIn = *reinterpret_cast<IType3D *>(sIn_ptr);
  auto &sOut = *reinterpret_cast<OType2x3D *>(sOut_ptr);
  auto &sOut_tr = *reinterpret_cast<OType2xt3D *>(sOut_tr_ptr);

  nvfp4_scale_t *sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t *sSFcolwise_ptr = reinterpret_cast<nvfp4_scale_t *>(
      dshmem + in_mem + out_mem_rowwise_data + out_mem_colwise_data + out_mem_rowwise_scales);
  auto &sSFrowwise = *reinterpret_cast<ScalesType2D *>(sSFrowwise_ptr);
  auto &sSFcolwise = *reinterpret_cast<ScalesTypeTr2D *>(sSFcolwise_ptr);

  constexpr int shmem_buff_size = BUFF_SIZE_ALIGNED_IN / BUFFS_NUM;

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  if (leading_thread) {
    fence_acquire_tensormap(&tensor_map_input);
    fence_acquire_tensormap(&tensor_map_output);
    if constexpr (RETURN_TRANSPOSE) {
      fence_acquire_tensormap(&tensor_map_output_t);
    }
  }

  {
    int first_block_id_Y = 0;
    int first_block_id_X = 0;
    work.current_block_ids(first_block_id_Y, first_block_id_X);
    const int first_block_offset_Y = first_block_id_Y * CHUNK_DIM_Y;
    const int first_block_offset_X = first_block_id_X * CHUNK_DIM_X;
#pragma unroll
    for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
      const int stage_Y = stage / STAGES_X;
      const int stage_X = stage % STAGES_X;
      const int stage_offset_Y = stage_Y * TILE_DIM_Y;
      const int stage_offset_X = stage_X * TILE_DIM_X;
      const int global_offset_Y = first_block_offset_Y + stage_offset_Y;
      const int global_offset_X = first_block_offset_X + stage_offset_X;
      if (leading_thread) {
        uint64_t *dst = reinterpret_cast<uint64_t *>(&sIn[stage]);
        const uint64_t *src = reinterpret_cast<const uint64_t *>(&tensor_map_input);
        uint64_t *barrier = &IN_buff_readable_mbar[stage];
        ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y,
                                                      barrier);
      }
    }
  }

  int buff_in = 0;
  int buff_out = 0;
  int buff_out_tr = 0;
  int IN_buff_readable_parity[BUFFS_NUM] = {0};

  bool job_finished = false;
  while (!job_finished) {
    int block_id_Y = 0;
    int block_id_X = 0;
    work.current_block_ids(block_id_Y, block_id_X);

    const int block_offset_Y = block_id_Y * CHUNK_DIM_Y;
    const int block_offset_X = block_id_X * CHUNK_DIM_X;
    const int block_offset_Y_tr = block_offset_X;
    const int block_offset_X_tr = block_offset_Y;
    const int chunk_rows = rows - block_offset_Y;
    const int chunk_cols = cols - block_offset_X;
    const int scales_block_offset_Y_rowwise = block_id_Y * CHUNK_DIM_Y;
    const int scales_block_offset_X_rowwise = block_id_X * SCALES_PER_CHUNK_X;
    const int scales_block_offset_Y_tr = block_id_X * CHUNK_DIM_X;
    const int scales_block_offset_X_tr = block_id_Y * SCALES_PER_CHUNK_Y;

    int prefetch_block_offset_Y = block_offset_Y;
    int prefetch_block_offset_X = block_offset_X;

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      const int stage_Y = stage / STAGES_X;
      const int stage_X = stage % STAGES_X;
      const int stage_offset_Y = stage_Y * TILE_DIM_Y;
      const int stage_offset_X = stage_X * TILE_DIM_X;

      if (stage == STAGES - PREFETCH_STAGES) {
        work.prepare_next(job_finished, prefetch_block_offset_Y, prefetch_block_offset_X);
      }

      if ((stage < STAGES - PREFETCH_STAGES) || !job_finished) {
        const int next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
        const int next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES;
        const int next_prefetch_stage_Y = next_prefetch_stage / STAGES_X;
        const int next_prefetch_stage_X = next_prefetch_stage % STAGES_X;
        const int next_prefetch_stage_offset_Y = next_prefetch_stage_Y * TILE_DIM_Y;
        const int next_prefetch_stage_offset_X = next_prefetch_stage_X * TILE_DIM_X;
        const bool prefetch_next_tile = (stage >= STAGES - PREFETCH_STAGES);
        const int prefetch_base_offset_Y =
            prefetch_next_tile ? prefetch_block_offset_Y : block_offset_Y;
        const int prefetch_base_offset_X =
            prefetch_next_tile ? prefetch_block_offset_X : block_offset_X;
        const int global_offset_Y = prefetch_base_offset_Y + next_prefetch_stage_offset_Y;
        const int global_offset_X = prefetch_base_offset_X + next_prefetch_stage_offset_X;

        if (leading_thread) {
          uint64_t *dst = reinterpret_cast<uint64_t *>(&sIn[next_prefetch_buff]);
          const uint64_t *src = reinterpret_cast<const uint64_t *>(&tensor_map_input);
          uint64_t *barrier = &IN_buff_readable_mbar[next_prefetch_buff];
          ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);
          ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y,
                                                        barrier);
        }
        ptx::fence_proxy_async_shared_cta();
      }

      ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                       IN_buff_readable_parity[buff_in]);
      IN_buff_readable_parity[buff_in] ^= 1;

      ptx::cp_async_bulk_wait_group_read<PREFETCH_STAGES>();
      __syncthreads();

      rowwise_scaling<ScalingTraits, USE_STOCHASTIC_ROUNDING, USE_FAST_MATH>(
          sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc_rowwise, stage_Y, stage_X, buff_in, buff_out,
          rng, random_uint4, rnd_idx);
      if constexpr (RETURN_TRANSPOSE) {
        colwise_scaling<ScalingTraits, USE_STOCHASTIC_ROUNDING, USE_FAST_MATH>(
            sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc_colwise, stage_Y, stage_X, buff_in,
            buff_out_tr, rng, random_uint4, rnd_idx);
      }

      ptx::fence_proxy_async_shared_cta();
      __syncthreads();

      if (leading_thread) {
        const int global_offset_Y = block_offset_Y + stage_offset_Y;
        const int global_offset_X = block_offset_X + stage_offset_X;
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&sOut[buff_out]));

        if constexpr (RETURN_TRANSPOSE) {
          const int global_offset_Y_tr = block_offset_Y_tr + stage_offset_X;
          const int global_offset_X_tr = block_offset_X_tr + stage_offset_Y;
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_tr,
              global_offset_Y_tr, reinterpret_cast<uint64_t *>(&sOut_tr[buff_out_tr]));
        }
        ptx::cp_async_bulk_commit_group();
      }

      buff_in = (buff_in + 1) % BUFFS_NUM_IN;
      buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
      buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
    }

    {
      using RowwiseScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_X>;
      const int rowwise_count =
          min(SCALES_PER_CHUNK_X, chunk_cols / static_cast<int>(NVFP4_SCALE_DIM));
      for (int row = threadIdx.x; row < CHUNK_DIM_Y; row += THREADS_NUM) {
        const int row_global_i = scales_block_offset_Y_rowwise + row;
        if (row_global_i < rows) {
          const size_t row_global = static_cast<size_t>(row_global_i);
          RowwiseScalesVec &scales_vec = *reinterpret_cast<RowwiseScalesVec *>(sSFrowwise[row]);
          const size_t scale_idx_global = row_global * scale_stride + scales_block_offset_X_rowwise;
          scales_vec.store_to_elts(&scales_rowwise[scale_idx_global], 0, rowwise_count);
        }
      }

      if constexpr (RETURN_TRANSPOSE) {
        using ColwiseScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
        const int colwise_count =
            min(SCALES_PER_CHUNK_Y, chunk_rows / static_cast<int>(NVFP4_SCALE_DIM));
        for (int row_tr = threadIdx.x; row_tr < CHUNK_DIM_X; row_tr += THREADS_NUM) {
          const int row_tr_global_i = scales_block_offset_Y_tr + row_tr;
          if (row_tr_global_i < cols) {
            const size_t row_tr_global = static_cast<size_t>(row_tr_global_i);
            ColwiseScalesVec &scales_vec = *reinterpret_cast<ColwiseScalesVec *>(sSFcolwise[row_tr]);
            const size_t scale_idx_global = row_tr_global * scale_stride_t + scales_block_offset_X_tr;
            scales_vec.store_to_elts(&scales_colwise[scale_idx_global], 0, colwise_count);
          }
        }
      }

      if (!job_finished) {
        work.commit_next();
        __syncthreads();
      }
    }
  }

  if (leading_thread) {
    ptx::cp_async_bulk_wait_group_read<0>();
  }
  __syncthreads();

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

template <ShapeRepresentation SHAPE_REP, bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH,
          bool RETURN_TRANSPOSE>
inline void launch_group_quantize_transpose_kernel(
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    nvfp4_scale_t *const scales_ptr, nvfp4_scale_t *const scales_t_ptr, const float *const noop_ptr,
    const float *const amax_rowwise_ptr, const float *const amax_colwise_ptr,
    const size_t amax_rowwise_numel, const size_t amax_colwise_numel,
    const size_t work_blocks_X, const size_t work_blocks_Y, const size_t *const rng_state,
    const int dshmem_size, cudaStream_t stream) {
  constexpr int CHUNK_DIM_Y = ScalingTraits::CHUNK_DIM_Y;
  constexpr int THREADS_NUM = ScalingTraits::THREADS_NUM;

  const int block_size = THREADS_NUM;
  auto kernel = group_quantize_transpose_nvfp4_tuned_1D_kernel
                <SHAPE_REP, USE_STOCHASTIC_ROUNDING, USE_FAST_MATH, RETURN_TRANSPOSE>;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));

  int active_blocks_per_sm = 0;
  NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_sm, kernel, block_size, dshmem_size));
  NVTE_CHECK(active_blocks_per_sm > 0, "Grouped NVFP4 optimized kernel has zero active blocks per SM.");

  const size_t sm_num = static_cast<size_t>(transformer_engine::cuda::sm_count());

  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    const size_t blocks_Y_per_tensor = DIVUP(rows_per_tensor, static_cast<size_t>(CHUNK_DIM_Y));
    const int rows = static_cast<int>(rows_per_tensor);
    const int cols = static_cast<int>(last_logical_dim);
    const int scale_stride = DIVUP_TO_MULTIPLE(DIVUP(last_logical_dim, static_cast<size_t>(NVFP4_SCALE_DIM)), 4);
    const int scale_stride_t = DIVUP_TO_MULTIPLE(DIVUP(rows_per_tensor, static_cast<size_t>(NVFP4_SCALE_DIM)), 4);
    const size_t requested_grid_size = sm_num * static_cast<size_t>(active_blocks_per_sm);
    const size_t workers_X_total = num_tensors * work_blocks_X;
    const size_t requested_workers_Y_per_tensor = std::max<size_t>(size_t{1}, requested_grid_size / workers_X_total);
    const size_t workers_Y_per_tensor = std::min(blocks_Y_per_tensor, requested_workers_Y_per_tensor);
    const dim3 grid(work_blocks_X, workers_Y_per_tensor, num_tensors);

    kernel<<<grid, block_size, dshmem_size, stream>>>(
        num_tensors, scales_ptr, scales_t_ptr, noop_ptr, amax_rowwise_ptr, amax_colwise_ptr,
        amax_rowwise_numel, amax_colwise_numel, rows, cols, scale_stride, scale_stride_t,
        blocks_Y_per_tensor, rng_state);
  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
    const int cols = static_cast<int>(last_logical_dim);
    const int scale_stride = DIVUP_TO_MULTIPLE(DIVUP(last_logical_dim, static_cast<size_t>(NVFP4_SCALE_DIM)), 4);
    const size_t requested_grid_size = sm_num * static_cast<size_t>(active_blocks_per_sm);
    const size_t workers_X_total = num_tensors * work_blocks_X;
    const size_t requested_workers_Y_per_tensor = std::max<size_t>(size_t{1}, requested_grid_size / workers_X_total);
    const size_t avg_blocks_Y_per_tensor = std::max<size_t>(size_t{1}, DIVUP(work_blocks_Y, num_tensors));
    const size_t workers_Y_per_tensor = std::min(avg_blocks_Y_per_tensor, requested_workers_Y_per_tensor);
    NVTE_CHECK(workers_Y_per_tensor > 0, "VARYING_FIRST_DIM persistent grid size must be greater than zero.");
    const dim3 grid(work_blocks_X, workers_Y_per_tensor, num_tensors);

    kernel<<<grid, block_size, dshmem_size, stream>>>(
        num_tensors, scales_ptr, scales_t_ptr, noop_ptr, amax_rowwise_ptr, amax_colwise_ptr,
        amax_rowwise_numel, amax_colwise_numel, 0, cols, scale_stride, 0, 0, rng_state);
  } else {
    const size_t total_work_blocks = work_blocks_X * work_blocks_Y;
    const size_t persistent_blocks_per_sm = std::min(active_blocks_per_sm, TunableConfig<SHAPE_REP>::BLOCKS_PER_SM);
    const size_t requested_workers_per_tensor = std::max<size_t>(size_t{1}, (sm_num * persistent_blocks_per_sm) / num_tensors);
    const size_t workers_per_tensor = std::min(total_work_blocks, requested_workers_per_tensor);
    NVTE_CHECK(workers_per_tensor > 0, "Tensor-local persistent grid size must be greater than zero.");
    const dim3 grid(workers_per_tensor, num_tensors);

    kernel<<<grid, block_size, dshmem_size, stream>>>(
        num_tensors, scales_ptr, scales_t_ptr, noop_ptr, amax_rowwise_ptr, amax_colwise_ptr,
        amax_rowwise_numel, amax_colwise_numel, 0, 0, 0, 0, 0, rng_state);
  }

  NVTE_CHECK_CUDA(cudaGetLastError());
}

#endif  // FP4_TYPE_SUPPORTED
}  // namespace group_quantize_transpose_tuned_kernel

inline void group_quantize_transpose(const GroupedTensor *input, const Tensor *noop,
                                     GroupedTensor *output, const QuantizationConfig *quant_config,
                                     cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_quantize_transpose_tuned_kernel;
  using namespace ptx;

  constexpr int CHUNK_DIM_Y = ScalingTraits::CHUNK_DIM_Y;
  constexpr int CHUNK_DIM_X = ScalingTraits::CHUNK_DIM_X;
  constexpr size_t ELTS_PER_CHUNK = ScalingTraits::ELTS_PER_CHUNK;
  constexpr int BUFF_DIM_Y = ScalingTraits::BUFF_DIM_Y;
  constexpr int BUFF_DIM_X = ScalingTraits::BUFF_DIM_X;
  constexpr int BUFF_SIZE_ALIGNED_IN = ScalingTraits::BUFF_SIZE_ALIGNED_IN;
  constexpr int BUFF_SIZE_ALIGNED_OUT = ScalingTraits::BUFF_SIZE_ALIGNED_OUT;
  constexpr int BUFF_SIZE_ALIGNED_OUT_TR = ScalingTraits::BUFF_SIZE_ALIGNED_OUT_TR;
  constexpr int BUFF_SIZE_ROWWISE_SCALES = ScalingTraits::BUFF_SIZE_ROWWISE_SCALES;
  constexpr int BUFF_SIZE_COLWISE_SCALES = ScalingTraits::BUFF_SIZE_COLWISE_SCALES;

  const bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;
  const bool use_fast_math = quant_config ? quant_config->use_fast_math : false;
  const bool return_transpose = output->has_columnwise_data();

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  NVTE_CHECK(input->num_tensors == output->num_tensors, "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(input->dtype() == DType::kBFloat16, "Optimized grouped NVFP4 kernel supports only BF16 input.");
  NVTE_CHECK(output->has_data(), "Grouped NVFP4 output tensor must be allocated.");
  NVTE_CHECK(is_fp4_dtype(output->dtype()), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "Output must have scales in compact format.");
  if (return_transpose) {
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype), "Transposed output must have FP4 type.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr, "Transposed scaling tensor must be allocated.");
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

  const bool use_single_work_grid = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                     shape_rep == ShapeRepresentation::VARYING_FIRST_DIM);

  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t elts_total = first_logical_dim * last_logical_dim;
  const size_t num_tensors = input->num_tensors;

  NVTE_CHECK(num_tensors <= MAX_SUPPORTED_TENSOR_DESCRIPTORS,
             "Number of tensors in a group is larger than the MAX number of supported "
             "descriptors (64).");
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS: {
      NVTE_CHECK(first_logical_dim % num_tensors == 0,
                 "First logical dimension of a grouped tensor must be divisible by the number of "
                 "tensors.");
      NVTE_CHECK((first_logical_dim / num_tensors) % 128 == 0,
                 "First dimension of each tensor in a group must be divisible by 128.");
      break;
    }
    case ShapeRepresentation::VARYING_FIRST_DIM: {
      NVTE_CHECK(first_logical_dim % 128 == 0,
                 "First logical dimension of a grouped tensor must be divisible by 128.");
      break;
    }
    case ShapeRepresentation::VARYING_LAST_DIM: {
      NVTE_CHECK(first_logical_dim % 128 == 0,
                 "First logical dimension of a grouped tensor must be divisible by 128.");
      NVTE_CHECK(last_logical_dim % 128 == 0,
                 "Last logical dimension of a grouped tensor must be divisible by 128.");
      break;
    }
    case ShapeRepresentation::VARYING_BOTH_DIMS: {
      NVTE_CHECK(last_logical_dim % ELTS_PER_CHUNK == 0,
                 "Last logical dimension of a grouped tensor must be divisible by ",
                 CHUNK_DIM_Y, "x", CHUNK_DIM_X, ".");
      break;
    }
  }

  size_t work_blocks_X = 0;
  size_t work_blocks_Y = 0;
  if (use_single_work_grid) {
    work_blocks_Y = DIVUP(first_logical_dim, static_cast<size_t>(CHUNK_DIM_Y));
    work_blocks_X = DIVUP(last_logical_dim, static_cast<size_t>(CHUNK_DIM_X));
  } else {
    work_blocks_Y = 1;
    work_blocks_X = DIVUP(elts_total, ELTS_PER_CHUNK);
  }

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);

  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  nvfp4_scale_t *const scales_t_ptr = reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float *const amax_rowwise_ptr = reinterpret_cast<const float *>(input->amax.dptr);
  const float *const amax_colwise_ptr = reinterpret_cast<const float *>(input->columnwise_amax.dptr);
  const size_t amax_rowwise_numel = input->amax.has_data() ? input->amax.numel() : 0;
  const size_t amax_colwise_numel = input->columnwise_amax.has_data() ? input->columnwise_amax.numel() : 0;

  if (input->amax.has_data()) {
    NVTE_CHECK(amax_rowwise_numel == 1 || amax_rowwise_numel == num_tensors,
               "Rowwise amax must contain either 1 value or num_tensors values, found ",
               amax_rowwise_numel, " values for num_tensors=", num_tensors, ".");
  }
  if (input->columnwise_amax.has_data()) {
    NVTE_CHECK(amax_colwise_numel == 1 || amax_colwise_numel == num_tensors,
               "Columnwise amax must contain either 1 value or num_tensors values, found ",
               amax_colwise_numel, " values for num_tensors=", num_tensors, ".");
  }

  const NVTETensor rng_state_tensor = (quant_config != nullptr) ? quant_config->rng_state : nullptr;
  const size_t *rng_state = nullptr;
  if (rng_state_tensor != nullptr) {
    Tensor &rng_state_te_tensor = *convertNVTETensor(rng_state_tensor);
    NVTE_CHECK(rng_state_te_tensor.dtype() == DType::kInt64,
               "RNG state should contain 2 64-bit values.");
    NVTE_CHECK(rng_state_te_tensor.data.shape == std::vector<size_t>{2},
               "Shape of the RNG state should be [2], but got ", rng_state_te_tensor.data.shape);
    rng_state = reinterpret_cast<const size_t *>(rng_state_te_tensor.data.dptr);
  }

  alignas(64) CUtensorMap tensor_map_input{};
  alignas(64) CUtensorMap tensor_map_act_input{};
  alignas(64) CUtensorMap tensor_map_output{};
  alignas(64) CUtensorMap tensor_map_output_transpose{};

  const size_t dummy_first_logical_dim = 32;
  const size_t dummy_last_logical_dim = 32;
  create_2D_tensor_map(tensor_map_input, input->data, dummy_first_logical_dim, 
                       dummy_last_logical_dim, BUFF_DIM_Y,
                       BUFF_DIM_X, dummy_last_logical_dim, 0, sizeof(IType) * 8);
  create_2D_tensor_map(tensor_map_output, output->data, dummy_first_logical_dim,
                       dummy_last_logical_dim, BUFF_DIM_Y,
                       BUFF_DIM_X, dummy_last_logical_dim, 0, 4);
  if (return_transpose) {
    create_2D_tensor_map(tensor_map_output_transpose, output->columnwise_data,
                         dummy_last_logical_dim, dummy_first_logical_dim,
                         BUFF_DIM_X, BUFF_DIM_Y,
                         dummy_first_logical_dim, 0, 4);
  }

  const int in_mem = BUFF_SIZE_ALIGNED_IN;
  const int out_data_mem = BUFF_SIZE_ALIGNED_OUT;
  const int out_data_transpose_mem = return_transpose ? BUFF_SIZE_ALIGNED_OUT_TR : 0;
  const int out_scales_mem = BUFF_SIZE_ROWWISE_SCALES;
  const int out_scales_transpose_mem = return_transpose ? BUFF_SIZE_COLWISE_SCALES : 0;
  const int out_mem = out_data_mem + out_data_transpose_mem;
  const int dshmem_size = in_mem + out_mem + out_scales_transpose_mem + out_scales_mem + TMA_SHMEM_ALIGNMENT;

  const IType *const input_dptr = reinterpret_cast<const IType *>(input->data.dptr);
  const void *const output_dptr = output->data.dptr;
  const void *const output_t_dptr = return_transpose ? output->columnwise_data.dptr : nullptr;

  update_tma_descriptors<IType, void, true>
      <<<num_tensors, 1, 0, stream>>>(
          tensor_map_input, tensor_map_act_input, tensor_map_output, tensor_map_output_transpose,
          input_dptr, nullptr, output_dptr, output_t_dptr, shape_rep, num_tensors,
          first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr, last_dims_ptr, true,
          return_transpose, false);
  NVTE_CHECK_CUDA(cudaGetLastError());

  TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
      shape_rep, SHAPE_REP, {
        TRANSFORMER_ENGINE_SWITCH_CONDITION(
            use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,
            TRANSFORMER_ENGINE_SWITCH_CONDITION(
                use_fast_math, USE_FAST_MATH,
                TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
                  launch_group_quantize_transpose_kernel<SHAPE_REP, USE_STOCHASTIC_ROUNDING,
                                                         USE_FAST_MATH, RETURN_TRANSPOSE>(
                      num_tensors, first_logical_dim, last_logical_dim, scales_ptr, scales_t_ptr,
                      noop_ptr, amax_rowwise_ptr, amax_colwise_ptr, amax_rowwise_numel,
                      amax_colwise_numel, work_blocks_X, work_blocks_Y, rng_state, dshmem_size,
                      stream);
                });););
      });
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_
