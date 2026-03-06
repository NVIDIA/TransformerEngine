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

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace group_quantize_transpose_tuned_kernel {

using namespace quantization_and_transposition_SF;
using namespace core;
using namespace ptx;
using namespace dispatch::common;

#if FP4_TYPE_SUPPORTED

struct TunableConfig {
  static constexpr size_t CHUNK_DIM_Y = 128;
  static constexpr size_t CHUNK_DIM_X = 128;
  static constexpr int PREFETCH_STAGES = 1;
  static constexpr bool PERSISTENT = true;
  static constexpr size_t STATIC_PERSISTENT_BLOCKS_PER_SM = 4;
};

constexpr size_t CHUNK_DIM_Y = TunableConfig::CHUNK_DIM_Y;
constexpr size_t CHUNK_DIM_X = TunableConfig::CHUNK_DIM_X;
constexpr int PREFETCH_STAGES = TunableConfig::PREFETCH_STAGES;
constexpr bool PERSISTENT = TunableConfig::PERSISTENT;
constexpr size_t STATIC_PERSISTENT_BLOCKS_PER_SM = TunableConfig::STATIC_PERSISTENT_BLOCKS_PER_SM;

constexpr size_t ELTS_PER_CHUNK = CHUNK_DIM_Y * CHUNK_DIM_X;

static_assert(!PERSISTENT || (STATIC_PERSISTENT_BLOCKS_PER_SM > 0),
              "STATIC_PERSISTENT_BLOCKS_PER_SM must be greater than zero in persistent mode.");

constexpr int MAX_SUPPORTED_TENSOR_DESCRIPTORS = 64;
__device__ alignas(128) CUtensorMap g_tensor_maps_input[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output[MAX_SUPPORTED_TENSOR_DESCRIPTORS];
__device__ alignas(128) CUtensorMap g_tensor_maps_output_t[MAX_SUPPORTED_TENSOR_DESCRIPTORS];

constexpr int SCALE_DIM = 16;  // NVFP4 block (x16 elts)
constexpr int THREADS_NUM = 128;
constexpr int ELTS_PER_THREAD = 16;
constexpr int TILE_DIM_Y = 64;
constexpr int TILE_DIM_X = 64;

static_assert(ELTS_PER_THREAD == SCALE_DIM && "Hardcoded and fixed parameter\0");

static_assert((THREADS_NUM * ELTS_PER_THREAD <= TILE_DIM_Y * TILE_DIM_X) &&
              "Unbalanced threads workload\0");

static_assert((CHUNK_DIM_Y % TILE_DIM_Y == 0) &&
              "Chunk size Y must be evenly divisible by the tile size Y\0");
static_assert((CHUNK_DIM_X % TILE_DIM_X == 0) &&
              "Chunk size X must be evenly divisible by the tile size X\0");

static_assert((TILE_DIM_Y % SCALE_DIM == 0) &&
              "Tile size Y must be evenly divisible by the scale dim\0");
static_assert((TILE_DIM_X % SCALE_DIM == 0) &&
              "Tile size X must be evenly divisible by the scale dim\0");

constexpr int TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;
constexpr int TILES_X = CHUNK_DIM_X / TILE_DIM_X;

constexpr int THREADS_PER_SCALE_ROWWISE = SCALE_DIM / ELTS_PER_THREAD;

constexpr int SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;
constexpr int SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;

constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
constexpr int SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;

constexpr int STAGES_Y = TILES_Y;
constexpr int STAGES_X = TILES_X;
constexpr int STAGES = STAGES_Y * STAGES_X;

constexpr int BUFFS_NUM = PREFETCH_STAGES + 1;
constexpr int BUFFS_NUM_IN = BUFFS_NUM;
constexpr int BUFFS_NUM_OUT = BUFFS_NUM;
constexpr int BUFFS_NUM_OUT_TR = 2;
constexpr int BUFF_DIM_Y = TILE_DIM_Y;
constexpr int BUFF_DIM_X = TILE_DIM_X;
constexpr int BUFF_SIZE = BUFF_DIM_Y * BUFF_DIM_X;
constexpr int BUFF_SIZE_TOTAL = BUFF_SIZE * BUFFS_NUM;

// Input buffer (BF16)
constexpr int BUFF_IN_DIM_Y = BUFF_DIM_Y;
constexpr int BUFF_IN_DIM_X = BUFF_DIM_X;
constexpr int BUFF_IN_SIZE = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;
constexpr int BUFF_IN_ELTS_NUM = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;

// Output buffer (NVFP4)
constexpr int BUFF_OUT_DIM_Y = BUFF_DIM_Y;
constexpr int BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
constexpr int BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

// Output transpose buffer (NVFP4)
constexpr int BUFF_OUT_TR_DIM_Y = BUFF_DIM_X;
constexpr int BUFF_OUT_TR_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr int BUFF_OUT_TR_SIZE = BUFF_OUT_TR_DIM_Y * BUFF_OUT_TR_DIM_X;

// Manual swizzling parameters to reduce SHMEM bank conflicts
constexpr int PACK_SIZE = 8;
constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;

constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;
constexpr int THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;

constexpr int THREADS_X_TR = TILE_DIM_X / 2;
constexpr int THREADS_Y_TR = THREADS_NUM / THREADS_X_TR;

constexpr int ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;
constexpr int ITERATIONS_TR = SCALES_PER_TILE_Y / THREADS_Y_TR;
static_assert(ITERATIONS_TR >= 1 && "Number of transpose iterations should be >=1\0");
static_assert((SCALES_PER_TILE_Y % THREADS_Y_TR == 0) &&
              "Partial transpose iterations are not supported\0");

constexpr int BUFF_OUT_IT_OFFSET = BUFF_OUT_TR_DIM_X / ITERATIONS_TR / STAGES;

static_assert(BUFF_DIM_Y >= SCALE_DIM &&
              "Number of buffer rows must be greater or equal to the size of the columwise "
              "scaling block\0");
static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE &&
              "Number of buffer rows must be greater or equal to the number of rowwise "
              "processing threads in Y dimension\0");

// Number of 4-bit elements that span 32 banks (4-byte each) of shared memory
constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

// Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory
constexpr int THREADS_PER_BANK = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;

using IType = bf16;
using IType2 = typename ptx::FPx2<IType>;
using IType3D = IType[BUFFS_NUM_IN][BUFF_IN_DIM_Y][BUFF_IN_DIM_X];
using IType2x3D = IType2[BUFFS_NUM_IN][BUFF_IN_DIM_Y][BUFF_IN_DIM_X / 2];
using OType2x3D = fp4e2m1x2[BUFFS_NUM_OUT][BUFF_OUT_DIM_Y][BUFF_OUT_DIM_X];
using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_TR_DIM_Y][BUFF_OUT_TR_DIM_X];
using ScalesType2D = nvfp4_scale_t[CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
using ScalesTypeTr2D = nvfp4_scale_t[CHUNK_DIM_X][SCALES_PER_CHUNK_Y];
using RNG_t = typename transformer_engine::curanddx::detail::philox4x32_native_state<10>;

template <bool USE_FAST_MATH>
struct SCALING_COEFFICIENT_TYPE {};
template <>
struct SCALING_COEFFICIENT_TYPE<false> {
  using type = float;
};
template <>
struct SCALING_COEFFICIENT_TYPE<true> {
  using type = bf16;
};

__device__ __forceinline__ float get_amax_of_pair(const IType2 pair) {
  return static_cast<float>(__hmax(__habs(pair.x), __habs(pair.y)));
}

// Compute "correct" per-block encoding scaling factor
template <typename SF_TYPE>
__device__ __forceinline__ SF_TYPE
compute_nvfp4_scaling_coefficient(const nvfp4_scale_t S_dec_block, const float S_enc) {
  NVTE_DEVICE_ERROR("Unsupported scaling-factor type. Only FP32 and BF16 are supported.");
}

template <>
__device__ __forceinline__ float compute_nvfp4_scaling_coefficient<float>(
    const nvfp4_scale_t S_dec_block, const float S_enc) {
  const float S_dec = 1.0f / S_enc;
  const float scale_rcp =
      fminf(1.0f / (static_cast<float>(S_dec_block) * S_dec), detail::TypeExtrema<float>::max);
  return scale_rcp;
}

template <>
__device__ __forceinline__ bf16
compute_nvfp4_scaling_coefficient<bf16>(const nvfp4_scale_t S_dec_block, const float S_enc) {
  const float scale_rcp =
      fminf(S_enc / (static_cast<float>(S_dec_block)), detail::TypeExtrema<bf16>::max);
  return static_cast<bf16>(scale_rcp);
}

template <bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH>
__device__ __forceinline__ void colwise_scaling(const IType *__restrict__ sIn_ptr,
                                                fp4e2m1x2 *__restrict__ sOut_tr_ptr,
                                                nvfp4_scale_t *__restrict__ sSFcolwise_ptr,
                                                const float S_enc_colwise, const int stage_Y,
                                                const int stage_X, const int buff_in,
                                                const int buff_out_tr, RNG_t &rng,
                                                uint4 &random_uint4, int &rnd_idx) {
  using scaling_coeff_type = typename SCALING_COEFFICIENT_TYPE<USE_FAST_MATH>::type;

  const auto &sIn2x = *reinterpret_cast<const IType2x3D *>(sIn_ptr);
  auto &sOut_tr = *reinterpret_cast<OType2xt3D *>(sOut_tr_ptr);
  auto &sSFcolwise = *reinterpret_cast<ScalesTypeTr2D *>(sSFcolwise_ptr);

  const int warp = threadIdx.x / THREADS_PER_WARP;
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;

  const int tid_Y_colwise = (thread_lane % 4 + warp) % 4;
  const int tid_X_colwise = thread_lane;

  const int thread_offset_Y_colwise = tid_Y_colwise * SCALE_DIM;
  const int thread_offset_X_colwise = tid_X_colwise * 2;

  const int in_thread_offset_Y = thread_offset_Y_colwise;
  const int in_thread_offset_X = thread_offset_X_colwise / 2;

  const int out_tr_thread_offset_Y = thread_offset_X_colwise;
  const int out_tr_thread_offset_X = thread_offset_Y_colwise / 2;

  const int scale_tr_offset_Y = (stage_X * TILE_DIM_X) + 2 * tid_X_colwise;
  const int scale_tr_offset_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y_colwise;

  __align__(8) IType rIn[2][SCALE_DIM];
  // Read (cache) a pair of input elements (S2R). Find NVFP4-block AMAX
  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
  for (int i = 0; i < SCALE_DIM; ++i) {
    const IType2 elt_pair =
        ptx::ld_shared_b32(&sIn2x[buff_in][in_thread_offset_Y + i][in_thread_offset_X]);
    rIn[0][i] = elt_pair.x;
    rIn[1][i] = elt_pair.y;
    ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, elt_pair);
  }
  const float block_amax[2] = {static_cast<float>(__habs(thread_amax_2x.x)),
                               static_cast<float>(__habs(thread_amax_2x.y))};
#pragma unroll
  for (int w = 0; w < 2; ++w) {
    const nvfp4_scale_t S_dec_b_fp8 = compute_decoding_scaling_factor(block_amax[w], S_enc_colwise);

    // Store scaling factors to SMEM buffer (R2S)
    sSFcolwise[scale_tr_offset_Y + w][scale_tr_offset_X] = S_dec_b_fp8;

    const scaling_coeff_type SFcoefficient =
        compute_nvfp4_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_colwise);

    // Scale elements
    __align__(8) uint32_t rOut[SCALE_DIM / 8];
#pragma unroll
    for (int e = 0; e < SCALE_DIM / 8; ++e) {
      const uint64_t elts03 = *reinterpret_cast<uint64_t *>(&rIn[w][8 * e]);
      const uint64_t elts47 = *reinterpret_cast<uint64_t *>(&rIn[w][8 * e + 4]);
      if constexpr (USE_STOCHASTIC_ROUNDING) {
        const uint32_t rbits03 = core::get_rbits(rng, random_uint4, rnd_idx);
        const uint32_t rbits47 = core::get_rbits(rng, random_uint4, rnd_idx);
        rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<scaling_coeff_type>(
            elts03, elts47, SFcoefficient, rbits03, rbits47);
      } else {
        rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<scaling_coeff_type>(elts03, elts47,
                                                                                   SFcoefficient);
      }
    }
    uint64_t &out_pack_16x = *reinterpret_cast<uint64_t *>(rOut);
    ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_thread_offset_Y + w][out_tr_thread_offset_X],
                       out_pack_16x);
  }
}

template <bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH>
__device__ __forceinline__ void rowwise_scaling(const IType *__restrict__ sIn_ptr,
                                                fp4e2m1x2 *__restrict__ sOut_ptr,
                                                nvfp4_scale_t *__restrict__ sSFrowwise_ptr,
                                                const float S_enc_rowwise, const int stage_Y,
                                                const int stage_X, const int buff_in,
                                                const int buff_out, RNG_t &rng, uint4 &random_uint4,
                                                int &rnd_idx) {
  using scaling_coeff_type = typename SCALING_COEFFICIENT_TYPE<USE_FAST_MATH>::type;

  const auto &sIn = *reinterpret_cast<const IType3D *>(sIn_ptr);
  auto &sOut = *reinterpret_cast<OType2x3D *>(sOut_ptr);
  auto &sSFrowwise = *reinterpret_cast<ScalesType2D *>(sSFrowwise_ptr);

  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  const int tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const int tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;

  const int thread_offset_Y_rowwise = tid_Y_rowwise;
  const int thread_offset_X_rowwise = tid_X_rowwise * ELTS_PER_THREAD;

  const int SF_thread_offset_rowwise_Y = tid_Y_rowwise;
  const int SF_thread_offset_rowwise_X = tid_X_rowwise / THREADS_PER_SCALE_ROWWISE;

  const bool SF_storing_thread = (tid_X_rowwise % THREADS_PER_SCALE_ROWWISE == 0);

  const int stage_rowwise_scales_offset_Y = SF_thread_offset_rowwise_Y + stage_Y * TILE_DIM_Y;
  const int stage_rowwise_scales_offset_X =
      SF_thread_offset_rowwise_X + stage_X * SCALES_PER_TILE_X;
#pragma unroll
  for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
    const int it_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

    __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];

    // Read (cache) input elements (S2R). Find NVFP4-block AMAX
    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;

      // Load elements
      __uint128_t &elts_8x = *reinterpret_cast<__uint128_t *>(&rIn[w]);
      elts_8x = ptx::ld_shared_b128(&sIn[buff_in][it_offset_Y_rowwise][swizzled_thread_idx]);
#pragma unroll
      for (int e = 0; e < PACK_SIZE / 2; ++e) {
        ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, rIn[w][e]);
      }
    }
    const float block_amax = get_amax_of_pair(thread_amax_2x);

    const nvfp4_scale_t S_dec_b_fp8 = compute_decoding_scaling_factor(block_amax, S_enc_rowwise);
    const scaling_coeff_type SFcoefficient =
        compute_nvfp4_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_rowwise);

    // Store scaling factors to SMEM buffer (R2S)
    if (SF_storing_thread) {
      const int scales_offset_Y = stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE;
      const int scales_offset_X = stage_rowwise_scales_offset_X;
      sSFrowwise[scales_offset_Y][scales_offset_X] = S_dec_b_fp8;
    }

// Scale elements
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const uint64_t elts03 = *reinterpret_cast<uint64_t *>(&rIn[w][0]);
      const uint64_t elts47 = *reinterpret_cast<uint64_t *>(&rIn[w][2]);

      uint32_t out_x8;
      if constexpr (USE_STOCHASTIC_ROUNDING) {
        const uint32_t rbits03 = core::get_rbits(rng, random_uint4, rnd_idx);
        const uint32_t rbits47 = core::get_rbits(rng, random_uint4, rnd_idx);
        out_x8 = ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<scaling_coeff_type>(
            elts03, elts47, SFcoefficient, rbits03, rbits47);
      } else {
        out_x8 = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<scaling_coeff_type>(elts03, elts47,
                                                                                  SFcoefficient);
      }

      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_idx = (swizzled_group_idx + thread_offset_X_rowwise) / 2;
      ptx::st_shared_b32(&sOut[buff_out][it_offset_Y_rowwise][swizzled_idx], out_x8);
    }
  }
}

__device__ __forceinline__ size_t get_current_tensor_id(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t current_offset,
    const size_t block_Y, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t current_row = block_Y * CHUNK_DIM_Y;
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    return current_row / rows_per_tensor;
  }

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

__device__ __forceinline__ size_t get_tensor_rows_num(
    const size_t tensor_id, const ShapeRepresentation shape_rep, const size_t first_logical_dim,
    const int64_t *const __restrict__ first_dims_ptr, const size_t num_tensors) {
  size_t rows_num = 0;
  switch (shape_rep) {
    case ShapeRepresentation::SAME_BOTH_DIMS:
      rows_num = first_logical_dim / num_tensors;
      break;
    case ShapeRepresentation::VARYING_LAST_DIM:
      rows_num = first_logical_dim;
      break;
    case ShapeRepresentation::VARYING_FIRST_DIM:
    case ShapeRepresentation::VARYING_BOTH_DIMS:
      rows_num = static_cast<size_t>(first_dims_ptr[tensor_id]);
      break;
  }
  if (rows_num % 128 != 0) {
    NVTE_DEVICE_ERROR("First dimension of each tensor in a group must be divisible by 128.");
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
      if (cols_num % 128 != 0) {
        NVTE_DEVICE_ERROR(
            "For non-single tensors, the last dimension of each tensor in a group "
            "must be divisible by 128.");
      }
      break;
  }
  return cols_num;
}

__device__ __forceinline__ size_t
get_tensor_base_offset(const size_t tensor_id, const ShapeRepresentation shape_rep,
                       const size_t first_logical_dim, const size_t last_logical_dim,
                       const size_t num_tensors, const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    return tensor_id * rows_per_tensor * last_logical_dim;
  }
  return static_cast<size_t>(offsets_ptr[tensor_id]);
}

__device__ __forceinline__ size_t get_nvfp4_scale_stride(const size_t block_scaled_dim) {
  return DIVUP_TO_MULTIPLE(DIVUP(block_scaled_dim, static_cast<size_t>(SCALE_DIM)),
                           static_cast<size_t>(4));
}

__device__ __forceinline__ size_t get_grouped_scale_base_offset(
    const size_t tensor_id, const ShapeRepresentation shape_rep, const size_t first_logical_dim,
    const size_t last_logical_dim, const size_t num_tensors,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, const bool rowwise) {
  size_t scale_base = 0;
  for (size_t t = 0; t < tensor_id; ++t) {
    const size_t rows =
        get_tensor_rows_num(t, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
    const size_t cols = get_tensor_cols_num(t, shape_rep, last_logical_dim, last_dims_ptr);

    const size_t scale_rows = rowwise ? rows : cols;
    const size_t stride_dim = rowwise ? cols : rows;
    const size_t scale_stride = get_nvfp4_scale_stride(stride_dim);
    scale_base += scale_rows * scale_stride;
  }
  return scale_base;
}

struct JobDescriptor {
  size_t block_id = 0;
  size_t block_global_offset = 0;
  size_t tensor_id = 0;
  size_t rows = 0;
  size_t cols = 0;
};

struct BlockDescriptor {
  size_t tensor_base = 0;
  size_t block_id_Y = 0;
  size_t block_id_X = 0;
  size_t block_offset_Y = 0;
  size_t block_offset_X = 0;
};

__device__ __forceinline__ JobDescriptor decode_job(
    const ShapeRepresentation shape_rep, const bool use_single_work_grid, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim, const size_t work_blocks_X,
    const int32_t ctaid_X, const int32_t ctaid_Y, const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr) {
  JobDescriptor job{};
  job.block_id = static_cast<size_t>(ctaid_Y) * work_blocks_X + static_cast<size_t>(ctaid_X);
  job.block_global_offset = use_single_work_grid
                                ? (ctaid_Y * CHUNK_DIM_Y * last_logical_dim + ctaid_X * CHUNK_DIM_X)
                                : (job.block_id * ELTS_PER_CHUNK);
  job.tensor_id = get_current_tensor_id(shape_rep, num_tensors, job.block_global_offset,
                                        static_cast<size_t>(ctaid_Y), first_logical_dim,
                                        last_logical_dim, offsets_ptr);
  job.rows =
      get_tensor_rows_num(job.tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  job.cols = get_tensor_cols_num(job.tensor_id, shape_rep, last_logical_dim, last_dims_ptr);
  return job;
}

__device__ __forceinline__ bool is_job_valid(const JobDescriptor &job,
                                             const ShapeRepresentation shape_rep,
                                             const size_t total_work_blocks,
                                             const int64_t *const __restrict__ offsets_ptr) {
  bool is_valid = (job.block_id < total_work_blocks) && (job.rows != 0) && (job.cols != 0);
  if (!is_valid || shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    return is_valid;
  }

  const size_t tensor_end_offset = static_cast<size_t>(offsets_ptr[job.tensor_id + 1]);
  if (job.block_global_offset >= tensor_end_offset) {
    return false;
  }

  return true;
}

__device__ __forceinline__ BlockDescriptor decode_block(
    const JobDescriptor &job, const ShapeRepresentation shape_rep, const bool use_single_work_grid,
    const size_t first_logical_dim, const size_t last_logical_dim, const size_t num_tensors,
    const int32_t ctaid_X, const int32_t ctaid_Y, const int64_t *const __restrict__ offsets_ptr) {
  BlockDescriptor block{};
  block.tensor_base = get_tensor_base_offset(job.tensor_id, shape_rep, first_logical_dim,
                                             last_logical_dim, num_tensors, offsets_ptr);

  const size_t blocks_X_num_in_current_tensor = DIVUP(job.cols, static_cast<size_t>(CHUNK_DIM_X));
  if (use_single_work_grid) {
    block.block_id_X = static_cast<size_t>(ctaid_X);
    if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
      const size_t rows_per_tensor = first_logical_dim / num_tensors;
      const size_t blocks_Y_per_tensor = DIVUP(rows_per_tensor, static_cast<size_t>(CHUNK_DIM_Y));
      block.block_id_Y = static_cast<size_t>(ctaid_Y) - job.tensor_id * blocks_Y_per_tensor;
    } else {
      const size_t tensor_base_row = block.tensor_base / job.cols;
      block.block_id_Y = static_cast<size_t>(ctaid_Y) - tensor_base_row / CHUNK_DIM_Y;
    }
  } else {
    const size_t block_id_in_current_tensor = job.block_id - block.tensor_base / ELTS_PER_CHUNK;
    block.block_id_Y = block_id_in_current_tensor / blocks_X_num_in_current_tensor;
    block.block_id_X = block_id_in_current_tensor % blocks_X_num_in_current_tensor;
  }

  block.block_offset_Y = block.block_id_Y * CHUNK_DIM_Y;
  block.block_offset_X = block.block_id_X * CHUNK_DIM_X;
  return block;
}

__device__ __forceinline__ uintptr_t get_pointer_with_offset_bits(const uintptr_t base_ptr,
                                                                  const size_t offset_elts,
                                                                  const size_t data_type_bits) {
  const size_t offset_bits = offset_elts * data_type_bits;
  if (offset_bits % 8 != 0) {
    NVTE_DEVICE_ERROR("Data offset is not byte-aligned.");
  }
  return base_ptr + offset_bits / 8;
}

__device__ __forceinline__ void modify_base_tensor_map(const CUtensorMap base_tensor_map,
                                                       CUtensorMap *global_tensor_map,
                                                       const uintptr_t global_data_ptr,
                                                       const size_t global_dim_Y,
                                                       const size_t global_dim_X,
                                                       const size_t data_type_bits) {
  __shared__ CUtensorMap shared_tensor_map;
  shared_tensor_map = base_tensor_map;
  constexpr bool is_blackwell = ARCH_BLACKWELL_FAMILY;
  if constexpr (is_blackwell) {
    const size_t global_stride_bits = global_dim_X * data_type_bits;
    if (global_stride_bits % 8 != 0) {
      NVTE_DEVICE_ERROR("Shape not supported. Data stride must be byte-aligned.");
    }
    const size_t global_stride_bytes = global_stride_bits / 8;
    if (global_stride_bytes % TMA_GMEM_ALIGNMENT != 0) {
      NVTE_DEVICE_ERROR("Shape not supported. Data stride must be 16B aligned.");
    }
    if (global_data_ptr % TMA_GMEM_ALIGNMENT != 0) {
      NVTE_DEVICE_ERROR("Tensor data pointer must be 16B aligned.");
    }
    asm volatile(
        "{\n\t"
        ".reg.b64 tensor_map_ptr; \n\t"
        "mov.b64 tensor_map_ptr, %0; \n\t"
        "tensormap.replace.tile.global_address.b1024.b64 [tensor_map_ptr], %1; \n\t"
        "tensormap.replace.tile.global_dim.b1024.b32 [tensor_map_ptr], 1, %2; \n\t"
        "tensormap.replace.tile.global_dim.b1024.b32 [tensor_map_ptr], 0, %3; \n\t"
        "tensormap.replace.tile.global_stride.b1024.b64 [tensor_map_ptr], 0, %4; \n\t"
        "}\n"
        :
        : "l"(reinterpret_cast<uintptr_t>(&shared_tensor_map)), "l"(global_data_ptr),
          "r"(static_cast<uint32_t>(global_dim_Y)), "r"(static_cast<uint32_t>(global_dim_X)),
          "l"(static_cast<uint64_t>(global_stride_bytes))
        : "memory");
    *global_tensor_map = shared_tensor_map;
  } else {
    NVTE_DEVICE_ERROR(
        "tensormap.replace is architecture-specific. "
        "Try recompiling with sm_XXXa instead of sm_XXX.");
  }
}

template <typename InType>
__global__ void update_tma_descriptors(
    const __grid_constant__ CUtensorMap base_tensor_map_input,
    const __grid_constant__ CUtensorMap base_tensor_map_output,
    const __grid_constant__ CUtensorMap base_tensor_map_output_t,
    const InType *const __restrict__ input_data_ptr, const void *const output_data_ptr,
    const void *const output_t_data_ptr, const ShapeRepresentation shape_rep,
    const size_t num_tensors, const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr, const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, const bool rowwise, const bool colwise) {
  const bool leading_thread = (threadIdx.x == 0);
  const size_t tensor_id = blockIdx.x;
  if (!leading_thread || tensor_id >= num_tensors) {
    return;
  }

  const size_t rows =
      get_tensor_rows_num(tensor_id, shape_rep, first_logical_dim, first_dims_ptr, num_tensors);
  const size_t cols = get_tensor_cols_num(tensor_id, shape_rep, last_logical_dim, last_dims_ptr);
  const size_t offset_elts = get_tensor_base_offset(tensor_id, shape_rep, first_logical_dim,
                                                    last_logical_dim, num_tensors, offsets_ptr);

  {
    const uintptr_t global_data_ptr = get_pointer_with_offset_bits(
        reinterpret_cast<uintptr_t>(input_data_ptr), offset_elts, TypeInfo<InType>::size);
    modify_base_tensor_map(base_tensor_map_input, &g_tensor_maps_input[tensor_id], global_data_ptr,
                           rows, cols, TypeInfo<InType>::size);
  }

  if (rowwise) {
    const uintptr_t global_data_ptr =
        get_pointer_with_offset_bits(reinterpret_cast<uintptr_t>(output_data_ptr), offset_elts, 4);
    modify_base_tensor_map(base_tensor_map_output, &g_tensor_maps_output[tensor_id],
                           global_data_ptr, rows, cols, 4);
  }

  if (colwise) {
    const uintptr_t global_data_ptr = get_pointer_with_offset_bits(
        reinterpret_cast<uintptr_t>(output_t_data_ptr), offset_elts, 4);
    modify_base_tensor_map(base_tensor_map_output_t, &g_tensor_maps_output_t[tensor_id],
                           global_data_ptr, cols, rows, 4);
  }
}

__device__ __forceinline__ void fence_acquire_tensormap(const CUtensorMap *tensor_map) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.tensormap::generic.acquire.cta [%0], 128;" ::"l"(tensor_map));
#else
  NVTE_DEVICE_ERROR("fence_acquire_tensormap is only supported on SM 9.0+.");
#endif
}

template <bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM) group_quantize_transpose_nvfp4_tuned_1D_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input_static,
    const __grid_constant__ CUtensorMap tensor_map_output_static,
    const __grid_constant__ CUtensorMap tensor_map_output_t_static,
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const int64_t *const __restrict__ offsets_ptr,
    const int64_t *const __restrict__ first_dims_ptr,
    const int64_t *const __restrict__ last_dims_ptr, nvfp4_scale_t *const scales_ptr,
    nvfp4_scale_t *const scales_t_ptr, const float *noop, const float *const amax_rowwise_ptr,
    const float *const amax_colwise_ptr, const size_t work_blocks_X, const size_t work_blocks_Y,
    const size_t *rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  (void)tensor_map_input_static;
  (void)tensor_map_output_static;
  (void)tensor_map_output_t_static;

  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t launch_block_id = static_cast<size_t>(blockIdx.y) * static_cast<size_t>(gridDim.x) +
                                 static_cast<size_t>(blockIdx.x);
  const size_t rng_sequence = threadIdx.x + launch_block_id * THREADS_NUM;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG_t rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0, 0, 0, 0};
  int rnd_idx = 0;

  const bool leading_thread = (threadIdx.x == 0);
  const bool use_single_work_grid = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                     shape_rep == ShapeRepresentation::VARYING_FIRST_DIM);

  constexpr int buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr int buff_elems_total_in = BUFFS_NUM_IN * buff_elems;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out_t =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);

  constexpr int in_mem = buff_size_aligned_in;
  constexpr int out_mem_rowwise_data = buff_size_aligned_out;
  constexpr int out_mem_colwise_data = RETURN_TRANSPOSE ? buff_size_aligned_out_t : 0;
  constexpr int out_mem_rowwise_scales = DIVUP_TO_MULTIPLE(
      CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char *dshmem = common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

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

  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const float S_enc_rowwise =
      (amax_rowwise_ptr == nullptr)
          ? 1.0f
          : core::compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
  const float S_enc_colwise =
      (amax_colwise_ptr == nullptr)
          ? S_enc_rowwise
          : core::compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  const size_t total_work_blocks = work_blocks_X * work_blocks_Y;
  int32_t ctaid_X = static_cast<int32_t>(blockIdx.x);
  int32_t ctaid_Y = static_cast<int32_t>(blockIdx.y);
  size_t static_next_block_id = 0;
  size_t static_block_stride = 0;
  if constexpr (PERSISTENT) {
    if (launch_block_id >= total_work_blocks) {
      return;
    }
    ctaid_X = static_cast<int32_t>(launch_block_id % work_blocks_X);
    ctaid_Y = static_cast<int32_t>(launch_block_id / work_blocks_X);
    static_block_stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(gridDim.y);
    static_next_block_id = launch_block_id + static_block_stride;
  }

  bool job_finished = false;
  int buff_in = 0;
  int buff_out = 0;
  int buff_out_tr = 0;
  int IN_buff_readable_parity[BUFFS_NUM] = {0};
  bool has_prefetched_current_job = true;

  {
    const JobDescriptor first_job = decode_job(
        shape_rep, use_single_work_grid, num_tensors, first_logical_dim, last_logical_dim,
        work_blocks_X, ctaid_X, ctaid_Y, offsets_ptr, first_dims_ptr, last_dims_ptr);
    if (!is_job_valid(first_job, shape_rep, total_work_blocks, offsets_ptr)) {
      return;
    }
    const BlockDescriptor first_block =
        decode_block(first_job, shape_rep, use_single_work_grid, first_logical_dim,
                     last_logical_dim, num_tensors, ctaid_X, ctaid_Y, offsets_ptr);
    const CUtensorMap &tensor_map_input = g_tensor_maps_input[first_job.tensor_id];
    if (leading_thread) {
      fence_acquire_tensormap(&tensor_map_input);
    }
#pragma unroll
    for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
      const int stage_Y = stage / STAGES_X;
      const int stage_X = stage % STAGES_X;
      const int stage_offset_Y = stage_Y * TILE_DIM_Y;
      const int stage_offset_X = stage_X * TILE_DIM_X;
      const int global_offset_Y = static_cast<int>(first_block.block_offset_Y) + stage_offset_Y;
      const int global_offset_X = static_cast<int>(first_block.block_offset_X) + stage_offset_X;
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

  while (!job_finished) {
    const JobDescriptor current_job = decode_job(
        shape_rep, use_single_work_grid, num_tensors, first_logical_dim, last_logical_dim,
        work_blocks_X, ctaid_X, ctaid_Y, offsets_ptr, first_dims_ptr, last_dims_ptr);
    const bool current_job_is_valid =
        is_job_valid(current_job, shape_rep, total_work_blocks, offsets_ptr);
    if (!current_job_is_valid) {
      if (has_prefetched_current_job) {
        ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                         IN_buff_readable_parity[buff_in]);
        IN_buff_readable_parity[buff_in] ^= 1;
        ptx::cp_async_bulk_wait_group_read<PREFETCH_STAGES>();
      }
      break;
    }

    const BlockDescriptor current_block =
        decode_block(current_job, shape_rep, use_single_work_grid, first_logical_dim,
                     last_logical_dim, num_tensors, ctaid_X, ctaid_Y, offsets_ptr);

    const size_t rows = current_job.rows;
    const size_t cols = current_job.cols;
    const size_t block_offset_Y = current_block.block_offset_Y;
    const size_t block_offset_X = current_block.block_offset_X;
    const size_t block_offset_Y_tr = block_offset_X;
    const size_t block_offset_X_tr = block_offset_Y;

    const size_t chunk_rows = rows - block_offset_Y;
    const size_t chunk_cols = cols - block_offset_X;

    const size_t scales_block_offset_Y_rowwise = current_block.block_id_Y * CHUNK_DIM_Y;
    const size_t scales_block_offset_X_rowwise = current_block.block_id_X * SCALES_PER_CHUNK_X;
    const size_t scales_block_offset_Y_tr = current_block.block_id_X * CHUNK_DIM_X;
    const size_t scales_block_offset_X_tr = current_block.block_id_Y * SCALES_PER_CHUNK_Y;

    const size_t scale_stride = get_nvfp4_scale_stride(cols);
    const size_t scale_stride_t = get_nvfp4_scale_stride(rows);

    const size_t rowwise_scale_base = get_grouped_scale_base_offset(
        current_job.tensor_id, shape_rep, first_logical_dim, last_logical_dim, num_tensors,
        first_dims_ptr, last_dims_ptr, true);
    const size_t colwise_scale_base = get_grouped_scale_base_offset(
        current_job.tensor_id, shape_rep, first_logical_dim, last_logical_dim, num_tensors,
        first_dims_ptr, last_dims_ptr, false);
    nvfp4_scale_t *const scales_rowwise = scales_ptr + rowwise_scale_base;
    nvfp4_scale_t *const scales_colwise =
        RETURN_TRANSPOSE ? (scales_t_ptr + colwise_scale_base) : nullptr;

    const CUtensorMap &tensor_map_input = g_tensor_maps_input[current_job.tensor_id];
    const CUtensorMap &tensor_map_output = g_tensor_maps_output[current_job.tensor_id];
    const CUtensorMap &tensor_map_output_t = g_tensor_maps_output_t[current_job.tensor_id];

    if (leading_thread) {
      fence_acquire_tensormap(&tensor_map_input);
      fence_acquire_tensormap(&tensor_map_output);
      if constexpr (RETURN_TRANSPOSE) {
        fence_acquire_tensormap(&tensor_map_output_t);
      }
    }

    bool prefetched_next_job = false;
#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      const int stage_Y = stage / STAGES_X;
      const int stage_X = stage % STAGES_X;
      const int stage_offset_Y = stage_Y * TILE_DIM_Y;
      const int stage_offset_X = stage_X * TILE_DIM_X;

      bool allow_next_job_prefetch = true;
      JobDescriptor prefetch_job = current_job;
      BlockDescriptor prefetch_block = current_block;

      if (stage == STAGES - PREFETCH_STAGES) {
        if constexpr (PERSISTENT) {
          if (static_next_block_id < total_work_blocks) {
            ctaid_X = static_cast<int32_t>(static_next_block_id % work_blocks_X);
            ctaid_Y = static_cast<int32_t>(static_next_block_id / work_blocks_X);
            static_next_block_id += static_block_stride;
          } else {
            ctaid_X = 0;
            ctaid_Y = static_cast<int32_t>(work_blocks_Y);
            allow_next_job_prefetch = false;
          }
        } else {
          ctaid_X = -1;
          ctaid_Y = -1;
        }
        if constexpr (!PERSISTENT) {
          if (ctaid_X == -1 && ctaid_Y == -1) {
            job_finished = true;
          }
        }
      }

      if ((stage >= STAGES - PREFETCH_STAGES) && allow_next_job_prefetch && !job_finished) {
        prefetch_job = decode_job(shape_rep, use_single_work_grid, num_tensors, first_logical_dim,
                                  last_logical_dim, work_blocks_X, ctaid_X, ctaid_Y, offsets_ptr,
                                  first_dims_ptr, last_dims_ptr);
        allow_next_job_prefetch =
            is_job_valid(prefetch_job, shape_rep, total_work_blocks, offsets_ptr);
        if (allow_next_job_prefetch) {
          prefetch_block =
              decode_block(prefetch_job, shape_rep, use_single_work_grid, first_logical_dim,
                           last_logical_dim, num_tensors, ctaid_X, ctaid_Y, offsets_ptr);
        }
      }

      if ((stage < STAGES - PREFETCH_STAGES) || (allow_next_job_prefetch && !job_finished)) {
        const int next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
        const int next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES;
        const int next_prefetch_stage_Y = next_prefetch_stage / STAGES_X;
        const int next_prefetch_stage_X = next_prefetch_stage % STAGES_X;
        const int next_prefetch_stage_offset_Y = next_prefetch_stage_Y * TILE_DIM_Y;
        const int next_prefetch_stage_offset_X = next_prefetch_stage_X * TILE_DIM_X;

        if (stage >= STAGES - PREFETCH_STAGES) {
          prefetched_next_job = true;
        }

        const int global_offset_Y =
            static_cast<int>(prefetch_block.block_offset_Y) + next_prefetch_stage_offset_Y;
        const int global_offset_X =
            static_cast<int>(prefetch_block.block_offset_X) + next_prefetch_stage_offset_X;

        const CUtensorMap &prefetch_tensor_map_input = g_tensor_maps_input[prefetch_job.tensor_id];
        if (leading_thread && stage == STAGES - PREFETCH_STAGES) {
          fence_acquire_tensormap(&prefetch_tensor_map_input);
        }

        if (leading_thread) {
          uint64_t *dst = reinterpret_cast<uint64_t *>(&sIn[next_prefetch_buff]);
          const uint64_t *src = reinterpret_cast<const uint64_t *>(&prefetch_tensor_map_input);
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

      rowwise_scaling<USE_STOCHASTIC_ROUNDING, USE_FAST_MATH>(
          sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc_rowwise, stage_Y, stage_X, buff_in, buff_out,
          rng, random_uint4, rnd_idx);
      if constexpr (RETURN_TRANSPOSE) {
        colwise_scaling<USE_STOCHASTIC_ROUNDING, USE_FAST_MATH>(
            sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc_colwise, stage_Y, stage_X, buff_in,
            buff_out_tr, rng, random_uint4, rnd_idx);
      }

      ptx::fence_proxy_async_shared_cta();
      __syncthreads();

      if (leading_thread) {
        const int global_offset_Y = static_cast<int>(block_offset_Y) + stage_offset_Y;
        const int global_offset_X = static_cast<int>(block_offset_X) + stage_offset_X;
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&sOut[buff_out]));

        if constexpr (RETURN_TRANSPOSE) {
          const int global_offset_Y_tr = static_cast<int>(block_offset_Y_tr) + stage_offset_X;
          const int global_offset_X_tr = static_cast<int>(block_offset_X_tr) + stage_offset_Y;
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
    has_prefetched_current_job = prefetched_next_job;

    {
      using RowwiseScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_X>;
      const int rowwise_count = min(SCALES_PER_CHUNK_X, static_cast<int>(chunk_cols / SCALE_DIM));
      for (size_t row = threadIdx.x; row < CHUNK_DIM_Y; row += THREADS_NUM) {
        const size_t row_global = scales_block_offset_Y_rowwise + row;
        if (row_global < rows) {
          RowwiseScalesVec &scales_vec = *reinterpret_cast<RowwiseScalesVec *>(sSFrowwise[row]);
          const size_t scale_idx_global = row_global * scale_stride + scales_block_offset_X_rowwise;
          scales_vec.store_to_elts(&scales_rowwise[scale_idx_global], 0, rowwise_count);
        }
      }

      if constexpr (RETURN_TRANSPOSE) {
        using ColwiseScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
        const int colwise_count = min(SCALES_PER_CHUNK_Y, static_cast<int>(chunk_rows / SCALE_DIM));
        for (size_t row_tr = threadIdx.x; row_tr < CHUNK_DIM_X; row_tr += THREADS_NUM) {
          const size_t row_tr_global = scales_block_offset_Y_tr + row_tr;
          if (row_tr_global < cols) {
            ColwiseScalesVec &scales_vec =
                *reinterpret_cast<ColwiseScalesVec *>(sSFcolwise[row_tr]);
            const size_t scale_idx_global =
                row_tr_global * scale_stride_t + scales_block_offset_X_tr;
            scales_vec.store_to_elts(&scales_colwise[scale_idx_global], 0, colwise_count);
          }
        }
      }

      if (!job_finished) {
        __syncthreads();
      }
    }
  }

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

#endif  // FP4_TYPE_SUPPORTED
}  // namespace group_quantize_transpose_tuned_kernel

inline void group_quantize_transpose(const GroupedTensor *input, const Tensor *noop,
                                     GroupedTensor *output, const QuantizationConfig *quant_config,
                                     cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace group_quantize_transpose_tuned_kernel;
  using namespace ptx;

  const bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;
  const bool use_fast_math = quant_config ? quant_config->use_fast_math : false;
  const bool return_transpose = output->has_columnwise_data();

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(input->dtype() == DType::kBFloat16,
             "Optimized grouped NVFP4 kernel supports only BF16 input.");
  NVTE_CHECK(output->has_data(), "Grouped NVFP4 output tensor must be allocated.");
  NVTE_CHECK(is_fp4_dtype(output->dtype()), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales, "Output must have scales in compact format.");
  if (return_transpose) {
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Transposed output must have FP4 type.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated.");
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
  if (shape_rep != ShapeRepresentation::VARYING_BOTH_DIMS) {
    NVTE_CHECK(first_logical_dim % 128 == 0,
               "First logical dimension of a grouped tensor must be divisible by 128.");
  }

  size_t work_blocks_X = 0;
  size_t work_blocks_Y = 0;
  if (use_single_work_grid) {
    work_blocks_Y = DIVUP(first_logical_dim, CHUNK_DIM_Y);
    work_blocks_X = DIVUP(last_logical_dim, CHUNK_DIM_X);
  } else {
    work_blocks_Y = 1;
    work_blocks_X = DIVUP(elts_total, ELTS_PER_CHUNK);
  }

  size_t launch_blocks_X = work_blocks_X;
  size_t launch_blocks_Y = work_blocks_Y;
  if constexpr (PERSISTENT) {
    const size_t sm_num = static_cast<size_t>(transformer_engine::cuda::sm_count());
    const size_t static_grid_size = sm_num * STATIC_PERSISTENT_BLOCKS_PER_SM;
    NVTE_CHECK(static_grid_size > 0, "Static persistent grid size must be greater than zero.");
    launch_blocks_X = static_grid_size;
    launch_blocks_Y = 1;
  }

  const dim3 grid(launch_blocks_X, launch_blocks_Y);
  const int block_size = THREADS_NUM;

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);

  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  nvfp4_scale_t *const scales_t_ptr =
      reinterpret_cast<nvfp4_scale_t *>(output->columnwise_scale_inv.dptr);

  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);
  const float *const amax_rowwise_ptr = reinterpret_cast<const float *>(output->amax.dptr);
  const float *const amax_colwise_ptr =
      reinterpret_cast<const float *>(output->columnwise_amax.dptr);

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
  alignas(64) CUtensorMap tensor_map_output{};
  alignas(64) CUtensorMap tensor_map_output_transpose{};

  const size_t dummy_first_logical_dim = 32;
  const size_t dummy_last_logical_dim = 32;
  create_2D_tensor_map(tensor_map_input, input->data, dummy_first_logical_dim,
                       dummy_last_logical_dim, BUFF_DIM_Y, BUFF_DIM_X, dummy_last_logical_dim, 0,
                       sizeof(IType) * 8);
  create_2D_tensor_map(tensor_map_output, output->data, dummy_first_logical_dim,
                       dummy_last_logical_dim, BUFF_DIM_Y, BUFF_DIM_X, dummy_last_logical_dim, 0,
                       4);
  if (return_transpose) {
    create_2D_tensor_map(tensor_map_output_transpose, output->columnwise_data,
                         dummy_last_logical_dim, dummy_first_logical_dim, BUFF_DIM_X, BUFF_DIM_Y,
                         dummy_first_logical_dim, 0, 4);
  }

  constexpr int buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr int buff_elems_total_in = BUFFS_NUM_IN * buff_elems;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out_t =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_scales = DIVUP_TO_MULTIPLE(
      CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_scales_transpose = DIVUP_TO_MULTIPLE(
      CHUNK_DIM_X * SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

  const int in_mem = buff_size_aligned_in;
  const int out_data_mem = buff_size_aligned_out;
  const int out_data_transpose_mem = return_transpose ? buff_size_aligned_out_t : 0;
  const int out_scales_mem = buff_size_scales;
  const int out_scales_transpose_mem = return_transpose ? buff_size_scales_transpose : 0;
  const int out_mem = out_data_mem + out_data_transpose_mem;
  const int dshmem_size =
      in_mem + out_mem + out_scales_transpose_mem + out_scales_mem + TMA_SHMEM_ALIGNMENT;

  const IType *const input_dptr = reinterpret_cast<const IType *>(input->data.dptr);
  const void *const output_dptr = output->data.dptr;
  const void *const output_t_dptr = return_transpose ? output->columnwise_data.dptr : nullptr;

  update_tma_descriptors<IType><<<num_tensors, 32, 0, stream>>>(
      tensor_map_input, tensor_map_output, tensor_map_output_transpose, input_dptr, output_dptr,
      output_t_dptr, shape_rep, num_tensors, first_logical_dim, last_logical_dim, offsets_ptr,
      first_dims_ptr, last_dims_ptr, true, return_transpose);
  NVTE_CHECK_CUDA(cudaGetLastError());

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          use_fast_math, USE_FAST_MATH,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE, {
            auto kernel =
                group_quantize_transpose_nvfp4_tuned_1D_kernel<USE_STOCHASTIC_ROUNDING,
                                                               USE_FAST_MATH, RETURN_TRANSPOSE>;

            NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size));
            kernel<<<grid, block_size, dshmem_size, stream>>>(
                tensor_map_input, tensor_map_output, tensor_map_output_transpose, shape_rep,
                num_tensors, first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr,
                last_dims_ptr, scales_ptr, scales_t_ptr, noop_ptr, amax_rowwise_ptr,
                amax_colwise_ptr, work_blocks_X, work_blocks_Y, rng_state);
            NVTE_CHECK_CUDA(cudaGetLastError());
          });););
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_
