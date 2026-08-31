/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file scaling_nvfp4_tuned_1D.cuh
 *  \brief Common scaling functions for tuned NVFP4 transpose kernels.
 */

#ifndef TRANSFORMER_ENGINE_SCALING_NVFP4_TUNED_1D_CUH_
#define TRANSFORMER_ENGINE_SCALING_NVFP4_TUNED_1D_CUH_

#include "../core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace tuned_1D_scaling_common {

#if FP4_TYPE_SUPPORTED

enum class KernelType : int {
  NON_GROUPED = 0,
  GROUPED = 1
};

template <KernelType kernel_type>
struct KernelTraits {
  static constexpr int CHUNK_DIM_Y = 128;
  static constexpr int CHUNK_DIM_X = (kernel_type == KernelType::GROUPED) ? 256 : 128;
  static constexpr int PREFETCH_STAGES = 1;

  static constexpr int THREADS_NUM = 128;
  static constexpr int ELTS_PER_THREAD = 16;
  static constexpr int TILE_DIM_Y = 64;
  static constexpr int TILE_DIM_X = 64;

  static_assert(ELTS_PER_THREAD == NVFP4_SCALE_DIM, "Hardcoded and fixed parameter\0");
  static_assert(THREADS_NUM * ELTS_PER_THREAD <= TILE_DIM_Y * TILE_DIM_X,
                "Unbalanced threads workload\0");
  static_assert(CHUNK_DIM_Y % TILE_DIM_Y == 0,
                "Chunk size Y must be evenly divisible by the tile size Y\0");
  static_assert(CHUNK_DIM_X % TILE_DIM_X == 0,
                "Chunk size X must be evenly divisible by the tile size X\0");
  static_assert(TILE_DIM_Y % NVFP4_SCALE_DIM == 0,
                "Tile size Y must be evenly divisible by the scale dim\0");
  static_assert(TILE_DIM_X % NVFP4_SCALE_DIM == 0,
                "Tile size X must be evenly divisible by the scale dim\0");

  static constexpr int TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;
  static constexpr int TILES_X = CHUNK_DIM_X / TILE_DIM_X;
  static constexpr int THREADS_PER_SCALE_ROWWISE = NVFP4_SCALE_DIM / ELTS_PER_THREAD;

  static constexpr int SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / NVFP4_SCALE_DIM;
  static constexpr int SCALES_PER_CHUNK_X = CHUNK_DIM_X / NVFP4_SCALE_DIM;
  static constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / NVFP4_SCALE_DIM;
  static constexpr int SCALES_PER_TILE_X = TILE_DIM_X / NVFP4_SCALE_DIM;

  static constexpr int STAGES_Y = TILES_Y;
  static constexpr int STAGES_X = TILES_X;
  static constexpr int STAGES = STAGES_Y * STAGES_X;

  static constexpr int BUFFS_NUM = PREFETCH_STAGES + 1;
  static constexpr int BUFFS_NUM_IN = BUFFS_NUM;
  static constexpr int BUFFS_NUM_OUT = BUFFS_NUM;
  static constexpr int BUFFS_NUM_OUT_TR = 2;
  static constexpr int BUFF_DIM_Y = TILE_DIM_Y;
  static constexpr int BUFF_DIM_X = TILE_DIM_X;
  static constexpr int BUFF_SIZE = BUFF_DIM_Y * BUFF_DIM_X;
  static constexpr int BUFF_SIZE_TOTAL = BUFF_SIZE * BUFFS_NUM;

  static constexpr int BUFF_IN_DIM_Y = BUFF_DIM_Y;
  static constexpr int BUFF_IN_DIM_X = BUFF_DIM_X;
  static constexpr int BUFF_IN_SIZE = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;
  static constexpr int BUFF_IN_ELTS_NUM = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;

  static constexpr int BUFF_OUT_DIM_Y = BUFF_DIM_Y;
  static constexpr int BUFF_OUT_DIM_X = (BUFF_DIM_X * 4) / 8;
  static constexpr int BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;

  static constexpr int BUFF_OUT_TR_DIM_Y = BUFF_DIM_X;
  static constexpr int BUFF_OUT_TR_DIM_X = (BUFF_DIM_Y * 4) / 8;
  static constexpr int BUFF_OUT_TR_SIZE = BUFF_OUT_TR_DIM_Y * BUFF_OUT_TR_DIM_X;

  using IType = bf16;
  using IType2 = typename ptx::FPx2<IType>;

  static constexpr int BUFF_ELEMS = BUFF_DIM_Y * BUFF_IN_DIM_X;
  static constexpr int BUFF_ELEMS_TOTAL_IN = BUFFS_NUM_IN * BUFF_ELEMS;
  static constexpr int BUFF_SIZE_ALIGNED_IN = DIVUP_TO_MULTIPLE(BUFF_ELEMS_TOTAL_IN * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  static constexpr int BUFF_SIZE_ALIGNED_OUT = DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  static constexpr int BUFF_SIZE_ALIGNED_OUT_TR = DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
  static constexpr int BUFF_SIZE_ROWWISE_SCALES = DIVUP_TO_MULTIPLE(CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
  static constexpr int BUFF_SIZE_COLWISE_SCALES = DIVUP_TO_MULTIPLE(CHUNK_DIM_X * SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

  static constexpr int PACK_SIZE = 8;
  static constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;

  static constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;
  static constexpr int THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;

  static constexpr int THREADS_X_TR = TILE_DIM_X / 2;
  static constexpr int THREADS_Y_TR = THREADS_NUM / THREADS_X_TR;

  static constexpr int ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;
  static constexpr int ITERATIONS_TR = SCALES_PER_TILE_Y / THREADS_Y_TR;
  static_assert(ITERATIONS_TR >= 1, "Number of transpose iterations should be >=1\0");
  static_assert(SCALES_PER_TILE_Y % THREADS_Y_TR == 0,
                "Partial transpose iterations are not supported\0");

  static constexpr int BUFF_OUT_IT_OFFSET = BUFF_OUT_TR_DIM_X / ITERATIONS_TR / STAGES;

  static_assert(BUFF_DIM_Y >= NVFP4_SCALE_DIM,
                "Number of buffer rows must be greater or equal to the size of the columwise "
                "scaling block\0");
  static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y);
  static_assert(BUFF_DIM_Y >= THREADS_Y_ROWWISE,
                "Number of buffer rows must be greater or equal to the number of rowwise "
                "processing threads in Y dimension\0");

  // Number of 4-bit elements that span 32 banks (4-byte each) of shared memory.
  static constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;  // 256

  // Number of threads (rowwise scaling) that span 32 banks (4-byte banks) of shared memory.
  static constexpr int THREADS_PER_BANK = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;

  static constexpr size_t ELTS_PER_CHUNK = static_cast<size_t>(CHUNK_DIM_Y) * static_cast<size_t>(CHUNK_DIM_X);

  using IType3D = IType[BUFFS_NUM_IN][BUFF_IN_DIM_Y][BUFF_IN_DIM_X];
  using IType2x3D = IType2[BUFFS_NUM_IN][BUFF_IN_DIM_Y][BUFF_IN_DIM_X / 2];
  using OType2x3D = fp4e2m1x2[BUFFS_NUM_OUT][BUFF_OUT_DIM_Y][BUFF_OUT_DIM_X];
  using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_TR_DIM_Y][BUFF_OUT_TR_DIM_X];
  using ScalesType2D = nvfp4_scale_t[CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
  using ScalesTypeTr2D = nvfp4_scale_t[CHUNK_DIM_X][SCALES_PER_CHUNK_Y];
};

using NonGroupedKernelTraits = KernelTraits<KernelType::NON_GROUPED>;
using GroupedKernelTraits = KernelTraits<KernelType::GROUPED>;

template <bool USE_FAST_MATH>
struct ScalingCoefficientType {};
template <>
struct ScalingCoefficientType<false> {
  using type = float;
};
template <>
struct ScalingCoefficientType<true> {
  using type = bf16;
};

template <typename PairType>
__device__ __forceinline__ float get_amax_of_pair(const PairType pair) {
  return static_cast<float>(__hmax(__habs(pair.x), __habs(pair.y)));
}

template <typename Traits, bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH, typename RngType>
__device__ __forceinline__ void colwise_scaling(
    const typename Traits::IType *__restrict__ sIn_ptr, fp4e2m1x2 *__restrict__ sOut_tr_ptr,
    nvfp4_scale_t *__restrict__ sSFcolwise_ptr, const float S_enc_colwise, const int stage_Y,
    const int stage_X, const int buff_in, const int buff_out_tr, RngType &rng,
    uint4 &random_uint4, int &rnd_idx) {
  using IType = typename Traits::IType;
  using IType2 = typename Traits::IType2;
  using IType2x3D = typename Traits::IType2x3D;
  using OType2xt3D = typename Traits::OType2xt3D;
  using ScalesTypeTr2D = typename Traits::ScalesTypeTr2D;
  using scaling_coeff_type = typename ScalingCoefficientType<USE_FAST_MATH>::type;

  constexpr int TILE_DIM_X = Traits::TILE_DIM_X;
  constexpr int SCALES_PER_TILE_Y = Traits::SCALES_PER_TILE_Y;

  const auto &sIn2x = *reinterpret_cast<const IType2x3D *>(sIn_ptr);
  auto &sOut_tr = *reinterpret_cast<OType2xt3D *>(sOut_tr_ptr);
  auto &sSFcolwise = *reinterpret_cast<ScalesTypeTr2D *>(sSFcolwise_ptr);

  const int warp = threadIdx.x / THREADS_PER_WARP;
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;

  const int tid_Y_colwise = (thread_lane / 2 + warp) % SCALES_PER_TILE_Y;
  const int tid_X_colwise = thread_lane;

  const int thread_offset_Y_colwise = tid_Y_colwise * NVFP4_SCALE_DIM;
  const int thread_offset_X_colwise = tid_X_colwise * 2;

  const int in_thread_offset_Y = thread_offset_Y_colwise;
  const int in_thread_offset_X = thread_offset_X_colwise / 2;

  const int out_tr_thread_offset_Y = thread_offset_X_colwise;
  const int out_tr_thread_offset_X = thread_offset_Y_colwise / 2;

  const int scale_tr_offset_Y = (stage_X * TILE_DIM_X) + 2 * tid_X_colwise;
  const int scale_tr_offset_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y_colwise;

  __align__(8) IType rIn[2][NVFP4_SCALE_DIM];
  // Read (cache) a pair of input elements (S2R). Find NVFP4-block AMAX.
  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
  for (int i = 0; i < NVFP4_SCALE_DIM; ++i) {
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
    const nvfp4_scale_t S_dec_b_fp8 =
        quantization_and_transposition_SF::compute_decoding_scaling_factor(block_amax[w],
                                                                           S_enc_colwise);

    // Store scaling factors to SMEM buffer (R2S).
    sSFcolwise[scale_tr_offset_Y + w][scale_tr_offset_X] = S_dec_b_fp8;

    const scaling_coeff_type SFcoefficient =
        core::compute_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_colwise);

    // Scale elements.
    __align__(8) uint32_t rOut[NVFP4_SCALE_DIM / 8];
    #pragma unroll
    for (int e = 0; e < NVFP4_SCALE_DIM / 8; ++e) {
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

template <typename Traits, bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH, bool ROW_SCALED_NVFP4,
          typename RngType>
__device__ __forceinline__ void rowwise_scaling(
    const typename Traits::IType *__restrict__ sIn_ptr, fp4e2m1x2 *__restrict__ sOut_ptr,
    nvfp4_scale_t *__restrict__ sSFrowwise_ptr, const float S_enc_rowwise, const int stage_Y,
    const int stage_X, const int buff_in, const int buff_out, const float *amax_rowwise_ptr,
    const size_t row_offset, const size_t rows, RngType &rng, uint4 &random_uint4, int &rnd_idx) {
  using IType = typename Traits::IType;
  using IType2 = typename Traits::IType2;
  using IType3D = typename Traits::IType3D;
  using OType2x3D = typename Traits::OType2x3D;
  using ScalesType2D = typename Traits::ScalesType2D;
  using scaling_coeff_type = typename ScalingCoefficientType<USE_FAST_MATH>::type;

  constexpr int TILE_DIM_Y = Traits::TILE_DIM_Y;
  constexpr int ELTS_PER_THREAD = Traits::ELTS_PER_THREAD;
  constexpr int PACK_SIZE = Traits::PACK_SIZE;
  constexpr int WAVES = Traits::WAVES;
  constexpr int THREADS_PER_BANK = Traits::THREADS_PER_BANK;
  constexpr int THREADS_X_ROWWISE = Traits::THREADS_X_ROWWISE;
  constexpr int THREADS_Y_ROWWISE = Traits::THREADS_Y_ROWWISE;
  constexpr int THREADS_PER_SCALE_ROWWISE = Traits::THREADS_PER_SCALE_ROWWISE;
  constexpr int SCALES_PER_TILE_X = Traits::SCALES_PER_TILE_X;
  constexpr int ITERATIONS_NORMAL = Traits::ITERATIONS_NORMAL;

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
  const int stage_rowwise_scales_offset_X = SF_thread_offset_rowwise_X + stage_X * SCALES_PER_TILE_X;
#pragma unroll
  for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
    const int it_offset_Y_rowwise = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;

    __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];

    // Read (cache) input elements (S2R). Find NVFP4-block AMAX.
    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;

      __uint128_t &elts_8x = *reinterpret_cast<__uint128_t *>(&rIn[w]);
      elts_8x = ptx::ld_shared_b128(&sIn[buff_in][it_offset_Y_rowwise][swizzled_thread_idx]);
      #pragma unroll
      for (int e = 0; e < PACK_SIZE / 2; ++e) {
        ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, rIn[w][e]);
      }
    }
    const float block_amax = get_amax_of_pair(thread_amax_2x);

    nvfp4_scale_t S_dec_b_fp8;
    scaling_coeff_type SFcoefficient;
    if constexpr (ROW_SCALED_NVFP4) {
      const size_t row_idx = row_offset + stage_Y * TILE_DIM_Y + it_offset_Y_rowwise;
      const float S_enc_rowwise_block = row_idx < rows
                                        ? core::compute_global_encode_scaling_factor_FP4(amax_rowwise_ptr[row_idx])
                                        : 1.0f;
      S_dec_b_fp8 = quantization_and_transposition_SF::compute_decoding_scaling_factor(block_amax, S_enc_rowwise_block);
      SFcoefficient = core::compute_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_rowwise_block);
    } else {
      S_dec_b_fp8 = quantization_and_transposition_SF::compute_decoding_scaling_factor(block_amax, S_enc_rowwise);
      SFcoefficient = core::compute_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_rowwise);
    }

    // Store scaling factors to SMEM buffer (R2S).
    if (SF_storing_thread) {
      const int scales_offset_Y = stage_rowwise_scales_offset_Y + it * THREADS_Y_ROWWISE;
      const int scales_offset_X = stage_rowwise_scales_offset_X;
      sSFrowwise[scales_offset_Y][scales_offset_X] = S_dec_b_fp8;
    }

// Scale elements.
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

template <typename Traits, bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_MATH, typename RngType>
__device__ __forceinline__ void rowwise_scaling(
    const typename Traits::IType *__restrict__ sIn_ptr, fp4e2m1x2 *__restrict__ sOut_ptr,
    nvfp4_scale_t *__restrict__ sSFrowwise_ptr, const float S_enc_rowwise, const int stage_Y,
    const int stage_X, const int buff_in, const int buff_out, RngType &rng,
    uint4 &random_uint4, int &rnd_idx) {
  rowwise_scaling<Traits, USE_STOCHASTIC_ROUNDING, USE_FAST_MATH, false>(
      sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc_rowwise, stage_Y, stage_X, buff_in, buff_out,
      nullptr, 0, 0, rng, random_uint4, rnd_idx);
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace tuned_1D_scaling_common
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_SCALING_NVFP4_TUNED_1D_CUH_
