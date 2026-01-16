/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_transpose_nvfp4_tuned_1D.cuh
 *  \brief Tuned kernel to cast to NVFP4 and transpose.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../../common.h"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"
#include "../../../utils.cuh"
#include "../core_nvfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace quantize_transpose_tuned_kernel {

using namespace quantization_and_transposition_SF;
using namespace core;
using namespace ptx;

#if FP4_TYPE_SUPPORTED

struct TunableConfig {
  static constexpr int CHUNK_DIM_Y = 128;
  static constexpr int CHUNK_DIM_X = 128;
  static constexpr int PREFETCH_STAGES = 1;
  static constexpr bool PERSISTENT = false;
};

constexpr int SCALE_DIM = 16;  // NVFP4 block (x16 elts)
constexpr int THREADS_NUM = 128;
constexpr int ELTS_PER_THREAD = 16;
constexpr int TILE_DIM_Y = 64;
constexpr int TILE_DIM_X = 64;

static_assert(ELTS_PER_THREAD == SCALE_DIM && "Hardcoded and fixed parameter\0");

static_assert((THREADS_NUM * ELTS_PER_THREAD <= TILE_DIM_Y * TILE_DIM_X) &&
              "Unbalanced threads workload\0");

static_assert((TunableConfig::CHUNK_DIM_Y % TILE_DIM_Y == 0) &&
              "Chunk size Y must be evenly divisible by the tile size Y\0");
static_assert((TunableConfig::CHUNK_DIM_X % TILE_DIM_X == 0) &&
              "Chunk size X must be evenly divisible by the tile size X\0");

static_assert((TILE_DIM_Y % SCALE_DIM == 0) &&
              "Tile size Y must be evenly divisible by the scale dim\0");
static_assert((TILE_DIM_X % SCALE_DIM == 0) &&
              "Tile size X must be evenly divisible by the scale dim\0");

constexpr int TILES_Y = TunableConfig::CHUNK_DIM_Y / TILE_DIM_Y;
constexpr int TILES_X = TunableConfig::CHUNK_DIM_X / TILE_DIM_X;

constexpr int THREADS_PER_SCALE_ROWWISE = SCALE_DIM / ELTS_PER_THREAD;

constexpr int SCALES_PER_CHUNK_Y = TunableConfig::CHUNK_DIM_Y / SCALE_DIM;
constexpr int SCALES_PER_CHUNK_X = TunableConfig::CHUNK_DIM_X / SCALE_DIM;

constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;
constexpr int SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;

constexpr int STAGES_Y = TILES_Y;
constexpr int STAGES_X = TILES_X;
constexpr int STAGES = STAGES_Y * STAGES_X;

constexpr int BUFFS_NUM = TunableConfig::PREFETCH_STAGES + 1;
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
constexpr int BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr int BUFF_OUT_T_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr int BUFF_OUT_T_SIZE = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;

// Manual swizzling parameters to reduce SHMEM bank conflicts
constexpr int PACK_SIZE = 8;
constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;

constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;
constexpr int THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;

constexpr int THREADS_X_TRANSP = TILE_DIM_X / 2;
constexpr int THREADS_Y_TRANSP = THREADS_NUM / THREADS_X_TRANSP;

constexpr int ITERATIONS_NORMAL = BUFF_DIM_Y / THREADS_Y_ROWWISE;
constexpr int ITERATIONS_TRANSPOSE = SCALES_PER_TILE_Y / THREADS_Y_TRANSP;
static_assert(ITERATIONS_TRANSPOSE >= 1 && "Number of transpose iterations should be >=1\0");
static_assert((SCALES_PER_TILE_Y % THREADS_Y_TRANSP == 0) &&
              "Partial transpose iterations are not supported\0");

constexpr int BUFF_OUT_IT_OFFSET = BUFF_OUT_T_DIM_X / ITERATIONS_TRANSPOSE / STAGES;

static_assert(BUFF_DIM_Y >= SCALE_DIM &&
              "Number of buffer rows must be greater or equal to the size of the columwise "
              "scaling block\0");
static_assert(TunableConfig::CHUNK_DIM_Y >= BUFF_DIM_Y);
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
using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_T_DIM_Y][BUFF_OUT_T_DIM_X];
using ScalesType2D = nvfp4_scale_t[TunableConfig::CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
using ScalesTypeTr2D = nvfp4_scale_t[TunableConfig::CHUNK_DIM_X][SCALES_PER_CHUNK_Y];
using RNG_t = typename transformer_engine::curanddx::detail::philox4x32_native_state<10>;

template <bool USE_FAST_NVFP4_SCALING>
struct SCALING_COEFFICIENT_TYPE {};
template<> struct SCALING_COEFFICIENT_TYPE<false> { using type = float; };
template<> struct SCALING_COEFFICIENT_TYPE<true>  { using type = bf16; };

__device__ __forceinline__ float get_amax_of_pair(const IType2 pair) {
  return static_cast<float>(__hmax(__habs(pair.x), __habs(pair.y)));
}

// Compute "correct" per-block encoding scaling factor
template <typename SF_TYPE>
__device__ __forceinline__ SF_TYPE
compute_nvfp4_scaling_coefficient(const nvfp4_scale_t S_dec_block, const float S_enc) {
  constexpr float float_max = detail::TypeExtrema<SF_TYPE>::max;
  const float scale_rcp = fminf(S_enc / static_cast<float>(S_dec_block), float_max);
  return static_cast<SF_TYPE>(scale_rcp);
}

template <bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_NVFP4_SCALING>
__device__ __forceinline__ void colwise_scaling(const IType *__restrict__ sIn_ptr,
                                                fp4e2m1x2 *__restrict__ sOut_tr_ptr,
                                                nvfp4_scale_t *__restrict__ sSFcolwise_ptr,
                                                const float S_enc_colwise, const int stage_Y,
                                                const int stage_X, const int buff_in,
                                                const int buff_out_tr, RNG_t &rng,
                                                uint4 &random_uint4, int &rnd_idx) {
  using scaling_coeff_type = typename SCALING_COEFFICIENT_TYPE<USE_FAST_NVFP4_SCALING>::type;

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

    const scaling_coeff_type SFcoefficient = compute_nvfp4_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_colwise);

    // Scale elements
    __align__(8) uint32_t rOut[SCALE_DIM / 8];
#pragma unroll
    for (int e = 0; e < SCALE_DIM / 8; ++e) {
      const uint64_t elts03 = *reinterpret_cast<uint64_t *>(&rIn[w][8 * e]);
      const uint64_t elts47 = *reinterpret_cast<uint64_t *>(&rIn[w][8 * e + 4]);
      if constexpr (USE_STOCHASTIC_ROUNDING) {
        const uint32_t rbits03 = core::get_rbits(rng, random_uint4, rnd_idx);
        const uint32_t rbits47 = core::get_rbits(rng, random_uint4, rnd_idx);
        rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<scaling_coeff_type>
                      (elts03, elts47, SFcoefficient, rbits03, rbits47);
      } else {
        rOut[e] = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<scaling_coeff_type>
                      (elts03, elts47, SFcoefficient);
      }
    }
    uint64_t &out_pack_16x = *reinterpret_cast<uint64_t *>(rOut);
    ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_thread_offset_Y + w][out_tr_thread_offset_X],
                       out_pack_16x);
  }
}

template <bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_NVFP4_SCALING>
__device__ __forceinline__ void rowwise_scaling(const IType *__restrict__ sIn_ptr,
                                                fp4e2m1x2 *__restrict__ sOut_ptr,
                                                nvfp4_scale_t *__restrict__ sSFrowwise_ptr,
                                                const float S_enc_rowwise, const int stage_Y,
                                                const int stage_X, const int buff_in,
                                                const int buff_out, RNG_t &rng, uint4 &random_uint4,
                                                int &rnd_idx) {
  using scaling_coeff_type = typename SCALING_COEFFICIENT_TYPE<USE_FAST_NVFP4_SCALING>::type;

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
    const scaling_coeff_type SFcoefficient = compute_nvfp4_scaling_coefficient<scaling_coeff_type>(S_dec_b_fp8, S_enc_rowwise);

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
        out_x8 = ptx::mul_cvt_bf16_to_fp4_8x_stochastic_rounding<scaling_coeff_type>
                    (elts03, elts47, SFcoefficient, rbits03, rbits47);
      } else {
        out_x8 = ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<scaling_coeff_type>
                    (elts03, elts47, SFcoefficient);
      }

      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_idx = (swizzled_group_idx + thread_offset_X_rowwise) / 2;
      ptx::st_shared_b32(&sOut[buff_out][it_offset_Y_rowwise][swizzled_idx], out_x8);
    }
  }
}

template <bool USE_STOCHASTIC_ROUNDING, bool USE_FAST_NVFP4_SCALING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM) quantize_transpose_nvfp4_tuned_1D_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    const __grid_constant__ CUtensorMap tensor_map_output_t, nvfp4_scale_t *const scales_ptr,
    nvfp4_scale_t *const scales_t_ptr, const float *noop, const float *const amax_rowwise_ptr,
    const float *const amax_colwise_ptr, const size_t rows, const size_t cols,
    const size_t scale_stride, const size_t scale_stride_t, const size_t *rng_state) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM + blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed = rng_state != nullptr ? rng_state[0] : 0;
  const size_t rng_offset = rng_state != nullptr ? rng_state[1] : 0;
  RNG_t rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0, 0, 0, 0};
  // Index of the random number. It increments each time when used and resets to 0 if reaches 4x
  int rnd_idx = 0;

  const bool leading_thread = (threadIdx.x == 0);

  constexpr int buff_elems = BUFF_DIM_Y * BUFF_IN_DIM_X;
  constexpr int buff_elems_total_in = BUFFS_NUM_IN * buff_elems;

  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out_t =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_T_SIZE, TMA_SHMEM_ALIGNMENT);

  constexpr int in_mem = buff_size_aligned_in;

  constexpr int out_mem_rowwise_data = buff_size_aligned_out;
  constexpr int out_mem_colwise_data = RETURN_TRANSPOSE ? buff_size_aligned_out_t : 0;
  constexpr int out_mem_rowwise_scales = DIVUP_TO_MULTIPLE(
      TunableConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
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

  // Compute a global encoding/decoding scaling factors for all S_dec_b
  const float S_enc_rowwise =
      (amax_rowwise_ptr == nullptr)
          ? 1.0f
          : core::compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);

  const float S_enc_colwise =
      (amax_colwise_ptr == nullptr)
          ? S_enc_rowwise
          : core::compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);

  __shared__ uint64_t workID_mbar;
  __shared__ __uint128_t workID_response;
  constexpr uint32_t workID_response_size = sizeof(workID_response);
  static_assert(workID_response_size == 16);

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];

  // Coordinates of the first chunk (CTA) to process
  int32_t ctaid_X = blockIdx.x;
  int32_t ctaid_Y = blockIdx.y;

  // Initialize shared memory barriers with the number of threads participating in them
  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::mbarrier_init(&workID_mbar, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  bool job_finished = false;
  int buff_in = 0;
  int buff_out = 0;
  int buff_out_tr = 0;
  int IN_buff_readable_parity[BUFFS_NUM] = {0, 0};
  int ctaid_parity = 0;

// Prefetch input data only when processing the first chunk,
// which enables the one-iteration overlap throughout the entire kernel life
#pragma unroll
  for (int stage = 0; stage < TunableConfig::PREFETCH_STAGES; ++stage) {
    const int buff_in = stage;
    const int stage_Y = stage / STAGES_X;
    const int stage_X = stage % STAGES_X;

    const int stage_offset_Y = stage_Y * TILE_DIM_Y;
    const int stage_offset_X = stage_X * TILE_DIM_X;

    const int block_offset_Y = ctaid_Y * TunableConfig::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * TunableConfig::CHUNK_DIM_X;

    const int global_offset_Y = block_offset_Y + stage_offset_Y;
    const int global_offset_X = block_offset_X + stage_offset_X;

    uint64_t *barrier = &IN_buff_readable_mbar[buff_in];
    if (leading_thread) {
      uint64_t *dst = reinterpret_cast<uint64_t *>(&sIn[buff_in]);
      const uint64_t *src = reinterpret_cast<const uint64_t *>(&tensor_map_input);

      // Arrive on the barrier and tell how many bytes are expected to come in
      ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);

      // Initiate bulk tensor copy
      ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y,
                                                    barrier);
    }
  }

  while (!job_finished) {
    const int block_offset_Y = ctaid_Y * TunableConfig::CHUNK_DIM_Y;
    const int block_offset_X = ctaid_X * TunableConfig::CHUNK_DIM_X;

    const int block_offset_Y_tr = ctaid_X * TunableConfig::CHUNK_DIM_X;
    const int block_offset_X_tr = ctaid_Y * TunableConfig::CHUNK_DIM_Y;

    const int chunk_rows = rows - block_offset_Y;
    const int chunk_cols = cols - block_offset_X;

    const int scales_block_offset_Y_rowwise = ctaid_Y * TunableConfig::CHUNK_DIM_Y;
    const int scales_block_offset_X_rowwise = ctaid_X * SCALES_PER_CHUNK_X;
    const int scales_block_offset_Y_tr = ctaid_X * TunableConfig::CHUNK_DIM_X;
    const int scales_block_offset_X_tr = ctaid_Y * SCALES_PER_CHUNK_Y;

    if constexpr (TunableConfig::PERSISTENT) {
      if (leading_thread) {
        ptx::mbarrier_arrive_expect_tx_cta_relaxed_shared_cta(&workID_mbar, workID_response_size);
        ptx::try_cancel_cta(&workID_mbar, &workID_response);
      }
    }

#pragma unroll
    for (int stage = 0; stage < STAGES; ++stage) {
      const int stage_Y = stage / STAGES_X;
      const int stage_X = stage % STAGES_X;

      const int stage_offset_Y = stage_Y * TILE_DIM_Y;
      const int stage_offset_X = stage_X * TILE_DIM_X;

      if (stage == STAGES - TunableConfig::PREFETCH_STAGES) {
        if constexpr (TunableConfig::PERSISTENT) {
          ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&workID_mbar, ctaid_parity);
          ptx::get_cancelled_cta_id_2D(&workID_response, ctaid_X, ctaid_Y);
          ctaid_parity ^= 1;
        } else {
          ctaid_X = -1;
          ctaid_Y = -1;
        }
        if (ctaid_X == -1 && ctaid_Y == -1) {
          job_finished = true;
        }
      }

      // Prefetch next stage Input data
      if (!job_finished || (stage < STAGES - TunableConfig::PREFETCH_STAGES)) {
        const int next_prefetch_buff = (buff_in + TunableConfig::PREFETCH_STAGES) % BUFFS_NUM;
        const int next_prefetch_stage = (stage + TunableConfig::PREFETCH_STAGES) % STAGES;
        const int next_prefetch_stage_Y = next_prefetch_stage / STAGES_X;
        const int next_prefetch_stage_X = next_prefetch_stage % STAGES_X;

        const int next_prefetch_stage_offset_Y = next_prefetch_stage_Y * TILE_DIM_Y;
        const int next_prefetch_stage_offset_X = next_prefetch_stage_X * TILE_DIM_X;

        // Offsets change, because coordinates of the next "to-be-prefetched" CTA do also chage
        const int block_offset_Y = ctaid_Y * TunableConfig::CHUNK_DIM_Y;
        const int block_offset_X = ctaid_X * TunableConfig::CHUNK_DIM_X;

        const int global_offset_Y = block_offset_Y + next_prefetch_stage_offset_Y;
        const int global_offset_X = block_offset_X + next_prefetch_stage_offset_X;

        uint64_t *barrier = &IN_buff_readable_mbar[next_prefetch_buff];
        if (leading_thread) {
          uint64_t *dst = reinterpret_cast<uint64_t *>(&sIn[next_prefetch_buff]);
          const uint64_t *src = reinterpret_cast<const uint64_t *>(&tensor_map_input);

          // Arrive on the barrier and tell how many bytes are expected to come in
          ptx::mbarrier_arrive_expect_tx(barrier, shmem_buff_size);

          // Initiate bulk tensor copy
          ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y,
                                                        barrier);
        }
        ptx::fence_proxy_async_shared_cta();
      }

      // Wait for the data to have arrived
      ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                       IN_buff_readable_parity[buff_in]);
      IN_buff_readable_parity[buff_in] ^= 1;

      // Wait for TMA transfer to have finished reading shared memory
      // I.e. the OUT buffer is ready to be written to
      ptx::cp_async_bulk_wait_group_read<TunableConfig::PREFETCH_STAGES>();

      // NVFP4 Quantization
      rowwise_scaling<USE_STOCHASTIC_ROUNDING, USE_FAST_NVFP4_SCALING>
        (sIn_ptr, sOut_ptr, sSFrowwise_ptr, S_enc_rowwise, stage_Y, stage_X,
         buff_in, buff_out, rng, random_uint4, rnd_idx);

      if constexpr (RETURN_TRANSPOSE) {
        colwise_scaling<USE_STOCHASTIC_ROUNDING, USE_FAST_NVFP4_SCALING>
        (sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, S_enc_colwise, stage_Y, stage_X,
          buff_in, buff_out_tr, rng, random_uint4, rnd_idx);
      }

      // Wait for shared memory writes to be visible to TMA engine
      ptx::fence_proxy_async_shared_cta();
      __syncthreads();
      // After syncthreads, writes by all threads are visible to TMA engine

      // Initiate TMA transfer to copy shared memory to global memory
      if (leading_thread) {
        const int global_offset_Y = block_offset_Y + stage_offset_Y;
        const int global_offset_X = block_offset_X + stage_offset_X;
        const int global_offset_Y_tr = block_offset_Y_tr + stage_offset_X;
        const int global_offset_X_tr = block_offset_X_tr + stage_offset_Y;

        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t *>(&tensor_map_output), global_offset_X,
            global_offset_Y, reinterpret_cast<uint64_t *>(&sOut[buff_out]));

        if constexpr (RETURN_TRANSPOSE) {
          ptx::cp_async_bulk_tensor_2d_shared_to_global(
              reinterpret_cast<const uint64_t *>(&tensor_map_output_t), global_offset_X_tr,
              global_offset_Y_tr, reinterpret_cast<uint64_t *>(&sOut_tr[buff_out_tr]));
        }

        // Create a "bulk async-group" out of the previous bulk copy operation
        ptx::cp_async_bulk_commit_group();
      }

      buff_in = (buff_in + 1) % BUFFS_NUM_IN;
      buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
      buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
    }  // end of stages

    // Vectorized store of scaling factors (S2G)
    {
      // Rowwise
      {
        using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_X>;
        // number of scales in X dimension of this chunk
        const int count = min(SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);

        for (size_t row = threadIdx.x; row < TunableConfig::CHUNK_DIM_Y; row += THREADS_NUM) {
          const size_t row_global = scales_block_offset_Y_rowwise + row;
          if (row_global < rows) {
            ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(sSFrowwise[row]);
            const size_t scale_idx_global =
                row_global * scale_stride + scales_block_offset_X_rowwise;
            scales_vec.store_to_elts(&scales_ptr[scale_idx_global], 0, count);
          }
        }
      }

      // Colwise
      if constexpr (RETURN_TRANSPOSE) {
        using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
        // number of scales in Y dimension of this chunk
        const int count = min(SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);

        for (size_t row_tr = threadIdx.x; row_tr < TunableConfig::CHUNK_DIM_X;
             row_tr += THREADS_NUM) {
          const size_t row_tr_global = scales_block_offset_Y_tr + row_tr;
          if (row_tr_global < cols) {
            ScalesVec &scales_vec = *reinterpret_cast<ScalesVec *>(sSFcolwise[row_tr]);
            const size_t scale_idx_global =
                row_tr_global * scale_stride_t + scales_block_offset_X_tr;
            scales_vec.store_to_elts(&scales_t_ptr[scale_idx_global], 0, count);
          }
        }
      }

      if (!job_finished) {
        // Ensures all reads from SFs buffer have completed and it's ready to be reused
        __syncthreads();
      }
    }
  }

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
    ptx::mbarrier_invalid(&workID_mbar);
  }
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

#endif  // FP4_TYPE_SUPPORTED
}  // namespace quantize_transpose_tuned_kernel

inline void quantize_transpose_tuned_1D(const Tensor &input, const Tensor *noop, Tensor *output,
                                        const QuantizationConfig *quant_config,
                                        cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace quantize_transpose_tuned_kernel;
  using namespace ptx;

  const bool use_stochastic_rounding = quant_config ? quant_config->stochastic_rounding : false;
  const bool use_fast_nvfp4_scaling = quant_config ? quant_config->fast_nvfp4_scaling : false;

  // If transposed output is allocated, return the transposed data
  // Otherwise, it's not necesary to return the transposed data.
  const bool return_transpose = output->has_columnwise_data();

  checkCuDriverContext(stream);
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output", false);

  NVTE_CHECK(input.has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(output->has_data(), "NVFP4 output tensor must be allocated.");
  NVTE_CHECK(is_fp4_dtype(output->data.dtype), "Output must have FP4 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  if (return_transpose) {
    NVTE_CHECK(is_fp4_dtype(output->columnwise_data.dtype),
               "Transposed output must have FP4 type.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Transposed scaling tensor must be allocated");
  }

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();

  NVTE_CHECK(rows % 32 == 0,
             "Number of tensor rows must be a multiple of 32");  // 16B alignment for TMA
  NVTE_CHECK(cols % 32 == 0,
             "Number of tensor cols must be a multiple of 32");  // 16B alignment for TMA

  const int blocks_Y = DIVUP(rows, static_cast<size_t>(TunableConfig::CHUNK_DIM_Y));
  const int blocks_X = DIVUP(cols, static_cast<size_t>(TunableConfig::CHUNK_DIM_X));
  const dim3 grid(blocks_X, blocks_Y);
  const int block_size = THREADS_NUM;

  const size_t scale_stride = output->scale_inv.shape[1];
  const size_t scale_stride_transpose =
      return_transpose ? output->columnwise_scale_inv.shape[1] : 0;

  nvfp4_scale_t *const scales_ptr = reinterpret_cast<nvfp4_scale_t *>(output->scale_inv.dptr);
  nvfp4_scale_t *const scales_transpose_ptr =
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

  create_2D_tensor_map(tensor_map_input, input.data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       sizeof(IType) * 8);

  create_2D_tensor_map(tensor_map_output, output->data, rows, cols, BUFF_DIM_Y, BUFF_DIM_X, cols, 0,
                       4);
  if (return_transpose) {
    create_2D_tensor_map(tensor_map_output_transpose, output->columnwise_data, cols, rows,
                         BUFF_DIM_X, BUFF_DIM_Y, rows, 0, 4);
  }

  constexpr int buff_elems = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr int buff_elems_total_in = BUFFS_NUM_IN * buff_elems;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out_t =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_T_SIZE, TMA_SHMEM_ALIGNMENT);

  constexpr int buff_size_scales = DIVUP_TO_MULTIPLE(
      TunableConfig::CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_scales_transpose = DIVUP_TO_MULTIPLE(
      TunableConfig::CHUNK_DIM_X * SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

  const int in_mem = buff_size_aligned_in;

  const int out_data_mem = buff_size_aligned_out;
  const int out_data_transpose_mem = return_transpose ? buff_size_aligned_out_t : 0;
  const int out_scales_mem = buff_size_scales;
  const int out_scales_transpose_mem = return_transpose ? buff_size_scales_transpose : 0;

  const int out_mem = out_data_mem + out_data_transpose_mem;

  const int dshmem_size =
      in_mem + out_mem + out_scales_transpose_mem + out_scales_mem + TMA_SHMEM_ALIGNMENT;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(use_stochastic_rounding, USE_STOCHASTIC_ROUNDING,
    TRANSFORMER_ENGINE_SWITCH_CONDITION(use_fast_nvfp4_scaling, USE_FAST_NVFP4_SCALING,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(return_transpose, RETURN_TRANSPOSE,
      {
        auto kernel = quantize_transpose_nvfp4_tuned_1D_kernel
                        <USE_STOCHASTIC_ROUNDING, USE_FAST_NVFP4_SCALING, RETURN_TRANSPOSE>;

        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
        kernel<<<grid, block_size, dshmem_size, stream>>>(
            tensor_map_input, tensor_map_output, tensor_map_output_transpose, scales_ptr,
            scales_transpose_ptr, noop_ptr, amax_rowwise_ptr, amax_colwise_ptr, rows, cols,
            scale_stride, scale_stride_transpose, rng_state);
      }
      ); 
    );
  );
#else
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif  // FP4_TYPE_SUPPORTED
}

}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_TRANSPOSE_NVFP4_TUNED_1D_CUH_
