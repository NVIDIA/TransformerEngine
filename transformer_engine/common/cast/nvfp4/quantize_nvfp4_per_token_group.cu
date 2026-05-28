/*************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_nvfp4_per_token_group.cu
 *  \brief Grouped NVFP4 per-token cast: bf16 input (sum_M, K), splits along
 *         M; K1 fused row+col amax + K2 row + col cast. Requires K % 128 == 0
 *         and every split_sections[i] % 128 == 0.
 */

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <transformer_engine/nvfp4_per_token.h>

#include <cstring>
#include <vector>

#include "common/cast/core/common.cuh"
#include "common/cast/nvfp4/core_nvfp4.cuh"
#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace nvfp4_per_token_group {

#if FP4_TYPE_SUPPORTED

using dispatch::nvfp4::nvfp4_scale_t;
using dispatch::nvfp4::core::compute_global_encode_scaling_factor_FP4;
using dispatch::nvfp4::quantization_SF::compute_decoding_scaling_factor;
using ptx::FPx2;

constexpr int kInnerK = 16;  // NVFP4 inner block: 16 elements per e4m3 SF

// 64-tensor cap so the args struct fits under the 4 KB launch-param limit.
constexpr int kMaxTensorsPerKernel = 64;

// Per-launch arg table; passed as __grid_constant__ for constant-cache reads.
struct NVFP4PerTokenMultiArgs {
  // K1 outputs (per-tensor pointers; one fp32 array per tensor)
  void* row_amax_list[kMaxTensorsPerKernel];  // each: float* (M_i,)
  void* col_amax_list[kMaxTensorsPerKernel];  // each: float* (K,)

  // K2 outputs (per-tensor pointers; FP4 codes + e4m3 inner SF)
  void* q_row_list[kMaxTensorsPerKernel];      // each: uint8* (M_i, K/2)
  void* s_dec_row_list[kMaxTensorsPerKernel];  // each: fp8e4m3* (M_i, K/16)
  void* q_col_list[kMaxTensorsPerKernel];      // each: uint8* (K, M_i/2)
  void* s_dec_col_list[kMaxTensorsPerKernel];  // each: fp8e4m3* (K, M_i/16)

  // Shared layout info
  int split_sections_range[kMaxTensorsPerKernel + 1];  // prefix sum w/ leading 0
  int num_tensors;
};

__device__ __forceinline__ int GetTensorId(const NVFP4PerTokenMultiArgs& args, int global_row) {
  const int n = args.num_tensors;
  if (global_row >= args.split_sections_range[n]) return n - 1;
  int tid = 0;
  while (args.split_sections_range[tid + 1] <= global_row) ++tid;
  return tid;
}

// Fused K1: TMA-loaded SMEM tile feeds row+col amax; routes atomicMax to the
// per-tensor buffer via tensor_id lookup at CTA entry.
namespace fused {

constexpr int CHUNK_DIM_Y = 128;  // CTA covers this many rows
constexpr int CHUNK_DIM_X = 128;  // CTA covers this many cols
constexpr int TILE_DIM_Y = 64;    // TMA bulk-2D box height
constexpr int TILE_DIM_X = 64;    // TMA bulk-2D box width
constexpr int THREADS_NUM = 128;
constexpr int PREFETCH_STAGES = 1;
constexpr int BUFFS_NUM = PREFETCH_STAGES + 1;
constexpr int TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;  // 2
constexpr int TILES_X = CHUNK_DIM_X / TILE_DIM_X;  // 2
constexpr int STAGES = TILES_Y * TILES_X;          // 4

constexpr int BUFF_IN_DIM_Y = TILE_DIM_Y;
constexpr int BUFF_IN_DIM_X = TILE_DIM_X;
constexpr int BUFF_IN_SIZE = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;

using FusedIType = bf16;
using FusedIType2 = ptx::FPx2<FusedIType>;
using FusedIType3D = FusedIType[BUFFS_NUM][BUFF_IN_DIM_Y][BUFF_IN_DIM_X];

// Randomized Hadamard Transform helpers (per-thread, 16-wide). Direct copy
// of the single-tensor helpers in quantize_nvfp4_per_token.cu; K1 and K2
// must consume identical output for FP4 + outer SF to be self-consistent.
// TODO: hoist into a shared core header.
__device__ __forceinline__ void apply_signed_fht16_inplace(float r[16], uint32_t random_sign_mask) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint32_t bits = __float_as_uint(r[i]);
    const uint32_t flip = ((random_sign_mask >> i) & 1u) << 31;
    r[i] = __uint_as_float(bits ^ flip);
  }
#pragma unroll
  for (int stride = 1; stride < 16; stride <<= 1) {
#pragma unroll
    for (int g = 0; g < 16; g += stride << 1) {
#pragma unroll
      for (int j = 0; j < stride; ++j) {
        const float a = r[g + j];
        const float b = r[g + j + stride];
        r[g + j] = a + b;
        r[g + j + stride] = a - b;
      }
    }
  }
}

__device__ __forceinline__ float amax_16_abs(const float r[16]) {
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(r[i]));
  return m;
}

// 1/sqrt(16) Hadamard normalization, folded once per 1x16 block.
constexpr float k16HadamardNorm = 0.25f;

// Pre-zero amax buffers (identity for atomicMax).
template <bool DO_ROW, bool DO_COL>
__global__ void group_per_token_fused_zero_amax_kernel(NVFP4PerTokenMultiArgs args, int K) {
  const int tensor_id = blockIdx.x;
  if (tensor_id >= args.num_tensors) return;
  if (DO_ROW) {
    float* row_amax = reinterpret_cast<float*>(args.row_amax_list[tensor_id]);
    if (row_amax != nullptr) {
      const int M_i =
          args.split_sections_range[tensor_id + 1] - args.split_sections_range[tensor_id];
      for (int m = threadIdx.x; m < M_i; m += blockDim.x) {
        row_amax[m] = 0.0f;
      }
    }
  }
  if (DO_COL) {
    float* col_amax = reinterpret_cast<float*>(args.col_amax_list[tensor_id]);
    if (col_amax != nullptr) {
      for (int k = threadIdx.x; k < K; k += blockDim.x) {
        col_amax[k] = 0.0f;
      }
    }
  }
}

// kWithRht=true: col-wise amax over RHT-rotated 16-row strips. Row direction
// never sees RHT.
template <bool DO_ROW, bool DO_COL, bool kWithRht>
__global__ void __launch_bounds__(THREADS_NUM)
    group_per_token_fused_amax_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                                      const __grid_constant__ NVFP4PerTokenMultiArgs args,
                                      const float* noop, const size_t rows, const size_t cols,
                                      const uint32_t random_sign_mask_t) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const bool leading_thread = (threadIdx.x == 0);
  const int tid = threadIdx.x;

  constexpr int buff_elems_total_in = BUFFS_NUM * BUFF_IN_SIZE;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(FusedIType), TMA_SHMEM_ALIGNMENT);

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char* dshmem = dispatch::common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);
  FusedIType* sIn_ptr = reinterpret_cast<FusedIType*>(dshmem);
  auto& sIn = *reinterpret_cast<FusedIType3D*>(sIn_ptr);

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];
  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const int32_t ctaid_X = blockIdx.x;
  const int32_t ctaid_Y = blockIdx.y;
  const int block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
  const int block_offset_X = ctaid_X * CHUNK_DIM_X;

  // Tile lies fully inside one tensor (split_sections[i] % 128 == 0).
  const int tensor_id = GetTensorId(args, block_offset_Y);
  const int local_row_base = block_offset_Y - args.split_sections_range[tensor_id];
  float* row_amax_out = DO_ROW ? reinterpret_cast<float*>(args.row_amax_list[tensor_id]) : nullptr;
  float* col_amax_out = DO_COL ? reinterpret_cast<float*>(args.col_amax_list[tensor_id]) : nullptr;

  // Each thread owns chunk-row `tid` (for row amax) and chunk-col `tid` (for col amax).
  float row_partial = 0.f;
  float col_partial = 0.f;
  const int my_row_stage_Y = tid / TILE_DIM_Y;
  const int my_col_stage_X = tid / TILE_DIM_X;
  const int my_row_in_subtile = tid % TILE_DIM_Y;
  const int my_col_in_subtile = tid % TILE_DIM_X;

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Prefetch stage 0.
#pragma unroll
  for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
    const int buff_in = stage;
    const int stage_Y = stage / TILES_X;
    const int stage_X = stage % TILES_X;
    const int global_offset_Y = block_offset_Y + stage_Y * TILE_DIM_Y;
    const int global_offset_X = block_offset_X + stage_X * TILE_DIM_X;
    if (leading_thread) {
      ptx::mbarrier_arrive_expect_tx(&IN_buff_readable_mbar[buff_in], shmem_buff_size);
      ptx::cp_async_bulk_tensor_2d_global_to_shared(
          reinterpret_cast<uint64_t*>(&sIn[buff_in]),
          reinterpret_cast<const uint64_t*>(&tensor_map_input), global_offset_X, global_offset_Y,
          &IN_buff_readable_mbar[buff_in]);
    }
  }

  int buff_in = 0;
  int IN_buff_readable_parity[BUFFS_NUM] = {0, 0};

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const int stage_Y = stage / TILES_X;
    const int stage_X = stage % TILES_X;

    // Prefetch next stage.
    if (stage < STAGES - PREFETCH_STAGES) {
      const int next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
      const int next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES;
      const int next_stage_Y = next_prefetch_stage / TILES_X;
      const int next_stage_X = next_prefetch_stage % TILES_X;
      const int next_global_offset_Y = block_offset_Y + next_stage_Y * TILE_DIM_Y;
      const int next_global_offset_X = block_offset_X + next_stage_X * TILE_DIM_X;
      if (leading_thread) {
        ptx::mbarrier_arrive_expect_tx(&IN_buff_readable_mbar[next_prefetch_buff], shmem_buff_size);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(&sIn[next_prefetch_buff]),
            reinterpret_cast<const uint64_t*>(&tensor_map_input), next_global_offset_X,
            next_global_offset_Y, &IN_buff_readable_mbar[next_prefetch_buff]);
      }
      ptx::fence_proxy_async_shared_cta();
    }

    // Wait for this stage's tile.
    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                     IN_buff_readable_parity[buff_in]);
    IN_buff_readable_parity[buff_in] ^= 1;

    // Row partial: rotate e-iter by bank group to split warp into 8 groups.
    if (DO_ROW && stage_Y == my_row_stage_Y) {
      float local_max = row_partial;
      const int row_bank_group = (my_row_in_subtile >> 2) & 0x7;
#pragma unroll
      for (int e_iter = 0; e_iter < 8; ++e_iter) {
        const int e = ((e_iter + row_bank_group) & 0x7) << 3;
        __uint128_t elts_8x = ptx::ld_shared_b128(&sIn[buff_in][my_row_in_subtile][e]);
        const FusedIType2* pairs = reinterpret_cast<const FusedIType2*>(&elts_8x);
        FusedIType2 amax_2x = {static_cast<FusedIType>(0.0f), static_cast<FusedIType>(0.0f)};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          ptx::abs_max_2x(amax_2x, amax_2x, pairs[p]);
        }
        local_max =
            fmaxf(local_max, static_cast<float>(__hmax(__habs(amax_2x.x), __habs(amax_2x.y))));
      }
      row_partial = local_max;
    }

    // Col partial: 1 thread per column scans down 64 rows of the sub-tile.
    if (DO_COL && stage_X == my_col_stage_X) {
      if constexpr (kWithRht) {
        // 4 contiguous 16-row blocks per sub-tile, one FHT per block; 0.25
        // is folded post-amax (exact, since 0.25 = 2^-2).
#pragma unroll
        for (int blk = 0; blk < TILE_DIM_Y / 16; ++blk) {
          float r[16];
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            r[i] = static_cast<float>(sIn[buff_in][blk * 16 + i][my_col_in_subtile]);
          }
          apply_signed_fht16_inplace(r, random_sign_mask_t);
          col_partial = fmaxf(col_partial, amax_16_abs(r) * k16HadamardNorm);
        }
      } else {
        float local_max = col_partial;
#pragma unroll
        for (int e = 0; e < TILE_DIM_Y; ++e) {
          const FusedIType v = sIn[buff_in][e][my_col_in_subtile];
          local_max = fmaxf(local_max, fabsf(static_cast<float>(v)));
        }
        col_partial = local_max;
      }
    }

    __syncthreads();
    buff_in = (buff_in + 1) % BUFFS_NUM;
  }

  // CTAs across (ctaid_X) share row_amax slots; across (ctaid_Y) share col_amax slots.
  if (DO_ROW) {
    atomicMaxFloat(&row_amax_out[local_row_base + tid], row_partial);
  }
  if (DO_COL) {
    atomicMaxFloat(&col_amax_out[block_offset_X + tid], col_partial);
  }

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
  }
#else
  (void)tensor_map_input;
  (void)args;
  (void)noop;
  (void)rows;
  (void)cols;
  (void)random_sign_mask_t;
  NVTE_DEVICE_ERROR("Fused grouped per-token amax kernel requires SM 10.0+ (Blackwell).");
#endif  // __CUDA_ARCH__ >= 1000
}

// K2 (encode) constants + helpers; byte-equal port of the single-tensor
// per-token cooperative 4x32 / 32x4 threading + ld_shared_b128 + mul_cvt_4x.
constexpr int ELTS_PER_THREAD = 16;                          // = NVFP4 block size = SCALE_DIM
constexpr int SCALE_DIM = 16;                                // NVFP4 inner block (1x16)
constexpr int SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;  // 8
constexpr int SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;  // 8
constexpr int SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;    // 4
constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;    // 4

// Rowwise pass: 4 (K-dim) x 32 (M-dim) -> 1 NVFP4 block per thread.
constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;         // 4
constexpr int THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;      // 32
constexpr int THREADS_PER_SCALE_ROWWISE = SCALE_DIM / ELTS_PER_THREAD;  // 1
constexpr int ITERATIONS_NORMAL = TILE_DIM_Y / THREADS_Y_ROWWISE;       // 2

// Output / SF SMEM buffer dims (sub-tile sized, double-buffered for ping-pong).
constexpr int BUFF_OUT_DIM_Y = TILE_DIM_Y;
constexpr int BUFF_OUT_DIM_X = (TILE_DIM_X * 4) / 8;  // 32 (fp4e2m1x2 bytes)
constexpr int BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;
constexpr int BUFF_OUT_TR_DIM_Y = TILE_DIM_X;
constexpr int BUFF_OUT_TR_DIM_X = (TILE_DIM_Y * 4) / 8;  // 32
constexpr int BUFF_OUT_TR_SIZE = BUFF_OUT_TR_DIM_Y * BUFF_OUT_TR_DIM_X;
constexpr int BUFFS_NUM_OUT = BUFFS_NUM;  // 2
constexpr int BUFFS_NUM_OUT_TR = 2;

// Manual SMEM swizzling parameters (matches single-tensor encode kernel).
constexpr int PACK_SIZE = 8;
constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;                     // 2
constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;                    // 256
constexpr int THREADS_PER_BANK = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;  // 16

using IType = FusedIType;
using IType2 = FusedIType2;
using IType2x3D = IType2[BUFFS_NUM][BUFF_IN_DIM_Y][BUFF_IN_DIM_X / 2];
using OType2x3D = fp4e2m1x2[BUFFS_NUM_OUT][BUFF_OUT_DIM_Y][BUFF_OUT_DIM_X];
using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_TR_DIM_Y][BUFF_OUT_TR_DIM_X];
using ScalesType2D = nvfp4_scale_t[CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
using ScalesTypeTr2D = nvfp4_scale_t[CHUNK_DIM_X][SCALES_PER_CHUNK_Y];

// Rowwise encode helper: reads sRowAmax (pre-populated by K1), writes FP4 +
// e4m3 SFs into sOut / sSFrowwise. Byte-equal to the single-tensor version.
__device__ __forceinline__ void rowwise_scaling_per_token(
    const IType* __restrict__ sIn_ptr, fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr, const float* __restrict__ sRowAmax,
    const int stage_Y, const int stage_X, const int buff_in, const int buff_out) {
  const auto& sIn = *reinterpret_cast<const FusedIType3D*>(sIn_ptr);
  auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
  auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);

  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  const int tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;  // 0..31
  const int tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;  // 0..3

  const int thread_offset_X_rowwise = tid_X_rowwise * ELTS_PER_THREAD;

  const int SF_thread_offset_rowwise_X = tid_X_rowwise / THREADS_PER_SCALE_ROWWISE;
  const bool SF_storing_thread = (tid_X_rowwise % THREADS_PER_SCALE_ROWWISE == 0);

  const int stage_rowwise_scales_offset_X =
      SF_thread_offset_rowwise_X + stage_X * SCALES_PER_TILE_X;

#pragma unroll
  for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
    const int it_offset_Y_rowwise = tid_Y_rowwise + it * THREADS_Y_ROWWISE;
    const int chunk_local_row = stage_Y * TILE_DIM_Y + it_offset_Y_rowwise;

    const float row_amax = sRowAmax[chunk_local_row];
    const float S_enc = compute_global_encode_scaling_factor_FP4(fmaxf(row_amax, 1e-12f));

    __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];

    IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;

      __uint128_t& elts_8x = *reinterpret_cast<__uint128_t*>(&rIn[w]);
      elts_8x = ptx::ld_shared_b128(&sIn[buff_in][it_offset_Y_rowwise][swizzled_thread_idx]);
#pragma unroll
      for (int e = 0; e < PACK_SIZE / 2; ++e) {
        ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, rIn[w][e]);
      }
    }
    const float block_amax =
        static_cast<float>(__hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));

    const fp8e4m3 s_dec = compute_decoding_scaling_factor(block_amax, S_enc);
    const float s_dec_f = static_cast<float>(s_dec);
    const float block_scale = (s_dec_f == 0.f) ? 0.f : __fdiv_rn(S_enc, s_dec_f);

    if (SF_storing_thread) {
      const int scales_offset_Y = chunk_local_row;
      const int scales_offset_X = stage_rowwise_scales_offset_X;
      sSFrowwise[scales_offset_Y][scales_offset_X] = s_dec;
    }

#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_idx = (swizzled_group_idx + thread_offset_X_rowwise) / 2;

      fp4e2m1x4 qu0{}, qu1{};
      ptx::mul_cvt_4x(qu0, rIn[w][0], rIn[w][1], block_scale);
      ptx::mul_cvt_4x(qu1, rIn[w][2], rIn[w][3], block_scale);

      uint32_t out_x8 = (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&qu0))) |
                        (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&qu1)) << 16);
      ptx::st_shared_b32(&sOut[buff_out][it_offset_Y_rowwise][swizzled_idx], out_x8);
    }
  }
}

// Colwise encode helper. kWithRht=true rotates each thread's 16-row strip
// via the FHT before block_amax + cast; K1 amax must have used the same
// mask so the per-col outer amax matches.
template <bool kWithRht = false>
__device__ __forceinline__ void colwise_scaling_per_token(
    const IType* __restrict__ sIn_ptr, fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr, const float* __restrict__ sColAmax,
    const int stage_Y, const int stage_X, const int buff_in, const int buff_out_tr,
    const uint32_t random_sign_mask_t = 0u) {
  const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
  auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
  auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

  const int warp = threadIdx.x / THREADS_PER_WARP;  // 0..3
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;

  const int tid_Y_colwise = (thread_lane % 4 + warp) % 4;  // 0..3
  const int tid_X_colwise = thread_lane;                   // 0..31

  const int thread_offset_Y_colwise = tid_Y_colwise * SCALE_DIM;
  const int thread_offset_X_colwise = tid_X_colwise * 2;

  const int in_thread_offset_Y = thread_offset_Y_colwise;
  const int in_thread_offset_X = thread_offset_X_colwise / 2;

  const int out_tr_thread_offset_Y = thread_offset_X_colwise;
  const int out_tr_thread_offset_X = thread_offset_Y_colwise / 2;

  const int scale_tr_offset_Y = (stage_X * TILE_DIM_X) + 2 * tid_X_colwise;
  const int scale_tr_offset_X = (stage_Y * SCALES_PER_TILE_Y) + tid_Y_colwise;

  __align__(8) IType rIn[2][SCALE_DIM];
  // RHT staging in fp32 (DCE'd in the non-RHT instantiation).
  float rRht[2][SCALE_DIM];

  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
  for (int i = 0; i < SCALE_DIM; ++i) {
    const IType2 elt_pair =
        ptx::ld_shared_b32(&sIn2x[buff_in][in_thread_offset_Y + i][in_thread_offset_X]);
    rIn[0][i] = elt_pair.x;
    rIn[1][i] = elt_pair.y;
    if constexpr (!kWithRht) {
      ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, elt_pair);
    }
  }

  float block_amax[2];
  if constexpr (kWithRht) {
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
      for (int i = 0; i < SCALE_DIM; ++i) {
        rRht[w][i] = static_cast<float>(rIn[w][i]);
      }
      apply_signed_fht16_inplace(rRht[w], random_sign_mask_t);
      float local_max = 0.f;
#pragma unroll
      for (int i = 0; i < SCALE_DIM; ++i) {
        local_max = fmaxf(local_max, fabsf(rRht[w][i]));
      }
      // amax(|r * 0.25|) == amax(|r|) * 0.25; 0.25 also folded into
      // block_scale_rht below (bit-exact: 0.25 = 2^-2).
      block_amax[w] = local_max * k16HadamardNorm;
    }
  } else {
    block_amax[0] = static_cast<float>(__habs(thread_amax_2x.x));
    block_amax[1] = static_cast<float>(__habs(thread_amax_2x.y));
  }

#pragma unroll
  for (int w = 0; w < 2; ++w) {
    const int chunk_local_col = scale_tr_offset_Y + w;
    const float col_amax = sColAmax[chunk_local_col];
    const float S_enc_col = compute_global_encode_scaling_factor_FP4(fmaxf(col_amax, 1e-12f));

    const fp8e4m3 s_dec = compute_decoding_scaling_factor(block_amax[w], S_enc_col);
    const float s_dec_f = static_cast<float>(s_dec);
    const float block_scale = (s_dec_f == 0.f) ? 0.f : __fdiv_rn(S_enc_col, s_dec_f);

    sSFcolwise[scale_tr_offset_Y + w][scale_tr_offset_X] = s_dec;

    fp4e2m1x4 qu[4];
    if constexpr (kWithRht) {
      // ptx::floatx2 keeps mul_cvt_4x's input fp32 (no bf16 round-trip).
      const float block_scale_rht = block_scale * k16HadamardNorm;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const ptx::floatx2 in01{rRht[w][4 * e + 0], rRht[w][4 * e + 1]};
        const ptx::floatx2 in23{rRht[w][4 * e + 2], rRht[w][4 * e + 3]};
        ptx::mul_cvt_4x(qu[e], in01, in23, block_scale_rht);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        IType2 in01{rIn[w][4 * e + 0], rIn[w][4 * e + 1]};
        IType2 in23{rIn[w][4 * e + 2], rIn[w][4 * e + 3]};
        ptx::mul_cvt_4x(qu[e], in01, in23, block_scale);
      }
    }

    uint64_t out_pack_16x = (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[0])) << 0) |
                            (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[1])) << 16) |
                            (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[2])) << 32) |
                            (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[3])) << 48);
    ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_thread_offset_Y + w][out_tr_thread_offset_X],
                       out_pack_16x);
  }
}

// Fused K2: TMA-loads input, runs cooperative row+col encode helpers, scatters
// FP4 + SFs to per-tensor outputs via st.global (multi-dest, no TMA store).
// kWithRht=true (and DO_COL=true): col-wise FHT with random_sign_mask_t,
// matching the K1 amax launch. Row direction never sees RHT.
template <bool DO_ROW, bool DO_COL, bool kWithRht>
__global__ void __launch_bounds__(THREADS_NUM)
    group_per_token_fused_cast_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                                      const __grid_constant__ NVFP4PerTokenMultiArgs args,
                                      const float* noop, const size_t rows, const size_t cols,
                                      const uint32_t random_sign_mask_t) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }
  (void)rows;

  const bool leading_thread = (threadIdx.x == 0);

  // Dynamic SMEM layout (~28 KiB): sIn (16K) + sOut (4K) + sOut_tr (4K) +
  // sSF_row (1K) + sSF_col (1K) + sRowAmax/sColAmax (512B each).
  constexpr int buff_elems_total_in = BUFFS_NUM * BUFF_IN_SIZE;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out_t =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int out_mem_rowwise_data = DO_ROW ? buff_size_aligned_out : 0;
  constexpr int out_mem_colwise_data = DO_COL ? buff_size_aligned_out_t : 0;
  constexpr int out_mem_rowwise_scales =
      DO_ROW ? DIVUP_TO_MULTIPLE(CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t),
                                 TMA_SHMEM_ALIGNMENT)
             : 0;
  constexpr int out_mem_colwise_scales =
      DO_COL ? DIVUP_TO_MULTIPLE(CHUNK_DIM_X * SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t),
                                 TMA_SHMEM_ALIGNMENT)
             : 0;
  (void)out_mem_colwise_scales;

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char* dshmem = dispatch::common::align_smem_ptr_per_TMA_requirements(dynamic_shmem);

  IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
  fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + buff_size_aligned_in);
  fp4e2m1x2* sOut_tr_ptr =
      reinterpret_cast<fp4e2m1x2*>(dshmem + buff_size_aligned_in + out_mem_rowwise_data);
  nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(
      dshmem + buff_size_aligned_in + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t* sSFcolwise_ptr =
      reinterpret_cast<nvfp4_scale_t*>(dshmem + buff_size_aligned_in + out_mem_rowwise_data +
                                       out_mem_colwise_data + out_mem_rowwise_scales);

  __shared__ float sRowAmax[CHUNK_DIM_Y];
  __shared__ float sColAmax[CHUNK_DIM_X];
  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];

  auto& sIn = *reinterpret_cast<FusedIType3D*>(sIn_ptr);

  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const int32_t ctaid_X = blockIdx.x;
  const int32_t ctaid_Y = blockIdx.y;
  const int block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
  const int block_offset_X = ctaid_X * CHUNK_DIM_X;

  // Chunk Y stays inside one tensor (split_sections[i] % 128 == 0).
  const int tensor_id = GetTensorId(args, block_offset_Y);
  const int local_row_base = block_offset_Y - args.split_sections_range[tensor_id];
  const int M_t = args.split_sections_range[tensor_id + 1] - args.split_sections_range[tensor_id];

  // Per-tensor output bases (one constant-cache lookup per CTA).
  uint8_t* const q_row_base =
      DO_ROW ? reinterpret_cast<uint8_t*>(args.q_row_list[tensor_id]) : nullptr;
  uint8_t* const q_col_base =
      DO_COL ? reinterpret_cast<uint8_t*>(args.q_col_list[tensor_id]) : nullptr;
  nvfp4_scale_t* const s_dec_row_base =
      DO_ROW ? reinterpret_cast<nvfp4_scale_t*>(args.s_dec_row_list[tensor_id]) : nullptr;
  nvfp4_scale_t* const s_dec_col_base =
      DO_COL ? reinterpret_cast<nvfp4_scale_t*>(args.s_dec_col_list[tensor_id]) : nullptr;
  const float* const row_amax_base =
      DO_ROW ? reinterpret_cast<const float*>(args.row_amax_list[tensor_id]) : nullptr;
  const float* const col_amax_base =
      DO_COL ? reinterpret_cast<const float*>(args.col_amax_list[tensor_id]) : nullptr;

  const size_t data_stride_row = static_cast<size_t>(cols) / 2;
  const size_t data_stride_col = static_cast<size_t>(M_t) / 2;
  const size_t scale_stride_row = static_cast<size_t>(cols) / SCALE_DIM;
  const size_t scale_stride_col = static_cast<size_t>(M_t) / SCALE_DIM;

  // Load per-row / per-col amax into SMEM cache.
  if (DO_ROW && threadIdx.x < CHUNK_DIM_Y) {
    sRowAmax[threadIdx.x] = row_amax_base[local_row_base + threadIdx.x];
  }
  if (DO_COL && threadIdx.x < CHUNK_DIM_X) {
    sColAmax[threadIdx.x] = col_amax_base[block_offset_X + threadIdx.x];
  }

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Prefetch stage 0.
#pragma unroll
  for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
    const int buff_in_p = stage;
    const int stage_Y = stage / TILES_X;
    const int stage_X = stage % TILES_X;
    const int global_offset_Y = block_offset_Y + stage_Y * TILE_DIM_Y;
    const int global_offset_X = block_offset_X + stage_X * TILE_DIM_X;
    if (leading_thread) {
      ptx::mbarrier_arrive_expect_tx(&IN_buff_readable_mbar[buff_in_p], shmem_buff_size);
      ptx::cp_async_bulk_tensor_2d_global_to_shared(
          reinterpret_cast<uint64_t*>(&sIn[buff_in_p]),
          reinterpret_cast<const uint64_t*>(&tensor_map_input), global_offset_X, global_offset_Y,
          &IN_buff_readable_mbar[buff_in_p]);
    }
  }

  int buff_in = 0;
  int buff_out = 0;
  int buff_out_tr = 0;
  int IN_buff_readable_parity[BUFFS_NUM] = {0, 0};

#pragma unroll
  for (int stage = 0; stage < STAGES; ++stage) {
    const int stage_Y = stage / TILES_X;
    const int stage_X = stage % TILES_X;

    if (stage < STAGES - PREFETCH_STAGES) {
      const int next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
      const int next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES;
      const int next_stage_Y = next_prefetch_stage / TILES_X;
      const int next_stage_X = next_prefetch_stage % TILES_X;
      const int next_global_offset_Y = block_offset_Y + next_stage_Y * TILE_DIM_Y;
      const int next_global_offset_X = block_offset_X + next_stage_X * TILE_DIM_X;
      if (leading_thread) {
        ptx::mbarrier_arrive_expect_tx(&IN_buff_readable_mbar[next_prefetch_buff], shmem_buff_size);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t*>(&sIn[next_prefetch_buff]),
            reinterpret_cast<const uint64_t*>(&tensor_map_input), next_global_offset_X,
            next_global_offset_Y, &IN_buff_readable_mbar[next_prefetch_buff]);
      }
      ptx::fence_proxy_async_shared_cta();
    }

    // Wait for current stage's input tile to land.
    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                     IN_buff_readable_parity[buff_in]);
    IN_buff_readable_parity[buff_in] ^= 1;

    // 4x32 cooperative row + col encode helpers.
    if (DO_ROW) {
      rowwise_scaling_per_token(sIn_ptr, sOut_ptr, sSFrowwise_ptr, sRowAmax, stage_Y, stage_X,
                                buff_in, buff_out);
    }
    if (DO_COL) {
      colwise_scaling_per_token<kWithRht>(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, sColAmax, stage_Y,
                                          stage_X, buff_in, buff_out_tr, random_sign_mask_t);
    }

    // Make helper SMEM writes visible before the scatter epilogue.
    __syncthreads();

    // Scatter sOut / sOut_tr to per-tensor buffers via cooperative b128 stores;
    // 2 threads per row/col x 16 B = 2048 B per sub-tile per direction.
    if (DO_ROW) {
      auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
      const int row_in_subtile = static_cast<int>(threadIdx.x) >> 1;  // 0..63
      const int half = static_cast<int>(threadIdx.x) & 1;             // 0..1
      const int local_row = local_row_base + stage_Y * TILE_DIM_Y + row_in_subtile;
      const int byte_off_X = (block_offset_X / 2) + stage_X * (TILE_DIM_X / 2) + half * 16;
      const uint4* src = reinterpret_cast<const uint4*>(&sOut[buff_out][row_in_subtile][half * 16]);
      uint4* dst = reinterpret_cast<uint4*>(
          q_row_base + static_cast<size_t>(local_row) * data_stride_row + byte_off_X);
      *dst = *src;
    }
    if (DO_COL) {
      auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
      const int col_in_subtile = static_cast<int>(threadIdx.x) >> 1;  // 0..63
      const int half = static_cast<int>(threadIdx.x) & 1;             // 0..1
      const int global_col = block_offset_X + stage_X * TILE_DIM_X + col_in_subtile;
      const int byte_off_M = (local_row_base / 2) + stage_Y * (TILE_DIM_Y / 2) + half * 16;
      const uint4* src =
          reinterpret_cast<const uint4*>(&sOut_tr[buff_out_tr][col_in_subtile][half * 16]);
      uint4* dst = reinterpret_cast<uint4*>(
          q_col_base + static_cast<size_t>(global_col) * data_stride_col + byte_off_M);
      *dst = *src;
    }

    // Sync so the scatter completes before next stage overwrites the buffer.
    __syncthreads();

    buff_in = (buff_in + 1) % BUFFS_NUM;
    buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
    buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
  }

  // SF epilogue: cooperative store of sSFrowwise / sSFcolwise to global.
  if (DO_ROW) {
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_X>;
    const size_t scales_block_offset_X_rowwise = static_cast<size_t>(ctaid_X) * SCALES_PER_CHUNK_X;
    for (int row = static_cast<int>(threadIdx.x); row < CHUNK_DIM_Y; row += THREADS_NUM) {
      ScalesVec& scales_vec = *reinterpret_cast<ScalesVec*>(sSFrowwise[row]);
      const size_t local_row = static_cast<size_t>(local_row_base) + row;
      const size_t scale_idx_global = local_row * scale_stride_row + scales_block_offset_X_rowwise;
      scales_vec.store_to_elts(&s_dec_row_base[scale_idx_global], 0, SCALES_PER_CHUNK_X);
    }
  }
  if (DO_COL) {
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
    // M-block offset within s_dec_col[global_col] (shape (K, M_i/16) row-major).
    const size_t local_block_offset_M = static_cast<size_t>(local_row_base) / SCALE_DIM;
    for (int row_tr = static_cast<int>(threadIdx.x); row_tr < CHUNK_DIM_X; row_tr += THREADS_NUM) {
      ScalesVec& scales_vec = *reinterpret_cast<ScalesVec*>(sSFcolwise[row_tr]);
      const size_t global_col = static_cast<size_t>(block_offset_X) + row_tr;
      const size_t scale_idx_global = global_col * scale_stride_col + local_block_offset_M;
      scales_vec.store_to_elts(&s_dec_col_base[scale_idx_global], 0, SCALES_PER_CHUNK_Y);
    }
  }

  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_invalid(&IN_buff_readable_mbar[buff]);
    }
  }
#else
  (void)tensor_map_input;
  (void)args;
  (void)noop;
  (void)rows;
  (void)cols;
  (void)random_sign_mask_t;
  NVTE_DEVICE_ERROR("Fused grouped per-token cast kernel requires SM 10.0+ (Blackwell).");
#endif  // __CUDA_ARCH__ >= 1000
}

// Host launcher for the fused K2 path. bf16-only.
// with_rht=true applies a 16-pt RHT on the col direction; K1 amax must have
// used the same flag + mask, else inner SF + FP4 saturate against mismatched
// data.
inline void launch_grouped_fused_cast_bf16(const NVFP4PerTokenMultiArgs& args,
                                           const SimpleTensor& input_data, int sum_M, int K,
                                           bool do_row, bool do_col, bool with_rht,
                                           uint32_t random_sign_mask_t, const float* noop,
                                           cudaStream_t stream) {
  if (!do_row && !do_col) return;

  checkCuDriverContext(stream);

  alignas(64) CUtensorMap tmap_in{};
  create_2D_tensor_map(tmap_in, input_data, sum_M, K, TILE_DIM_Y, TILE_DIM_X, K, 0,
                       sizeof(FusedIType) * 8);

  dim3 grid(static_cast<unsigned>(K / CHUNK_DIM_X), static_cast<unsigned>(sum_M / CHUNK_DIM_Y), 1);
  dim3 block(THREADS_NUM, 1, 1);

  // Collapse to kWithRht=false when no colwise output is requested.
  const bool with_rht_effective = with_rht && do_col;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      do_row, DO_ROW,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          do_col, DO_COL, TRANSFORMER_ENGINE_SWITCH_CONDITION(with_rht_effective, kWithRht, {
            constexpr int sz_in = DIVUP_TO_MULTIPLE(BUFFS_NUM * BUFF_IN_SIZE * sizeof(FusedIType),
                                                    TMA_SHMEM_ALIGNMENT);
            constexpr int sz_out_r =
                DO_ROW ? DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT) : 0;
            constexpr int sz_out_c =
                DO_COL ? DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT)
                       : 0;
            constexpr int sz_sf_r =
                DO_ROW ? DIVUP_TO_MULTIPLE(CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t),
                                           TMA_SHMEM_ALIGNMENT)
                       : 0;
            constexpr int sz_sf_c =
                DO_COL ? DIVUP_TO_MULTIPLE(CHUNK_DIM_X * SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t),
                                           TMA_SHMEM_ALIGNMENT)
                       : 0;
            constexpr int dshmem_size =
                sz_in + sz_out_r + sz_out_c + sz_sf_r + sz_sf_c + TMA_SHMEM_ALIGNMENT;
            auto kernel = group_per_token_fused_cast_kernel<DO_ROW, DO_COL, kWithRht>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
            kernel<<<grid, block, dshmem_size, stream>>>(
                tmap_in, args, noop, static_cast<size_t>(sum_M), static_cast<size_t>(K),
                random_sign_mask_t);
          })););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Host launcher for the fused K1 path. bf16-only.
// with_rht=true applies a 16-pt RHT on the col amax (rowwise raw). The
// downstream K2 cast MUST use the same flag + mask.
inline void launch_grouped_fused_amax_bf16(const NVFP4PerTokenMultiArgs& args,
                                           const SimpleTensor& input_data, int sum_M, int K,
                                           bool do_row, bool do_col, bool with_rht,
                                           uint32_t random_sign_mask_t, const float* noop,
                                           cudaStream_t stream) {
  if (!do_row && !do_col) return;

  // Pre-zero amax slots (atomicMax identity).
  {
    dim3 grid_zero(static_cast<unsigned>(args.num_tensors));
    dim3 block_zero(256);
    if (do_row && do_col) {
      group_per_token_fused_zero_amax_kernel<true, true>
          <<<grid_zero, block_zero, 0, stream>>>(args, K);
    } else if (do_row) {
      group_per_token_fused_zero_amax_kernel<true, false>
          <<<grid_zero, block_zero, 0, stream>>>(args, K);
    } else {
      group_per_token_fused_zero_amax_kernel<false, true>
          <<<grid_zero, block_zero, 0, stream>>>(args, K);
    }
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  checkCuDriverContext(stream);

  alignas(64) CUtensorMap tmap_in{};
  create_2D_tensor_map(tmap_in, input_data, sum_M, K, TILE_DIM_Y, TILE_DIM_X, K, 0,
                       sizeof(FusedIType) * 8);

  constexpr int buff_elems_total_in = BUFFS_NUM * BUFF_IN_SIZE;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(FusedIType), TMA_SHMEM_ALIGNMENT);
  constexpr int dshmem_size = buff_size_aligned_in + TMA_SHMEM_ALIGNMENT;

  dim3 grid(static_cast<unsigned>(K / CHUNK_DIM_X), static_cast<unsigned>(sum_M / CHUNK_DIM_Y), 1);
  dim3 block(THREADS_NUM, 1, 1);

  // Collapse to kWithRht=false when no colwise amax is requested.
  const bool with_rht_effective = with_rht && do_col;
  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      do_row, DO_ROW,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          do_col, DO_COL, TRANSFORMER_ENGINE_SWITCH_CONDITION(with_rht_effective, kWithRht, {
            auto kernel = group_per_token_fused_amax_kernel<DO_ROW, DO_COL, kWithRht>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
            kernel<<<grid, block, dshmem_size, stream>>>(
                tmap_in, args, noop, static_cast<size_t>(sum_M), static_cast<size_t>(K),
                random_sign_mask_t);
          })););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace fused

// Populate per-tensor pointer tables + split_sections prefix-sum.
// which_buffers bitmask: kBufRowAmax | kBufColAmax | kBufRowCast | kBufColCast.
enum BufferFlags : int {
  kBufRowAmax = 0x1,
  kBufColAmax = 0x2,
  kBufRowCast = 0x4,
  kBufColCast = 0x8,
};

void populate_args(NVFP4PerTokenMultiArgs* args, std::vector<Tensor*>& outputs,
                   const size_t* split_sections, size_t num_tensors, int which_buffers,
                   int expected_sum_M, int K) {
  std::memset(args, 0, sizeof(*args));
  args->num_tensors = static_cast<int>(num_tensors);
  args->split_sections_range[0] = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    Tensor* o = outputs[i];
    NVTE_CHECK(split_sections[i] % 128 == 0, "split_sections[", i, "] = ", split_sections[i],
               " must be a multiple of 128");
    args->split_sections_range[i + 1] =
        args->split_sections_range[i] + static_cast<int>(split_sections[i]);
    if (split_sections[i] == 0) continue;
    if (which_buffers & kBufRowAmax) {
      NVTE_CHECK(o->amax.dptr != nullptr, "NVFP4 per-token grouped: outputs[", i,
                 "].amax must be allocated for rowwise");
      args->row_amax_list[i] = o->amax.dptr;
    }
    if (which_buffers & kBufColAmax) {
      NVTE_CHECK(o->columnwise_amax.dptr != nullptr, "NVFP4 per-token grouped: outputs[", i,
                 "].columnwise_amax must be allocated for columnwise");
      args->col_amax_list[i] = o->columnwise_amax.dptr;
    }
    if (which_buffers & kBufRowCast) {
      NVTE_CHECK(o->data.dptr != nullptr && o->scale_inv.dptr != nullptr,
                 "NVFP4 per-token grouped: outputs[", i,
                 "].data + .scale_inv must be allocated for rowwise cast");
      args->q_row_list[i] = o->data.dptr;
      args->s_dec_row_list[i] = o->scale_inv.dptr;
    }
    if (which_buffers & kBufColCast) {
      NVTE_CHECK(o->columnwise_data.dptr != nullptr && o->columnwise_scale_inv.dptr != nullptr,
                 "NVFP4 per-token grouped: outputs[", i,
                 "].columnwise_data + .columnwise_scale_inv must be allocated for columnwise cast");
      args->q_col_list[i] = o->columnwise_data.dptr;
      args->s_dec_col_list[i] = o->columnwise_scale_inv.dptr;
    }
  }
  NVTE_CHECK(args->split_sections_range[num_tensors] == expected_sum_M,
             "NVFP4 per-token grouped: sum(split_sections) = ",
             args->split_sections_range[num_tensors], " must equal input rows ", expected_sum_M);
  (void)K;
}

// Host entry. do_amax / do_cast select K1 / K2 phases (composite = both).
// with_rht / mask are threaded into BOTH K1 and K2; the caller must use the
// same flag/mask if they invoke amax + cast separately.
void quantize_per_token_grouped(const Tensor& input, std::vector<Tensor*>& outputs,
                                const size_t* split_sections, size_t num_tensors, bool rowwise,
                                bool columnwise, bool do_amax, bool do_cast, bool with_rht,
                                uint32_t random_sign_mask_t, cudaStream_t stream) {
  NVTE_CHECK(num_tensors > 0, "NVFP4 per-token grouped: num_tensors must be > 0");
  NVTE_CHECK(num_tensors <= static_cast<size_t>(kMaxTensorsPerKernel),
             "NVFP4 per-token grouped: num_tensors (", num_tensors,
             ") exceeds kMaxTensorsPerKernel = ", kMaxTensorsPerKernel);
  NVTE_CHECK(rowwise || columnwise,
             "NVFP4 per-token grouped: at least one of rowwise/columnwise must be true");
  NVTE_CHECK(input.has_data(), "NVFP4 per-token grouped: input has no data");
  NVTE_CHECK(input.dtype() == DType::kBFloat16,
             "NVFP4 per-token grouped: input dtype must be bf16 (got ",
             static_cast<int>(input.dtype()), ")");

  const int sum_M = static_cast<int>(input.flat_first_dim());
  const int K = static_cast<int>(input.flat_last_dim());
  if (sum_M == 0 || K == 0) return;
  NVTE_CHECK(K % 128 == 0, "NVFP4 per-token grouped: K (", K, ") must be a multiple of 128");

  int which_buffers = 0;
  if ((do_amax || do_cast) && rowwise) which_buffers |= kBufRowAmax;
  if ((do_amax || do_cast) && columnwise) which_buffers |= kBufColAmax;
  if (do_cast && rowwise) which_buffers |= kBufRowCast;
  if (do_cast && columnwise) which_buffers |= kBufColCast;

  NVFP4PerTokenMultiArgs args;
  populate_args(&args, outputs, split_sections, num_tensors, which_buffers, sum_M, K);

  // K1 + K2 = 2 fused launches; K1 must complete before K2 reads its amax.
  if (do_amax) {
    fused::launch_grouped_fused_amax_bf16(args, input.data, sum_M, K,
                                          /*do_row=*/rowwise,
                                          /*do_col=*/columnwise,
                                          /*with_rht=*/with_rht,
                                          /*random_sign_mask_t=*/random_sign_mask_t,
                                          /*noop=*/nullptr, stream);
  }
  if (do_cast) {
    fused::launch_grouped_fused_cast_bf16(args, input.data, sum_M, K,
                                          /*do_row=*/rowwise,
                                          /*do_col=*/columnwise,
                                          /*with_rht=*/with_rht,
                                          /*random_sign_mask_t=*/random_sign_mask_t,
                                          /*noop=*/nullptr, stream);
  }
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace nvfp4_per_token_group
}  // namespace transformer_engine

// C-API entries.
namespace {

std::vector<transformer_engine::Tensor*> collect_outputs(NVTETensor* outputs, size_t num_tensors) {
  std::vector<transformer_engine::Tensor*> v;
  v.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    v.push_back(transformer_engine::convertNVTETensorCheck(outputs[i]));
  }
  return v;
}

}  // namespace

void nvte_group_nvfp4_per_token_amax(const NVTETensor input, NVTETensor* outputs,
                                     const size_t* split_sections, size_t num_tensors, bool rowwise,
                                     bool columnwise, int with_rht, int random_sign_mask_t,
                                     cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_group_nvfp4_per_token_amax);
  using namespace transformer_engine;
  if (num_tensors == 0) return;
  const Tensor* in = convertNVTETensorCheck(input);
  std::vector<Tensor*> outs = collect_outputs(outputs, num_tensors);
  // C-API mirrors nvte_nvfp4_per_token_amax: `int` for cross-language ABI
  // safety; internal kernel arg is uint32_t with only the low 16 bits used.
  nvfp4_per_token_group::quantize_per_token_grouped(
      *in, outs, split_sections, num_tensors, rowwise, columnwise,
      /*do_amax=*/true, /*do_cast=*/false,
      /*with_rht=*/with_rht != 0,
      /*random_sign_mask_t=*/
      static_cast<uint32_t>(random_sign_mask_t) & 0xFFFFu, stream);
#else
  (void)input;
  (void)outputs;
  (void)split_sections;
  (void)num_tensors;
  (void)rowwise;
  (void)columnwise;
  (void)with_rht;
  (void)random_sign_mask_t;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

void nvte_group_nvfp4_per_token_cast(const NVTETensor input, NVTETensor* outputs,
                                     const size_t* split_sections, size_t num_tensors, bool rowwise,
                                     bool columnwise, int with_rht, int random_sign_mask_t,
                                     cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_group_nvfp4_per_token_cast);
  using namespace transformer_engine;
  if (num_tensors == 0) return;
  const Tensor* in = convertNVTETensorCheck(input);
  std::vector<Tensor*> outs = collect_outputs(outputs, num_tensors);
  nvfp4_per_token_group::quantize_per_token_grouped(
      *in, outs, split_sections, num_tensors, rowwise, columnwise,
      /*do_amax=*/false, /*do_cast=*/true,
      /*with_rht=*/with_rht != 0,
      /*random_sign_mask_t=*/
      static_cast<uint32_t>(random_sign_mask_t) & 0xFFFFu, stream);
#else
  (void)input;
  (void)outputs;
  (void)split_sections;
  (void)num_tensors;
  (void)rowwise;
  (void)columnwise;
  (void)with_rht;
  (void)random_sign_mask_t;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

void nvte_group_nvfp4_per_token_quantize(const NVTETensor input, NVTETensor* outputs,
                                         const size_t* split_sections, size_t num_tensors,
                                         bool rowwise, bool columnwise, int with_rht,
                                         int random_sign_mask_t, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_group_nvfp4_per_token_quantize);
  using namespace transformer_engine;
  if (num_tensors == 0) return;
  const Tensor* in = convertNVTETensorCheck(input);
  std::vector<Tensor*> outs = collect_outputs(outputs, num_tensors);
  nvfp4_per_token_group::quantize_per_token_grouped(
      *in, outs, split_sections, num_tensors, rowwise, columnwise,
      /*do_amax=*/true, /*do_cast=*/true,
      /*with_rht=*/with_rht != 0,
      /*random_sign_mask_t=*/
      static_cast<uint32_t>(random_sign_mask_t) & 0xFFFFu, stream);
#else
  (void)input;
  (void)outputs;
  (void)split_sections;
  (void)num_tensors;
  (void)rowwise;
  (void)columnwise;
  (void)with_rht;
  (void)random_sign_mask_t;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}
