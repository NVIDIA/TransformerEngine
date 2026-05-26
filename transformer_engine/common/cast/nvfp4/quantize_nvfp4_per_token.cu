/*************************************************************************
 * Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_nvfp4_per_token.cu
 *  \brief NVFP4 per-token cast on the bf16 fast path:
 *         TMA + mbarrier + 64x64 sub-tile + 2-buffer ping-pong.
 *
 *  Pipeline structure mirrors the per-tensor cast kernel
 *  (``quantize_transpose_nvfp4_tuned_1D_kernel``) and the RHT
 *  amax kernel (``HadamardAmaxTmaKernel``):
 *
 *    * Two-kernel design (amax pass + encode pass). Output of amax fed
 *      into encode via per-row/per-col buffers in ``output->amax``
 *      and ``output->columnwise_amax`` (sized [M] and [K] respectively).
 *    * Each CTA covers a 128x128 chunk decomposed as 4 sequential 64x64
 *      sub-tiles, double-buffered. Each sub-tile is one TMA bulk-2D
 *      tensor transaction. mbarrier expect_tx + parity wait gives
 *      one-iteration-overlap between HBM and compute.
 *    * Encode pass reads the input tile ONCE into SMEM, then dispatches
 *      both the rowwise (FP4 + per-row scale) and the columnwise (FP4
 *      transpose + per-col scale) outputs from that same staged copy.
 *      Outer scaling factors S_enc are loaded from
 *      ``row_amax_in[M]`` / ``col_amax_in[K]`` once per CTA into a small
 *      SMEM cache (1 KiB total).
 */

#include <transformer_engine/nvfp4_per_token.h>

#include "common/cast/core/common.cuh"
#include "common/cast/nvfp4/core_nvfp4.cuh"
#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace nvfp4_per_token {

#if FP4_TYPE_SUPPORTED

using dispatch::common::align_smem_ptr_per_TMA_requirements;
using dispatch::nvfp4::nvfp4_scale_t;
using dispatch::nvfp4::core::compute_global_encode_scaling_factor_FP4;
using dispatch::nvfp4::quantization_SF::compute_decoding_scaling_factor;

constexpr int CHUNK_DIM_Y = 128;                // CTA covers this many rows of input
constexpr int CHUNK_DIM_X = 128;                // CTA covers this many cols of input
constexpr int TILE_DIM_Y = 64;                  // TMA bulk-2D box height
constexpr int TILE_DIM_X = 64;                  // TMA bulk-2D box width
constexpr int THREADS_NUM = 128;                // threads per CTA
constexpr int ELTS_PER_THREAD = 16;             // = NVFP4 block size = SCALE_DIM
constexpr int SCALE_DIM = 16;                   // NVFP4 inner block (1x16)
constexpr int PREFETCH_STAGES = 1;              // 1-stage prefetch overlap
constexpr int BUFFS_NUM = PREFETCH_STAGES + 1;  // = 2 ping-pong input buffers

// Derived (chunk / tile / stage)
constexpr int TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;  // 2
constexpr int TILES_X = CHUNK_DIM_X / TILE_DIM_X;  // 2
constexpr int STAGES = TILES_Y * TILES_X;          // 4

constexpr int SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;  // 8 inner blocks per row of the chunk
constexpr int SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;  // 8 inner blocks per col of the chunk
constexpr int SCALES_PER_TILE_X = TILE_DIM_X / SCALE_DIM;    // 4
constexpr int SCALES_PER_TILE_Y = TILE_DIM_Y / SCALE_DIM;    // 4

// Encode helpers' thread layout (rowwise pass: 4x32 = K-dim x M-dim)
constexpr int THREADS_X_ROWWISE = TILE_DIM_X / ELTS_PER_THREAD;     // 4
constexpr int THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;  // 32
constexpr int THREADS_PER_SCALE_ROWWISE =
    SCALE_DIM / ELTS_PER_THREAD;  // 1 (each block owned by 1 thread)
constexpr int ITERATIONS_NORMAL = TILE_DIM_Y / THREADS_Y_ROWWISE;  // 2

// Encode helpers' thread layout (colwise pass: tid.X for col, warp for M-block)
constexpr int THREADS_X_TR = TILE_DIM_X / 2;              // 32 cols per warp
constexpr int THREADS_Y_TR = THREADS_NUM / THREADS_X_TR;  // 4 (warps)

// Buffer dimensions (input bf16 SMEM tiles + FP4 output SMEM tiles for TMA store)
constexpr int BUFF_IN_DIM_Y = TILE_DIM_Y;
constexpr int BUFF_IN_DIM_X = TILE_DIM_X;
constexpr int BUFF_IN_SIZE = BUFF_IN_DIM_Y * BUFF_IN_DIM_X;  // elements
constexpr int BUFF_OUT_DIM_Y = TILE_DIM_Y;
constexpr int BUFF_OUT_DIM_X = (TILE_DIM_X * 4) / 8;  // 32 (2 fp4 per byte)
constexpr int BUFF_OUT_SIZE = BUFF_OUT_DIM_Y * BUFF_OUT_DIM_X;
constexpr int BUFF_OUT_TR_DIM_Y = TILE_DIM_X;
constexpr int BUFF_OUT_TR_DIM_X = (TILE_DIM_Y * 4) / 8;  // 32
constexpr int BUFF_OUT_TR_SIZE = BUFF_OUT_TR_DIM_Y * BUFF_OUT_TR_DIM_X;
constexpr int BUFFS_NUM_OUT = BUFFS_NUM;  // 2 ping-pong (matches input)
constexpr int BUFFS_NUM_OUT_TR = 2;       // 2 ping-pong for transpose

// Manual swizzling parameters to reduce SMEM bank conflicts on rowwise loads
constexpr int PACK_SIZE = 8;
constexpr int WAVES = ELTS_PER_THREAD / PACK_SIZE;                     // 2
constexpr int TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;                    // 256
constexpr int THREADS_PER_BANK = TOTAL_BANKS_WIDTH / ELTS_PER_THREAD;  // 16

using IType = bf16;
using IType2 = ptx::FPx2<IType>;  // = ptx::bf16x2
using IType3D = IType[BUFFS_NUM][BUFF_IN_DIM_Y][BUFF_IN_DIM_X];
using IType2x3D = IType2[BUFFS_NUM][BUFF_IN_DIM_Y][BUFF_IN_DIM_X / 2];
using OType2x3D = fp4e2m1x2[BUFFS_NUM_OUT][BUFF_OUT_DIM_Y][BUFF_OUT_DIM_X];
using OType2xt3D = fp4e2m1x2[BUFFS_NUM_OUT_TR][BUFF_OUT_TR_DIM_Y][BUFF_OUT_TR_DIM_X];
using ScalesType2D = nvfp4_scale_t[CHUNK_DIM_Y][SCALES_PER_CHUNK_X];
using ScalesTypeTr2D = nvfp4_scale_t[CHUNK_DIM_X][SCALES_PER_CHUNK_Y];

// Compute the per-block (1x16) byte-equal arithmetic and emit FP4 codes into
// SMEM rowwise output buffer + e4m3 scale into SMEM rowwise scale buffer.
__device__ __forceinline__ void rowwise_scaling_per_token(
    const IType* __restrict__ sIn_ptr, fp4e2m1x2* __restrict__ sOut_ptr,
    nvfp4_scale_t* __restrict__ sSFrowwise_ptr,
    const float* __restrict__ sRowAmax,  // [CHUNK_DIM_Y], indexed by chunk-local row
    const int stage_Y, const int stage_X, const int buff_in, const int buff_out) {
  const auto& sIn = *reinterpret_cast<const IType3D*>(sIn_ptr);
  auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
  auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);

  const int thread_lane = threadIdx.x % THREADS_PER_WARP;
  const int bank_group = thread_lane / THREADS_PER_BANK;

  const int tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;  // 0..31
  const int tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;  // 0..3

  const int thread_offset_X_rowwise =
      tid_X_rowwise * ELTS_PER_THREAD;  // K-elt offset in tile (0/16/32/48)

  const int SF_thread_offset_rowwise_X =
      tid_X_rowwise / THREADS_PER_SCALE_ROWWISE;  // = tid_X_rowwise here
  const bool SF_storing_thread = (tid_X_rowwise % THREADS_PER_SCALE_ROWWISE == 0);

  const int stage_rowwise_scales_offset_X =
      SF_thread_offset_rowwise_X + stage_X * SCALES_PER_TILE_X;

#pragma unroll
  for (int it = 0; it < ITERATIONS_NORMAL; ++it) {
    const int it_offset_Y_rowwise = tid_Y_rowwise + it * THREADS_Y_ROWWISE;  // 0..63 over 2 iters
    const int chunk_local_row = stage_Y * TILE_DIM_Y + it_offset_Y_rowwise;  // 0..127

    // Per-row S_enc (look up from CTA-cached row amax buffer)
    const float row_amax = sRowAmax[chunk_local_row];
    const float S_enc = compute_global_encode_scaling_factor_FP4(fmaxf(row_amax, 1e-12f));

    __align__(16) IType2 rIn[WAVES][PACK_SIZE / 2];

    // Read 16 elements (in PACK_SIZE=8 waves), swizzled to avoid bank conflicts,
    // and reduce to a 1x16 block amax.
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

    // Byte-equal compute path (matches the Python reference in
    // ``NVFP4QuantizerPerTokenRef``):
    const fp8e4m3 s_dec = compute_decoding_scaling_factor(block_amax, S_enc);
    const float s_dec_f = static_cast<float>(s_dec);
    const float block_scale = (s_dec_f == 0.f) ? 0.f : __fdiv_rn(S_enc, s_dec_f);

    // Store e4m3 scale to SMEM SF buffer (1 thread per 1x16 block stores).
    if (SF_storing_thread) {
      const int scales_offset_Y = chunk_local_row;
      const int scales_offset_X = stage_rowwise_scales_offset_X;
      sSFrowwise[scales_offset_Y][scales_offset_X] = s_dec;
    }

    // Cast 16 elements to FP4 using mul_cvt_4x (4 elements per call, the
    // byte-equal path against the Python reference). We've already pre-loaded
    // into rIn[WAVES][4].
    //   WAVES = 2, PACK_SIZE/2 = 4 elements per wave
    //   Total per iteration: 2 waves * (4 IType2 elts) = 16 elements
#pragma unroll
    for (int w = 0; w < WAVES; ++w) {
      const int swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % ELTS_PER_THREAD;
      const int swizzled_idx = (swizzled_group_idx + thread_offset_X_rowwise) / 2;

      // 4 fp4 quads from 8 bf16 elements (in PACK_SIZE=8 waves):
      //   rIn[w][0..3] = 4 IType2 pairs = 8 elements.
      //   Each mul_cvt_4x packs 4 elements; we need 2 calls per wave.
      fp4e2m1x4 qu0{}, qu1{};
      ptx::mul_cvt_4x(qu0, rIn[w][0], rIn[w][1], block_scale);
      ptx::mul_cvt_4x(qu1, rIn[w][2], rIn[w][3], block_scale);

      // Pack into a 32-bit word and store to SMEM out (b32 store)
      uint32_t out_x8 = (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&qu0))) |
                        (static_cast<uint32_t>(*reinterpret_cast<uint16_t*>(&qu1)) << 16);
      ptx::st_shared_b32(&sOut[buff_out][it_offset_Y_rowwise][swizzled_idx], out_x8);
    }
  }
}

// Compute the per-block (1x16, along M) byte-equal arithmetic for the columnwise
// pass; emit transposed FP4 + e4m3 scale into SMEM.
__device__ __forceinline__ void colwise_scaling_per_token(
    const IType* __restrict__ sIn_ptr, fp4e2m1x2* __restrict__ sOut_tr_ptr,
    nvfp4_scale_t* __restrict__ sSFcolwise_ptr,
    const float* __restrict__ sColAmax,  // [CHUNK_DIM_X], indexed by chunk-local col
    const int stage_Y, const int stage_X, const int buff_in, const int buff_out_tr) {
  const auto& sIn2x = *reinterpret_cast<const IType2x3D*>(sIn_ptr);
  auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
  auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);

  const int warp = threadIdx.x / THREADS_PER_WARP;  // 0..3
  const int thread_lane = threadIdx.x % THREADS_PER_WARP;

  const int tid_Y_colwise = (thread_lane % 4 + warp) % 4;  // 0..3 (M-block index in tile)
  const int tid_X_colwise = thread_lane;                   // 0..31 (col-pair index in tile)

  const int thread_offset_Y_colwise = tid_Y_colwise * SCALE_DIM;  // 0/16/32/48
  const int thread_offset_X_colwise = tid_X_colwise * 2;          // 0/2/.../62 (2 cols per thread)

  const int in_thread_offset_Y = thread_offset_Y_colwise;
  const int in_thread_offset_X = thread_offset_X_colwise / 2;  // index into IType2[]

  const int out_tr_thread_offset_Y = thread_offset_X_colwise;      // transpose: X becomes Y
  const int out_tr_thread_offset_X = thread_offset_Y_colwise / 2;  // /2 for fp4e2m1x2 byte index

  const int scale_tr_offset_Y =
      (stage_X * TILE_DIM_X) + 2 * tid_X_colwise;  // chunk-local col index (×1)
  const int scale_tr_offset_X =
      (stage_Y * SCALES_PER_TILE_Y) + tid_Y_colwise;  // chunk-local M-block index

  __align__(8) IType rIn[2][SCALE_DIM];
  // Read 2 columns x 16 rows, accumulate per-column amax.
  IType2 thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
  for (int i = 0; i < SCALE_DIM; ++i) {
    const IType2 elt_pair =
        ptx::ld_shared_b32(&sIn2x[buff_in][in_thread_offset_Y + i][in_thread_offset_X]);
    rIn[0][i] = elt_pair.x;
    rIn[1][i] = elt_pair.y;
    ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, elt_pair);
  }
  // NOTE: thread_amax_2x.x is the amax of column .x; thread_amax_2x.y is amax of column .y.
  const float block_amax[2] = {static_cast<float>(__habs(thread_amax_2x.x)),
                               static_cast<float>(__habs(thread_amax_2x.y))};

#pragma unroll
  for (int w = 0; w < 2; ++w) {
    // Per-col S_enc lookup (each of the 2 cols this thread owns has its own amax/S_enc).
    const int chunk_local_col = scale_tr_offset_Y + w;
    const float col_amax = sColAmax[chunk_local_col];
    const float S_enc_col = compute_global_encode_scaling_factor_FP4(fmaxf(col_amax, 1e-12f));

    const fp8e4m3 s_dec = compute_decoding_scaling_factor(block_amax[w], S_enc_col);
    const float s_dec_f = static_cast<float>(s_dec);
    const float block_scale = (s_dec_f == 0.f) ? 0.f : __fdiv_rn(S_enc_col, s_dec_f);

    // Store e4m3 scale to SMEM colwise SF buffer.
    sSFcolwise[scale_tr_offset_Y + w][scale_tr_offset_X] = s_dec;

    // Cast 16 elements to FP4 via 4x mul_cvt_4x (4 elements per call -> 4 calls).
    // The 16 rIn[w][...] values are bf16; pack into IType2 pairs.
    fp4e2m1x4 qu[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      IType2 in01{rIn[w][4 * e + 0], rIn[w][4 * e + 1]};
      IType2 in23{rIn[w][4 * e + 2], rIn[w][4 * e + 3]};
      ptx::mul_cvt_4x(qu[e], in01, in23, block_scale);
    }

    // Pack 4 fp4e2m1x4 (= 16 fp4) into a 64-bit value and store to SMEM transpose buffer.
    uint64_t out_pack_16x = (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[0])) << 0) |
                            (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[1])) << 16) |
                            (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[2])) << 32) |
                            (static_cast<uint64_t>(*reinterpret_cast<uint16_t*>(&qu[3])) << 48);
    ptx::st_shared_b64(&sOut_tr[buff_out_tr][out_tr_thread_offset_Y + w][out_tr_thread_offset_X],
                       out_pack_16x);
  }
}

// =============================================================================
// Kernel 2: per-token encode (rowwise + optional colwise transpose).
// =============================================================================
template <bool DO_ROW, bool DO_COL>
__global__ void __launch_bounds__(THREADS_NUM)
    per_token_encode_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                            const __grid_constant__ CUtensorMap tensor_map_output,
                            const __grid_constant__ CUtensorMap tensor_map_output_t,
                            nvfp4_scale_t* const scales_ptr, nvfp4_scale_t* const scales_t_ptr,
                            const float* const row_amax_in, const float* const col_amax_in,
                            const float* noop, const size_t rows, const size_t cols,
                            const size_t scale_stride, const size_t scale_stride_t) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const bool leading_thread = (threadIdx.x == 0);

  // -------------------------------------------------------------------------
  // Dynamic SMEM layout
  //   sIn:        2 buffers x (64x64 bf16)        = 16 KiB
  //   sOut:       2 buffers x (64x32 fp4 packed)  = 4 KiB   (rowwise FP4)
  //   sOut_tr:    2 buffers x (64x32 fp4 packed)  = 4 KiB   (colwise FP4)
  //   sSFrowwise: 128 x 8 e4m3                    = 1 KiB
  //   sSFcolwise: 128 x 8 e4m3                    = 1 KiB
  //   sRowAmax:   128 fp32                        = 512 B
  //   sColAmax:   128 fp32                        = 512 B
  //   IN_buff_readable_mbar: 2 x 8 B              = 16 B
  // Total: ~27 KiB + alignment padding.
  // -------------------------------------------------------------------------
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

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char* dshmem = align_smem_ptr_per_TMA_requirements(dynamic_shmem);

  IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
  fp4e2m1x2* sOut_ptr = reinterpret_cast<fp4e2m1x2*>(dshmem + buff_size_aligned_in);
  fp4e2m1x2* sOut_tr_ptr =
      reinterpret_cast<fp4e2m1x2*>(dshmem + buff_size_aligned_in + out_mem_rowwise_data);

  nvfp4_scale_t* sSFrowwise_ptr = reinterpret_cast<nvfp4_scale_t*>(
      dshmem + buff_size_aligned_in + out_mem_rowwise_data + out_mem_colwise_data);
  nvfp4_scale_t* sSFcolwise_ptr =
      reinterpret_cast<nvfp4_scale_t*>(dshmem + buff_size_aligned_in + out_mem_rowwise_data +
                                       out_mem_colwise_data + out_mem_rowwise_scales);

  // Per-CTA row/col amax SMEM cache (128 floats each).
  __shared__ float sRowAmax[CHUNK_DIM_Y];
  __shared__ float sColAmax[CHUNK_DIM_X];
  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];

  auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);

  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const int32_t ctaid_X = blockIdx.x;
  const int32_t ctaid_Y = blockIdx.y;
  const int block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
  const int block_offset_X = ctaid_X * CHUNK_DIM_X;
  // Transpose-output block offsets: row-CTA(X) -> col-tensor's M; col-CTA(Y) -> col-tensor's N.
  const int block_offset_Y_tr = ctaid_X * CHUNK_DIM_X;
  const int block_offset_X_tr = ctaid_Y * CHUNK_DIM_Y;

  const int scales_block_offset_Y_rowwise = ctaid_Y * CHUNK_DIM_Y;
  const int scales_block_offset_X_rowwise = ctaid_X * SCALES_PER_CHUNK_X;
  const int scales_block_offset_Y_tr = ctaid_X * CHUNK_DIM_X;
  const int scales_block_offset_X_tr = ctaid_Y * SCALES_PER_CHUNK_Y;

  // Load per-row / per-col amax into SMEM cache (cooperative, full chunk = 128 entries each).
  if (DO_ROW && threadIdx.x < CHUNK_DIM_Y) {
    sRowAmax[threadIdx.x] = row_amax_in[block_offset_Y + threadIdx.x];
  }
  if (DO_COL && threadIdx.x < CHUNK_DIM_X) {
    sColAmax[threadIdx.x] = col_amax_in[block_offset_X + threadIdx.x];
  }

  // Initialize mbarriers.
  if (leading_thread) {
#pragma unroll
    for (int buff = 0; buff < BUFFS_NUM; ++buff) {
      ptx::mbarrier_init(&IN_buff_readable_mbar[buff], 1);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // Prefetch stage 0 (one-iteration overlap throughout main loop).
#pragma unroll
  for (int stage = 0; stage < PREFETCH_STAGES; ++stage) {
    const int buff_in = stage;
    const int stage_Y = stage / TILES_X;
    const int stage_X = stage % TILES_X;
    const int global_offset_Y = block_offset_Y + stage_Y * TILE_DIM_Y;
    const int global_offset_X = block_offset_X + stage_X * TILE_DIM_X;
    if (leading_thread) {
      uint64_t* dst = reinterpret_cast<uint64_t*>(&sIn[buff_in]);
      const uint64_t* src = reinterpret_cast<const uint64_t*>(&tensor_map_input);
      ptx::mbarrier_arrive_expect_tx(&IN_buff_readable_mbar[buff_in], shmem_buff_size);
      ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, global_offset_X, global_offset_Y,
                                                    &IN_buff_readable_mbar[buff_in]);
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
    const int stage_offset_Y = stage_Y * TILE_DIM_Y;
    const int stage_offset_X = stage_X * TILE_DIM_X;

    // Prefetch next stage's input (skip after the second-to-last stage).
    if (stage < STAGES - PREFETCH_STAGES) {
      const int next_prefetch_buff = (buff_in + PREFETCH_STAGES) % BUFFS_NUM;
      const int next_prefetch_stage = (stage + PREFETCH_STAGES) % STAGES;
      const int next_stage_Y = next_prefetch_stage / TILES_X;
      const int next_stage_X = next_prefetch_stage % TILES_X;
      const int next_global_offset_Y = block_offset_Y + next_stage_Y * TILE_DIM_Y;
      const int next_global_offset_X = block_offset_X + next_stage_X * TILE_DIM_X;

      if (leading_thread) {
        uint64_t* dst = reinterpret_cast<uint64_t*>(&sIn[next_prefetch_buff]);
        const uint64_t* src = reinterpret_cast<const uint64_t*>(&tensor_map_input);
        ptx::mbarrier_arrive_expect_tx(&IN_buff_readable_mbar[next_prefetch_buff], shmem_buff_size);
        ptx::cp_async_bulk_tensor_2d_global_to_shared(dst, src, next_global_offset_X,
                                                      next_global_offset_Y,
                                                      &IN_buff_readable_mbar[next_prefetch_buff]);
      }
      ptx::fence_proxy_async_shared_cta();
    }

    // Wait for current stage's input to land.
    ptx::mbarrier_wait_parity_acquire_cta_shared_cta(&IN_buff_readable_mbar[buff_in],
                                                     IN_buff_readable_parity[buff_in]);
    IN_buff_readable_parity[buff_in] ^= 1;

    // Wait for any prior TMA store to have finished reading the output SMEM
    // buffers (so we can overwrite them).
    ptx::cp_async_bulk_wait_group_read<PREFETCH_STAGES>();

    // ----- Compute: rowwise + colwise from the same SMEM tile -----
    if (DO_ROW) {
      rowwise_scaling_per_token(sIn_ptr, sOut_ptr, sSFrowwise_ptr, sRowAmax, stage_Y, stage_X,
                                buff_in, buff_out);
    }
    if (DO_COL) {
      colwise_scaling_per_token(sIn_ptr, sOut_tr_ptr, sSFcolwise_ptr, sColAmax, stage_Y, stage_X,
                                buff_in, buff_out_tr);
    }

    // Fence + sync so all threads' SMEM writes are visible to TMA store.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    // Issue TMA store(s) for this stage's outputs.
    if (leading_thread) {
      const int global_offset_Y = block_offset_Y + stage_offset_Y;
      const int global_offset_X = block_offset_X + stage_offset_X;
      const int global_offset_Y_tr = block_offset_Y_tr + stage_offset_X;
      const int global_offset_X_tr = block_offset_X_tr + stage_offset_Y;

      if (DO_ROW) {
        auto& sOut = *reinterpret_cast<OType2x3D*>(sOut_ptr);
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output), global_offset_X, global_offset_Y,
            reinterpret_cast<uint64_t*>(&sOut[buff_out]));
      }
      if (DO_COL) {
        auto& sOut_tr = *reinterpret_cast<OType2xt3D*>(sOut_tr_ptr);
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            reinterpret_cast<const uint64_t*>(&tensor_map_output_t), global_offset_X_tr,
            global_offset_Y_tr, reinterpret_cast<uint64_t*>(&sOut_tr[buff_out_tr]));
      }
      ptx::cp_async_bulk_commit_group();
    }

    buff_in = (buff_in + 1) % BUFFS_NUM;
    buff_out = (buff_out + 1) % BUFFS_NUM_OUT;
    buff_out_tr = (buff_out_tr + 1) % BUFFS_NUM_OUT_TR;
  }  // end of stages

  // Vectorized SF scatter to global (chunk-end batch). Mirrors the
  // production tuned 1D scale-store epilogue.
  if (DO_ROW) {
    auto& sSFrowwise = *reinterpret_cast<ScalesType2D*>(sSFrowwise_ptr);
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_X>;
    const int chunk_cols = static_cast<int>(cols) - block_offset_X;
    const int count = min(SCALES_PER_CHUNK_X, chunk_cols / SCALE_DIM);

    for (size_t row = threadIdx.x; row < CHUNK_DIM_Y; row += THREADS_NUM) {
      const size_t row_global = scales_block_offset_Y_rowwise + row;
      if (row_global < rows) {
        ScalesVec& scales_vec = *reinterpret_cast<ScalesVec*>(sSFrowwise[row]);
        const size_t scale_idx_global = row_global * scale_stride + scales_block_offset_X_rowwise;
        scales_vec.store_to_elts(&scales_ptr[scale_idx_global], 0, count);
      }
    }
  }
  if (DO_COL) {
    auto& sSFcolwise = *reinterpret_cast<ScalesTypeTr2D*>(sSFcolwise_ptr);
    using ScalesVec = Vec<nvfp4_scale_t, SCALES_PER_CHUNK_Y>;
    const int chunk_rows = static_cast<int>(rows) - block_offset_Y;
    const int count = min(SCALES_PER_CHUNK_Y, chunk_rows / SCALE_DIM);

    for (size_t row_tr = threadIdx.x; row_tr < CHUNK_DIM_X; row_tr += THREADS_NUM) {
      const size_t row_tr_global = scales_block_offset_Y_tr + row_tr;
      if (row_tr_global < cols) {
        ScalesVec& scales_vec = *reinterpret_cast<ScalesVec*>(sSFcolwise[row_tr]);
        const size_t scale_idx_global = row_tr_global * scale_stride_t + scales_block_offset_X_tr;
        scales_vec.store_to_elts(&scales_t_ptr[scale_idx_global], 0, count);
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
  NVTE_DEVICE_ERROR("Per-token encode kernel requires SM 10.0+ (Blackwell).");
#endif  // __CUDA_ARCH__ >= 1000
}

// =============================================================================
// Kernel 1: per-token amax (rowwise + colwise atomicMaxFloat).
//
// Same TMA + mbarrier + 64x64 sub-tile + ping-pong pipeline as the encode
// kernel above, just with compute = abs + reduce instead of FP4 encode.
//
// Compute mapping (one thread per output slot):
//   tid t in [0, 128):
//     row partial: max over (cols 0..127) for row (row_base + t)
//     col partial: max over (rows 0..127) for col (col_base + t)
//   For each 64x64 sub-tile in stage (stage_Y, stage_X):
//     if t in [stage_Y*64, stage_Y*64+64):  scan 64 cols of sub-tile for row t
//     if t in [stage_X*64, stage_X*64+64):  scan 64 rows of sub-tile for col t
// After all 4 stages, emit one atomicMaxFloat per row slot + one per col slot.
// =============================================================================
template <bool DO_ROW, bool DO_COL>
__global__ void __launch_bounds__(THREADS_NUM)
    per_token_amax_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                          float* __restrict__ row_amax_out,  // [M], nullptr if !DO_ROW
                          float* __restrict__ col_amax_out,  // [K], nullptr if !DO_COL
                          const float* noop, const size_t rows, const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  const bool leading_thread = (threadIdx.x == 0);
  const int tid = threadIdx.x;

  constexpr int buff_elems_total_in = BUFFS_NUM * BUFF_IN_SIZE;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);

  extern __shared__ unsigned char dynamic_shmem[];
  unsigned char* dshmem = align_smem_ptr_per_TMA_requirements(dynamic_shmem);
  IType* sIn_ptr = reinterpret_cast<IType*>(dshmem);
  auto& sIn = *reinterpret_cast<IType3D*>(sIn_ptr);

  __shared__ uint64_t IN_buff_readable_mbar[BUFFS_NUM];
  constexpr int shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  const int32_t ctaid_X = blockIdx.x;
  const int32_t ctaid_Y = blockIdx.y;
  const int block_offset_Y = ctaid_Y * CHUNK_DIM_Y;
  const int block_offset_X = ctaid_X * CHUNK_DIM_X;

  // Per-thread row & col partial accumulators (each thread owns 1 of each).
  float row_partial = 0.f;
  float col_partial = 0.f;

  // Which row / col does THIS thread own within the 128x128 chunk?
  //   row owned: row_base + tid  -> needs sub-tile rows [stage_Y*64, +64)
  //              i.e., this thread contributes to row partial in stages
  //              where stage_Y == tid / 64.
  //   col owned: col_base + tid  -> stage_X == tid / 64.
  const int my_row_stage_Y = tid / TILE_DIM_Y;     // 0 or 1
  const int my_col_stage_X = tid / TILE_DIM_X;     // 0 or 1
  const int my_row_in_subtile = tid % TILE_DIM_Y;  // 0..63
  const int my_col_in_subtile = tid % TILE_DIM_X;  // 0..63

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

    // ----- Row partial update: walk this thread's row across the sub-tile -----
    if (DO_ROW && stage_Y == my_row_stage_Y) {
      // 32 warp lanes each own a distinct row but read col-offset e in lockstep;
      // SMEM row stride is 64*sizeof(bf16) = 128 B = exactly 32 banks, so every
      // lane lands on the same bank set -> 32-way bank conflict per LDS.128.
      // Rotate the e-iter visit order by (my_row_in_subtile >> 2) so that lanes
      // in distinct row-quads pick distinct e values per iter, splitting the
      // warp into 8 disjoint bank groups (4-way conflict, 8x reduction).
      // Per-thread data set unchanged; max() is associative & commutative => byte-equal.
      float local_max = row_partial;
      const int row_bank_group = (my_row_in_subtile >> 2) & 0x7;
#pragma unroll
      for (int e_iter = 0; e_iter < 8; ++e_iter) {
        const int e = ((e_iter + row_bank_group) & 0x7) << 3;
        __uint128_t elts_8x = ptx::ld_shared_b128(&sIn[buff_in][my_row_in_subtile][e]);
        const IType2* pairs = reinterpret_cast<const IType2*>(&elts_8x);
        IType2 amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          ptx::abs_max_2x(amax_2x, amax_2x, pairs[p]);
        }
        local_max =
            fmaxf(local_max, static_cast<float>(__hmax(__habs(amax_2x.x), __habs(amax_2x.y))));
      }
      row_partial = local_max;
    }

    // ----- Col partial update: walk this thread's col down the sub-tile -----
    if (DO_COL && stage_X == my_col_stage_X) {
      // Scan 64 rows for our col. Single-column access pattern (1 byte stride
      // per row in SMEM); we read 1 bf16 at a time. Bank conflicts mitigated
      // by 64-wide tile (column stride = TILE_DIM_X * 2 = 128 bytes, which is
      // 1 bank * 32 rows; with 32 threads on different cols, conflicts hit
      // groups of 32 -> serialized 32-way, accepted for v1).
      float local_max = col_partial;
#pragma unroll
      for (int e = 0; e < TILE_DIM_Y; ++e) {
        const IType v = sIn[buff_in][e][my_col_in_subtile];
        local_max = fmaxf(local_max, fabsf(static_cast<float>(v)));
      }
      col_partial = local_max;
    }

    __syncthreads();
    buff_in = (buff_in + 1) % BUFFS_NUM;
  }

  // ----- Cross-CTA reduction: 1 atomicMaxFloat per row/col slot per CTA -----
  if (DO_ROW) {
    atomicMaxFloat(&row_amax_out[block_offset_Y + tid], row_partial);
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
  NVTE_DEVICE_ERROR("Per-token amax kernel requires SM 10.0+ (Blackwell).");
#endif  // __CUDA_ARCH__ >= 1000
}

#endif  // FP4_TYPE_SUPPORTED (closes the kernels block opened at line 69)

// =============================================================================
// Launchers
// =============================================================================

#if FP4_TYPE_SUPPORTED
// Launch Kernel 1 (amax). Writes only to output->amax / output->columnwise_amax;
// other output fields untouched. Pre-zeroes the amax buffers (atomicMax identity).
inline void launch_amax(const Tensor& input, Tensor* output, const Tensor& noop,
                        cudaStream_t stream) {
  const size_t M = input.flat_first_dim();
  const size_t K = input.flat_last_dim();

  const bool do_row = (output->amax.dptr != nullptr);
  const bool do_col = (output->columnwise_amax.dptr != nullptr);
  if (!do_row && !do_col) return;

  // Pre-zero amax buffers (atomicMaxFloat identity for non-negative values).
  if (do_row) {
    NVTE_CHECK(output->amax.numel() == M, "Per-token amax: output->amax numel must equal M = ", M,
               ", got ", output->amax.numel());
    NVTE_CHECK_CUDA(cudaMemsetAsync(output->amax.dptr, 0, M * sizeof(float), stream));
  }
  if (do_col) {
    NVTE_CHECK(output->columnwise_amax.numel() == K,
               "Per-token amax: output->columnwise_amax numel must equal K = ", K, ", got ",
               output->columnwise_amax.numel());
    NVTE_CHECK_CUDA(cudaMemsetAsync(output->columnwise_amax.dptr, 0, K * sizeof(float), stream));
  }

  checkCuDriverContext(stream);

  alignas(64) CUtensorMap tmap_in{};
  create_2D_tensor_map(tmap_in, input.data, M, K, TILE_DIM_Y, TILE_DIM_X, K, 0, sizeof(IType) * 8);

  constexpr int buff_elems_total_in = BUFFS_NUM * BUFF_IN_SIZE;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int dshmem_size = buff_size_aligned_in + TMA_SHMEM_ALIGNMENT;  // + align pad

  dim3 grid(static_cast<unsigned>(K / CHUNK_DIM_X), static_cast<unsigned>(M / CHUNK_DIM_Y), 1);
  dim3 block(THREADS_NUM, 1, 1);

  const float* noop_ptr =
      (noop.data.dptr != nullptr) ? reinterpret_cast<const float*>(noop.data.dptr) : nullptr;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      do_row, DO_ROW, TRANSFORMER_ENGINE_SWITCH_CONDITION(do_col, DO_COL, {
        auto kernel = per_token_amax_kernel<DO_ROW, DO_COL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
        kernel<<<grid, block, dshmem_size, stream>>>(
            tmap_in, do_row ? reinterpret_cast<float*>(output->amax.dptr) : nullptr,
            do_col ? reinterpret_cast<float*>(output->columnwise_amax.dptr) : nullptr, noop_ptr, M,
            K);
      }););
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Launch Kernel 2 (encode). Requires output->amax / columnwise_amax to be pre-filled
// (by a prior launch_amax call or by an external caller); writes
// output->data / scale_inv / columnwise_data / columnwise_scale_inv.
inline void launch_encode(const Tensor& input, Tensor* output, const Tensor& noop,
                          cudaStream_t stream) {
  const size_t M = input.flat_first_dim();
  const size_t K = input.flat_last_dim();

  const bool do_row = output->has_data();
  const bool do_col = output->has_columnwise_data();
  if (!do_row && !do_col) return;

  if (do_row) {
    NVTE_CHECK(output->amax.dptr != nullptr,
               "Per-token encode: output->amax (per-row, [M]) must be pre-filled.");
    NVTE_CHECK(output->data.dptr != nullptr,
               "Per-token encode: output->data (rowwise FP4) must be allocated.");
    NVTE_CHECK(output->scale_inv.dptr != nullptr,
               "Per-token encode: output->scale_inv must be allocated.");
  }
  if (do_col) {
    NVTE_CHECK(output->columnwise_amax.dptr != nullptr,
               "Per-token encode: output->columnwise_amax (per-col, [K]) must be pre-filled.");
    NVTE_CHECK(output->columnwise_data.dptr != nullptr,
               "Per-token encode: output->columnwise_data must be allocated.");
    NVTE_CHECK(output->columnwise_scale_inv.dptr != nullptr,
               "Per-token encode: output->columnwise_scale_inv must be allocated.");
  }

  checkCuDriverContext(stream);

  alignas(64) CUtensorMap tmap_in{};
  alignas(64) CUtensorMap tmap_out{};
  alignas(64) CUtensorMap tmap_out_t{};

  create_2D_tensor_map(tmap_in, input.data, M, K, TILE_DIM_Y, TILE_DIM_X, K, 0, sizeof(IType) * 8);
  if (do_row) {
    create_2D_tensor_map(tmap_out, output->data, M, K, TILE_DIM_Y, TILE_DIM_X, K, 0, 4);
  }
  if (do_col) {
    create_2D_tensor_map(tmap_out_t, output->columnwise_data, K, M, TILE_DIM_X, TILE_DIM_Y, M, 0,
                         4);
  }

  constexpr int buff_elems_total_in = BUFFS_NUM * BUFF_IN_SIZE;
  constexpr int buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total_in * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT * BUFF_OUT_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_aligned_out_t =
      DIVUP_TO_MULTIPLE(BUFFS_NUM_OUT_TR * BUFF_OUT_TR_SIZE, TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_scales = DIVUP_TO_MULTIPLE(
      CHUNK_DIM_Y * SCALES_PER_CHUNK_X * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);
  constexpr int buff_size_scales_t = DIVUP_TO_MULTIPLE(
      CHUNK_DIM_X * SCALES_PER_CHUNK_Y * sizeof(nvfp4_scale_t), TMA_SHMEM_ALIGNMENT);

  // Total dyn SMEM: input + output FP4 (row + col) + SF (row + col) + 128B align.
  const int dshmem_size = buff_size_aligned_in + (do_row ? buff_size_aligned_out : 0) +
                          (do_col ? buff_size_aligned_out_t : 0) + (do_row ? buff_size_scales : 0) +
                          (do_col ? buff_size_scales_t : 0) + TMA_SHMEM_ALIGNMENT;

  dim3 grid(static_cast<unsigned>(K / CHUNK_DIM_X), static_cast<unsigned>(M / CHUNK_DIM_Y), 1);
  dim3 block(THREADS_NUM, 1, 1);

  const float* noop_ptr =
      (noop.data.dptr != nullptr) ? reinterpret_cast<const float*>(noop.data.dptr) : nullptr;
  const size_t scale_stride = do_row ? output->scale_inv.shape[1] : 0;
  const size_t scale_stride_t = do_col ? output->columnwise_scale_inv.shape[1] : 0;

  nvfp4_scale_t* scales_ptr =
      do_row ? reinterpret_cast<nvfp4_scale_t*>(output->scale_inv.dptr) : nullptr;
  nvfp4_scale_t* scales_t_ptr =
      do_col ? reinterpret_cast<nvfp4_scale_t*>(output->columnwise_scale_inv.dptr) : nullptr;
  const float* row_amax_in = do_row ? reinterpret_cast<const float*>(output->amax.dptr) : nullptr;
  const float* col_amax_in =
      do_col ? reinterpret_cast<const float*>(output->columnwise_amax.dptr) : nullptr;

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      do_row, DO_ROW, TRANSFORMER_ENGINE_SWITCH_CONDITION(do_col, DO_COL, {
        auto kernel = per_token_encode_kernel<DO_ROW, DO_COL>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dshmem_size);
        kernel<<<grid, block, dshmem_size, stream>>>(tmap_in, tmap_out, tmap_out_t, scales_ptr,
                                                     scales_t_ptr, row_amax_in, col_amax_in,
                                                     noop_ptr, M, K, scale_stride, scale_stride_t);
      }););
  NVTE_CHECK_CUDA(cudaGetLastError());
}
#endif  // FP4_TYPE_SUPPORTED

// =============================================================================
// Impls (validation + dispatch). The K1 amax / K2 encode passes are exposed
// as separately callable entry points alongside the composite K1+K2 entry,
// to enable per-kernel benchmarking and diagnostic use.
// =============================================================================

#if FP4_TYPE_SUPPORTED
// Common input + shape validation, shared by all 3 entry points.
// Output constraints differ by entry point (see validate_*_output helpers below).
inline void validate_input_shape(const Tensor& input) {
  NVTE_CHECK(input.has_data(), "Per-token cast: input has no data.");
  NVTE_CHECK(input.dtype() == DType::kBFloat16, "Per-token cast is bf16-only. Got dtype enum ",
             static_cast<int>(input.dtype()));
  const size_t M = input.flat_first_dim();
  const size_t K = input.flat_last_dim();
  NVTE_CHECK(M % CHUNK_DIM_Y == 0, "Per-token cast: M must be a multiple of ", CHUNK_DIM_Y,
             ", got M=", M);
  NVTE_CHECK(K % CHUNK_DIM_X == 0, "Per-token cast: K must be a multiple of ", CHUNK_DIM_X,
             ", got K=", K);
}

// K1 (amax-only) requires at least one amax buffer allocated; FP4 output is not used.
inline void validate_amax_output(const Tensor* output) {
  NVTE_CHECK(output->amax.dptr != nullptr || output->columnwise_amax.dptr != nullptr,
             "Per-token K1 (amax): at least one of rowwise/columnwise amax buffer "
             "must be allocated.");
}

// K2 (encode) and composite require at least one FP4 output buffer allocated.
inline void validate_encode_output(const Tensor* output) {
  NVTE_CHECK(output->has_data() || output->has_columnwise_data(),
             "Per-token K2 (encode): at least one of rowwise/columnwise FP4 output "
             "must be allocated.");
  NVTE_CHECK(!output->with_gemm_swizzled_scales,
             "Per-token cast emits compact (non-swizzled) inner SF.");
}

void per_token_amax_blocked_impl(const Tensor& input, const Tensor& noop, Tensor* output,
                                 cudaStream_t stream) {
  validate_input_shape(input);
  validate_amax_output(output);
  if (input.flat_first_dim() == 0 || input.flat_last_dim() == 0) return;
  launch_amax(input, output, noop, stream);
}

void per_token_encode_blocked_impl(const Tensor& input, const Tensor& noop, Tensor* output,
                                   cudaStream_t stream) {
  validate_input_shape(input);
  validate_encode_output(output);
  if (input.flat_first_dim() == 0 || input.flat_last_dim() == 0) return;
  launch_encode(input, output, noop, stream);
}

void per_token_quantize_blocked_impl(const Tensor& input, const Tensor& noop, Tensor* output,
                                     cudaStream_t stream) {
  validate_input_shape(input);
  validate_encode_output(output);
  if (input.flat_first_dim() == 0 || input.flat_last_dim() == 0) return;
  launch_amax(input, output, noop, stream);
  launch_encode(input, output, noop, stream);
}

bool can_use_per_token(size_t M, size_t K, DType dtype) {
  return (dtype == DType::kBFloat16) && (M % CHUNK_DIM_Y == 0) && (K % CHUNK_DIM_X == 0);
}
#else   // !FP4_TYPE_SUPPORTED
void per_token_amax_blocked_impl(const Tensor&, const Tensor&, Tensor*, cudaStream_t) {
  NVTE_ERROR("NVFP4 requires SM100 (Blackwell); build with sm_100a/sm_100f.");
}
void per_token_encode_blocked_impl(const Tensor&, const Tensor&, Tensor*, cudaStream_t) {
  NVTE_ERROR("NVFP4 requires SM100 (Blackwell); build with sm_100a/sm_100f.");
}
void per_token_quantize_blocked_impl(const Tensor&, const Tensor&, Tensor*, cudaStream_t) {
  NVTE_ERROR("NVFP4 requires SM100 (Blackwell); build with sm_100a/sm_100f.");
}
bool can_use_per_token(size_t, size_t, DType) { return false; }
#endif  // FP4_TYPE_SUPPORTED

}  // namespace nvfp4_per_token
}  // namespace transformer_engine

// =============================================================================
// C-API entry points
// =============================================================================

void nvte_nvfp4_per_token_amax(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                               cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_per_token_amax);
  using namespace transformer_engine;
  const Tensor* input_tensor = convertNVTETensorCheck(input);
  Tensor* output_tensor = convertNVTETensorCheck(output);
  Tensor dummy_noop;
  const Tensor* noop_tensor = (noop != nullptr) ? convertNVTETensorCheck(noop) : &dummy_noop;
  nvfp4_per_token::per_token_amax_blocked_impl(*input_tensor, *noop_tensor, output_tensor, stream);
#else
  (void)input;
  (void)noop;
  (void)output;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

void nvte_nvfp4_per_token_encode(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                                 cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_per_token_encode);
  using namespace transformer_engine;
  const Tensor* input_tensor = convertNVTETensorCheck(input);
  Tensor* output_tensor = convertNVTETensorCheck(output);
  Tensor dummy_noop;
  const Tensor* noop_tensor = (noop != nullptr) ? convertNVTETensorCheck(noop) : &dummy_noop;
  nvfp4_per_token::per_token_encode_blocked_impl(*input_tensor, *noop_tensor, output_tensor,
                                                 stream);
#else
  (void)input;
  (void)noop;
  (void)output;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

void nvte_nvfp4_per_token_quantize(const NVTETensor input, const NVTETensor noop, NVTETensor output,
                                   cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  NVTE_API_CALL(nvte_nvfp4_per_token_quantize);
  using namespace transformer_engine;
  const Tensor* input_tensor = convertNVTETensorCheck(input);
  Tensor* output_tensor = convertNVTETensorCheck(output);
  Tensor dummy_noop;
  const Tensor* noop_tensor = (noop != nullptr) ? convertNVTETensorCheck(noop) : &dummy_noop;
  nvfp4_per_token::per_token_quantize_blocked_impl(*input_tensor, *noop_tensor, output_tensor,
                                                   stream);
#else
  (void)input;
  (void)noop;
  (void)output;
  (void)stream;
  NVTE_ERROR("FP4 support requires CUDA 12.8+, but compile-time CUDA version is ", CUDA_VERSION);
#endif
}

int nvte_nvfp4_per_token_can_dispatch(size_t M, size_t K, int input_dtype_enum) {
  using namespace transformer_engine;
  const DType dtype = static_cast<DType>(input_dtype_enum);
  return nvfp4_per_token::can_use_per_token(M, K, dtype) ? 1 : 0;
}
