/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_gated_mxfp8_regtile.cuh
 *  \brief Register-resident MXFP8 bidimensional quantize fused with SwiGLU or its gradient.
 *
 *  The gated counterpart of quantize_mxfp8_regtile.cuh. Input [M, 2K] holds the activated half
 *  and the gate half side by side. The forward writes silu(act) * gate as [M, K]; the backward
 *  writes the gradients of both halves as [M, 2K], quantized independently, one after the other
 *  from the same 32-row band. A CTA covers 256 output columns and walks 32-row bands; the band
 *  stays in registers, and both scale directions are computed from there.
 *
 *  It writes the bytes the generic gated kernel (gated_mxfp8.cuh) writes: the same math, the
 *  same rounding to BF16, and scales only inside the tensor.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_GATED_MXFP8_REGTILE_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_GATED_MXFP8_REGTILE_CUH_

#include "quantize_mxfp8_regtile.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace regtile {
namespace gated {

constexpr int kThreadsPerCta = 256;
constexpr int kWarpsPerCta = kThreadsPerCta / 32;
constexpr int kRowsPerWarp = kRowsPerMxBlock / kWarpsPerCta;
constexpr int kRowsInFlight = 2;
// Resident CTAs per SM, the second __launch_bounds__ argument (GB200, 32768x7168): the forward
// is issue-bound and gains 16% from 6 CTAs over 4; the backward, which keeps more values live,
// loses 30% at 5.
template <bool IS_BWD>
constexpr int kMinBlocksPerSm = IS_BWD ? 4 : 6;
static_assert(kVecLoadsPerRow == 1, "One 16-byte vector per row and lane.");

/*! \brief One output element of the gated activation, as the generic gated kernel computes it.
 *
 *  kPart 0: the forward output, or the gradient of the activated half. kPart 1: the gradient of
 *  the gate half. Rounded to BF16 by the caller.
 */
template <bool IS_BWD, int kPart>
__device__ __forceinline__ float gated_element(const float act_elt, const float gate_elt,
                                               const float grad_elt) {
  if constexpr (!IS_BWD) {
    return silu<float, float>(act_elt, {}) * gate_elt;
  } else {
    const float x = act_elt;
    const float s = sigmoidf(x);
    if constexpr (kPart == 0) {
      const float dact_x = x * s * (1 - s) + s;
      return dact_x * grad_elt * gate_elt;
    } else {
      const float act_x = x * s;
      const float dgate_elt = 1.0f;
      return dgate_elt * act_x * grad_elt;
    }
  }
}

/*! \brief The gated activation of one BF16 pair of each input, rounded to a BF16 pair. */
template <bool IS_BWD, int kPart>
__device__ __forceinline__ unsigned gated_word(const unsigned act, const unsigned gate,
                                               const unsigned grad) {
  const ptx::floatx2 a = widen_bf16x2(act);
  const ptx::floatx2 g = widen_bf16x2(gate);
  const ptx::floatx2 d = widen_bf16x2(grad);
  return to_bf16x2(
      {gated_element<IS_BWD, kPart>(a.x, g.x, d.x), gated_element<IS_BWD, kPart>(a.y, g.y, d.y)});
}

template <bool IS_BWD, typename OType, bool WITH_GEMM_SWIZZLED_SCALES, bool kTails>
__global__ void __launch_bounds__(kThreadsPerCta, kMinBlocksPerSm<IS_BWD>)
    quantize_gated_kernel(const uint4* __restrict__ input, const uint4* __restrict__ grad,
                          unsigned char* __restrict__ rowwise_out,
                          unsigned char* __restrict__ rowwise_scales,
                          unsigned char* __restrict__ colwise_out,
                          unsigned char* __restrict__ colwise_scales, int M, int K,
                          int rowwise_scale_stride, int colwise_scale_stride, int iters) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  __shared__ __align__(16) unsigned col_amax_partials[kWarpsPerCta][kTileCols / 2];
  __shared__ __align__(16) unsigned col_scale_rcp_smem[kTileCols / 2];
  __shared__ __align__(8) unsigned char row_scale_bytes[kRowsPerMxBlock * kMxGroupsPerRow];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col0 = blockIdx.x * kTileCols;
  // Output columns; the backward writes both halves.
  const int out_cols = IS_BWD ? 2 * K : K;
  const int k_vecs = K >> 3;  // uint4 (8 BF16 values) per K columns
  const int k_groups = (K + 31) / 32;
  const bool full_width = !kTails || col0 + kTileCols <= K;
  const bool lane_in_cols = col0 + kColsPerLane * lane < K;

#pragma unroll 1
  for (int it = 0; it < iters; ++it) {
    const int row0 = (it * gridDim.y + blockIdx.y) * kRowsPerMxBlock;
    const int warp_row0 = row0 + warp * kRowsPerWarp;
    // Quantize one output half (kPart; the forward has only part 0) of the band from its tile.
    const auto quantize_part = [&](auto edge, auto part, const unsigned* tile_words) {
      constexpr bool kEdge = decltype(edge)::value;
      constexpr int kPart = decltype(part)::value;
      const auto in_bounds = [&](int j) { return !kEdge || (lane_in_cols && warp_row0 + j < M); };

      // ---- rowwise ----------------------------------------------------------------------------
      const size_t store_offset =
          (size_t)warp_row0 * out_cols + kPart * K + col0 + kColsPerLane * lane;
      {
        unsigned char* row_out_ptr = rowwise_out + store_offset;
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; ++j) {
          const unsigned* v = tile_words + j * kWordsPerLane;
          unsigned amax = abs_max_bf16x2(abs_max_bf16x2(v[0], v[1]), abs_max_bf16x2(v[2], v[3]));
#pragma unroll
          for (int lane_mask = 1; lane_mask < kLanesPerMxGroup; lane_mask <<= 1)
            amax = abs_max_bf16x2(amax, __shfl_xor_sync(0xffffffffu, amax, lane_mask));
          amax = abs_max_bf16x2(amax, __byte_perm(amax, amax, 0x1032));
          const unsigned scale_byte =
              amax_to_e8m0_2x<OType>(0.f, __uint_as_float(amax & 0x7fff0000u));
          if ((lane & (kLanesPerMxGroup - 1)) == 0)
            row_scale_bytes[(warp * kRowsPerWarp + j) * kMxGroupsPerRow +
                            (lane / kLanesPerMxGroup)] = (unsigned char)scale_byte;
          const unsigned rcp = e8m0_to_bf16x2_reciprocal(scale_byte);
          unsigned row_out[kWordsPerLane / 2];
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            row_out[m] = quantize_x4<OType>(v[2 * m], v[2 * m + 1], rcp, rcp);
          if (in_bounds(j)) store_quantized(row_out_ptr + (size_t)j * out_cols, row_out);
        }
      }

      // ---- colwise amax: per-warp partial in registers, cross-warp fold in shared ---------------
      {
        unsigned col_amax[kWordsPerLane];
#pragma unroll
        for (int m = 0; m < kWordsPerLane; ++m) col_amax[m] = tile_words[m];
#pragma unroll
        for (int j = 1; j < kRowsPerWarp; ++j)
#pragma unroll
          for (int m = 0; m < kWordsPerLane; ++m)
            col_amax[m] = abs_max_bf16x2(col_amax[m], tile_words[j * kWordsPerLane + m]);
        *(uint4*)(&col_amax_partials[warp][kWordsPerLane * lane]) = *(const uint4*)col_amax;
      }
      __syncthreads();
      if (tid < kTileCols / 2) {
        unsigned col_amax = col_amax_partials[0][tid];
#pragma unroll
        for (int w = 1; w < kWarpsPerCta; ++w)
          col_amax = abs_max_bf16x2(col_amax, col_amax_partials[w][tid]);
        col_amax &= 0x7fff7fffu;
        const ptx::floatx2 amax_2x = widen_bf16x2(col_amax);
        const unsigned scale_pair = amax_to_e8m0_2x<OType>(amax_2x.y, amax_2x.x);
        const int col = col0 + 2 * tid;
        // The generic gated kernel writes scales only inside the tensor.
        if (!kEdge || (col < K && row0 < M)) {
          if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
            const size_t row_block_tiles = DIVUP(static_cast<size_t>(M), size_t{128});
            colwise_scales[swizzle::gemm_swizzled_scale_idx(kPart * K + col, row0 / 32,
                                                            row_block_tiles)] =
                static_cast<unsigned char>(scale_pair);
            colwise_scales[swizzle::gemm_swizzled_scale_idx(kPart * K + col + 1, row0 / 32,
                                                            row_block_tiles)] =
                static_cast<unsigned char>(scale_pair >> 8);
          } else {
            *(unsigned short*)(colwise_scales + (size_t)(row0 / 32) * colwise_scale_stride +
                               kPart * K + col) = (unsigned short)scale_pair;
          }
        }
        const ptx::bf16x2 scale_rcp =
            ptx::exp2f_rcp_2x_per_lane(__byte_perm(scale_pair, 0, 0x4140));
        col_scale_rcp_smem[tid] = reinterpret_cast<const unsigned&>(scale_rcp);
      }
      // Drain the row scales: one thread per row, one byte per group inside the tensor.
      if (tid >= kTileCols / 2 && tid < kTileCols / 2 + kRowsPerMxBlock) {
        const int r = tid - kTileCols / 2;
        if (!kEdge || row0 + r < M) {
          const int groups = full_width ? kMxGroupsPerRow : (K - col0 + 31) / 32;
          const size_t col_tiles = DIVUP(static_cast<size_t>(out_cols), size_t{128});
          for (int g = 0; g < groups; ++g) {
            const int group = kPart * k_groups + (col0 >> 5) + g;
            size_t idx;
            if constexpr (WITH_GEMM_SWIZZLED_SCALES) {
              idx = swizzle::gemm_swizzled_scale_idx(row0 + r, group, col_tiles);
            } else {
              idx = (size_t)(row0 + r) * rowwise_scale_stride + group;
            }
            rowwise_scales[idx] = row_scale_bytes[r * kMxGroupsPerRow + g];
          }
        }
      }
      __syncthreads();

      // ---- colwise quantization ---------------------------------------------------------------
      {
        const uint4 rcp_vec = *(const uint4*)(col_scale_rcp_smem + kWordsPerLane * lane);
        const unsigned* rcp = (const unsigned*)&rcp_vec;
        unsigned char* col_out_ptr = colwise_out + store_offset;
#pragma unroll
        for (int j = 0; j < kRowsPerWarp; ++j) {
          const unsigned* v = tile_words + j * kWordsPerLane;
          unsigned col_out[kWordsPerLane / 2];
#pragma unroll
          for (int m = 0; m < kWordsPerLane / 2; ++m)
            col_out[m] = quantize_x4<OType>(v[2 * m], v[2 * m + 1], rcp[2 * m], rcp[2 * m + 1]);
          if (in_bounds(j)) store_quantized(col_out_ptr + (size_t)j * out_cols, col_out);
        }
      }
    };
    // Load the band once and compute its tiles: one for the forward, both gradient halves for the
    // backward, which then quantizes them one after the other.
    const auto parts = [&](auto edge) {
      constexpr bool kEdge = decltype(edge)::value;
      const auto in_bounds = [&](int j) { return !kEdge || (lane_in_cols && warp_row0 + j < M); };
      unsigned tile0[kRowsPerWarp * kWordsPerLane];
      unsigned tile1[IS_BWD ? kRowsPerWarp * kWordsPerLane : 1];
      const size_t in_offset = (size_t)warp_row0 * (2 * k_vecs) + (col0 >> 3) + lane;
      const uint4* act_ptr = input + in_offset;
      const uint4* gate_ptr = act_ptr + k_vecs;
      const uint4* grad_ptr = grad + (size_t)warp_row0 * k_vecs + (col0 >> 3) + lane;
#pragma unroll
      for (int h = 0; h < kRowsPerWarp / kRowsInFlight; ++h) {
        uint4 act_vec[kRowsInFlight], gate_vec[kRowsInFlight], grad_vec[kRowsInFlight];
        asm volatile("" ::: "memory");
#pragma unroll
        for (int t = 0; t < kRowsInFlight; ++t) {
          const int j = h * kRowsInFlight + t;
          const uint4 zero = make_uint4(0, 0, 0, 0);
          act_vec[t] = in_bounds(j) ? act_ptr[(size_t)j * (2 * k_vecs)] : zero;
          gate_vec[t] = in_bounds(j) ? gate_ptr[(size_t)j * (2 * k_vecs)] : zero;
          grad_vec[t] = IS_BWD && in_bounds(j) ? grad_ptr[(size_t)j * k_vecs] : zero;
        }
        asm volatile("" ::: "memory");
#pragma unroll
        for (int t = 0; t < kRowsInFlight; ++t) {
          const unsigned* a = (const unsigned*)&act_vec[t];
          const unsigned* g = (const unsigned*)&gate_vec[t];
          const unsigned* d = (const unsigned*)&grad_vec[t];
#pragma unroll
          for (int m = 0; m < 4; ++m) {
            const int w = (h * kRowsInFlight + t) * kWordsPerLane + m;
            tile0[w] = gated_word<IS_BWD, 0>(a[m], g[m], d[m]);
            if constexpr (IS_BWD) tile1[w] = gated_word<IS_BWD, 1>(a[m], g[m], d[m]);
          }
        }
      }
      quantize_part(edge, std::integral_constant<int, 0>{}, tile0);
      if constexpr (IS_BWD) quantize_part(edge, std::integral_constant<int, 1>{}, tile1);
    };
    if constexpr (kTails) {
      if (full_width && row0 + kRowsPerMxBlock <= M) {
        parts(std::false_type{});
      } else {
        parts(std::true_type{});
      }
    } else {
      parts(std::false_type{});
    }
  }
#endif
}

//! Whether the gated regtile kernel takes this request: BF16 [M, 2K] -> E4M3 or E5M2, both directions,
//! K % 32 == 0 (the generic gated kernel's requirement), aligned buffers.
inline bool can_use(const Tensor& input, const Tensor& grad, const Tensor& output,
                    const bool is_bwd) {
  if (!getenv<bool>("NVTE_MXFP8_REGTILE", true)) return false;
  if (transformer_engine::cuda::sm_arch() < 100) return false;
  if (input.dtype() != DType::kBFloat16 ||
      (output.dtype() != DType::kFloat8E4M3 && output.dtype() != DType::kFloat8E5M2)) {
    return false;
  }
  if (!output.has_data() || !output.has_columnwise_data()) return false;
  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim() / 2;
  constexpr size_t kMaxRows = std::numeric_limits<int>::max() - 64;
  if (rows == 0 || cols == 0 || cols % 32 != 0 || rows > kMaxRows ||
      2 * cols > std::numeric_limits<unsigned>::max() / kRowsPerMxBlock) {
    return false;
  }
  return is_aligned_ptr(input.data.dptr, 16) && (!is_bwd || is_aligned_ptr(grad.data.dptr, 16)) &&
         is_aligned_ptr(output.data.dptr, 8) && is_aligned_ptr(output.columnwise_data.dptr, 8) &&
         is_aligned_ptr(output.columnwise_scale_inv.dptr, 2);
}

template <bool IS_BWD, typename OType, bool WITH_GEMM_SWIZZLED_SCALES>
void quantize(const Tensor& input, const Tensor& grad, Tensor* output, cudaStream_t stream) {
  const int M = static_cast<int>(input.flat_first_dim());
  const int K = static_cast<int>(input.flat_last_dim() / 2);
  const int bands = (M + kRowsPerMxBlock - 1) / kRowsPerMxBlock;
  const int grid_cols = (K + kTileCols - 1) / kTileCols;
  int min_walk = kMinWalk;
  if ((long long)grid_cols * (bands / min_walk) < 2LL * cuda::sm_count()) min_walk = 1;
  int iters = pick_walk_length(bands, grid_cols, kCtaTargetActivation, min_walk);
  if (iters > kMaxWalkDact) iters = kMaxWalkDact;
  const dim3 grid(grid_cols, bands / iters);
  const auto run = [&](auto tails) {
    constexpr bool kTails = decltype(tails)::value;
    quantize_gated_kernel<IS_BWD, OType, WITH_GEMM_SWIZZLED_SCALES, kTails>
        <<<grid, kThreadsPerCta, 0, stream>>>(
            (const uint4*)input.data.dptr, (const uint4*)(IS_BWD ? grad.data.dptr : nullptr),
            (unsigned char*)output->data.dptr, (unsigned char*)output->scale_inv.dptr,
            (unsigned char*)output->columnwise_data.dptr,
            (unsigned char*)output->columnwise_scale_inv.dptr, M, K,
            static_cast<int>(output->scale_inv.shape[1]),
            static_cast<int>(output->columnwise_scale_inv.shape[1]), iters);
  };
  if (M % kRowsPerMxBlock == 0 && K % kTileCols == 0) {
    run(std::false_type{});
  } else {
    run(std::true_type{});
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace gated
}  // namespace regtile
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_GATED_MXFP8_REGTILE_CUH_
