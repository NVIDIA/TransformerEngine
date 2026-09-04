/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_rowwise.cu
 *  \brief Register-resident rowwise MXFP8 quantization kernel.
 */

#include <cuda_runtime.h>

#include "../../../common.h"
#include "../../../util/ptx.cuh"
#include "../../../util/ptx_arch_spec.cuh"
#include "../../../utils.cuh"
#include "cast_rowwise.h"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

namespace ptx = transformer_engine::ptx;

// This kernel casts BF16 to rowwise-scaled MXFP8: every run of 32 consecutive
// elements within a row forms one MX block that shares a single E8M0 scale,
// chosen so the block's largest magnitude lands at the top of the FP8E4M3
// range.
//
// Unlike the TMA-based kernel in mxfp8/specialized, this one keeps its tile
// entirely in registers.  There is no shared memory, no barrier, and no
// two-dimensional tiling: for a row-major tensor whose scale array is also
// contiguous, MX blocks never straddle a row boundary, so the whole tensor is
// just a flat sequence of M*(K/32) independent blocks.  That reduces the
// kernel to a pure streaming problem, and what is left to tune is how the
// input and output streams share L2.
//
// A tensor whose scale array is padded (scale_stride > K/32) breaks the flat
// view; those shapes go to quantize_strided_kernel below.

// Elements in one MX block, all sharing a single E8M0 scale.
constexpr int32_t kBlockElems = 32;

// Two lanes cooperate on each MX block.  A lane's half of a block is 16 BF16
// values = 32 bytes = one 256-bit load, the widest the ISA offers; splitting
// the block any further would waste load width, and any less would exceed it.
constexpr int32_t kLanesPerBlock = 2;
constexpr int32_t kElemsPerLane = kBlockElems / kLanesPerBlock;

// Both tensors are addressed as 32-bit words: BF16 packs 2 elements per word,
// FP8 packs 4.  All the packed-math PTX below operates on those words.
constexpr int32_t kInElemsPerWord = sizeof(uint32_t) / sizeof(bf16);
// Every MXFP8 output type is a single byte, so a 32-bit word holds four.
constexpr int32_t kOutElemsPerWord = 4;

constexpr int32_t kInWordsPerLane = kElemsPerLane / kInElemsPerWord;    // 8 -> 256-bit load
constexpr int32_t kOutWordsPerLane = kElemsPerLane / kOutElemsPerWord;  // 4 -> 128-bit store
constexpr int32_t kInWordsPerBlock = kBlockElems / kInElemsPerWord;     // 16
constexpr int32_t kOutWordsPerBlock = kBlockElems / kOutElemsPerWord;   // 8

// MX blocks a single warp covers in one pass over its registers.
constexpr int32_t kBlocksPerWarp = THREADS_PER_WARP / kLanesPerBlock;  // 16

// The block-wide maximum is formed with a single shuffle that swaps a lane
// with its odd/even partner, which only covers a two-lane group.
static_assert(kLanesPerBlock == 2, "A wider lane group would need a multi-step reduction.");

/*! \brief Launch parameters for one tensor-size regime.
 *
 * The kernel is bandwidth-bound, so the best configuration tracks how the
 * working set compares with L2 rather than the shape itself.  These were
 * selected by autotuning over a B200 shape sweep; see kTierMaxBytes below for
 * the one threshold that has since been re-measured.
 */
struct LaunchConfig {
  //! CTA width.  Trades occupancy against per-CTA scheduling overhead.
  int32_t threads_per_cta;
  //! MX blocks each lane pair handles per launch.  Raising this unrolls the
  //! body, giving more independent loads in flight at the cost of registers.
  int32_t blocks_per_lane;
  //! Percentage of CTAs that let their input settle in L2 normally; the
  //! remainder tag their loads evict_first so the data streams past without
  //! displacing anything.  0 streams the entire input.
  //!
  //! Streaming everything is right once the input dwarfs L2, since nothing
  //! would survive to be reused anyway.  When the input is only a few times
  //! L2, holding part of it back leaves capacity for the output write-back
  //! instead of thrashing on input lines.
  int32_t l2_cached_cta_percent;
};

// Output bytes (one FP8 byte per element, i.e. M*K) separating the regimes.
//
// The first threshold is 12 MiB rather than the 24 MiB the original sweep
// picked.  The single-block-per-lane configuration of tier 0 stops paying off
// well before 24 MiB: measured on B200, a 16 MiB tensor ran 8.29 us on tier 0
// against 6.78 us on tier 1, and a 24 MiB tensor 11.58 us against 9.82 us.
// Both were also slower than the TMA kernel this one replaces, so the tier 0
// range is cut where the crossover actually lies.
constexpr int64_t kTierMaxBytes[] = {12ll << 20, 48ll << 20, 96ll << 20};

constexpr LaunchConfig kTierConfigs[] = {
    {/*threads_per_cta=*/256, /*blocks_per_lane=*/1, /*l2_cached_cta_percent=*/0},
    {/*threads_per_cta=*/256, /*blocks_per_lane=*/2, /*l2_cached_cta_percent=*/0},
    {/*threads_per_cta=*/128, /*blocks_per_lane=*/2, /*l2_cached_cta_percent=*/40},
    {/*threads_per_cta=*/256, /*blocks_per_lane=*/2, /*l2_cached_cta_percent=*/40},
};
constexpr int32_t kNumTiers = sizeof(kTierConfigs) / sizeof(kTierConfigs[0]);
static_assert(kNumTiers == sizeof(kTierMaxBytes) / sizeof(kTierMaxBytes[0]) + 1,
              "Each size threshold must separate two tiers.");

namespace {

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

/*! \brief Reduce eight BF16 pairs to the largest magnitude among them. */
__device__ __forceinline__ ptx::bf16x2 block_half_amax(const uint32_t (&words)[kInWordsPerLane]) {
  const ptx::bf16x2 *pairs = reinterpret_cast<const ptx::bf16x2 *>(words);

  // Balanced tree: depth 3 instead of the 7 of a serial chain, so the
  // independent maxima issue back to back.
  ptx::bf16x2 level[kInWordsPerLane / 2];
#pragma unroll
  for (int32_t i = 0; i < kInWordsPerLane / 2; ++i) {
    ptx::abs_max_2x(level[i], pairs[2 * i], pairs[2 * i + 1]);
  }
#pragma unroll
  for (int32_t i = 0; i < kInWordsPerLane / 4; ++i) {
    ptx::abs_max_2x(level[i], level[i], level[i + kInWordsPerLane / 4]);
  }
  ptx::bf16x2 result;
  ptx::abs_max_2x(result, level[0], level[1]);
  return result;
}

/*! \brief Widen a BF16 pair's larger magnitude to FP32.
 *
 * `max.xorsign.abs` keeps the magnitude of the larger operand but sets the
 * result sign to the XOR of the input signs, so an accumulator built from it
 * can come out negative.  Only the magnitude means anything for a scale, and
 * feeding a negative value to the unsigned E8M0 conversion would saturate it
 * to zero, so the sign is cleared here.
 */
__device__ __forceinline__ float pair_amax_to_float(ptx::bf16x2 pair) {
  const uint32_t bits = reinterpret_cast<const uint32_t &>(pair);
  // Fold the two halves against each other, then keep the low BF16 sans sign.
  const uint32_t folded = __byte_perm(bits, bits, 0x1032);
  ptx::bf16x2 a, b;
  reinterpret_cast<uint32_t &>(a) = bits;
  reinterpret_cast<uint32_t &>(b) = folded;
  ptx::bf16x2 wide;
  ptx::abs_max_2x(wide, a, b);
  const uint32_t magnitude = reinterpret_cast<const uint32_t &>(wide) & 0x7FFFu;
  return __int_as_float(magnitude << 16);
}

/*! \brief Scale and convert one lane's 16 BF16 values into 16 FP8E4M3 bytes. */
template <typename OType>
__device__ __forceinline__ void scale_and_convert(const uint32_t (&in)[kInWordsPerLane],
                                                  ptx::bf16x2 scale_reciprocal,
                                                  uint32_t (&out)[kOutWordsPerLane]) {
#pragma unroll
  for (int32_t i = 0; i < kOutWordsPerLane; ++i) {
    ptx::mul_cvt_4x(reinterpret_cast<ptx::FPx4<OType> &>(out[i]),
                    reinterpret_cast<const ptx::bf16x4 &>(in[2 * i]), scale_reciprocal);
  }
}

/*! \brief Quantize one whole MX block held by a single thread.
 *
 * Used by the remainder and strided kernels, which handle far too little data
 * to be worth the two-lane split of the main kernel.
 */
template <typename OType>
__device__ __forceinline__ void quantize_one_block(const uint32_t *__restrict__ in,
                                                   uint32_t *__restrict__ out,
                                                   e8m0_t *__restrict__ scale,
                                                   uint64_t output_policy) {
  uint32_t words[kInWordsPerBlock];
  ptx::ld_global_nc_b32x8(reinterpret_cast<uint32_t(&)[8]>(words[0]), in);
  ptx::ld_global_nc_b32x8(reinterpret_cast<uint32_t(&)[8]>(words[kInWordsPerLane]),
                          in + kInWordsPerLane);

  ptx::bf16x2 amax_lo = block_half_amax(reinterpret_cast<const uint32_t(&)[8]>(words[0]));
  ptx::bf16x2 amax_hi =
      block_half_amax(reinterpret_cast<const uint32_t(&)[8]>(words[kInWordsPerLane]));
  ptx::bf16x2 amax_pair;
  ptx::abs_max_2x(amax_pair, amax_lo, amax_hi);

  const float amax = pair_amax_to_float(amax_pair);
  const e8m0_t biased_exponent = ptx::float_to_e8m0(amax * Quantized_Limits<OType>::max_norm_rcp);
  *scale = biased_exponent;

  const ptx::bf16x2 scale_reciprocal = ptx::exp2f_rcp_2x(biased_exponent);
  uint32_t out_words[kOutWordsPerBlock];
  scale_and_convert<OType>(reinterpret_cast<const uint32_t(&)[8]>(words[0]), scale_reciprocal,
                           reinterpret_cast<uint32_t(&)[4]>(out_words[0]));
  scale_and_convert<OType>(reinterpret_cast<const uint32_t(&)[8]>(words[kInWordsPerLane]),
                           scale_reciprocal,
                           reinterpret_cast<uint32_t(&)[4]>(out_words[kOutWordsPerLane]));

  ptx::st_global_b32x4(out, reinterpret_cast<const uint32_t(&)[4]>(out_words[0]), output_policy);
  ptx::st_global_b32x4(out + kOutWordsPerLane,
                       reinterpret_cast<const uint32_t(&)[4]>(out_words[kOutWordsPerLane]),
                       output_policy);
}

#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

}  // namespace

/*! \brief Quantize a contiguous run of MX blocks, two lanes per block.
 *
 * \tparam PARTIAL_L2_CACHING  When false every CTA streams its input, which
 *                             the ISA expresses as a static load modifier and
 *                             so costs no policy register.  When true the
 *                             decision varies per CTA and needs a runtime
 *                             policy token.  See LaunchConfig.
 *
 * \param[in]  input                BF16 input, viewed as 32-bit words.
 * \param[out] output               FP8E4M3 output, viewed as 32-bit words.
 * \param[out] scales               One E8M0 byte per MX block.
 * \param[in]  first_streaming_cta  CTAs at or above this index stream their
 *                                  input; earlier ones cache normally.  Only
 *                                  read when PARTIAL_L2_CACHING is true.
 */
template <typename OType, int32_t THREADS_PER_CTA, int32_t BLOCKS_PER_LANE, bool PARTIAL_L2_CACHING>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    quantize_contiguous_kernel(const uint32_t *__restrict__ input, uint32_t *__restrict__ output,
                               e8m0_t *__restrict__ scales, uint32_t first_streaming_cta) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr int32_t kWarpsPerCta = THREADS_PER_CTA / THREADS_PER_WARP;
  constexpr int32_t kBlocksPerWarpPass = kBlocksPerWarp * BLOCKS_PER_LANE;

  const int32_t lane = threadIdx.x % THREADS_PER_WARP;
  const int64_t warp_id =
      static_cast<int64_t>(blockIdx.x) * kWarpsPerCta + threadIdx.x / THREADS_PER_WARP;
  const int64_t first_block = warp_id * kBlocksPerWarpPass;

  // Output is marked evict_last so its lines linger long enough to coalesce on
  // write-back.
  const uint64_t output_policy = ptx::create_l2_policy_evict_last();

  // Lanes pair up as (even, odd); the even lane of each pair owns the scale.
  const int32_t block_in_warp = lane / kLanesPerBlock;
  const bool owns_scale = (lane % kLanesPerBlock) == 0;

  uint32_t in_words[BLOCKS_PER_LANE][kInWordsPerLane];
  if constexpr (PARTIAL_L2_CACHING) {
    const uint64_t input_policy =
        ptx::create_l2_policy_evict_first(blockIdx.x >= first_streaming_cta ? 1.0f : 0.0f);
#pragma unroll
    for (int32_t u = 0; u < BLOCKS_PER_LANE; ++u) {
      const int64_t group_base = first_block + static_cast<int64_t>(u) * kBlocksPerWarp;
      ptx::ld_global_nc_b32x8(in_words[u],
                              input + group_base * kInWordsPerBlock + lane * kInWordsPerLane,
                              input_policy);
    }
  } else {
#pragma unroll
    for (int32_t u = 0; u < BLOCKS_PER_LANE; ++u) {
      const int64_t group_base = first_block + static_cast<int64_t>(u) * kBlocksPerWarp;
      ptx::ld_global_nc_evict_first_b32x8(
          in_words[u], input + group_base * kInWordsPerBlock + lane * kInWordsPerLane);
    }
  }

#pragma unroll
  for (int32_t u = 0; u < BLOCKS_PER_LANE; ++u) {
    const int64_t group_base = first_block + static_cast<int64_t>(u) * kBlocksPerWarp;

    // Each lane reduces its own half, then swaps with its partner so both
    // arrive at the block-wide maximum.
    ptx::bf16x2 half_amax = block_half_amax(in_words[u]);
    ptx::bf16x2 partner;
    reinterpret_cast<uint32_t &>(partner) =
        __shfl_xor_sync(0xFFFFFFFFu, reinterpret_cast<const uint32_t &>(half_amax), /*laneMask=*/1);
    ptx::bf16x2 block_amax;
    ptx::abs_max_2x(block_amax, half_amax, partner);

    const e8m0_t biased_exponent =
        ptx::float_to_e8m0(pair_amax_to_float(block_amax) * Quantized_Limits<OType>::max_norm_rcp);
    if (owns_scale) {
      scales[group_base + block_in_warp] = biased_exponent;
    }

    uint32_t out_words[kOutWordsPerLane];
    scale_and_convert<OType>(in_words[u], ptx::exp2f_rcp_2x(biased_exponent), out_words);
    ptx::st_global_b32x4(output + group_base * kOutWordsPerBlock + lane * kOutWordsPerLane,
                         out_words, output_policy);
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

/*! \brief Quantize the MX blocks left over when the block count does not
 *         divide evenly among the main kernel's CTAs.  One block per thread. */
template <typename OType>
__global__ void __launch_bounds__(128)
    quantize_remainder_kernel(const uint32_t *__restrict__ input, uint32_t *__restrict__ output,
                              e8m0_t *__restrict__ scales, int64_t first_block,
                              int64_t num_blocks) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int64_t block = first_block + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= num_blocks) {
    return;
  }
  quantize_one_block<OType>(input + block * kInWordsPerBlock, output + block * kOutWordsPerBlock,
                            scales + block, ptx::create_l2_policy_evict_last());
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

/*! \brief Quantize a tensor whose scale rows are padded.
 *
 * With scale_stride > K/32 the scale array is no longer a flat image of the
 * block sequence, so blocks are indexed two-dimensionally.  One block per
 * thread; grid.y walks the rows.
 */
template <typename OType>
__global__ void __launch_bounds__(128)
    quantize_strided_kernel(const uint32_t *__restrict__ input, uint32_t *__restrict__ output,
                            e8m0_t *__restrict__ scales, int32_t blocks_per_row,
                            int32_t scale_stride) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int32_t block_in_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_in_row >= blocks_per_row) {
    return;
  }
  const int64_t block = static_cast<int64_t>(blockIdx.y) * blocks_per_row + block_in_row;
  quantize_one_block<OType>(input + block * kInWordsPerBlock, output + block * kOutWordsPerBlock,
                            scales + static_cast<int64_t>(blockIdx.y) * scale_stride + block_in_row,
                            ptx::create_l2_policy_evict_last());
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

namespace {

//! Threads per CTA for the two block-per-thread helper kernels.
constexpr int32_t kHelperThreads = 128;

/*! \brief Launch quantize_contiguous_kernel for a configuration resolved at
 *         run time, instantiating only the combinations the tier table uses. */
template <typename OType>
void launch_contiguous(const LaunchConfig &config, int64_t grid, uint32_t first_streaming_cta,
                       const uint32_t *input, uint32_t *output, e8m0_t *scales,
                       cudaStream_t stream) {
  const dim3 blocks(static_cast<unsigned>(grid));
  const dim3 threads(static_cast<unsigned>(config.threads_per_cta));

  const bool partial = config.l2_cached_cta_percent != 0;

  if (config.threads_per_cta == 256 && config.blocks_per_lane == 1 && !partial) {
    quantize_contiguous_kernel<OType, 256, 1, false>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta);
  } else if (config.threads_per_cta == 256 && config.blocks_per_lane == 2 && !partial) {
    quantize_contiguous_kernel<OType, 256, 2, false>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta);
  } else if (config.threads_per_cta == 128 && config.blocks_per_lane == 2 && partial) {
    quantize_contiguous_kernel<OType, 128, 2, true>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta);
  } else if (config.threads_per_cta == 256 && config.blocks_per_lane == 2 && partial) {
    quantize_contiguous_kernel<OType, 256, 2, true>
        <<<blocks, threads, 0, stream>>>(input, output, scales, first_streaming_cta);
  } else {
    NVTE_ERROR("No quantize_contiguous_kernel instantiation for ", config.threads_per_cta,
               " threads, ", config.blocks_per_lane, " blocks per lane, partial L2 caching ",
               partial, ".");
  }
}

}  // namespace

template <typename OType>
void launch_cast_rowwise(const void *input, void *output, void *scales, int rows, int cols,
                         int scale_stride, cudaStream_t stream) {
  NVTE_CHECK(cols % kBlockElems == 0, "Rowwise MXFP8 requires the column count (", cols,
             ") to be a multiple of the MX block size (", kBlockElems, ").");

  const int32_t blocks_per_row = cols / kBlockElems;
  const uint32_t *in = reinterpret_cast<const uint32_t *>(input);
  uint32_t *out = reinterpret_cast<uint32_t *>(output);
  e8m0_t *scale_out = reinterpret_cast<e8m0_t *>(scales);

  // A padded scale array breaks the flat block view the fast path relies on.
  if (scale_stride != blocks_per_row) {
    const dim3 grid(DIVUP(blocks_per_row, kHelperThreads), rows);
    quantize_strided_kernel<OType>
        <<<grid, kHelperThreads, 0, stream>>>(in, out, scale_out, blocks_per_row, scale_stride);
    NVTE_CHECK_CUDA(cudaGetLastError());
    return;
  }

  const int64_t num_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  const int64_t output_bytes = static_cast<int64_t>(rows) * cols;

  int32_t tier = 0;
  while (tier < kNumTiers - 1 && output_bytes > kTierMaxBytes[tier]) {
    ++tier;
  }
  const LaunchConfig config = kTierConfigs[tier];

  // Every CTA covers a whole number of MX blocks; the leftovers, if any, go to
  // the remainder kernel rather than costing the main kernel a bounds check.
  const int64_t blocks_per_cta =
      static_cast<int64_t>(config.threads_per_cta) / kLanesPerBlock * config.blocks_per_lane;
  const int64_t grid = num_blocks / blocks_per_cta;

  if (grid > 0) {
    const uint32_t first_streaming_cta =
        static_cast<uint32_t>(grid * config.l2_cached_cta_percent / 100);
    launch_contiguous<OType>(config, grid, first_streaming_cta, in, out, scale_out, stream);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }

  const int64_t blocks_done = grid * blocks_per_cta;
  if (blocks_done < num_blocks) {
    const int64_t remaining = num_blocks - blocks_done;
    quantize_remainder_kernel<OType>
        <<<DIVUP(remaining, static_cast<int64_t>(kHelperThreads)), kHelperThreads, 0, stream>>>(
            in, out, scale_out, blocks_done, num_blocks);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

// The MXFP8 output types the specialized dispatch can reach; see hasSpec.
template void launch_cast_rowwise<fp8e4m3>(const void *, void *, void *, int, int, int,
                                           cudaStream_t);
template void launch_cast_rowwise<fp8e5m2>(const void *, void *, void *, int, int, int,
                                           cudaStream_t);

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine
