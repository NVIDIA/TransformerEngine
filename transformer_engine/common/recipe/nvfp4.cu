/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <cassert>
#include <limits>

#include "../common.h"
#include "../utils.cuh"
#include "../util/ptx.cuh"

namespace transformer_engine {
namespace nvfp4_recipe {

/*
 * ---------------------------------------------------------------------------
 * NVFP4 2D PARTIAL-SHARD KERNEL DESIGN
 *
 * These kernels mirror the FP8 block-scaling helpers but operate on shard-local
 * slices and nibble-packed FP4 rowwise buffers. One CUDA block covers a logical
 * 16x16 tile (grid = ceil(W/16) x ceil(H/16), blockDim = 256 threads).
 *
 * 1) Partial Amax (`nvfp4_2d_compute_partial_amax_kernel`)
 *    - Warps sweep the tile using nested loops, accumulating local maxima only
 *      for elements in [start_offset, start_offset + len).
 *    - Shared memory reduces the 8 warp maxima; the block writes a float into
 *      `amax_ptr[tile_row * stride_h + tile_col * stride_w]`.
 *
 *      Tile/warp mapping (each '#' = elements visited by that warp):
 *
 *          +------------------+
 *          |########..........|  Warp 0
 *          |########..........|  Warp 1
 *          |   ...            |
 *          |########..........|  Warp 7
 *          +------------------+
 *
 * 2) Partial Cast (`nvfp4_2d_partial_cast_kernel`)
 *    - Stage the tile into shared memory (same pattern as FP8).
 *    - For each 4-value group, build float2 pairs and call
 *      `ptx::mul_cvt_fp32_to_fp4_4x`, producing packed FP4 nibbles.
 *    - Compute a shard-local byte index and update only the owned nibble(s)
 *      using read-modify-write:
 *
 *          packed_bits = [mw3 | mw2 | mw1 | mw0]
 *          byte_idx    = (ref_elem_idx - start_offset) >> 1
 *          if elem_idx % 2 == 0:  // low nibble
 *              byte = (byte & 0xF0) | nibble
 *          else:                  // high nibble
 *              byte = (byte & 0x0F) | (nibble << 4)
 *
 *      Thread coverage inside a tile:
 *
 *          rows:   16           columns: 16
 *          Warp 0 -> rows 0-1   lanes sweep cols 0..3, 4..7, ...
 *          Warp 1 -> rows 2-3   (groups of 4 elements per thread)
 *          ...
 *          Warp 7 -> rows 14-15
 *
 * The host helper `_cast_master_weights_to_nvfp4_2d` reduces per-tile amax
 * values, packs the resulting FP32 scales into the uint8 `_rowwise_scale_inv`,
 * and launches `tex.nvfp4_2d_partial_cast`. The resulting bytes match TE’s full
 * NVFP4 quantizer, so downstream GEMMs/checkpoints remain unchanged.
 * ---------------------------------------------------------------------------
 */

// constexpr float factor = 6.0 * 6.0 * 448.0 * 448.0;
constexpr float factor_inv = 1.0 / (6.0 * 6.0 * 448.0 * 448.0);
constexpr int kTileDim = 16;
constexpr int kThreadsPerBlock = 256;

// Kernel to compute alpha *= amax_A * amax_B / factor
__global__ void compute_nvfp4_per_tensor_scale_kernel(float alpha_in, const float *amax_A,
                                                      const float *amax_B, float *alpha_out) {
  // factor is defined in the enclosing namespace
  *alpha_out = alpha_in * (*amax_A) * (*amax_B) * factor_inv;
}

template <typename IType>
__global__ void __launch_bounds__(kThreadsPerBlock)
    nvfp4_2d_compute_partial_amax_kernel(const IType *input, float *amax_ptr,
                                         const size_t amax_stride_h, const size_t amax_stride_w,
                                         const size_t h, const size_t w, const size_t start_offset,
                                         const size_t len) {
  constexpr int kThreadsPerWarp = 32;
  constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;
  static_assert(kTileDim * kTileDim == kThreadsPerBlock);

  const size_t tile_col = blockIdx.x;
  const size_t tile_row = blockIdx.y;
  const size_t end_offset = start_offset + len;
  const IType *input_minus_offset = input - start_offset;

  __shared__ float smem[kNumWarps];
  float amax = 0.0f;

  size_t r = tile_row * kTileDim + threadIdx.x / kTileDim;
  size_t c = tile_col * kTileDim + threadIdx.x % kTileDim;
  size_t idx = r * w + c;
  if (r < h && c < w && idx >= start_offset && idx < end_offset) {
    amax = fabs(static_cast<float>(input_minus_offset[idx]));
  }

  for (int delta = kThreadsPerWarp / 2; delta > 0; delta /= 2) {
    float other_amax = __shfl_down_sync(0xFFFFFFFF, amax, delta);
    __builtin_assume(amax >= 0);
    __builtin_assume(other_amax >= 0);
    amax = fmaxf(amax, other_amax);
  }

  if (threadIdx.x % kThreadsPerWarp == 0) {
    smem[threadIdx.x / kThreadsPerWarp] = amax;
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 0; i < kNumWarps; ++i) {
      float other_amax = smem[i];
      __builtin_assume(amax >= 0);
      __builtin_assume(other_amax >= 0);
      amax = fmaxf(amax, other_amax);
    }
    amax_ptr[tile_row * amax_stride_h + tile_col * amax_stride_w] = amax;
  }
}

template <typename IType, bool kWidthAligned>
__global__ void __launch_bounds__(kThreadsPerBlock)
    nvfp4_2d_partial_cast_kernel(const IType *input, uint8_t *output, const float *decode_scale_ptr,
                                 const size_t scale_stride_h, const size_t scale_stride_w,
                                 const float *global_scale_ptr, const size_t h, const size_t w,
                                 const size_t start_offset, const size_t len) {
  constexpr int kNumOutputElemsPerBank = 4;
  constexpr int kThreadsPerWarp = 32;
  constexpr int kLoopsPerRow = (kTileDim + kThreadsPerWarp - 1) / kThreadsPerWarp;
  constexpr int kNumWarps = kThreadsPerBlock / kThreadsPerWarp;
  constexpr int kRowsPerWarp = (kTileDim + kNumWarps - 1) / kNumWarps;

  __shared__ float smem[kTileDim][kTileDim + kNumOutputElemsPerBank];

  const int tile_w = blockIdx.x;
  const int tile_h = blockIdx.y;
  const size_t shard_end = start_offset + len;
  const IType *input_minus_offset = input - start_offset;

  float global_encode_scale = global_scale_ptr[0];
  if (global_encode_scale <= 0.f) {
    global_encode_scale = 1.f;
  }
  const float global_decode_scale = 1.0f / global_encode_scale;

  float tile_decode_scale =
      decode_scale_ptr[tile_h * scale_stride_h + tile_w * scale_stride_w];
  tile_decode_scale = static_cast<float>(static_cast<fp8e4m3>(tile_decode_scale));
  constexpr float kFp32Max = 3.402823466e+38F;
  float tile_encode_val = (tile_decode_scale > 0.f)
                              ? 1.0f / (tile_decode_scale * global_decode_scale)
                              : kFp32Max;
  tile_encode_val = fminf(tile_encode_val, kFp32Max);
  const float2 scale_vec = make_float2(tile_encode_val, tile_encode_val);

  bool skip_store = true;
  for (int i = 0; i < kRowsPerWarp; ++i) {
    for (int j = 0; j < kLoopsPerRow; ++j) {
      const int h_in_smem = threadIdx.x / kThreadsPerWarp * kRowsPerWarp + i;
      const int w_in_smem = threadIdx.x % kThreadsPerWarp + kThreadsPerWarp * j;
      if (h_in_smem >= kTileDim || w_in_smem >= kTileDim) {
        continue;
      }
      const int h_in_input = tile_h * kTileDim + h_in_smem;
      const int w_in_input = tile_w * kTileDim + w_in_smem;
      const size_t idx_in_input = static_cast<size_t>(h_in_input) * w + w_in_input;
      if (h_in_input < h && w_in_input < w && idx_in_input >= start_offset &&
          idx_in_input < shard_end) {
        smem[h_in_smem][w_in_smem] = static_cast<float>(input_minus_offset[idx_in_input]);
        skip_store = false;
      }
    }
  }

  for (int delta = kThreadsPerWarp / 2; delta > 0; delta /= 2) {
    bool other = __shfl_down_sync(0xFFFFFFFF, skip_store, delta);
    skip_store = skip_store && other;
  }
  skip_store = __shfl_sync(0xFFFFFFFF, skip_store, 0);
  if (skip_store) {
    return;
  }

  for (int i = 0; i < kRowsPerWarp; ++i) {
    const int row_in_smem = threadIdx.x / kThreadsPerWarp * kRowsPerWarp + i;
    const int row_in_output = tile_h * kTileDim + row_in_smem;
    if (row_in_output >= h) {
      continue;
    }
    const int col_in_smem = threadIdx.x % kThreadsPerWarp * kNumOutputElemsPerBank;
    if (col_in_smem >= kTileDim) {
      continue;
    }
    const int col_in_output = tile_w * kTileDim + col_in_smem;

    float vals[kNumOutputElemsPerBank];
    bool mask[kNumOutputElemsPerBank];
    size_t elem_idx[kNumOutputElemsPerBank];
    bool any_valid = false;

    for (int j = 0; j < kNumOutputElemsPerBank; ++j) {
      const int col = col_in_output + j;
      const bool in_width = col < w;
      const size_t idx = static_cast<size_t>(row_in_output) * w + col;
      elem_idx[j] = idx;
      const bool in_shard = in_width && idx >= start_offset && idx < shard_end;
      mask[j] = in_shard;
      const bool in_tile = (col_in_smem + j) < kTileDim;
      const float tile_val =
          in_tile ? smem[row_in_smem][col_in_smem + j] : 0.0f;
      vals[j] = in_shard ? tile_val : 0.0f;
      any_valid |= in_shard;
    }

    if (!any_valid) {
      continue;
    }

    const float2 in01 = make_float2(vals[0], vals[1]);
    const float2 in23 = make_float2(vals[2], vals[3]);
    const auto packed =
        transformer_engine::ptx::mul_cvt_fp32_to_fp4_4x<false>(in01, in23, scale_vec, 0);
    const uint16_t packed_bits = reinterpret_cast<const uint16_t &>(packed);

    for (int pair = 0; pair < 2; ++pair) {
      const int first = pair * 2;
      const int second = first + 1;
      if (!mask[first] && !mask[second]) {
        continue;
      }
      const size_t ref_idx = mask[first] ? elem_idx[first] : elem_idx[second];
      const size_t byte_idx = (ref_idx - start_offset) >> 1;
      uint8_t byte = output[byte_idx];

      if (mask[first]) {
        const uint8_t nibble =
            static_cast<uint8_t>((packed_bits >> (4 * first)) & 0xF);
        if ((elem_idx[first] & 1u) == 0) {
          byte = static_cast<uint8_t>((byte & 0xF0u) | nibble);
        } else {
          byte = static_cast<uint8_t>((byte & 0x0Fu) | (nibble << 4));
        }
      }

      if (mask[second]) {
        const uint8_t nibble =
            static_cast<uint8_t>((packed_bits >> (4 * second)) & 0xF);
        if ((elem_idx[second] & 1u) == 0) {
          byte = static_cast<uint8_t>((byte & 0xF0u) | nibble);
        } else {
          byte = static_cast<uint8_t>((byte & 0x0Fu) | (nibble << 4));
        }
      }

      output[byte_idx] = byte;
    }
  }
}

void nvfp4_2d_compute_partial_amax(const Tensor inp, Tensor amax, size_t h, size_t w,
                                   size_t amax_stride_h, size_t amax_stride_w,
                                   size_t start_offset, size_t block_len, cudaStream_t stream) {
  NVTE_CHECK(block_len == 16, "NVFP4 2D supports 16x16 tiles only (block_len = 16).");

  size_t len = inp.numel();

  assert(h > 0 && w > 0);
  assert(start_offset < h * w);
  assert(start_offset + len <= h * w);

  size_t blocks_x = (w + kTileDim - 1) / kTileDim;
  size_t blocks_y = (h + kTileDim - 1) / kTileDim;
  assert(blocks_x <= std::numeric_limits<unsigned int>::max());
  assert(blocks_y <= std::numeric_limits<unsigned int>::max());
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      inp.dtype(), inp_dtype,
      nvfp4_2d_compute_partial_amax_kernel<inp_dtype>
      <<<grid, kThreadsPerBlock, 0, stream>>>(reinterpret_cast<const inp_dtype *>(inp.data.dptr),
                                              reinterpret_cast<float *>(amax.data.dptr),
                                              amax_stride_h, amax_stride_w, h, w, start_offset,
                                              len);)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvfp4_2d_partial_cast(const Tensor inp, Tensor out, const Tensor scale,
                           const Tensor global_scale, size_t h, size_t w, size_t scale_stride_h,
                           size_t scale_stride_w, size_t start_offset, size_t block_len,
                           cudaStream_t stream) {
  NVTE_CHECK(block_len == 16, "NVFP4 2D supports 16x16 tiles only (block_len = 16).");
  NVTE_CHECK(out.dtype() == DType::kByte, "NVFP4 rowwise data must be uint8.");

  size_t len = inp.numel();

  assert(h > 0 && w > 0);
  assert(start_offset < h * w);
  assert(start_offset + len <= h * w);

  size_t blocks_x = (w + kTileDim - 1) / kTileDim;
  size_t blocks_y = (h + kTileDim - 1) / kTileDim;
  assert(blocks_x <= std::numeric_limits<unsigned int>::max());
  assert(blocks_y <= std::numeric_limits<unsigned int>::max());
  dim3 grid(blocks_x, blocks_y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      inp.dtype(), inp_dtype,
      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          w % kTileDim == 0, kWidthAligned,
          nvfp4_2d_partial_cast_kernel<inp_dtype, kWidthAligned>
              <<<grid, kThreadsPerBlock, 0, stream>>>(
                  reinterpret_cast<const inp_dtype *>(inp.data.dptr),
                  reinterpret_cast<uint8_t *>(out.data.dptr),
                  reinterpret_cast<const float *>(scale.data.dptr), scale_stride_h, scale_stride_w,
                  reinterpret_cast<const float *>(global_scale.data.dptr), h, w, start_offset,
                  len);))
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace nvfp4_recipe
}  // namespace transformer_engine

void nvte_nvfp4_2d_compute_partial_amax(const NVTETensor inp, NVTETensor amax, size_t h,
                                                 size_t w, size_t amax_stride_h,
                                                 size_t amax_stride_w, size_t start_offset,
                                                 size_t block_len, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_compute_partial_amax);
  using namespace transformer_engine;
  nvfp4_recipe::nvfp4_2d_compute_partial_amax(
      *convertNVTETensorCheck(inp), *convertNVTETensorCheck(amax), h, w, amax_stride_h,
      amax_stride_w, start_offset, block_len, stream);
}

void nvte_nvfp4_2d_partial_cast(const NVTETensor inp, NVTETensor out, const NVTETensor scale,
                                const NVTETensor global_scale, size_t h, size_t w,
                                size_t scale_stride_h, size_t scale_stride_w, size_t start_offset,
                                size_t block_len, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_2d_partial_cast);
  using namespace transformer_engine;
  nvfp4_recipe::nvfp4_2d_partial_cast(
      *convertNVTETensorCheck(inp), *convertNVTETensorCheck(out), *convertNVTETensorCheck(scale),
      *convertNVTETensorCheck(global_scale), h, w, scale_stride_h, scale_stride_w, start_offset,
      block_len, stream);
}

void nvte_nvfp4_compute_per_tensor_scale(const NVTETensor inpA, const bool use_rowwise_amax_A,
                                         const NVTETensor inpB, const bool use_rowwise_amax_B,
                                         float alpha_in, NVTETensor alpha_out,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_compute_per_tensor_scale);
  using namespace transformer_engine;

  auto *tA = convertNVTETensor(inpA);
  auto *tB = convertNVTETensor(inpB);
  auto *tOut = convertNVTETensor(alpha_out);

  void *amax_A_ptr = use_rowwise_amax_A ? tA->amax.dptr : tA->columnwise_amax.dptr;
  void *amax_B_ptr = use_rowwise_amax_B ? tB->amax.dptr : tB->columnwise_amax.dptr;
  void *alpha_ptr = tOut->data.dptr;

  // check for not null pointers
  NVTE_CHECK(amax_A_ptr != nullptr, "amax_A_ptr is null");
  NVTE_CHECK(amax_B_ptr != nullptr, "amax_B_ptr is null");
  NVTE_CHECK(alpha_ptr != nullptr, "alpha_ptr is null");

  nvfp4_recipe::compute_nvfp4_per_tensor_scale_kernel<<<1, 1, 0, stream>>>(
      alpha_in, reinterpret_cast<const float *>(amax_A_ptr),
      reinterpret_cast<const float *>(amax_B_ptr), reinterpret_cast<float *>(alpha_ptr));
  NVTE_CHECK_CUDA(cudaGetLastError());
}




