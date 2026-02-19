/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*
 * ---------------------------------------------------------------------------
 * NVFP4 TRANSPOSE KERNEL
 *
 * Unlike FP8, NVFP4 packs two 4-bit values into each byte. A simple byte-wise
 * transpose doesn't work because the packing changes:
 *   - Before transpose: elements [m, 2c] and [m, 2c+1] share a byte
 *   - After transpose:  elements [k, 2*m_packed] and [k, 2*m_packed+1] share a byte
 *                       which were originally [2*m_packed, k] and [2*m_packed+1, k]
 * ---------------------------------------------------------------------------
 */

// Vectorized transpose kernel parameters
constexpr int TRANSPOSE_TILE_DIM = 64;       // Logical FP4 elements per tile dimension
constexpr int TRANSPOSE_TILE_PACKED = 32;    // TILE_DIM / 2 bytes
constexpr int TRANSPOSE_BLOCK_SIZE = 256;    // threads per block

// Shared memory: store unpacked 4-bit values as bytes for easy transpose
// Size: TILE_DIM x (TILE_DIM + 4) to avoid bank conflicts
constexpr int TRANSPOSE_SHMEM_STRIDE = TRANSPOSE_TILE_DIM + 4;

/*
 * Vectorized transpose kernel with uint2 loads/stores (256 threads)
 * Tile: 64x64 logical FP4 = 64x32 packed bytes
 */
__global__ void __launch_bounds__(TRANSPOSE_BLOCK_SIZE)
nvfp4_transpose_kernel(const uint8_t* __restrict__ input, 
                       uint8_t* __restrict__ output,
                       const size_t M, const size_t K) {
  const size_t K_packed = K / 2;
  const size_t M_packed = M / 2;

  const size_t tile_m_start = blockIdx.x * TRANSPOSE_TILE_DIM;
  const size_t tile_k_start = blockIdx.y * TRANSPOSE_TILE_DIM;

  __shared__ uint8_t shmem[TRANSPOSE_TILE_DIM][TRANSPOSE_SHMEM_STRIDE];

  const int tid = threadIdx.x;
  
  // Phase 1: Load input tile with VECTORIZED uint2 reads
  // 256 threads, each loads 8 bytes (uint2) = 2048 bytes total
  // Input tile: [64 rows, 32 cols] = 2048 bytes
  {
    const int thread_row = tid / 4;           // 64 rows, 4 threads per row
    const int thread_col = (tid % 4) * 8;     // 4 x 8 = 32 bytes per row
    
    const size_t global_m = tile_m_start + thread_row;
    const size_t global_k_packed_base = tile_k_start / 2 + thread_col;
    
    // Load 8 bytes as uint2
    uint2 loaded = make_uint2(0, 0);
    if (global_m < M && global_k_packed_base + 7 < K_packed) {
      loaded = *reinterpret_cast<const uint2*>(&input[global_m * K_packed + global_k_packed_base]);
    } else if (global_m < M) {
      // Boundary: scalar loads
      uint8_t* bytes = reinterpret_cast<uint8_t*>(&loaded);
      #pragma unroll
      for (int b = 0; b < 8; ++b) {
        size_t col = global_k_packed_base + b;
        bytes[b] = (col < K_packed) ? input[global_m * K_packed + col] : 0;
      }
    }
    
    // Unpack 8 bytes -> 16 nibbles and store to shared memory
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&loaded);
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int k0 = thread_col * 2 + b * 2;
      const int k1 = k0 + 1;
      shmem[thread_row][k0] = bytes[b] & 0x0F;
      shmem[thread_row][k1] = (bytes[b] >> 4) & 0x0F;
    }
  }

  __syncthreads();

  // Phase 2: Write output with VECTORIZED uint2 stores
  // Output tile: [64 rows, 32 cols] = 2048 bytes
  {
    const int thread_row = tid / 4;            // output K dimension [0, 64)
    const int thread_col_base = (tid % 4) * 8; // output M_packed [0, 32) in steps of 8
    
    const size_t global_k = tile_k_start + thread_row;
    const size_t global_m_packed_base = tile_m_start / 2 + thread_col_base;
    
    if (global_k >= K) return;
    
    // Build 8 output bytes in registers
    uint8_t out_bytes[8];
    
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int out_m_packed = thread_col_base + b;
      
      if (global_m_packed_base + b >= M_packed) {
        out_bytes[b] = 0;
        continue;
      }
      
      // Two M positions that pack into this output byte
      const int m0 = out_m_packed * 2;
      const int m1 = out_m_packed * 2 + 1;
      const int k = thread_row;
      
      // Read from shared memory (transposed access)
      const uint8_t val0 = shmem[m0][k];
      const uint8_t val1 = shmem[m1][k];
      
      out_bytes[b] = val0 | (val1 << 4);
    }
    
    // Vectorized store as uint2
    if (global_m_packed_base + 7 < M_packed) {
      *reinterpret_cast<uint2*>(&output[global_k * M_packed + global_m_packed_base]) = 
          *reinterpret_cast<uint2*>(out_bytes);
    } else {
      // Boundary: scalar stores
      for (int b = 0; b < 8 && global_m_packed_base + b < M_packed; ++b) {
        output[global_k * M_packed + global_m_packed_base + b] = out_bytes[b];
      }
    }
  }
}

void nvfp4_transpose(const Tensor input, Tensor output, cudaStream_t stream) {
  // Input has logical shape [M, K], stored as [M, K/2] bytes
  // Output has logical shape [K, M], stored as [K, M/2] bytes

  NVTE_CHECK(input.dtype() == DType::kByte, "NVFP4 transpose input must be uint8.");
  NVTE_CHECK(output.dtype() == DType::kByte, "NVFP4 transpose output must be uint8.");

  // Get dimensions from packed storage
  // input.shape() = [M, K/2], so M = shape[0], K = shape[1] * 2
  const auto in_shape = input.shape();
  NVTE_CHECK(in_shape.size() == 2, "NVFP4 transpose expects 2D input (packed), got ", in_shape.size(), "D.");
  const size_t M = in_shape[0];
  const size_t K_packed = in_shape[1];
  const size_t K = K_packed * 2;

  // Output should be [K, M/2]
  const size_t M_packed = M / 2;
  NVTE_CHECK(M % 2 == 0, "NVFP4 transpose requires M (", M, ") to be even.");

  const auto out_shape = output.shape();
  NVTE_CHECK(out_shape.size() == 2, "NVFP4 transpose expects 2D output.");
  NVTE_CHECK(out_shape[0] == K && out_shape[1] == M_packed,
             "NVFP4 transpose output shape mismatch. Expected [", K, ", ", M_packed,
             "], got [", out_shape[0], ", ", out_shape[1], "].");

  if (M == 0 || K == 0) return;

  // Use vectorized kernel (faster than TMA for pure transpose)
  // 128x128 tiles with 512 threads and uint4 vectorized access
  dim3 block(TRANSPOSE_BLOCK_SIZE);
  dim3 grid((M + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
            (K + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);
  
  nvfp4_transpose_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(input.data.dptr),
      reinterpret_cast<uint8_t *>(output.data.dptr), M, K);
  
  NVTE_CHECK_CUDA(cudaGetLastError());
}

/*
 * ---------------------------------------------------------------------------
 * NVFP4 SCALE TRANSPOSE KERNEL
 *
 * Transposes tile-level scales from rowwise to columnwise format.
 * Scale values are stored as E4M3 (fp8) in uint8 tensors.
 * 
 * Input (rowwise_scale_inv): [M_padded, K_tiles] where scales are stored
 *   at every 16th row (i.e., row 0, 16, 32, ... contain the actual scales,
 *   and each row i within a tile block has the same scale as row (i // 16) * 16).
 *
 * Output (columnwise_scale_inv): [K_padded, M_tiles] where scales are
 *   repeated 16 times per tile row.
 *
 * Mapping:
 *   output[k_tile * 16 + i, m_tile] = input[m_tile * 16, k_tile]
 *   for i in [0, 16) and valid (k_tile, m_tile) indices.
 * ---------------------------------------------------------------------------
 */
__global__ void nvfp4_scale_transpose_kernel(
    const uint8_t* __restrict__ input,   // [M_padded, K_tiles], E4M3 stored as uint8
    uint8_t* __restrict__ output,        // [K_padded, M_tiles], E4M3 stored as uint8
    const size_t M_tiles,              // Number of M tiles
    const size_t K_tiles,              // Number of K tiles
    const size_t input_stride,         // K_tiles (input row stride)
    const size_t output_stride,        // M_tiles (output row stride)
    const size_t K_padded              // Output height
) {
    // Each thread handles one output element
    const size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_row >= K_padded || out_col >= M_tiles) return;
    
    // Determine which tile row this belongs to
    const size_t k_tile = out_row / kTileDim;
    
    // Read from input: row = m_tile * 16 (first row of the tile), col = k_tile
    // m_tile = out_col
    if (k_tile < K_tiles) {
        const size_t in_row = out_col * kTileDim;  // m_tile * 16
        const uint8_t scale = input[in_row * input_stride + k_tile];
        output[out_row * output_stride + out_col] = scale;
    } else {
        output[out_row * output_stride + out_col] = 0;
    }
}

void nvfp4_scale_transpose(const Tensor input, Tensor output, 
                           size_t M_tiles, size_t K_tiles,
                           cudaStream_t stream) {
    NVTE_CHECK(input.dtype() == DType::kByte, "NVFP4 scale transpose input must be uint8 (E4M3).");
    NVTE_CHECK(output.dtype() == DType::kByte, "NVFP4 scale transpose output must be uint8 (E4M3).");
    
    const auto in_shape = input.shape();
    const auto out_shape = output.shape();
    NVTE_CHECK(in_shape.size() == 2, "NVFP4 scale transpose expects 2D input.");
    NVTE_CHECK(out_shape.size() == 2, "NVFP4 scale transpose expects 2D output.");
    
    const size_t input_stride = in_shape[1];   // K_tiles
    const size_t output_stride = out_shape[1]; // M_tiles
    const size_t K_padded = out_shape[0];
    
    if (M_tiles == 0 || K_tiles == 0 || K_padded == 0) return;
    
    constexpr int kBlockDim = 16;
    dim3 block(kBlockDim, kBlockDim);
    dim3 grid((M_tiles + kBlockDim - 1) / kBlockDim, 
              (K_padded + kBlockDim - 1) / kBlockDim);
    
    nvfp4_scale_transpose_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(input.data.dptr),
        reinterpret_cast<uint8_t*>(output.data.dptr),
        M_tiles, K_tiles, input_stride, output_stride, K_padded);
    NVTE_CHECK_CUDA(cudaGetLastError());
}

/*
 * ---------------------------------------------------------------------------
 * NVFP4 SCALE EXPANSION KERNEL
 *
 * Expands tile-level scales to row-level scales and converts to FP8 E4M3, used in partial cast.
 * 
 * Input (per_block_decode_scale): [tile_rows, tile_cols] in float32
 * Output (target_scale): [rows_padded, tile_cols] in uint8 (E4M3)
 *
 * Each tile row's scale is repeated block_len times in the output.
 * ---------------------------------------------------------------------------
 */
__global__ void nvfp4_expand_scale_to_fp8_kernel(
    const float* __restrict__ input,   // [tile_rows, tile_cols]
    uint8_t* __restrict__ output,      // [rows_padded, tile_cols]
    const size_t tile_rows,
    const size_t tile_cols,
    const size_t rows_padded,
    const size_t block_len
) {
    const size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (out_row >= rows_padded || out_col >= tile_cols) return;
    
    // Determine which tile row this output row belongs to
    const size_t tile_row = out_row / block_len;
    
    float scale_val = 0.0f;
    if (tile_row < tile_rows) {
        scale_val = input[tile_row * tile_cols + out_col];
    }
    
    // Convert float32 to FP8 E4M3
    // Clamp to FP8 E4M3 range and convert
    fp8e4m3 fp8_val = static_cast<fp8e4m3>(scale_val);
    output[out_row * tile_cols + out_col] = reinterpret_cast<const uint8_t&>(fp8_val);
}

void nvfp4_expand_scale_to_fp8(const Tensor input, Tensor output,
                               size_t tile_rows, size_t tile_cols,
                               size_t rows_padded, size_t block_len,
                               cudaStream_t stream) {
    NVTE_CHECK(input.dtype() == DType::kFloat32, "Scale input must be float32.");
    NVTE_CHECK(output.dtype() == DType::kByte, "Scale output must be uint8 (E4M3).");
    
    if (tile_rows == 0 || tile_cols == 0 || rows_padded == 0) return;
    
    constexpr int kBlockDim = 16;
    dim3 block(kBlockDim, kBlockDim);
    dim3 grid((tile_cols + kBlockDim - 1) / kBlockDim,
              (rows_padded + kBlockDim - 1) / kBlockDim);
    
    nvfp4_expand_scale_to_fp8_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float*>(input.data.dptr),
        reinterpret_cast<uint8_t*>(output.data.dptr),
        tile_rows, tile_cols, rows_padded, block_len);
    NVTE_CHECK_CUDA(cudaGetLastError());
}

/*
 * ---------------------------------------------------------------------------
 * NVFP4 COMPUTE PER-BLOCK DECODE SCALE KERNEL
 *
 * Computes per-block decode scale from block amax and global amax:
 *   global_scale = (fp8_max * fp4_max) / global_amax = 2688 / global_amax
 *   per_block_decode_scale = block_amax / fp4_max * global_scale
 *                          = block_amax * 448 / global_amax
 *
 * This matches the CUDA device function compute_decoding_scaling_factor() in core_nvfp4.cuh
 *
 * Input (block_amax): [tile_rows, tile_cols] in float32
 * Input (global_amax): scalar float32 (per-tensor amax after all-reduce)
 * Output (scale): [tile_rows, tile_cols] in float32
 * Output (global_scale_out): scalar float32 (the computed global encode scale)
 * ---------------------------------------------------------------------------
 */
__global__ void nvfp4_compute_per_block_scale_kernel(
    const float* __restrict__ block_amax,  // [tile_rows, tile_cols]
    float* __restrict__ scale,             // [tile_rows, tile_cols]
    const float* __restrict__ global_amax_ptr,  // Pointer to single float value (avoids D2H)
    const size_t numel
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    constexpr float fp4_max = 6.0f;
    constexpr float fp8_max = 448.0f;
    constexpr float flt_max = 3.402823466e+38f;
    constexpr float tiny = 1.17549435e-38f;  // FLT_MIN
    
    // Read global_amax from device memory (avoids D2H transfer)
    float global_amax = *global_amax_ptr;
    
    // Compute global encode scale: S_enc = (fp8_max * fp4_max) / global_amax
    float safe_global_amax = fmaxf(global_amax, tiny);
    float global_scale = (global_amax > 0.0f) ? 
        fminf((fp8_max * fp4_max) / safe_global_amax, flt_max) : 1.0f;
    
    // Compute per-block decode scale: S_dec_b = block_amax / fp4_max * S_enc
    float amax_val = block_amax[idx];
    float result = fminf((amax_val / fp4_max) * global_scale, flt_max);
    scale[idx] = result;
}

// Simple kernel to compute global encode scale from global amax
__global__ void nvfp4_compute_global_scale_kernel(
    const float* __restrict__ global_amax,  // [num_params]
    float* __restrict__ global_scale,       // [num_params]
    const size_t num_params
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;
    
    constexpr float fp4_max = 6.0f;
    constexpr float fp8_max = 448.0f;
    constexpr float flt_max = 3.402823466e+38f;
    constexpr float tiny = 1.17549435e-38f;  // FLT_MIN
    
    float amax = global_amax[idx];
    float safe_amax = fmaxf(amax, tiny);
    float scale = (amax > 0.0f) ? fminf((fp8_max * fp4_max) / safe_amax, flt_max) : 1.0f;
    global_scale[idx] = scale;
}

void nvfp4_compute_per_block_scale(const Tensor block_amax, Tensor scale,
                                   const Tensor global_amax, cudaStream_t stream) {
    NVTE_CHECK(block_amax.dtype() == DType::kFloat32, "Block amax must be float32.");
    NVTE_CHECK(scale.dtype() == DType::kFloat32, "Scale must be float32.");
    NVTE_CHECK(global_amax.dtype() == DType::kFloat32, "Global amax must be float32.");
    NVTE_CHECK(global_amax.numel() == 1, "Global amax must be a single element tensor.");
    
    size_t numel = block_amax.numel();
    if (numel == 0) return;
    
    constexpr int kBlockSize = 256;
    int grid_size = (numel + kBlockSize - 1) / kBlockSize;
    
    nvfp4_compute_per_block_scale_kernel<<<grid_size, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float*>(block_amax.data.dptr),
        reinterpret_cast<float*>(scale.data.dptr),
        reinterpret_cast<const float*>(global_amax.data.dptr),
        numel);
    NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvfp4_compute_global_scale(const Tensor global_amax, Tensor global_scale,
                                cudaStream_t stream) {
    NVTE_CHECK(global_amax.dtype() == DType::kFloat32, "Global amax must be float32.");
    NVTE_CHECK(global_scale.dtype() == DType::kFloat32, "Global scale must be float32.");
    
    size_t num_params = global_amax.numel();
    if (num_params == 0) return;
    
    constexpr int kBlockSize = 256;
    int grid_size = (num_params + kBlockSize - 1) / kBlockSize;
    
    nvfp4_compute_global_scale_kernel<<<grid_size, kBlockSize, 0, stream>>>(
        reinterpret_cast<const float*>(global_amax.data.dptr),
        reinterpret_cast<float*>(global_scale.data.dptr),
        num_params);
    NVTE_CHECK_CUDA(cudaGetLastError());
}

/*
 * ---------------------------------------------------------------------------
 * FUSED NVFP4 SCALE COMPUTATION KERNEL
 *
 * Fuses three operations into one kernel:
 * 1. nvfp4_compute_per_block_scale: compute tile-level decode scales from block amax
 * 2. target_amax.copy_: copy global amax to target tensor
 * 3. nvfp4_expand_scale_to_fp8: expand to row-level and convert to FP8 E4M3
 *
 * Input (block_amax): [tile_rows, tile_cols] float32
 * Input (global_amax): [1] float32
 * Output (per_block_scale): [tile_rows, tile_cols] float32 (intermediate, for partial_cast)
 * Output (target_scale): [rows_padded, tile_cols] uint8 (E4M3)
 * Output (target_amax): [1] float32 (copy of global_amax)
 *
 * Saves 2 kernel launches per parameter (eliminates nvfp4_compute_per_block_scale and
 * nvfp4_expand_scale_to_fp8 as separate calls, plus the amax copy).
 * ---------------------------------------------------------------------------
 */
 __global__ void nvfp4_fused_scale_kernel(
  const float* __restrict__ block_amax,    // [tile_rows, tile_cols]
  const float* __restrict__ global_amax,   // [1]
  float* __restrict__ per_block_scale,     // [tile_rows, tile_cols] - for partial_cast
  uint8_t* __restrict__ target_scale,      // [rows_padded, tile_cols]
  float* __restrict__ target_amax,         // [1]
  const size_t tile_rows,
  const size_t tile_cols,
  const size_t rows_padded,
  const size_t block_len
) {
  const size_t out_row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t out_col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Read global amax once per thread (broadcast)
  const float g_amax = *global_amax;
  
  // Thread (0,0) copies global_amax to target_amax
  if (out_row == 0 && out_col == 0) {
      *target_amax = g_amax;
  }
  
  if (out_row >= rows_padded || out_col >= tile_cols) return;
  
  // Determine which tile row this output row belongs to
  const size_t tile_row = out_row / block_len;
  
  // Compute the scale value
  constexpr float fp4_max = 6.0f;
  constexpr float fp8_max = 448.0f;
  constexpr float flt_max = 3.402823466e+38f;
  constexpr float tiny = 1.17549435e-38f;
  
  float scale_val = 0.0f;
  if (tile_row < tile_rows) {
      float safe_global_amax = fmaxf(g_amax, tiny);
      float global_scale = (g_amax > 0.0f) ? 
          fminf((fp8_max * fp4_max) / safe_global_amax, flt_max) : 1.0f;
      
      // Read block amax and compute per-block decode scale
      float amax_val = block_amax[tile_row * tile_cols + out_col];
      scale_val = fminf((amax_val / fp4_max) * global_scale, flt_max);
      
      // Write per-block scale (only once per tile, when out_row % block_len == 0)
      if (out_row % block_len == 0) {
          per_block_scale[tile_row * tile_cols + out_col] = scale_val;
      }
  }
  
  // Convert float32 to FP8 E4M3 and write expanded scale
  fp8e4m3 fp8_val = static_cast<fp8e4m3>(scale_val);
  target_scale[out_row * tile_cols + out_col] = reinterpret_cast<const uint8_t&>(fp8_val);
}

void nvfp4_fused_scale(const Tensor block_amax, const Tensor global_amax,
                     Tensor per_block_scale, Tensor target_scale, Tensor target_amax,
                     size_t tile_rows, size_t tile_cols,
                     size_t rows_padded, size_t block_len,
                     cudaStream_t stream) {
  NVTE_CHECK(block_amax.dtype() == DType::kFloat32, "Block amax must be float32.");
  NVTE_CHECK(global_amax.dtype() == DType::kFloat32, "Global amax must be float32.");
  NVTE_CHECK(per_block_scale.dtype() == DType::kFloat32, "Per-block scale must be float32.");
  NVTE_CHECK(target_scale.dtype() == DType::kByte, "Target scale must be uint8 (E4M3).");
  NVTE_CHECK(target_amax.dtype() == DType::kFloat32, "Target amax must be float32.");
  NVTE_CHECK(global_amax.numel() == 1, "Global amax must be a single element tensor.");
  NVTE_CHECK(target_amax.numel() == 1, "Target amax must be a single element tensor.");
  
  if (tile_rows == 0 || tile_cols == 0 || rows_padded == 0) return;
  
  constexpr int kBlockDim = 16;
  dim3 block(kBlockDim, kBlockDim);
  dim3 grid((tile_cols + kBlockDim - 1) / kBlockDim,
            (rows_padded + kBlockDim - 1) / kBlockDim);
  
  nvfp4_fused_scale_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const float*>(block_amax.data.dptr),
      reinterpret_cast<const float*>(global_amax.data.dptr),
      reinterpret_cast<float*>(per_block_scale.data.dptr),
      reinterpret_cast<uint8_t*>(target_scale.data.dptr),
      reinterpret_cast<float*>(target_amax.data.dptr),
      tile_rows, tile_cols, rows_padded, block_len);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
}  // namespace nvfp4_recipe
}  // namespace transformer_engine

void nvte_nvfp4_expand_scale_to_fp8(const NVTETensor input, NVTETensor output,
                                    size_t tile_rows, size_t tile_cols,
                                    size_t rows_padded, size_t block_len,
                                    cudaStream_t stream) {
    NVTE_API_CALL(nvte_nvfp4_expand_scale_to_fp8);
    using namespace transformer_engine;
    nvfp4_recipe::nvfp4_expand_scale_to_fp8(*convertNVTETensorCheck(input),
                                            *convertNVTETensorCheck(output),
                                            tile_rows, tile_cols, rows_padded, block_len, stream);
}

void nvte_nvfp4_compute_per_block_scale(const NVTETensor block_amax, NVTETensor scale,
                                        const NVTETensor global_amax, cudaStream_t stream) {
    NVTE_API_CALL(nvte_nvfp4_compute_per_block_scale);
    using namespace transformer_engine;
    nvfp4_recipe::nvfp4_compute_per_block_scale(*convertNVTETensorCheck(block_amax),
                                                *convertNVTETensorCheck(scale),
                                                *convertNVTETensorCheck(global_amax),
                                                stream);
}

void nvte_nvfp4_compute_global_scale(const NVTETensor global_amax, NVTETensor global_scale,
                                     cudaStream_t stream) {
    NVTE_API_CALL(nvte_nvfp4_compute_global_scale);
    using namespace transformer_engine;
    nvfp4_recipe::nvfp4_compute_global_scale(*convertNVTETensorCheck(global_amax),
                                             *convertNVTETensorCheck(global_scale),
                                             stream);
}

void nvte_nvfp4_scale_transpose(const NVTETensor input, NVTETensor output,
                                size_t M_tiles, size_t K_tiles, cudaStream_t stream) {
    NVTE_API_CALL(nvte_nvfp4_scale_transpose);
    using namespace transformer_engine;
    nvfp4_recipe::nvfp4_scale_transpose(*convertNVTETensorCheck(input),
                                        *convertNVTETensorCheck(output),
                                        M_tiles, K_tiles, stream);
}

void nvte_nvfp4_transpose(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_nvfp4_transpose);
  using namespace transformer_engine;
  nvfp4_recipe::nvfp4_transpose(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                                stream);
}

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

void nvte_nvfp4_fused_scale(const NVTETensor block_amax, const NVTETensor global_amax,
                            NVTETensor per_block_scale, NVTETensor target_scale,
                            NVTETensor target_amax,
                            size_t tile_rows, size_t tile_cols,
                            size_t rows_padded, size_t block_len,
                            cudaStream_t stream) {
    NVTE_API_CALL(nvte_nvfp4_fused_scale);
    using namespace transformer_engine;
    nvfp4_recipe::nvfp4_fused_scale(*convertNVTETensorCheck(block_amax),
                                    *convertNVTETensorCheck(global_amax),
                                    *convertNVTETensorCheck(per_block_scale),
                                    *convertNVTETensorCheck(target_scale),
                                    *convertNVTETensorCheck(target_amax),
                                    tile_rows, tile_cols, rows_padded, block_len,
                                    stream);
}
