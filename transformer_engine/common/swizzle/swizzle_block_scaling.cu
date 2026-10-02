/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/swizzle.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace {
constexpr uint32_t WARP_SIZE = 32;
}  // namespace
namespace swizzle_kernel_1d {
constexpr uint32_t WARPS_X_PER_TB = 2;  // configurable
constexpr uint32_t WARPS_Y_PER_TB = 2;  // configurable

// Transposes a 4x4 matrix of bytes stored across four threads with consecutive thread ids where
// each thread stores a single row (of four bytes).
// Example:
//   lane0.row = 0x00010203
//   lane1.row = 0x04050607
//   lane2.row = 0x08090a0b
//   lane3.row = 0x0c0d0e0f
// Becomes:
//   lane0.row = 0x0004080c
//   lane1.row = 0x0105090d
//   lane2.row = 0x02060a0e
//   lane3.row = 0x03070b0f
uint32_t __device__ __forceinline__ transpose_4x4_byte_matrix(const uint32_t row,
                                                              const uint32_t lane,
                                                              const uint32_t active_mask) {
  using cu = const uint32_t;

  // Threads operate in groups of 4, and each thread stores 4 bytes at a time.
  // The bytes in this 4x4 matrix are labeled in hex. We shuffle around bytes
  // until we have transposed the 4x4 matrix.
  cu m_0123_4567_89ab_cdef = row;
  cu m_4567_0123_cdef_89ab = __shfl_xor_sync(active_mask, m_0123_4567_89ab_cdef, 1, 4);
  cu m_0426_4062_8cae_c8ea = __byte_perm(m_0123_4567_89ab_cdef, m_4567_0123_cdef_89ab, 0x6240);
  cu m_5173_1537_d9fb_9dbf = __byte_perm(m_0123_4567_89ab_cdef, m_4567_0123_cdef_89ab, 0x3715);
  cu m_0426_1537_8cae_9dbf = (lane & 1) ? m_5173_1537_d9fb_9dbf : m_0426_4062_8cae_c8ea;
  cu m_8cae_9dbf_0426_1537 = __shfl_xor_sync(active_mask, m_0426_1537_8cae_9dbf, 2, 4);
  cu m_048c_159d_8c04_9d15 = __byte_perm(m_0426_1537_8cae_9dbf, m_8cae_9dbf_0426_1537, 0x5410);
  cu m_ae26_bf37_26ae_37bf = __byte_perm(m_0426_1537_8cae_9dbf, m_8cae_9dbf_0426_1537, 0x3276);
  cu m_048c_159d_26ae_37bf = (lane & 2) ? m_ae26_bf37_26ae_37bf : m_048c_159d_8c04_9d15;

  return m_048c_159d_26ae_37bf;
}

// Expands a uint32_t to a uint4 by duplicating each byte four times.
// Example: 0x01020304u becomes uint4{0x01010101, 0x02020202, 0x03030303, 0x04040404}
uint4 __device__ __forceinline__ broadcast_uint32_t_to_uint4(uint32_t x) {
  return {__byte_perm(x, 0, 0x0000), __byte_perm(x, 0, 0x1111), __byte_perm(x, 0, 0x2222),
          __byte_perm(x, 0, 0x3333)};
}

// Tag struct denoting whether the number of rows of the input fp8 block scaling tensor's data
// matrix is divisible by 128. If it is not, some threads could read out of bounds scaling factors.
struct no_oob_tag_t {};
constexpr no_oob_tag_t NO_OOB_TAG;

// Converts the scaling factors of one 128x128 data tile. `in` and `out` point at the tensor's
// compact 1D block scales and its swizzled MXFP8 scales. All lanes of the warp must call this
// with the same tile coordinates.
template <typename OOBT>
__device__ __forceinline__ void swizzle_tile(const void* __restrict__ const in,
                                             void* __restrict__ const out,
                                             const uint32_t out_tile_x, const uint32_t out_tile_y,
                                             const uint32_t tiles_y, const uint32_t in_y_stride,
                                             const uint32_t out_y_stride, const OOBT first_oob,
                                             const uint32_t lane) {
  // resolve kernel variant
  constexpr bool no_oob = std::is_same_v<OOBT, no_oob_tag_t>;
  static_assert(no_oob || std::is_same_v<OOBT, uint32_t>);

  const uint32_t in_tile_y = out_tile_x;
  const uint32_t in_tile_x = out_tile_y;

  // calculate this warp's input base pointer
  constexpr uint32_t in_x_stride = WARP_SIZE * sizeof(uint4);
  const void* const warp_src =
      (reinterpret_cast<const uint8_t*>(in) + in_tile_y * in_y_stride + in_tile_x * in_x_stride);

  // load scaling factors for this lane's initial four 1x128 tiles
  uint4 sf;
  if constexpr (no_oob) {
    sf = reinterpret_cast<const uint4*>(warp_src)[lane];
  } else {
    if ((out_tile_y < tiles_y - 1) || lane < first_oob) {
      sf = reinterpret_cast<const uint4*>(warp_src)[lane];
    } else {
      sf = uint4{0, 0, 0, 0};
    }
  }

  // pack the exponent bits of the scaling factors
  uint32_t packed_exponents = ((sf.x >> 23) & 0xFF) | (((sf.y >> 23) & 0xFF) << 8) |
                              (((sf.z >> 23) & 0xFF) << 16) | (((sf.w >> 23) & 0xFF) << 24);

  // partially swizzle the scaling factors
  constexpr uint32_t ACTIVE_MASK = 0xFFFFFFFF;  // no divergent branches
  const uint32_t lane_load_idx = (lane % 4) * 8 + (lane / 4);
  packed_exponents = __shfl_sync(ACTIVE_MASK, packed_exponents, lane_load_idx);

  // transpose 4x4 matrices of scaling factors
  packed_exponents = transpose_4x4_byte_matrix(packed_exponents, lane % 4, ACTIVE_MASK);

  // broadcast the scaling factors for sixteen 1x32 tiles
  sf = broadcast_uint32_t_to_uint4(packed_exponents);

  // store them cooperatively for 512 1x32 tiles in a 128x128 tile
  constexpr uint32_t out_x_stride = 512;
  void* const warp_dst =
      (reinterpret_cast<uint8_t*>(out) + out_tile_y * out_y_stride + out_tile_x * out_x_stride);
  reinterpret_cast<uint4*>(warp_dst)[lane] = sf;
}

template <typename OOBT>
void __global__ __launch_bounds__(WARPS_X_PER_TB* WARPS_Y_PER_TB* WARP_SIZE)
    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel(
        const void* __restrict__ const in, void* __restrict__ const out, const uint32_t tiles_x,
        const uint32_t tiles_y, const uint32_t in_y_stride, const uint32_t out_y_stride,
        OOBT first_oob) {
  // load thread indices
  const uint32_t lane = threadIdx.x;
  __builtin_assume(lane < WARP_SIZE);
  const uint32_t warp_x = threadIdx.z;
  __builtin_assume(warp_x < WARPS_X_PER_TB);
  const uint32_t warp_y = threadIdx.y;
  __builtin_assume(warp_y < WARPS_Y_PER_TB);

  // compute tile indices
  const uint32_t out_tile_y = blockIdx.y * WARPS_Y_PER_TB + warp_y;
  const uint32_t out_tile_x = blockIdx.x * WARPS_X_PER_TB + warp_x;

  // bounds check; uniform branch
  if (out_tile_y >= tiles_y || out_tile_x >= tiles_x) {
    return;
  }

  swizzle_tile(in, out, out_tile_x, out_tile_y, tiles_y, in_y_stride, out_y_stride, first_oob,
               lane);
}

void launch_kernel(const void* const in, void* const out, uint32_t data_rows, uint32_t data_cols,
                   cudaStream_t stream) {
  NVTE_CHECK(is_aligned_ptr(in, alignof(uint4)), "Input scaling factor pointer must be aligned to ",
             alignof(uint4), " bytes");
  NVTE_CHECK(is_aligned_ptr(out, alignof(uint4)),
             "Output scaling factor pointer must be aligned to ", alignof(uint4), " bytes");
  NVTE_CHECK(data_rows % 4 == 0, "Input tensor must not have any padding scaling factors");

  const uint32_t tiles_x = DIVUP(data_cols, 128u);
  const uint32_t tiles_y = DIVUP(data_rows, 128u);
  const dim3 grid_dim{DIVUP(tiles_x, WARPS_X_PER_TB), DIVUP(tiles_y, WARPS_Y_PER_TB), 1};
  const dim3 block_dim{WARP_SIZE, WARPS_Y_PER_TB, WARPS_X_PER_TB};

  // Each 128x128 tile in the data corresponds to a 128x1 tile in the input scales
  // and a 128x4 tile in the output scales. The input scales are in transposed order.
  const uint32_t input_scale_inv_cols = DIVUP(data_rows, 4u) * 4;
  const uint32_t output_scale_inv_cols = tiles_x * 128 * 4;
  const uint32_t in_y_stride = input_scale_inv_cols * sizeof(float);
  const uint32_t out_y_stride = output_scale_inv_cols * sizeof(uint8_t);

  const uint32_t first_oob = (input_scale_inv_cols % 128) / 4;

  if (first_oob == 0) {
    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in, out, tiles_x, tiles_y, in_y_stride, out_y_stride, NO_OOB_TAG);
  } else {
    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in, out, tiles_x, tiles_y, in_y_stride, out_y_stride, first_oob);
  }
}
}  // namespace swizzle_kernel_1d
namespace swizzle_kernel_2d {
constexpr uint32_t WARPS_X_PER_TB = 2;  // configurable
constexpr uint32_t WARPS_Y_PER_TB = 2;  // configurable

// Converts the scaling factor of one 128x128 data tile. All lanes of the warp must call this
// with the same tile coordinates.
__device__ __forceinline__ void swizzle_tile(const void* __restrict__ const in,
                                             void* __restrict__ const out,
                                             const uint32_t out_tile_x, const uint32_t out_tile_y,
                                             const uint32_t in_y_stride,
                                             const uint32_t out_y_stride, const uint32_t lane) {
  const uint32_t in_tile_y = out_tile_y;
  const uint32_t in_tile_x = out_tile_x;

  // calculate this warp's input base pointer
  constexpr uint32_t in_x_stride = sizeof(float);
  const void* const warp_src =
      (reinterpret_cast<const uint8_t*>(in) + in_tile_y * in_y_stride + in_tile_x * in_x_stride);

  // load scaling factor for this warp's 128x128 tile
  uint32_t sf = *reinterpret_cast<const uint32_t*>(warp_src);

  // broadcast it to four scaling factors for 1x32 tiles
  // extract and broadcast the exponent byte to four bytes for E8M0 format
  uint32_t exp_byte = (sf >> 23) & 0xFF;
  sf = exp_byte | (exp_byte << 8) | (exp_byte << 16) | (exp_byte << 24);

  // broadcast it to sixteen scaling factors for 1x32 tiles
  const uint4 sf4{sf, sf, sf, sf};

  // store it cooperatively for 512 1x32 tiles in a 128x128 tile
  constexpr uint32_t out_x_stride = 512;
  void* const warp_dst =
      (reinterpret_cast<uint8_t*>(out) + out_tile_y * out_y_stride + out_tile_x * out_x_stride);
  reinterpret_cast<uint4*>(warp_dst)[lane] = sf4;
}

void __global__ __launch_bounds__(WARPS_X_PER_TB* WARPS_Y_PER_TB* WARP_SIZE)
    swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel(
        const void* __restrict__ const in, void* __restrict__ const out, const uint32_t tiles_x,
        const uint32_t tiles_y, const uint32_t in_y_stride, const uint32_t out_y_stride) {
  // load thread indices
  const uint32_t lane = threadIdx.x;
  __builtin_assume(lane < WARP_SIZE);
  const uint32_t warp_x = threadIdx.z;
  __builtin_assume(warp_x < WARPS_X_PER_TB);
  const uint32_t warp_y = threadIdx.y;
  __builtin_assume(warp_y < WARPS_Y_PER_TB);

  // compute tile indices
  const uint32_t out_tile_y = blockIdx.y * WARPS_Y_PER_TB + warp_y;
  const uint32_t out_tile_x = blockIdx.x * WARPS_X_PER_TB + warp_x;

  // bounds check; uniform branch
  if (out_tile_y >= tiles_y || out_tile_x >= tiles_x) {
    return;
  }

  swizzle_tile(in, out, out_tile_x, out_tile_y, in_y_stride, out_y_stride, lane);
}

void launch_kernel(const void* const in, void* const out, uint32_t data_rows, uint32_t data_cols,
                   cudaStream_t stream) {
  NVTE_CHECK(is_aligned_ptr(in, alignof(float)), "Input scaling factor pointer must be aligned to ",
             alignof(float), " bytes");
  NVTE_CHECK(is_aligned_ptr(out, alignof(uint4)),
             "Output scaling factor pointer must be aligned to ", alignof(uint4), " bytes");

  const uint32_t tiles_x = DIVUP(data_cols, 128u);
  const uint32_t tiles_y = DIVUP(data_rows, 128u);
  const dim3 grid_dim{DIVUP(tiles_x, WARPS_X_PER_TB), DIVUP(tiles_y, WARPS_Y_PER_TB), 1};
  const dim3 block_dim{WARP_SIZE, WARPS_Y_PER_TB, WARPS_X_PER_TB};

  // Each 128x128 tile in the data corresponds to a 1x1 tile in the input scales
  // and a 128x4 tile in the output scales.
  const uint32_t input_scale_inv_cols = DIVUP(data_cols, 512u) * 4;
  const uint32_t output_scale_inv_cols = tiles_x * 128 * 4;
  const uint32_t in_y_stride = input_scale_inv_cols * sizeof(float);
  const uint32_t out_y_stride = output_scale_inv_cols * sizeof(uint8_t);

  swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel<<<grid_dim, block_dim, 0, stream>>>(
      in, out, tiles_x, tiles_y, in_y_stride, out_y_stride);
}
}  // namespace swizzle_kernel_2d
namespace swizzle_kernel_grouped {
constexpr uint32_t WARPS_PER_TB = 4;
constexpr size_t kBlockLen = 128;
constexpr size_t kMaxBlocksPerSM = 8;
// Each warp converts one 128x128 data tile, writing 128x4 = 512 scale bytes.
constexpr size_t kTileOutputBytes = 512;

__host__ __device__ __forceinline__ size_t ceil_div(const size_t a, const size_t b) {
  return (a + b - 1) / b;
}
__host__ __device__ __forceinline__ size_t round_up(const size_t a, const size_t b) {
  return ceil_div(a, b) * b;
}

// Per-tensor scale sizes for a tensor with rowwise data [rows, cols]. These must match
// padded_block_{1d,2d}_scale_inv_floats(.., rowwise=true) and
// padded_mxfp8_scale_inv_bytes(.., rowwise=true) in gemm/cublaslt_grouped_gemm.cu, which locate
// each tensor's scales in the same grouped buffers.
template <bool kIs2D>
__device__ __forceinline__ size_t input_scale_floats(const size_t rows, const size_t cols) {
  if constexpr (kIs2D) {
    return ceil_div(rows, kBlockLen) * round_up(ceil_div(cols, kBlockLen), 4);
  } else {
    return ceil_div(cols, kBlockLen) * round_up(rows, 4);
  }
}
__device__ __forceinline__ size_t output_scale_bytes(const size_t rows, const size_t cols) {
  return round_up(rows, kBlockLen) * ceil_div(cols, kBlockLen) * 4;
}

// Shared memory for the per-tensor tables: first tile (num_tensors + 1 entries, the last being the
// total), first input scale and first output scale of every tensor.
__host__ __device__ __forceinline__ size_t table_smem_bytes(const size_t num_tensors) {
  return (3 * num_tensors + 1) * sizeof(size_t);
}

// Persistent kernel: each CTA tabulates every tensor's first tile and scale offsets in shared
// memory, then its warps walk 128x128 data tiles across all tensors in the group and find each
// tile's tensor by binary search. The per-tensor dims are read on the device, so a captured launch
// stays correct when they change between CUDA graph replays.
template <bool kIs2D>
__global__ void __launch_bounds__(WARPS_PER_TB* WARP_SIZE)
    grouped_swizzle_block_scaling_to_mxfp8_kernel(
        const float* __restrict__ const in, uint8_t* __restrict__ const out,
        const int64_t* __restrict__ const first_dims, const int64_t* __restrict__ const last_dims,
        const size_t uniform_first, const size_t uniform_last, const size_t num_tensors) {
  extern __shared__ size_t tables[];
  size_t* const tile_start = tables;
  size_t* const in_start = tile_start + num_tensors + 1;
  size_t* const out_start = in_start + num_tensors;

  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const auto tensor_rows = [=](const size_t t) {
    return first_dims != nullptr ? static_cast<size_t>(first_dims[t]) : uniform_first;
  };
  const auto tensor_cols = [=](const size_t t) {
    return last_dims != nullptr ? static_cast<size_t>(last_dims[t]) : uniform_last;
  };

  // Exclusive prefix sums over the tensors, 32 tensors per step.
  if (threadIdx.x < WARP_SIZE) {
    constexpr uint32_t kFullMask = 0xFFFFFFFF;
    size_t tile_carry = 0;
    size_t in_carry = 0;
    size_t out_carry = 0;
    for (size_t base = 0; base < num_tensors; base += WARP_SIZE) {
      const size_t t = base + lane;
      size_t tiles = 0;
      size_t in_size = 0;
      size_t out_size = 0;
      if (t < num_tensors) {
        const size_t rows = tensor_rows(t);
        const size_t cols = tensor_cols(t);
        tiles = ceil_div(rows, kBlockLen) * ceil_div(cols, kBlockLen);
        in_size = input_scale_floats<kIs2D>(rows, cols);
        out_size = output_scale_bytes(rows, cols);
      }
      size_t tiles_sum = tiles;
      size_t in_sum = in_size;
      size_t out_sum = out_size;
#pragma unroll
      for (uint32_t d = 1; d < WARP_SIZE; d *= 2) {
        const size_t tiles_up = __shfl_up_sync(kFullMask, tiles_sum, d);
        const size_t in_up = __shfl_up_sync(kFullMask, in_sum, d);
        const size_t out_up = __shfl_up_sync(kFullMask, out_sum, d);
        if (lane >= d) {
          tiles_sum += tiles_up;
          in_sum += in_up;
          out_sum += out_up;
        }
      }
      if (t < num_tensors) {
        tile_start[t] = tile_carry + tiles_sum - tiles;
        in_start[t] = in_carry + in_sum - in_size;
        out_start[t] = out_carry + out_sum - out_size;
      }
      tile_carry += __shfl_sync(kFullMask, tiles_sum, WARP_SIZE - 1);
      in_carry += __shfl_sync(kFullMask, in_sum, WARP_SIZE - 1);
      out_carry += __shfl_sync(kFullMask, out_sum, WARP_SIZE - 1);
    }
    if (lane == 0) {
      tile_start[num_tensors] = tile_carry;
    }
  }
  __syncthreads();

  const size_t total_tiles = tile_start[num_tensors];
  const size_t warp_stride = static_cast<size_t>(gridDim.x) * WARPS_PER_TB;
  for (size_t tile_id = static_cast<size_t>(blockIdx.x) * WARPS_PER_TB + threadIdx.x / WARP_SIZE;
       tile_id < total_tiles; tile_id += warp_stride) {
    // Last tensor whose first tile is at or before this one. A tensor without tiles shares its
    // first tile with the next tensor, so it is never selected.
    size_t t = 0;
    size_t hi = num_tensors;
    while (hi - t > 1) {
      const size_t mid = (t + hi) / 2;
      if (tile_start[mid] <= tile_id) {
        t = mid;
      } else {
        hi = mid;
      }
    }

    const size_t rows = tensor_rows(t);
    const uint32_t tiles_x = static_cast<uint32_t>(ceil_div(tensor_cols(t), kBlockLen));
    const uint32_t local_tile = static_cast<uint32_t>(tile_id - tile_start[t]);
    const uint32_t out_tile_y = local_tile / tiles_x;
    const uint32_t out_tile_x = local_tile - out_tile_y * tiles_x;
    const uint32_t out_y_stride = tiles_x * static_cast<uint32_t>(kBlockLen) * 4;
    const float* const tensor_in = in + in_start[t];
    uint8_t* const tensor_out = out + out_start[t];
    if constexpr (kIs2D) {
      const uint32_t in_y_stride = static_cast<uint32_t>(round_up(tiles_x, 4) * sizeof(float));
      swizzle_kernel_2d::swizzle_tile(tensor_in, tensor_out, out_tile_x, out_tile_y, in_y_stride,
                                      out_y_stride, lane);
    } else {
      const uint32_t tiles_y = static_cast<uint32_t>(ceil_div(rows, kBlockLen));
      const uint32_t in_y_stride = static_cast<uint32_t>(round_up(rows, 4) * sizeof(float));
      const uint32_t first_oob = static_cast<uint32_t>((round_up(rows, 4) % kBlockLen) / 4);
      if (first_oob == 0) {
        swizzle_kernel_1d::swizzle_tile(tensor_in, tensor_out, out_tile_x, out_tile_y, tiles_y,
                                        in_y_stride, out_y_stride, swizzle_kernel_1d::NO_OOB_TAG,
                                        lane);
      } else {
        swizzle_kernel_1d::swizzle_tile(tensor_in, tensor_out, out_tile_x, out_tile_y, tiles_y,
                                        in_y_stride, out_y_stride, first_oob, lane);
      }
    }
  }
}

// Host-computable upper bound (bytes) on the swizzled MXFP8 scales of all tensors, from the
// logical shape alone (per-tensor dims live on the device). The PyTorch binding
// (convert_grouped_block_scaling_to_mxfp8_tensor) allocates exactly this many bytes and the
// public header (swizzle.h) documents these formulas, so keep all three in sync.
//   uniform:        n * roundup(F / n, 128) * ceil(L / 128) * 4     (exact)
//   varying first:  (F + 127 n) * ceil(L / 128) * 4
//   varying last:   roundup(F, 128) * 4 * (ceil(L / 128) + n)
size_t grouped_mxfp8_scale_bytes_upper_bound(const GroupedTensor& t) {
  const size_t n = t.num_tensors;
  const size_t F = t.logical_shape.data[0];
  const size_t L = t.logical_shape.data[1];
  if (F == 0 || L == 0) {
    return 0;
  }
  if (t.first_dims.has_data()) {
    return (F + (kBlockLen - 1) * n) * ceil_div(L, kBlockLen) * 4;
  }
  if (t.last_dims.has_data()) {
    return round_up(F, kBlockLen) * 4 * (ceil_div(L, kBlockLen) + n);
  }
  return n * round_up(F / n, kBlockLen) * ceil_div(L, kBlockLen) * 4;
}
}  // namespace swizzle_kernel_grouped

void swizzle_block_scaling_to_mxfp8_scaling_factors(const Tensor* input, Tensor* output,
                                                    cudaStream_t stream) {
  // Do nothing if tensor is empty
  if (input->data.numel() == 0) {
    return;
  }

  CheckInputTensor(*input, "block_scaling_scaling_factor_input");
  CheckInputTensor(*output, "mxfp8_scaling_factor_output");

  const NVTEScalingMode scaling_mode = input->scaling_mode;
  NVTE_CHECK(scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D,
             "Input tensor must be a block scaling tensor");
  NVTE_CHECK(output->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Output tensor must be an mxfp8 tensor");

  NVTE_CHECK(input->data.dtype == transformer_engine::DType::kFloat8E4M3 ||
                 input->data.dtype == transformer_engine::DType::kFloat8E5M2,
             "Input data must have FP8E4M3 or FP8E5M2 dtype to be compatible with MXFP8");
  NVTE_CHECK(output->data.dtype == input->data.dtype,
             "Output data must have the same dtype as input data");
  NVTE_CHECK(input->scale_inv.dtype == DType::kFloat32, "Input must have FP32 scaling factors");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat8E8M0,
             "Output must have E8M0 scaling factors");

  NVTE_CHECK(output->with_gemm_swizzled_scales,
             "Expected output tensor with scales in GEMM swizzled format.");

  NVTE_CHECK(input->data.dptr != nullptr, "Input must have rowwise data");
  NVTE_CHECK(output->data.dptr == input->data.dptr, "Output must share data with input");
  NVTE_CHECK(input->scale_inv.dptr != nullptr, "Input must have rowwise scaling factors");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Output must have rowwise scaling factors");

  NVTE_CHECK(input->data.shape.size() == 2, "Input data must be a matrix");
  NVTE_CHECK(output->data.shape == input->data.shape,
             "Output data must have the same shape as input data");
  NVTE_CHECK(input->scale_inv.shape.size() == 2, "Input scaling factors must be a matrix");
  NVTE_CHECK(output->scale_inv.shape.size() == 2, "Output scaling factors must be a matrix");

  const size_t data_rows = input->data.shape[0];
  const size_t data_cols = input->data.shape[1];
  const size_t input_scale_inv_rows = input->scale_inv.shape[0];
  const size_t input_scale_inv_cols = input->scale_inv.shape[1];
  const size_t output_scale_inv_rows = output->scale_inv.shape[0];
  const size_t output_scale_inv_cols = output->scale_inv.shape[1];

  NVTE_CHECK(output_scale_inv_rows == DIVUP<size_t>(data_rows, 128) * 128,
             "Expected the output scaling factor matrix to have ",
             DIVUP<size_t>(data_rows, 128) * 128, " rows, but it has ", output_scale_inv_rows,
             " rows instead.");
  NVTE_CHECK(output_scale_inv_cols == DIVUP<size_t>(data_cols, 128) * 4,
             "Expected the output scaling factor matrix to have ",
             DIVUP<size_t>(data_cols, 128) * 4, " columns, but it has ", output_scale_inv_cols,
             " columns instead.");

  if (scaling_mode == NVTE_BLOCK_SCALING_1D) {
    NVTE_CHECK(input_scale_inv_rows == DIVUP<size_t>(data_cols, 128),
               "Expected the input scaling factor matrix to have ", DIVUP<size_t>(data_cols, 128),
               " rows, but it has ", input_scale_inv_rows, " rows instead.");
    NVTE_CHECK(input_scale_inv_cols == DIVUP<size_t>(data_rows, 4) * 4,
               "Expected the input scaling factor matrix to have ", DIVUP<size_t>(data_rows, 4) * 4,
               " columns, but it has ", input_scale_inv_cols, " columns instead.");

    swizzle_kernel_1d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, data_rows,
                                     data_cols, stream);
  } else {  // scaling_mode == NVTE_BLOCK_SCALING_2D
    NVTE_CHECK(input_scale_inv_rows == DIVUP<size_t>(data_rows, 128),
               "Expected the input scaling factor matrix to have ", DIVUP<size_t>(data_rows, 128),
               " rows, but it has ", input_scale_inv_rows, " rows instead.");
    NVTE_CHECK(input_scale_inv_cols == DIVUP<size_t>(data_cols, 512) * 4,
               "Expected the input scaling factor matrix to have ",
               DIVUP<size_t>(data_cols, 512) * 4, " columns, but it has ", input_scale_inv_cols,
               " columns instead.");

    swizzle_kernel_2d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, data_rows,
                                     data_cols, stream);
  }
}

void swizzle_grouped_block_scaling_to_mxfp8_scaling_factors(const GroupedTensor* input,
                                                            GroupedTensor* output,
                                                            cudaStream_t stream) {
  using namespace swizzle_kernel_grouped;

  const NVTEScalingMode scaling_mode = input->scaling_mode;
  NVTE_CHECK(scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D,
             "Input grouped tensor must be a block scaling tensor");
  NVTE_CHECK(output->scaling_mode == NVTE_MXFP8_1D_SCALING,
             "Output grouped tensor must be an mxfp8 tensor");
  NVTE_CHECK(output->with_gemm_swizzled_scales,
             "Expected output grouped tensor with scales in GEMM swizzled format.");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Input and output grouped tensors must have the same number of tensors");
  NVTE_CHECK(input->logical_shape.ndim == 2, "Input grouped tensor must have a 2D logical shape");
  const bool varying_first = input->first_dims.has_data();
  const bool varying_last = input->last_dims.has_data();
  NVTE_CHECK(!(varying_first && varying_last),
             "Converting FP8 block scaling to MXFP8 scales does not support grouped tensors whose "
             "first and last dims both vary");
  // The kernel reads the per-tensor dims as int64.
  NVTE_CHECK(!varying_first || input->first_dims.dtype == DType::kInt64,
             "Grouped tensor first_dims must have int64 dtype");
  NVTE_CHECK(!varying_last || input->last_dims.dtype == DType::kInt64,
             "Grouped tensor last_dims must have int64 dtype");

  const size_t bound = grouped_mxfp8_scale_bytes_upper_bound(*input);
  if (bound == 0) {
    return;
  }

  NVTE_CHECK(input->data.dptr != nullptr, "Input must have rowwise data");
  NVTE_CHECK(output->data.dptr == input->data.dptr, "Output must share data with input");
  NVTE_CHECK(input->data.dtype == DType::kFloat8E4M3 || input->data.dtype == DType::kFloat8E5M2,
             "Input data must have FP8E4M3 or FP8E5M2 dtype to be compatible with MXFP8");
  NVTE_CHECK(output->data.dtype == input->data.dtype,
             "Output data must have the same dtype as input data");
  NVTE_CHECK(input->scale_inv.dptr != nullptr && input->scale_inv.dtype == DType::kFloat32,
             "Input must have FP32 rowwise scaling factors");
  NVTE_CHECK(output->scale_inv.dptr != nullptr && output->scale_inv.dtype == DType::kFloat8E8M0,
             "Output must have E8M0 rowwise scaling factors");
  NVTE_CHECK(output->scale_inv.numel() >= bound, "Output scaling factor buffer holds ",
             output->scale_inv.numel(), " bytes, but the conversion may write up to ", bound,
             " bytes.");
  NVTE_CHECK(is_aligned_ptr(input->scale_inv.dptr, alignof(uint4)),
             "Input scaling factor pointer must be aligned to ", alignof(uint4), " bytes");
  NVTE_CHECK(is_aligned_ptr(output->scale_inv.dptr, alignof(uint4)),
             "Output scaling factor pointer must be aligned to ", alignof(uint4), " bytes");

  const size_t uniform_first = varying_first ? 0 : input->get_common_first_dim();
  const size_t uniform_last = varying_last ? 0 : input->get_common_last_dim();
  if (scaling_mode == NVTE_BLOCK_SCALING_1D && !varying_first) {
    NVTE_CHECK(uniform_first % 4 == 0, "Input tensor must not have any padding scaling factors");
  }
  const auto* first_dims =
      varying_first ? reinterpret_cast<const int64_t*>(input->first_dims.dptr) : nullptr;
  const auto* last_dims =
      varying_last ? reinterpret_cast<const int64_t*>(input->last_dims.dptr) : nullptr;

  const size_t max_blocks = DIVUP(DIVUP(bound, kTileOutputBytes), size_t{WARPS_PER_TB});
  const size_t sm_count = static_cast<size_t>(cuda::sm_count(cuda::current_device()));
  const dim3 grid_dim{static_cast<uint32_t>(std::min(max_blocks, sm_count * kMaxBlocksPerSM))};
  const dim3 block_dim{WARPS_PER_TB * WARP_SIZE};
  const auto* in = reinterpret_cast<const float*>(input->scale_inv.dptr);
  auto* out = reinterpret_cast<uint8_t*>(output->scale_inv.dptr);
  const size_t smem_bytes = table_smem_bytes(input->num_tensors);
  auto launch = [&](auto kernel) {
    constexpr size_t kDefaultMaxSmemBytes = 48 * 1024;
    if (smem_bytes > kDefaultMaxSmemBytes) {
      int max_smem_bytes = 0;
      NVTE_CHECK_CUDA(cudaDeviceGetAttribute(
          &max_smem_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda::current_device()));
      NVTE_CHECK(smem_bytes <= static_cast<size_t>(max_smem_bytes), "Too many tensors (",
                 input->num_tensors, ") to convert FP8 block scaling to MXFP8 scales: needs ",
                 smem_bytes, " bytes of shared memory, but the device allows ", max_smem_bytes,
                 ".");
      NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem_bytes)));
    }
    kernel<<<grid_dim, block_dim, smem_bytes, stream>>>(
        in, out, first_dims, last_dims, uniform_first, uniform_last, input->num_tensors);
  };
  if (scaling_mode == NVTE_BLOCK_SCALING_2D) {
    launch(grouped_swizzle_block_scaling_to_mxfp8_kernel<true>);
  } else {
    launch(grouped_swizzle_block_scaling_to_mxfp8_kernel<false>);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(const NVTETensor input, NVTETensor output,
                                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_block_scaling_to_mxfp8_scaling_factors);
  using namespace transformer_engine;
  swizzle_block_scaling_to_mxfp8_scaling_factors(convertNVTETensorCheck(input),
                                                 convertNVTETensorCheck(output), stream);
}

void nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors(const NVTEGroupedTensor input,
                                                                 NVTEGroupedTensor output,
                                                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_grouped_block_scaling_to_mxfp8_scaling_factors);
  using namespace transformer_engine;
  swizzle_grouped_block_scaling_to_mxfp8_scaling_factors(
      convertNVTEGroupedTensorCheck(input), convertNVTEGroupedTensorCheck(output), stream);
}
