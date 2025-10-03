/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/swizzle.h>

#include <cstdint>
#include <type_traits>

#include "../common.h"
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

template <typename OOBT>
void __global__ __launch_bounds__(WARPS_X_PER_TB* WARPS_Y_PER_TB* WARP_SIZE)
    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel(
        const void* __restrict__ const in, void* __restrict__ const out, const uint32_t tiles_x,
        const uint32_t tiles_y, const uint32_t in_y_stride, const uint32_t out_y_stride,
        OOBT first_oob) {
  // resolve kernel variant
  constexpr bool no_oob = std::is_same_v<OOBT, no_oob_tag_t>;
  static_assert(no_oob || std::is_same_v<OOBT, uint32_t>);

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
  const uint32_t in_tile_y = out_tile_x;
  const uint32_t in_tile_x = out_tile_y;

  // bounds check; uniform branch
  if (out_tile_y >= tiles_y || out_tile_x >= tiles_x) {
    return;
  }

  // calculate this warp's input base pointer
  constexpr uint32_t in_x_stride = WARP_SIZE * sizeof(uint4);
  const void* const warp_src = in + in_tile_y * in_y_stride + in_tile_x * in_x_stride;

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
  uint32_t packed_exponents = (sf.x >> 23) | (sf.y >> 15) | (sf.z >> 7) | (sf.w << 1);

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
  void* const warp_dst = out + out_tile_y * out_y_stride + out_tile_x * out_x_stride;
  reinterpret_cast<uint4*>(warp_dst)[lane] = sf;
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
  const uint32_t in_tile_y = out_tile_y;
  const uint32_t in_tile_x = out_tile_x;

  // bounds check; uniform branch
  if (out_tile_y >= tiles_y || out_tile_x >= tiles_x) {
    return;
  }

  // calculate this warp's input base pointer
  constexpr uint32_t in_x_stride = sizeof(float);
  const void* const warp_src = in + in_tile_y * in_y_stride + in_tile_x * in_x_stride;

  // load scaling factor for this warp's 128x128 tile
  uint32_t sf = *reinterpret_cast<const uint32_t*>(warp_src);

  // broadcast it to four scaling factors for 1x32 tiles
  sf = (sf << 1) | (sf >> 7);
  sf = sf | (sf >> 16);

  // broadcast it to sixteen scaling factors for 1x32 tiles
  const uint4 sf4{sf, sf, sf, sf};

  // store it cooperatively for 512 1x32 tiles in a 128x128 tile
  constexpr uint32_t out_x_stride = 512;
  void* const warp_dst = out + out_tile_y * out_y_stride + out_tile_x * out_x_stride;
  reinterpret_cast<uint4*>(warp_dst)[lane] = sf4;
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

}  // namespace transformer_engine

void nvte_swizzle_block_scaling_to_mxfp8_scaling_factors(const NVTETensor input, NVTETensor output,
                                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_block_scaling_to_mxfp8_scaling_factors);
  using namespace transformer_engine;
  swizzle_block_scaling_to_mxfp8_scaling_factors(convertNVTETensorCheck(input),
                                                 convertNVTETensorCheck(output), stream);
}
