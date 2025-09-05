/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/swizzle.h>

#include <cstdint>

#include "../common.h"
#include "../util/logging.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace {
  constexpr uint32_t WARP_SIZE = 32;

  template<typename T>
  T __device__ convert_block_scaling_to_mxfp8_scaling_factors(T sf) {
    sf = (sf << 1) | (sf >> 7);
    sf = sf | (sf >> 16);
    return sf;
  }
} // namespace
namespace swizzle_kernel_1d {
  constexpr uint32_t WARPS_X_PER_TB = 2; // configurable
  constexpr uint32_t WARPS_Y_PER_TB = 2; // configurable
  
  union sf_block {
    // A thread loads 4 float scaling factors
    // float4 f32x4; conceptually
    
    // A thread treats them as two uint64 for swizzling
    uint64_t u64[2];

    // A thread treats them as four uint32 for storing to memory
    uint32_t u32[4];
    uint4 u32x4;
  };

  __device__ __forceinline__ uint32_t transpose4x4_bytes(uint32_t v, unsigned lane_in_quad, unsigned mask) {
    uint32_t u;
    u = __shfl_xor_sync(mask, v, 1, 4);
    v = (lane_in_quad & 1) ? __byte_perm(v, u, 0x7531)  // [b,f,d,h]
                        : __byte_perm(v, u, 0x6240); // [a,e,c,g]
    u = __shfl_xor_sync(mask, v, 2, 4);
    v = (lane_in_quad & 2) ? __byte_perm(v, u, 0x7632)  // [x2,x3,y2,y3]
                        : __byte_perm(v, u, 0x5410); // [x0,x1,y0,y1]
    return v;
  }

  __device__ __forceinline__ uint4 expand_bytes1(uint32_t x) {
    uint4 r;
    r.x = __byte_perm(x, 0, 0x0000);  // [a a a a]
    r.y = __byte_perm(x, 0, 0x1111);  // [b b b b]
    r.z = __byte_perm(x, 0, 0x2222);  // [c c c c]
    r.w = __byte_perm(x, 0, 0x3333);  // [d d d d]
    return r;
  }

  void __global__ __launch_bounds__(WARPS_X_PER_TB * WARPS_Y_PER_TB * WARP_SIZE)
  swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel(const void* __restrict__ const in,
                                                           void* __restrict__ const out,
                                                           const uint32_t tiles_x,
                                                           const uint32_t tiles_y,
                                                           const uint32_t in_y_stride,
                                                           const uint32_t out_y_stride) {
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

    // calculate warp input base pointer
    constexpr uint32_t in_x_stride = WARP_SIZE * sizeof(sf_block);
    const void* const warp_src = in + in_tile_y * in_y_stride + in_tile_x * in_x_stride;

    const sf_block* const tile = reinterpret_cast<const sf_block*>(warp_src);
    sf_block sf = tile[(lane % 4) * 8];

    uint32_t packed_exponents = (sf.u32[0] << 1) | (sf.u32[1] >> 7) | (sf.u32[2] >> 15) | (sf.u32[3] >> 23);
    
    constexpr uint32_t ACTIVE_MASK = 0xFFFFFFFF; // no divergent branches

    packed_exponents = transpose4x4_bytes(packed_exponents, lane % 4, ACTIVE_MASK);

    sf.u32x4 = expand_bytes1(packed_exponents);

    // store them cooperatively for 512 1x32 tiles in a 128x128 tile
    constexpr uint32_t out_x_stride = 512;
    void* const warp_dst = out + out_tile_y * out_y_stride + out_tile_x * out_x_stride;
    reinterpret_cast<uint4*>(warp_dst)[lane] = sf.u32x4;
  }

  void launch_kernel(const void* const in, void* const out, uint32_t data_rows, uint32_t data_cols,
                     cudaStream_t stream) {
    NVTE_CHECK(is_aligned_ptr(in, alignof(sf_block)),
               "Input scaling factor pointer must be aligned to ", alignof(sf_block), " bytes");
    NVTE_CHECK(is_aligned_ptr(out, alignof(uint4)),
               "Output scaling factor pointer must be aligned to ", alignof(uint4), " bytes");
    NVTE_CHECK(data_rows % 128 == 0,
               "Input scaling factors have to be available for full 128x128 tiles");

    const uint32_t tiles_x = DIVUP(data_cols, 128u);
    const uint32_t tiles_y = DIVUP(data_rows, 128u);
    const dim3 grid_dim{DIVUP(tiles_x, WARPS_X_PER_TB), DIVUP(tiles_y, WARPS_Y_PER_TB), 1};
    const dim3 block_dim{WARP_SIZE, WARPS_Y_PER_TB, WARPS_X_PER_TB};

    const uint32_t input_scale_inv_cols = DIVUP(data_rows, 4u) * 4;
    const uint32_t in_y_stride = input_scale_inv_cols * sizeof(float);
    
    const uint32_t out_y_stride = tiles_x * 512;

    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in, out, tiles_x, tiles_y, in_y_stride, out_y_stride);
  }
}  // namespace swizzle_kernel_1d
namespace swizzle_kernel_2d {
  constexpr uint32_t WARPS_X_PER_TB = 2; // configurable
  constexpr uint32_t WARPS_Y_PER_TB = 2; // configurable

  void __global__ __launch_bounds__(WARPS_X_PER_TB * WARPS_Y_PER_TB * WARP_SIZE)
  swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel(const void* __restrict__ const in,
                                                           void* __restrict__ const out,
                                                           const uint32_t tiles_x,
                                                           const uint32_t tiles_y,
                                                           const uint32_t in_y_stride,
                                                           const uint32_t out_y_stride) {
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

    // load scaling factor for a 128x128 tile
    constexpr uint32_t in_x_stride = sizeof(float);
    const void* const warp_src = in + in_tile_y * in_y_stride + in_tile_x * in_x_stride;
    uint32_t sf = reinterpret_cast<const uint32_t*>(warp_src)[0];

    // convert it to four scaling factors for 1x32 tiles
    sf = convert_block_scaling_to_mxfp8_scaling_factors(sf);

    // broadcast it to sixteen scaling factors for 1x32 tiles
    const uint4 sf4{sf, sf, sf, sf};

    // store it cooperatively for 512 1x32 tiles in a 128x128 tile
    constexpr uint32_t out_x_stride = 512;
    void* const warp_dst = out + out_tile_y * out_y_stride + out_tile_x * out_x_stride;
    reinterpret_cast<uint4*>(warp_dst)[lane] = sf4;
  }

  void launch_kernel(const void* const in, void* const out, uint32_t data_rows, uint32_t data_cols,
                     cudaStream_t stream) {
    NVTE_CHECK(is_aligned_ptr(in, alignof(float)),
               "Input scaling factor pointer must be aligned to ", alignof(float), " bytes");
    NVTE_CHECK(is_aligned_ptr(out, alignof(uint4)),
               "Output scaling factor pointer must be aligned to ", alignof(uint4), " bytes");

    const uint32_t tiles_x = DIVUP(data_cols, 128u);
    const uint32_t tiles_y = DIVUP(data_rows, 128u);
    const dim3 grid_dim{DIVUP(tiles_x, WARPS_X_PER_TB), DIVUP(tiles_y, WARPS_Y_PER_TB), 1};
    const dim3 block_dim{WARP_SIZE, WARPS_Y_PER_TB, WARPS_X_PER_TB};
    
    const uint32_t input_scale_inv_cols = DIVUP(data_cols, 512u) * 4;
    const uint32_t in_y_stride = input_scale_inv_cols * sizeof(float);
    
    const uint32_t out_y_stride = tiles_x * 512;
    
    swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in, out, tiles_x, tiles_y, in_y_stride, out_y_stride);
  }
} // namespace swizzle_kernel_2d

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

  NVTE_CHECK(input->scale_inv.dptr != nullptr, "Input must have rowwise scaling factors");
  NVTE_CHECK(input->scale_inv.dtype == DType::kFloat32, "Input must have FP32 scaling factors");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Output must have rowwise scaling factors");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat8E8M0,
             "Output must have E8M0 scaling factors");

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
               "Expected the input scaling factor matrix to have ",
               DIVUP<size_t>(data_cols, 128), " rows, but it has ", input_scale_inv_rows,
               " rows instead.");
    NVTE_CHECK(input_scale_inv_cols == DIVUP<size_t>(data_rows, 4) * 4,
               "Expected the input scaling factor matrix to have ",
               DIVUP<size_t>(data_rows, 4) * 4, "columns, but it has ", input_scale_inv_cols,
               " columns instead.");

    swizzle_kernel_1d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, data_rows,
                                     data_cols, stream);
  } else { // scaling_mode == NVTE_BLOCK_SCALING_2D
    NVTE_CHECK(input_scale_inv_rows == DIVUP<size_t>(data_rows, 128),
               "Expected the output scaling factor matrix to have ",
               DIVUP<size_t>(data_rows, 128), " rows, but it has ", input_scale_inv_rows,
               " rows instead.");
    NVTE_CHECK(input_scale_inv_cols == DIVUP<size_t>(data_cols, 512) * 4,
               "Expected the input scaling factor matrix to have ",
               DIVUP<size_t>(data_cols, 512) * 4, "columns, but it has ", input_scale_inv_cols,
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
                                                 convertNVTETensorCheck(output),
                                                 stream);
}
