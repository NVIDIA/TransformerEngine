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
  constexpr uint32_t WARPS_X_PER_TB = 2;
  constexpr uint32_t WARPS_Y_PER_TB = 2;
  constexpr uint32_t WARPS_PER_TB = WARPS_X_PER_TB * WARPS_Y_PER_TB;
  constexpr uint32_t SF_PER_THREAD = 4;
  constexpr uint32_t SF_PER_WARP = SF_PER_THREAD * WARP_SIZE;
  constexpr uint32_t TB_SIZE = WARP_SIZE * WARPS_PER_TB;
  constexpr uint32_t SF_PER_TB = WARPS_PER_TB * SF_PER_WARP;
  constexpr uint32_t MXFP8_SWIZZLE_STRIDE = 32;
  static_assert(MXFP8_SWIZZLE_STRIDE == WARP_SIZE);
  
  union sf_block {
    // A thread loads 4 float scaling factors
    // float4 f32x4; conceptually
    static_assert(SF_PER_THREAD == 4);
    
    // A thread treats them as two uint64 for swizzling
    uint64_t u64[2];

    // A thread treats them as four uint32 for storing to memory
    uint32_t u32[4];
    uint4 u32x4;
  };

  void __global__ swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel(const void* const in,
                                                                           void* const out,
                                                                           const uint32_t rows,
                                                                           const uint32_t cols) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warp = tid / WARP_SIZE;
    const uint32_t lane = tid % WARP_SIZE;

    // uniform branch
    const uint32_t sz = rows * cols;
    if (warp * SF_PER_WARP >= sz) {
      return;
    }

    // load scaling factors for four 1x128 tiles
    sf_block sf = reinterpret_cast<const sf_block*>(in)[tid];

    // convert them to sixteen scaling factors for 1x32 tiles
    for (int i = 0; i < 2; ++i) {
      sf.u64[i] = convert_block_scaling_to_mxfp8_scaling_factors(sf.u64[i]);
    }

    // swizzle the scaling factors
    constexpr uint32_t ACTIVE_MASK = 0xFFFFFFFF; // no divergent branches
    const sf_block offered = sf;
    for (int i = 0; i < 4; ++i) {
        const uint32_t dst_comp = (lane + i) % 4;
        const uint32_t src_lane = (lane / 4) + ((lane + i) % 4) * 8;
        const uint32_t offered_comp = (lane / 8 + 4 - i) % 4;
        
        sf.u32[dst_comp] = __shfl_sync(ACTIVE_MASK, offered.u32[offered_comp], src_lane);
    }

    // store them in swizzled manner for 512 1x32 tiles in a 128x128 tile
    const uint32_t dst_tile_col = warp % rows;
    const uint32_t dst_tile_row = warp / rows;
    constexpr uint32_t TILE_SZ = 512;
    void* const dst_tile = out + dst_tile_row * rows * TILE_SZ + dst_tile_col * TILE_SZ;
    reinterpret_cast<uint4*>(dst_tile)[lane] = sf.u32x4;
  }

  void launch_kernel(const void* const in, void* const out, uint32_t data_rows, uint32_t data_cols,
                     cudaStream_t stream) {
    static_assert(SF_PER_WARP == 128);
    NVTE_CHECK(data_cols % SF_PER_WARP == 0, "Input data has to be divisible into 128x128 tiles");

    const uint32_t tiles_x = DIVUP(data_cols, 128u);
    const uint32_t tiles_y = DIVUP(data_rows, 128u);
    const dim3 grid_dim{DIVUP(tiles_x, WARPS_X_PER_TB), DIVUP(tiles_y, WARPS_Y_PER_TB), 1};
    const dim3 block_dim{WARPS_X_PER_TB, WARPS_Y_PER_TB, WARP_SIZE};

    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in, out, ...);
  }
}  // namespace swizzle_kernel_1d
namespace swizzle_kernel_2d {
  constexpr uint32_t WARPS_X_PER_TB = 1; // configurable
  constexpr uint32_t WARPS_Y_PER_TB = 1; // configurable

  void __global__ __launch_bounds__(WARPS_X_PER_TB * WARPS_Y_PER_TB * WARP_SIZE)
  swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel(const void* __restrict__ const in,
                                                           void* __restrict__ const out,
                                                           const uint32_t tiles_x,
                                                           const uint32_t tiles_y,
                                                           const uint32_t in_y_stride,
                                                           const uint32_t out_y_stride) {
    // load thread indices
    const uint32_t warp_x = threadIdx.x;
    __builtin_assume(warp_x < WARPS_X_PER_TB);
    const uint32_t warp_y = threadIdx.y;
    __builtin_assume(warp_y < WARPS_Y_PER_TB);
    const uint32_t lane = threadIdx.z;
    __builtin_assume(warp_y < WARP_SIZE);

    // compute tile indices
    const uint32_t tile_y = blockIdx.y * WARPS_Y_PER_TB + warp_y;
    const uint32_t tile_x = blockIdx.x * WARPS_X_PER_TB + warp_x;

    // bounds check; uniform branch
    if (tile_y >= tiles_y || tile_x >= tiles_x) {
      return;
    }

    // load scaling factor for a 128x128 tile
    constexpr uint32_t in_x_stride = sizeof(float);
    uint32_t sf = 
        *reinterpret_cast<const uint32_t*>(in + tile_y * in_y_stride + tile_x * in_x_stride);

    // convert it to four scaling factors for 1x32 tiles
    sf = convert_block_scaling_to_mxfp8_scaling_factors(sf);

    // broadcast it to sixteen scaling factors for 1x32 tiles
    const uint4 sf4{sf, sf, sf, sf};

    // store it cooperatively for 512 1x32 tiles in a 128x128 tile
    constexpr uint32_t out_x_stride = 512;
    void* const warp_dst = out + tile_y * out_y_stride + tile_x * out_x_stride;
    reinterpret_cast<uint4*>(warp_dst)[lane] = sf4;
  }

  void launch_kernel(const void* const in, void* const out, uint32_t data_rows, uint32_t data_cols,
                     cudaStream_t stream) {
    const uint32_t tiles_x = DIVUP(data_cols, 128u);
    const uint32_t tiles_y = DIVUP(data_rows, 128u);
    const dim3 grid_dim{DIVUP(tiles_x, WARPS_X_PER_TB), DIVUP(tiles_y, WARPS_Y_PER_TB), 1};
    const dim3 block_dim{WARPS_X_PER_TB, WARPS_Y_PER_TB, WARP_SIZE};
    
    const uint32_t input_scale_inv_cols = DIVUP(data_cols, 512u) * 4;
    const uint32_t output_scale_inv_cols = DIVUP<size_t>(data_cols, 128) * 4;
    
    const uint32_t in_y_stride = input_scale_inv_cols * sizeof(float);
    const uint32_t out_y_stride = output_scale_inv_cols * sizeof(uint8_t);
    
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
