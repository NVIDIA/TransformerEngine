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
  constexpr size_t WARP_SIZE = 32;

  template<typename T>
  T __device__ convert_block_scaling_to_mxfp8_scaling_factors(T sf) {
    sf = (sf << 1) | (sf >> 7);
    sf = sf | (sf >> 16);
    return sf;
  }
} // namespace
namespace swizzle_kernel_1d {
  constexpr size_t WARPS_PER_TB = 4; // configurable
  constexpr size_t SF_PER_THREAD = 4;
  constexpr size_t SF_PER_WARP = SF_PER_THREAD * WARP_SIZE;
  constexpr size_t TB_SIZE = WARP_SIZE * WARPS_PER_TB;
  constexpr size_t SF_PER_TB = WARPS_PER_TB * SF_PER_WARP;
  constexpr size_t MXFP8_SWIZZLE_STRIDE = 32;
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
                                                                           const size_t rows,
                                                                           const size_t cols) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t warp = tid / WARP_SIZE;
    const size_t lane = tid % WARP_SIZE;

    // uniform branch
    const size_t sz = rows * cols;
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
    constexpr size_t ACTIVE_MASK = 0xFFFFFFFF; // no divergent branches
    const sf_block offered = sf;
    for (int i = 0; i < 4; ++i) {
        const size_t dst_comp = (lane + i) % 4;
        const size_t src_lane = (lane / 4) + ((lane + i) % 4) * 8;
        const size_t offered_comp = (lane / 8 + 4 - i) % 4;
        
        sf.u32[dst_comp] = __shfl_sync(ACTIVE_MASK, offered.u32[offered_comp], src_lane);
    }

    // store them in swizzled manner for 512 1x32 tiles in a 128x128 tile
    const size_t dst_tile_col = warp % rows;
    const size_t dst_tile_row = warp / rows;
    constexpr size_t TILE_SZ = 512;
    void* const dst_tile = out + dst_tile_row * rows * TILE_SZ + dst_tile_col * TILE_SZ;
    reinterpret_cast<uint4*>(dst_tile)[lane] = sf.u32x4;
  }

  void launch_kernel(const void* const in, void* const out, size_t rows, size_t cols,
                     cudaStream_t stream) {
    static_assert(SF_PER_WARP == 128);
    NVTE_CHECK(cols % SF_PER_WARP == 0, "Input data has to be divisible into 128x128 tiles");
    const size_t sz = rows * cols;
    const size_t blocks = DIVUP(sz, SF_PER_TB);
    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel<<<blocks, TB_SIZE, 0, stream>>>(in,
                                                                                             out,
                                                                                             rows,
                                                                                             cols);
  }
}  // namespace swizzle_kernel_1d
namespace swizzle_kernel_2d {
  constexpr size_t WARPS_PER_TB = 4; // configurable
  constexpr size_t SF_PER_WARP = 1;
  constexpr size_t TB_SIZE = WARP_SIZE * WARPS_PER_TB;
  constexpr size_t SF_PER_TB = WARPS_PER_TB * SF_PER_WARP;

  void __global__ swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel(const void* const in,
                                                                           void* const out,
                                                                           const size_t sz) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    static_assert(SF_PER_WARP == 1);
    const size_t warp = tid / WARP_SIZE;
    const size_t lane = tid % WARP_SIZE;

    // uniform branch
    if (warp >= sz) {
      return;
    }

    // load scaling factor for a 128x128 tile
    uint32_t sf = reinterpret_cast<const uint32_t*>(in)[warp];

    // convert it to four scaling factors for 1x32 tiles
    sf = convert_block_scaling_to_mxfp8_scaling_factors(sf);

    // boradcast it to sixteen scaling factors for 1x32 tiles
    const uint4 sf4{sf, sf, sf, sf};

    // store it cooperatively for 512 1x32 tiles in a 128x128 tile
    void* const warp_dst = out + 512 * warp;
    void* const dst = warp_dst + lane * sizeof(uint4);
    *reinterpret_cast<uint4*>(dst) = sf4;
  }

  void launch_kernel(const void* const in, void* const out, size_t rows, size_t cols,
                     cudaStream_t stream) {
    const size_t sz = rows * cols;
    const size_t blocks = DIVUP(sz, SF_PER_TB);
    swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel<<<blocks, TB_SIZE, 0, stream>>>(in,
                                                                                             out,
                                                                                             sz);
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

  const size_t input_rows = input->scale_inv.shape[0];
  const size_t input_cols = input->scale_inv.shape[1];
  const size_t output_rows = output->scale_inv.shape[0];
  const size_t output_cols = output->scale_inv.shape[1];
  if (scaling_mode == NVTE_BLOCK_SCALING_1D) {
    NVTE_CHECK(output_rows == input_cols && output_cols == input_rows * 4,
               "Output MXFP8 scaling factors should have shape (", input_cols, ", ",
               input_rows * 4, "), but has shape (", output_rows, ", ", output_cols, ")");

    swizzle_kernel_1d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, input_rows,
                                     input_cols, stream);
  } else { // scaling_mode == NVTE_BLOCK_SCALING_2D
    NVTE_CHECK(output_rows == input_rows * 128 && output_cols == input_cols * 4,
               "Output MXFP8 scaling factors should have shape (", input_rows * 128, ", ",
               input_cols * 4, "), but has shape (", output_rows, ", ", output_cols, ")");

    swizzle_kernel_2d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, input_rows,
                                     input_cols, stream);
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
