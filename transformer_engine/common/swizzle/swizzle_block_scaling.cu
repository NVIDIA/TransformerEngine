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
  };

  void __global__ swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel(const void* const in,
                                                                           void* const out,
                                                                           size_t sz) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // uniform branch (sz % 128 == 0)
    if (tid * SF_PER_THREAD >= sz) {
      return;
    }

    // load scaling factors for four 1x128 tiles
    sf_block sf = reinterpret_cast<const sf_block*>(in)[tid];

    // convert them to sixteen scaling factors for 1x32 tiles
    for (int i = 0; i < 2; ++i) {
      sf.u64[i] = convert_block_scaling_to_mxfp8_scaling_factors(sf.u64[i]);
    }

    // store them in swizzled manner for 512 1x32 tiles in a 128x128 tile
    uint32_t* const dst = reinterpret_cast<uint32_t*>(out) + tid;
    for (int i = 0; i < 4; ++i) {
      dst[i * MXFP8_SWIZZLE_STRIDE] = sf.u32[i];
    }
  }

  void launch_kernel(const void* const in, void* const out, size_t sz, cudaStream_t stream) {
    NVTE_CHECK(sz % 128 == 0, "Input has to be divisible into 128x128 tiles");
    const size_t blocks = DIVUP(sz, SF_PER_TB);
    swizzle_block_scaling_1d_to_mxfp8_scaling_factors_kernel<<<blocks, TB_SIZE, 0, stream>>>(in,
                                                                                             out,
                                                                                             sz);
  }
}  // namespace swizzle_kernel_1d
namespace swizzle_kernel_2d {
  constexpr size_t WARPS_PER_TB = 4; // configurable
  constexpr size_t SF_PER_WARP = 1;
  constexpr size_t TB_SIZE = WARP_SIZE * WARPS_PER_TB;
  constexpr size_t SF_PER_TB = WARPS_PER_TB * SF_PER_WARP;

  void __global__ swizzle_block_scaling_2d_to_mxfp8_scaling_factors_kernel(const void* const in,
                                                                           void* const out,
                                                                           size_t sz) {
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

  void launch_kernel(const void* const in, void* const out, size_t sz, cudaStream_t stream) {
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

  const size_t numel = input->scale_inv.numel();
  if (scaling_mode == NVTE_BLOCK_SCALING_1D) {
    NVTE_CHECK(output->scale_inv.numel() == numel * 4,
               "Output should have 4 scaling factors (for 4 1x32 tiles) "
               "for every input scaling factor (for a 1x128 tile)");

    swizzle_kernel_1d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, numel, stream);
  } else { // scaling_mode == NVTE_BLOCK_SCALING_2D
    NVTE_CHECK(output->scale_inv.numel() == numel * 512,
               "Output should have 512 scaling factors (for 512 1x32 tiles) "
               "for every input scaling factor (for a 128x128 tile)");

    swizzle_kernel_2d::launch_kernel(input->scale_inv.dptr, output->scale_inv.dptr, numel, stream);
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
