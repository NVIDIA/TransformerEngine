/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_nvfp4.cuh
 *  \brief CUDA kernels to dequantize from NVFP4.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_NVFP4_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"

#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif  // FP4_TYPE_SUPPORTED

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {
namespace dequantize_kernel {
#if FP4_TYPE_SUPPORTED
template <typename OType>
__global__ void __launch_bounds__(512)
    dequantize_fp4_kernel(const void *const input, OType *output, const fp8e4m3 *const scales,
                          const float *const tensor_amax, const size_t N, const size_t M,
                          const size_t scale_stride) {
  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t x = thread_idx % M;
  const size_t y = thread_idx / M;

  if (y >= N) {
    return;
  }

  union fp4vec {
    uint64_t vec;
    fp4e2m1x4 small_vec[4];
  };
  using OVec = Vec<OType, 4>;
  const uint64_t *const input_vectorized = reinterpret_cast<const uint64_t *>(input);
  OVec *output_vec = reinterpret_cast<OVec *>(output);

  const size_t my_index = x + y * M;
  const size_t my_scale_index = x + y * scale_stride;
  const size_t my_output_index = (x + y * M) * 4;
  fp4vec value;
  value.vec = input_vectorized[my_index];
  fp8e4m3 scale = scales[my_scale_index];
  float amax = *tensor_amax;
  constexpr float factor_inv = 1.0 / (6.0 * 448.0);
  float final_scale = static_cast<float>(scale) * amax * factor_inv;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float4 current = static_cast<float4>(value.small_vec[i]);
    OVec out;
    out.data.elt[0] = static_cast<OType>(current.x * final_scale);
    out.data.elt[1] = static_cast<OType>(current.y * final_scale);
    out.data.elt[2] = static_cast<OType>(current.z * final_scale);
    out.data.elt[3] = static_cast<OType>(current.w * final_scale);
    output_vec[my_output_index + i] = out;
  }
}
#endif  // FP4_TYPE_SUPPORTED
}  // namespace dequantize_kernel

inline void dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
#if FP4_TYPE_SUPPORTED
  using namespace dequantize_kernel;
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output");
  NVTE_CHECK(input.data.dtype == DType::kFloat4E2M1, "Input must have FP4 type.");
  NVTE_CHECK(is_high_precision_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  constexpr int FP4_BLOCK_SIZE = 16;
  const size_t N = input.flat_first_dim();
  const size_t M = input.flat_last_dim();

  NVTE_CHECK(M % FP4_BLOCK_SIZE == 0, "Last dimension of FP4 tensors needs to be divisible by ",
             FP4_BLOCK_SIZE, ", but got ", input.data.shape, ".");

  const size_t Mread = M / FP4_BLOCK_SIZE;
  const size_t total = N * Mread;
  const size_t threads = 512;
  const size_t blocks = DIVUP(total, threads);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->data.dtype, OType,

      dequantize_fp4_kernel<<<blocks, threads, 0, stream>>>(
          input.data.dptr, reinterpret_cast<OType *>(output->data.dptr),
          reinterpret_cast<fp8e4m3 *>(input.scale_inv.dptr),
          reinterpret_cast<float *>(input.amax.dptr), N, Mread,
          input.scale_inv.shape.back()););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif  // FP4_TYPE_SUPPORTED
}
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_NVFP4_CUH_
