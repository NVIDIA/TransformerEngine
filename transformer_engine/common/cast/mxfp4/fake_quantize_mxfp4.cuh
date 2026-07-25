/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/


#ifndef TRANSFORMER_ENGINE_CAST_MXFP4_FAKE_QUANTIZE_MXFP4_CUH_
#define TRANSFORMER_ENGINE_CAST_MXFP4_FAKE_QUANTIZE_MXFP4_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace fake_quantize_kernel {

constexpr size_t MXFP4_BLOCK_SIZE = 32;
constexpr size_t THREADS_PER_CHUNK = 256;
constexpr float E2M1_MAX = 6.0f;
constexpr int SCALE_EXP_MIN = -126;
constexpr int SCALE_EXP_MAX = 125;

__device__ __forceinline__ float round_e2m1_magnitude(const float y) {
  if (y <= 2.0f) {
    return rintf(y * 2.0f) * 0.5f;
  }
  if (y <= 4.0f) {
    return rintf(y);
  }
  return rintf(y * 0.5f) * 2.0f;
}

__device__ __forceinline__ int block_scale_exponent_from_bits(const int bits) {
  const int exp_field = bits >> 23;
  const int mantissa = bits & 0x7FFFFF;
  int e = exp_field - 129 + (mantissa > 0x400000 ? 1 : 0);
  if (exp_field == 0) {
    e = SCALE_EXP_MIN;
  }
  return max(SCALE_EXP_MIN, min(SCALE_EXP_MAX, e));
}

__device__ __forceinline__ float exp2_from_int(const int e) {
  return __int_as_float((e + 127) << 23);
}

__device__ __forceinline__ float mul_rn_no_ftz(const float a, const float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

template <typename IType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    fake_quantize_mxfp4_kernel(const IType *__restrict__ input, IType *__restrict__ output,
                               const size_t num_vectors) {
  constexpr size_t VEC_ELTS = 16 / sizeof(IType);
  constexpr int LANES_PER_BLOCK = MXFP4_BLOCK_SIZE / VEC_ELTS;

  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t vec_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vec_idx < num_vectors; vec_idx += stride) {
    Vec<IType, VEC_ELTS> in_vec;
    in_vec.load_from(input, vec_idx);

    float elts[VEC_ELTS];
    int amax_bits = 0;
#pragma unroll
    for (int i = 0; i < static_cast<int>(VEC_ELTS); ++i) {
      elts[i] = static_cast<float>(in_vec.data.elt[i]);
      amax_bits = max(amax_bits, __float_as_int(elts[i]) & 0x7fffffff);
    }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned group_mask =
        (LANES_PER_BLOCK == 32)
            ? 0xffffffffu
            : (((1u << LANES_PER_BLOCK) - 1u) << (lane & ~(LANES_PER_BLOCK - 1u)));
#pragma unroll
    for (int offset = LANES_PER_BLOCK / 2; offset > 0; offset >>= 1) {
      amax_bits = max(amax_bits, __shfl_xor_sync(group_mask, amax_bits, offset));
    }
    const int nonfinite = amax_bits >= 0x7f800000;

    Vec<IType, VEC_ELTS> out_vec;
    if (nonfinite) {
      const float qnan = __int_as_float(0x7fc00000);
#pragma unroll
      for (int i = 0; i < static_cast<int>(VEC_ELTS); ++i) {
        out_vec.data.elt[i] = static_cast<IType>(qnan);
      }
    } else {
      const int e = (amax_bits > 0) ? block_scale_exponent_from_bits(amax_bits) : 0;
      const float scale = exp2_from_int(e);
      const float inv_scale = exp2_from_int(-e);
#pragma unroll
      for (int i = 0; i < static_cast<int>(VEC_ELTS); ++i) {
        const float av = __int_as_float(__float_as_int(elts[i]) & 0x7fffffff);
        const float y = fminf(mul_rn_no_ftz(av, inv_scale), E2M1_MAX);
        const float q = copysignf(round_e2m1_magnitude(y), elts[i]);
        out_vec.data.elt[i] = static_cast<IType>(mul_rn_no_ftz(q, scale));
      }
    }
    out_vec.store_to(output, vec_idx);
  }
}

}

template <typename IType>
void fake_quantize_mxfp4_launch(const IType *input, IType *output, const size_t numel,
                                cudaStream_t stream) {
  using namespace fake_quantize_kernel;
  constexpr size_t VEC_ELTS = 16 / sizeof(IType);
  const size_t num_vectors = numel / VEC_ELTS;
  const size_t blocks_needed = (num_vectors + THREADS_PER_CHUNK - 1) / THREADS_PER_CHUNK;
  const int grid = static_cast<int>(blocks_needed < 65535 ? blocks_needed : 65535);
  fake_quantize_mxfp4_kernel<IType>
      <<<grid, THREADS_PER_CHUNK, 0, stream>>>(input, output, num_vectors);
}

}
}
}

#endif
