/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include <limits>
// Stringstream is a big hammer, but I want to rely on operator<< for dtype.
#include <sstream>

#include "common/recipe/recipe_common.cuh"
#include "common/utils.cuh"
#include "multi_tensor_apply.cuh"
#include "type_shim.h"

#define BLOCK_SIZE 256

struct ComputeScaleAndScaleInvFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<3> &tl,  // NOLINT(*)
                                             float max_fp8, bool force_pow_2_scales,
                                             float epsilon) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float *amax = reinterpret_cast<float *>(tl.addresses[0][tensor_loc]);
    amax += chunk_idx * chunk_size;

    float *scale = reinterpret_cast<float *>(tl.addresses[1][tensor_loc]);
    scale += chunk_idx * chunk_size;

    float *scale_inv = reinterpret_cast<float *>(tl.addresses[2][tensor_loc]);
    scale_inv += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    for (int i_start = threadIdx.x; i_start < n && i_start < chunk_size; i_start += blockDim.x) {
      float scale_val = transformer_engine::compute_scale_from_amax(
          amax[i_start], max_fp8, force_pow_2_scales, epsilon, std::numeric_limits<float>::max());
      scale[i_start] = scale_val;
      transformer_engine::reciprocal(scale_inv + i_start, scale_val);
    }
  }
};

void multi_tensor_compute_scale_and_scale_inv_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    float max_fp8, bool force_pow_2_scales, float epsilon) {
  using namespace at;

  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                        ComputeScaleAndScaleInvFunctor(), max_fp8, force_pow_2_scales, epsilon);
  AT_CUDA_CHECK(cudaGetLastError());
}
