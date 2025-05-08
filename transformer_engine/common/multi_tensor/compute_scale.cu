/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <limits>
// Stringstream is a big hammer, but I want to rely on operator<< for dtype.
#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include <sstream>

#include "../recipe/recipe_common.cuh"
#include "../utils.cuh"
#include "multi_tensor_apply.cuh"

namespace transformer_engine {
namespace multi_tensor_compute_scale {

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

void multi_tensor_compute_scale_and_scale_inv_cuda(int chunk_size, Tensor noop_flag,
                                                   std::vector<std::vector<Tensor *>> tensor_lists,
                                                   float max_fp8, bool force_pow_2_scales,
                                                   float epsilon, const int device_id,
                                                   cudaStream_t stream) {
  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                        ComputeScaleAndScaleInvFunctor(), device_id, stream, max_fp8,
                        force_pow_2_scales, epsilon);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_compute_scale
}  // namespace transformer_engine

void nvte_multi_tensor_compute_scale_and_scale_inv_cuda(
    int chunk_size, NVTETensor noop_flag, NVTETensor **tensor_lists, const size_t num_tensor_lists,
    const size_t num_tensors_per_list, float max_fp8, int force_pow_2_scales, float epsilon,
    const int device_id, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_compute_scale_and_scale_inv_cuda);
  using namespace transformer_engine;

  multi_tensor_compute_scale::multi_tensor_compute_scale_and_scale_inv_cuda(
      chunk_size, *reinterpret_cast<Tensor *>(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), max_fp8,
      force_pow_2_scales, epsilon, device_id, stream);
}
