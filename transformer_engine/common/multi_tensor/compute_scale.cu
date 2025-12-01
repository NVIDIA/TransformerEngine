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
#include "../util/ptx.cuh"
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

struct ComputeScaleInvE8M0Functor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *unused,
                                             TensorListMetadata<2> &tl) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    bf16 *amax = reinterpret_cast<bf16 *>(tl.addresses[0][tensor_loc]);
    amax += chunk_idx * chunk_size;

    e8m0_t *scale_inv = reinterpret_cast<e8m0_t *>(tl.addresses[1][tensor_loc]);
    scale_inv += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    for (int i_start = threadIdx.x; i_start < n && i_start < chunk_size; i_start += blockDim.x) {
      scale_inv[i_start] = ptx::float_to_e8m0(static_cast<float>(amax[i_start]) *
                                              Quantized_Limits<fp8e4m3>::max_norm_rcp);
    }
  }
};

void multi_tensor_compute_scale_and_scale_inv_cuda(int chunk_size, Tensor noop_flag,
                                                   std::vector<std::vector<Tensor *>> tensor_lists,
                                                   float max_fp8, bool force_pow_2_scales,
                                                   float epsilon, cudaStream_t stream) {
  multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                        ComputeScaleAndScaleInvFunctor(), stream, max_fp8, force_pow_2_scales,
                        epsilon);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void multi_tensor_compute_scale_inv_e8m0_cuda(int chunk_size,
                                              std::vector<std::vector<Tensor *>> tensor_lists,
                                              cudaStream_t stream) {
  NVTE_CHECK(tensor_lists[0][0]->data.dtype == DType::kBFloat16, "amax should be bf16");
  auto scale_inv_dtype = tensor_lists[1][0]->data.dtype;
  NVTE_CHECK(scale_inv_dtype == DType::kByte || scale_inv_dtype == DType::kFloat8E8M0,
             "scale_inv should be e8m0/uint8");
  Tensor dummy;
  multi_tensor_apply<2>(BLOCK_SIZE, chunk_size, dummy, tensor_lists, ComputeScaleInvE8M0Functor(),
                        stream);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_compute_scale
}  // namespace transformer_engine

void nvte_multi_tensor_compute_scale_and_scale_inv_cuda(int chunk_size, NVTETensor noop_flag,
                                                        NVTETensor **tensor_lists,
                                                        const size_t num_tensor_lists,
                                                        const size_t num_tensors_per_list,
                                                        float max_fp8, int force_pow_2_scales,
                                                        float epsilon, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_compute_scale_and_scale_inv_cuda);
  using namespace transformer_engine;

  multi_tensor_compute_scale::multi_tensor_compute_scale_and_scale_inv_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), max_fp8,
      force_pow_2_scales, epsilon, stream);
}

void nvte_multi_tensor_compute_scale_inv_e8m0_cuda(int chunk_size, NVTETensor **tensor_lists,
                                                   const size_t num_tensor_lists,
                                                   const size_t num_tensors_per_list,
                                                   cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_compute_scale_inv_e8m0_cuda);
  using namespace transformer_engine;

  multi_tensor_compute_scale::multi_tensor_compute_scale_inv_e8m0_cuda(
      chunk_size, convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      stream);
}
