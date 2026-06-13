/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include "../utils.cuh"
#include "multi_tensor_apply.cuh"

namespace transformer_engine {
namespace multi_tensor_raw_moments {

#define BLOCK_SIZE 512
#define ILP 4
#define RAW_MOMENT_FIELDS 5

template <typename T>
__device__ __forceinline__ bool is_aligned(T *p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T *dst, T *src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT *)dst)[dst_offset] = ((LT *)src)[src_offset];  // NOLINT(*)
}

__device__ __forceinline__ float reduce_block_sum(float *x, float val) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  float final = 0.f;
  if (tid < 32) {
    if (blockSize >= 64) {
      final = x[tid] + x[tid + 32];
    } else {
      final = val;
    }

#pragma unroll
    for (int i = 16; i >= 1; i >>= 1) final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  __syncthreads();
  return final;
}

template <typename x_t>
struct RawMomentsFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int *noop_gmem,
                                             TensorListMetadata<1> &tl,  // NOLINT(*)
                                             float *output_per_tensor, int max_chunks_per_tensor) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_idx = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t *x = reinterpret_cast<x_t *>(tl.addresses[0][tensor_loc]);
    x += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;
    int elements_this_chunk = n < chunk_size ? n : chunk_size;

    __shared__ float s_vals[RAW_MOMENT_FIELDS - 1][BLOCK_SIZE];

    float sum_1 = 0.f;
    float sum_2 = 0.f;
    float sum_3 = 0.f;
    float sum_4 = 0.f;

    x_t r_x[ILP];
    for (int i = 0; i < ILP; i++) r_x[i] = 0;

    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x)) {
      for (int i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size;
           i_start += blockDim.x) {
        load_store(r_x, x, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          float val = static_cast<float>(r_x[ii]);
          float val_2 = val * val;
          sum_1 += val;
          sum_2 += val_2;
          sum_3 += val_2 * val;
          sum_4 += val_2 * val_2;
        }
      }
    } else {
      for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          int i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) {
            float val = static_cast<float>(x[i]);
            float val_2 = val * val;
            sum_1 += val;
            sum_2 += val_2;
            sum_3 += val_2 * val;
            sum_4 += val_2 * val_2;
          }
        }
      }
    }

    float final_sum_1 = reduce_block_sum(s_vals[0], sum_1);
    float final_sum_2 = reduce_block_sum(s_vals[1], sum_2);
    float final_sum_3 = reduce_block_sum(s_vals[2], sum_3);
    float final_sum_4 = reduce_block_sum(s_vals[3], sum_4);

    if (threadIdx.x == 0) {
      if (!isfinite(final_sum_1) || !isfinite(final_sum_2) || !isfinite(final_sum_3) ||
          !isfinite(final_sum_4)) {
        *noop_gmem = 1;
      }
      float *row = output_per_tensor +
                   (tensor_idx * max_chunks_per_tensor + chunk_idx) * RAW_MOMENT_FIELDS;
      row[0] = static_cast<float>(elements_this_chunk);
      row[1] = final_sum_1;
      row[2] = final_sum_2;
      row[3] = final_sum_3;
      row[4] = final_sum_4;
    }
  }
};

__global__ void cleanup(float *output_per_tensor, float *ret, int max_chunks_per_tensor) {
  int tensor_idx = blockIdx.x;
  int field_idx = blockIdx.y;
  __shared__ float vals[BLOCK_SIZE];

  float *chunks =
      output_per_tensor + tensor_idx * max_chunks_per_tensor * RAW_MOMENT_FIELDS + field_idx;

  float val = 0.f;
  for (int i = threadIdx.x; i < max_chunks_per_tensor; i += blockDim.x) {
    val += chunks[i * RAW_MOMENT_FIELDS];
  }

  float final = reduce_block_sum(vals, val);
  if (threadIdx.x == 0) ret[tensor_idx * RAW_MOMENT_FIELDS + field_idx] = final;
}

void multi_tensor_raw_moments_cuda(int chunk_size, Tensor noop_flag,
                                   std::vector<std::vector<Tensor *>> tensor_lists,
                                   Tensor output_per_tensor, Tensor ret,
                                   int max_chunks_per_tensor, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      tensor_lists[0][0]->dtype(), dtype,
      multi_tensor_apply<1>(
          BLOCK_SIZE, chunk_size, noop_flag, tensor_lists, RawMomentsFunctor<dtype>(), stream,
          reinterpret_cast<float *>(output_per_tensor.data.dptr), max_chunks_per_tensor);)

  NVTE_CHECK_CUDA(cudaGetLastError());

  dim3 grid(tensor_lists[0].size(), RAW_MOMENT_FIELDS);
  cleanup<<<grid, BLOCK_SIZE, 0, stream>>>(
      reinterpret_cast<float *>(output_per_tensor.data.dptr),
      reinterpret_cast<float *>(ret.data.dptr), max_chunks_per_tensor);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_raw_moments
}  // namespace transformer_engine

void nvte_multi_tensor_raw_moments_cuda(int chunk_size, NVTETensor noop_flag,
                                        NVTETensor **tensor_lists,
                                        const size_t num_tensor_lists,
                                        const size_t num_tensors_per_list,
                                        NVTETensor output_per_tensor, NVTETensor ret,
                                        int max_chunks_per_tensor, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_raw_moments_cuda);
  using namespace transformer_engine;

  multi_tensor_raw_moments::multi_tensor_raw_moments_cuda(
      chunk_size, *convertNVTETensorCheck(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list),
      *convertNVTETensorCheck(output_per_tensor), *convertNVTETensorCheck(ret),
      max_chunks_per_tensor, stream);
}
