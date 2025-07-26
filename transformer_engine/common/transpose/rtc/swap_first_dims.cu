/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Parameters
using Type = __TYPE__;
constexpr size_t block_size = __BLOCK_SIZE__;

}  // namespace

__global__ void __launch_bounds__(block_size)
    swap_first_dims_kernel(const Type* __restrict__ const input,
                           Type* __restrict__ const output,
                           const size_t dim0,
                           const size_t dim1,
                           const size_t dim2) {
  const size_t gid = threadIdx.x + blockIdx.x * block_size;
#if __SINGLE_LOAD_STORE__
  const size_t idx = gid;
#else
  const size_t nthreads = gridDim.x * block_size;
  for (size_t idx = gid; idx < dim0 * dim1 * dim2; idx += nthreads)
#endif // __SINGLE_LOAD_STORE__
  {
    const size_t idx2 = idx % dim2;
    const size_t idx1 = (idx / dim2) % dim1;
    const size_t idx0 = (idx / dim2) / dim1;
    const size_t in_offset = idx0 * dim1 * dim2 + idx1 * dim2 + idx2;
    const size_t out_offset = idx1 * dim0 * dim2 + idx0 * dim2 + idx2;
    output[out_offset] = input[in_offset];
  }
}
