/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transpose.h>

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

#include "../common.h"
#include "../util/logging.h"

namespace transformer_engine {

namespace {

// Hard-coded kernel parameters
constexpr size_t block_size = 256;

template <typename Type>
__global__ void __launch_bounds__(block_size)
    swap_first_dims_untuned_kernel(const Type * __restrict__ input,
                                   Type * __restrict__ output,
                                   const size_t dim0,
                                   const size_t dim1,
                                   const size_t dim2) {
  const size_t gid = threadIdx.x + blockIdx.x * block_size;
  const size_t nthreads = gridDim.x * block_size;
  for (size_t idx = gid; idx < dim0 * dim1 * dim2; idx += nthreads) {
    const size_t idx2 = idx % dim2;
    const size_t idx1 = (idx / dim2) % dim1;
    const size_t idx0 = (idx / dim2) / dim1;
    const size_t in_offset = idx0 * dim1 * dim2 + idx1 * dim2 + idx2;
    const size_t out_offset = idx1 * dim0 * dim2 + idx0 * dim2 + idx2;
    output[out_offset] = input[in_offset];
  }
}

}  // namespace

void swap_first_dims(const Tensor &input, Tensor &output, cudaStream_t stream) {

  // Check tensors
  NVTE_CHECK(input.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be simple tensor, but scaling mode is ",
             to_string(input.scaling_mode), ".");
  NVTE_CHECK(output.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Output tensor must be simple tensor, but scaling mode is ",
             to_string(output.scaling_mode), ".");
  NVTE_CHECK(input.dtype() == output.dtype(),
             "Input tensor (dtype=", to_string(input.dtype()), ") and output tensor (dtype=",
             to_string(output.dtype()), ") do not match.");
  NVTE_CHECK(input.data.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(output.data.dptr != nullptr, "Output is not allocated.");

  // Check tensor dimensions
  const auto input_shape = input.shape();
  const auto output_shape = output.shape();
  NVTE_CHECK(input_shape.size() >= 2, "Invalid input tensor dimensions (shape=", input_shape, ").");
  NVTE_CHECK(output_shape.size() == input_shape.size(),
             "Input tensor (shape=", input_shape, ") and output tensor (shape=", output_shape,
             ") do not match.");
  NVTE_CHECK(input_shape[0] == output_shape[1],
             "Input tensor (shape=", input_shape, ") and output tensor (shape=", output_shape,
             ") do not match.");
  NVTE_CHECK(input_shape[1] == output_shape[0],
             "Input tensor (shape=", input_shape, ") and output tensor (shape=", output_shape,
             ") do not match.");
  for (size_t i = 2; i < input_shape.size(); ++i) {
    NVTE_CHECK(input_shape[i] == output_shape[i],
               "Input tensor (shape=", input_shape, ") and output tensor (shape=", output_shape,
               ") do not match.");
  }

  // Reinterpret tensors as 3D tensors of bytes
  const size_t dim0 = input_shape[0];
  const size_t dim1 = input_shape[1];
  size_t dim2 = 1;
  for (size_t i = 2; i < input_shape.size(); ++i) {
    dim2 *= input_shape[i];
  }
  dim2 = get_buffer_size_bytes(dim2, input.dtype());

  // Launch kernel
  /// TODO Vectorization
  /// TODO NVRTC
  size_t num_blocks = DIVUP(dim0 * dim1 * dim2, block_size);
  num_blocks = std::min(num_blocks, static_cast<size_t>(2147483647ull));
  swap_first_dims_untuned_kernel
    <<<num_blocks, block_size, 0, stream>>>(static_cast<const uint8_t*>(input.data.dptr),
                                            static_cast<uint8_t*>(output.data.dptr),
                                            dim0, dim1, dim2);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace transformer_engine

void nvte_swap_first_dims(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swap_first_dims);
  using namespace transformer_engine;
  swap_first_dims(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output), stream);
}
