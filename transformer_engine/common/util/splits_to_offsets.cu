/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine::splits_to_offsets {

namespace {

constexpr size_t kThreadsPerBlock = 256;

struct KernelArgs {
  static constexpr size_t kMaxNumOutputs = 8;
  const void *split_sizes = nullptr;
  DType split_sizes_dtype = DType::kNumTypes;
  size_t num_splits = 0;
  void *outputs[kMaxNumOutputs] = {};
  DType outputs_dtype[kMaxNumOutputs] = {};
  int64_t strides[kMaxNumOutputs] = {};
  bool include_leading_zero[kMaxNumOutputs] = {};
  size_t num_outputs = 0;
};

__device__ __forceinline__ int64_t load_split_size(const void *__restrict__ split_sizes,
                                                   DType dtype, size_t idx) {
  switch (dtype) {
    case DType::kInt32:
      return static_cast<int64_t>(static_cast<const int32_t *>(split_sizes)[idx]);
    case DType::kInt64:
      return static_cast<const int64_t *>(split_sizes)[idx];
    default:
      NVTE_DEVICE_ERROR("Unsupported dtype for split_sizes (expected int32 or int64).");
      return 0;
  }
}

__device__ __forceinline__ void store_output(void *__restrict__ output, DType dtype, size_t idx,
                                             int64_t value) {
  switch (dtype) {
    case DType::kInt32:
      static_cast<int32_t *>(output)[idx] = static_cast<int32_t>(value);
      return;
    case DType::kInt64:
      static_cast<int64_t *>(output)[idx] = value;
      return;
    default:
      NVTE_DEVICE_ERROR("Unsupported dtype for output (expected int32 or int64).");
  }
}

__global__ void __launch_bounds__(kThreadsPerBlock) kernel(KernelArgs args) {
  const size_t tid = threadIdx.x;

  // Fill leading zeros if needed
  if (tid == 0) {
    for (size_t out_idx = 0; out_idx < args.num_outputs; ++out_idx) {
      if (args.include_leading_zero[out_idx]) {
        store_output(args.outputs[out_idx], args.outputs_dtype[out_idx], 0, 0);
      }
    }
  }

  // Workspace for prefix sum chunk
  __shared__ int64_t block_scan[kThreadsPerBlock];

  // Sum from previous chunks
  __shared__ int64_t chunk_prefix;
  if (tid == 0) {
    chunk_prefix = 0;
  }
  __syncthreads();

  // Perform prefix sum in chunks
  for (size_t chunk_start = 0; chunk_start < args.num_splits; chunk_start += kThreadsPerBlock) {
    const size_t idx = chunk_start + tid;

    // Load input from global memory into shared memory
    if (idx < args.num_splits) {
      block_scan[tid] = load_split_size(args.split_sizes, args.split_sizes_dtype, idx);
    } else {
      block_scan[tid] = 0;
    }
    __syncthreads();

    // Prefix sum in shared memory
    for (size_t offset = 1; offset < kThreadsPerBlock; offset <<= 1) {
      const int64_t addend = (tid >= offset) ? block_scan[tid - offset] : 0;
      __syncthreads();
      block_scan[tid] += addend;
      __syncthreads();
    }

    // Compute global prefix sum, apply strides, and store to output
    if (idx < args.num_splits) {
      const int64_t prefix = chunk_prefix + block_scan[tid];
      for (size_t out_idx = 0; out_idx < args.num_outputs; ++out_idx) {
        const size_t write_idx = idx + (args.include_leading_zero[out_idx] ? 1 : 0);
        store_output(args.outputs[out_idx], args.outputs_dtype[out_idx], write_idx,
                     prefix * args.strides[out_idx]);
      }
    }

    // Update sum for later chunks
    __syncthreads();
    if (tid == kThreadsPerBlock - 1) {
      chunk_prefix += block_scan[tid];
    }
    __syncthreads();
  }
}

}  // namespace

}  // namespace transformer_engine::splits_to_offsets

void nvte_splits_to_offsets(const int64_t *split_sizes, int64_t *output, size_t num_splits,
                            int64_t stride, cudaStream_t stream) {
  NVTE_API_CALL(nvte_splits_to_offsets);
  NVTE_CHECK(output != nullptr, "Output pointer is NULL.");
  NVTE_CHECK(num_splits > 0, "num_splits must be greater than 0.");
  NVTE_CHECK(split_sizes != nullptr, "split_sizes pointer is NULL.");
  NVTE_CHECK(stride > 0, "stride must be greater than 0.");

  using namespace transformer_engine;
  namespace s2o = transformer_engine::splits_to_offsets;

  s2o::KernelArgs args = {};
  args.split_sizes = split_sizes;
  args.split_sizes_dtype = DType::kInt64;
  args.num_splits = num_splits;
  args.outputs[0] = output;
  args.outputs_dtype[0] = DType::kInt64;
  args.strides[0] = stride;
  args.include_leading_zero[0] = true;
  args.num_outputs = 1;
  s2o::kernel<<<1, s2o::kThreadsPerBlock, 0, stream>>>(args);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvte_splits_to_offsets_multi(NVTETensor split_sizes, NVTETensor *outputs,
                                  const int64_t *strides, const int *include_leading_zero,
                                  size_t num_outputs, cudaStream_t stream) {
  NVTE_API_CALL(nvte_splits_to_offsets_multi);
  using namespace transformer_engine;
  namespace s2o = transformer_engine::splits_to_offsets;

  if (num_outputs == 0) {
    return;
  }
  NVTE_CHECK(outputs != nullptr, "outputs is NULL.");
  NVTE_CHECK(strides != nullptr, "strides is NULL.");
  NVTE_CHECK(include_leading_zero != nullptr, "include_leading_zero is NULL.");

  // Check if dtype is supported
  const auto is_integer_dtype = [](DType dtype) {
    return dtype == DType::kInt32 || dtype == DType::kInt64;
  };

  // Check input tensor
  const auto *split_sizes_tensor = convertNVTETensorCheck(split_sizes);
  const auto split_sizes_dtype = split_sizes_tensor->dtype();
  const auto num_splits = split_sizes_tensor->numel();
  NVTE_CHECK(num_splits > 0 && split_sizes_tensor->dim() == 1,
             "split_sizes must be a non-empty 1D tensor, but got shape=",
             split_sizes_tensor->shape(), ".");
  NVTE_CHECK(is_integer_dtype(split_sizes_dtype),
             "split_sizes must be an int32/int64 tensor, but got dtype=",
             split_sizes_dtype, ".");

  // Check output tensors
  std::vector<const Tensor *> output_tensors(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto *out_tensor = convertNVTETensorCheck(outputs[i]);
    const auto out_dtype = out_tensor->dtype();
    const bool has_leading_zero = include_leading_zero[i] != 0;
    const Shape expected_shape = {num_splits + (has_leading_zero ? 1 : 0)};
    NVTE_CHECK(out_tensor->shape() == expected_shape,
               "Expected outputs[", i, "] to have shape=", expected_shape,
               ", but got shape=", out_tensor->shape(), ".");
    NVTE_CHECK(is_integer_dtype(out_dtype),
               "Expected outputs[", i, "] to be an int32/int64 tensor, but got dtype=",
               out_dtype, ".");
    output_tensors[i] = out_tensor;
  }

  // Chunk outputs to fit in kernel arguments and launch kernels
  for (size_t chunk_start = 0; chunk_start < num_outputs;
       chunk_start += s2o::KernelArgs::kMaxNumOutputs) {
    const size_t chunk_size = std::min(s2o::KernelArgs::kMaxNumOutputs, num_outputs - chunk_start);
    s2o::KernelArgs args = {};
    args.split_sizes = split_sizes_tensor->data.dptr;
    args.split_sizes_dtype = split_sizes_dtype;
    args.num_splits = num_splits;
    args.num_outputs = chunk_size;
    for (size_t i = 0; i < chunk_size; ++i) {
      const size_t out_idx = chunk_start + i;
      args.outputs[i] = output_tensors[out_idx]->data.dptr;
      args.outputs_dtype[i] = output_tensors[out_idx]->dtype();
      args.strides[i] = strides[out_idx];
      args.include_leading_zero[i] = include_leading_zero[out_idx] != 0;
    }
    s2o::kernel<<<1, s2o::kThreadsPerBlock, 0, stream>>>(args);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}
