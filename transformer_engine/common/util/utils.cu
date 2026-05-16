/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/utils.h>

#include <algorithm>
#include <cstring>

#include <cuda_runtime.h>

#include "../common.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace load_value_on_device {
namespace {

union Payload {
  static constexpr size_t max_bytes = 2048;
  static constexpr size_t vector_size = 4;
  uint8_t bytes[max_bytes];
  uint32_t vectors[max_bytes / vector_size];
};

constexpr size_t block_size = 512;
constexpr size_t num_blocks = DIVUP(Payload::max_bytes / Payload::vector_size, block_size);

__global__ void __launch_bounds__(block_size) kernel(Payload payload, size_t num_bytes, void *out) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (Payload::vector_size * (tid + 1) <= num_bytes) {
    reinterpret_cast<uint32_t *>(out)[tid] = payload.vectors[tid];
  } else {
    for (int64_t i = Payload::vector_size * tid; i < num_bytes; ++i) {
      static_cast<uint8_t *>(out)[i] = payload.bytes[i];
    }
  }
}

}  // namespace
}  // namespace load_value_on_device
}  // namespace transformer_engine

void nvte_load_value_on_device(const void *host_ptr, void *device_ptr, size_t num_bytes,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_load_value_on_device);
  using namespace transformer_engine::load_value_on_device;

  // Nothing to be done if size is zero
  if (num_bytes == 0) {
    return;
  }

  // Check pointers
  NVTE_CHECK(host_ptr != nullptr, "Attempting to read ", num_bytes, " bytes from a null pointer.");
  NVTE_CHECK(device_ptr != nullptr, "Attempting to write ", num_bytes, " bytes into a null pointer.");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(device_ptr) % Payload::vector_size == 0,
             "Device pointer is not aligned to ", Payload::vector_size, " bytes.");

  // Chunk data to fit in kernel arguments and launch kernels
  const uint8_t *src = static_cast<const uint8_t *>(host_ptr);
  uint8_t *dst = static_cast<uint8_t *>(device_ptr);
  for (size_t offset = 0; offset < num_bytes; offset += Payload::max_bytes) {
    const size_t chunk_size = std::min(num_bytes - offset, Payload::max_bytes);
    Payload payload;
    std::memcpy(payload.bytes, src + offset, chunk_size);
    kernel<<<num_blocks, block_size, 0, stream>>>(payload, chunk_size, dst + offset);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void nvte_convert_pointers_to_tensor(const uint64_t *host_ptrs, NVTETensor output, int64_t count,
                                     cudaStream_t stream) {
  using namespace transformer_engine;
  Tensor *out_tensor = convertNVTETensorCheck(output);
  nvte_load_value_on_device(host_ptrs, out_tensor->data.dptr,
                            static_cast<size_t>(count) * sizeof(uint64_t), stream);
}
