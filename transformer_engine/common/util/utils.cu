/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/utils.h>

#include <algorithm>
#include <cstring>

#include "../common.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace copy_host_to_device_via_kernel {
namespace {

union Payload {
  static constexpr size_t kMaxBytes = 2048;
  static constexpr size_t kVectorSize = 4;
  static constexpr size_t kMaxVectors = kMaxBytes / kVectorSize;
  uint8_t bytes[kMaxBytes];
  uint32_t vectors[kMaxVectors];
};

constexpr size_t block_size = 512;
constexpr size_t num_blocks = DIVUP(Payload::kMaxVectors, block_size);

__global__ void __launch_bounds__(block_size) kernel(Payload payload, size_t num_bytes, void *out) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (Payload::kVectorSize * (tid + 1) <= num_bytes) {
    reinterpret_cast<uint32_t *>(out)[tid] = payload.vectors[tid];
  } else {
    for (size_t i = Payload::kVectorSize * tid; i < num_bytes; ++i) {
      reinterpret_cast<uint8_t *>(out)[i] = payload.bytes[i];
    }
  }
}

}  // namespace
}  // namespace copy_host_to_device_via_kernel
}  // namespace transformer_engine

void nvte_copy_host_to_device_via_kernel(const void *host_ptr, void *device_ptr, size_t num_bytes,
                                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_copy_host_to_device_via_kernel);
  using namespace transformer_engine::copy_host_to_device_via_kernel;

  // Nothing to be done if size is zero
  if (num_bytes == 0) {
    return;
  }

  // Check pointers
  NVTE_CHECK(host_ptr != nullptr, "Attempting to read ", num_bytes, " bytes from a null pointer.");
  NVTE_CHECK(device_ptr != nullptr, "Attempting to write ", num_bytes,
             " bytes into a null pointer.");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(device_ptr) % Payload::kVectorSize == 0,
             "Device pointer is not aligned to ", Payload::kVectorSize, " bytes.");

  // Chunk data to fit in kernel arguments and launch kernels
  const uint8_t *src = static_cast<const uint8_t *>(host_ptr);
  uint8_t *dst = static_cast<uint8_t *>(device_ptr);
  for (size_t offset = 0; offset < num_bytes; offset += Payload::kMaxBytes) {
    const size_t chunk_size = std::min(num_bytes - offset, Payload::kMaxBytes);
    Payload payload{};
    std::memcpy(payload.bytes, src + offset, chunk_size);
    kernel<<<num_blocks, block_size, 0, stream>>>(payload, chunk_size, dst + offset);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void nvte_convert_pointers_to_tensor(const uint64_t *host_ptrs, NVTETensor output, int64_t count,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_convert_pointers_to_tensor);
  using namespace transformer_engine;
  Tensor *out_tensor = convertNVTETensorCheck(output);
  nvte_copy_host_to_device_via_kernel(host_ptrs, out_tensor->data.dptr,
                                      static_cast<size_t>(count) * sizeof(uint64_t), stream);
}
