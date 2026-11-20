/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/utils.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "../common.h"
#include "../util/cuda_runtime.h"
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

namespace transformer_engine {

namespace {

// Occupancy probe for cuda::max_active_clusters: empty on purpose, only its launch bounds and
// cluster dim matter.
__global__ void cluster_probe_kernel() {}

}  // namespace

namespace cuda {

int max_active_clusters(int cluster_size) {
  static std::mutex mutex;
  // Occupancy depends on the device and the cluster size, so the cache keys on both.
  static std::unordered_map<int64_t, int> clusters_by_device_and_size;
  const int device = current_device();
  const int64_t key = (static_cast<int64_t>(device) << 32) | static_cast<uint32_t>(cluster_size);
  std::lock_guard<std::mutex> lock(mutex);
  auto it = clusters_by_device_and_size.find(key);
  if (it == clusters_by_device_and_size.end()) {
    // All the shared memory a CTA may take: one CTA per SM.
    int smem = 0;
    NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(cluster_probe_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = cluster_size;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(cluster_size, 1, 1);
    config.blockDim = dim3(128, 1, 1);
    config.dynamicSmemBytes = smem;
    config.attrs = &attr;
    config.numAttrs = 1;
    int clusters = 0;
    NVTE_CHECK_CUDA(cudaOccupancyMaxActiveClusters(&clusters, cluster_probe_kernel, &config));
    it = clusters_by_device_and_size.emplace(key, clusters).first;
  }
  return it->second;
}

}  // namespace cuda

}  // namespace transformer_engine
