/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/resource_handle.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "common.h"
#include "resource_handle.h"

struct NVTEResourceHandleOpaque {
  int device = -1;
  CUcontext context = nullptr;
  size_t capacity = 0;
  transformer_engine::GroupedTmaStorageView grouped_tma;
  std::vector<void *> allocations;

  void reserve_grouped_tma(size_t num_tensors, cudaStream_t stream) {
    if (num_tensors == 0) {
      return;
    }

    transformer_engine::cuda_driver::ensure_context_exists();
    CUcontext current_context = nullptr;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxGetCurrent, &current_context);
    int current_device = -1;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
    NVTE_CHECK(context == nullptr || context == current_context,
               "Resource handle is bound to a different CUDA context.");
    NVTE_CHECK(device < 0 || device == current_device,
               "Resource handle is bound to CUDA device ", device,
               ", but the current device is ", current_device, ".");
    if (num_tensors <= capacity) {
      return;
    }

    cudaStreamCaptureStatus capture_status;
    NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture_status));
    NVTE_CHECK(capture_status == cudaStreamCaptureStatusNone,
               "Grouped TMA storage cannot grow during CUDA graph capture. "
               "Reserve enough descriptors or run the operation before capture.");

    constexpr size_t alignment = 128;
    constexpr size_t bytes_per_group = 4 * sizeof(CUtensorMap) + 3 * sizeof(size_t);
    const size_t max_count =
        (std::numeric_limits<size_t>::max() - 4 * (alignment - 1)) / bytes_per_group;
    NVTE_CHECK(num_tensors <= max_count, "Grouped TMA descriptor count is too large.");
    const size_t new_capacity =
        capacity <= max_count / 2 ? std::max(num_tensors, capacity * 2) : num_tensors;
    const size_t map_bytes =
        ((new_capacity * sizeof(CUtensorMap) + alignment - 1) / alignment) * alignment;
    const size_t metadata_bytes = new_capacity * sizeof(size_t);
    const size_t total_bytes = 4 * map_bytes + 3 * metadata_bytes;

    allocations.reserve(allocations.size() + 1);
    void *allocation = nullptr;
    NVTE_CHECK_CUDA(cudaMalloc(&allocation, total_bytes));
    // Retain previous allocations until destruction: previously enqueued kernels and
    // captured graphs may still refer to them after this handle grows.
    allocations.push_back(allocation);
    device = current_device;
    context = current_context;
    auto *base = static_cast<char *>(allocation);
    transformer_engine::GroupedTmaStorageView storage;
    storage.input = reinterpret_cast<CUtensorMap *>(base);
    storage.act_input = reinterpret_cast<CUtensorMap *>(base + map_bytes);
    storage.output_rowwise = reinterpret_cast<CUtensorMap *>(base + 2 * map_bytes);
    storage.output_colwise = reinterpret_cast<CUtensorMap *>(base + 3 * map_bytes);
    storage.rows = reinterpret_cast<size_t *>(base + 4 * map_bytes);
    storage.cols = reinterpret_cast<size_t *>(base + 4 * map_bytes + metadata_bytes);
    storage.offsets = reinterpret_cast<size_t *>(base + 4 * map_bytes + 2 * metadata_bytes);
    grouped_tma = storage;
    capacity = new_capacity;
  }

  void release() {
    if (allocations.empty()) {
      return;
    }
    transformer_engine::cuda_driver::ensure_context_exists();
    CUcontext current_context = nullptr;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxGetCurrent, &current_context);
    NVTE_CHECK(current_context == context,
               "Destroy the resource handle in its bound CUDA context.");
    int current_device = -1;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
    NVTE_CHECK(current_device == device,
               "Destroy the resource handle on its bound CUDA device ", device, ".");
    while (!allocations.empty()) {
      NVTE_CHECK_CUDA(cudaFree(allocations.back()));
      allocations.pop_back();
    }
  }
};

NVTEResourceHandle nvte_create_resource_handle(void) {
  NVTE_API_CALL(nvte_create_resource_handle);
  return new NVTEResourceHandleOpaque;
}

void nvte_resource_handle_reserve_grouped_tma(NVTEResourceHandle handle, size_t num_tensors,
                                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_resource_handle_reserve_grouped_tma);
  NVTE_CHECK(handle != nullptr, "Resource handle must not be NULL.");
  handle->reserve_grouped_tma(num_tensors, stream);
}

void nvte_destroy_resource_handle(NVTEResourceHandle handle) {
  NVTE_API_CALL(nvte_destroy_resource_handle);
  if (handle != nullptr) {
    handle->release();
    delete handle;
  }
}

namespace transformer_engine {

GroupedTmaStorageView get_grouped_tma_storage(NVTEResourceHandle handle, size_t num_tensors,
                                               cudaStream_t stream) {
  if (handle == nullptr) {
    return {};
  }
  handle->reserve_grouped_tma(num_tensors, stream);
  return handle->grouped_tma;
}

}  // namespace transformer_engine
