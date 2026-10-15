/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/resource_handle.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "resource_handle.h"

namespace {

struct GroupedTmaResource {
  size_t capacity = 0;
  transformer_engine::GroupedTmaStorageView storage;
  std::vector<void *> allocations;

  // The owning host thread alone resizes this stream-local resource.
  void reserve(size_t num_tensors, cudaStream_t stream) {
    if (num_tensors <= capacity) {
      return;
    }

    cudaStreamCaptureStatus capture_status;
    NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture_status));
    NVTE_CHECK(capture_status == cudaStreamCaptureStatusNone,
               "Grouped TMA storage cannot grow during CUDA graph capture. "
               "Reserve enough descriptors on this host thread and stream before capture.");

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
    // Older allocations may still be referenced by queued kernels or captured graphs.
    allocations.push_back(allocation);
    auto *base = static_cast<char *>(allocation);
    storage = transformer_engine::GroupedTmaStorageView(base, map_bytes, metadata_bytes);
    capacity = new_capacity;
  }

  void release() {
    while (!allocations.empty()) {
      NVTE_CHECK_CUDA(cudaFree(allocations.back()));
      allocations.pop_back();
    }
  }
};

// Mutable resources used by one host thread on one CUDA stream belong here.
// Resource types with another lifetime or sharing policy can be owned by the
// handle itself rather than placed in this stream-local container.
struct ThreadStreamResources {
  const CUcontext context;
  const int device;
  GroupedTmaResource grouped_tma;

  ThreadStreamResources(CUcontext context, int device) : context(context), device(device) {}

  void release() { grouped_tma.release(); }
};

struct SlotKey {
  std::uint64_t owner_id;
  cudaStream_t stream;

  bool operator==(const SlotKey &other) const {
    return owner_id == other.owner_id && stream == other.stream;
  }
};

struct SlotKeyHash {
  size_t operator()(const SlotKey &key) const {
    const size_t a = std::hash<std::uint64_t>{}(key.owner_id);
    const size_t b = std::hash<cudaStream_t>{}(key.stream);
    return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
  }
};

std::atomic<std::uint64_t> next_handle_id{1};
std::atomic<std::uint64_t> next_thread_id{1};

std::uint64_t current_thread_id() {
  static thread_local const std::uint64_t id =
      next_thread_id.fetch_add(1, std::memory_order_relaxed);
  return id;
}

// The CUDA driver wrapper's generic call path locks a global symbol cache.
// Resolve this fixed symbol once so steady-state slot lookup has no TE mutex.
CUcontext current_context() {
  transformer_engine::cuda_driver::ensure_context_exists();
  using GetCurrent = CUresult (*)(CUcontext *);
  static auto *get_current = reinterpret_cast<GetCurrent>(
      transformer_engine::cuda_driver::get_symbol("cuCtxGetCurrent"));
  CUcontext context = nullptr;
  NVTE_CHECK_CUDA_DRIVER(get_current(&context));
  return context;
}

}  // namespace

struct NVTEResourceHandleOpaque {
  const std::uint64_t id = next_handle_id.fetch_add(1, std::memory_order_relaxed);
  std::mutex registry_mutex;
  CUcontext context = nullptr;
  int device = -1;
  // Per-thread/stream mutable resources. Shared device resources can live alongside them.
  std::unordered_map<SlotKey, std::unique_ptr<ThreadStreamResources>, SlotKeyHash> slots;

  ThreadStreamResources *get_slot(cudaStream_t stream) {
    const CUcontext active_context = current_context();
    int active_device = -1;
    NVTE_CHECK_CUDA(cudaGetDevice(&active_device));

    // The cache itself is thread-local. A slot never changes owners, so hits
    // require no lock even if other threads register new slots concurrently.
    static thread_local std::unordered_map<SlotKey, ThreadStreamResources *, SlotKeyHash> cache;
    const SlotKey key{id, stream};
    auto found = cache.find(key);
    if (found != cache.end()) {
      auto *slot = found->second;
      NVTE_CHECK(slot->context == active_context,
                 "Resource handle is bound to a different CUDA context.");
      NVTE_CHECK(slot->device == active_device,
                 "Resource handle is bound to CUDA device ", slot->device,
                 ", but the current device is ", active_device, ".");
      return slot;
    }

    ThreadStreamResources *slot = nullptr;
    {
      std::lock_guard<std::mutex> lock(registry_mutex);
      NVTE_CHECK(context == nullptr || context == active_context,
                 "Resource handle is bound to a different CUDA context.");
      NVTE_CHECK(device < 0 || device == active_device,
                 "Resource handle is bound to CUDA device ", device,
                 ", but the current device is ", active_device, ".");
      const SlotKey registry_key{current_thread_id(), stream};
      auto existing = slots.find(registry_key);
      if (existing != slots.end()) {
        slot = existing->second.get();
      } else {
        auto new_slot = std::make_unique<ThreadStreamResources>(active_context, active_device);
        slot = new_slot.get();
        slots.emplace(registry_key, std::move(new_slot));
        context = active_context;
        device = active_device;
      }
    }
    // Bound the per-thread metadata when applications create many short-lived
    // handles. Handle IDs prevent entries from aliasing a later handle.
    if (cache.size() >= 128) {
      cache.clear();
    }
    cache.emplace(key, slot);
    return slot;
  }

  void release() {
    if (context == nullptr) {
      return;
    }
    NVTE_CHECK(current_context() == context,
               "Destroy the resource handle in its bound CUDA context.");
    int active_device = -1;
    NVTE_CHECK_CUDA(cudaGetDevice(&active_device));
    NVTE_CHECK(active_device == device,
               "Destroy the resource handle on its bound CUDA device ", device, ".");
    for (auto &entry : slots) {
      entry.second->release();
    }
  }
};

NVTEResourceHandle nvte_create_resource_handle(void) {
  NVTE_API_CALL(nvte_create_resource_handle);
  return new NVTEResourceHandleOpaque;
}

void nvte_resource_handle_reserve(NVTEResourceHandle handle, NVTEResourceType resource_type,
                                  size_t capacity, cudaStream_t stream) {
  NVTE_API_CALL(nvte_resource_handle_reserve);
  NVTE_CHECK(handle != nullptr, "Resource handle must not be NULL.");
  switch (resource_type) {
    case NVTE_RESOURCE_GROUPED_TMA_DESCRIPTORS:
      if (capacity != 0) {
        handle->get_slot(stream)->grouped_tma.reserve(capacity, stream);
      }
      return;
    default:
      NVTE_ERROR("Unknown resource type: ", static_cast<int>(resource_type), ".");
  }
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
  auto &grouped_tma = handle->get_slot(stream)->grouped_tma;
  grouped_tma.reserve(num_tensors, stream);
  return grouped_tma.storage;
}

}  // namespace transformer_engine
