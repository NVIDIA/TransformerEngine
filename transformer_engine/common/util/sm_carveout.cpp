/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/sm_carveout.h"

#include <cuda.h>

#include <list>
#include <mutex>
#include <unordered_map>

#include "../util/cuda_driver.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace cuda {

namespace {

// A hardware SM partition, cached per (device, margin) for the process lifetime. `nullptr`
// means no carveout (platform/driver couldn't create one).
CUgreenCtx get_green_context(int device_id, int sm_margin) {
  const int device_sm_count = sm_count(device_id);
  NVTE_CHECK(sm_margin >= 0 && sm_margin < device_sm_count, "SM margin (", sm_margin,
             ") must be between 0 (inclusive) and the device SM count (", device_sm_count,
             ") (exclusive).");

#if CUDART_VERSION >= 12040
  static std::unordered_map<int, std::unordered_map<int, CUgreenCtx>> cache;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  auto &per_device = cache[device_id];
  auto it = per_device.find(sm_margin);
  if (it != per_device.end()) return it->second;

  CUgreenCtx green_ctx = nullptr;
  try {
    CUdevice cu_device;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &cu_device, device_id);

    // These symbols postdate call()'s default resolution version (12.1) and their ABI has
    // changed since, so resolve against the exact CUDA version this library was built with.
    CUdevResource sm_resource;
    NVTE_CALL_CHECK_CUDA_DRIVER_VERSIONED(cuDeviceGetDevResource, CUDART_VERSION, cu_device,
                                          &sm_resource, CU_DEV_RESOURCE_TYPE_SM);

    CUdevResource carved_resource;
    unsigned int num_groups = 1;
    NVTE_CALL_CHECK_CUDA_DRIVER_VERSIONED(cuDevSmResourceSplitByCount, CUDART_VERSION,
                                          &carved_resource, &num_groups, &sm_resource, nullptr, 0u,
                                          static_cast<unsigned int>(device_sm_count - sm_margin));
    NVTE_CHECK(num_groups > 0, "Failed to partition ", device_sm_count - sm_margin, " SMs.");

    // Platform partition granularity may not match the requested count exactly.
    const int actual_sm_count = static_cast<int>(carved_resource.sm.smCount);
    if (actual_sm_count != device_sm_count - sm_margin) {
      NVTE_WARN("Requested reserving ", sm_margin,
                " SMs, but the platform's green-context partition granularity actually "
                "reserves ",
                device_sm_count - actual_sm_count, " SMs.");
    }

    CUdevResourceDesc desc;
    NVTE_CALL_CHECK_CUDA_DRIVER_VERSIONED(cuDevResourceGenerateDesc, CUDART_VERSION, &desc,
                                          &carved_resource, 1u);

    NVTE_CALL_CHECK_CUDA_DRIVER_VERSIONED(cuGreenCtxCreate, CUDART_VERSION, &green_ctx, desc,
                                          cu_device, CU_GREEN_CTX_DEFAULT_STREAM);
  } catch (const std::exception &e) {
    // Platform/driver doesn't support green contexts; ignore the margin rather than failing
    // operations that would otherwise work fine unrestricted.
    NVTE_WARN("Could not reserve SMs via a green context (", e.what(),
              "); the requested margin will be ignored.");
    green_ctx = nullptr;
  }
  per_device[sm_margin] = green_ctx;
  return green_ctx;
#else
  NVTE_WARN(
      "Reserving SMs via a green context requires CUDA 12.4 or later; the requested margin "
      "will be ignored.");
  return nullptr;
#endif
}

// A stream + fork/join events, cached per (device, margin, caller stream) so independent
// callers don't serialize on one shared carveout stream.
struct PerCallerStream {
  cudaStream_t carveout_stream = nullptr;
  cudaEvent_t fork_event = nullptr;
  cudaEvent_t join_event = nullptr;
};

struct PerCallerStreamKey {
  int device_id;
  int sm_margin;
  cudaStream_t caller_stream;
  bool operator==(const PerCallerStreamKey &other) const {
    return device_id == other.device_id && sm_margin == other.sm_margin &&
           caller_stream == other.caller_stream;
  }
};

struct PerCallerStreamKeyHash {
  size_t operator()(const PerCallerStreamKey &key) const {
    size_t h = std::hash<int>()(key.device_id);
    h = h * 31 + std::hash<int>()(key.sm_margin);
    h = h * 31 + std::hash<void *>()(static_cast<void *>(key.caller_stream));
    return h;
  }
};

void destroy_per_caller_stream(const PerCallerStream &entry) {
  if (entry.carveout_stream != nullptr) cudaStreamDestroy(entry.carveout_stream);
  if (entry.fork_event != nullptr) cudaEventDestroy(entry.fork_event);
  if (entry.join_event != nullptr) cudaEventDestroy(entry.join_event);
}

// Bounds the number of distinct caller streams kept alive at once so a workload that churns
// through many short-lived streams doesn't accumulate green-context streams/events forever.
constexpr size_t kMaxCachedPerCallerStreams = 64;

PerCallerStream get_per_caller_stream(int device_id, int sm_margin, cudaStream_t caller_stream) {
  struct Node {
    PerCallerStream stream;
    std::list<PerCallerStreamKey>::iterator lru_it;
  };
  static std::unordered_map<PerCallerStreamKey, Node, PerCallerStreamKeyHash> cache;
  static std::list<PerCallerStreamKey> lru_order;  // front = most recently used
  static std::mutex mutex;

  const PerCallerStreamKey key{device_id, sm_margin, caller_stream};
  std::lock_guard<std::mutex> lock(mutex);

  auto it = cache.find(key);
  if (it != cache.end()) {
    lru_order.splice(lru_order.begin(), lru_order, it->second.lru_it);
    return it->second.stream;
  }

  PerCallerStream entry;
  const CUgreenCtx green_ctx = get_green_context(device_id, sm_margin);
  if (green_ctx != nullptr) {
    CUstream cu_stream;
    NVTE_CALL_CHECK_CUDA_DRIVER_VERSIONED(cuGreenCtxStreamCreate, CUDART_VERSION, &cu_stream,
                                          green_ctx, CU_STREAM_NON_BLOCKING, 0);
    entry.carveout_stream = reinterpret_cast<cudaStream_t>(cu_stream);
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&entry.fork_event, cudaEventDisableTiming));
    NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&entry.join_event, cudaEventDisableTiming));
  }

  lru_order.push_front(key);
  cache[key] = Node{entry, lru_order.begin()};

  if (cache.size() > kMaxCachedPerCallerStreams) {
    const PerCallerStreamKey evict_key = lru_order.back();
    destroy_per_caller_stream(cache.at(evict_key).stream);
    cache.erase(evict_key);
    lru_order.pop_back();
  }

  return entry;
}

}  // namespace

cudaStream_t sm_carveout_stream_begin(cudaStream_t stream, int sm_margin) {
  if (sm_margin == 0) return stream;
  const int device_id = current_device();
  PerCallerStream entry = get_per_caller_stream(device_id, sm_margin, stream);
  if (entry.carveout_stream == nullptr) return stream;

  NVTE_CHECK_CUDA(cudaEventRecord(entry.fork_event, stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(entry.carveout_stream, entry.fork_event, 0));
  return entry.carveout_stream;
}

void sm_carveout_stream_end(cudaStream_t stream, cudaStream_t carveout_stream, int sm_margin) {
  if (carveout_stream == stream) return;
  PerCallerStream entry = get_per_caller_stream(current_device(), sm_margin, stream);

  NVTE_CHECK_CUDA(cudaEventRecord(entry.join_event, carveout_stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream, entry.join_event, 0));
}

}  // namespace cuda
}  // namespace transformer_engine
