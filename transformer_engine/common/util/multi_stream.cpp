/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
#define TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_

#include "multi_stream.h"

#include <transformer_engine/multi_stream.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "cuda_context.h"
#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine::detail {

namespace {
struct ComputeResources : cuda_context::Resource {
  struct Slot {
    std::atomic<cudaStream_t> stream{nullptr};
    std::atomic<cudaEvent_t> event{nullptr};
  };
  explicit ComputeResources(int device) : slots(nvte_get_num_compute_streams()) {
    bind_context(device);
  }
  void invalidate_context() noexcept override {
    for (auto &slot : slots) {
      slot.stream.store(nullptr, std::memory_order_release);
      slot.event.store(nullptr, std::memory_order_release);
    }
  }
  std::mutex mutex;
  std::vector<Slot> slots;
};

ComputeResources &compute_resources() {
  static auto &mutex = *new std::mutex;
  static auto &devices = *new std::map<int, std::unique_ptr<ComputeResources>>;
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  static thread_local std::map<int, ComputeResources *> local;
  const auto found = local.find(device);
  if (found != local.end()) return *found->second;
  std::lock_guard<std::mutex> lock(mutex);
  auto &resources = devices[device];
  if (!resources) resources = std::make_unique<ComputeResources>(device);
  local.emplace(device, resources.get());
  return *resources;
}
}  // namespace

cudaStream_t get_compute_stream(int idx) {
  const size_t num_streams = nvte_get_num_compute_streams();
  NVTE_CHECK(0 <= idx && idx < num_streams, "Invalid compute stream (requested idx ", idx,
             ", but there are ", num_streams, " streams)");
  auto &resources = compute_resources();
  auto &cached = resources.slots[idx].stream;
  auto stream = cached.load(std::memory_order_acquire);
  if (stream != nullptr) return stream;
  std::lock_guard<std::mutex> lock(resources.mutex);
  stream = cached.load(std::memory_order_acquire);
  if (stream == nullptr) {
    NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
    cached.store(stream, std::memory_order_release);
  }
  return stream;
}

cudaEvent_t get_compute_stream_event(int idx) {
  const size_t num_streams = nvte_get_num_compute_streams();
  NVTE_CHECK(0 <= idx && idx < num_streams, "Invalid compute stream (requested idx ", idx,
             ", but there are ", num_streams, " streams)");
  auto &resources = compute_resources();
  auto &cached = resources.slots[idx].event;
  auto event = cached.load(std::memory_order_acquire);
  if (event != nullptr) return event;
  std::lock_guard<std::mutex> lock(resources.mutex);
  event = cached.load(std::memory_order_acquire);
  if (event == nullptr) {
    NVTE_CHECK_CUDA(cudaEventCreate(&event));
    cached.store(event, std::memory_order_release);
  }
  return event;
}

int get_num_compute_streams() {
  static constexpr int num_compute_streams = 4;
  return num_compute_streams;
}

}  // namespace transformer_engine::detail

int nvte_get_num_compute_streams() { return transformer_engine::detail::get_num_compute_streams(); }

cudaStream_t nvte_get_compute_stream(const int idx) {
  return transformer_engine::detail::get_compute_stream(idx);
}

cudaEvent_t nvte_get_compute_stream_event(const int idx) {
  return transformer_engine::detail::get_compute_stream_event(idx);
}

#endif  // TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
