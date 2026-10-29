/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cuda_context.h"

#include <transformer_engine/cuda_context.h>

#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <set>

#include "logging.h"

namespace transformer_engine::cuda_context {

struct DeviceState {
  std::atomic<uint64_t> generation{1};
  std::set<Resource *> resources;
};

namespace {
struct Registry {
  std::mutex mutex;
  std::map<int, DeviceState> devices;
  std::atomic<uint64_t> generation{1};
};

Registry &registry() {
  // TLS resource destructors can run after ordinary static destruction.
  static auto &instance = *new Registry;
  return instance;
}
}  // namespace

Resource::~Resource() {
  if (context_ != nullptr) {
    auto &state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    context_->resources.erase(this);
  }
}

void Resource::bind_context(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  if (device_ == device) return;
  auto &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto &context = state.devices[device];
  if (context_ == &context) return;
  // Allocate the new registration before removing the old one.
  context.resources.insert(this);
  if (context_ != nullptr) context_->resources.erase(this);
  context_ = &context;
  device_ = device;
}

uint64_t generation() { return registry().generation.load(std::memory_order_acquire); }

Generation capture_generation(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto &counter = state.devices[device].generation;
  return {&counter, counter.load(std::memory_order_acquire)};
}

void reset(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto &context = state.devices[device];
  const auto next = context.generation.load(std::memory_order_relaxed) + 1;
  NVTE_CHECK(next != 0 && state.generation.load(std::memory_order_relaxed) !=
                              std::numeric_limits<uint64_t>::max(),
             "CUDA context generation overflow.");
  // Callbacks must not destroy old CUDA handles or mutate registry membership. The caller has
  // stopped all users, including other threads' TLS resource destructors.
  // Invalidate shared RAII snapshots before callbacks release active owners.
  // Public device-generation queries take this mutex and cannot observe a
  // partially invalidated registry.
  context.generation.store(next, std::memory_order_release);
  for (auto *resource : context.resources) resource->invalidate_context();
  state.generation.fetch_add(1, std::memory_order_release);
}

uint64_t device_generation(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.devices[device].generation.load(std::memory_order_acquire);
}

}  // namespace transformer_engine::cuda_context

void nvte_reset_cuda_state(int device) { transformer_engine::cuda_context::reset(device); }

uint64_t nvte_get_cuda_context_generation(int device) {
  return transformer_engine::cuda_context::device_generation(device);
}
