/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cuda_context.h"

#include <transformer_engine/cuda_context.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine::cuda_context {

struct DeviceState {
  std::atomic<uint64_t> generation{1};
  std::set<Resource *> resources;
};

namespace {
struct Registry {
  // Lifecycle callbacks may destroy other registered objects. Membership uses
  // its own mutex, so their destructors can unregister without re-entering the
  // lifecycle lock. Public generation queries wait for a complete operation.
  std::mutex lifecycle_mutex;
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

ExternalResource::ExternalResource(int device, const char *owner) : owner_(owner) {
  bind_context(device);
}

void ExternalResource::check_context_loss() const {
  NVTE_ERROR("Cannot recover active ", owner_,
             " after CUDA context loss. Its communicator/IPC/VMM state must be released "
             "before context destruction; otherwise restart the worker.");
}

void ExternalResource::check_release() const {}

void ExternalResource::release_context() {
  NVTE_ERROR("Release all owners of ", owner_,
             " while the CUDA context and communicator are usable, then release TE state.");
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

void reset(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
  std::unique_lock<std::mutex> lock(state.mutex);
  auto &context = state.devices[device];
  const auto next = context.generation.load(std::memory_order_relaxed) + 1;
  NVTE_CHECK(next != 0 && state.generation.load(std::memory_order_relaxed) !=
                              std::numeric_limits<uint64_t>::max(),
             "CUDA context generation overflow.");
  for (auto *resource : context.resources) resource->check_context_loss();
  const auto resources = context.resources;
  // Callbacks must not destroy old CUDA handles or register new resources. The
  // caller has stopped all users, including other threads' TLS destructors.
  // Public device-generation queries take this mutex and cannot observe a
  // partially invalidated registry.
  for (auto *resource : resources) {
    if (context.resources.count(resource) == 0) continue;
    lock.unlock();
    resource->invalidate_context();
    lock.lock();
  }
  context.generation.store(next, std::memory_order_release);
  state.generation.fetch_add(1, std::memory_order_release);
}

void release(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
  std::unique_lock<std::mutex> lock(state.mutex);
  auto &context = state.devices[device];
  const auto next = context.generation.load(std::memory_order_relaxed) + 1;
  NVTE_CHECK(next != 0 && state.generation.load(std::memory_order_relaxed) !=
                              std::numeric_limits<uint64_t>::max(),
             "CUDA context generation overflow.");
  for (auto *resource : context.resources) resource->check_release();
  std::vector<Resource *> resources(context.resources.begin(), context.resources.end());
  std::stable_sort(resources.begin(), resources.end(),
                   [](auto *a, auto *b) { return a->release_order() < b->release_order(); });
  int previous;
  NVTE_CHECK_CUDA(cudaGetDevice(&previous));
  NVTE_CHECK_CUDA(cudaSetDevice(device));
  std::exception_ptr error;
  for (auto *resource : resources) {
    if (context.resources.count(resource) == 0) continue;
    lock.unlock();
    try {
      resource->release_context();
    } catch (...) {
      if (!error) error = std::current_exception();
    }
    lock.lock();
  }
  context.generation.store(next, std::memory_order_release);
  state.generation.fetch_add(1, std::memory_order_release);
  const auto restore_status = cudaSetDevice(previous);
  if (error) std::rethrow_exception(error);
  NVTE_CHECK_CUDA(restore_status);
}

uint64_t device_generation(int device) {
  NVTE_CHECK(device >= 0, "Invalid CUDA device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
  std::lock_guard<std::mutex> lock(state.mutex);
  return state.devices[device].generation.load(std::memory_order_acquire);
}

}  // namespace transformer_engine::cuda_context

struct NVTECudaStateRegistrationImpl : transformer_engine::cuda_context::Resource {
  NVTECudaStateRegistrationImpl(int device, NVTECudaStateCallbacks callbacks, void *data)
      : callbacks(callbacks), data(data) {
    NVTE_CHECK(callbacks.release != nullptr && callbacks.invalidate != nullptr,
               "CUDA state release/invalidate callbacks must not be null.");
    bind_context(device);
  }
  void check_context_loss() const override {
    if (callbacks.check_context_loss) callbacks.check_context_loss(data);
  }
  void check_release() const override {
    if (callbacks.check_release) callbacks.check_release(data);
  }
  int release_order() const override { return callbacks.release_order; }
  void release_context() override { callbacks.release(data); }
  void invalidate_context() noexcept override { callbacks.invalidate(data); }
  NVTECudaStateCallbacks callbacks;
  void *data;
};

NVTECudaStateRegistration nvte_register_cuda_state(int device, NVTECudaStateCallbacks callbacks,
                                                   void *user_data) {
  return new NVTECudaStateRegistrationImpl(device, callbacks, user_data);
}

void nvte_unregister_cuda_state(NVTECudaStateRegistration registration) { delete registration; }

void nvte_reset_cuda_state(int device) { transformer_engine::cuda_context::reset(device); }

void nvte_release_cuda_state(int device) { transformer_engine::cuda_context::release(device); }

uint64_t nvte_get_cuda_state_version() { return transformer_engine::cuda_context::generation(); }

uint64_t nvte_get_cuda_context_generation(int device) {
  return transformer_engine::cuda_context::device_generation(device);
}
