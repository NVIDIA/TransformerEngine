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
#include <mutex>
#include <vector>

#include "cuda_driver.h"
#include "logging.h"

namespace transformer_engine::cuda_context {

namespace {
struct Registry {
  // Lifecycle callbacks may destroy other registered objects. Membership uses
  // its own mutex, so their destructors can unregister without re-entering the
  // lifecycle lock. The global version is published after the full operation.
  std::mutex lifecycle_mutex;
  std::mutex mutex;
  std::vector<Resource *> resources;
  std::atomic<uint64_t> generation{1};
};

Registry &registry() {
  // TLS resource destructors can run after ordinary static destruction.
  static auto &instance = *new Registry;
  return instance;
}
// Only attach primary contexts that already exist. Cached RTC device entries
// may be registered for devices that have never had a context in this process.
template <typename Function>
void with_primary_context(int ordinal, Function &&function) {
  if (ordinal < 0) {
    function();
    return;
  }
  NVTE_CALL_CHECK_CUDA_DRIVER(cuInit, 0);
  CUdevice device;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &device, ordinal);
  unsigned flags;
  int active;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxGetState, device, &flags, &active);
  if (!active) {
    function();
    return;
  }
  CUcontext context;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxRetain, &context, device);
  const auto push_status = cuda_driver::call("cuCtxPushCurrent", context);
  if (push_status != CUDA_SUCCESS) {
    cuda_driver::call("cuDevicePrimaryCtxRelease", device);
    NVTE_CHECK_CUDA_DRIVER(push_status);
  }
  CUcontext popped;
  try {
    function();
  } catch (...) {
    cuda_driver::call("cuCtxPopCurrent", &popped);
    cuda_driver::call("cuDevicePrimaryCtxRelease", device);
    throw;
  }
  const auto pop_status = cuda_driver::call("cuCtxPopCurrent", &popped);
  const auto release_status = cuda_driver::call("cuDevicePrimaryCtxRelease", device);
  NVTE_CHECK_CUDA_DRIVER(pop_status);
  NVTE_CHECK_CUDA_DRIVER(release_status);
}
}  // namespace

Resource::~Resource() {
  if (registered_) {
    auto &state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto &resources = state.resources;
    resources.erase(std::remove(resources.begin(), resources.end(), this), resources.end());
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
  NVTE_CHECK(device >= -1, "Invalid CUDA cleanup device ordinal.");
  auto &state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!registered_) {
    state.resources.push_back(this);
    registered_ = true;
  }
  device_ = device;
}

uint64_t generation() { return registry().generation.load(std::memory_order_acquire); }

void reset() {
  auto &state = registry();
  std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
  std::unique_lock<std::mutex> lock(state.mutex);
  NVTE_CHECK(
      state.generation.load(std::memory_order_relaxed) != std::numeric_limits<uint64_t>::max(),
      "CUDA state generation overflow.");
  for (auto *resource : state.resources) resource->check_context_loss();
  const auto resources = state.resources;
  // All users are stopped and all contexts backing registered resources have
  // been lost. No CUDA calls or new registrations are allowed during reset.
  for (auto *resource : resources) {
    if (std::find(state.resources.begin(), state.resources.end(), resource) ==
        state.resources.end())
      continue;
    lock.unlock();
    resource->invalidate_context();
    lock.lock();
  }
  state.generation.fetch_add(1, std::memory_order_release);
}

void release() {
  auto &state = registry();
  std::lock_guard<std::mutex> lifecycle_lock(state.lifecycle_mutex);
  std::unique_lock<std::mutex> lock(state.mutex);
  NVTE_CHECK(
      state.generation.load(std::memory_order_relaxed) != std::numeric_limits<uint64_t>::max(),
      "CUDA state generation overflow.");
  for (auto *resource : state.resources) resource->check_release();
  auto resources = state.resources;
  std::stable_sort(resources.begin(), resources.end(),
                   [](auto *a, auto *b) { return a->release_order() < b->release_order(); });
  std::exception_ptr error;
  for (auto *resource : resources) {
    if (std::find(state.resources.begin(), state.resources.end(), resource) ==
        state.resources.end())
      continue;
    lock.unlock();
    try {
      with_primary_context(resource->device_, [&] { resource->release_context(); });
    } catch (...) {
      if (!error) error = std::current_exception();
    }
    lock.lock();
  }
  state.generation.fetch_add(1, std::memory_order_release);
  if (error) std::rethrow_exception(error);
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

void nvte_reset_cuda_state() { transformer_engine::cuda_context::reset(); }

void nvte_release_cuda_state() { transformer_engine::cuda_context::release(); }

uint64_t nvte_get_cuda_state_version() { return transformer_engine::cuda_context::generation(); }
