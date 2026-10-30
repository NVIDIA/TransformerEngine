/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_
#define TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_

#include <atomic>
#include <cstdint>
#include <mutex>

namespace transformer_engine::cuda_context {

struct DeviceState;

// Registered objects keep stable addresses. Reset and destruction must be
// externally serialized with all TE calls. Registration is lazy and host-only.
class Resource {
 public:
  Resource() = default;
  Resource(const Resource &) = delete;
  Resource &operator=(const Resource &) = delete;
  virtual ~Resource();
  void bind_context(int device);

 private:
  friend void reset(int device);
  friend void release(int device);
  // Preflight runs before any resource or generation changes. An owner can
  // reject a lifecycle operation that its dependency cannot safely perform.
  virtual void check_context_loss() const {}
  virtual void check_release() const {}
  virtual int release_order() const { return 0; }
  virtual void release_context() = 0;
  virtual void invalidate_context() noexcept = 0;
  DeviceState *context_ = nullptr;
  int device_ = -1;
};

// Communicators and VMM/IPC registrations are not owned by a CUDA context.
// Their public owner must be destroyed while its transport is still usable.
// Refuse a false-success reset if a caller retains such an owner.
class ExternalResource final : public Resource {
 public:
  ExternalResource(int device, const char *owner);

 private:
  void check_context_loss() const override;
  void check_release() const override;
  void release_context() override;
  int release_order() const override { return 1000; }
  void invalidate_context() noexcept override {}
  const char *owner_;
};

// A resettable once flag for context-local kernel attributes. Keep one stable
// instance per device. The fast path does not query CUDA or lock the registry.
class InitializationFlag final : public Resource {
 public:
  template <typename Function>
  void call(int device, Function &&initialize) {
    if (initialized_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_.load(std::memory_order_relaxed)) return;
    bind_context(device);
    initialize();
    initialized_.store(true, std::memory_order_release);
  }

 private:
  void release_context() override { invalidate_context(); }
  void invalidate_context() noexcept override {
    initialized_.store(false, std::memory_order_release);
  }
  std::atomic<bool> initialized_{false};
  std::mutex mutex_;
};

// Process-wide notification for host TLS that needs to recheck CUDA following
// ANY device reset. Device resource ownership remains scoped to the device.
uint64_t generation();
void reset(int device);
void release(int device);

}  // namespace transformer_engine::cuda_context

#endif  // TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_
