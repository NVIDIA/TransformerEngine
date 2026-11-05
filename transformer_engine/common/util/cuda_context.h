/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_
#define TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_

#include <transformer_engine/cuda_context.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

namespace transformer_engine::cuda_context {

// Registered objects keep stable addresses. Reset and destruction must be
// externally serialized with all TE calls and resource construction. Base
// construction registers the address using host operations only.
class Resource {
 public:
  explicit Resource(int device = -1, int release_order = NVTE_CUDA_STATE_RELEASE_PLANS);
  Resource(const Resource &) = delete;
  Resource &operator=(const Resource &) = delete;
  virtual ~Resource();
  // Device affinity selects a live cleanup context; -1 is host-only or multi-device.
  // Updating affinity does not change the original registration order.
  void set_device(int device);

 private:
  friend void reset();
  friend void release();
  // Preflight runs before any resource or generation changes. An owner can
  // reject a lifecycle operation that its dependency cannot safely perform.
  virtual void check_context_loss() const {}
  virtual void check_release() const {}
  virtual void release_context() = 0;
  virtual void invalidate_context() noexcept = 0;
  int device_;
  const int release_order_;
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
  void invalidate_context() noexcept override {}
  const char *owner_;
};

// Synchronizes successful lazy initialization. A throwing initializer leaves
// the state invalid and may be retried. Invalidation requires quiescent users.
class InitializationState {
 public:
  template <typename Function>
  void ensure_valid(Function &&initialize) {
    if (initialized_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_.load(std::memory_order_relaxed)) return;
    initialize();
    initialized_.store(true, std::memory_order_release);
  }
  void invalidate() noexcept { initialized_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> initialized_{false};
  std::mutex mutex_;
};

// Optional lazy initialization for owners such as library handles and modules.
// Parameters belong to the caller's initializer, not to a universal virtual init.
class LazyResource : public Resource {
 public:
  using Resource::Resource;
  template <typename Function>
  void ensure_valid(Function &&initialize) {
    initialization_.ensure_valid(std::forward<Function>(initialize));
  }

 private:
  void release_context() final {
    // Always visit the owner: failed initialization may have acquired part of
    // its state before throwing. Empty owners must support no-op cleanup.
    release_resources();
    initialization_.invalidate();
  }
  void invalidate_context() noexcept final {
    invalidate_resources();
    initialization_.invalidate();
  }
  virtual void release_resources() = 0;
  virtual void invalidate_resources() noexcept = 0;
  InitializationState initialization_;
};

// Resettable context-local kernel attributes share the same initialization state.
class InitializationFlag final : public LazyResource {
 public:
  template <typename Function>
  void call(int device, Function &&initialize) {
    ensure_valid([&] {
      set_device(device);
      initialize();
    });
  }

 private:
  void release_resources() override {}
  void invalidate_resources() noexcept override {}
};

// Process-wide notification for host TLS that needs to recheck CUDA following
// a global lifecycle operation. Device affinity is only used for live cleanup.
uint64_t generation();
void reset();
void release();

}  // namespace transformer_engine::cuda_context

#endif  // TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_
