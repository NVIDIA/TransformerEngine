/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_
#define TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_

#include <memory>
#include <vector>

#include "cuda_context.h"
#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine::detail {

template <typename Handle, void Create(Handle*), void Destroy(Handle)>
class HandleManager {
 public:
  static HandleManager& Instance() {
    static thread_local HandleManager instance;
    return instance;
  }

  Handle GetHandle() {
    const int device_id = cuda::current_device();
    NVTE_CHECK(0 <= device_id && device_id < handles_.size(), "invalid CUDA device ID");
    auto& slot = handles_[device_id];
    if (!slot) slot = acquire_slot(device_id);
    if (slot->handle == nullptr) {
      Handle handle = nullptr;
      Create(&handle);
      slot->handle = handle;
    }
    return slot->handle;
  }

 private:
  struct Slot : cuda_context::Resource {
    explicit Slot(int device) { bind_context(device); }
    // Explicit release owns teardown. Calling CUDA from TLS/static destruction
    // is unsafe when the runtime is already terminating.
    int release_order() const override { return 100; }
    void release_context() override {
      if (handle == nullptr) return;
      Destroy(handle);
      handle = nullptr;
    }
    void invalidate_context() noexcept override {
      // cudnnDestroy may dereference memory released by cudaDeviceReset.
      handle = nullptr;
    }
    Handle handle = nullptr;
    Slot* next_idle = nullptr;
  };

  HandleManager() : handles_(cuda::num_devices()) {}
  ~HandleManager() {
    // A finished thread must not strand vendor allocations outside the reset
    // registry. Return ownership to the process pool using host operations only.
    for (size_t device = 0; device < handles_.size(); ++device) {
      if (auto* slot = handles_[device]) {
        auto& state = pool();
        std::lock_guard<std::mutex> lock(state.mutex);
        slot->next_idle = state.idle[device];
        state.idle[device] = slot;
      }
    }
  }

  struct Pool {
    explicit Pool(int devices) : idle(devices, nullptr) {}
    std::mutex mutex;
    std::vector<std::unique_ptr<Slot>> owners;
    std::vector<Slot*> idle;
  };
  static Pool& pool() {
    // Avoid CUDA calls during process/static teardown. The number of slots is
    // bounded by peak concurrent thread use, and live release frees their handles.
    static auto& state = *new Pool(cuda::num_devices());
    return state;
  }
  static Slot* acquire_slot(int device) {
    auto& state = pool();
    std::lock_guard<std::mutex> lock(state.mutex);
    auto* slot = state.idle[device];
    if (slot) {
      state.idle[device] = slot->next_idle;
      slot->next_idle = nullptr;
      return slot;
    }
    auto owner = std::make_unique<Slot>(device);
    slot = owner.get();
    state.owners.emplace_back(std::move(owner));
    return slot;
  }

  std::vector<Slot*> handles_;
};

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_
