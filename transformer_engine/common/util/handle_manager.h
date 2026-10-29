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

template <typename Handle, void Create(Handle*), void Destroy(Handle) = nullptr>
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
    if (!slot) slot = std::make_unique<Slot>(device_id);
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
    ~Slot() override {
      if (Destroy != nullptr && handle != nullptr) Destroy(handle);
    }
    void invalidate_context() noexcept override {
      // cudnnDestroy may dereference memory released by cudaDeviceReset.
      handle = nullptr;
    }
    Handle handle = nullptr;
  };

  HandleManager() : handles_(cuda::num_devices()) {}

  std::vector<std::unique_ptr<Slot>> handles_;
};

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_
