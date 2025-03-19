/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_
#define TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_

#include <vector>

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
    static thread_local std::vector<bool> initialized(handles_.size(), false);
    const int device_id = cuda::current_device();
    NVTE_CHECK(0 <= device_id && device_id < handles_.size(), "invalid CUDA device ID");
    if (!initialized[device_id]) {
      Create(&(handles_[device_id]));
      initialized[device_id] = true;
    }
    return handles_[device_id];
  }

  ~HandleManager() {
    if (Destroy != nullptr) {
      for (auto& handle : handles_) {
        Destroy(handle);
      }
    }
  }

 private:
  HandleManager() : handles_(cuda::num_devices(), nullptr) {}

  std::vector<Handle> handles_ = nullptr;
};

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_UTIL_HANDLE_MANAGER_H_
