/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_
#define TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_

#include <atomic>
#include <cstdint>

namespace transformer_engine::cuda_context {

struct DeviceState;

// A snapshot for shared RAII owners that can outlive their active catalog.
// The counter has process lifetime. A lost generation must never call CUDA.
struct Generation {
  const std::atomic<uint64_t> *counter = nullptr;
  uint64_t value = 0;
  bool current() const {
    return counter == nullptr || counter->load(std::memory_order_acquire) == value;
  }
};
Generation capture_generation(int device);

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
  virtual void invalidate_context() noexcept = 0;
  DeviceState *context_ = nullptr;
  int device_ = -1;
};

// Process-wide notification for host TLS that needs to recheck CUDA following
// ANY device reset. Device resource ownership remains scoped to the device.
uint64_t generation();
void reset(int device);

}  // namespace transformer_engine::cuda_context

#endif  // TRANSFORMER_ENGINE_UTIL_CUDA_CONTEXT_H_
