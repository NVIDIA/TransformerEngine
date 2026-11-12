/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_CUDA_STATE_REGISTRATION_H_
#define TRANSFORMER_ENGINE_UTIL_CUDA_STATE_REGISTRATION_H_

#include <transformer_engine/cuda_context.h>

namespace transformer_engine::cuda_context {

// Framework bindings use only the common C ABI, not common's hidden C++ RTTI.
class FrontendResource {
 public:
  explicit FrontendResource(int device = -1, int release_order = NVTE_CUDA_STATE_RELEASE_PLANS) {
    // Store the explicit priority; virtual dispatch is not available for the
    // derived owner during base construction. Trampolines run after construction.
    const NVTECudaStateCallbacks callbacks{
        [](void *p) { static_cast<FrontendResource *>(p)->check_context_loss(); },
        [](void *p) { static_cast<FrontendResource *>(p)->check_release(); },
        [](void *p) { static_cast<FrontendResource *>(p)->release_context(); },
        [](void *p) { static_cast<FrontendResource *>(p)->invalidate_context(); }, release_order};
    registration_ = nvte_register_cuda_state(device, callbacks, this);
  }
  FrontendResource(const FrontendResource &) = delete;
  FrontendResource &operator=(const FrontendResource &) = delete;
  virtual ~FrontendResource() { nvte_unregister_cuda_state(registration_); }
  void set_device(int device) { nvte_set_cuda_state_device(registration_, device); }

 private:
  virtual void check_context_loss() const {}
  virtual void check_release() const {}
  virtual void release_context() = 0;
  virtual void invalidate_context() noexcept = 0;
  NVTECudaStateRegistration registration_ = nullptr;
};

}  // namespace transformer_engine::cuda_context

#endif  // TRANSFORMER_ENGINE_UTIL_CUDA_STATE_REGISTRATION_H_
