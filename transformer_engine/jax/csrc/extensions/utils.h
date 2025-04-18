/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>
#include <transformer_engine/fused_attn.h>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/util/logging.h"

namespace transformer_engine {
namespace jax {

int GetCudaRuntimeVersion();
size_t GetCudnnRuntimeVersion();
int GetDeviceComputeCapability(int gpu_id);

class cudaDevicePropertiesManager {
 public:
  static cudaDevicePropertiesManager &Instance() {
    static thread_local cudaDevicePropertiesManager instance;
    return instance;
  }

  int GetMultiProcessorCount() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.multiProcessorCount;
  }

  int GetMajor() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.major;
  }

 private:
  bool prop_queried_ = false;
  cudaDeviceProp prop_;
};

}  // namespace jax
}  // namespace transformer_engine
