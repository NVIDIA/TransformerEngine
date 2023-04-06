/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace cuda {

int num_devices() {
  static int num_devices_ = -1;
  if (num_devices_ < 0) {
    NVTE_CHECK_CUDA(cudaGetDeviceCount(&num_devices_));
  }
  return num_devices_;
}

int current_device() {
  int device_id;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  return device_id;
}

int sm_arch(int device_id) {
  static std::vector<int> cache(num_devices(), -1);
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  if (cache[device_id] < 0) {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = 10*prop.major + prop.minor;
  }
  return cache[device_id];
}

const std::string &include_directory() {
  static std::string path;
  if (path.empty()) {
    /// TODO Make configurable
    /// TODO Make abspath
    path = "/usr/local/cuda/include";
  }
  return path;
}

}  // namespace cuda

}  // namespace transformer_engine
