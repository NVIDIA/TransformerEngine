/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <filesystem>
#include <mutex>  // NOLINT(*)

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"

namespace transformer_engine {

namespace cuda {

int num_devices() {
  static int num_devices_ = -1;
  static std::once_flag flag;
  auto init = [&] () {
    NVTE_CHECK_CUDA(cudaGetDeviceCount(&num_devices_));
  }
  std::call_once(flag, init);
  return num_devices_;
}

int current_device() {
  int device_id;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  return device_id;
}

int sm_arch(int device_id) {
  static std::vector<int> cache(num_devices(), -1);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&] () {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = 10*prop.major + prop.minor;
  }
  std::call_once(flags[device_id], init);
  return cache[device_id];
}

int sm_count(int device_id) {
  static std::vector<int> cache(num_devices(), -1);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&] () {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = prop.multiProcessorCount;
  }
  std::call_once(flags[device_id], init);
  return cache[device_id];
}

const std::string &include_directory(bool required) {
  static std::string path;

  // Update cached path if needed
  static bool need_to_check_env = true;
  if (path.empty() && required) {
    need_to_check_env = true;
  }
  if (need_to_check_env) {
    // Search for CUDA headers in common paths
    using Path = std::filesystem::path;
    std::vector<std::pair<std::string, Path>> search_paths = {
      {"NVTE_CUDA_INCLUDE_DIR", ""},
      {"CUDA_HOME", ""},
      {"CUDA_DIR", ""},
      {"", "/usr/local/cuda"}};
    for (auto &[env, p] : search_paths) {
      if (p.empty()) {
        p = getenv<Path>(env);
      }
      if (!p.empty()) {
        if (file_exists(p / "cuda_runtime.h")) {
          path = p;
          break;
        }
        if (file_exists(p / "include" / "cuda_runtime.h")) {
          path = p / "include";
          break;
        }
      }
    }

    // Throw exception if path is required but not found
    if (path.empty() && required) {
      std::string message;
      message.reserve(2048);
      message += "Could not find cuda_runtime.h in";
      bool is_first = true;
      for (const auto &[env, p] : search_paths) {
        message += is_first ? " " : ", ";
        is_first = false;
        if (!env.empty()) {
          message += env;
          message += "=";
        }
        if (p.empty()) {
          message += "<unset>";
        } else {
          message += p;
        }
      }
      message += (". "
                  "Specify path to CUDA Toolkit headers "
                  "with NVTE_CUDA_INCLUDE_DIR "
                  "or disable NVRTC support with NVTE_DISABLE_NVRTC=1.");
      NVTE_ERROR(message);
    }
    need_to_check_env = false;
  }

  // Return cached path
  return path;
}

}  // namespace cuda

}  // namespace transformer_engine
