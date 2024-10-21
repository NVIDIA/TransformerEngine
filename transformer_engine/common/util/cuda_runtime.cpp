/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/cuda_runtime.h"

#include <filesystem>
#include <mutex>

#include "../common.h"
#include "../util/cuda_driver.h"
#include "../util/system.h"
#include "common/util/cuda_runtime.h"

namespace transformer_engine {

namespace cuda {

namespace {

// String with build-time CUDA include path
#include "string_path_cuda_include.h"

}  // namespace

int num_devices() {
  auto query_num_devices = []() -> int {
    int count;
    NVTE_CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
  };
  static int num_devices_ = query_num_devices();
  return num_devices_;
}

int current_device() {
  // Return 0 if CUDA context is not initialized
  CUcontext context;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxGetCurrent, &context);
  if (context == nullptr) {
    return 0;
  }

  // Query device from CUDA runtime
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
  auto init = [&]() {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = 10 * prop.major + prop.minor;
  };
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
  auto init = [&]() {
    cudaDeviceProp prop;
    NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    cache[device_id] = prop.multiProcessorCount;
  };
  std::call_once(flags[device_id], init);
  return cache[device_id];
}

bool supports_multicast(int device_id) {
  static std::vector<bool> cache(num_devices(), false);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id  = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&]() {
    int cuda_runtime_version;
    NVTE_CHECK_CUDA(cudaRuntimeGetVersion(&cuda_runtime_version));
    if (cuda_runtime_version >= 12010) {
      // On CUDA 12.1+, we can directly query driver for the multicast support property
      CUdevice cudev;
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &cudev, device_id);
      int result;
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGetAttribute, &result,
                                  CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cudev);
      cache[device_id] = static_cast<bool>(result);
    } else {
      // Otherwise, we can assume CUDA Multicast is supported with CUDA 12.0+ and SM arch 9.0+
      cache[device_id] = (cuda_runtime_version >= 12000 && sm_arch() > 90);
    }
  };
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
    std::vector<std::pair<std::string, Path>> search_paths = {{"NVTE_CUDA_INCLUDE_DIR", ""},
                                                              {"CUDA_HOME", ""},
                                                              {"CUDA_DIR", ""},
                                                              {"", string_path_cuda_include},
                                                              {"", "/usr/local/cuda"}};
    for (auto &[env, p] : search_paths) {
      if (p.empty()) {
        p = getenv<Path>(env.c_str());
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
      message +=
          (". "
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
