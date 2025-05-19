/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

void stream_priority_range(int *low_priority, int *high_priority, int device_id) {
  static std::vector<std::pair<int, int>> cache(num_devices());
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&]() {
    int ori_dev = current_device();
    if (device_id != ori_dev) NVTE_CHECK_CUDA(cudaSetDevice(device_id));
    int min_pri, max_pri;
    NVTE_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&min_pri, &max_pri));
    if (device_id != ori_dev) NVTE_CHECK_CUDA(cudaSetDevice(ori_dev));
    cache[device_id] = std::make_pair(min_pri, max_pri);
  };
  std::call_once(flags[device_id], init);
  *low_priority = cache[device_id].first;
  *high_priority = cache[device_id].second;
}

bool supports_multicast(int device_id) {
#if CUDART_VERSION >= 12010
  // NOTE: This needs to be guarded at compile time because the
  //       CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED enum is not defined in earlier CUDA versions.
  static std::vector<bool> cache(num_devices(), false);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&]() {
    CUdevice cudev;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &cudev, device_id);
    // Multicast support requires both CUDA12.1 UMD + KMD
    int result = 0;
    // Check if KMD >= 12.1
    int driver_version;
    NVTE_CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    if (driver_version >= 12010) {
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGetAttribute, &result,
                                  CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cudev);
    }
    cache[device_id] = static_cast<bool>(result);
  };
  std::call_once(flags[device_id], init);
  return cache[device_id];
#else
  return false;
#endif
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
