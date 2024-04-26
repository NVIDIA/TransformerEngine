/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <filesystem>
#include <mutex>

#include "../common.h"
#include "../util/cuda_driver.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"

namespace transformer_engine {

namespace cuda {

namespace {

// String with build-time CUDA include path
#include "string_path_cuda_include.h"

}  // namespace

int num_devices() {
  auto query_num_devices = []() -> int {
    try {
  int count;
    /*
    DPCT1009:294: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    NVTE_CHECK_CUDA(
        DPCT_CHECK_ERROR(count = dpct::dev_mgr::instance().device_count()));
    return count;
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
  };
  static int num_devices_ = query_num_devices();
  return num_devices_;
}

int current_device() try {
  // Return 0 if CUDA context is not initialized
  int context;
  NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxGetCurrent, &context);
  if (context == nullptr) {
    return 0;
  }

  // Query device from CUDA runtime
  int device_id;
  /*
  DPCT1009:297: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  NVTE_CHECK_CUDA(DPCT_CHECK_ERROR(
      device_id = dpct::dev_mgr::instance().current_device_id()));
  return device_id;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int sm_arch(int device_id) {
  static std::vector<int> cache(num_devices(), -1);
  static std::vector<std::once_flag> flags(num_devices());
  if (device_id < 0) {
    device_id = current_device();
  }
  NVTE_CHECK(0 <= device_id && device_id < num_devices(), "invalid CUDA device ID");
  auto init = [&]() {
    try {
  dpct::device_info prop;
    /*
    DPCT1009:298: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    NVTE_CHECK_CUDA(DPCT_CHECK_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device_id))));
    /*
    DPCT1005:299: The SYCL device version is different from CUDA Compute
    Compatibility. You may need to rewrite this code.
    */
    cache[device_id] = 10 * prop.get_major_version() + prop.get_minor_version();
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }
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
    try {
  dpct::device_info prop;
    /*
    DPCT1009:300: SYCL uses exceptions to report errors and does not use the
    error codes. The original code was commented out and a warning string was
    inserted. You need to rewrite this code.
    */
    NVTE_CHECK_CUDA(DPCT_CHECK_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device_id))));
    cache[device_id] = prop.get_max_compute_units();
  }
  catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
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
    std::vector<std::pair<std::string, Path>> search_paths = {
      {"NVTE_CUDA_INCLUDE_DIR", ""},
      {"CUDA_HOME", ""},
      {"CUDA_DIR", ""},
      {"", string_path_cuda_include},
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
