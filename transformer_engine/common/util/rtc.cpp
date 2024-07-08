/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../util/rtc.h"

#include <cstdlib>
#include <iostream>
#include <utility>

#include "../common.h"
#include "../util/cuda_driver.h"
#include "../util/string.h"
#include "../util/system.h"

namespace transformer_engine {

namespace rtc {

namespace {

// Strings with headers for RTC kernels
#include "string_code_util_math_h.h"
#include "string_code_utils_cuh.h"

/*! \brief Latest compute capability that NVRTC supports
 *
 * \return Compute capability as int. Last digit is minor revision,
 *         remaining digits are major revision.
 */
inline int max_supported_sm_arch() {
  static int arch_ = -1;
  if (arch_ < 0) {
    int num_archs = 0;
    NVTE_CHECK_NVRTC(nvrtcGetNumSupportedArchs(&num_archs));
    NVTE_CHECK(num_archs > 0, "Could not determine SM archs that NVRTC supports");
    std::vector<int> archs(num_archs);
    NVTE_CHECK_NVRTC(nvrtcGetSupportedArchs(archs.data()));
    arch_ = archs.back();
  }
  return arch_;
}

}  // namespace

bool is_enabled() {
  static bool is_enabled_ = false;
  static bool need_to_check_env = true;
  if (need_to_check_env) {
    is_enabled_ = !getenv<bool>("NVTE_DISABLE_NVRTC");
    need_to_check_env = false;
  }
  return is_enabled_;
}

Kernel::Kernel(std::string mangled_name, std::string compiled_code)
    : mangled_name_{std::move(mangled_name)},
      compiled_code_{std::move(compiled_code)},
      modules_(cuda::num_devices(), null_module),
      functions_(cuda::num_devices(), null_function),
      init_flags_{std::make_unique<std::vector<std::once_flag>>(cuda::num_devices())} {}

Kernel::~Kernel() {
  for (int device_id = 0; device_id < static_cast<int>(modules_.size()); ++device_id) {
    // Unload CUDA modules if needed
    if (modules_[device_id] != null_module) {
      CUdevice device;
      CUcontext context;
      if (cuda_driver::call("cuDeviceGet", &device, device_id) != CUDA_SUCCESS) {
        continue;
      }
      if (cuda_driver::call("cuDevicePrimaryCtxRetain", &context, device) != CUDA_SUCCESS) {
        continue;
      }
      if (cuda_driver::call("cuCtxSetCurrent", context) != CUDA_SUCCESS) {
        continue;
      }
      cuda_driver::call("cuModuleUnload", modules_[device_id]);
      cuda_driver::call("cuDevicePrimaryCtxRelease", device);
    }
  }
}

Kernel::Kernel(Kernel&& other) noexcept { swap(*this, other); }

Kernel& Kernel::operator=(Kernel other) noexcept {
  // Copy-and-swap idiom
  swap(*this, other);
  return *this;
}

void swap(Kernel& first, Kernel& second) noexcept {
  using std::swap;
  swap(first.mangled_name_, second.mangled_name_);
  swap(first.compiled_code_, second.compiled_code_);
  swap(first.modules_, second.modules_);
  swap(first.functions_, second.functions_);
  swap(first.init_flags_, second.init_flags_);
}

CUfunction Kernel::get_function(int device_id) {
  // Load kernel on device if needed
  auto load_on_device = [&]() {
    // Set driver context to proper device
    CUdevice device;
    CUcontext context;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &device, device_id);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxRetain, &context, device);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxSetCurrent, context);

    // Load function into driver context
    NVTE_CALL_CHECK_CUDA_DRIVER(cuModuleLoadDataEx, &modules_[device_id], compiled_code_.c_str(),
                                0,         // numOptions
                                nullptr,   // options
                                nullptr);  // optionValues
    NVTE_CALL_CHECK_CUDA_DRIVER(cuModuleGetFunction, &functions_[device_id], modules_[device_id],
                                mangled_name_.c_str());

    // Reset driver context
    NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxRelease, device);
  };
  std::call_once(init_flags_->at(device_id), load_on_device);

  // Return CUDA function
  return functions_[device_id];
}

void Kernel::set_function_cache_config(int device_id, CUfunc_cache cache_config) {
  NVTE_CALL_CHECK_CUDA_DRIVER(cuFuncSetCacheConfig, get_function(device_id), cache_config);
}

KernelManager& KernelManager::instance() {
  NVTE_CHECK(is_enabled(), "NVRTC support is not enabled");
  static KernelManager instance_;
  return instance_;
}

void KernelManager::compile(const std::string& kernel_label, const std::string& kernel_name,
                            const std::string& code, const std::string& filename) {
  std::lock_guard<std::mutex> lock_guard_(lock_);

  // Choose whether to compile to PTX or cubin
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);
  const int compile_sm_arch = std::min(sm_arch_, max_supported_sm_arch());
  const bool compile_ptx = (CUDA_VERSION <= 11000) || (sm_arch_ != compile_sm_arch);

  // Compilation flags
  std::vector<std::string> opts = {
#if NDEBUG == 0
      "-G",
#endif
      "--std=c++17"};
  if (compile_ptx) {
    opts.push_back(concat_strings("--gpu-architecture=compute_", compile_sm_arch));
  } else {
    opts.push_back(concat_strings("--gpu-architecture=sm_", compile_sm_arch));
  }
  opts.push_back(concat_strings("-I", cuda::include_directory(true)));
  std::vector<const char*> opts_ptrs;
  for (const auto& opt : opts) {
    opts_ptrs.push_back(opt.c_str());
  }

  // Compile source
  nvrtcProgram program;
  constexpr int num_headers = 2;
  constexpr const char* headers[num_headers] = {string_code_utils_cuh, string_code_util_math_h};
  constexpr const char* include_names[num_headers] = {"utils.cuh", "util/math.h"};
  NVTE_CHECK_NVRTC(nvrtcCreateProgram(&program, code.c_str(), filename.c_str(), num_headers,
                                      headers, include_names));
  NVTE_CHECK_NVRTC(nvrtcAddNameExpression(program, kernel_name.c_str()));
  const nvrtcResult compile_result =
      nvrtcCompileProgram(program, opts_ptrs.size(), opts_ptrs.data());
  if (compile_result != NVRTC_SUCCESS) {
    // Display log if compilation failed
    std::string log = concat_strings("NVRTC compilation log for ", filename, ":\n");
    const size_t log_offset = log.size();
    size_t log_size;
    NVTE_CHECK_NVRTC(nvrtcGetProgramLogSize(program, &log_size));
    log.resize(log_offset + log_size);
    NVTE_CHECK_NVRTC(nvrtcGetProgramLog(program, &log[log_offset]));
    log.back() = '\n';
    std::cerr << log;
    NVTE_CHECK_NVRTC(compile_result);
  }

  // Get mangled function name
  const char* mangled_name;
  NVTE_CHECK_NVRTC(nvrtcGetLoweredName(program, kernel_name.c_str(), &mangled_name));

  // Get compiled code
  std::string compiled_code;
  if (compile_ptx) {
    size_t compiled_size;
    NVTE_CHECK_NVRTC(nvrtcGetPTXSize(program, &compiled_size));
    compiled_code.resize(compiled_size);
    NVTE_CHECK_NVRTC(nvrtcGetPTX(program, compiled_code.data()));
  } else {
    size_t compiled_size;
    NVTE_CHECK_NVRTC(nvrtcGetCUBINSize(program, &compiled_size));
    compiled_code.resize(compiled_size);
    NVTE_CHECK_NVRTC(nvrtcGetCUBIN(program, compiled_code.data()));
  }

  // Cache compiled code
  const auto key = get_kernel_cache_key(kernel_label, device_id);
  kernel_cache_.insert({key, Kernel(mangled_name, std::move(compiled_code))});
  kernel_cache_.at(key).get_function(device_id);  // Make sure kernel is available on device

  // Clean up
  NVTE_CHECK_NVRTC(nvrtcDestroyProgram(&program));
}

void KernelManager::set_cache_config(const std::string& kernel_label, CUfunc_cache cache_config) {
  const int device_id = cuda::current_device();
  const auto key = get_kernel_cache_key(kernel_label, device_id);
  NVTE_CHECK(kernel_cache_.count(key) > 0, "Attempted to configure RTC kernel before compilation");
  kernel_cache_.at(key).set_function_cache_config(device_id, cache_config);
}

bool KernelManager::is_compiled(const std::string& kernel_label, int device_id) const {
  const auto key = get_kernel_cache_key(kernel_label, device_id);
  return kernel_cache_.count(key) > 0;
}

std::string KernelManager::get_kernel_cache_key(const std::string& kernel_label,
                                                int device_id) const {
  return concat_strings("sm=", cuda::sm_arch(device_id), ",", kernel_label);
}

}  // namespace rtc

}  // namespace transformer_engine
