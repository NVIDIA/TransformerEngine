/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dlfcn.h>
#if !(defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__))
#include <execinfo.h>
#include <unistd.h>
#endif
#include <cstdio>
#include <filesystem>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "transformer_engine/musify.h"

namespace transformer_engine {

namespace cuda_driver {

class Library {
 public:
  explicit Library(const char *filename) {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
    // TODO Windows support
    NVTE_ERROR("Shared library initialization is not supported with Windows");
#else
    handle_ = dlopen(filename, RTLD_LAZY | RTLD_LOCAL);
    NVTE_CHECK(handle_ != nullptr, "Lazy library initialization failed");
#endif  // _WIN32 or _WIN64 or __WINDOW__
  }

  ~Library() {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
    // TODO Windows support
#else
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
#endif  // _WIN32 or _WIN64 or __WINDOW__
  }

  Library(const Library &) = delete;  // move-only

  Library(Library &&other) noexcept { swap(*this, other); }

  Library &operator=(Library other) noexcept {
    // Copy-and-swap idiom
    swap(*this, other);
    return *this;
  }

  friend void swap(Library &first, Library &second) noexcept;

  void *get() noexcept { return handle_; }

  const void *get() const noexcept { return handle_; }

  /*! \brief Get pointer corresponding to symbol in shared library */
  void *get_symbol(const char *symbol) {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
    // TODO Windows support
    NVTE_ERROR("Shared library initialization is not supported with Windows");
#else
    void *ptr = dlsym(handle_, symbol);
    if (ptr == nullptr) {
      std::fprintf(stderr, "Could not find symbol: %s\n", symbol);
      void *callstack[64];
      const int frames = backtrace(callstack, 64);
      backtrace_symbols_fd(callstack, frames, STDERR_FILENO);
      std::fflush(stderr);
    }
    NVTE_CHECK(ptr != nullptr, "Could not find symbol in lazily-initialized library");
    return ptr;
#endif  // _WIN32 or _WIN64 or __WINDOW__
  }

 private:
  void *handle_ = nullptr;
};

void swap(Library &first, Library &second) noexcept {
  using std::swap;
  swap(first.handle_, second.handle_);
}

Library &musa_driver_lib() {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  constexpr char lib_name[] = "nvcuda.dll";
#else
  constexpr char lib_name[] = "libmusa.so";
#endif
  static Library lib(lib_name);
  return lib;
}

typedef cudaError_t (*VersionedGetEntryPoint)(const char *, void **, unsigned int,
                                              unsigned long long,  // NOLINT(*)
                                              cudaDriverEntryPointQueryResult *);
typedef cudaError_t (*GetEntryPoint)(const char *, void **, unsigned long long,  // NOLINT(*)
                                     cudaDriverEntryPointQueryResult *);

void *get_symbol(const char *symbol, int cuda_version) {
#ifndef NVTE_SKIP_MUSA_UNCOMPATIBLE
  constexpr char driver_entrypoint[] = "cudaGetDriverEntryPoint";
  constexpr char driver_entrypoint_versioned[] = "cudaGetDriverEntryPointByVersion";
  // We link to the libcudart.so already, so can search for it in the current context
  static GetEntryPoint driver_entrypoint_fun =
      reinterpret_cast<GetEntryPoint>(dlsym(RTLD_DEFAULT, driver_entrypoint));
  static VersionedGetEntryPoint driver_entrypoint_versioned_fun =
      reinterpret_cast<VersionedGetEntryPoint>(dlsym(RTLD_DEFAULT, driver_entrypoint_versioned));

  cudaDriverEntryPointQueryResult driver_result;
  void *entry_point = nullptr;
  if (driver_entrypoint_versioned_fun != nullptr) {
    // Found versioned entrypoint function
    NVTE_CHECK_CUDA(driver_entrypoint_versioned_fun(symbol, &entry_point, cuda_version,
                                                    cudaEnableDefault, &driver_result));
  } else {
    NVTE_CHECK(driver_entrypoint_fun != nullptr, "Error finding the CUDA Runtime-Driver interop.");
    // Versioned entrypoint function not found
    NVTE_CHECK_CUDA(driver_entrypoint_fun(symbol, &entry_point, cudaEnableDefault, &driver_result));
  }
  NVTE_CHECK(driver_result == cudaDriverEntryPointSuccess,
             "Could not find CUDA driver entry point for ", symbol);
  return entry_point;
#else
  return musa_driver_lib().get_symbol(symbol);
#endif
}

void ensure_context_exists() {
  static thread_local bool need_check = []() {
    CUcontext context;
    NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxGetCurrent, &context);
    if (context == nullptr) {
      // Add primary context to context stack
      CUdevice device;
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDeviceGet, &device, cuda::current_device());
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxRetain, &context, device);
      NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxSetCurrent, context);
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxRelease, device);
    }
    return false;
  }();
}

}  // namespace cuda_driver

}  // namespace transformer_engine
