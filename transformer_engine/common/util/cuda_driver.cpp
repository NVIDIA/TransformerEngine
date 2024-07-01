/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dlfcn.h>

#include <filesystem>

#include "../common.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace {

/*! \brief Wrapper class for a shared library
 *
 * \todo Windows support
 */
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

/*! \brief Lazily-initialized shared library for CUDA driver */
Library &cuda_driver_lib() {
#if defined(_WIN32) || defined(_WIN64) || defined(__WINDOWS__)
  constexpr char lib_name[] = "nvcuda.dll";
#else
  constexpr char lib_name[] = "libcuda.so.1";
#endif
  static Library lib(lib_name);
  return lib;
}

}  // namespace

namespace cuda_driver {

void *get_symbol(const char *symbol) {
  void *entry_point;
  cudaDriverEntryPointQueryResult driver_result;
  NVTE_CHECK_CUDA(cudaGetDriverEntryPoint(symbol, &entry_point, cudaEnableDefault, &driver_result));
  NVTE_CHECK(driver_result == cudaDriverEntryPointSuccess,
             "Could not find CUDA driver entry point for ", symbol);
  return entry_point;
}

}  // namespace cuda_driver

}  // namespace transformer_engine
