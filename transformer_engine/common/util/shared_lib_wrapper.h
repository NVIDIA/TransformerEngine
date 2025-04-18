/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_SHARED_LIB_WRAPPER_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_SHARED_LIB_WRAPPER_H_

#include <dlfcn.h>

namespace transformer_engine {

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

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_SHARED_LIB_WRAPPER_H_
