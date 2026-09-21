/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_SHARED_LIB_WRAPPER_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_SHARED_LIB_WRAPPER_H_

#include <dlfcn.h>

#include <filesystem>

namespace transformer_engine {

/*! \brief Return the shared object containing an address. */
inline std::filesystem::path shared_library_path(const void *anchor) {
  Dl_info library_info{};
  if (anchor == nullptr || dladdr(anchor, &library_info) == 0 ||
      library_info.dli_fname == nullptr) {
    return {};
  }

  std::filesystem::path library_path = library_info.dli_fname;
  if (library_path.is_relative()) {
    std::error_code error;
    library_path = std::filesystem::absolute(library_path, error);
    if (error) {
      return {};
    }
  }
  return library_path;
}

/*! \brief Return the directory containing the shared object for an address. */
inline std::filesystem::path shared_library_directory(const void *anchor) {
  return shared_library_path(anchor).parent_path();
}

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
