/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
#define TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_

#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "transformer_engine/logging.h"

namespace transformer_engine {
namespace jax {

template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");
  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
std::string PackDescriptorAsString(const T &descriptor) {
  return std::string(bit_cast<const char *>(&descriptor), sizeof(T));
}

template <typename T>
pybind11::bytes PackDescriptor(const T &descriptor) {
  return pybind11::bytes(PackDescriptorAsString(descriptor));
}

template <typename T>
const T *UnpackDescriptor(const char *opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T *>(opaque);
}

template <typename T>
pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

class cublasLtMetaManager {
 public:
  static cublasLtMetaManager &Instance() {
    static thread_local cublasLtMetaManager instance;
    return instance;
  }

  cublasLtMetaManager() {}
  ~cublasLtMetaManager() { Clear_(); }

  void *GetWorkspace(size_t size = 4194304) {
    ReallocateIfNeed_(size);
    return workspace_;
  }

 private:
  void *workspace_ = nullptr;
  size_t size_ = 0;

  void Clear_() {
    if (workspace_ != nullptr) {
      NVTE_CHECK_CUDA(cudaFree(workspace_));
    }
    workspace_ = nullptr;
    size_ = 0;
  }

  void Allocate_(size_t new_size) {
    NVTE_CHECK_CUDA(cudaMalloc(&workspace_, new_size));
    size_ = new_size;
  }

  void ReallocateIfNeed_(size_t new_size) {
    if (new_size > size_) {
      Clear_();
      Allocate_(new_size);
    }
  }
};

class cudaDevicePropertiesManager {
 public:
  static cudaDevicePropertiesManager &Instance() {
    static thread_local cudaDevicePropertiesManager instance;
    return instance;
  }

  int GetMultiProcessorCount() {
    if (!prop_queried_) {
      int device_id;
      NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
      cudaGetDeviceProperties(&prop_, device_id);
      prop_queried_ = true;
    }
    return prop_.multiProcessorCount;
  }

 private:
  bool prop_queried_ = false;
  cudaDeviceProp prop_;
};

}  // namespace jax
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_JAX_CSRC_UTILS_H_
