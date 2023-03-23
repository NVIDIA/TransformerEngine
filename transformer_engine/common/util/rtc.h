/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

namespace transformer_engine {

namespace rtc {

/* \brief Wrapper class for a runtime-compiled CUDA kernel */
class Kernel {
public:
  Kernel(std::string mangled_name, std::string compiled_code);
  ~Kernel();
  Kernel(const Kernel&) = delete;  // move-only
  Kernel(Kernel&&) noexcept;
  Kernel& operator=(Kernel) noexcept;
  friend void swap(Kernel& first, Kernel& second) noexcept;

  /* \brief Launch CUDA kernel
   *
   * Loads the kernel into the device the first time the device is
   * accessed.
   */
  void launch(int device_id,
              const dim3 grid_dim,
              const dim3 block_dim,
              unsigned int shared_mem_bytes,
              cudaStream_t stream,
              std::vector<void*> &args);

  /* /brief CUDA function for given CUDA device
   *
   * Loads the kernel into the device the first time the device is
   * accessed.
   */
  CUfunction get_function(int device_id);

private:
  // Mangled function name
  std::string mangled_name_;
  // Compiled assembly, either in PTX or cubin format
  std::string compiled_code_;
  // CUDA module for each CUDA device
  std::vector<CUmodule> modules_;
  // CUDA function for each CUDA device
  std::vector<CUfunction> functions_;

  // Uninitialized CUDA module
  static constexpr CUmodule null_module = static_cast<CUmodule>(nullptr);
  // Uninitialized CUDA function
  static constexpr CUfunction null_function = static_cast<CUfunction>(nullptr);
};

/* \brief Singleton class to manage runtime-compiled CUDA kernels */
class KernelManager {
public:
  /* \brief Access singleton */
  static KernelManager& instance();

  /* \brief Compile CUDA kernel */
  void compile(const std::string &kernel_name,
               const std::string &code,
               const std::string &filename,
               int device_id,
               const std::string &parameters);

  /* \brief Whether a CUDA kernel has already been compiled */
  bool is_compiled(const std::string &kernel_name,
                   int device_id,
                   const std::string &parameters) const;

  /* \brief Launch a CUDA kernel
   *
   * Assumes the kernel has already been compiled.
   */
  void launch(const std::string &kernel_name,
              int device_id,
              const std::string &parameters,
              const dim3 grid_dim,
              const dim3 block_dim,
              unsigned int shared_mem_bytes,
              cudaStream_t stream,
              std::vector<void*> &args);

private:
  /* /brief Compiled kernels */
  std::unordered_map<std::string, Kernel> kernel_cache_;
  /* /brief Mutex for thread-safe compilation */
  std::mutex lock_;

  KernelManager() = default;
  ~KernelManager() = default;
  KernelManager(const KernelManager&) = delete;
  KernelManager& operator=(const KernelManager&) = delete;

  /* /brief Identifying string for a compiled kernel */
  std::string get_kernel_cache_key(const std::string &kernel_name,
                                   int device_id,
                                   const std::string &parameters) const; /// TODO Consider variadic template

};

}  // namespace rtc

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_
