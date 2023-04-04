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
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include "../util/cuda_driver.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace rtc {

/*! \brief Whether NVRTC support is enabled */
bool is_enabled();

/*! \brief Wrapper class for a runtime-compiled CUDA kernel */
class Kernel {
public:
  Kernel(std::string mangled_name, std::string compiled_code);
  ~Kernel();
  Kernel(const Kernel&) = delete;  // move-only
  Kernel(Kernel&&) noexcept;
  Kernel& operator=(Kernel) noexcept;
  friend void swap(Kernel& first, Kernel& second) noexcept;

  /*! \brief Launch CUDA kernel
   *
   * Loads the kernel into the device the first time the device is
   * accessed.
   */
  template <typename... ArgTs>
  void launch(int device_id,
              const dim3 grid_dim,
              const dim3 block_dim,
              unsigned int shared_mem_bytes,
              cudaStream_t stream,
              ArgTs&... args);

  /*! \brief CUDA function for given CUDA device
   *
   * Loads the kernel into the device the first time the device is
   * accessed.
   */
  CUfunction get_function(int device_id);

private:
  /*! \brief Mangled function name */
  std::string mangled_name_;
  /*! \brief  Compiled assembly, either in PTX or cubin format */
  std::string compiled_code_;
  /*! CUDA module for each CUDA device */
  std::vector<CUmodule> modules_;
  /*! CUDA function for each CUDA device */
  std::vector<CUfunction> functions_;

  /*! \brief Mutex for thread-safe kernel initialization */
  std::mutex lock_;

  /*! \brief Uninitialized CUDA module */
  static constexpr CUmodule null_module = static_cast<CUmodule>(nullptr);
  /*! Uninitialized CUDA function */
  static constexpr CUfunction null_function = static_cast<CUfunction>(nullptr);
};

/*! \brief Singleton class to manage runtime-compiled CUDA kernels */
class KernelManager {
public:
  /*! \brief Access singleton */
  static KernelManager& instance();

  /*! \brief Compile CUDA kernel for current CUDA device
   */
  void compile(const std::string &kernel_label,
               const std::string &kernel_name,
               const std::string &code,
               const std::string &filename);

  /*! \brief Whether CUDA kernel has been compiled for current CUDA
   * device.
   */
  bool is_compiled(const std::string &kernel_label) const;

  /*! \brief Launch CUDA kernel on current CUDA device
   *
   * Assumes the kernel has already been compiled.
   */
  template <typename... ArgTs>
  void launch(const std::string &kernel_label,
              const dim3 grid_dim,
              const dim3 block_dim,
              unsigned int shared_mem_bytes,
              cudaStream_t stream,
              ArgTs&... args);

private:
  /*! \brief Compiled kernels */
  std::unordered_map<std::string, Kernel> kernel_cache_;
  /*! \brief Mutex for thread-safe compilation */
  std::mutex lock_;

  KernelManager() = default;
  ~KernelManager() = default;
  KernelManager(const KernelManager&) = delete;
  KernelManager& operator=(const KernelManager&) = delete;

  /*! \brief Construct key for kernel cache */
  std::string get_kernel_cache_key(const std::string &kernel_label,
                                   int device_id) const;
};

}  // namespace rtc

}  // namespace transformer_engine

////////////////////////////////////////////////////////////////////////////////////////////////////
// Template implementations
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace transformer_engine {

namespace rtc {

template <typename... ArgTs>
void Kernel::launch(int device_id,
                    const dim3 grid_dim,
                    const dim3 block_dim,
                    unsigned int shared_mem_bytes,
                    cudaStream_t stream,
                    ArgTs&... args) {
  void* arg_ptrs[] = { const_cast<void*>(static_cast<const void*>(&args))... };
  NVTE_CHECK_CUDA_DRIVER(cuLaunchKernel(get_function(device_id),
                                        grid_dim.x,
                                        grid_dim.y,
                                        grid_dim.z,
                                        block_dim.x,
                                        block_dim.y,
                                        block_dim.z,
                                        shared_mem_bytes,
                                        static_cast<CUstream>(stream),
                                        arg_ptrs,
                                        nullptr));
}

template <typename... ArgTs>
void KernelManager::launch(const std::string &cache_key,
                           const dim3 grid_dim,
                           const dim3 block_dim,
                           unsigned int shared_mem_bytes,
                           cudaStream_t stream,
                           ArgTs&... args) {
  const int device_id = cuda::current_device();
  const auto key = get_kernel_cache_key(cache_key, device_id);
  NVTE_CHECK(kernel_cache_.count(key) > 0,
             "Attempted to launch RTC kernel before compilation");
  kernel_cache_.at(key).launch(device_id,
                               grid_dim,
                               block_dim,
                               shared_mem_bytes,
                               stream,
                               args...);

}

}  // namespace rtc

}  // namespace transformer_engine


#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_
