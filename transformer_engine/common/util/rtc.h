/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../common.h"
#include "../util/cuda_driver.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace rtc {

/*! \brief Whether NVRTC support is enabled
 *
 * NVRTC support can be disabled by setting NVTE_DISABLE_NVRTC=1 in
 * the environment.
 */
bool is_enabled();

/*! \brief Wrapper class for a runtime-compiled CUDA kernel */
class Kernel {
 public:
  Kernel(std::string mangled_name, std::string compiled_code);
  ~Kernel();
  Kernel(const Kernel &) = delete;  // move-only
  Kernel(Kernel &&) noexcept;
  Kernel &operator=(Kernel) noexcept;
  friend void swap(Kernel &first, Kernel &second) noexcept;

  /*! \brief Launch CUDA kernel
   *
   * Loads the kernel into the device the first time the device is
   * accessed.
   *
   * \param[in] device_id        CUDA device
   * \param[in] grid_dim         Grid dimensions in blocks
   * \param[in] block_dim        Thread block dimensions
   * \param[in] shared_mem_bytes Dynamic shared-memory size per thread block in
   *                             bytes
   * \param[in] stream           CUDA stream
   * \param[in] args             Kernel arguments
   */
  template <typename... ArgTs>
  void launch(int device_id, const dim3 grid_dim, const dim3 block_dim,
              unsigned int shared_mem_bytes, cudaStream_t stream, ArgTs &&...args) {
    void *arg_ptrs[] = {const_cast<void *>(static_cast<const void *>(&args))...};
    NVTE_CALL_CHECK_CUDA_DRIVER(cuLaunchKernel, get_function(device_id), grid_dim.x, grid_dim.y,
                                grid_dim.z, block_dim.x, block_dim.y, block_dim.z, shared_mem_bytes,
                                static_cast<CUstream>(stream), arg_ptrs, nullptr);
  }

  /*! \brief CUDA function for given CUDA device
   *
   * Loads the kernel into the device the first time the device is
   * accessed.
   */
  CUfunction get_function(int device_id);

  /*! \brief Sets the preferred cache configuration for a function
   *
   * Wrapper of the CUDA Driver API function "cuFuncSetCacheConfig"
   */
  void set_function_cache_config(int device_id, CUfunc_cache cache_config);

 private:
  /*! \brief Mangled function name */
  std::string mangled_name_;
  /*! \brief  Compiled assembly, either in PTX or cubin format */
  std::string compiled_code_;
  /*! CUDA module for each CUDA device */
  std::vector<CUmodule> modules_;
  /*! CUDA function for each CUDA device */
  std::vector<CUfunction> functions_;

  /*! Flags for thread-safe kernel initialization */
  std::unique_ptr<std::vector<std::once_flag>> init_flags_;

  /*! \brief Uninitialized CUDA module */
  static constexpr CUmodule null_module = static_cast<CUmodule>(nullptr);
  /*! Uninitialized CUDA function */
  static constexpr CUfunction null_function = static_cast<CUfunction>(nullptr);
};

/*! \brief Singleton class to manage runtime-compiled CUDA kernels */
class KernelManager {
 public:
  /*! \brief Get singleton instance */
  static KernelManager &instance();

  /*! \brief Compile CUDA kernel for current CUDA device
   *
   * The compiled kernel is cached and made available for launching.
   *
   * \param[in] kernel_label Unique identifying string for kernel
   * \param[in] kernel_name  Kernel name within source code
   * \param[in] code         Kernel source code
   * \param[in] filename     Path to associate with source code,
   *                         primarily for debugging
   */
  void compile(const std::string &kernel_label, const std::string &kernel_name,
               const std::string &code, const std::string &filename);

  /*! \brief Whether CUDA kernel has been compiled for CUDA device
   *
   * \param[in] kernel_label Unique identifying string for kernel
   * \param[in] device_id    CUDA device (default is current device)

   * \return Whether kernel has been compiled
   */
  bool is_compiled(const std::string &kernel_label, int device_id = -1) const;

  /*! \brief Launch CUDA kernel on current CUDA device
   *
   * Assumes the kernel has already been compiled.
   *
   * \param[in] kernel_label     Unique identifying string for kernel
   * \param[in] grid_dim         Grid dimensions in blocks
   * \param[in] block_dim        Thread block dimensions
   * \param[in] shared_mem_bytes Dynamic shared-memory size per thread block in
   *                             bytes
   * \param[in] stream           CUDA stream
   * \param[in] args             Kernel arguments
   */
  template <typename... ArgTs>
  void launch(const std::string &kernel_label, const dim3 grid_dim, const dim3 block_dim,
              unsigned int shared_mem_bytes, cudaStream_t stream, ArgTs &&...args) {
    const int device_id = cuda::current_device();
    const auto key = get_kernel_cache_key(kernel_label, device_id);
    NVTE_CHECK(kernel_cache_.count(key) > 0, "Attempted to launch RTC kernel before compilation");
    kernel_cache_.at(key).launch(device_id, grid_dim, block_dim, shared_mem_bytes, stream,
                                 std::forward<ArgTs>(args)...);
  }

  /*! \brief Sets the preferred cache configuration for a function in the context
   *
   * Assumes the kernel has already been compiled.
   *
   * \param[in] kernel_label     Unique identifying string for kernel
   * \param[in] cache_config     Prefered cache configuration
   */
  void set_cache_config(const std::string &kernel_label, CUfunc_cache cache_config);

 private:
  /*! \brief Compiled kernels */
  std::unordered_map<std::string, Kernel> kernel_cache_;
  /*! \brief Mutex for thread-safe compilation */
  std::mutex lock_;

  KernelManager() = default;
  ~KernelManager() = default;
  KernelManager(const KernelManager &) = delete;
  KernelManager &operator=(const KernelManager &) = delete;

  /*! \brief Construct key for kernel cache
   *
   * \param[in] kernel_label     Unique identifying string for kernel
   * \param[in] device_id    CUDA device (default is current device)
   *
   * \return Key for kernel cache
   */
  std::string get_kernel_cache_key(const std::string &kernel_label, int device_id) const;
};

}  // namespace rtc

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_RTC_H_
