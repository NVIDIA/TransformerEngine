/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <filesystem>

#include "../common.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace cuda_driver {

typedef cudaError_t (*VersionedGetEntryPoint)(const char *, void **, unsigned int,
                                              unsigned long long,  // NOLINT(*)
                                              cudaDriverEntryPointQueryResult *);
typedef cudaError_t (*GetEntryPoint)(const char *, void **, unsigned long long,  // NOLINT(*)
                                     cudaDriverEntryPointQueryResult *);

void *get_symbol(const char *symbol, int cuda_version) {
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
}

void ensure_context_exists() {
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
}

}  // namespace cuda_driver

}  // namespace transformer_engine
