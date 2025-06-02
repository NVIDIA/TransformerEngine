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

void *get_symbol(const char *symbol, int cuda_version) {
  void *entry_point;
  cudaDriverEntryPointQueryResult driver_result;
#if CUDA_VERSION < 12050
  NVTE_CHECK_CUDA(cudaGetDriverEntryPoint(symbol, &entry_point, cudaEnableDefault, &driver_result));
#else
  NVTE_CHECK_CUDA(cudaGetDriverEntryPointByVersion(symbol, &entry_point, cuda_version,
                                                   cudaEnableDefault, &driver_result));
#endif
  NVTE_CHECK(driver_result == cudaDriverEntryPointSuccess,
             "Could not find CUDA driver entry point for ", symbol);
  return entry_point;
}

}  // namespace cuda_driver

}  // namespace transformer_engine
