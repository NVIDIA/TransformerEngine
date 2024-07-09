/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dlfcn.h>

#include <filesystem>
#include <string_view>
#include <unordered_map

#include "../common.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {

namespace cuda_driver {

void *get_symbol(const char *symbol) {
  static thread_local std::unordered_list<std::string_view, void*> cache;
  const std::string_view key(symbol);
  if (cache.count(symbol) == 0) {
    void *entry_point;
    cudaDriverEntryPointQueryResult driver_result;
    NVTE_CHECK_CUDA(cudaGetDriverEntryPoint(symbol, &entry_point, cudaEnableDefault, &driver_result));
    NVTE_CHECK(driver_result == cudaDriverEntryPointSuccess,
               "Could not find CUDA driver entry point for ", symbol);
    cache[key] = entry_point;
  }
  return cache.at(key);
}

}  // namespace cuda_driver

}  // namespace transformer_engine
