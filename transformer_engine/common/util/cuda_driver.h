/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_

#include <string>

#include <cuda.h>

#include "../common.h"

namespace {

inline void check_cuda_driver_(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char *description;
    cuGetErrorString(status, &description);
    NVTE_ERROR("CUDA Error: " + std::string(description));
  }
}

}  // namespace

#define NVTE_CHECK_CUDA_DRIVER(ans) { check_cuda_driver_(ans); }

// TODO Indirectly access CUDA driver funcs in case libcuda.so
// isn't available

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_
