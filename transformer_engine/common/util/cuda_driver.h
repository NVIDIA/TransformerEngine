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
#include "../util/string.h"

namespace transformer_engine {

namespace cuda_driver {

/*! \brief Get pointer corresponding to symbol in CUDA driver library */
void *get_symbol(const char *symbol);

/*! \brief Call function in CUDA driver library
 *
 * The CUDA driver library (libcuda.so.1 on Linux) may be different at
 * compile-time and run-time. In particular, the CUDA SDK provides
 * stubs for the driver library in case compilation is on a system
 * without GPUs. Indirect function calls into a lazily-initialized
 * library ensures we are accessing the correct version.
 */
template <typename... ArgTs>
inline CUresult call(const char *symbol, ArgTs... args) {
  using FuncT = CUresult(ArgTs...);
  FuncT *func = reinterpret_cast<FuncT*>(get_symbol(symbol));
  return (*func)(args...);
}

}  // namespace cuda_driver

}  // namespace transformer_engine

namespace {

/*! \brief Throw exception if CUDA driver call has failed */
inline void check_cuda_driver_(CUresult status) {
  if (status != CUDA_SUCCESS) {
    const char *description;
    transformer_engine::cuda_driver::call("cuGetErrorString", &description);
    NVTE_ERROR(transformer_engine::concat_strings("CUDA Error: ",description));
  }
}

/*! \brief Call CUDA driver function and throw exception if it fails */
template <typename... ArgTs>
inline void call_and_check_cuda_driver_(const char *symbol,
                                        ArgTs &&... args) {
  check_cuda_driver_(transformer_engine::cuda_driver::call(symbol,
                                                           std::forward<ArgTs>(args)...));
}

}  // namespace

#define NVTE_CHECK_CUDA_DRIVER(ans) { check_cuda_driver_(ans); }

#define NVTE_CALL_CHECK_CUDA_DRIVER(func, ...) \
  { call_and_check_cuda_driver_(#func, __VA_ARGS__); }

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_
