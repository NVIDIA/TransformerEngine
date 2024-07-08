/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_

#include <cuda.h>

#include <string>

#include "../common.h"
#include "../util/string.h"

namespace transformer_engine {

namespace cuda_driver {

/*! \brief Get pointer corresponding to symbol in CUDA driver library */
void *get_symbol(const char *symbol);

/*! \brief Call function in CUDA driver library
 *
 * The CUDA driver library (libcuda.so.1 on Linux) may be different at
 * compile-time and run-time. In particular, the CUDA Toolkit provides
 * stubs for the driver library in case compilation is on a system
 * without GPUs. Indirect function calls into a lazily-initialized
 * library ensures we are accessing the correct version.
 *
 * \param[in] symbol Function name
 * \param[in] args   Function arguments
 */
template <typename... ArgTs>
inline CUresult call(const char *symbol, ArgTs... args) {
  using FuncT = CUresult(ArgTs...);
  FuncT *func = reinterpret_cast<FuncT *>(get_symbol(symbol));
  return (*func)(args...);
}

}  // namespace cuda_driver

}  // namespace transformer_engine

#define NVTE_CHECK_CUDA_DRIVER(expr)                                                             \
  do {                                                                                           \
    const CUresult status_NVTE_CHECK_CUDA_DRIVER = (expr);                                       \
    if (status_NVTE_CHECK_CUDA_DRIVER != CUDA_SUCCESS) {                                         \
      const char *desc_NVTE_CHECK_CUDA_DRIVER;                                                   \
      ::transformer_engine::cuda_driver::call("cuGetErrorString", status_NVTE_CHECK_CUDA_DRIVER, \
                                              &desc_NVTE_CHECK_CUDA_DRIVER);                     \
      NVTE_ERROR("CUDA Error: ", desc_NVTE_CHECK_CUDA_DRIVER);                                   \
    }                                                                                            \
  } while (false)

#define NVTE_CALL_CHECK_CUDA_DRIVER(symbol, ...)                                           \
  do {                                                                                     \
    NVTE_CHECK_CUDA_DRIVER(::transformer_engine::cuda_driver::call(#symbol, __VA_ARGS__)); \
  } while (false)

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_
