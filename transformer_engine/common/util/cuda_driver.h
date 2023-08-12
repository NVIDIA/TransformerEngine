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
  FuncT *func = reinterpret_cast<FuncT*>(get_symbol(symbol));
  return (*func)(args...);
}

}  // namespace cuda_driver

}  // namespace transformer_engine

#define NVTE_CHECK_CUDA_DRIVER(ans)                                            \
  do {                                                                         \
    if (status != CUDA_SUCCESS) {                                              \
      const char *description;                                                 \
      transformer_engine::cuda_driver::call("cuGetErrorString", status,        \
                                            &description);                     \
      NVTE_ERROR(                                                              \
          transformer_engine::concat_strings("CUDA Error: ", description));    \
    }                                                                          \
    while (false)

#define NVTE_CALL_CHECK_CUDA_DRIVER(func, ...)                                 \
  NVTE_CHECK_CUDA_DRIVER(                                                      \
      transformer_engine::cuda_driver::call(symbol, __VA_ARGS__))

#endif // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_DRIVER_H_
