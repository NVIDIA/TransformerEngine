/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_NVML_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_NVML_H_

#include <nvml.h>

#include <string>

#include "../common.h"
#include "../util/string.h"

namespace transformer_engine {

namespace cuda_nvml {

/*! \brief Get pointer corresponding to symbol in CUDA NVML library */
void *get_symbol(const char *symbol);

/*! \brief Call function in CUDA NVML library
 *
 * The CUDA NVML library (libnvidia-ml.so.1 on Linux) may be different at
 * compile-time and run-time.
 *
 * \param[in] symbol Function name
 * \param[in] args   Function arguments
 */
template <typename... ArgTs>
inline nvmlReturn_t call(const char *symbol, ArgTs... args) {
  using FuncT = nvmlReturn_t(ArgTs...);
  FuncT *func = reinterpret_cast<FuncT *>(get_symbol(symbol));
  return (*func)(args...);
}

/*! \brief Get NVML error string
 *
 * \param[in] rc NVML return code
 */
inline const char *get_nvml_error_string(nvmlReturn_t rc) {
  using FuncT = const char *(nvmlReturn_t);
  FuncT *func = reinterpret_cast<FuncT *>(get_symbol("nvmlErrorString"));
  return (*func)(rc);
}

}  // namespace cuda_nvml

}  // namespace transformer_engine

#define NVTE_CHECK_CUDA_NVML(expr)                                                             \
  do {                                                                                         \
    const nvmlReturn_t status_NVTE_CHECK_CUDA_NVML = (expr);                                   \
    if (status_NVTE_CHECK_CUDA_NVML != NVML_SUCCESS) {                                         \
      const char *desc_NVTE_CHECK_CUDA_NVML =                                                  \
          ::transformer_engine::cuda_nvml::get_nvml_error_string(status_NVTE_CHECK_CUDA_NVML); \
      NVTE_ERROR("NVML Error: ", desc_NVTE_CHECK_CUDA_NVML);                                   \
    }                                                                                          \
  } while (false)

#define VA_ARGS(...) , ##__VA_ARGS__
#define NVTE_CALL_CHECK_CUDA_NVML(symbol, ...)                                                 \
  do {                                                                                         \
    NVTE_CHECK_CUDA_NVML(::transformer_engine::cuda_nvml::call(#symbol VA_ARGS(__VA_ARGS__))); \
  } while (false)

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_NVML_H_
