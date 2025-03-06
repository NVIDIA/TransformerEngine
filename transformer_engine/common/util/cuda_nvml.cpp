/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "cuda_nvml.h"

#include "shared_lib_wrapper.h"

namespace transformer_engine {

namespace cuda_nvml {

/*! \brief Lazily-initialized shared library for CUDA NVML */
Library &cuda_nvml_lib() {
  constexpr char lib_name[] = "libnvidia-ml.so.1";
  static Library lib(lib_name);
  return lib;
}

void *get_symbol(const char *symbol) { return cuda_nvml_lib().get_symbol(symbol); }

}  // namespace cuda_nvml

}  // namespace transformer_engine
