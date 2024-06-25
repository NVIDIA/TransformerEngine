/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util/pybind_helper.h"

namespace transformer_engine {

PYBIND11_MODULE(transformer_engine_common, m) {
  NVTE_ADD_COMMON_PYBIND11_BINDINGS(m);
}  // PYBIND11_MODULE

}  // namespace transformer_engine
