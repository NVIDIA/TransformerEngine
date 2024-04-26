/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NVTX_H_
#define TRANSFORMER_ENGINE_COMMON_NVTX_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <string>

namespace transformer_engine::nvtx {

struct NVTXWrapper {
  explicit NVTXWrapper(const std::string &name) {
    /*
    DPCT1007:217: Migration of nvtxRangePushA is not supported.
    */
    nvtxRangePush(name.c_str());
  }

  ~NVTXWrapper() {
    /*
    DPCT1007:218: Migration of nvtxRangePop is not supported.
    */
    nvtxRangePop();
  }
};

}  // namespace transformer_engine::nvtx

#endif  // TRANSFORMER_ENGINE_COMMON_NVTX_H_
