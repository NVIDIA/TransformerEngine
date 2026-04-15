/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NVTX_H_
#define TRANSFORMER_ENGINE_COMMON_NVTX_H_

#ifndef NVTE_SKIP_MUSA_UNCOMPATIBLE
#include <nvtx3/nvToolsExt.h>
#endif

#include <string>

namespace transformer_engine::nvtx {

struct NVTXWrapper {
  explicit NVTXWrapper(const std::string &name) {
#ifndef NVTE_SKIP_MUSA_UNCOMPATIBLE
    nvtxRangePush(name.c_str());
#endif
  }

  ~NVTXWrapper() {
#ifndef NVTE_SKIP_MUSA_UNCOMPATIBLE
    nvtxRangePop();
#endif
  }
};

}  // namespace transformer_engine::nvtx

#endif  // TRANSFORMER_ENGINE_COMMON_NVTX_H_
