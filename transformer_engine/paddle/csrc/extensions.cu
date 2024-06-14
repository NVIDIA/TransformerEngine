/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

#include "common/util/pybind_helper.h"

namespace transformer_engine {
namespace paddle_ext {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

PYBIND11_MODULE(transformer_engine_paddle, m) {
  // Load nvte = py::module_::import("transformer_engine_common") into TE/Paddle. This makes
  // essential NVTE enums available through `import transformer_engine_paddle` without requiring
  // an additional `import transformer_engine_common as tex`.
  NVTE_ADD_COMMON_PYBIND11_BINDINGS(m)

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");
  m.def("get_fused_attn_backend", &get_fused_attn_backend, "Get Fused Attention backend");
  m.def("get_nvte_qkv_layout", &get_nvte_qkv_layout, "Get qkv layout enum by the string");
}
}  // namespace paddle_ext
}  // namespace transformer_engine
