/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common.h"

namespace transformer_engine {
namespace paddle_ext {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

PYBIND11_MODULE(transformer_engine_paddle, m) {
    // Misc
    m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");
    // Data structures
    py::enum_<DType>(m, "DType", py::module_local())
        .value("kByte", DType::kByte)
        .value("kInt32", DType::kInt32)
        .value("kFloat32", DType::kFloat32)
        .value("kFloat16", DType::kFloat16)
        .value("kBFloat16", DType::kBFloat16)
        .value("kFloat8E4M3", DType::kFloat8E4M3)
        .value("kFloat8E5M2", DType::kFloat8E5M2);
}
}  // namespace paddle_ext
}  // namespace transformer_engine
