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
    py::class_<FP8TensorMeta>(m, "FP8TensorMeta")
        .def(py::init<>())
        .def_readwrite("scale", &FP8TensorMeta::scale)
        .def_readwrite("scale_inv", &FP8TensorMeta::scale_inv)
        .def_readwrite("amax_history", &FP8TensorMeta::amax_history);

    py::enum_<DType>(m, "DType", py::module_local())
        .value("kByte", DType::kByte)
        .value("kInt32", DType::kInt32)
        .value("kFloat32", DType::kFloat32)
        .value("kFloat16", DType::kFloat16)
        .value("kBFloat16", DType::kBFloat16)
        .value("kFloat8E4M3", DType::kFloat8E4M3)
        .value("kFloat8E5M2", DType::kFloat8E5M2);

    py::enum_<FP8FwdTensors>(m, "FP8FwdTensors")
        .value("GEMM1_INPUT", FP8FwdTensors::GEMM1_INPUT)
        .value("GEMM1_WEIGHT", FP8FwdTensors::GEMM1_WEIGHT)
        .value("GEMM1_OUTPUT", FP8FwdTensors::GEMM1_OUTPUT)
        .value("GEMM2_INPUT", FP8FwdTensors::GEMM2_INPUT)
        .value("GEMM2_WEIGHT", FP8FwdTensors::GEMM2_WEIGHT)
        .value("GEMM2_OUTPUT", FP8FwdTensors::GEMM2_OUTPUT);

    py::enum_<FP8BwdTensors>(m, "FP8BwdTensors")
        .value("GRAD_OUTPUT1", FP8BwdTensors::GRAD_OUTPUT1)
        .value("GRAD_INPUT1", FP8BwdTensors::GRAD_INPUT1)
        .value("GRAD_OUTPUT2", FP8BwdTensors::GRAD_OUTPUT2)
        .value("GRAD_INPUT2", FP8BwdTensors::GRAD_INPUT2);
}
}  // namespace paddle_ext
}  // namespace transformer_engine
