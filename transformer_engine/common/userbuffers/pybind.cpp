/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>

#include "transformer_engine/transformer_engine.h"
#include "userbuffers.h"
#include "executor.h"

#include "../common.h"

namespace py = pybind11;

namespace transformer_engine {

namespace userbuffers {

PYBIND11_MODULE(transformer_engine_userbuffers, m) {
  py::enum_<UbufCommType>(m, "UbufCommType", py::module_local())
    .value("RS", UbufCommType::RS)
    .value("AG", UbufCommType::AG);

  py::enum_<UbufOverlapAlgo>(m, "UbufOverlapAlgo", py::module_local())
    .value("BULK_OVERLAP_RS", UbufOverlapAlgo::BULK_OVERLAP_RS)
    .value("BULK_OVERLAP_AG", UbufOverlapAlgo::BULK_OVERLAP_AG)
    .value("SPLIT_PIPELINED_RS", UbufOverlapAlgo::SPLIT_PIPELINED_RS)
    .value("SPLIT_PIPELINED_RS_P2P", UbufOverlapAlgo::SPLIT_PIPELINED_RS_P2P)
    .value("SPLIT_PIPELINED_AG_P2P", UbufOverlapAlgo::SPLIT_PIPELINED_AG_P2P)
    .value("ATOMIC_GEMM_RS", UbufOverlapAlgo::ATOMIC_GEMM_RS)
    .value("ATOMIC_GEMM_RS_P2P", UbufOverlapAlgo::ATOMIC_GEMM_AG_P2P)
    .value("ATOMIC_GEMM_AG_P2P", UbufOverlapAlgo::ATOMIC_GEMM_AG_P2P);

  py::enum_<DType>(m, "TEDType", py::module_local())
    .value("kByte", DType::kByte)
    .value("kInt32", DType::kInt32)
    .value("kInt64", DType::kInt64)
    .value("kFloat32", DType::kFloat32)
    .value("kFloat16", DType::kFloat16)
    .value("kBFloat16", DType::kBFloat16)
    .value("kFloat8E4M3", DType::kFloat8E4M3)
    .value("kFloat8E5M2", DType::kFloat8E5M2);

  py::class_<UbufExecutorBase>(m, "UbufExecutorBase", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool>())
    .def_readwrite("allgather", &UbufExecutorBase::_allgather)
    .def_readwrite("barrier", &UbufExecutorBase::_barrier);

  py::class_<UbufExecutor, UbufExecutorBase>(m, "UbufExecutor", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool>());

  py::class_<UbufExecutorP2P, UbufExecutorBase>(m, "UbufExecutorP2P", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool, bool>());
}  // PYBIND11_MODULE

}  // namespace userbuffers

}  // namespace transformer_engine
