/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <dlpack/dlpack.h>

#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/fused_attn.h>

#include "../userbuffers/comm_gemm_overlap.h"

namespace transformer_engine {

namespace ub = userbuffers;

PYBIND11_MODULE(transformer_engine_common_cpp, m) {
  py::enum_<DType>(m, "DType", py::module_local())
    .value("kByte", DType::kByte)
    .value("kInt32", DType::kInt32)
    .value("kInt64", DType::kInt64)
    .value("kFloat8E4M3", DType::kFloat8E4M3)
    .value("kFloat8E5M2", DType::kFloat8E5M2)
    .value("kBFloat16", DType::kBFloat16)
    .value("kFloat16", DType::kFloat16)
    .value("kFloat32", DType::kFloat32);

  auto m_fa = m.def_submodule("fused_attn");

  py::enum_<NVTE_Bias_Type>(m_fa, "NVTE_Bias_Type", py::module_local())
      .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)
      .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)
      .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)
      .value("NVTE_ALIBI", NVTE_Bias_Type::NVTE_ALIBI);

  py::enum_<NVTE_Mask_Type>(m_fa, "NVTE_Mask_Type", py::module_local())
      .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)
      .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)
      .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK)
      .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK);

  py::enum_<NVTE_QKV_Layout>(m_fa, "NVTE_QKV_Layout", py::module_local())
      .value("NVTE_SB3HD", NVTE_QKV_Layout::NVTE_SB3HD)
      .value("NVTE_SBH3D", NVTE_QKV_Layout::NVTE_SBH3D)
      .value("NVTE_SBHD_SB2HD", NVTE_QKV_Layout::NVTE_SBHD_SB2HD)
      .value("NVTE_SBHD_SBH2D", NVTE_QKV_Layout::NVTE_SBHD_SBH2D)
      .value("NVTE_SBHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD)
      .value("NVTE_BS3HD", NVTE_QKV_Layout::NVTE_BS3HD)
      .value("NVTE_BSH3D", NVTE_QKV_Layout::NVTE_BSH3D)
      .value("NVTE_BSHD_BS2HD", NVTE_QKV_Layout::NVTE_BSHD_BS2HD)
      .value("NVTE_BSHD_BSH2D", NVTE_QKV_Layout::NVTE_BSHD_BSH2D)
      .value("NVTE_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD)
      .value("NVTE_T3HD", NVTE_QKV_Layout::NVTE_T3HD)
      .value("NVTE_TH3D", NVTE_QKV_Layout::NVTE_TH3D)
      .value("NVTE_THD_T2HD", NVTE_QKV_Layout::NVTE_THD_T2HD)
      .value("NVTE_THD_TH2D", NVTE_QKV_Layout::NVTE_THD_TH2D)
      .value("NVTE_THD_THD_THD", NVTE_QKV_Layout::NVTE_THD_THD_THD);

  py::enum_<NVTE_Fused_Attn_Backend>(m_fa, "NVTE_Fused_Attn_Backend", py::module_local())
      .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)
      .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)
      .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8)
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend);

  auto m_ub = m.def_submodule("userbuffers");

  m_ub.attr("_NUM_MAX_UB_STREAMS") = py::int_(ub::_NUM_MAX_UB_STREAMS);

  py::enum_<ub::CommGemmOverlapType>(m_ub, "CommGemmOverlapType", py::module_local())
    .value("RS", ub::CommGemmOverlapType::RS)
    .value("AG", ub::CommGemmOverlapType::AG);

  py::enum_<ub::CommGemmOverlapAlgo>(m_ub, "CommGemmOverlapAlgo", py::module_local())
    .value("BULK_OVERLAP_RS", ub::CommGemmOverlapAlgo::BULK_OVERLAP_RS)
    .value("BULK_OVERLAP_AG", ub::CommGemmOverlapAlgo::BULK_OVERLAP_AG)
    .value("SPLIT_PIPELINED_RS", ub::CommGemmOverlapAlgo::SPLIT_PIPELINED_RS)
    .value("SPLIT_PIPELINED_RS_P2P", ub::CommGemmOverlapAlgo::SPLIT_PIPELINED_RS_P2P)
    .value("SPLIT_PIPELINED_AG_P2P", ub::CommGemmOverlapAlgo::SPLIT_PIPELINED_AG_P2P)
    .value("ATOMIC_GEMM_RS", ub::CommGemmOverlapAlgo::ATOMIC_GEMM_RS)
    .value("ATOMIC_GEMM_RS_P2P", ub::CommGemmOverlapAlgo::ATOMIC_GEMM_AG_P2P)
    .value("ATOMIC_GEMM_AG_P2P", ub::CommGemmOverlapAlgo::ATOMIC_GEMM_AG_P2P);

  py::class_<ub::CommGemmOverlapBase>(m_ub, "CommGemmOverlapBase", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool>())
    .def("register_gpu_buffer",
      static_cast<void (ub::CommGemmOverlapBase::*)(py::capsule&, bool)>(
        &ub::CommGemmOverlapBase::register_gpu_buffer))
    .def("set_collective_callbacks", &ub::CommGemmOverlapBase::set_collective_callbacks)
    .def("is_atomic_gemm", &ub::CommGemmOverlapBase::is_atomic_gemm)
    .def("is_p2p_overlap", &ub::CommGemmOverlapBase::is_p2p_overlap);

  py::class_<ub::CommGemmOverlap,
             ub::CommGemmOverlapBase>(m_ub, "CommGemmOverlap", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool>());

  py::class_<ub::CommGemmOverlapP2P,
             ub::CommGemmOverlapBase>(m_ub, "CommGemmOverlapP2P", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool, bool, bool>());
}  // PYBIND11_MODULE

}  // namespace transformer_engine
