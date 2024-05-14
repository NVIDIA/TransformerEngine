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
#include <transformer_engine/activation.h>

#include "userbuffers/comm_gemm_overlap.h"

namespace transformer_engine {

namespace ub = userbuffers;

PYBIND11_MODULE(transformer_engine_pybind, m) {
  py::enum_<DType>(m, "DType", py::module_local())
    .value("kByte", DType::kByte)
    .value("kInt32", DType::kInt32)
    .value("kInt64", DType::kInt64)
    .value("kFloat8E4M3", DType::kFloat8E4M3)
    .value("kFloat8E5M2", DType::kFloat8E5M2)
    .value("kBFloat16", DType::kBFloat16)
    .value("kFloat16", DType::kFloat16)
    .value("kFloat32", DType::kFloat32);

  pybind11::enum_<NVTE_Activation_Type>(m, "NVTE_Activation_Type", pybind11::module_local())
        .value("GELU", NVTE_Activation_Type::GELU)
        .value("GEGLU", NVTE_Activation_Type::GEGLU)
        .value("SILU", NVTE_Activation_Type::SILU)
        .value("SWIGLU", NVTE_Activation_Type::SWIGLU)
        .value("RELU", NVTE_Activation_Type::RELU)
        .value("REGLU", NVTE_Activation_Type::REGLU)
        .value("QGELU", NVTE_Activation_Type::QGELU)
        .value("QGEGLU", NVTE_Activation_Type::QGEGLU)
        .value("SRELU", NVTE_Activation_Type::SRELU)
        .value("SREGLU", NVTE_Activation_Type::SREGLU);

  py::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type", py::module_local())
      .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)
      .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)
      .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)
      .value("NVTE_ALIBI", NVTE_Bias_Type::NVTE_ALIBI);

  py::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type", py::module_local())
      .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)
      .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)
      .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK)
      .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK);

  py::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout", py::module_local())
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

  py::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", py::module_local())
      .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)
      .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)
      .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8)
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend);

  m.attr("NVTE_MAX_USERBUFFER_STREAMS") = py::int_(NVTE_MAX_USERBUFFER_STREAMS);

  py::enum_<NVTE_Comm_Overlap_Type>(m, "NVTE_Comm_Overlap_Type", py::module_local())
    .value("RS", NVTE_Comm_Overlap_Type::REDUCE_SCATTER)
    .value("AG", NVTE_Comm_Overlap_Type::ALL_GATHER);

  py::enum_<NVTE_Comm_Overlap_Algo>(m, "NVTE_Comm_Overlap_Algo", py::module_local())
    .value("BULK_OVERLAP_RS", NVTE_Comm_Overlap_Algo::BULK_OVERLAP_RS)
    .value("BULK_OVERLAP_AG", NVTE_Comm_Overlap_Algo::BULK_OVERLAP_AG)
    .value("SPLIT_PIPELINED_RS", NVTE_Comm_Overlap_Algo::SPLIT_PIPELINED_RS)
    .value("SPLIT_PIPELINED_RS_P2P", NVTE_Comm_Overlap_Algo::SPLIT_PIPELINED_RS_P2P)
    .value("SPLIT_PIPELINED_AG_P2P", NVTE_Comm_Overlap_Algo::SPLIT_PIPELINED_AG_P2P)
    .value("ATOMIC_GEMM_RS", NVTE_Comm_Overlap_Algo::ATOMIC_GEMM_RS)
    .value("ATOMIC_GEMM_RS_P2P", NVTE_Comm_Overlap_Algo::ATOMIC_GEMM_AG_P2P)
    .value("ATOMIC_GEMM_AG_P2P", NVTE_Comm_Overlap_Algo::ATOMIC_GEMM_AG_P2P);
}  // PYBIND11_MODULE

}  // namespace transformer_engine
