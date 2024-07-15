/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <transformer_engine/activation.h>
#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/transformer_engine.h>

#define NVTE_ADD_COMMON_PYBIND11_BINDINGS(m)                                                       \
  pybind11::enum_<transformer_engine::DType>(m, "DType", pybind11::module_local())                 \
      .value("kByte", transformer_engine::DType::kByte)                                            \
      .value("kInt32", transformer_engine::DType::kInt32)                                          \
      .value("kInt64", transformer_engine::DType::kInt64)                                          \
      .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)                                \
      .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2)                                \
      .value("kBFloat16", transformer_engine::DType::kBFloat16)                                    \
      .value("kFloat16", transformer_engine::DType::kFloat16)                                      \
      .value("kFloat32", transformer_engine::DType::kFloat32);                                     \
  pybind11::enum_<NVTE_Activation_Type>(m, "NVTE_Activation_Type", pybind11::module_local())       \
      .value("GELU", NVTE_Activation_Type::GELU)                                                   \
      .value("GEGLU", NVTE_Activation_Type::GEGLU)                                                 \
      .value("SILU", NVTE_Activation_Type::SILU)                                                   \
      .value("SWIGLU", NVTE_Activation_Type::SWIGLU)                                               \
      .value("RELU", NVTE_Activation_Type::RELU)                                                   \
      .value("REGLU", NVTE_Activation_Type::REGLU)                                                 \
      .value("QGELU", NVTE_Activation_Type::QGELU)                                                 \
      .value("QGEGLU", NVTE_Activation_Type::QGEGLU)                                               \
      .value("SRELU", NVTE_Activation_Type::SRELU)                                                 \
      .value("SREGLU", NVTE_Activation_Type::SREGLU);                                              \
  pybind11::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type", pybind11::module_local())                   \
      .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)                                         \
      .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)                           \
      .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)                         \
      .value("NVTE_ALIBI", NVTE_Bias_Type::NVTE_ALIBI);                                            \
  pybind11::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type", pybind11::module_local())                   \
      .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)                                         \
      .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)                               \
      .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK)                                 \
      .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK);                \
  pybind11::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout", pybind11::module_local())                 \
      .value("NVTE_SB3HD", NVTE_QKV_Layout::NVTE_SB3HD)                                            \
      .value("NVTE_SBH3D", NVTE_QKV_Layout::NVTE_SBH3D)                                            \
      .value("NVTE_SBHD_SB2HD", NVTE_QKV_Layout::NVTE_SBHD_SB2HD)                                  \
      .value("NVTE_SBHD_SBH2D", NVTE_QKV_Layout::NVTE_SBHD_SBH2D)                                  \
      .value("NVTE_SBHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD)                          \
      .value("NVTE_BS3HD", NVTE_QKV_Layout::NVTE_BS3HD)                                            \
      .value("NVTE_BSH3D", NVTE_QKV_Layout::NVTE_BSH3D)                                            \
      .value("NVTE_BSHD_BS2HD", NVTE_QKV_Layout::NVTE_BSHD_BS2HD)                                  \
      .value("NVTE_BSHD_BSH2D", NVTE_QKV_Layout::NVTE_BSHD_BSH2D)                                  \
      .value("NVTE_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD)                          \
      .value("NVTE_T3HD", NVTE_QKV_Layout::NVTE_T3HD)                                              \
      .value("NVTE_TH3D", NVTE_QKV_Layout::NVTE_TH3D)                                              \
      .value("NVTE_THD_T2HD", NVTE_QKV_Layout::NVTE_THD_T2HD)                                      \
      .value("NVTE_THD_TH2D", NVTE_QKV_Layout::NVTE_THD_TH2D)                                      \
      .value("NVTE_THD_THD_THD", NVTE_QKV_Layout::NVTE_THD_THD_THD);                               \
  pybind11::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", pybind11::module_local()) \
      .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)            \
      .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)      \
      .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8)                                        \
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend);                         \
  pybind11::enum_<NVTE_Comm_Overlap_Type>(m, "NVTE_Comm_Overlap_Type", pybind11::module_local())   \
      .value("AG", NVTE_Comm_Overlap_Type::ALL_GATHER)                                             \
      .value("RS", NVTE_Comm_Overlap_Type::REDUCE_SCATTER);                                        \
  pybind11::enum_<NVTE_Comm_Overlap_Algo>(m, "NVTE_Comm_Overlap_Algo", pybind11::module_local())   \
      .value("BULK_OVERLAP_RS", NVTE_Comm_Overlap_Algo::BULK_OVERLAP_RS)                           \
      .value("BULK_OVERLAP_AG", NVTE_Comm_Overlap_Algo::BULK_OVERLAP_AG)                           \
      .value("SPLIT_PIPELINED_RS", NVTE_Comm_Overlap_Algo::SPLIT_PIPELINED_RS)                     \
      .value("SPLIT_PIPELINED_RS_P2P", NVTE_Comm_Overlap_Algo::SPLIT_PIPELINED_RS_P2P)             \
      .value("SPLIT_PIPELINED_AG_P2P", NVTE_Comm_Overlap_Algo::SPLIT_PIPELINED_AG_P2P)             \
      .value("ATOMIC_GEMM_RS", NVTE_Comm_Overlap_Algo::ATOMIC_GEMM_RS)                             \
      .value("ATOMIC_GEMM_RS_P2P", NVTE_Comm_Overlap_Algo::ATOMIC_GEMM_RS_P2P)                     \
      .value("ATOMIC_GEMM_AG_P2P", NVTE_Comm_Overlap_Algo::ATOMIC_GEMM_AG_P2P);                    \
  m.attr("NVTE_COMM_OVERLAP_MAX_STREAMS") = pybind11::int_(NVTE_COMM_OVERLAP_MAX_STREAMS);         \
  m.def("device_supports_multicast", &transformer_engine::device_supports_multicast,               \
        pybind11::call_guard<pybind11::gil_scoped_release>());

#endif  // TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_
