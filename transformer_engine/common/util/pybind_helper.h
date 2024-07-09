/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_

#include <pybind11/pybind11.h>

#define NVTE_ADD_COMMON_PYBIND11_BINDINGS(m)                                                \
  auto nvte = pybind11::module_::import("transformer_engine_common");                       \
  m.attr("DType") = nvte.attr("DType");                                                     \
  m.attr("NVTE_Activation_Type") = nvte.attr("NVTE_Activation_Type");                       \
  m.attr("NVTE_Bias_Type") = nvte.attr("NVTE_Bias_Type");                                   \
  m.attr("NVTE_Mask_Type") = nvte.attr("NVTE_Mask_Type");                                   \
  m.attr("NVTE_QKV_Layout") = nvte.attr("NVTE_QKV_Layout");                                 \
  m.attr("NVTE_Fused_Attn_Backend") = nvte.attr("NVTE_Fused_Attn_Backend");                 \
  m.attr("NVTE_Comm_Overlap_Type") = nvte.attr("NVTE_Comm_Overlap_Type");                   \
  m.attr("NVTE_Comm_Overlap_Algo") = nvte.attr("NVTE_Comm_Overlap_Algo");                   \
  m.attr("NVTE_COMM_OVERLAP_MAX_STREAMS") = nvte.attr("NVTE_COMM_OVERLAP_MAX_STREAMS");     \
  m.attr("comm_overlap_supports_multicast") = nvte.attr("comm_overlap_supports_multicast"); \

#endif  // TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_
