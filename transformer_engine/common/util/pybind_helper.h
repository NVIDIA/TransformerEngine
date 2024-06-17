/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_

#include <pybind11/pybind11.h>

#define NVTE_ADD_COMMON_PYBIND11_BINDINGS(mod)                                                \
  auto nvte = pybind11::module_::import("transformer_engine_common");                         \
  mod.attr("DType") = nvte.attr("DType");                                                     \
  mod.attr("NVTE_Activation_Type") = nvte.attr("NVTE_Activation_Type");                       \
  mod.attr("NVTE_Bias_Type") = nvte.attr("NVTE_Bias_Type");                                   \
  mod.attr("NVTE_Mask_Type") = nvte.attr("NVTE_Mask_Type");                                   \
  mod.attr("NVTE_QKV_Layout") = nvte.attr("NVTE_QKV_Layout");                                 \
  mod.attr("NVTE_Fused_Attn_Backend") = nvte.attr("NVTE_Fused_Attn_Backend");                 \
  mod.attr("NVTE_COMM_OVERLAP_MAX_STREAMS") = nvte.attr("NVTE_COMM_OVERLAP_MAX_STREAMS");     \
  mod.attr("comm_overlap_supports_multicast") = nvte.attr("comm_overlap_supports_multicast"); \
  mod.attr("NVTE_Comm_Overlap_Type") = nvte.attr("NVTE_Comm_Overlap_Type");                   \
  mod.attr("NVTE_Comm_Overlap_Algo") = nvte.attr("NVTE_Comm_Overlap_Algo");

#endif  // TRANSFORMER_ENGINE_COMMON_PYBIND_HELPER_H_
