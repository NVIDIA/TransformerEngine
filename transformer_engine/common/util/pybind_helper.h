/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>

#define NVTE_ADD_COMMON_PYBIND11_BINDINGS(mod)                                                \
    auto nvte = pybind11::module_::import("transformer_engine_common");    \
    mod.attr("DType") = nvte.attr("DType");                                                   \
    mod.attr("NVTE_Activation_Type") = nvte.attr("NVTE_Activation_Type");                     \
    mod.attr("NVTE_Bias_Type") = nvte.attr("NVTE_Bias_Type");                                 \
    mod.attr("NVTE_Mask_Type") = nvte.attr("NVTE_Mask_Type");                                 \
    mod.attr("NVTE_QKV_Layout") = nvte.attr("NVTE_QKV_Layout");                               \
    mod.attr("NVTE_Fused_Attn_Backend") = nvte.attr("NVTE_Fused_Attn_Backend");               \
    mod.attr("NVTE_Comm_Overlap_Type") = nvte.attr("NVTE_Comm_Overlap_Type");                 \
    mod.attr("NVTE_Comm_Overlap_Algo") = nvte.attr("NVTE_Comm_Overlap_Algo");                 \
    mod.attr("overlapped_comm_type") = nvte.attr("overlapped_comm_type");                     \
    mod.attr("comm_overlap_is_p2p") = nvte.attr("comm_overlap_is_p2p");
