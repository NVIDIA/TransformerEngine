/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>

#define NVTE_ADD_PYBIND11_BINDINGS(ext_mod)                                                      \
    auto te_common = pybind11::module_::import("transformer_engine_pybind");                     \
    ext_mod.attr("DType") = te_common.attr("DType");                                             \
    ext_mod.attr("NVTE_Activation_Type") = te_common.attr("NVTE_Activation_Type");               \
    ext_mod.attr("NVTE_Bias_Type") = te_common.attr("NVTE_Bias_Type");                           \
    ext_mod.attr("NVTE_Mask_Type") = te_common.attr("NVTE_Mask_Type");                           \
    ext_mod.attr("NVTE_QKV_Layout") = te_common.attr("NVTE_QKV_Layout");                         \
    ext_mod.attr("NVTE_Fused_Attn_Backend") = te_common.attr("NVTE_Fused_Attn_Backend");         \
    ext_mod.attr("NVTE_Comm_Overlap_Type") = te_common.attr("NVTE_Comm_Overlap_Type");           \
    ext_mod.attr("NVTE_Comm_Overlap_Algo") = te_common.attr("NVTE_Comm_Overlap_Algo");           \
    ext_mod.attr("CommGemmOverlapBase") = te_common.attr("CommGemmOverlapBase");
