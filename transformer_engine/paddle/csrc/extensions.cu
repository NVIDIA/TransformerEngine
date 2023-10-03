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
    m.def("get_fused_attn_backend", &get_fused_attn_backend, "Get Fused Attention backend");
    // Data structures
    py::enum_<DType>(m, "DType", py::module_local())
        .value("kByte", DType::kByte)
        .value("kInt32", DType::kInt32)
        .value("kFloat32", DType::kFloat32)
        .value("kFloat16", DType::kFloat16)
        .value("kBFloat16", DType::kBFloat16)
        .value("kFloat8E4M3", DType::kFloat8E4M3)
        .value("kFloat8E5M2", DType::kFloat8E5M2);

    py::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type")
        .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)
        .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)
        .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);

    py::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type")
        .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)
        .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)
        .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK);

    py::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout")
        .value("NVTE_NOT_INTERLEAVED", NVTE_QKV_Layout::NVTE_NOT_INTERLEAVED)
        .value("NVTE_QKV_INTERLEAVED", NVTE_QKV_Layout::NVTE_QKV_INTERLEAVED)
        .value("NVTE_KV_INTERLEAVED", NVTE_QKV_Layout::NVTE_KV_INTERLEAVED);

    py::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", py::module_local())
        .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)
        .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)
        .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8)
        .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend);
}
}  // namespace paddle_ext
}  // namespace transformer_engine
