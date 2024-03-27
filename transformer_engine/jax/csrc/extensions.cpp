/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cublasLt.h>

#include "common/include/transformer_engine/fused_attn.h"
#include "common/include/transformer_engine/transformer_engine.h"
#include "jax/csrc/modules.h"
#include "jax/csrc/utils.h"

namespace transformer_engine {
namespace jax {

template <typename T>
pybind11::capsule EncapsulateFunction(T *fn) {
    return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["te_transpose"] = EncapsulateFunction(Transpose);
    dict["te_cast_transpose"] = EncapsulateFunction(CastTranspose);
    dict["te_gelu"] = EncapsulateFunction(Gelu);
    dict["te_gelu_fp8"] = EncapsulateFunction(GeluFP8);
    dict["te_dgelu"] = EncapsulateFunction(DGelu);
    dict["te_dgelu_dbias_cast_transpose"] = EncapsulateFunction(DGeluDBiasCastTranspose);
    dict["te_gated_gelu"] = EncapsulateFunction(GatedGelu);
    dict["te_gated_gelu_fp8"] = EncapsulateFunction(GatedGeluFP8);
    dict["te_dgated_gelu"] = EncapsulateFunction(DGatedGelu);
    dict["te_dgated_gelu_cast_transpose"] = EncapsulateFunction(DGatedGeluCastTranspose);
    dict["te_layernorm_forward"] = EncapsulateFunction(LayerNormForward);
    dict["te_layernorm_forward_fp8"] = EncapsulateFunction(LayerNormForwardFP8);
    dict["te_layernorm_backward"] = EncapsulateFunction(LayerNormBackward);
    dict["te_rmsnorm_forward"] = EncapsulateFunction(RMSNormForward);
    dict["te_rmsnorm_forward_fp8"] = EncapsulateFunction(RMSNormForwardFP8);
    dict["te_rmsnorm_backward"] = EncapsulateFunction(RMSNormBackward);
    dict["te_quantize"] = EncapsulateFunction(Quantize);
    dict["te_dequantize"] = EncapsulateFunction(Dequantize);
    dict["te_scaled_softmax_forward"] = EncapsulateFunction(ScaledSoftmaxForward);
    dict["te_scaled_softmax_backward"] = EncapsulateFunction(ScaledSoftmaxBackward);
    dict["te_scaled_masked_softmax_forward"] = EncapsulateFunction(ScaledMaskedSoftmaxForward);
    dict["te_scaled_masked_softmax_backward"] = EncapsulateFunction(ScaledMaskedSoftmaxBackward);
    dict["te_scaled_upper_triang_masked_softmax_forward"] =
        EncapsulateFunction(ScaledUpperTriangMaskedSoftmaxForward);
    dict["te_scaled_upper_triang_masked_softmax_backward"] =
        EncapsulateFunction(ScaledUpperTriangMaskedSoftmaxBackward);
    dict["te_fused_attn_forward"] = EncapsulateFunction(FusedAttnForward);
    dict["te_fused_attn_backward"] = EncapsulateFunction(FusedAttnBackward);
    return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
    m.def("registrations", &Registrations);
    m.def("pack_common_descriptor", &PackCustomCallCommonDescriptor);
    m.def("pack_common_wk_descriptor", &PackCustomCallCommonWkDescriptor);
    m.def("pack_norm_descriptor", &PackCustomCallNormDescriptor);
    m.def("pack_softmax_descriptor", &PackCustomCallSoftmaxDescriptor);
    m.def("pack_fused_attn_descriptor", &PackCustomCallFusedAttnDescriptor);
    m.def("get_fused_attn_backend", &GetFusedAttnBackend);
    m.def("get_cuda_version", &GetCudaRuntimeVersion);
    m.def("get_device_compute_capability", &GetDeviceComputeCapability);
    m.def("get_cublasLt_version", &cublasLtGetVersion);
    m.def("get_dgelu_dbias_ct_workspace_sizes", &GetDGeluDBiasCastTransposeWorkspaceSizes);
    m.def("get_layernorm_fwd_workspace_sizes", &GetLayerNormForwardWorkspaceSizes);
    m.def("get_layernorm_bwd_workspace_sizes", &GetLayerNormBackwardWorkspaceSizes);
    m.def("get_fused_attn_fwd_workspace_sizes", &GetFusedAttnForwardWorkspaceSizes);
    m.def("get_fused_attn_bwd_workspace_sizes", &GetFusedAttnBackwardWorkspaceSizes);

    pybind11::enum_<DType>(m, "DType", pybind11::module_local())
        .value("kByte", DType::kByte)
        .value("kInt32", DType::kInt32)
        .value("kInt64", DType::kInt64)
        .value("kFloat32", DType::kFloat32)
        .value("kFloat16", DType::kFloat16)
        .value("kBFloat16", DType::kBFloat16)
        .value("kFloat8E4M3", DType::kFloat8E4M3)
        .value("kFloat8E5M2", DType::kFloat8E5M2);

    pybind11::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type", pybind11::module_local())
        .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)
        .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)
        .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);

    pybind11::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type", pybind11::module_local())
        .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)
        .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)
        .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK)
        .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK);

    pybind11::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout", pybind11::module_local())
        .value("NVTE_BS3HD", NVTE_QKV_Layout::NVTE_BS3HD)
        .value("NVTE_BSHD_BS2HD", NVTE_QKV_Layout::NVTE_BSHD_BS2HD)
        .value("NVTE_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD);

    pybind11::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", pybind11::module_local())
        .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend)
        .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)
        .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)
        .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8);
}

}  // namespace jax
}  // namespace transformer_engine
