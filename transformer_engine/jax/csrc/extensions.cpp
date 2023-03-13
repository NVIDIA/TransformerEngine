/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    dict["te_gated_gelu"] = EncapsulateFunction(GatedGelu);
    dict["te_gated_gelu_fp8"] = EncapsulateFunction(GatedGeluFP8);
    dict["te_dgated_gelu"] = EncapsulateFunction(DGatedGelu);
    dict["te_dgated_gelu_cast_transpose"] = EncapsulateFunction(DGatedGeluCastTranspose);
    dict["te_gemm"] = EncapsulateFunction(Gemm);
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
    return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
    m.def("registrations", &Registrations);
    m.def("pack_common_descriptor", &PackCustomCallCommonDescriptor);
    m.def("pack_gemm_descriptor", &PackCustomCallGemmDescriptor);
    m.def("pack_norm_descriptor", &PackCustomCallNormDescriptor);
    m.def("pack_softmax_descriptor", &PackCustomCallSoftmaxDescriptor);

    pybind11::enum_<DType>(m, "DType", pybind11::module_local())
        .value("kByte", DType::kByte)
        .value("kInt32", DType::kInt32)
        .value("kFloat32", DType::kFloat32)
        .value("kFloat16", DType::kFloat16)
        .value("kBFloat16", DType::kBFloat16)
        .value("kFloat8E4M3", DType::kFloat8E4M3)
        .value("kFloat8E5M2", DType::kFloat8E5M2);
}

}  // namespace jax
}  // namespace transformer_engine
