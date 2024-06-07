/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cublasLt.h>

#include "modules.h"
#include "utils.h"

#include "common/util/pybind_helper.h"

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

    dict["te_act_lu"] = EncapsulateFunction(ActLu);
    dict["te_act_lu_fp8"] = EncapsulateFunction(ActLuFP8);
    dict["te_dact_lu"] = EncapsulateFunction(DActLu);
    dict["te_dbias_cast_transpose"] = EncapsulateFunction(DBiasCastTranspose);
    dict["te_dact_lu_dbias_cast_transpose"] = EncapsulateFunction(DActLuDBiasCastTranspose);
    dict["te_dgated_act_lu_cast_transpose"] = EncapsulateFunction(DGatedActLuCastTranspose);

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
    // Load nvte = py::module_::import("transformer_engine_common") into TE/JAX. This makes
    // essential NVTE enums available through `import transformer_engine_jax` without requiring an
    // additional `import transformer_engine_common`.
    NVTE_ADD_COMMON_PYBIND11_BINDINGS(m);

    m.def("registrations", &Registrations);
    m.def("pack_common_descriptor", &PackCustomCallCommonDescriptor,
          pybind11::arg(), pybind11::arg(), pybind11::arg(), pybind11::arg("act_num") = 0);
    m.def("pack_common_wk_descriptor", &PackCustomCallCommonWkDescriptor,
          pybind11::arg(), pybind11::arg(), pybind11::arg(),
          pybind11::arg(), pybind11::arg(), pybind11::arg("act_num") = 0);
    m.def("pack_norm_descriptor", &PackCustomCallNormDescriptor);
    m.def("pack_softmax_descriptor", &PackCustomCallSoftmaxDescriptor);
    m.def("pack_fused_attn_descriptor", &PackCustomCallFusedAttnDescriptor);
    m.def("get_fused_attn_backend", &GetFusedAttnBackend);
    m.def("get_cuda_version", &GetCudaRuntimeVersion);
    m.def("get_device_compute_capability", &GetDeviceComputeCapability);
    m.def("get_cublasLt_version", &cublasLtGetVersion);
    m.def("get_dact_dbias_ct_workspace_sizes", &GetDActDBiasCastTransposeWorkspaceSizes);
    m.def("get_dbias_ct_workspace_sizes", &GetDBiasCastTransposeWorkspaceSizes);
    m.def("get_layernorm_fwd_workspace_sizes", &GetLayerNormForwardWorkspaceSizes);
    m.def("get_layernorm_bwd_workspace_sizes", &GetLayerNormBackwardWorkspaceSizes);
    m.def("get_fused_attn_fwd_workspace_sizes", &GetFusedAttnForwardWorkspaceSizes);
    m.def("get_fused_attn_bwd_workspace_sizes", &GetFusedAttnBackwardWorkspaceSizes);
}

}  // namespace jax
}  // namespace transformer_engine
