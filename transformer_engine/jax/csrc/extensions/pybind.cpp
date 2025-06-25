/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

#include "common/util/pybind_helper.h"
namespace transformer_engine {
namespace jax {

template <typename T>
pybind11::capsule EncapsulateFFI(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;

  // Activation
  dict["te_act_lu_ffi"] = EncapsulateFFI(ActLuHandler);
  dict["te_dact_dbias_quantize_ffi"] = EncapsulateFFI(DActLuDBiasQuantizeHandler);

  // Quantization
  dict["te_dbias_quantize_ffi"] = EncapsulateFFI(DBiasQuantizeHandler);
  dict["te_grouped_quantize_ffi"] = EncapsulateFFI(GroupedQuantizeHandler);
  dict["te_dequantize_ffi"] = EncapsulateFFI(DequantizeHandler);

  // Softmax
  dict["te_scaled_softmax_forward_ffi"] = EncapsulateFFI(ScaledSoftmaxForwardHandler);
  dict["te_scaled_softmax_backward_ffi"] = EncapsulateFFI(ScaledSoftmaxBackwardHandler);
  dict["te_scaled_masked_softmax_forward_ffi"] = EncapsulateFFI(ScaledMaskedSoftmaxForwardHandler);
  dict["te_scaled_masked_softmax_backward_ffi"] =
      EncapsulateFFI(ScaledMaskedSoftmaxBackwardHandler);
  dict["te_scaled_upper_triang_masked_softmax_forward_ffi"] =
      EncapsulateFFI(ScaledUpperTriangMaskedSoftmaxForwardHandler);
  dict["te_scaled_upper_triang_masked_softmax_backward_ffi"] =
      EncapsulateFFI(ScaledUpperTriangMaskedSoftmaxBackwardHandler);

  // Normalization
  dict["te_norm_forward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(NormForwardHandler));
  dict["te_norm_backward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(NormBackwardHandler));

  // Attention
  dict["te_fused_attn_forward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(FusedAttnForwardHandler));
  dict["te_fused_attn_backward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(FusedAttnBackwardHandler));

  // GEMM
  dict["te_gemm_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CublasHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GemmHandler));

  // Grouped GEMM
  dict["te_grouped_gemm_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CublasHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GroupedGemmHandler));

  return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
  NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)

  m.def("registrations", &Registrations);
  m.def("get_fused_attn_backend", &GetFusedAttnBackend);
  m.def("get_cuda_version", &GetCudaRuntimeVersion);
  m.def("get_cudnn_version", &GetCudnnRuntimeVersion);
  m.def("get_device_compute_capability", &GetDeviceComputeCapability);
  m.def("get_cublasLt_version", &cublasLtGetVersion);
  m.def("get_dact_dbias_quantize_workspace_sizes", &GetDActDBiasQuantizeWorkspaceSizes);
  m.def("get_dbias_quantize_workspace_sizes", &GetDBiasQuantizeWorkspaceSizes);
  m.def("get_norm_fwd_workspace_sizes", &GetNormForwardWorkspaceSizes);
  m.def("get_norm_bwd_workspace_sizes", &GetNormBackwardWorkspaceSizes);
  m.def("get_fused_attn_fwd_workspace_sizes", &GetFusedAttnForwardWorkspaceSizes);
  m.def("get_fused_attn_bwd_workspace_sizes", &GetFusedAttnBackwardWorkspaceSizes);

  pybind11::enum_<JAXX_Scaling_Mode>(m, "JAXX_Scaling_Mode", pybind11::module_local())
      .value("NO_SCALING", JAXX_Scaling_Mode::NO_SCALING)
      .value("DELAYED_TENSOR_SCALING", JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
      .value("MXFP8_1D_SCALING", JAXX_Scaling_Mode::MXFP8_1D_SCALING)
      .value("CURRENT_TENSOR_SCALING", JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING);

  pybind11::enum_<transformer_engine::jax::QuantizeLayout>(m, "QuantizeLayout",
                                                           pybind11::module_local())
      .value("ROWWISE", transformer_engine::jax::QuantizeLayout::ROWWISE)
      .value("COLWISE", transformer_engine::jax::QuantizeLayout::COLWISE)
      .value("ROWWISE_COLWISE", transformer_engine::jax::QuantizeLayout::ROWWISE_COLWISE);
}

}  // namespace jax
}  // namespace transformer_engine
