/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common/util/cuda_runtime.h"

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
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CollectiveGemmInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GemmHandler));

  // Grouped GEMM
  dict["te_grouped_gemm_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CublasHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GroupedGemmHandler));

  return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
  m.def("registrations", &Registrations);
  m.def("get_fused_attn_backend", &GetFusedAttnBackend);
  m.def("get_cuda_version", &GetCudaRuntimeVersion);
  m.def("get_cudnn_version", &GetCudnnRuntimeVersion);
  m.def("get_device_compute_capability", &GetDeviceComputeCapability);
  m.def("get_num_compute_streams", &nvte_get_num_compute_streams);
  m.def("get_cublasLt_version", &cublasLtGetVersion);
  m.def("get_dact_dbias_quantize_workspace_sizes", &GetDActDBiasQuantizeWorkspaceSizes);
  m.def("get_dbias_quantize_workspace_sizes", &GetDBiasQuantizeWorkspaceSizes);
  m.def("get_norm_fwd_workspace_sizes", &GetNormForwardWorkspaceSizes);
  m.def("get_norm_bwd_workspace_sizes", &GetNormBackwardWorkspaceSizes);
  m.def("get_fused_attn_fwd_workspace_sizes", &GetFusedAttnForwardWorkspaceSizes);
  m.def("get_fused_attn_bwd_workspace_sizes", &GetFusedAttnBackwardWorkspaceSizes);
  m.def("nvte_get_qkv_format", &nvte_get_qkv_format);
  m.def("is_non_nt_fp8_gemm_supported", &nvte_is_non_tn_fp8_gemm_supported);

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
      .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)
      .value("NVTE_CAUSAL_BOTTOM_RIGHT_MASK", NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK)
      .value("NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK",
             NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);

  pybind11::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout", pybind11::module_local())
      .value("NVTE_BS3HD", NVTE_QKV_Layout::NVTE_BS3HD)
      .value("NVTE_BSHD_BS2HD", NVTE_QKV_Layout::NVTE_BSHD_BS2HD)
      .value("NVTE_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD)
      .value("NVTE_T3HD", NVTE_QKV_Layout::NVTE_T3HD)
      .value("NVTE_THD_T2HD", NVTE_QKV_Layout::NVTE_THD_T2HD)
      .value("NVTE_THD_THD_THD", NVTE_QKV_Layout::NVTE_THD_THD_THD);

  pybind11::enum_<NVTE_QKV_Format>(m, "NVTE_QKV_Format", pybind11::module_local())
      .value("NVTE_SBHD", NVTE_QKV_Format::NVTE_SBHD)
      .value("NVTE_BSHD", NVTE_QKV_Format::NVTE_BSHD)
      .value("NVTE_THD", NVTE_QKV_Format::NVTE_THD);

  pybind11::enum_<NVTE_Activation_Type>(m, "NVTE_Activation_Type", pybind11::module_local())
      .value("GELU", NVTE_Activation_Type::GELU)
      .value("GEGLU", NVTE_Activation_Type::GEGLU)
      .value("SILU", NVTE_Activation_Type::SILU)
      .value("SWIGLU", NVTE_Activation_Type::SWIGLU)
      .value("RELU", NVTE_Activation_Type::RELU)
      .value("REGLU", NVTE_Activation_Type::REGLU)
      .value("QGELU", NVTE_Activation_Type::QGELU)
      .value("QGEGLU", NVTE_Activation_Type::QGEGLU)
      .value("SRELU", NVTE_Activation_Type::SRELU)
      .value("SREGLU", NVTE_Activation_Type::SREGLU)
      .export_values();

  pybind11::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", pybind11::module_local())
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend)
      .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)
      .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)
      .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8);

  pybind11::enum_<NVTE_Norm_Type>(m, "NVTE_Norm_Type", pybind11::module_local())
      .value("LayerNorm", NVTE_Norm_Type::LayerNorm)
      .value("RMSNorm", NVTE_Norm_Type::RMSNorm)
      .export_values();

  pybind11::enum_<JAXX_Scaling_Mode>(m, "JAXX_Scaling_Mode", pybind11::module_local())
      .value("NO_SCALING", JAXX_Scaling_Mode::NO_SCALING)
      .value("DELAYED_TENSOR_SCALING", JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING)
      .value("MXFP8_1D_SCALING", JAXX_Scaling_Mode::MXFP8_1D_SCALING)
      .value("CURRENT_TENSOR_SCALING", JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING)
      .export_values();

  pybind11::enum_<transformer_engine::jax::QuantizeLayout>(m, "QuantizeLayout",
                                                           pybind11::module_local())
      .value("ROWWISE", transformer_engine::jax::QuantizeLayout::ROWWISE)
      .value("COLWISE", transformer_engine::jax::QuantizeLayout::COLWISE)
      .value("ROWWISE_COLWISE", transformer_engine::jax::QuantizeLayout::ROWWISE_COLWISE)
      .export_values();

  pybind11::enum_<JAXX_Collective_Op>(m, "JAXX_Collective_Op", pybind11::module_local())
      .value("NONE", JAXX_Collective_Op::NONE)
      .value("ALL_GATHER", JAXX_Collective_Op::ALL_GATHER)
      .value("REDUCE_SCATTER", JAXX_Collective_Op::REDUCE_SCATTER)
      .export_values();
}

}  // namespace jax
}  // namespace transformer_engine
