/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/util/pybind_helper.h"
#include "extensions.h"

namespace transformer_engine {

namespace jax {

template <typename T>
pybind11::capsule EncapsulateFunction(T *fn) {
  return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

template <typename T>
pybind11::capsule EncapsulateFFI(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
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
  dict["te_gemm"] = EncapsulateFunction(Gemm);
  dict["te_copy_into_overlap_buffer"] = EncapsulateFunction(CopyIntoOverlapBuffer);
  dict["te_comm_gemm_overlap"] = EncapsulateFunction(CommGemmOverlap);

  // Transpose
  dict["te_transpose_ffi"] = EncapsulateFFI(TransposeHandler);
  dict["te_cast_transpose_ffi"] = EncapsulateFFI(CastTransposeHandler);
  dict["te_dbias_cast_transpose_ffi"] = EncapsulateFFI(DBiasCastTransposeHandler);

  // Activation
  dict["te_act_lu_ffi"] = EncapsulateFFI(ActLuHandler);
  dict["te_act_lu_fp8_ffi"] = EncapsulateFFI(ActLuFP8Handler);
  dict["te_dact_lu_ffi"] = EncapsulateFFI(DActLuHandler);
  dict["te_dact_lu_dbias_cast_transpose_ffi"] =
      EncapsulateFunction(DActLuDBiasCastTransposeHandler);
  dict["te_dgated_act_lu_cast_transpose_ffi"] =
      EncapsulateFunction(DGatedActLuCastTransposeHandler);

  // Quantization
  dict["te_quantize_ffi"] = EncapsulateFFI(QuantizeHandler);
  dict["te_dequantize_ffi"] = EncapsulateFFI(DequantizeHandler);

  // Softmax
  dict["te_scaled_softmax_forward_ffi"] = EncapsulateFunction(ScaledSoftmaxForwardHandler);
  dict["te_scaled_softmax_backward_ffi"] = EncapsulateFunction(ScaledSoftmaxBackwardHandler);
  dict["te_scaled_masked_softmax_forward_ffi"] =
      EncapsulateFunction(ScaledMaskedSoftmaxForwardHandler);
  dict["te_scaled_masked_softmax_backward_ffi"] =
      EncapsulateFunction(ScaledMaskedSoftmaxBackwardHandler);
  dict["te_scaled_upper_triang_masked_softmax_forward_ffi"] =
      EncapsulateFunction(ScaledUpperTriangMaskedSoftmaxForwardHandler);
  dict["te_scaled_upper_triang_masked_softmax_backward_ffi"] =
      EncapsulateFunction(ScaledUpperTriangMaskedSoftmaxBackwardHandler);

  // Normalization
  dict["te_layernorm_forward_ffi"] = EncapsulateFFI(LayerNormForwardHandler);
  dict["te_layernorm_forward_fp8_ffi"] = EncapsulateFFI(LayerNormForwardFP8Handler);
  dict["te_layernorm_backward_ffi"] = EncapsulateFFI(LayerNormBackwardHandler);
  dict["te_rmsnorm_forward_ffi"] = EncapsulateFunction(RMSNormForwardHandler);
  dict["te_rmsnorm_forward_fp8_ffi"] = EncapsulateFunction(RMSNormForwardFP8Handler);
  dict["te_rmsnorm_backward_ffi"] = EncapsulateFunction(RMSNormBackwardHandler);

  // Attention
  pybind11::dict fused_attn_forward_ffi;
  fused_attn_forward_ffi["prepare"] = EncapsulateFFI(CudnnHandleInitHandler);
  fused_attn_forward_ffi["execute"] = EncapsulateFFI(FusedAttnForwardHandler);
  dict["te_fused_attn_forward_ffi"] = fused_attn_forward_ffi;

  pybind11::dict fused_attn_backward_ffi;
  fused_attn_backward_ffi["prepare"] = EncapsulateFFI(CudnnHandleInitHandler);
  fused_attn_backward_ffi["execute"] = EncapsulateFFI(FusedAttnBackwardHandler);
  dict["te_fused_attn_backward_ffi"] = fused_attn_backward_ffi;

  dict["te_gemm_ffi"] = EncapsulateFFI(GemmHandler);
  dict["te_copy_into_overlap_buffer_ffi"] = EncapsulateFFI(CopyIntoOverlapBufferHandler);
  dict["te_comm_gemm_overlap_ffi"] = EncapsulateFFI(CommGemmOverlapHandler);
  return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
  NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)

  m.def("registrations", &Registrations);
  m.def("pack_common_descriptor", &PackCustomCallCommonDescriptor, pybind11::arg(), pybind11::arg(),
        pybind11::arg(), pybind11::arg("act_num") = 0);
  m.def("pack_common_wk_descriptor", &PackCustomCallCommonWkDescriptor, pybind11::arg(),
        pybind11::arg(), pybind11::arg(), pybind11::arg(), pybind11::arg(),
        pybind11::arg("act_num") = 0);
  m.def("pack_norm_descriptor", &PackCustomCallNormDescriptor);
  m.def("pack_softmax_descriptor", &PackCustomCallSoftmaxDescriptor);
  m.def("pack_fused_attn_descriptor", &PackCustomCallFusedAttnDescriptor);
  m.def("pack_gemm_descriptor", &PackCustomCallGemmDescriptor);
  m.def("pack_buffer_descriptor", &PackCustomCallBufferDescriptor);
  m.def("pack_overlap_descriptor", &PackCustomCallOverlapDescriptor);
  m.def("get_fused_attn_backend", &GetFusedAttnBackend);
  m.def("get_cuda_version", &GetCudaRuntimeVersion);
  m.def("get_cudnn_version", &GetCudnnRuntimeVersion);
  m.def("get_device_compute_capability", &GetDeviceComputeCapability, pybind11::arg("gpu_id") = -1);
  m.def("get_cublasLt_version", &cublasLtGetVersion);
  m.def("get_dact_dbias_ct_workspace_sizes", &GetDActDBiasCastTransposeWorkspaceSizes);
  m.def("get_dbias_ct_workspace_sizes", &GetDBiasCastTransposeWorkspaceSizes);
  m.def("get_layernorm_fwd_workspace_sizes", &GetLayerNormForwardWorkspaceSizes);
  m.def("get_layernorm_bwd_workspace_sizes", &GetLayerNormBackwardWorkspaceSizes);
  m.def("get_fused_attn_fwd_workspace_sizes", &GetFusedAttnForwardWorkspaceSizes);
  m.def("get_fused_attn_bwd_workspace_sizes", &GetFusedAttnBackwardWorkspaceSizes);
  m.def("nvte_get_qkv_format", &nvte_get_qkv_format);
  m.def("bootstrap_comm_gemm_overlap", &BootstrapCommGemmOverlap);
  m.def("destroy_comm_gemm_overlaps", &DestroyCommGemmOverlap);
  m.def("set_buffer_scale_inv", &SetOverlapBufferScaleInverse, pybind11::arg(), pybind11::arg(),
        pybind11::arg("grad") = false);
  m.def("get_overlap_buffer", &GetOverlapBuffer);
  m.def("overlap_buffer_is_fp8", &OverlapBufferIsFp8);
}

}  // namespace jax

}  // namespace transformer_engine
