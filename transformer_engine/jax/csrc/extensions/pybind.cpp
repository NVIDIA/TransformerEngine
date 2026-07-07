/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "cgemm_helper.h"
#include "common/util/cuda_runtime.h"
#include "transformer_engine/gemm.h"

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
  dict["te_act_lu_ffi"] =
      pybind11::dict(pybind11::arg("initialize") = EncapsulateFFI(ActLuInitializeHandler),
                     pybind11::arg("execute") = EncapsulateFFI(ActLuHandler));
  dict["te_dact_dbias_quantize_ffi"] = pybind11::dict(
      pybind11::arg("initialize") = EncapsulateFFI(DActLuDBiasQuantizeInitializeHandler),
      pybind11::arg("execute") = EncapsulateFFI(DActLuDBiasQuantizeHandler));

  // Quantization
  dict["te_dbias_quantize_ffi"] = EncapsulateFFI(DBiasQuantizeHandler);
  dict["te_grouped_quantize_ffi"] = EncapsulateFFI(GroupedQuantizeHandler);
  dict["te_grouped_quantize_v2_ffi"] = EncapsulateFFI(GroupedQuantizeV2Handler);
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
                     pybind11::arg("initialize") = EncapsulateFFI(NormForwardInitializeHandler),
                     pybind11::arg("execute") = EncapsulateFFI(NormForwardHandler));
  dict["te_norm_backward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("initialize") = EncapsulateFFI(NormBackwardInitializeHandler),
                     pybind11::arg("execute") = EncapsulateFFI(NormBackwardHandler));

  // Attention
  dict["te_fused_attn_forward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(FusedAttnForwardHandler));
  dict["te_fused_attn_backward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(FusedAttnBackwardHandler));
  dict["te_fused_attn_score_mod_forward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(FusedAttnScoreModForwardHandler));
  dict["te_fused_attn_score_mod_backward_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CudnnHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(FusedAttnScoreModBackwardHandler));

  // GEMM
  dict["te_gemm_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CollectiveGemmInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GemmHandler));

  dict["te_gemm_v2_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(GemmInitV2Handler),
                     pybind11::arg("execute") = EncapsulateFFI(GemmV2Handler));

  // Grouped GEMM
  dict["te_grouped_gemm_d2h_group_sizes_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CublasHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GroupedGemmD2HGroupSizesHandler));
  dict["te_grouped_gemm_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CublasHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GroupedGemmHandler));
  dict["te_grouped_gemm_v2_ffi"] =
      pybind11::dict(pybind11::arg("prepare") = EncapsulateFFI(CublasHandleInitHandler),
                     pybind11::arg("execute") = EncapsulateFFI(GroupedGemmV2Handler));

  // Amax
  dict["te_rht_amax_ffi"] = pybind11::dict(
      pybind11::arg("initialize") = EncapsulateFFI(RHTAmaxCalculationInitializeHandler),
      pybind11::arg("execute") = EncapsulateFFI(RHTAmaxCalculationHandler));

  dict["te_inspect_ffi"] =
      pybind11::dict(pybind11::arg("execute") = EncapsulateFFI(InspectHandler));

  // Router
  dict["te_fused_topk_with_score_function_forward_ffi"] =
      EncapsulateFFI(FusedTopkWithScoreFunctionForwardHandler);
  dict["te_fused_topk_with_score_function_backward_ffi"] =
      EncapsulateFFI(FusedTopkWithScoreFunctionBackwardHandler);
  dict["te_fused_moe_aux_loss_forward_ffi"] = EncapsulateFFI(FusedMoEAuxLossForwardHandler);
  dict["te_fused_moe_aux_loss_backward_ffi"] = EncapsulateFFI(FusedMoEAuxLossBackwardHandler);

#ifdef NVTE_WITH_NCCL_EP
  // Expert Parallelism (instantiate handler pins NCCL comm to executable lifetime).
  dict["te_ep_prepare_ffi"] =
      pybind11::dict(pybind11::arg("instantiate") = EncapsulateFFI(EpInstantiateHandler),
                     pybind11::arg("execute") = EncapsulateFFI(EpPrepareHandler));
  dict["te_ep_dispatch_ffi"] =
      pybind11::dict(pybind11::arg("instantiate") = EncapsulateFFI(EpInstantiateHandler),
                     pybind11::arg("execute") = EncapsulateFFI(EpDispatchHandler));
  dict["te_ep_combine_ffi"] =
      pybind11::dict(pybind11::arg("instantiate") = EncapsulateFFI(EpInstantiateHandler),
                     pybind11::arg("execute") = EncapsulateFFI(EpCombineHandler));
  dict["te_ep_dispatch_bwd_ffi"] =
      pybind11::dict(pybind11::arg("instantiate") = EncapsulateFFI(EpInstantiateHandler),
                     pybind11::arg("execute") = EncapsulateFFI(EpDispatchBwdHandler));
  dict["te_ep_combine_bwd_ffi"] =
      pybind11::dict(pybind11::arg("instantiate") = EncapsulateFFI(EpInstantiateHandler),
                     pybind11::arg("execute") = EncapsulateFFI(EpCombineBwdHandler));
#endif  // NVTE_WITH_NCCL_EP

  // TopK
  dict["te_topk_ffi"] = EncapsulateFFI(TopkHandler);

  return dict;
}

PYBIND11_MODULE(transformer_engine_jax, m) {
  m.def("registrations", &Registrations);
  m.def("get_fused_attn_backend", &GetFusedAttnBackend);
  m.def("get_cuda_version", &GetCudaRuntimeVersion);
  m.def("get_cudnn_version", &GetCudnnRuntimeVersion);
  m.def("get_cudnn_frontend_version", &GetCudnnFrontendVersion);
  m.def("get_device_compute_capability", &GetDeviceComputeCapability);
  m.def("get_num_compute_streams", &nvte_get_num_compute_streams);
  m.def("get_cublasLt_version", &cublasLtGetVersion);
  m.def("get_dact_dbias_quantize_workspace_sizes", &GetDActDBiasQuantizeWorkspaceSizes);
  m.def("get_dbias_quantize_workspace_sizes", &GetDBiasQuantizeWorkspaceSizes);
  m.def("get_norm_fwd_workspace_sizes", &GetNormForwardWorkspaceSizes);
  m.def("get_norm_bwd_workspace_sizes", &GetNormBackwardWorkspaceSizes);
  m.def("get_fused_attn_fwd_workspace_sizes", &GetFusedAttnForwardWorkspaceSizes);
  m.def("get_fused_attn_bwd_workspace_sizes", &GetFusedAttnBackwardWorkspaceSizes);
  m.def("get_topk_workspace_sizes", &GetTopkWorkspaceSizes);
  m.def("nvte_get_qkv_format", &nvte_get_qkv_format);
  m.def("is_non_nt_fp8_gemm_supported", &nvte_is_non_tn_fp8_gemm_supported);
  m.def("nvte_built_with_cublasmp", &::nvte_built_with_cublasmp);
  m.def("initialize_cgemm_communicator", &InitializeCgemmCommunicator);
  m.def("is_collective_gemm_with_cublasmp", &IsCollectiveGemmWithCublasmp);
  m.def("get_cgemm_num_max_streams", &GetCgemmNumMaxStreams);
  m.def("get_grouped_gemm_setup_workspace_size", &nvte_get_grouped_gemm_setup_workspace_size);
#ifdef NVTE_WITH_NCCL_EP
  m.def("set_ep_bootstrap_params", &SetEpBootstrapParams, pybind11::arg("unique_id_bytes"),
        pybind11::arg("ep_size"), pybind11::arg("rank_within_group"), pybind11::arg("num_experts"),
        pybind11::arg("max_tokens_per_rank"), pybind11::arg("max_recv_tokens_per_rank"),
        pybind11::arg("hidden_dim"), pybind11::arg("max_num_sms"),
        pybind11::arg("max_token_dtype"));
  m.def("release_ep_resources", &ReleaseEpResources);
  m.def("ep_handle_mem_size", &EpHandleMemSize, pybind11::arg("top_k"),
        pybind11::arg("dispatch_output_per_expert_alignment") = 0);
  m.def("get_ep_instance_state_type_id", &GetEpInstanceStateTypeIdCapsule);
  m.def("get_ep_instance_state_type_info", &GetEpInstanceStateTypeInfoCapsule);
#endif  // NVTE_WITH_NCCL_EP

  pybind11::enum_<DType>(m, "DType", pybind11::module_local())
      .value("kByte", DType::kByte)
      .value("kInt32", DType::kInt32)
      .value("kInt64", DType::kInt64)
      .value("kFloat32", DType::kFloat32)
      .value("kFloat16", DType::kFloat16)
      .value("kBFloat16", DType::kBFloat16)
      .value("kFloat8E4M3", DType::kFloat8E4M3)
      .value("kFloat8E5M2", DType::kFloat8E5M2)
      .value("kFloat8E8M0", DType::kFloat8E8M0)
      .value("kFloat4E2M1", DType::kFloat4E2M1);

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
      .value("NVTE_THD_THD_THD", NVTE_QKV_Layout::NVTE_THD_THD_THD)
      .value("NVTE_QKV_Layout_NOT_SET", NVTE_QKV_Layout::NVTE_QKV_Layout_NOT_SET);

  pybind11::enum_<NVTE_QKV_Format>(m, "NVTE_QKV_Format", pybind11::module_local())
      .value("NVTE_SBHD", NVTE_QKV_Format::NVTE_SBHD)
      .value("NVTE_BSHD", NVTE_QKV_Format::NVTE_BSHD)
      .value("NVTE_THD", NVTE_QKV_Format::NVTE_THD)
      .value("NVTE_QKV_Format_NOT_SET", NVTE_QKV_Format::NVTE_QKV_Format_NOT_SET);

  pybind11::enum_<NVTE_Softmax_Type>(m, "NVTE_Softmax_Type", pybind11::module_local())
      .value("NVTE_VANILLA_SOFTMAX", NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX)
      .value("NVTE_OFF_BY_ONE_SOFTMAX", NVTE_Softmax_Type::NVTE_OFF_BY_ONE_SOFTMAX)
      .value("NVTE_LEARNABLE_SOFTMAX", NVTE_Softmax_Type::NVTE_LEARNABLE_SOFTMAX);

  pybind11::enum_<NVTE_Activation_Type>(m, "NVTE_Activation_Type", pybind11::module_local())
      .value("GELU", NVTE_Activation_Type::GELU)
      .value("GEGLU", NVTE_Activation_Type::GEGLU)
      .value("GLU", NVTE_Activation_Type::GLU)
      .value("SILU", NVTE_Activation_Type::SILU)
      .value("SWIGLU", NVTE_Activation_Type::SWIGLU)
      .value("RELU", NVTE_Activation_Type::RELU)
      .value("REGLU", NVTE_Activation_Type::REGLU)
      .value("QGELU", NVTE_Activation_Type::QGELU)
      .value("QGEGLU", NVTE_Activation_Type::QGEGLU)
      .value("SRELU", NVTE_Activation_Type::SRELU)
      .value("SREGLU", NVTE_Activation_Type::SREGLU)
      .value("CLAMPED_SWIGLU", NVTE_Activation_Type::CLAMPED_SWIGLU)
      .export_values();

  pybind11::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", pybind11::module_local())
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend)
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
      .value("NVFP4_1D_SCALING", JAXX_Scaling_Mode::NVFP4_1D_SCALING)
      .value("NVFP4_2D_SCALING", JAXX_Scaling_Mode::NVFP4_2D_SCALING)
      .export_values();

  pybind11::enum_<NVTEScalingMode>(m, "NVTEScalingMode", pybind11::module_local())
      .value("NVTE_DELAYED_TENSOR_SCALING", NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING)
      .value("NVTE_MXFP8_1D_SCALING", NVTEScalingMode::NVTE_MXFP8_1D_SCALING)
      .value("NVTE_BLOCK_SCALING_1D", NVTEScalingMode::NVTE_BLOCK_SCALING_1D)
      .value("NVTE_BLOCK_SCALING_2D", NVTEScalingMode::NVTE_BLOCK_SCALING_2D)
      .value("NVTE_NVFP4_1D_SCALING", NVTEScalingMode::NVTE_NVFP4_1D_SCALING)
      .value("NVTE_INVALID_SCALING", NVTEScalingMode::NVTE_INVALID_SCALING);

  pybind11::enum_<JAXX_Quantize_Layout>(m, "JAXX_Quantize_Layout", pybind11::module_local())
      .value("ROWWISE", JAXX_Quantize_Layout::ROWWISE)
      .value("COLWISE", JAXX_Quantize_Layout::COLWISE)
      .value("ROWWISE_COLWISE", JAXX_Quantize_Layout::ROWWISE_COLWISE)
      .export_values();

  pybind11::enum_<JAXX_Score_Function>(m, "JAXX_Score_Function", pybind11::module_local())
      .value("SIGMOID", JAXX_Score_Function::SIGMOID)
      .value("SOFTMAX", JAXX_Score_Function::SOFTMAX)
      .export_values();

  pybind11::enum_<JAXX_Routing_Map_Format>(m, "JAXX_Routing_Map_Format", pybind11::module_local())
      .value("BYTEMAP", JAXX_Routing_Map_Format::BYTEMAP)
      .value("BITMAP_U8", JAXX_Routing_Map_Format::BITMAP_U8)
      .export_values();

  pybind11::enum_<JAXX_Collective_Op>(m, "JAXX_Collective_Op", pybind11::module_local())
      .value("NONE", JAXX_Collective_Op::NONE)
      .value("ALL_GATHER", JAXX_Collective_Op::ALL_GATHER)
      .value("REDUCE_SCATTER", JAXX_Collective_Op::REDUCE_SCATTER)
      .export_values();
}

}  // namespace jax
}  // namespace transformer_engine
