/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/functional.h>

#include "../comm_gemm_overlap.h"
#include "../extensions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Softmax functions
  m.def("scaled_softmax_forward", &scaled_softmax_forward, "Scaled Softmax FWD");
  m.def("scaled_softmax_backward", &scaled_softmax_backward, "Scaled Softmax BWD");
  m.def("scaled_masked_softmax_forward", &scaled_masked_softmax_forward,
        "Scaled Masked Softmax FWD");
  m.def("scaled_masked_softmax_backward", &scaled_masked_softmax_backward,
        "Scaled Masked Softmax BWD");
  m.def("scaled_upper_triang_masked_softmax_forward", &scaled_upper_triang_masked_softmax_forward,
        "Scaled Upper-Triangular Masked Softmax FWD");
  m.def("scaled_upper_triang_masked_softmax_backward", &scaled_upper_triang_masked_softmax_backward,
        "Scaled Upper-Triangular Masked Softmax BWD");
  m.def("scaled_aligned_causal_masked_softmax_forward",
        &scaled_aligned_causal_masked_softmax_forward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax FWD");
  m.def("scaled_aligned_causal_masked_softmax_backward",
        &scaled_aligned_causal_masked_softmax_backward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax BWD");

  // Other granular functions
  m.def("layernorm_fwd_fp8", &layernorm_fwd_fp8, "LN FWD FP8", py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("eps"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"),
        py::arg("otype"), py::arg("sm_margin"), py::arg("zero_centered_gamma"),
        py::arg("scale_offset") = 0, py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("layernorm_fwd_fp8_noalloc", &layernorm_fwd_fp8_noalloc, "LN FWD FP8", py::arg("input"),
        py::arg("weight"), py::arg("bias"), py::arg("eps"), py::arg("scale"), py::arg("ln_out"),
        py::arg("amax"), py::arg("scale_inv"), py::arg("otype"), py::arg("sm_margin"),
        py::arg("zero_centered_gamma"), py::arg("scale_offset") = 0, py::arg("amax_offset") = 0,
        py::arg("scale_inv_offset") = 0);
  m.def("layernorm_bwd", &layernorm_bwd, "LN BWD");
  m.def("layernorm_fwd", &layernorm_fwd, "LN FWD");
  m.def("layernorm_fwd_noalloc", &layernorm_fwd_noalloc, "LN FWD");
  m.def("rmsnorm_fwd_fp8", &rmsnorm_fwd_fp8, "RMSNorm FWD FP8", py::arg("input"), py::arg("weight"),
        py::arg("eps"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"), py::arg("otype"),
        py::arg("sm_margin"), py::arg("zero_centered_gamma"), py::arg("scale_offset") = 0,
        py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("rmsnorm_fwd_fp8_noalloc", &rmsnorm_fwd_fp8_noalloc, "RMSNorm FWD FP8", py::arg("input"),
        py::arg("weight"), py::arg("eps"), py::arg("scale"), py::arg("ln_out"), py::arg("amax"),
        py::arg("scale_inv"), py::arg("otype"), py::arg("sm_margin"),
        py::arg("zero_centered_gamma"), py::arg("scale_offset") = 0, py::arg("amax_offset") = 0,
        py::arg("scale_inv_offset") = 0);
  m.def("rmsnorm_bwd", &rmsnorm_bwd, "RMSNorm BWD");
  m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm FWD");
  m.def("rmsnorm_fwd_noalloc", &rmsnorm_fwd_noalloc, "RMSNorm FWD");
  m.def("fused_cast_transpose", &fused_cast_transpose, "Fused Cast + Transpose");
  m.def("fused_cast_transpose_noop", &fused_cast_transpose_noop,
        "Fused Cast + Transpose with noop option", py::arg("input"), py::arg("noop"),
        py::arg("scale"), py::arg("amax"), py::arg("scale_inv"), py::arg("input_cast"),
        py::arg("input_transpose"), py::arg("otype"), py::arg("scale_offset") = 0,
        py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_cast_transpose_bgrad", &fused_cast_transpose_bgrad, "Fused Cast + Transpose + BGRAD",
        py::arg("grad_output"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"),
        py::arg("otype"), py::arg("scale_offset") = 0, py::arg("amax_offset") = 0,
        py::arg("scale_inv_offset") = 0);
  m.def("fused_fp8_transpose_bgrad", &fused_fp8_transpose_bgrad, "Fused FP8 Transpose + BGRAD",
        py::arg("grad_output"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"),
        py::arg("otype"), py::arg("grad_bias_type"), py::arg("scale_offset") = 0,
        py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_cast_transpose_bgrad_dgelu", &fused_cast_transpose_bgrad_dgelu,
        "Fused Cast + Transpose + BGRAD + DGELU", py::arg("grad_output"), py::arg("gelu_input"),
        py::arg("scale"), py::arg("amax"), py::arg("scale_inv"), py::arg("otype"),
        py::arg("scale_offset") = 0, py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_multi_cast_transpose", &fused_multi_cast_transpose,
        "Fused Multi-tensor Cast + Transpose");
  m.def("cast_to_fp8", &cast_to_fp8, "Cast to FP8");
  m.def("cast_to_fp8_noalloc", &cast_to_fp8_noalloc, "Cast to FP8");
  m.def("cast_from_fp8", &cast_from_fp8, "Cast from FP8");
  m.def("te_gemm", &te_gemm, "CublasLt GEMM");
  m.def("fused_attn_fwd_qkvpacked", &fused_attn_fwd_qkvpacked,
        "Fused Attention FP8/BF16/FP16 FWD with packed QKV");
  m.def("fused_attn_bwd_qkvpacked", &fused_attn_bwd_qkvpacked,
        "Fused Attention FP8/BF16/FP16 BWD with packed QKV");
  m.def("fused_attn_fwd_kvpacked", &fused_attn_fwd_kvpacked,
        "Fused Attention FP8/BF16/FP16 FWD with packed KV");
  m.def("fused_attn_bwd_kvpacked", &fused_attn_bwd_kvpacked,
        "Fused Attention FP8/BF16/FP16 BWD with packed KV");
  m.def("fused_attn_fwd", &fused_attn_fwd,
        "Fused Attention FP8/BF16/FP16 FWD with separate Q, K and V");
  m.def("fused_attn_bwd", &fused_attn_bwd,
        "Fused Attention FP8/BF16/FP16 BWD with separate Q, K and V");
  m.def("fp8_transpose", &fp8_transpose, "Transpose with FP8 I/O");
  m.def("fp8_transpose_noalloc", &fp8_transpose_noalloc, "Transpose with FP8 I/O");
  m.def("fp8_transpose_noalloc_noop", &fp8_transpose_noalloc_noop,
        "Transpose with FP8 I/O with noop option.");
  m.def("gelu", &gelu, "GeLU with FP8 output");
  m.def("relu", &relu, "ReLU with FP8 output");
  m.def("geglu", &geglu, "GeGLU with FP8 output");
  m.def("reglu", &reglu, "ReGLU with FP8 output");
  m.def("swiglu", &swiglu, "SwiGLU with FP8 output");
  m.def("qgelu", &qgelu, "QuickGELU with FP8 output");
  m.def("srelu", &srelu, "Squared ReLU with FP8 output");
  m.def("dgelu", &dgelu, "Backward of GeLU");
  m.def("drelu", &drelu, "Backward of ReLU");
  m.def("dgeglu", &dgeglu, "Backward of GeGLU");
  m.def("dreglu", &dreglu, "Backward of ReGLU");
  m.def("dswiglu", &dswiglu, "Backward of SwiGLU");
  m.def("dqgelu", &dqgelu, "Backward of QuickGELU");
  m.def("dsrelu", &dsrelu, "Backward of Squared ReLU");
  m.def("fa_prepare_fwd", &fa_prepare_fwd, "Prepare QKV for Flash Attention");
  m.def("fa_prepare_bwd", &fa_prepare_bwd, "Backward of QKV preparation for Flash Attention");
  m.def("get_fused_attn_backend", &get_fused_attn_backend, "Get Fused Attention backend");
  m.def("fused_amax_and_scale_update_after_reduction", &fused_amax_and_scale_update_after_reduction,
        "Update amax history and FP8 scale/scale_inv after reduction");

  // fused apply rope
  m.def("fused_rope_forward", &fused_rope_forward, "Fused Apply RoPE FWD");
  m.def("fused_rope_backward", &fused_rope_backward, "Fused Apply RoPE BWD");
  m.def("fused_rope_thd_forward", &fused_rope_thd_forward, "Fused Apply RoPE FWD for thd format");
  m.def("fused_rope_thd_backward", &fused_rope_thd_backward, "Fused Apply RoPE BWD for thd format");

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version");
  m.def("get_cudnn_version", &get_cudnn_version, "Get cuDNN version");

  // Support THD format for Context Parallel
  m.def("thd_read_half_tensor", &thd_read_half_tensor,
        "Read the first half(half_idx=0) or the second half(half_idx=1) of each sequence in a THD "
        "tensor");
  m.def("thd_second_half_lse_correction", &thd_second_half_lse_correction,
        "Correct the second half of the softmax_lse");
  m.def("thd_read_second_half_lse", &thd_read_second_half_lse,
        "Read the second half of the softmax_lse");
  m.def("thd_out_correction", &thd_out_correction,
        "Correct the THD format output of context parallelism in forward pass");
  m.def("thd_grad_correction", &thd_grad_correction,
        "Correct the THD format gradients of context parallelism in backward pass");
  m.def("thd_get_partitioned_indices", &thd_get_partitioned_indices,
        "Generate partitioned indices for inputs in THD format");

  // multi-tensor functions
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_unscale_l2norm", &multi_tensor_unscale_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors after unscaling (unscaling is only "
        "performed for L2 norm computation, and tensors are not updated)");
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_adam_capturable", &multi_tensor_adam_capturable_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support and LR scheduling");
  m.def("multi_tensor_adam_capturable_master", &multi_tensor_adam_capturable_master_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support, LR scheduling and FP32 master weights");
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors");

  // Data structures
  py::class_<transformer_engine::FP8TensorMeta>(m, "FP8TensorMeta")
      .def(py::init<>())
      .def_readwrite("scale", &transformer_engine::FP8TensorMeta::scale)
      .def_readwrite("scale_inv", &transformer_engine::FP8TensorMeta::scale_inv)
      .def_readwrite("amax_history", &transformer_engine::FP8TensorMeta::amax_history);

  // comm+GEMM overlap w/ userbuffers
  m.def("set_ubuf_bootstrap_callbacks", &ubuf::set_ubuf_bootstrap_callbacks);

  py::enum_<ubuf::UBOverlapAlgo>(m, "UbufOverlapAlgo")
      .value("BULK_OVERLAP_AG", ubuf::UBOverlapAlgo::BULK_OVERLAP_AG)
      .value("BULK_OVERLAP_RS", ubuf::UBOverlapAlgo::BULK_OVERLAP_RS)
      .value("SPLIT_PIPELINED_RS", ubuf::UBOverlapAlgo::SPLIT_PIPELINED_RS)
      .value("SPLIT_PIPELINED_RS_P2P", ubuf::UBOverlapAlgo::SPLIT_PIPELINED_RS_P2P)
      .value("SPLIT_PIPELINED_AG_P2P", ubuf::UBOverlapAlgo::SPLIT_PIPELINED_AG_P2P)
      .value("ATOMIC_GEMM_RS", ubuf::UBOverlapAlgo::ATOMIC_GEMM_RS)
      .value("ATOMIC_GEMM_AG_P2P", ubuf::UBOverlapAlgo::ATOMIC_GEMM_AG_P2P)
      .value("ATOMIC_GEMM_RS_P2P", ubuf::UBOverlapAlgo::ATOMIC_GEMM_RS_P2P);

  py::class_<ubuf::UbufCommOverlap>(m, "UbufCommOverlap")
      .def(py::init<torch::Tensor&, int, int, int, int, int, int, int, bool, int, bool,
                    torch::Tensor>())
      .def("bulk_overlap", &ubuf::UbufCommOverlap::bulk_overlap)
      .def("split_overlap_rs", &ubuf::UbufCommOverlap::split_overlap_rs)
      .def("set_ubuf_scale_inv", &ubuf::UbufCommOverlap::set_ubuf_scale_inv)
      .def("atomic_gemm_overlap_rs", &ubuf::UbufCommOverlap::atomic_gemm_overlap_rs)
      .def("is_fp8_ubuf", &ubuf::UbufCommOverlap::is_fp8_ubuf)
      .def("copy_input_to_ubuf", &ubuf::UbufCommOverlap::copy_input_to_ubuf)
      .def("get_ubuf_output", &ubuf::UbufCommOverlap::get_ubuf_output)
      .def("is_atomic_gemm", &ubuf::UbufCommOverlap::is_atomic_gemm)
      .def("is_p2p_overlap", &ubuf::UbufCommOverlap::is_p2p_overlap);

  py::class_<ubuf::UbufP2PCommOverlap>(m, "UbufP2PCommOverlap")
      .def(py::init<torch::Tensor&, int, int, int, int, int, int, bool, bool, int, bool, bool, bool,
                    torch::Tensor>())
      .def("split_overlap_ag_p2p", &ubuf::UbufP2PCommOverlap::split_overlap_ag)
      .def("split_overlap_rs_p2p", &ubuf::UbufP2PCommOverlap::split_overlap_rs)
      .def("atomic_gemm_overlap_ag_p2p", &ubuf::UbufP2PCommOverlap::atomic_gemm_overlap_ag)
      .def("atomic_gemm_overlap_rs_p2p", &ubuf::UbufP2PCommOverlap::atomic_gemm_overlap_rs)
      .def("copy_input_to_ubuf", &ubuf::UbufP2PCommOverlap::copy_input_to_ubuf)
      .def("get_ubuf_output", &ubuf::UbufP2PCommOverlap::get_ubuf_output)
      .def("is_fp8_ubuf", &ubuf::UbufP2PCommOverlap::is_fp8_ubuf)
      .def("is_atomic_gemm", &ubuf::UbufP2PCommOverlap::is_atomic_gemm)
      .def("is_p2p_overlap", &ubuf::UbufP2PCommOverlap::is_p2p_overlap)
      .def("set_ubuf_scale_inv", &ubuf::UbufP2PCommOverlap::set_ubuf_scale_inv);

  py::enum_<transformer_engine::DType>(m, "DType", py::module_local())
      .value("kByte", transformer_engine::DType::kByte)
      .value("kInt32", transformer_engine::DType::kInt32)
      .value("kFloat32", transformer_engine::DType::kFloat32)
      .value("kFloat16", transformer_engine::DType::kFloat16)
      .value("kBFloat16", transformer_engine::DType::kBFloat16)
      .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)
      .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", transformer_engine::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", transformer_engine::FP8FwdTensors::GEMM2_OUTPUT)
      .value("GEMM3_INPUT", transformer_engine::FP8FwdTensors::GEMM3_INPUT)
      .value("GEMM3_WEIGHT", transformer_engine::FP8FwdTensors::GEMM3_WEIGHT)
      .value("GEMM3_OUTPUT", transformer_engine::FP8FwdTensors::GEMM3_OUTPUT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", transformer_engine::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", transformer_engine::FP8BwdTensors::GRAD_INPUT2)
      .value("GRAD_OUTPUT3", transformer_engine::FP8BwdTensors::GRAD_OUTPUT3)
      .value("GRAD_INPUT3", transformer_engine::FP8BwdTensors::GRAD_INPUT3);

  py::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type")
      .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)
      .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)
      .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)
      .value("NVTE_ALIBI", NVTE_Bias_Type::NVTE_ALIBI);

  py::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type")
      .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)
      .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)
      .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK)
      .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK);

  py::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout")
      .value("NVTE_SB3HD", NVTE_QKV_Layout::NVTE_SB3HD)
      .value("NVTE_SBH3D", NVTE_QKV_Layout::NVTE_SBH3D)
      .value("NVTE_SBHD_SB2HD", NVTE_QKV_Layout::NVTE_SBHD_SB2HD)
      .value("NVTE_SBHD_SBH2D", NVTE_QKV_Layout::NVTE_SBHD_SBH2D)
      .value("NVTE_SBHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD)
      .value("NVTE_BS3HD", NVTE_QKV_Layout::NVTE_BS3HD)
      .value("NVTE_BSH3D", NVTE_QKV_Layout::NVTE_BSH3D)
      .value("NVTE_BSHD_BS2HD", NVTE_QKV_Layout::NVTE_BSHD_BS2HD)
      .value("NVTE_BSHD_BSH2D", NVTE_QKV_Layout::NVTE_BSHD_BSH2D)
      .value("NVTE_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD)
      .value("NVTE_T3HD", NVTE_QKV_Layout::NVTE_T3HD)
      .value("NVTE_TH3D", NVTE_QKV_Layout::NVTE_TH3D)
      .value("NVTE_THD_T2HD", NVTE_QKV_Layout::NVTE_THD_T2HD)
      .value("NVTE_THD_TH2D", NVTE_QKV_Layout::NVTE_THD_TH2D)
      .value("NVTE_THD_THD_THD", NVTE_QKV_Layout::NVTE_THD_THD_THD);

  py::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend")
      .value("NVTE_F16_max512_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_max512_seqlen)
      .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)
      .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8)
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend);
}
