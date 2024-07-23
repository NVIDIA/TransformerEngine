/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <common/util/pybind_helper.h>
#include <pybind11/functional.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "../common.h"
#include "../extensions.h"

namespace py = pybind11;
namespace te_torch = transformer_engine_torch;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Load nvte = py::module_::import("transformer_engine_common") into TE/PyTorch. This makes
  // essential NVTE enums available through `import transformer_engine_torch` without requiring an
  // additional `import transformer_engine_common`.
  NVTE_ADD_COMMON_PYBIND11_BINDINGS(m)

  // Softmax functions
  m.def("scaled_softmax_forward", &scaled_softmax_forward, "Scaled Softmax FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("scaled_softmax_backward", &scaled_softmax_backward, "Scaled Softmax BWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("scaled_masked_softmax_forward", &scaled_masked_softmax_forward,
        "Scaled Masked Softmax FWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_masked_softmax_backward", &scaled_masked_softmax_backward,
        "Scaled Masked Softmax BWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_upper_triang_masked_softmax_forward", &scaled_upper_triang_masked_softmax_forward,
        "Scaled Upper-Triangular Masked Softmax FWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_upper_triang_masked_softmax_backward", &scaled_upper_triang_masked_softmax_backward,
        "Scaled Upper-Triangular Masked Softmax BWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_aligned_causal_masked_softmax_forward",
        &scaled_aligned_causal_masked_softmax_forward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("scaled_aligned_causal_masked_softmax_backward",
        &scaled_aligned_causal_masked_softmax_backward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax BWD",
        py::call_guard<py::gil_scoped_release>());

  // Other granular functions
  m.def("layernorm_fwd_fp8", &layernorm_fwd_fp8, "LN FWD FP8",
        py::call_guard<py::gil_scoped_release>(), py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("eps"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"),
        py::arg("otype"), py::arg("sm_margin"), py::arg("zero_centered_gamma"),
        py::arg("scale_offset") = 0, py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("layernorm_fwd_fp8_noalloc", &layernorm_fwd_fp8_noalloc, "LN FWD FP8",
        py::call_guard<py::gil_scoped_release>(), py::arg("input"), py::arg("weight"),
        py::arg("bias"), py::arg("eps"), py::arg("scale"), py::arg("ln_out"), py::arg("amax"),
        py::arg("scale_inv"), py::arg("otype"), py::arg("sm_margin"),
        py::arg("zero_centered_gamma"), py::arg("scale_offset") = 0, py::arg("amax_offset") = 0,
        py::arg("scale_inv_offset") = 0);
  m.def("layernorm_bwd", &layernorm_bwd, "LN BWD", py::call_guard<py::gil_scoped_release>());
  m.def("layernorm_fwd", &layernorm_fwd, "LN FWD", py::call_guard<py::gil_scoped_release>());
  m.def("layernorm_fwd_noalloc", &layernorm_fwd_noalloc, "LN FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("rmsnorm_fwd_fp8", &rmsnorm_fwd_fp8, "RMSNorm FWD FP8",
        py::call_guard<py::gil_scoped_release>(), py::arg("input"), py::arg("weight"),
        py::arg("eps"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"), py::arg("otype"),
        py::arg("sm_margin"), py::arg("zero_centered_gamma"), py::arg("scale_offset") = 0,
        py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("rmsnorm_fwd_fp8_noalloc", &rmsnorm_fwd_fp8_noalloc, "RMSNorm FWD FP8",
        py::call_guard<py::gil_scoped_release>(), py::arg("input"), py::arg("weight"),
        py::arg("eps"), py::arg("scale"), py::arg("ln_out"), py::arg("amax"), py::arg("scale_inv"),
        py::arg("otype"), py::arg("sm_margin"), py::arg("zero_centered_gamma"),
        py::arg("scale_offset") = 0, py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("rmsnorm_bwd", &rmsnorm_bwd, "RMSNorm BWD", py::call_guard<py::gil_scoped_release>());
  m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm FWD", py::call_guard<py::gil_scoped_release>());
  m.def("rmsnorm_fwd_noalloc", &rmsnorm_fwd_noalloc, "RMSNorm FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_cast_transpose", &fused_cast_transpose, "Fused Cast + Transpose",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_cast_transpose_noop", &fused_cast_transpose_noop,
        "Cast + Transpose with noop option", py::call_guard<py::gil_scoped_release>(),
        py::arg("input"), py::arg("noop"), py::arg("scale"), py::arg("amax"), py::arg("scale_inv"),
        py::arg("input_cast"), py::arg("input_transpose"), py::arg("otype"),
        py::arg("scale_offset") = 0, py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_cast_transpose_bgrad", &fused_cast_transpose_bgrad, "Fused Cast + Transpose + BGRAD",
        py::call_guard<py::gil_scoped_release>(), py::arg("grad_output"), py::arg("scale"),
        py::arg("amax"), py::arg("scale_inv"), py::arg("otype"), py::arg("scale_offset") = 0,
        py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_fp8_transpose_bgrad", &fused_fp8_transpose_bgrad, "Fused FP8 Transpose + BGRAD",
        py::call_guard<py::gil_scoped_release>(), py::arg("grad_output"), py::arg("scale"),
        py::arg("amax"), py::arg("scale_inv"), py::arg("otype"), py::arg("grad_bias_type"),
        py::arg("scale_offset") = 0, py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_cast_transpose_bgrad_dgelu", &fused_cast_transpose_bgrad_dgelu,
        "Fused Cast + Transpose + BGRAD + DGELU", py::call_guard<py::gil_scoped_release>(),
        py::arg("grad_output"), py::arg("gelu_input"), py::arg("scale"), py::arg("amax"),
        py::arg("scale_inv"), py::arg("otype"), py::arg("scale_offset") = 0,
        py::arg("amax_offset") = 0, py::arg("scale_inv_offset") = 0);
  m.def("fused_multi_cast_transpose", &fused_multi_cast_transpose,
        "Fused Multi-tensor Cast + Transpose", py::call_guard<py::gil_scoped_release>());
  m.def("cast_to_fp8", &cast_to_fp8, "Cast to FP8", py::call_guard<py::gil_scoped_release>());
  m.def("cast_to_fp8_noalloc", &cast_to_fp8_noalloc, "Cast to FP8",
        py::call_guard<py::gil_scoped_release>());
  m.def("cast_from_fp8", &cast_from_fp8, "Cast from FP8", py::call_guard<py::gil_scoped_release>());
  m.def("te_gemm", &te_gemm, "CublasLt GEMM");  /// TODO Think
  m.def("te_grouped_gemm", &te_grouped_gemm, "Grouped GEMM");
  m.def("fused_attn_fwd_qkvpacked", &fused_attn_fwd_qkvpacked,
        "Fused Attention FP8/BF16/FP16 FWD with packed QKV",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_attn_bwd_qkvpacked", &fused_attn_bwd_qkvpacked,
        "Fused Attention FP8/BF16/FP16 BWD with packed QKV",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_attn_fwd_kvpacked", &fused_attn_fwd_kvpacked,
        "Fused Attention FP8/BF16/FP16 FWD with packed KV",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_attn_bwd_kvpacked", &fused_attn_bwd_kvpacked,
        "Fused Attention FP8/BF16/FP16 BWD with packed KV",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_attn_fwd", &fused_attn_fwd,
        "Fused Attention FP8/BF16/FP16 FWD with separate Q, K and V",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_attn_bwd", &fused_attn_bwd,
        "Fused Attention FP8/BF16/FP16 BWD with separate Q, K and V",
        py::call_guard<py::gil_scoped_release>());
  m.def("fp8_transpose", &fp8_transpose, "Transpose with FP8 I/O",
        py::call_guard<py::gil_scoped_release>());
  m.def("fp8_transpose_noalloc", &fp8_transpose_noalloc, "Transpose with FP8 I/O",
        py::call_guard<py::gil_scoped_release>());
  m.def("fp8_transpose_noalloc_noop", &fp8_transpose_noalloc_noop,
        "Transpose with FP8 I/O with noop option.", py::call_guard<py::gil_scoped_release>());
  m.def("gelu", &gelu, "GeLU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("relu", &relu, "ReLU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("geglu", &geglu, "GeGLU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("reglu", &reglu, "ReGLU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("swiglu", &swiglu, "SwiGLU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("qgelu", &qgelu, "QuickGELU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("srelu", &srelu, "Squared ReLU with FP8 output", py::call_guard<py::gil_scoped_release>());
  m.def("dgelu", &dgelu, "Backward of GeLU", py::call_guard<py::gil_scoped_release>());
  m.def("drelu", &drelu, "Backward of ReLU", py::call_guard<py::gil_scoped_release>());
  m.def("dgeglu", &dgeglu, "Backward of GeGLU", py::call_guard<py::gil_scoped_release>());
  m.def("dreglu", &dreglu, "Backward of ReGLU", py::call_guard<py::gil_scoped_release>());
  m.def("dswiglu", &dswiglu, "Backward of SwiGLU", py::call_guard<py::gil_scoped_release>());
  m.def("dqgelu", &dqgelu, "Backward of QuickGELU", py::call_guard<py::gil_scoped_release>());
  m.def("dsrelu", &dsrelu, "Backward of Squared ReLU", py::call_guard<py::gil_scoped_release>());
  m.def("fa_prepare_fwd", &fa_prepare_fwd, "Prepare QKV for Flash Attention",
        py::call_guard<py::gil_scoped_release>());
  m.def("fa_prepare_bwd", &fa_prepare_bwd, "Backward of QKV preparation for Flash Attention",
        py::call_guard<py::gil_scoped_release>());
  m.def("get_fused_attn_backend", &get_fused_attn_backend, "Get Fused Attention backend",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_amax_and_scale_update_after_reduction", &fused_amax_and_scale_update_after_reduction,
        "Update amax history and FP8 scale/scale_inv after reduction",
        py::call_guard<py::gil_scoped_release>());

  // fused apply rope
  m.def("fused_rope_forward", &fused_rope_forward, "Fused Apply RoPE FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_rope_backward", &fused_rope_backward, "Fused Apply RoPE BWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_rope_thd_forward", &fused_rope_thd_forward, "Fused Apply RoPE FWD for thd format",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_rope_thd_backward", &fused_rope_thd_backward, "Fused Apply RoPE BWD for thd format",
        py::call_guard<py::gil_scoped_release>());

  // Misc
  m.def("get_cublasLt_version", &get_cublasLt_version, "Get cublasLt version",
        py::call_guard<py::gil_scoped_release>());
  m.def("get_cudnn_version", &get_cudnn_version, "Get cuDNN version",
        py::call_guard<py::gil_scoped_release>());

  // Support THD format for Context Parallel
  m.def("thd_read_half_tensor", &thd_read_half_tensor,
        "Read the first half(half_idx=0) or the second half(half_idx=1) of each sequence in a THD "
        "tensor",
        py::call_guard<py::gil_scoped_release>());
  m.def("thd_second_half_lse_correction", &thd_second_half_lse_correction,
        "Correct the second half of the softmax_lse", py::call_guard<py::gil_scoped_release>());
  m.def("thd_read_second_half_lse", &thd_read_second_half_lse,
        "Read the second half of the softmax_lse", py::call_guard<py::gil_scoped_release>());
  m.def("thd_out_correction", &thd_out_correction,
        "Correct the THD format output of context parallelism in forward pass",
        py::call_guard<py::gil_scoped_release>());
  m.def("thd_grad_correction", &thd_grad_correction,
        "Correct the THD format gradients of context parallelism in backward pass",
        py::call_guard<py::gil_scoped_release>());
  m.def("thd_get_partitioned_indices", &thd_get_partitioned_indices,
        "Generate partitioned indices for inputs in THD format",
        py::call_guard<py::gil_scoped_release>());

  // multi-tensor functions
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_unscale_l2norm", &multi_tensor_unscale_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors after unscaling (unscaling is only "
        "performed for L2 norm computation, and tensors are not updated)",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam_capturable", &multi_tensor_adam_capturable_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support and LR scheduling",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam_capturable_master", &multi_tensor_adam_capturable_master_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support, LR scheduling and FP32 master weights",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors",
        py::call_guard<py::gil_scoped_release>());

  // Data structures
  py::class_<te_torch::FP8TensorMeta>(m, "FP8TensorMeta", py::module_local())
      .def(py::init<>())
      .def_readwrite("scale", &te_torch::FP8TensorMeta::scale)
      .def_readwrite("scale_inv", &te_torch::FP8TensorMeta::scale_inv)
      .def_readwrite("amax_history", &te_torch::FP8TensorMeta::amax_history);

  py::enum_<te_torch::FP8FwdTensors>(m, "FP8FwdTensors", py::module_local())
      .value("GEMM1_INPUT", te_torch::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", te_torch::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", te_torch::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", te_torch::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", te_torch::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", te_torch::FP8FwdTensors::GEMM2_OUTPUT)
      .value("GEMM3_INPUT", te_torch::FP8FwdTensors::GEMM3_INPUT)
      .value("GEMM3_WEIGHT", te_torch::FP8FwdTensors::GEMM3_WEIGHT)
      .value("GEMM3_OUTPUT", te_torch::FP8FwdTensors::GEMM3_OUTPUT)
      .export_values();

  py::enum_<te_torch::FP8BwdTensors>(m, "FP8BwdTensors", py::module_local())
      .value("GRAD_OUTPUT1", te_torch::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", te_torch::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", te_torch::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", te_torch::FP8BwdTensors::GRAD_INPUT2)
      .value("GRAD_OUTPUT3", te_torch::FP8BwdTensors::GRAD_OUTPUT3)
      .value("GRAD_INPUT3", te_torch::FP8BwdTensors::GRAD_INPUT3)
      .export_values();

  // Comm+GEMM Overlap
  m.def("set_comm_overlap_callbacks", &set_comm_overlap_callbacks);

  py::class_<te_torch::CommGemmOverlap>(m, "CommGemmOverlap", py::module_local())
      .def(py::init</* sample_tensor */ torch::Tensor &, /* world_rank */ int, /* world_size */ int,
                    /* local_rank */ int, /* local_size */ int, /* node_id */ int,
                    /* num_nodes */ int, /* tp_size */ int, /* num_splits */ int,
                    /* num_max_streams */ int, /* cga_size */ int, /* num_comm_sm */ int,
                    /* set_sm_margin */ bool, /* use_ce */ bool, /* atomic_gemm */ bool>())
      .def("bulk_overlap", &te_torch::CommGemmOverlap::bulk_overlap,
           py::call_guard<py::gil_scoped_release>())
      .def("split_overlap_rs", &te_torch::CommGemmOverlap::split_overlap_rs,
           py::call_guard<py::gil_scoped_release>())
      .def("atomic_gemm_overlap_rs", &te_torch::CommGemmOverlap::atomic_gemm_overlap_rs,
           py::call_guard<py::gil_scoped_release>())
      .def("copy_input_to_ubuf", &te_torch::CommGemmOverlap::copy_input_to_ubuf,
           py::call_guard<py::gil_scoped_release>())
      .def("get_ubuf_output", &te_torch::CommGemmOverlap::get_ubuf_output,
           py::call_guard<py::gil_scoped_release>())
      .def("set_ubuf_scale_inv", &te_torch::CommGemmOverlap::set_ubuf_scale_inv,
           py::call_guard<py::gil_scoped_release>())
      .def("is_fp8_ubuf", &te_torch::CommGemmOverlap::is_fp8_ubuf,
           py::call_guard<py::gil_scoped_release>())
      .def("is_atomic_gemm", &te_torch::CommGemmOverlap::is_atomic_gemm,
           py::call_guard<py::gil_scoped_release>())
      .def("is_p2p_overlap", &te_torch::CommGemmOverlap::is_p2p_overlap,
           py::call_guard<py::gil_scoped_release>());

  py::class_<te_torch::CommGemmOverlapP2P>(m, "CommGemmOverlapP2P", py::module_local())
      .def(py::init</* sample_tensor */ torch::Tensor &, /* world_rank */ int, /* world_size */ int,
                    /* local_rank */ int, /* local_size */ int, /* node_id */ int,
                    /* num_nodes */ int, /* tp_size */ int, /* num_max_streams */ int,
                    /* cga_size */ int, /* num_comm_sm */ int, /* set_sm_margin */ bool,
                    /* use_ce */ bool, /* atomic_gemm */ bool, /* aggregate */ bool,
                    /* is_reduce_scatter */ bool>())
      .def("split_overlap_ag_p2p", &te_torch::CommGemmOverlapP2P::split_overlap_ag,
           py::call_guard<py::gil_scoped_release>())
      .def("split_overlap_rs_p2p", &te_torch::CommGemmOverlapP2P::split_overlap_rs,
           py::call_guard<py::gil_scoped_release>())
      .def("atomic_gemm_overlap_ag_p2p", &te_torch::CommGemmOverlapP2P::atomic_gemm_overlap_ag,
           py::call_guard<py::gil_scoped_release>())
      .def("atomic_gemm_overlap_rs_p2p", &te_torch::CommGemmOverlapP2P::atomic_gemm_overlap_rs,
           py::call_guard<py::gil_scoped_release>())
      .def("copy_input_to_ubuf", &te_torch::CommGemmOverlapP2P::copy_input_to_ubuf,
           py::call_guard<py::gil_scoped_release>())
      .def("get_ubuf_output", &te_torch::CommGemmOverlapP2P::get_ubuf_output,
           py::call_guard<py::gil_scoped_release>())
      .def("set_ubuf_scale_inv", &te_torch::CommGemmOverlapP2P::set_ubuf_scale_inv,
           py::call_guard<py::gil_scoped_release>())
      .def("is_fp8_ubuf", &te_torch::CommGemmOverlapP2P::is_fp8_ubuf,
           py::call_guard<py::gil_scoped_release>())
      .def("is_atomic_gemm", &te_torch::CommGemmOverlapP2P::is_atomic_gemm,
           py::call_guard<py::gil_scoped_release>())
      .def("is_p2p_overlap", &te_torch::CommGemmOverlapP2P::is_p2p_overlap,
           py::call_guard<py::gil_scoped_release>());
}  // PYBIND11_MODULE
