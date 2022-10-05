/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <torch/extension.h>

#include "ATen/cuda/CUDAContext.h"
#include "softmax.h"

// BMM + Reduction
// at::Tensor bmm_tn(const at::Tensor & A, const at::Tensor & B, at::Tensor &
// C);

// Relu(A) + BMM
void bmm_nt(const at::Tensor &A, const at::Tensor &B, at::Tensor C);
void bmm_nn(const at::Tensor &A, const at::Tensor &B, at::Tensor C);

// Masked Softmax + Dropout_draw/encode

at::Tensor softmax_fwd(const at::Tensor &x,           // BxHxSqxSk
                       const at::Tensor &cu_seqlens,  // B+1
                       const float scale_pre_softmax, const float p_dropout,
                       const softmax::MaskMode mask_mode,
                       c10::optional<at::Generator> gen_);

at::Tensor softmax_bwd(const at::Tensor &dz,          // BxHxSqxSk
                       const at::Tensor &smat_dmask,  // BxHxSqxSk
                       const float scale_pre_softmax, float p_dropout);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA Components for Multi-head Attention";

  m.def("softmax_fwd", &softmax_fwd,
        "CUDA implementation of masked softmax+dropout-draw/encode fwd",
        py::arg("x"), py::arg("cu_seqlens"), py::arg("scale_pre_softmax"),
        py::arg("p_dropout"), py::arg("mask_mode"), py::arg("gen_"));
  m.def("softmax_bwd", &softmax_bwd,
        "CUDA implementation of masked softmax+dropout-draw/encode bwd",
        py::arg("dz"), py::arg("smat_dmask"), py::arg("scale_pre_softmax"),
        py::arg("p_dropout"));

  py::enum_<softmax::MaskMode>(m, "MaskMode")
      .value("SELF", softmax::SELF, "Self-attention Mask")
      .value("CAUSAL", softmax::CAUSAL, "Causal Mask")
      .export_values();

  m.def("relu_bmm_nn", &bmm_nn, "CUDA implementation of bmm(relu(A), B)",
        py::arg("A"), py::arg("B"), py::arg("C"));
  m.def("relu_bmm_nt", &bmm_nt, "CUDA implementation of bmm(relu(A).T, B)",
        py::arg("A"), py::arg("B"), py::arg("C"));

  // m.def("bmm_tn", &bmm_tn, "CUDA implementation of bmm(A,B.T)");
}
