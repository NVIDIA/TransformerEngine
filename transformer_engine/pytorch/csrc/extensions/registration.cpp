/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../stable_common.h"

// This file defines the transformer_engine library namespace.
// All other stable ABI files use STABLE_TORCH_LIBRARY_FRAGMENT to add schemas
// and STABLE_TORCH_LIBRARY_IMPL to add implementations.
STABLE_TORCH_LIBRARY(transformer_engine, m) {
  // Softmax ops
  m.def("scaled_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def(
      "scaled_softmax_backward(Tensor output_grad, Tensor softmax_results, float scale_factor) -> "
      "Tensor");
  m.def("scaled_masked_softmax_forward(Tensor input, Tensor mask, float scale_factor) -> Tensor");
  m.def(
      "scaled_masked_softmax_backward(Tensor output_grad, Tensor softmax_results, float "
      "scale_factor) -> Tensor");
  m.def("scaled_upper_triang_masked_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def(
      "scaled_upper_triang_masked_softmax_backward(Tensor output_grads, Tensor softmax_results, "
      "float scale_factor) -> Tensor");
  m.def("scaled_aligned_causal_masked_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def(
      "scaled_aligned_causal_masked_softmax_backward(Tensor output_grad, Tensor softmax_results, "
      "float scale_factor) -> Tensor");
}
