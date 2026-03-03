/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <optional>

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

at::Tensor cumsum(at::Tensor input, std::optional<at::Tensor> out) {
  init_extension();

  // Operate on a contiguous int64 CUDA tensor.
  auto contiguous_input = input.contiguous();
  NVTE_CHECK(contiguous_input.is_cuda(), "Expected input to be on CUDA.");
  NVTE_CHECK(contiguous_input.scalar_type() == at::kLong, "Expected input dtype to be int64.");
  NVTE_CHECK(contiguous_input.dim() == 1, "Expected 1D input tensor.");

  const auto num_elements = static_cast<size_t>(contiguous_input.numel());
  if (!out) {
    out = at::empty({static_cast<int64_t>(num_elements + 1)}, contiguous_input.options());
  }
  NVTE_CHECK(out->is_cuda(), "Expected output to be on CUDA.");
  NVTE_CHECK(out->scalar_type() == at::kLong, "Expected output dtype to be int64.");
  NVTE_CHECK(out->dim() == 1, "Expected 1D output tensor.");
  NVTE_CHECK(static_cast<size_t>(out->numel()) == num_elements + 1, "Expected output length ",
             num_elements + 1, " but got ", out->numel(), ".");
  NVTE_CHECK(out->is_contiguous(), "Expected output to be contiguous.");

  nvte_cumsum(contiguous_input.data_ptr<int64_t>(), out->data_ptr<int64_t>(), num_elements,
              at::cuda::getCurrentCUDAStream());
  return std::move(*out);
}

}  // namespace transformer_engine::pytorch
