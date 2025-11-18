/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"

namespace transformer_engine::pytorch {

void nvfp4_2d_compute_partial_amax(const at::Tensor &tensor, at::Tensor amax, size_t h, size_t w,
                                   size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  TORCH_CHECK(amax.scalar_type() == at::ScalarType::Float, "amax must be a float tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float ||
                  tensor.scalar_type() == at::ScalarType::BFloat16,
              "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(tensor.contiguous());
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_nvfp4_2d_compute_partial_amax(
      tensor_cu.data(), amax_cu.data(), h, w, amax.stride(0), amax.stride(1), start_offset,
      block_len, at::cuda::getCurrentCUDAStream());
}

void nvfp4_2d_partial_cast(const at::Tensor &inp, py::handle out, const at::Tensor &scale,
                           size_t h, size_t w, size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 16, "Currently only block_len = 16 is supported for NVFP4 2D");
  TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
  TORCH_CHECK(inp.scalar_type() == at::ScalarType::Float ||
                  inp.scalar_type() == at::ScalarType::BFloat16,
              "input must be a float or bfloat16 tensor");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(inp.contiguous());
  const TensorWrapper out_cu = makeTransformerEngineTensor(out, py::none());
  const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);

  nvte_nvfp4_2d_partial_cast(inp_cu.data(), out_cu.data(), scale_cu.data(), h, w, scale.stride(0),
                             scale.stride(1), start_offset, block_len,
                             at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch


