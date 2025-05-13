/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

namespace transformer_engine::pytorch {

void fp8_block_scaling_compute_partial_amax(const at::Tensor &tensor, at::Tensor amax, size_t h,
                                            size_t w, size_t start_offset, size_t block_len) {
  TORCH_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  TORCH_CHECK(amax.dim() == 2, "amax must be a 2D tensor");
  TORCH_CHECK(amax.scalar_type() == at::ScalarType::Float, "amax must be a float tensor");
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Float ||
                  tensor.scalar_type() == at::ScalarType::BFloat16,
              "tensor must be a float or bfloat16 tensor");

  const TensorWrapper tensor_cu = makeTransformerEngineTensor(tensor);
  TensorWrapper amax_cu = makeTransformerEngineTensor(amax);

  nvte_fp8_block_scaling_compute_partial_amax(tensor_cu.data(), amax_cu.data(), h, w,
                                              amax.stride(0), amax.stride(1), start_offset,
                                              block_len, at::cuda::getCurrentCUDAStream());
}

void fp8_block_scaling_partial_cast(const at::Tensor &inp, at::Tensor out, const at::Tensor &scale,
                                    size_t h, size_t w, size_t start_offset, size_t block_len,
                                    const transformer_engine::DType out_dtype) {
  TORCH_CHECK(block_len == 128, "Currently only block_len = 128 is supported");
  TORCH_CHECK(scale.dim() == 2, "scale must be a 2D tensor");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float, "scale must be a float tensor");
  TORCH_CHECK(
      inp.scalar_type() == at::ScalarType::Float || inp.scalar_type() == at::ScalarType::BFloat16,
      "input must be a float or bfloat16 tensor");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Byte, "output must be a uint8 tensor");
  TORCH_CHECK(out_dtype == transformer_engine::DType::kFloat8E4M3 ||
                  out_dtype == transformer_engine::DType::kFloat8E5M2,
              "out_dtype must be kFloat8E4M3 or kFloat8E5M2");

  const TensorWrapper inp_cu = makeTransformerEngineTensor(inp);
  TensorWrapper out_cu = makeTransformerEngineTensor(out);
  const TensorWrapper scale_cu = makeTransformerEngineTensor(scale);

  nvte_fp8_block_scaling_partial_cast(
      inp_cu.data(), out_cu.data(), scale_cu.data(), h, w, scale.stride(0), scale.stride(1),
      start_offset, block_len, static_cast<NVTEDType>(out_dtype), at::cuda::getCurrentCUDAStream());
}

}  // namespace transformer_engine::pytorch
