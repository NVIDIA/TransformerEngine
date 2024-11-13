/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor cast_to_fp8(const at::Tensor& input, const at::Tensor& scale, at::Tensor amax,
                       at::Tensor scale_inv, transformer_engine::DType otype,
                       const int scale_offset, const int amax_offset, const int scale_inv_offset) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  if (input.numel() == 0) return output;

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax_dptr,
                                               scale_dptr, scale_inv_dptr);

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

void cast_to_fp8_noalloc(const at::Tensor& input, const at::Tensor& scale, at::Tensor output,
                         at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype,
                         const int scale_offset, const int amax_offset,
                         const int scale_inv_offset) {
  using namespace transformer_engine;
  size_t N = static_cast<size_t>(input.size(0));
  size_t H = static_cast<size_t>(input.size(1));

  // Get pointers for FP8 scale, amax, scale-inverse
  void* scale_dptr = getDataPtr(scale, scale_offset);
  void* amax_dptr = getDataPtr(amax, amax_offset);
  void* scale_inv_dptr = getDataPtr(scale_inv, scale_inv_offset);

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, H}, otype, amax_dptr,
                                               scale_dptr, scale_inv_dptr);

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return;
}

at::Tensor cast_from_fp8(const at::Tensor& input, const at::Tensor& scale_inv,
                         transformer_engine::DType itype, transformer_engine::DType otype,
                         const int scale_inv_offset) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), shape, itype, nullptr, nullptr,
                                              getDataPtr(scale_inv, scale_inv_offset));
  auto output_cu = makeTransformerEngineTensor(output);

  nvte_fp8_dequantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}
