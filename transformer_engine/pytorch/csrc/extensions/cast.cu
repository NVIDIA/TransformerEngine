/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor cast_to_fp8(const at::Tensor &input, const at::Tensor &scale, at::Tensor amax,
                       at::Tensor scale_inv, transformer_engine::DType otype) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  if (input.numel() == 0) return output;

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), shape, otype, amax.data_ptr(),
                                               scale.data_ptr(), scale_inv.data_ptr());

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}

void cast_to_fp8_noalloc(const at::Tensor &input, const at::Tensor &scale, at::Tensor output,
                         at::Tensor amax, at::Tensor scale_inv, transformer_engine::DType otype) {
  using namespace transformer_engine;
  size_t N = static_cast<size_t>(input.size(0));
  size_t H = static_cast<size_t>(input.size(1));

  auto input_cu = makeTransformerEngineTensor(input);
  auto output_cu = makeTransformerEngineTensor(output.data_ptr(), {N, H}, otype, amax.data_ptr(),
                                               scale.data_ptr(), scale_inv.data_ptr());

  nvte_fp8_quantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return;
}

at::Tensor cast_from_fp8(const at::Tensor &input, const at::Tensor &scale_inv,
                         transformer_engine::DType itype, transformer_engine::DType otype) {
  using namespace transformer_engine;
  auto input_shape = input.sizes().vec();
  std::vector<size_t> shape{input_shape.begin(), input_shape.end()};

  auto output = at::empty_like(input, at::CUDA(GetATenDType(otype)));

  auto input_cu = makeTransformerEngineTensor(input.data_ptr(), shape, itype, nullptr, nullptr,
                                              scale_inv.data_ptr());
  auto output_cu = makeTransformerEngineTensor(output);

  nvte_fp8_dequantize(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return output;
}
