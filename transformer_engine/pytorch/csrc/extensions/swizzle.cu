/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

at::Tensor swizzle_scaling_factors(at::Tensor input, at::Tensor scale_inv,
                                   std::vector<int64_t> scaling_mode) {
  using namespace transformer_engine;

  auto options = at::TensorOptions().dtype(scale_inv.dtype()).device(torch::kCUDA);
  auto swizzled_scale_inv = at::empty_like(scale_inv, options);

  void* scale_inv_dptr = getDataPtr(scale_inv, 0);
  void* swizzled_scale_inv_dptr = getDataPtr(swizzled_scale_inv, 0);

  NVTEScalingMode nvte_scaling_mode = {static_cast<int>(scaling_mode[0]),
                                       static_cast<int>(scaling_mode[1]),
                                       static_cast<int>(scaling_mode[2])};

  // Construct Transformer Engine tensors
  DType dtype = GetTransformerEngineDType(input.scalar_type());
  auto input_cu =
      makeTransformerEngineTensor(input.data_ptr(), getTensorShape(input), dtype, nullptr, nullptr,
                                  scale_inv_dptr, getTensorShape(scale_inv), nvte_scaling_mode);
  auto output_cu = makeTransformerEngineTensor(
      input.data_ptr(), getTensorShape(input), dtype, nullptr, nullptr, swizzled_scale_inv_dptr,
      getTensorShape(swizzled_scale_inv), nvte_scaling_mode);

  // Launch kernel
  nvte_swizzle_scaling_factors(input_cu.data(), output_cu.data(), at::cuda::getCurrentCUDAStream());

  return swizzled_scale_inv;
}
