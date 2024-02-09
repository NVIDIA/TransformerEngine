/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

#include <string>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void fused_amax_and_scale_update(const at::Tensor &amax_history,
                                 const at::Tensor &scale,
                                 const at::Tensor &scale_inv,
                                 const at::Tensor &scale_inv_mask,
                                 at::Tensor updated_amax_history,
                                 at::Tensor updated_scale,
                                 at::Tensor updated_scale_inv,
                                 const std::string& amax_compute_algo,
                                 transformer_engine::DType fp8_dtype,
                                 float margin) {
  nvte_delayed_scaling_recipe_amax_and_scale_update(
    makeTransformerEngineTensor(amax_history).data(),
    makeTransformerEngineTensor(scale).data(),
    makeTransformerEngineTensor(scale_inv).data(),
    makeTransformerEngineTensor(scale_inv_mask).data(),
    makeTransformerEngineTensor(updated_amax_history).data(),
    makeTransformerEngineTensor(updated_scale).data(),
    makeTransformerEngineTensor(updated_scale_inv).data(),
    amax_compute_algo.c_str(),
    static_cast<NVTEDType>(fp8_dtype),
    margin,
    at::cuda::getCurrentCUDAStream());
}
