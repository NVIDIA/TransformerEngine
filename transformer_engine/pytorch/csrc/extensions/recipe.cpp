/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <string>

#include "common/common.h"
#include "extensions.h"

void fused_amax_and_scale_update_after_reduction(const at::Tensor &amax_reduction_buffer,
                                                 std::vector<at::Tensor> amax_histories,
                                                 std::vector<at::Tensor> scales,
                                                 const std::string &amax_compute_algo,
                                                 transformer_engine::DType fp8_dtype,
                                                 float margin) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;
  size_t num_tensors = amax_histories.size();
  std::vector<Tensor> t_amax_histories(num_tensors);
  std::vector<Tensor> t_scales(num_tensors);
  std::vector<NVTETensor> te_amax_histories(num_tensors);
  std::vector<NVTETensor> te_scales(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    t_amax_histories[i].data.dptr = amax_histories[i].data_ptr();
    auto amax_sizes = amax_histories[i].sizes().vec();
    std::vector<size_t> amax_shape{amax_sizes.begin(), amax_sizes.end()};
    t_amax_histories[i].data.shape = amax_shape;
    t_amax_histories[i].data.dtype = DType::kFloat32;

    t_scales[i].data.dptr = scales[i].data_ptr();
    auto scale_sizes = scales[i].sizes().vec();
    std::vector<size_t> scale_shape{scale_sizes.begin(), scale_sizes.end()};
    t_scales[i].data.shape = scale_shape;
    t_scales[i].data.dtype = DType::kFloat32;

    te_amax_histories[i] = reinterpret_cast<NVTETensor>(&t_amax_histories[i]);
    te_scales[i] = reinterpret_cast<NVTETensor>(&t_scales[i]);
  }
  nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
      makeTransformerEngineTensor(amax_reduction_buffer).data(), te_amax_histories, te_scales,
      amax_compute_algo.c_str(), static_cast<NVTEDType>(fp8_dtype), margin,
      at::cuda::getCurrentCUDAStream());
}
