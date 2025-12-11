/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <string>

#include "../extensions.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

void compute_amax(const at::Tensor& tensor, at::Tensor& amax) {
  auto input_tensor = tensor.contiguous();
  const TensorWrapper& te_input = makeTransformerEngineTensor(input_tensor);

  TORCH_CHECK(amax.scalar_type() == at::kFloat, "amax must be a float tensor");
  TORCH_CHECK(amax.numel() == 1, "amax must have exactly one element");
  auto* amax_ptr = amax.data_ptr<float>();
  TensorWrapper fake_te_output(
      nullptr, te_input.shape(),
      DType::kFloat8E4M3,  // It doesn't matter because we only compute amax.
      amax_ptr);

  nvte_compute_amax(te_input.data(), fake_te_output.data(), at::cuda::getCurrentCUDAStream());
}

void fused_amax_and_scale_update_after_reduction(const at::Tensor& amax_reduction_buffer,
                                                 std::vector<at::Tensor> amax_histories,
                                                 std::vector<at::Tensor> scales,
                                                 const std::string& amax_compute_algo,
                                                 DType fp8_dtype, float margin) {
  size_t num_tensors = amax_histories.size();
  std::vector<NVTETensor> te_amax_histories;
  std::vector<NVTETensor> te_scales;
  te_amax_histories.reserve(num_tensors);
  te_scales.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    te_amax_histories.push_back(nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING));
    NVTETensor& amax_history = te_amax_histories.back();
    NVTEShape amax_shape = convertTorchShape(amax_histories[i].sizes());
    NVTEBasicTensor amax_history_data = {amax_histories[i].data_ptr(),
                                         static_cast<NVTEDType>(DType::kFloat32), amax_shape};
    nvte_set_tensor_param(&amax_history, kNVTERowwiseData, &amax_history_data);

    te_scales.push_back(nvte_create_tensor(NVTE_DELAYED_TENSOR_SCALING));
    NVTETensor& scale = te_scales.back();
    NVTEShape scale_shape = convertTorchShape(scales[i].sizes());
    NVTEBasicTensor scale_data = {scales[i].data_ptr(), static_cast<NVTEDType>(DType::kFloat32),
                                  scale_shape};
    nvte_set_tensor_param(&scale, kNVTERowwiseData, &scale_data);
  }
  nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
      makeTransformerEngineTensor(amax_reduction_buffer).data(), te_amax_histories, te_scales,
      amax_compute_algo.c_str(), static_cast<NVTEDType>(fp8_dtype), margin,
      at::cuda::getCurrentCUDAStream());
  for (auto& t : te_amax_histories) {
    nvte_destroy_tensor(t);
  }
  for (auto& t : te_scales) {
    nvte_destroy_tensor(t);
  }
}

}  // namespace transformer_engine::pytorch
