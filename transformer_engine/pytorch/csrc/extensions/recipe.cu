/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <string>

#include "extensions.h"

void fused_amax_and_scale_update_after_reduction(
    const at::Tensor &amax_reduction_buffer, std::vector<at::Tensor> amax_histories,
    std::vector<at::Tensor> scales, std::vector<at::Tensor> scale_invs,
    const std::string &amax_compute_algo, transformer_engine::DType fp8_dtype, float margin) {
  using namespace transformer_engine;
  size_t num_tensors = amax_histories.size();
  std::vector<Tensor> t_amax_histories(num_tensors);
  std::vector<Tensor> t_scales(num_tensors);
  std::vector<Tensor> t_scale_invs(num_tensors);
  std::vector<NVTETensor> te_amax_histories(num_tensors);
  std::vector<NVTETensor> te_scales(num_tensors);
  std::vector<NVTETensor> te_scale_invs(num_tensors);
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

    t_scale_invs[i].data.dptr = scale_invs[i].data_ptr();
    auto scale_inv_sizes = scale_invs[i].sizes().vec();
    std::vector<size_t> scale_inv_shape{scale_inv_sizes.begin(), scale_inv_sizes.end()};
    t_scale_invs[i].data.shape = scale_inv_shape;
    t_scale_invs[i].data.dtype = DType::kFloat32;

    te_amax_histories[i] = reinterpret_cast<NVTETensor>(&t_amax_histories[i]);
    te_scales[i] = reinterpret_cast<NVTETensor>(&t_scales[i]);
    te_scale_invs[i] = reinterpret_cast<NVTETensor>(&t_scale_invs[i]);
  }
  nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
      makeTransformerEngineTensor(amax_reduction_buffer).data(), te_amax_histories, te_scales,
      te_scale_invs, amax_compute_algo.c_str(), static_cast<NVTEDType>(fp8_dtype), margin,
      at::cuda::getCurrentCUDAStream());
}

namespace {

__global__ void __launch_bounds__(1) scalar_reciprocal_kernel(const float* __restrict__ src,
                                                              float* __restrict__ dst,
                                                              const float* __restrict__ noop) {
  if (noop != nullptr && *noop == 1.f) {
    return;
  }
  *dst = __frcp_rn(*src);
}

}  // namespace

at::Tensor scalar_reciprocal(const at::Tensor &src,
                             std::optional<at::Tensor> dst,
                             int64_t src_offset,
                             int64_t dst_offset,
                             const std::optional<at::Tensor> &noop_flag) {
  using namespace transformer_engine;

  // Allocate output tensor if needed
  NVTE_CHECK(dst || dst_offset == 0,
             "Provided offset in output tensor without providing output tensor");
  at::Tensor dst_val = dst ? dst.value() : allocateTorchTensor(1, DType::kFloat32);

  // Get pointers
  const float* src_ptr = reinterpret_cast<const float*>(getDataPtr(src, src_offset));
  float* dst_ptr = reinterpret_cast<float*>(getDataPtr(dst_val, dst_offset));
  const float* noop_ptr = nullptr;
  if (noop_flag) {
    noop_ptr = reinterpret_cast<const float*>(getDataPtr(noop_flag.value()));
  }

  // Launch kernel
  scalar_reciprocal_kernel<<<1,1,0,at::cuda::getCurrentCUDAStream()>>>(src_ptr, dst_ptr, noop_ptr);

  return dst_val;
}
