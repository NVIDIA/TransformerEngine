/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "common/common.h"
#include "extensions.h"
#include "util.h"

namespace transformer_engine::pytorch {

namespace {

at::Tensor collect_pointers_in_device_tensor(const std::vector<uint64_t>& host_ptrs,
                                             const at::Device& device, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(host_ptrs.size());
  auto out = at::empty({count}, at::TensorOptions().dtype(at::kLong).device(device));
  nvte_store_value_on_device(host_ptrs.data(), out.data_ptr(),
                             static_cast<size_t>(count) * sizeof(uint64_t), stream);
  return out;
}

}  // namespace

std::vector<at::Tensor> convert_host_pointers_to_tensor(
    std::vector<std::vector<at::Tensor>> tensor_lists) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensor_lists.size());
  auto stream = at::cuda::getCurrentCUDAStream();

  for (const auto& tensor_list : tensor_lists) {
    NVTE_CHECK(!tensor_list.empty(), "Tensor list is empty.");
    const auto& first_tensor = tensor_list[0];
    NVTE_CHECK(first_tensor.is_cuda(), "Tensor list must be on CUDA.");
    const auto device = first_tensor.device();
    const int64_t count = static_cast<int64_t>(tensor_list.size());
    std::vector<uint64_t> host_ptrs(count);
    for (int64_t i = 0; i < count; ++i) {
      host_ptrs[i] = reinterpret_cast<uintptr_t>(tensor_list[static_cast<size_t>(i)].data_ptr());
    }
    outputs.push_back(collect_pointers_in_device_tensor(host_ptrs, device, stream));
  }

  return outputs;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> get_device_pointer_for_data_and_scales(
    std::vector<at::Tensor> data_tensors, std::vector<at::Tensor> scale_tensors, bool swizzle,
    bool rowwise, transformer_engine::DType data_dtype) {
  const size_t num_tensors = data_tensors.size();
  NVTE_CHECK(num_tensors > 0, "data_tensors must not be empty.");
  NVTE_CHECK(num_tensors == scale_tensors.size(),
             "data_tensors and scale_tensors must have the same size.");
  NVTE_CHECK(data_tensors[0].is_cuda(), "data_tensors must be on CUDA.");
  const auto device = data_tensors[0].device();
  auto stream = at::cuda::getCurrentCUDAStream();

  // Infer data shape from the first data tensor (expected 2D: n x k)
  NVTE_CHECK(data_tensors[0].dim() == 2,
             "data_tensors elements must be 2D, got dim=", data_tensors[0].dim());
  NVTEShape data_shape{};
  data_shape.ndim = 2;
  data_shape.data[0] = static_cast<size_t>(data_tensors[0].size(0));
  data_shape.data[1] = static_cast<size_t>(data_tensors[0].size(1));

  // Collect data device pointers
  std::vector<uint64_t> data_host_ptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    data_host_ptrs[i] = reinterpret_cast<uintptr_t>(data_tensors[i].data_ptr());
  }

  // Swizzle scales and collect scale pointers
  at::Tensor swizzled_scales_keepalive;
  std::vector<uint64_t> scale_host_ptrs(num_tensors);

  if (swizzle) {
    // Determine scaling mode and scale dtype from data dtype
    NVTEScalingMode scaling_mode;
    transformer_engine::DType scale_dtype;
    if (is_fp8_dtype(data_dtype)) {
      scaling_mode = NVTE_MXFP8_1D_SCALING;
      scale_dtype = transformer_engine::DType::kFloat8E8M0;
    } else if (is_fp4_dtype(data_dtype)) {
      scaling_mode = NVTE_NVFP4_1D_SCALING;
      scale_dtype = transformer_engine::DType::kFloat8E4M3;
    } else {
      NVTE_ERROR("data_dtype must be an FP8 or FP4 type for swizzling.");
    }

    // Build TensorWrappers for swizzle
    std::vector<transformer_engine::TensorWrapper> scale_wrappers;
    scale_wrappers.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      NVTEShape scale_shape = convertTorchShape(scale_tensors[i].sizes());
      void* scale_ptr = scale_tensors[i].data_ptr();
      scale_wrappers.emplace_back(scaling_mode);
      auto& wrapper = scale_wrappers.back();
      if (rowwise) {
        wrapper.set_rowwise_data(nullptr, data_dtype, data_shape);
        wrapper.set_rowwise_scale_inv(scale_ptr, scale_dtype, scale_shape);
      } else {
        wrapper.set_columnwise_data(nullptr, data_dtype, data_shape);
        wrapper.set_columnwise_scale_inv(scale_ptr, scale_dtype, scale_shape);
      }
    }

    // Swizzle scales; wrappers are updated in-place with swizzled pointers
    auto result = multi_tensor_swizzle_scales_for_gemm(scale_wrappers, rowwise, !rowwise);
    NVTE_CHECK(result.has_value(), "Scale swizzle returned no output buffer.");
    swizzled_scales_keepalive = std::move(*result);

    // Collect swizzled scale pointers from updated wrappers
    for (size_t i = 0; i < num_tensors; ++i) {
      const auto scales_nvte = rowwise ? scale_wrappers[i].get_rowwise_scale_inv()
                                       : scale_wrappers[i].get_columnwise_scale_inv();
      scale_host_ptrs[i] = reinterpret_cast<uintptr_t>(scales_nvte.data_ptr);
    }
  } else {
    swizzled_scales_keepalive = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    for (size_t i = 0; i < num_tensors; ++i) {
      scale_host_ptrs[i] = reinterpret_cast<uintptr_t>(scale_tensors[i].data_ptr());
    }
  }

  // Convert pointer arrays to device tensors
  auto data_ptrs = collect_pointers_in_device_tensor(data_host_ptrs, device, stream);
  auto scale_ptrs = collect_pointers_in_device_tensor(scale_host_ptrs, device, stream);

  return {std::move(data_ptrs), std::move(scale_ptrs), std::move(swizzled_scales_keepalive)};
}

}  // namespace transformer_engine::pytorch
