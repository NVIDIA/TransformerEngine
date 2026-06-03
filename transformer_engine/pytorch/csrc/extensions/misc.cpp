/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "../extensions.h"
#include "common/common.h"

namespace transformer_engine::pytorch {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

at::Tensor splits_to_offsets(const at::Tensor &first_dims, int64_t logical_last_dim) {
  NVTE_CHECK(first_dims.is_cuda(), "first_dims must be on CUDA.");
  NVTE_CHECK(first_dims.scalar_type() == at::kLong, "first_dims must have dtype int64.");
  NVTE_CHECK(first_dims.dim() == 1, "first_dims must be a 1D tensor.");
  NVTE_CHECK(logical_last_dim > 0, "logical_last_dim must be greater than 0.");

  auto first_dims_contiguous = first_dims.contiguous();
  const auto num_tensors = static_cast<size_t>(first_dims_contiguous.numel());
  auto output = at::empty({static_cast<int64_t>(num_tensors) + 1},
                          first_dims_contiguous.options().dtype(at::kLong));

  nvte_splits_to_offsets(static_cast<const int64_t *>(first_dims_contiguous.data_ptr()),
                         static_cast<int64_t *>(output.data_ptr()), num_tensors, logical_last_dim,
                         at::cuda::getCurrentCUDAStream());

  return output;
}

at::Tensor copy_data_ptrs_to_device(const std::vector<at::Tensor> &tensors,
                                    const c10::Device &device) {
  // Collect data pointers
  std::vector<uint64_t> ptrs_host;
  ptrs_host.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    ptrs_host.push_back(reinterpret_cast<uintptr_t>(tensor.data_ptr()));
  }

  // Allocate device buffer
  auto ptrs_device = at::empty({static_cast<int64_t>(tensors.size())},
                               at::TensorOptions().dtype(at::kLong).device(device));

  // Load pointers on device
  nvte_copy_host_to_device_via_kernel(ptrs_host.data(), ptrs_device.data_ptr(),
                                      tensors.size() * sizeof(uint64_t),
                                      at::cuda::getCurrentCUDAStream());

  return ptrs_device;
}

}  // namespace transformer_engine::pytorch
