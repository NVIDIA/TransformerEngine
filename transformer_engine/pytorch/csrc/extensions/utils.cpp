/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "common/common.h"
#include "extensions.h"

namespace transformer_engine::pytorch {

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
