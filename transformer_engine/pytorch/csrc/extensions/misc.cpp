/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <tuple>
#include <utility>
#include <vector>

#include "../extensions.h"
#include "common/common.h"
#include "pybind.h"

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

std::tuple<at::Tensor, std::vector<at::Tensor>> splits_to_offsets_multi(
    const at::Tensor &split_sizes, const c10::Device &device, const std::vector<int64_t> &strides,
    const std::vector<bool> &include_leading_zero, const std::vector<at::ScalarType> &dtypes,
    bool bulk_allocate_outputs) {
  const size_t num_outputs = strides.size();
  const size_t num_splits = static_cast<size_t>(split_sizes.numel());

  // Check inputs.
  NVTE_CHECK(include_leading_zero.size() == num_outputs && dtypes.size() == num_outputs,
             "strides, include_leading_zero, and dtypes must have matching lengths, but got ",
             strides.size(), ", ", include_leading_zero.size(), ", and ", dtypes.size(), ".");
  NVTE_CHECK(device.is_cuda(), "device must be CUDA, but got ", device.str(), ".");

  // Convert split sizes to int64 GPU tensor.
  const at::Tensor split_sizes_i64 =
      split_sizes.scalar_type() == at::kLong ? split_sizes : split_sizes.to(at::kLong);
  const at::Tensor split_sizes_out =
      split_sizes_i64.device() == device ? split_sizes_i64 : split_sizes_i64.to(device);

  // Allocate outputs.
  std::vector<at::Tensor> outputs;
  outputs.reserve(num_outputs);
  if (bulk_allocate_outputs) {
    std::vector<std::vector<size_t>> shapes;
    shapes.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
      const size_t length = num_splits + (include_leading_zero[i] ? 1 : 0);
      shapes.emplace_back(std::vector<size_t>{length});
    }
    // cuDNN CuTe DSL grouped GEMM kernels require padded_offsets
    // aligned to 16 bytes.
    const std::vector<size_t> alignments(num_outputs, 16);
    outputs = bulk_allocate(shapes, dtypes, device, alignments);
  } else {
    for (size_t i = 0; i < num_outputs; ++i) {
      const int64_t length = static_cast<int64_t>(num_splits) + (include_leading_zero[i] ? 1 : 0);
      outputs.emplace_back(
          at::empty({length}, at::TensorOptions().dtype(dtypes[i]).device(device)));
    }
  }

  // Construct NVTETensors.
  MultiTensorWrapper outputs_nvte(num_outputs);
  std::vector<int> include_leading_zero_int(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    const size_t length = num_splits + (include_leading_zero[i] ? 1 : 0);
    NVTEShape shape = nvte_make_shape(&length, 1);
    NVTEBasicTensor data = {outputs[i].data_ptr(),
                            static_cast<NVTEDType>(GetTransformerEngineDType(dtypes[i])), shape};
    nvte_set_tensor_param_v2(outputs_nvte[i], kNVTERowwiseData, &data, sizeof(data));
    include_leading_zero_int[i] = include_leading_zero[i] ? 1 : 0;
  }

  auto split_sizes_nvte = makeTransformerEngineTensor(split_sizes_out);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_splits_to_offsets_multi(split_sizes_nvte.data(), outputs_nvte.data(), strides.data(),
                                 include_leading_zero_int.data(), num_outputs,
                                 at::cuda::getCurrentCUDAStream());
  });

  return {split_sizes_out, std::move(outputs)};
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
