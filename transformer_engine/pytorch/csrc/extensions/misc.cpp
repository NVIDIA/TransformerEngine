/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "pybind.h"

namespace transformer_engine::pytorch {

size_t get_cublasLt_version() { return cublasLtGetVersion(); }

size_t get_cudnn_version() { return cudnnGetVersion(); }

namespace {

std::vector<at::Tensor> prepare_grouped_splits_impl(const at::Tensor &split_sizes,
                                                    int64_t num_groups,
                                                    const std::vector<int64_t> &logical_last_dims) {
  NVTE_CHECK(split_sizes.scalar_type() == at::kInt || split_sizes.scalar_type() == at::kLong,
             "split_sizes must have dtype int32 or int64.");
  NVTE_CHECK(split_sizes.dim() == 1, "split_sizes must be a 1D tensor.");
  NVTE_CHECK(num_groups > 0, "num_groups must be greater than 0.");
  NVTE_CHECK(split_sizes.numel() == num_groups, "split_sizes must have length ", num_groups, ".");
  NVTE_CHECK(!logical_last_dims.empty(), "logical_last_dims must be non-empty.");
  const size_t list_size = logical_last_dims.size();
  for (const auto logical_last_dim : logical_last_dims) {
    NVTE_CHECK(logical_last_dim >= 0, "logical_last_dim values must be non-negative.");
  }
  const c10::Device device = c10::Device(c10::kCUDA, c10::cuda::current_device());

  at::Tensor split_sizes_for_kernel;
  if (split_sizes.is_cuda()) {
    NVTE_CHECK(split_sizes.device() == device, "CUDA split_sizes must be on current CUDA device ",
               device.index(), ", but got CUDA device ", split_sizes.device().index(), ".");
    split_sizes_for_kernel = split_sizes;
  } else {
    // Preserve the legacy eager path: host m_splits are copied to the target
    // CUDA device here, then all derived metadata is produced by one CUDA kernel.
    split_sizes_for_kernel =
        split_sizes.to(at::TensorOptions().dtype(split_sizes.scalar_type()).device(device),
                       /*non_blocking=*/true);
  }

  const int64_t offsets_length = num_groups + 1;
  auto split_sizes_i64 = split_sizes_for_kernel.scalar_type() == at::kLong
                             ? split_sizes_for_kernel
                             : split_sizes_for_kernel.to(at::kLong);

  // Return order is part of the Python contract:
  //   0. split_sizes_i64: int64[num_groups], canonical TE GroupedTensor first dims.
  //   1. split_points: int32[num_groups], cumsum(split_sizes) without the leading 0
  //      for cuDNN grouped GEMM padded offsets.  This is intentionally int32
  //      even though TE grouped tensor metadata uses int64 below.
  //   2..: tensor_offsets[i]: int64[num_groups + 1],
  //      [0, cumsum(split_sizes)] * logical_last_dims[i].  Callers that need
  //      base_offsets request logical_last_dim=1 and name that output locally.
  //
  // ``logical_last_dims`` is the complete offset stride list passed to the
  // common kernel.  Each element produces one returned offset vector.
  //
  // Force 16-byte alignment on every output so ``split_points`` (consumed by
  // cuDNN CuTe-DSL grouped GEMM as ``padded_offsets``, which requires 16-byte
  // alignment) lands on a 16-byte boundary inside the bulk buffer.
  // Allocation order mirrors the return order after ``split_sizes_i64``:
  //   split_points, offsets for each requested logical_last_dim...
  std::vector<std::vector<size_t>> shapes = {{static_cast<size_t>(num_groups)}};
  std::vector<at::ScalarType> dtypes = {at::kInt};
  std::vector<size_t> alignments = {16};
  shapes.reserve(1 + list_size);
  dtypes.reserve(1 + list_size);
  alignments.reserve(1 + list_size);
  for (size_t i = 0; i < list_size; ++i) {
    shapes.emplace_back(std::vector<size_t>{static_cast<size_t>(offsets_length)});
    dtypes.emplace_back(at::kLong);
    alignments.emplace_back(16);
  }
  auto outputs = bulk_allocate(shapes, dtypes, device, alignments);
  auto split_points = outputs[0];

  std::vector<int64_t> stride_list(list_size, 0);
  std::vector<NVTETensor> split_offsets_list(list_size, nullptr);
  std::vector<TensorWrapper> split_offsets_nvte;
  split_offsets_nvte.reserve(list_size);
  auto split_sizes_nvte = makeTransformerEngineTensor(split_sizes_for_kernel);
  auto split_points_nvte = makeTransformerEngineTensor(split_points);
  for (size_t list_idx = 0; list_idx < list_size; ++list_idx) {
    stride_list[list_idx] = logical_last_dims[list_idx];
    split_offsets_nvte.emplace_back(makeTransformerEngineTensor(outputs[1 + list_idx]));
    split_offsets_list[list_idx] = split_offsets_nvte.back().data();
  }
  NVTE_CHECK(
      stride_list.size() == list_size && split_offsets_list.size() == list_size,
      "Internal error: stride_list and split_offsets_list must both have list_size entries.");

  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_splits_to_offsets(split_sizes_nvte.data(), stride_list.data(),
                                 split_points_nvte.data(), split_offsets_list.data(), list_size,
                                 at::cuda::getCurrentCUDAStream());
  });

  std::vector<at::Tensor> ret;
  ret.reserve(2 + list_size);
  ret.emplace_back(split_sizes_i64);
  ret.emplace_back(split_points);
  for (size_t i = 0; i < list_size; ++i) {
    ret.emplace_back(outputs[1 + i]);
  }
  return ret;
}

}  // namespace

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

std::vector<at::Tensor> prepare_grouped_splits(const at::Tensor &split_sizes, int64_t num_groups,
                                               const std::vector<int64_t> &logical_last_dims) {
  return prepare_grouped_splits_impl(split_sizes, num_groups, logical_last_dims);
}

}  // namespace transformer_engine::pytorch
