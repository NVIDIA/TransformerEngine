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
    split_sizes_for_kernel =
        split_sizes.to(at::TensorOptions().dtype(split_sizes.scalar_type()).device(device),
                       /*non_blocking=*/true);
  }

  const int64_t offsets_length = num_groups + 1;
  auto split_sizes_i64 = split_sizes_for_kernel.scalar_type() == at::kLong
                             ? split_sizes_for_kernel
                             : split_sizes_for_kernel.to(at::kLong);

  // Return order is part of the Python contract:
  //   0. split_sizes_i64: int64[num_groups]   - canonical TE GroupedTensor first dims.
  //   1. split_points:    int32[num_groups]   - inclusive scan of split_sizes,
  //                                             no leading zero. Consumed by cuDNN
  //                                             CuTe-DSL grouped GEMM as
  //                                             padded_offsets (requires 16-byte align).
  //   2..: tensor_offsets[i]: int64[num_groups + 1] - inclusive scan with leading
  //                                                   zero, scaled by logical_last_dims[i].
  //
  // 16-byte alignment is forced on every output so split_points (cuDNN's
  // padded_offsets) lands on a 16-byte boundary inside the bulk-allocated buffer.
  const size_t num_outputs = 1 + list_size;
  std::vector<std::vector<size_t>> shapes;
  std::vector<at::ScalarType> dtypes;
  std::vector<size_t> alignments(num_outputs, 16);
  shapes.reserve(num_outputs);
  dtypes.reserve(num_outputs);
  shapes.emplace_back(std::vector<size_t>{static_cast<size_t>(num_groups)});
  dtypes.emplace_back(at::kInt);
  for (size_t i = 0; i < list_size; ++i) {
    shapes.emplace_back(std::vector<size_t>{static_cast<size_t>(offsets_length)});
    dtypes.emplace_back(at::kLong);
  }
  auto outputs = bulk_allocate(shapes, dtypes, device, alignments);

  // Pack output NVTETensors as a single batch so the kernel sees all metadata
  // from one nvte_create_tensors call rather than num_outputs separate calls.
  MultiTensorWrapper output_tensors_nvte(num_outputs);
  std::vector<NVTETensor> nvte_outputs(num_outputs);
  std::vector<int64_t> strides(num_outputs);
  std::vector<int> include_leading_zero(num_outputs);

  const size_t num_groups_sz = static_cast<size_t>(num_groups);
  const size_t offsets_length_sz = static_cast<size_t>(offsets_length);
  const auto set_output = [&](size_t out_idx, at::Tensor &tensor, DType dtype, size_t numel,
                              int64_t stride, int with_leading_zero) {
    NVTEShape shape = nvte_make_shape(&numel, 1);
    NVTEBasicTensor data = {tensor.data_ptr(), static_cast<NVTEDType>(dtype), shape};
    nvte_set_tensor_param_v2(output_tensors_nvte[out_idx], kNVTERowwiseData, &data, sizeof(data));
    nvte_outputs[out_idx] = output_tensors_nvte[out_idx];
    strides[out_idx] = stride;
    include_leading_zero[out_idx] = with_leading_zero;
  };

  set_output(0, outputs[0], DType::kInt32, num_groups_sz, /*stride=*/1,
             /*with_leading_zero=*/0);
  for (size_t i = 0; i < list_size; ++i) {
    set_output(1 + i, outputs[1 + i], DType::kInt64, offsets_length_sz,
               /*stride=*/logical_last_dims[i], /*with_leading_zero=*/1);
  }

  auto split_sizes_nvte = makeTransformerEngineTensor(split_sizes_for_kernel);
  NVTE_SCOPED_GIL_RELEASE({
    nvte_splits_to_offsets_multi(split_sizes_nvte.data(), nvte_outputs.data(), strides.data(),
                                 include_leading_zero.data(), num_outputs,
                                 at::cuda::getCurrentCUDAStream());
  });

  std::vector<at::Tensor> ret;
  ret.reserve(1 + num_outputs);
  ret.emplace_back(split_sizes_i64);
  for (size_t i = 0; i < num_outputs; ++i) {
    ret.emplace_back(outputs[i]);
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
