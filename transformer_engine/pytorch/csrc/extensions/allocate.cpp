/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <memory>
#include <vector>

#include "../extensions.h"

namespace transformer_engine {
namespace pytorch {

std::vector<at::Tensor> bulk_allocate(const std::vector<std::vector<size_t>> &shapes,
                                      const std::vector<at::ScalarType> &dtypes,
                                      const std::vector<size_t> &alignments) {
  const size_t n = shapes.size();
  NVTE_CHECK(dtypes.size() == n, "Got ", shapes.size(), " shapes and ", dtypes.size(), " dtypes.");
  NVTE_CHECK(alignments.size() == n, "Got ", shapes.size(), " shapes and ", alignments.size(),
             " alignments.");
  if (n == 0) return {};

  // Compute per-tensor sizes and offsets
  size_t total_bytes = 0;
  std::vector<size_t> byte_sizes(n);
  std::vector<size_t> offsets(n);
  for (size_t i = 0; i < n; ++i) {
    if (alignments[i] > 0) {
      total_bytes = roundup(total_bytes, alignments[i]);
    }
    offsets[i] = total_bytes;
    byte_sizes[i] = product(shapes[i]) * at::elementSize(dtypes[i]);
    total_bytes += byte_sizes[i];
  }

  // Single backing allocation
  auto buffer = std::make_shared<at::Tensor>(
      at::empty({static_cast<int64_t>(total_bytes)}, at::device(at::kCUDA).dtype(torch::kUInt8)));
  uint8_t *data_ptr = buffer->data_ptr<uint8_t>();

  // Create views into the buffer
  std::vector<at::Tensor> out;
  out.reserve(n);
  std::vector<int64_t> shape_int64;
  for (size_t i = 0; i < n; ++i) {
    shape_int64.assign(shapes[i].begin(), shapes[i].end());
    if (byte_sizes[i] == 0) {
      // Work around problems with from_blob when constructing an
      // empty tensor. Passing a null pointer fails because it checks
      // that the pointer is on GPU. Passing a non-null pointer can
      // cause bugs in TE kernels.
      out.emplace_back(at::empty(shape_int64, at::device(at::kCUDA).dtype(dtypes[i])));
    } else {
      out.emplace_back(at::from_blob(
          data_ptr + offsets[i], shape_int64, [buffer](void *) {},  // Deleter keeps buffer alive
          at::device(at::kCUDA).dtype(dtypes[i])));
    }
  }
  return out;
}

}  // namespace pytorch
}  // namespace transformer_engine
