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
                                      std::optional<c10::Device> device,
                                      std::optional<std::vector<size_t>> alignments) {
  // Check shapes and dtypes
  const size_t n = shapes.size();
  NVTE_CHECK(dtypes.size() == n, "Got ", shapes.size(), " shapes and ", dtypes.size(), " dtypes.");
  NVTE_CHECK(!alignments || alignments->size() == n, "Got ", shapes.size(), " shapes and ",
             alignments->size(), " alignments.");

  // Return immediately if no tensors are needed
  if (n == 0) return {};

  // Set defaults for optional arguments
  if (!device) {
    device = c10::Device(c10::kCUDA);
  }
  if (!alignments) {
    alignments = std::vector<size_t>{};
    alignments->reserve(n);
    for (const auto &dtype : dtypes) {
      alignments->push_back(c10::elementSize(dtype));
    }
  }

  // Compute offsets in base buffer
  std::vector<size_t> byte_sizes(n);
  std::vector<size_t> offsets(n);
  size_t base_byte_size = 0;
  size_t base_alignment = 1;
  for (size_t i = 0; i < n; ++i) {
    byte_sizes[i] = product(shapes[i]) * at::elementSize(dtypes[i]);
    offsets[i] = roundup(base_byte_size, (*alignments)[i]);
    base_byte_size = offsets[i] + byte_sizes[i];
    base_alignment = std::max(base_alignment, (*alignments)[i]);
  }
  if (base_alignment > 1) {
    // Pad in case data pointer is not aligned
    base_byte_size += base_alignment;
  }

  // Allocate base buffer
  auto base_buffer = std::make_shared<at::Tensor>(
      at::empty({static_cast<int64_t>(base_byte_size)}, at::device(*device).dtype(torch::kUInt8)));
  uint8_t *base_ptr = base_buffer->data_ptr<uint8_t>();
  base_ptr =
      reinterpret_cast<uint8_t *>(roundup(reinterpret_cast<uintptr_t>(base_ptr), base_alignment));

  // Create views into base buffer
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
      out.emplace_back(at::empty(shape_int64, at::device(*device).dtype(dtypes[i])));
    } else {
      // Construct tensor with custom deleter to keep base buffer alive
      out.emplace_back(at::from_blob(
          base_ptr + offsets[i], shape_int64, [base_buffer](void *) {},
          at::device(*device).dtype(dtypes[i])));
    }
  }
  return out;
}

}  // namespace pytorch
}  // namespace transformer_engine
