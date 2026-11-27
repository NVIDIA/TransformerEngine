/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <vector>

#include "../extensions.h"

namespace transformer_engine {
namespace pytorch {

/* Allocate multiple PyTorch tensors backed by the same buffer.
 *
 * Use with caution and avoid exposing externally.
 *
 * The outputs share allocator-owned storage so record_stream on any
 * output protects the backing allocation across streams. The backing
 * buffer is deallocated once the final tensor is destroyed.
 */
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
  auto base_buffer =
      at::empty({static_cast<int64_t>(base_byte_size)}, at::device(*device).dtype(torch::kUInt8));
  const auto base_ptr = reinterpret_cast<uintptr_t>(base_buffer.data_ptr());
  const size_t alignment_offset = roundup(base_ptr, base_alignment) - base_ptr;

  // Create views into base buffer
  std::vector<at::Tensor> out;
  out.reserve(n);
  std::vector<int64_t> shape_int64;
  for (size_t i = 0; i < n; ++i) {
    shape_int64.assign(shapes[i].begin(), shapes[i].end());
    if (byte_sizes[i] == 0) {
      // Keep empty tensors separate: TE kernels expect a null data pointer.
      out.emplace_back(at::empty(shape_int64, at::device(*device).dtype(dtypes[i])));
    } else {
      // Preserve the allocator-owned storage instead of wrapping its raw pointer.
      const size_t byte_offset = alignment_offset + offsets[i];
      const size_t element_size = at::elementSize(dtypes[i]);
      NVTE_CHECK(byte_offset % element_size == 0, "Bulk tensor offset is not dtype-aligned");
      out.emplace_back(at::empty({0}, at::device(*device).dtype(dtypes[i]))
                           .set_(base_buffer.storage(), byte_offset / element_size, shape_int64));
    }
  }
  return out;
}

}  // namespace pytorch
}  // namespace transformer_engine
