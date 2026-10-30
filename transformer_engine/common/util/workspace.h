/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_WORKSPACE_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_WORKSPACE_H_

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

#include "logging.h"

namespace transformer_engine {

// Minimum alignment of the allocation and every nonempty block. An operation
// may request a stricter power-of-two alignment for an individual block.
constexpr size_t kWorkspaceAlignment = 128;

// Byte offsets are independent of the allocation address. An empty block is
// optional storage and maps to nullptr. Layouts and blocks own no memory.
struct WorkspaceBlock {
  size_t offset = 0;
  size_t size = 0;
  size_t alignment = kWorkspaceAlignment;
};

// Build an operation-specific layout on the host, using the same sequence of
// additions for a size query and execution. Keep named blocks in the operation's
// own layout struct. Appending blocks preserves existing offsets; changing the
// size/order of earlier blocks requires recomputing subsequent offsets.
class WorkspaceLayout {
 public:
  WorkspaceBlock add_bytes(size_t bytes, size_t alignment = kWorkspaceAlignment) {
    alignment = normalize_alignment(alignment);
    if (bytes == 0) {
      return {};
    }
    const size_t offset = align_up(end_, alignment);
    NVTE_CHECK(bytes <= std::numeric_limits<size_t>::max() - offset,
               "Workspace block size overflows size_t.");
    const size_t end = offset + bytes;
    const size_t layout_alignment = std::max(alignment_, alignment);
    const size_t size = align_up(end, layout_alignment);
    // Commit only after all overflow checks succeed.
    end_ = end;
    size_ = size;
    alignment_ = layout_alignment;
    return {offset, bytes, alignment};
  }

  template <typename T>
  WorkspaceBlock add(size_t count, size_t alignment = kWorkspaceAlignment) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Workspace elements must be trivially copyable.");
    NVTE_CHECK(count <= std::numeric_limits<size_t>::max() / sizeof(T),
               "Workspace element count overflows size_t.");
    return add_bytes(count * sizeof(T), std::max(normalize_alignment(alignment), alignof(T)));
  }

  // Includes final padding, so the allocation can also be used as an aligned
  // region of a larger workspace. An empty layout requires zero bytes.
  size_t size() const { return size_; }
  size_t alignment() const { return alignment_; }

 private:
  static size_t normalize_alignment(size_t alignment) {
    NVTE_CHECK(alignment != 0 && (alignment & (alignment - 1)) == 0,
               "Workspace alignment must be a nonzero power of two.");
    return std::max(alignment, kWorkspaceAlignment);
  }

  static size_t align_up(size_t value, size_t alignment) {
    const size_t padding = (alignment - value % alignment) % alignment;
    NVTE_CHECK(padding <= std::numeric_limits<size_t>::max() - value,
               "Workspace alignment padding overflows size_t.");
    return value + padding;
  }

  size_t end_ = 0;
  size_t size_ = 0;
  size_t alignment_ = kWorkspaceAlignment;
};

// Resolve blocks in a caller-owned allocation on the host. The caller supplies
// at least layout.size() bytes at an address aligned to layout.alignment(); this
// view does not validate the allocation. Pass the resulting device pointers (or
// the base pointer and block offsets) to kernels.
// This view neither initializes storage nor tracks asynchronous uses. After
// reallocation, reconstruct views and reinitialize or explicitly migrate valid
// cache blocks. Keep old allocations alive until all kernels/graphs using them
// are finished. Scratch contents have no preservation guarantee.
class WorkspaceView {
 public:
  WorkspaceView(void *data, const WorkspaceLayout &layout)
      : data_(static_cast<char *>(data)), size_(layout.size()) {}

  template <typename T>
  T *get(const WorkspaceBlock &block) const {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Workspace elements must be trivially copyable.");
    if (block.size == 0) {
      return nullptr;
    }
    NVTE_CHECK(block.offset <= size_ && block.size <= size_ - block.offset,
               "Workspace block is outside the layout.");
    NVTE_CHECK(
        block.alignment >= kWorkspaceAlignment && (block.alignment & (block.alignment - 1)) == 0,
        "Invalid workspace block alignment.");
    NVTE_CHECK(block.size % sizeof(T) == 0,
               "Workspace block size does not match the element type.");
    NVTE_CHECK(block.alignment >= alignof(T) && block.offset % block.alignment == 0,
               "Workspace block does not satisfy the required alignment.");
    return reinterpret_cast<T *>(data_ + block.offset);
  }

 private:
  char *data_;
  size_t size_;
};

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_WORKSPACE_H_
