/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_CORE_GROUPED_TMA_WORKSPACE_H_
#define TRANSFORMER_ENGINE_COMMON_CAST_CORE_GROUPED_TMA_WORKSPACE_H_

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace common {

constexpr size_t kGroupedTmaWorkspaceAlignment = 128;

// Non-owning view passed by value to descriptor updater and consumer kernels.
struct GroupedTmaStorageView {
  CUtensorMap *input = nullptr;
  CUtensorMap *act_input = nullptr;
  CUtensorMap *output_rowwise = nullptr;
  CUtensorMap *output_colwise = nullptr;
  size_t *rows = nullptr;
  size_t *cols = nullptr;
  size_t *offsets = nullptr;
};

// One operation's descriptor arrays and mapper metadata. Query and execution
// reconstruct the same offsets from the group count, without retained state.
struct GroupedTmaWorkspaceLayout {
  explicit GroupedTmaWorkspaceLayout(size_t num_tensors) {
    input = reserve<CUtensorMap>(num_tensors);
    act_input = reserve<CUtensorMap>(num_tensors);
    output_rowwise = reserve<CUtensorMap>(num_tensors);
    output_colwise = reserve<CUtensorMap>(num_tensors);
    rows = reserve<size_t>(num_tensors);
    cols = reserve<size_t>(num_tensors);
    offsets = reserve<size_t>(num_tensors);
  }

  size_t size() const { return size_; }
  GroupedTmaStorageView view(void *data) const {
    if (size_ == 0) return {};
    NVTE_CHECK(
        data != nullptr && reinterpret_cast<uintptr_t>(data) % kGroupedTmaWorkspaceAlignment == 0,
        "Grouped TMA workspace must have a 128-byte aligned base pointer.");
    auto *base = static_cast<char *>(data);
    return {reinterpret_cast<CUtensorMap *>(base + input),
            reinterpret_cast<CUtensorMap *>(base + act_input),
            reinterpret_cast<CUtensorMap *>(base + output_rowwise),
            reinterpret_cast<CUtensorMap *>(base + output_colwise),
            reinterpret_cast<size_t *>(base + rows),
            reinterpret_cast<size_t *>(base + cols),
            reinterpret_cast<size_t *>(base + offsets)};
  }

  size_t input, act_input, output_rowwise, output_colwise, rows, cols, offsets;

 private:
  template <typename T>
  size_t reserve(size_t count) {
    constexpr size_t max_bytes = std::numeric_limits<size_t>::max();
    NVTE_CHECK(count <= max_bytes / sizeof(T), "Grouped TMA workspace size overflows size_t.");
    const size_t bytes = count * sizeof(T);
    const size_t padding = (kGroupedTmaWorkspaceAlignment - bytes % kGroupedTmaWorkspaceAlignment) %
                           kGroupedTmaWorkspaceAlignment;
    NVTE_CHECK(bytes <= max_bytes - padding && bytes + padding <= max_bytes - size_,
               "Grouped TMA workspace size overflows size_t.");
    const size_t offset = size_;
    size_ += bytes + padding;
    return offset;
  }
  size_t size_ = 0;
};

inline size_t get_grouped_tma_workspace_size(size_t num_tensors) {
  // One byte distinguishes execution from an empty-tensor size query when the
  // selected kernel needs no descriptor arrays.
  return std::max(size_t{1}, GroupedTmaWorkspaceLayout(num_tensors).size());
}

inline size_t grouped_tma_workspace_bytes(const GroupedTensor &tensor) {
  return tensor.all_same_shape() || tensor.all_same_last_dim()
             ? 0
             : get_grouped_tma_workspace_size(tensor.num_tensors);
}

inline GroupedTmaStorageView get_grouped_tma_storage(Tensor *workspace, size_t num_tensors) {
  if (num_tensors == 0) return {};
  return GroupedTmaWorkspaceLayout(num_tensors).view(workspace->data.dptr);
}

// The caller owns the buffer and orders all uses of it. Separate independently
// executing streams/graphs must receive separate mutable storage.
template <typename Launch>
void dispatch_grouped_tma_workspace(NVTETensor workspace, size_t bytes, Launch &&launch) {
  if (workspace == nullptr) {
    NVTE_CHECK(bytes == 0,
               "Grouped MXFP8 with varying last dimensions requires caller-owned workspace. "
               "Use the workspace-enabled _v2 API (e.g. nvte_group_quantize_v2).");
    launch(nullptr);
    return;
  }
  auto *tensor = convertNVTETensorCheck(workspace);
  const size_t required_bytes = std::max(size_t{1}, bytes);
  if (tensor->data.dptr == nullptr) {
    // Preserve backend input validation before reporting the size. Empty
    // workspace calls return before launching GPU work.
    launch(tensor);
    tensor->data.dtype = DType::kByte;
    tensor->data.shape = {required_bytes};
    return;
  }
  NVTE_CHECK(tensor->data.buffer_size_bytes() >= required_bytes,
             "Workspace tensor is smaller than the queried allocation requirement.");
  NVTE_CHECK(
      bytes == 0 ||
          reinterpret_cast<uintptr_t>(tensor->data.dptr) % kGroupedTmaWorkspaceAlignment == 0,
      "Workspace allocation does not satisfy the queried base alignment.");
  launch(tensor);
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_CORE_GROUPED_TMA_WORKSPACE_H_
