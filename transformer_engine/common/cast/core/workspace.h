/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_CAST_CORE_WORKSPACE_H_
#define TRANSFORMER_ENGINE_COMMON_CAST_CORE_WORKSPACE_H_

#include <algorithm>
#include <limits>

#include "../../common.h"
#include "../../workspace.h"

namespace transformer_engine {
namespace dispatch {
namespace common {

constexpr const char *kDBiasWorkspaceRegion = "te.dbias";
constexpr const char *kGroupedTmaWorkspaceRegion = "te.grouped_tma";

// Translate the existing backend's size query into a byte reservation. This
// checks arithmetic in the requirements, never the user's allocation capacity.
inline size_t workspace_requirement_bytes(const TensorWrapper &requirement) {
  size_t bytes = typeToSize(requirement.dtype());
  const auto shape = requirement.shape();
  for (size_t i = 0; i < shape.ndim; ++i) {
    NVTE_CHECK(shape.data[i] == 0 || bytes <= std::numeric_limits<size_t>::max() / shape.data[i],
               "Workspace size requirement overflows size_t.");
    bytes *= shape.data[i];
  }
  return bytes;
}

// An empty tensor queries requirements and commits host metadata only. Existing
// reservations remain in place, so several operations can query the same empty
// tensor before the caller allocates its final buffer.
inline void query_workspace(NVTETensor workspace, const char *name, size_t bytes,
                            const TensorWrapper &requirement,
                            NVTEWorkspaceLifetime lifetime = kNVTEWorkspaceScratch) {
  auto &storage = convertNVTETensorToWorkspace(workspace);
  const auto region = storage.find(name);
  if (!storage.planned() || (bytes != 0 && (region == 0 || storage.info(region).size < bytes))) {
    WorkspacePlan plan(storage);
    plan.require(name, nullptr, 0, bytes, 128, lifetime);
    storage.commit(plan);
  }
  auto &data = convertNVTETensorCheck(workspace)->data;
  data.dtype = requirement.dtype();
  const auto shape = requirement.shape();
  const size_t required_bytes = workspace_requirement_bytes(requirement);
  if (required_bytes != 0 && required_bytes >= storage.size()) {
    // Preserve the selected backend's dtype/shape when it covers the catalog.
    data.shape = Shape(shape);
  } else {
    const size_t element_bytes = typeToSize(requirement.dtype());
    const size_t total_bytes = std::max(size_t{1}, storage.size());
    data.shape = {total_bytes / element_bytes + (total_bytes % element_bytes != 0)};
  }
}

// A fresh tensor over a scratch-only allocation can reconstruct the operation's
// layout. Multi-stream reuse must retain the tensor/catalog so stream slots are
// shared by all calls, including calls from different host threads.
inline Workspace &execution_workspace(NVTETensor workspace, const char *name, size_t bytes,
                                      cudaStream_t stream,
                                      NVTEWorkspaceLifetime lifetime = kNVTEWorkspaceScratch) {
  const auto *tensor = convertNVTETensorCheck(workspace);
  auto &storage = convertNVTETensorToWorkspace(workspace);
  if (!storage.planned()) {
    WorkspacePlan plan(storage);
    plan.require(name, nullptr, 0, bytes, 128, lifetime);
    storage.commit(plan);
  }
  storage.require_planned(name, bytes);
  NVTE_CHECK(tensor->data.buffer_size_bytes() >= storage.size(),
             "Workspace tensor is smaller than the queried allocation requirement.");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(tensor->data.dptr) % storage.alignment() == 0,
             "Workspace allocation does not satisfy the queried base alignment.");
  // Nonempty regions register/resolve the stream once in data(region, stream).
  if (bytes == 0) storage.begin_use(stream);
  return storage;
}

inline void bind_dbias_workspace(Workspace &workspace, TensorWrapper *requirement,
                                 cudaStream_t stream) {
  const auto region =
      workspace.require_planned(kDBiasWorkspaceRegion, workspace_requirement_bytes(*requirement));
  requirement->set_rowwise_data(workspace.data(region, stream), requirement->dtype(),
                                requirement->shape());
}

template <typename Launch>
void dispatch_dbias_workspace(NVTETensor workspace, cudaStream_t stream, Launch &&launch) {
  const auto *tensor = convertNVTETensorCheck(workspace);
  if (tensor->data.dptr == nullptr) {
    TensorWrapper requirement;
    launch(requirement.data());
    query_workspace(workspace, kDBiasWorkspaceRegion, workspace_requirement_bytes(requirement),
                    requirement);
    return;
  }
  // The public tensor describes the whole allocation. Recover the backend's
  // typed view even for a fresh handle (e.g. JAX), since the queried allocation
  // may include final alignment padding and have a different shape.
  TensorWrapper adapter;
  launch(adapter.data());
  auto &storage = execution_workspace(workspace, kDBiasWorkspaceRegion,
                                      workspace_requirement_bytes(adapter), stream);
  bind_dbias_workspace(storage, &adapter, stream);
  launch(adapter.data());
}

}  // namespace common
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_CAST_CORE_WORKSPACE_H_
