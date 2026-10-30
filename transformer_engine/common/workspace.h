/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_WORKSPACE_H_
#define TRANSFORMER_ENGINE_COMMON_WORKSPACE_H_

#include <transformer_engine/workspace.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "util/cuda_context.h"
#include "util/workspace.h"

namespace transformer_engine {

// Plans share this object with the active catalog. Abandoning a dead context's
// event must affect every copy, including snapshots destroyed after recovery.
struct WorkspaceReadyEvent {
  cudaEvent_t event = nullptr;
  cuda_context::Generation context;
  ~WorkspaceReadyEvent() {
    if (event != nullptr && context.current()) cudaEventDestroy(event);
  }
};

struct WorkspaceRegionRecord {
  std::string name;
  std::vector<unsigned char> key;
  WorkspaceBlock block;
  // Additional stream slots follow block at this stride; zero for shared tables.
  size_t slot_stride = 0;
  NVTEWorkspaceLifetime lifetime;
  uint64_t initialized_generation = 0;
  std::shared_ptr<WorkspaceReadyEvent> ready;
  std::array<bool, kMaxWorkspaceStreamSlots> ready_streams{};
};

class Workspace;

// Plans contain proposed metadata only, never device pointers or ownership.
class WorkspacePlan {
 public:
  explicit WorkspacePlan(const Workspace &workspace);
  NVTEWorkspaceRegion require(const char *name, const void *key, size_t key_size, size_t bytes,
                              size_t alignment, NVTEWorkspaceLifetime lifetime);
  size_t size() const { return layout_.size(); }
  size_t alignment() const { return layout_.alignment(); }

 private:
  friend class Workspace;
  const Workspace *owner_;
  uint64_t revision_;
  WorkspaceLayout layout_;
  std::vector<WorkspaceRegionRecord> regions_;
};

// Host-side catalog for a caller-owned device allocation. The owner serializes
// metadata access and keeps current/retired allocations alive for their users.
class Workspace : public cuda_context::Resource {
 public:
  Workspace() = default;
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;
  void commit(const WorkspacePlan &plan);
  void bind(void *data);
  bool planned() const { return planned_; }
  size_t size() const { return layout_.size(); }
  size_t alignment() const { return layout_.alignment(); }
  uint64_t generation() const { return generation_; }
  size_t num_regions() const { return regions_.size(); }
  NVTEWorkspaceRegion find(const char *name, const void *key = nullptr, size_t key_size = 0) const;
  NVTEWorkspaceRegionInfo info(NVTEWorkspaceRegion region) const;
  void *data(NVTEWorkspaceRegion region) const;
  void *data(NVTEWorkspaceRegion region, cudaStream_t stream);
  // Check the host reservation, never the size of the caller's allocation.
  NVTEWorkspaceRegion require_planned(const char *name, size_t bytes) const;
  size_t stream_slot(cudaStream_t stream);
  void begin_use(cudaStream_t stream) { stream_slot(stream); }
  bool initialized(NVTEWorkspaceRegion region) const;
  void mark_initialized(NVTEWorkspaceRegion region, cudaStream_t stream);
  void initialize(NVTEWorkspaceRegion region, cudaStream_t stream,
                  NVTEWorkspaceInitializer initializer, void *user_data);
  void invalidate(NVTEWorkspaceRegion region);
  void invalidate_context() noexcept override;

 private:
  friend class WorkspacePlan;
  const WorkspaceRegionRecord &record(NVTEWorkspaceRegion region) const;
  WorkspaceRegionRecord &record(NVTEWorkspaceRegion region);
  WorkspaceLayout layout_;
  std::vector<WorkspaceRegionRecord> regions_;
  char *data_ = nullptr;
  bool planned_ = false;
  bool bound_ = false;
  uint64_t revision_ = 0;
  uint64_t generation_ = 0;
  int device_ = -1;
  size_t num_streams_ = 0;
  std::array<uint64_t, kMaxWorkspaceStreamSlots> stream_ids_{};
};

}  // namespace transformer_engine

struct NVTEWorkspacePlanImpl : transformer_engine::WorkspacePlan {
  explicit NVTEWorkspacePlanImpl(const transformer_engine::Workspace &workspace)
      : transformer_engine::WorkspacePlan(workspace) {}
};

namespace transformer_engine {

// NVTETensor is an allocator handle, not a pointer to Workspace.
inline Workspace &convertNVTETensorToWorkspace(NVTETensor workspace) {
  auto *tensor = convertNVTETensorCheck(workspace);
  if (!tensor->workspace) {
    tensor->workspace = std::make_shared<Workspace>();
    if (tensor->data.dptr != nullptr) {
      tensor->workspace->bind(tensor->data.dptr);
    }
  }
  return *tensor->workspace;
}

inline WorkspacePlan &convertNVTEWorkspacePlan(NVTEWorkspacePlan plan) {
  NVTE_CHECK(plan != nullptr, "Workspace plan must not be NULL.");
  return *plan;
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_WORKSPACE_H_
