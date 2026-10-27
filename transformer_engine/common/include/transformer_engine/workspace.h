/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file workspace.h
 *  \brief Caller-owned device storage with a host-side region catalog.
 */

#ifndef TRANSFORMER_ENGINE_WORKSPACE_H_
#define TRANSFORMER_ENGINE_WORKSPACE_H_

#include <stddef.h>
#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
#include <utility>
extern "C" {
#endif

/*! A proposed layout. Keep its source workspace alive until the plan is destroyed. */
typedef struct NVTEWorkspacePlanImpl *NVTEWorkspacePlan;
/*! Workspace-local stable region ID. Zero denotes absent/zero-sized storage. */
typedef uint64_t NVTEWorkspaceRegion;

typedef enum {
  kNVTEWorkspaceScratch = 0,
  kNVTEWorkspacePersistent = 1,
} NVTEWorkspaceLifetime;

/*! Region reservation, in bytes. Name/key pointers are borrowed until the next
 *  layout commit or workspace destruction. Keys are compared byte for byte;
 *  producers must use a canonical encoding without uninitialized structure padding.
 */
typedef struct {
  const char *name;
  const void *key;
  size_t key_size;
  size_t offset;
  size_t size;
  size_t alignment;
  NVTEWorkspaceLifetime lifetime;
} NVTEWorkspaceRegionInfo;

/*! Workspace metadata is attached lazily to an ordinary NVTETensor. Use
 *  nvte_create_tensor/nvte_destroy_tensor or TensorWrapper for its lifetime.
 *  Committing sets its rowwise shape/dtype to the required byte storage.
 *  Setting rowwise data through either tensor setter binds a new generation,
 *  even for the same address. Keep the tensor alive for all plans and uses.
 */

/*! Snapshot the active layout. Planning does not change the active buffer/catalog. */
NVTEWorkspacePlan nvte_create_workspace_plan(NVTETensor workspace);
void nvte_destroy_workspace_plan(NVTEWorkspacePlan plan);

/*! Reserve a named, keyed region. Names/keys are copied from host memory. Existing reservations
 *  are reused when sufficient; growing one appends its replacement and preserves
 *  other regions' offsets. Alignment is a power of two, with a minimum of 128.
 *  A zero-byte request returns zero without changing the plan.
 */
NVTEWorkspaceRegion nvte_workspace_plan_require(NVTEWorkspacePlan plan, const char *name,
                                                const void *key, size_t key_size, size_t bytes,
                                                size_t alignment, NVTEWorkspaceLifetime lifetime);
size_t nvte_workspace_plan_size(NVTEWorkspacePlan plan);
size_t nvte_workspace_plan_alignment(NVTEWorkspacePlan plan);

/*! Commit a plan from this workspace's current revision. The caller first
 *  arranges sufficient storage. Unchanged regions retain initialization state;
 *  new/relocated regions are uninitialized. A committed plan cannot be committed again.
 *  To replace the allocation, commit then bind the new buffer before execution.
 */
void nvte_workspace_commit_plan(NVTETensor workspace, NVTEWorkspacePlan plan);

/*! Bind a new caller-owned allocation. Every call starts a new generation and
 *  invalidates all persistent contents, even when the pointer value is unchanged.
 *  Reusing the same live allocation does not require rebinding. Neither capacity
 *  nor base alignment is checked. No contents are initialized, copied, or freed.
 *  The caller keeps old allocations alive until their outstanding uses finish.
 */
void nvte_workspace_bind(NVTETensor workspace, void *data);
size_t nvte_workspace_size(NVTETensor workspace);
size_t nvte_workspace_alignment(NVTETensor workspace);
uint64_t nvte_workspace_generation(NVTETensor workspace);
size_t nvte_workspace_num_regions(NVTETensor workspace);

/*! Find a region by name and exact key; returns zero if absent. IDs 1 through
 *  num_regions can also be used to inspect the current catalog.
 */
NVTEWorkspaceRegion nvte_workspace_find_region(NVTETensor workspace, const char *name,
                                               const void *key, size_t key_size);
NVTEWorkspaceRegionInfo nvte_workspace_get_region_info(NVTETensor workspace,
                                                       NVTEWorkspaceRegion region);
/*! Resolve base + offset. Region zero resolves to NULL. */
void *nvte_workspace_get_region_data(NVTETensor workspace, NVTEWorkspaceRegion region);

/*! Establish one CUDA stream for this allocation generation. Subsequent uses
 *  must use the same stream; use a separate workspace for another stream.
 *  No CUDA calls or synchronization are performed. Host access is serialized by
 *  the caller. Plan/allocate/initialize before graph capture; execution may resolve
 *  pointers and reuse prepared regions during capture.
 */
void nvte_workspace_begin_use(NVTETensor workspace, cudaStream_t stream);
/*! True means initialization was enqueued in the workspace stream for the
 *  current generation. It does not imply GPU completion. Only persistent regions
 *  can be marked initialized. Enqueue the producer successfully before marking.
 */
int nvte_workspace_region_initialized(NVTETensor workspace, NVTEWorkspaceRegion region);
void nvte_workspace_mark_region_initialized(NVTETensor workspace, NVTEWorkspaceRegion region,
                                            cudaStream_t stream);
void nvte_workspace_invalidate_region(NVTETensor workspace, NVTEWorkspaceRegion region);

#ifdef __cplusplus
}  // extern "C"

namespace transformer_engine {

/*! Move-only owner of a layout plan, with no ownership of device storage. */
class WorkspacePlanWrapper {
 public:
  explicit WorkspacePlanWrapper(NVTETensor workspace)
      : plan_(nvte_create_workspace_plan(workspace)) {}
  explicit WorkspacePlanWrapper(NVTEWorkspacePlan plan) : plan_(plan) {}
  ~WorkspacePlanWrapper() { nvte_destroy_workspace_plan(plan_); }
  WorkspacePlanWrapper(const WorkspacePlanWrapper &) = delete;
  WorkspacePlanWrapper &operator=(const WorkspacePlanWrapper &) = delete;
  WorkspacePlanWrapper(WorkspacePlanWrapper &&other) noexcept
      : plan_(std::exchange(other.plan_, nullptr)) {}
  WorkspacePlanWrapper &operator=(WorkspacePlanWrapper &&other) noexcept {
    if (this != &other) {
      nvte_destroy_workspace_plan(plan_);
      plan_ = std::exchange(other.plan_, nullptr);
    }
    return *this;
  }
  NVTEWorkspacePlan data() const { return plan_; }
  NVTEWorkspaceRegion require(const char *name, size_t bytes,
                              NVTEWorkspaceLifetime lifetime = kNVTEWorkspaceScratch,
                              size_t alignment = 128, const void *key = nullptr,
                              size_t key_size = 0) {
    return nvte_workspace_plan_require(plan_, name, key, key_size, bytes, alignment, lifetime);
  }
  size_t size() const { return nvte_workspace_plan_size(plan_); }
  size_t alignment() const { return nvte_workspace_plan_alignment(plan_); }

 private:
  NVTEWorkspacePlan plan_;
};

/*! Optional convenience wrapper around TensorWrapper and the workspace functions.
 *  Plain NVTETensor/TensorWrapper handles support the same protocol. Device
 *  storage remains caller-owned; destroying the tensor releases host metadata.
 */
class WorkspaceWrapper {
 public:
  WorkspaceWrapper() = default;
  WorkspaceWrapper(const WorkspaceWrapper &) = delete;
  WorkspaceWrapper &operator=(const WorkspaceWrapper &) = delete;
  WorkspaceWrapper(WorkspaceWrapper &&) = default;
  WorkspaceWrapper &operator=(WorkspaceWrapper &&) = default;
  NVTETensor data() const { return workspace_.data(); }
  WorkspacePlanWrapper plan() const { return WorkspacePlanWrapper(data()); }
  void commit(const WorkspacePlanWrapper &plan) { nvte_workspace_commit_plan(data(), plan.data()); }
  void commit(const WorkspacePlanWrapper &plan, void *new_data) {
    commit(plan);
    bind(new_data);
  }
  void bind(void *new_data) { nvte_workspace_bind(data(), new_data); }
  size_t size() const { return nvte_workspace_size(data()); }
  size_t alignment() const { return nvte_workspace_alignment(data()); }
  uint64_t generation() const { return nvte_workspace_generation(data()); }
  size_t num_regions() const { return nvte_workspace_num_regions(data()); }
  NVTEWorkspaceRegion find(const char *name, const void *key = nullptr, size_t key_size = 0) const {
    return nvte_workspace_find_region(data(), name, key, key_size);
  }
  NVTEWorkspaceRegionInfo info(NVTEWorkspaceRegion region) const {
    return nvte_workspace_get_region_info(data(), region);
  }
  void begin_use(cudaStream_t stream) { nvte_workspace_begin_use(data(), stream); }
  template <typename T = unsigned char>
  T *get(NVTEWorkspaceRegion region) const {
    return static_cast<T *>(nvte_workspace_get_region_data(data(), region));
  }
  bool initialized(NVTEWorkspaceRegion region) const {
    return nvte_workspace_region_initialized(data(), region) != 0;
  }
  void mark_initialized(NVTEWorkspaceRegion region, cudaStream_t stream) {
    nvte_workspace_mark_region_initialized(data(), region, stream);
  }
  void invalidate(NVTEWorkspaceRegion region) { nvte_workspace_invalidate_region(data(), region); }

 private:
  TensorWrapper workspace_;
};

}  // namespace transformer_engine
#endif  // __cplusplus

#endif  // TRANSFORMER_ENGINE_WORKSPACE_H_
