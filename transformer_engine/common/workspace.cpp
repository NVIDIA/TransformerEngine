/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "workspace.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>

namespace transformer_engine {
namespace {

void check_key(const char *name, const void *key, size_t key_size) {
  NVTE_CHECK(name != nullptr && name[0] != '\0', "Workspace region name must not be empty.");
  NVTE_CHECK(key_size == 0 || key != nullptr, "Workspace key data must not be NULL.");
}

bool matches(const WorkspaceRegionRecord &record, const char *name, const void *key,
             size_t key_size) {
  return record.name == name && record.key.size() == key_size &&
         (key_size == 0 || std::memcmp(record.key.data(), key, key_size) == 0);
}

bool same_block(const WorkspaceBlock &a, const WorkspaceBlock &b) {
  return a.offset == b.offset && a.size == b.size && a.alignment == b.alignment;
}

void reserve_region(WorkspaceLayout &layout, WorkspaceRegionRecord &region, size_t bytes,
                    size_t alignment) {
  region.block = layout.add_bytes(bytes, alignment);
  region.initialized_generation = 0;
  region.ready.reset();
}

uint64_t next_counter(uint64_t value) {
  NVTE_CHECK(value < std::numeric_limits<uint64_t>::max(), "Workspace counter overflow.");
  return value + 1;
}

}  // namespace

WorkspacePlan::WorkspacePlan(const Workspace &workspace)
    : owner_(&workspace),
      revision_(workspace.revision_),
      layout_(workspace.layout_),
      regions_(workspace.regions_) {}

NVTEWorkspaceRegion WorkspacePlan::require(const char *name, const void *key, size_t key_size,
                                           size_t bytes, size_t alignment,
                                           NVTEWorkspaceLifetime lifetime) {
  check_key(name, key, key_size);
  NVTE_CHECK(lifetime == kNVTEWorkspaceScratch || lifetime == kNVTEWorkspacePersistent ||
                 lifetime == kNVTEWorkspaceStreamPersistent,
             "Invalid workspace region lifetime.");
  NVTE_CHECK(alignment != 0 && (alignment & (alignment - 1)) == 0,
             "Workspace alignment must be a nonzero power of two.");
  alignment = std::max(alignment, kWorkspaceAlignment);
  if (bytes == 0) {
    return 0;
  }
  for (size_t i = 0; i < regions_.size(); ++i) {
    auto &region = regions_[i];
    if (!matches(region, name, key, key_size)) {
      continue;
    }
    NVTE_CHECK(region.lifetime == lifetime, "Workspace region lifetime cannot change.");
    if (region.block.size >= bytes && region.block.alignment >= alignment) {
      return i + 1;
    }
    WorkspaceLayout proposed = layout_;
    reserve_region(proposed, region, std::max(bytes, region.block.size),
                   std::max(alignment, region.block.alignment));
    layout_ = proposed;
    return i + 1;
  }
  NVTE_CHECK(regions_.size() < std::numeric_limits<NVTEWorkspaceRegion>::max(),
             "Too many workspace regions.");
  WorkspaceLayout proposed = layout_;
  WorkspaceRegionRecord region;
  region.name = name;
  if (key_size != 0) {
    const auto *bytes_key = static_cast<const unsigned char *>(key);
    region.key.assign(bytes_key, bytes_key + key_size);
  }
  region.lifetime = lifetime;
  reserve_region(proposed, region, bytes, alignment);
  regions_.push_back(std::move(region));
  layout_ = proposed;
  return regions_.size();
}

void Workspace::commit(const WorkspacePlan &plan) {
  NVTE_CHECK(plan.owner_ == this && plan.revision_ == revision_,
             "Workspace plan belongs to another workspace or is stale.");
  const auto revision = next_counter(revision_);
  auto regions = plan.regions_;
  // Initialization can be enqueued after taking the plan snapshot. Preserve the
  // current state of unchanged regions rather than the snapshot's older state.
  for (size_t i = 0; i < regions_.size(); ++i) {
    if (same_block(regions[i].block, regions_[i].block)) {
      regions[i].initialized_generation = regions_[i].initialized_generation;
      regions[i].ready = regions_[i].ready;
    }
  }
  regions_ = std::move(regions);
  layout_ = plan.layout_;
  revision_ = revision;
  planned_ = true;
}

void Workspace::bind(void *data) {
  const auto generation = next_counter(generation_);
  const auto revision = next_counter(revision_);
  data_ = static_cast<char *>(data);
  bound_ = true;
  generation_ = generation;
  revision_ = revision;
  for (auto &region : regions_) {
    region.initialized_generation = 0;
    region.ready.reset();
  }
}

NVTEWorkspaceRegion Workspace::find(const char *name, const void *key, size_t key_size) const {
  check_key(name, key, key_size);
  for (size_t i = 0; i < regions_.size(); ++i) {
    if (matches(regions_[i], name, key, key_size)) {
      return i + 1;
    }
  }
  return 0;
}

const WorkspaceRegionRecord &Workspace::record(NVTEWorkspaceRegion region) const {
  NVTE_CHECK(region != 0 && region <= regions_.size(), "Invalid workspace region ID.");
  return regions_[region - 1];
}

WorkspaceRegionRecord &Workspace::record(NVTEWorkspaceRegion region) {
  return const_cast<WorkspaceRegionRecord &>(std::as_const(*this).record(region));
}

NVTEWorkspaceRegionInfo Workspace::info(NVTEWorkspaceRegion region) const {
  const auto &entry = record(region);
  return {entry.name.c_str(), entry.key.data(),      entry.key.size(), entry.block.offset,
          entry.block.size,   entry.block.alignment, entry.lifetime};
}

void *Workspace::data(NVTEWorkspaceRegion region) const {
  if (region == 0) {
    return nullptr;
  }
  const auto &entry = record(region);
  NVTE_CHECK(bound_ && data_ != nullptr, "Bind workspace storage before resolving a region.");
  return data_ + entry.block.offset;
}

NVTEWorkspaceRegion Workspace::require_planned(const char *name, size_t bytes) const {
  if (bytes == 0) {
    return 0;
  }
  const auto region = find(name);
  NVTE_CHECK(region != 0 && record(region).block.size >= bytes, "Workspace region '", name,
             "' needs an empty-tensor size query before execution.");
  return region;
}

bool Workspace::initialized(NVTEWorkspaceRegion region) const {
  if (region == 0) {
    return false;
  }
  const auto &entry = record(region);
  return generation_ != 0 && entry.initialized_generation == generation_;
}

void Workspace::mark_initialized(NVTEWorkspaceRegion region, cudaStream_t stream) {
  auto &entry = record(region);
  NVTE_CHECK(entry.lifetime == kNVTEWorkspacePersistent,
             "Only shared immutable workspace regions have initialization state.");
  NVTE_CHECK(!initialized(region), "Immutable workspace region is already initialized.");
  NVTE_CHECK(bound_ && data_ != nullptr, "Bind workspace storage before initialization.");
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  NVTE_CHECK(capture == cudaStreamCaptureStatusNone,
             "Initialize immutable workspace regions before CUDA graph capture.");
  auto ready = std::shared_ptr<cudaEvent_t>(new cudaEvent_t{}, [](cudaEvent_t *event) {
    if (*event != nullptr) cudaEventDestroy(*event);
    delete event;
  });
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(ready.get(), cudaEventDisableTiming));
  NVTE_CHECK_CUDA(cudaEventRecord(*ready, stream));
  entry.ready = std::move(ready);
  entry.initialized_generation = generation_;
}

void Workspace::initialize(NVTEWorkspaceRegion region, cudaStream_t stream,
                           NVTEWorkspaceInitializer initializer, void *user_data) {
  auto &entry = record(region);
  NVTE_CHECK(entry.lifetime == kNVTEWorkspacePersistent,
             "Only shared immutable workspace regions support lazy initialization.");
  NVTE_CHECK(bound_ && data_ != nullptr, "Bind workspace storage before initialization.");
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  if (initialized(region)) {
    if (capture != cudaStreamCaptureStatusNone) {
      // Initialization is outside the graph. Require completion so replay
      // never depends on uncaptured work remaining on the original stream.
      NVTE_CHECK(cudaEventQuery(*entry.ready) == cudaSuccess,
                 "Complete immutable workspace initialization before CUDA graph capture.");
    }
    return;
  }
  NVTE_CHECK(capture == cudaStreamCaptureStatusNone,
             "Initialize immutable workspace regions before CUDA graph capture.");
  NVTE_CHECK(initializer != nullptr, "An uninitialized region requires an initializer.");
  initializer(data(region), entry.block.size, stream, user_data);
  mark_initialized(region, stream);
}

void Workspace::invalidate(NVTEWorkspaceRegion region) {
  auto &entry = record(region);
  entry.initialized_generation = 0;
  entry.ready.reset();
}

}  // namespace transformer_engine

using transformer_engine::convertNVTETensorToWorkspace;
using transformer_engine::convertNVTEWorkspacePlan;

NVTEWorkspacePlan nvte_create_workspace_plan(NVTETensor workspace) {
  return new NVTEWorkspacePlanImpl(convertNVTETensorToWorkspace(workspace));
}
void nvte_destroy_workspace_plan(NVTEWorkspacePlan plan) { delete plan; }
NVTEWorkspaceRegion nvte_workspace_plan_require(NVTEWorkspacePlan plan, const char *name,
                                                const void *key, size_t key_size, size_t bytes,
                                                size_t alignment, NVTEWorkspaceLifetime lifetime) {
  return convertNVTEWorkspacePlan(plan).require(name, key, key_size, bytes, alignment, lifetime);
}
size_t nvte_workspace_plan_size(NVTEWorkspacePlan plan) {
  return convertNVTEWorkspacePlan(plan).size();
}
size_t nvte_workspace_plan_alignment(NVTEWorkspacePlan plan) {
  return convertNVTEWorkspacePlan(plan).alignment();
}
void nvte_workspace_commit_plan(NVTETensor workspace, NVTEWorkspacePlan plan) {
  auto &storage = convertNVTETensorToWorkspace(workspace);
  storage.commit(convertNVTEWorkspacePlan(plan));
  auto &data = transformer_engine::convertNVTETensorCheck(workspace)->data;
  data.shape = {storage.size()};
  data.dtype = transformer_engine::DType::kByte;
}
void nvte_workspace_bind(NVTETensor workspace, void *data) {
  auto &storage = convertNVTETensorToWorkspace(workspace);
  const size_t bytes = storage.size();
  const NVTEBasicTensor buffer{data, kNVTEByte, nvte_make_shape(&bytes, 1)};
  nvte_set_tensor_param_v2(workspace, kNVTERowwiseData, &buffer, sizeof(buffer));
}
size_t nvte_workspace_size(NVTETensor workspace) {
  return convertNVTETensorToWorkspace(workspace).size();
}
size_t nvte_workspace_alignment(NVTETensor workspace) {
  return convertNVTETensorToWorkspace(workspace).alignment();
}
uint64_t nvte_workspace_generation(NVTETensor workspace) {
  return convertNVTETensorToWorkspace(workspace).generation();
}
size_t nvte_workspace_num_regions(NVTETensor workspace) {
  return convertNVTETensorToWorkspace(workspace).num_regions();
}
NVTEWorkspaceRegion nvte_workspace_find_region(NVTETensor workspace, const char *name,
                                               const void *key, size_t key_size) {
  return convertNVTETensorToWorkspace(workspace).find(name, key, key_size);
}
NVTEWorkspaceRegionInfo nvte_workspace_get_region_info(NVTETensor workspace,
                                                       NVTEWorkspaceRegion region) {
  return convertNVTETensorToWorkspace(workspace).info(region);
}
void *nvte_workspace_get_region_data(NVTETensor workspace, NVTEWorkspaceRegion region) {
  return convertNVTETensorToWorkspace(workspace).data(region);
}
void nvte_workspace_initialize_region(NVTETensor workspace, NVTEWorkspaceRegion region,
                                      cudaStream_t stream, NVTEWorkspaceInitializer initializer,
                                      void *user_data) {
  convertNVTETensorToWorkspace(workspace).initialize(region, stream, initializer, user_data);
}
int nvte_workspace_region_initialized(NVTETensor workspace, NVTEWorkspaceRegion region) {
  return convertNVTETensorToWorkspace(workspace).initialized(region);
}
void nvte_workspace_mark_region_initialized(NVTETensor workspace, NVTEWorkspaceRegion region,
                                            cudaStream_t stream) {
  convertNVTETensorToWorkspace(workspace).mark_initialized(region, stream);
}
void nvte_workspace_invalidate_region(NVTETensor workspace, NVTEWorkspaceRegion region) {
  convertNVTETensorToWorkspace(workspace).invalidate(region);
}
