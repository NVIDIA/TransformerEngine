/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <transformer_engine/workspace.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace transformer_engine;

TEST(WorkspaceApiTest, EmptyAndOptionalRegions) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  EXPECT_EQ(plan.require("optional", 0), 0);
  EXPECT_EQ(plan.size(), 0);
  EXPECT_EQ(plan.alignment(), 128);
  workspace.commit(plan);
  EXPECT_EQ(workspace.size(), 0);
  EXPECT_EQ(workspace.generation(), 0);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 0);
  EXPECT_EQ(workspace.get(0), nullptr);
  EXPECT_FALSE(workspace.initialized(0));
  EXPECT_THROW(workspace.initialized(1), std::runtime_error);
  EXPECT_EQ(nvte_workspace_find_region(workspace.data(), "missing", nullptr, 0), 0);
}

TEST(WorkspaceApiTest, QueryIsIsolatedAndCanBeDiscarded) {
  WorkspaceWrapper workspace;
  {
    auto plan = workspace.plan();
    plan.require("dbias", 129);
    plan.require("gelu", 257, kNVTEWorkspacePersistent);
    EXPECT_EQ(plan.size(), kMaxWorkspaceStreamSlots * 256 + 384);
    EXPECT_EQ(workspace.size(), 0);
    EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 0);
  }
  EXPECT_EQ(workspace.size(), 0);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 0);
}

TEST(WorkspaceApiTest, AppendingTablesPreservesScratchOffsets) {
  WorkspaceWrapper workspace;
  auto first = workspace.plan();
  const auto dbias = first.require("dbias", 129);
  alignas(128) std::array<unsigned char, 2048> data{};
  workspace.commit(first, data.data());
  const auto dbias_offset = nvte_workspace_get_region_info(workspace.data(), dbias).offset;

  auto extended = workspace.plan();
  const auto table = extended.require("gelu", 257, kNVTEWorkspacePersistent);
  EXPECT_EQ(nvte_workspace_find_region(workspace.data(), "gelu", nullptr, 0), 0);
  workspace.commit(extended);
  EXPECT_EQ(workspace.get(dbias), data.data() + dbias_offset);
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), table).offset,
            kMaxWorkspaceStreamSlots * 256);
  EXPECT_EQ(workspace.size(), kMaxWorkspaceStreamSlots * 256 + 384);
  EXPECT_EQ(workspace.generation(), 1);
  EXPECT_FALSE(workspace.initialized(table));
}

TEST(WorkspaceApiTest, GrowingScratchPreservesPersistentTable) {
  WorkspaceWrapper workspace;
  auto first = workspace.plan();
  const auto dbias = first.require("dbias", 64);
  const auto table = first.require("gelu", 64, kNVTEWorkspacePersistent);
  alignas(128) std::array<unsigned char, 2048> data{};
  workspace.commit(first, data.data());
  workspace.get<float>(table)[0] = 7.5f;
  workspace.mark_initialized(table, nullptr);
  const auto table_offset = nvte_workspace_get_region_info(workspace.data(), table).offset;

  auto extended = workspace.plan();
  EXPECT_EQ(extended.require("dbias", 257), dbias);
  workspace.commit(extended);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 2);
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), table).offset, table_offset);
  EXPECT_GE(nvte_workspace_get_region_info(workspace.data(), dbias).offset, 256);
  EXPECT_TRUE(workspace.initialized(table));
  EXPECT_EQ(workspace.get<float>(table)[0], 7.5f);
}

TEST(WorkspaceApiTest, ReservationsAreReusedWithoutShrinking) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("table", 256, kNVTEWorkspacePersistent, 256);
  EXPECT_EQ(plan.require("table", 1, kNVTEWorkspacePersistent), table);
  EXPECT_EQ(plan.size(), 256);
  alignas(256) std::array<unsigned char, 512> data{};
  workspace.commit(plan, data.data());
  workspace.mark_initialized(table, nullptr);
  auto reused = workspace.plan();
  EXPECT_EQ(reused.require("table", 16, kNVTEWorkspacePersistent), table);
  workspace.commit(reused);
  EXPECT_TRUE(workspace.initialized(table));
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), table).size, 256);
  EXPECT_EQ(workspace.alignment(), 256);
}

TEST(WorkspaceApiTest, ExactKeysAndNamesAreCopied) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  char name[] = "activation";
  std::array<unsigned char, 3> key{1, 2, 3};
  const auto first = plan.require(name, 128, kNVTEWorkspacePersistent, 128, key.data(), key.size());
  name[0] = 'X';
  key[2] = 4;
  const auto second =
      plan.require("activation", 128, kNVTEWorkspacePersistent, 128, key.data(), key.size());
  EXPECT_NE(first, second);
  workspace.commit(plan);
  const std::array<unsigned char, 3> original{1, 2, 3};
  EXPECT_EQ(
      nvte_workspace_find_region(workspace.data(), "activation", original.data(), original.size()),
      first);
  EXPECT_EQ(nvte_workspace_find_region(workspace.data(), "activation", key.data(), key.size()),
            second);
  const auto info = nvte_workspace_get_region_info(workspace.data(), first);
  EXPECT_STREQ(info.name, "activation");
  EXPECT_EQ(info.key_size, original.size());
  EXPECT_EQ(static_cast<const unsigned char *>(info.key)[2], 3);
  EXPECT_EQ(info.lifetime, kNVTEWorkspacePersistent);
}

TEST(WorkspaceApiTest, RebindingAlwaysInvalidatesTables) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("gelu", 128, kNVTEWorkspacePersistent);
  alignas(128) std::array<unsigned char, 256> first{}, second{};
  workspace.commit(plan, first.data());
  workspace.mark_initialized(table, nullptr);
  EXPECT_TRUE(workspace.initialized(table));
  workspace.bind(first.data());
  EXPECT_EQ(workspace.generation(), 2);
  EXPECT_FALSE(workspace.initialized(table));
  workspace.mark_initialized(table, nullptr);
  workspace.bind(second.data());
  EXPECT_EQ(workspace.generation(), 3);
  EXPECT_FALSE(workspace.initialized(table));
  EXPECT_EQ(workspace.get(table), second.data());
}

TEST(WorkspaceApiTest, GrowingPersistentRegionInvalidatesOnlyThatRegion) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto first = plan.require("gelu", 64, kNVTEWorkspacePersistent);
  const auto second = plan.require("silu", 64, kNVTEWorkspacePersistent);
  alignas(128) std::array<unsigned char, 1024> data{};
  workspace.commit(plan, data.data());
  workspace.mark_initialized(first, nullptr);
  workspace.mark_initialized(second, nullptr);
  auto extended = workspace.plan();
  EXPECT_EQ(extended.require("gelu", 257, kNVTEWorkspacePersistent), first);
  workspace.commit(extended);
  EXPECT_FALSE(workspace.initialized(first));
  EXPECT_TRUE(workspace.initialized(second));
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), second).offset, 128);
}

TEST(WorkspaceApiTest, ExplicitMigrationAfterNewBinding) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("gelu", 64, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("dbias", 128);
  alignas(128) std::array<unsigned char, 1024> old_data{};
  workspace.commit(plan, old_data.data());
  workspace.get<float>(table)[0] = 7.5f;
  workspace.mark_initialized(table, nullptr);
  const auto *old_table = workspace.get(table);
  const size_t table_bytes = workspace.info(table).size;

  auto extended = workspace.plan();
  extended.require("tma", 257);
  alignas(128) std::array<unsigned char, 4096> new_data;
  new_data.fill(0xcc);
  workspace.commit(extended, new_data.data());
  EXPECT_EQ(workspace.find("gelu"), table);
  EXPECT_EQ(workspace.num_regions(), 3);
  EXPECT_FALSE(workspace.initialized(table));
  std::memcpy(workspace.get(table), old_table, table_bytes);
  workspace.mark_initialized(table, nullptr);
  EXPECT_TRUE(workspace.initialized(table));
  EXPECT_EQ(workspace.get<float>(table)[0], 7.5f);
  EXPECT_EQ(workspace.get(scratch)[0], 0xcc);
  workspace.invalidate(table);
  EXPECT_FALSE(workspace.initialized(table));
}

TEST(WorkspaceApiTest, CommitPreservesInitializationEnqueuedAfterSnapshot) {
  WorkspaceWrapper workspace;
  auto first = workspace.plan();
  const auto table = first.require("gelu", 128, kNVTEWorkspacePersistent);
  alignas(128) std::array<unsigned char, 1024> data{};
  workspace.commit(first, data.data());
  auto extended = workspace.plan();
  extended.require("dbias", 128);
  workspace.mark_initialized(table, nullptr);
  workspace.commit(extended);
  EXPECT_TRUE(workspace.initialized(table));
}

TEST(WorkspaceApiTest, StreamSlotsFollowFirstUseNotHandleOrder) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto scratch = plan.require("scratch", 129);
  const auto descriptors = plan.require("tma", 128, kNVTEWorkspaceStreamPersistent);
  const auto table = plan.require("gelu", 128, kNVTEWorkspacePersistent);
  alignas(128) std::array<unsigned char, 4096> data{};
  workspace.commit(plan, data.data());
  std::array<cudaStream_t, kMaxWorkspaceStreamSlots> streams{};
  for (auto &stream : streams) {
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  }
  EXPECT_EQ(nvte_workspace_max_stream_slots(), kMaxWorkspaceStreamSlots);
  // Deliberately register the last-created stream first and include stream zero.
  std::vector<cudaStream_t> order{streams.back(), nullptr};
  for (int i = kMaxWorkspaceStreamSlots - 2; i > 0; --i) order.push_back(streams[i]);
  ASSERT_EQ(order.size(), kMaxWorkspaceStreamSlots);
  for (size_t slot = 0; slot < order.size(); ++slot) {
    EXPECT_EQ(workspace.stream_slot(order[slot]), slot);
    EXPECT_EQ(workspace.get(scratch, order[slot]), workspace.get(scratch) + slot * 256);
    EXPECT_EQ(workspace.get(descriptors, order[slot]), workspace.get(descriptors) + slot * 128);
    EXPECT_EQ(workspace.get(table, order[slot]), workspace.get(table));
  }
  for (size_t slot = order.size(); slot-- > 0;) {
    EXPECT_EQ(workspace.stream_slot(order[slot]), slot);
  }
  EXPECT_THROW(workspace.begin_use(streams.front()), std::runtime_error);
  EXPECT_EQ(workspace.stream_slot(order.front()), 0);
  // Rebinding resets mapping, including when the numerical address is unchanged.
  workspace.bind(data.data());
  EXPECT_EQ(workspace.stream_slot(streams.front()), 0);
  for (auto stream : streams) EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(WorkspaceApiTest, InvalidationAndInitializationProtocol) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("gelu", 128, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("dbias", 128);
  workspace.commit(plan);
  EXPECT_THROW(workspace.get(table), std::runtime_error);
  EXPECT_THROW(workspace.mark_initialized(table, nullptr), std::runtime_error);
  alignas(128) std::array<unsigned char, 256> data{};
  workspace.bind(data.data());
  EXPECT_THROW(workspace.mark_initialized(scratch, nullptr), std::runtime_error);
  workspace.mark_initialized(table, nullptr);
  nvte_workspace_invalidate_region(workspace.data(), table);
  EXPECT_FALSE(workspace.initialized(table));
  EXPECT_THROW(nvte_workspace_get_region_info(workspace.data(), 0), std::runtime_error);
  EXPECT_THROW(workspace.get(3), std::runtime_error);
}

TEST(WorkspaceApiTest, RecreatedStreamDoesNotReuseRetiredSlot) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  plan.require("scratch", 128);
  alignas(128) std::array<unsigned char, 128 * kMaxWorkspaceStreamSlots> data{};
  workspace.commit(plan, data.data());
  cudaStream_t first, second;
  ASSERT_EQ(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
  EXPECT_EQ(workspace.stream_slot(first), 0);
  ASSERT_EQ(cudaStreamDestroy(first), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
  // CUDA may recycle the handle address; its unique ID still identifies a new stream.
  EXPECT_EQ(workspace.stream_slot(second), 1);
  EXPECT_EQ(cudaStreamDestroy(second), cudaSuccess);
}

TEST(WorkspaceApiTest, ForeignAndStalePlansAreRejected) {
  WorkspaceWrapper workspace, other;
  auto first = workspace.plan();
  auto stale = workspace.plan();
  first.require("dbias", 128);
  EXPECT_THROW(other.commit(first), std::runtime_error);
  workspace.commit(first);
  EXPECT_THROW(workspace.commit(first), std::runtime_error);
  EXPECT_THROW(workspace.commit(stale), std::runtime_error);
  auto before_binding = workspace.plan();
  alignas(128) std::array<unsigned char, 128> data{};
  workspace.bind(data.data());
  EXPECT_THROW(workspace.commit(before_binding), std::runtime_error);
}

TEST(WorkspaceApiTest, InvalidRequirementsLeavePlanUnchanged) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  plan.require("dbias", 128);
  EXPECT_THROW(plan.require("", 1), std::runtime_error);
  EXPECT_THROW(plan.require("bad-key", 1, kNVTEWorkspaceScratch, 128, nullptr, 1),
               std::runtime_error);
  EXPECT_THROW(plan.require("dbias", 1, kNVTEWorkspacePersistent), std::runtime_error);
  EXPECT_THROW(plan.require("dbias", 1, kNVTEWorkspaceScratch, 3), std::runtime_error);
  EXPECT_THROW(plan.require("huge", std::numeric_limits<size_t>::max()), std::runtime_error);
  EXPECT_EQ(plan.size(), kMaxWorkspaceStreamSlots * 128);
  workspace.commit(plan);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 1);
}

TEST(WorkspaceApiTest, MoveOnlyWrappersPreservePlanAndBinding) {
  static_assert(!std::is_copy_constructible<WorkspaceWrapper>::value);
  static_assert(!std::is_copy_constructible<WorkspacePlanWrapper>::value);
  WorkspaceWrapper original;
  auto plan = original.plan();
  const auto region = plan.require("dbias", 128);
  WorkspaceWrapper moved(std::move(original));
  EXPECT_EQ(original.data(), nullptr);
  WorkspacePlanWrapper moved_plan(std::move(plan));
  EXPECT_EQ(plan.data(), nullptr);
  alignas(128) std::array<unsigned char, 512> data{};
  moved.commit(moved_plan, data.data());
  WorkspaceWrapper assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.get(region), data.data());
  EXPECT_EQ(moved.data(), nullptr);
}

TEST(WorkspaceApiTest, PlainTensorWrapperOwnsCatalogAndByteRequirements) {
  TensorWrapper tensor;
  WorkspacePlanWrapper plan(tensor.data());
  const auto table = plan.require("activation", 257, kNVTEWorkspacePersistent);
  nvte_workspace_commit_plan(tensor.data(), plan.data());
  EXPECT_EQ(tensor.dtype(), DType::kByte);
  ASSERT_EQ(tensor.shape().ndim, 1);
  EXPECT_EQ(tensor.shape().data[0], plan.size());

  alignas(128) std::array<unsigned char, 512> data{};
  tensor.set_rowwise_data(data.data(), tensor.dtype(), tensor.shape());
  EXPECT_EQ(nvte_workspace_generation(tensor.data()), 1);
  EXPECT_EQ(nvte_workspace_get_region_data(tensor.data(), table), data.data());
  nvte_workspace_mark_region_initialized(tensor.data(), table, nullptr);
  EXPECT_TRUE(nvte_workspace_region_initialized(tensor.data(), table));

  // Both tensor setters must treat rebinding the same address as new storage.
  const NVTEBasicTensor buffer = tensor.get_rowwise_data();
  NVTETensor handle = tensor.data();
  nvte_set_tensor_param(&handle, kNVTERowwiseData, &buffer);
  EXPECT_EQ(nvte_workspace_generation(handle), 2);
  EXPECT_FALSE(nvte_workspace_region_initialized(handle, table));
  nvte_workspace_mark_region_initialized(handle, table, nullptr);
  nvte_set_tensor_param_v2(handle, kNVTERowwiseData, &buffer, sizeof(buffer));
  EXPECT_EQ(nvte_workspace_generation(handle), 3);
  EXPECT_FALSE(nvte_workspace_region_initialized(handle, table));
}

TEST(WorkspaceApiTest, ExistingTensorStorageCanAcquireCatalog) {
  alignas(128) std::array<unsigned char, 512> data{};
  TensorWrapper tensor(data.data(), std::vector<size_t>{data.size()}, DType::kByte);
  WorkspacePlanWrapper plan(tensor.data());
  const auto region = plan.require("scratch", 128);
  nvte_workspace_commit_plan(tensor.data(), plan.data());
  EXPECT_EQ(nvte_workspace_generation(tensor.data()), 1);
  EXPECT_EQ(nvte_workspace_get_region_data(tensor.data(), region), data.data());
}

TEST(WorkspaceApiTest, DestroyedTensorDoesNotLeakCatalogIntoReusedHandle) {
  {
    TensorWrapper tensor;
    WorkspacePlanWrapper plan(tensor.data());
    plan.require("activation", 128, kNVTEWorkspacePersistent);
    nvte_workspace_commit_plan(tensor.data(), plan.data());
    EXPECT_EQ(nvte_workspace_num_regions(tensor.data()), 1);
  }
  TensorWrapper next;
  EXPECT_EQ(nvte_workspace_num_regions(next.data()), 0);
  EXPECT_EQ(nvte_workspace_size(next.data()), 0);
  EXPECT_EQ(nvte_workspace_generation(next.data()), 0);
}
