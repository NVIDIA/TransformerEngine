/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <transformer_engine/workspace.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "util/workspace.h"

using namespace transformer_engine;

namespace {

// Independent expected packing: round the byte count up to a whole alignment unit.
// Test sizes are small; overflow cases are exercised directly against WorkspaceLayout.
constexpr size_t PaddedBytes(size_t bytes, size_t alignment = kWorkspaceAlignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

}  // namespace

TEST(WorkspaceApiTest, EmptyAndOptionalRegions) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  EXPECT_EQ(plan.require("optional", 0), 0);
  EXPECT_EQ(plan.size(), 0);
  EXPECT_EQ(plan.alignment(), kWorkspaceAlignment);
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
  // Each unaligned request needs padding; scratch has one copy per stream slot.
  constexpr size_t kScratchBytes = kWorkspaceAlignment + 1;
  constexpr size_t kScratchSlotStride = PaddedBytes(kScratchBytes);
  constexpr size_t kTableBytes = 2 * kWorkspaceAlignment + 1;
  constexpr size_t kTablePaddedBytes = PaddedBytes(kTableBytes);
  constexpr size_t kTableOffset = kMaxWorkspaceStreamSlots * kScratchSlotStride;
  constexpr size_t kWorkspaceBytes = kTableOffset + kTablePaddedBytes;
  WorkspaceWrapper workspace;
  {
    auto plan = workspace.plan();
    plan.require("dbias", kScratchBytes);
    plan.require("gelu", kTableBytes, kNVTEWorkspacePersistent);
    EXPECT_EQ(plan.size(), kWorkspaceBytes);
    EXPECT_EQ(workspace.size(), 0);
    EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 0);
  }
  EXPECT_EQ(workspace.size(), 0);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 0);
}

TEST(WorkspaceApiTest, AppendingTablesPreservesScratchOffsets) {
  // Each unaligned request needs padding; scratch has one copy per stream slot.
  constexpr size_t kScratchBytes = kWorkspaceAlignment + 1;
  constexpr size_t kScratchSlotStride = PaddedBytes(kScratchBytes);
  constexpr size_t kTableBytes = 2 * kWorkspaceAlignment + 1;
  constexpr size_t kTablePaddedBytes = PaddedBytes(kTableBytes);
  constexpr size_t kTableOffset = kMaxWorkspaceStreamSlots * kScratchSlotStride;
  constexpr size_t kWorkspaceBytes = kTableOffset + kTablePaddedBytes;
  WorkspaceWrapper workspace;
  auto first = workspace.plan();
  const auto dbias = first.require("dbias", kScratchBytes);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kWorkspaceBytes> data{};
  workspace.commit(first, data.data());
  const auto dbias_offset = nvte_workspace_get_region_info(workspace.data(), dbias).offset;

  auto extended = workspace.plan();
  const auto table = extended.require("gelu", kTableBytes, kNVTEWorkspacePersistent);
  EXPECT_EQ(nvte_workspace_find_region(workspace.data(), "gelu", nullptr, 0), 0);
  workspace.commit(extended);
  EXPECT_EQ(workspace.get(dbias), data.data() + dbias_offset);
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), table).offset, kTableOffset);
  EXPECT_EQ(workspace.size(), kWorkspaceBytes);
  EXPECT_EQ(workspace.generation(), 1);
  EXPECT_FALSE(workspace.initialized(table));
}

TEST(WorkspaceApiTest, GrowingScratchPreservesPersistentTable) {
  constexpr size_t kInitialRegionBytes = kWorkspaceAlignment / 2;
  constexpr size_t kOriginalBytes = (kMaxWorkspaceStreamSlots + 1) * kWorkspaceAlignment;
  constexpr size_t kGrownScratchBytes = 2 * kWorkspaceAlignment + 1;
  constexpr size_t kGrownScratchSlotStride = PaddedBytes(kGrownScratchBytes);
  // Growth appends all scratch copies after the original scratch copies and shared table.
  constexpr size_t kGrownScratchOffset = kOriginalBytes;
  constexpr size_t kExtendedBytes =
      kGrownScratchOffset + kMaxWorkspaceStreamSlots * kGrownScratchSlotStride;
  constexpr float kCachedTableValue = 7.5f;
  WorkspaceWrapper workspace;
  auto first = workspace.plan();
  const auto dbias = first.require("dbias", kInitialRegionBytes);
  const auto table = first.require("gelu", kInitialRegionBytes, kNVTEWorkspacePersistent);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kExtendedBytes> data{};
  workspace.commit(first, data.data());
  workspace.get<float>(table)[0] = kCachedTableValue;
  workspace.mark_initialized(table, nullptr);
  EXPECT_EQ(workspace.size(), kOriginalBytes);
  const auto table_offset = nvte_workspace_get_region_info(workspace.data(), table).offset;

  auto extended = workspace.plan();
  EXPECT_EQ(extended.require("dbias", kGrownScratchBytes), dbias);
  workspace.commit(extended);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 2);
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), table).offset, table_offset);
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), dbias).offset, kGrownScratchOffset);
  EXPECT_EQ(workspace.size(), kExtendedBytes);
  EXPECT_TRUE(workspace.initialized(table));
  EXPECT_EQ(workspace.get<float>(table)[0], kCachedTableValue);
}

TEST(WorkspaceApiTest, ReservationsAreReusedWithoutShrinking) {
  constexpr size_t kTableAlignment = 2 * kWorkspaceAlignment;
  constexpr size_t kReservedTableBytes = kTableAlignment;
  constexpr size_t kSmallerRequestBytes = kReservedTableBytes / 16;
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table =
      plan.require("table", kReservedTableBytes, kNVTEWorkspacePersistent, kTableAlignment);
  EXPECT_EQ(plan.require("table", 1, kNVTEWorkspacePersistent), table);
  EXPECT_EQ(plan.size(), kReservedTableBytes);
  alignas(kTableAlignment) std::array<unsigned char, kReservedTableBytes> data{};
  workspace.commit(plan, data.data());
  workspace.mark_initialized(table, nullptr);
  auto reused = workspace.plan();
  EXPECT_EQ(reused.require("table", kSmallerRequestBytes, kNVTEWorkspacePersistent), table);
  workspace.commit(reused);
  EXPECT_TRUE(workspace.initialized(table));
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), table).size, kReservedTableBytes);
  EXPECT_EQ(workspace.alignment(), kTableAlignment);
}

TEST(WorkspaceApiTest, ExactKeysAndNamesAreCopied) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  char name[] = "activation";
  std::array<unsigned char, 3> key{1, 2, 3};
  const auto first = plan.require(name, kWorkspaceAlignment, kNVTEWorkspacePersistent,
                                  kWorkspaceAlignment, key.data(), key.size());
  name[0] = 'X';
  key.back() = 4;
  const auto second = plan.require("activation", kWorkspaceAlignment, kNVTEWorkspacePersistent,
                                   kWorkspaceAlignment, key.data(), key.size());
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
  EXPECT_EQ(static_cast<const unsigned char *>(info.key)[original.size() - 1], original.back());
  EXPECT_EQ(info.lifetime, kNVTEWorkspacePersistent);
}

TEST(WorkspaceApiTest, RebindingAlwaysInvalidatesTables) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("gelu", kWorkspaceAlignment, kNVTEWorkspacePersistent);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kWorkspaceAlignment> first{}, second{};
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
  constexpr size_t kInitialTableBytes = kWorkspaceAlignment / 2;
  constexpr size_t kSecondTableOffset = kWorkspaceAlignment;
  constexpr size_t kOriginalBytes = 2 * kWorkspaceAlignment;
  constexpr size_t kGrownTableBytes = 2 * kWorkspaceAlignment + 1;
  constexpr size_t kGrownTablePaddedBytes = PaddedBytes(kGrownTableBytes);
  constexpr size_t kExtendedBytes = kOriginalBytes + kGrownTablePaddedBytes;
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto first = plan.require("gelu", kInitialTableBytes, kNVTEWorkspacePersistent);
  const auto second = plan.require("silu", kInitialTableBytes, kNVTEWorkspacePersistent);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kExtendedBytes> data{};
  workspace.commit(plan, data.data());
  workspace.mark_initialized(first, nullptr);
  workspace.mark_initialized(second, nullptr);
  auto extended = workspace.plan();
  EXPECT_EQ(extended.require("gelu", kGrownTableBytes, kNVTEWorkspacePersistent), first);
  workspace.commit(extended);
  EXPECT_EQ(workspace.info(first).offset, kOriginalBytes);
  EXPECT_EQ(workspace.size(), kExtendedBytes);
  EXPECT_FALSE(workspace.initialized(first));
  EXPECT_TRUE(workspace.initialized(second));
  EXPECT_EQ(nvte_workspace_get_region_info(workspace.data(), second).offset, kSecondTableOffset);
}

TEST(WorkspaceApiTest, ExplicitMigrationAfterNewBinding) {
  constexpr size_t kTableBytes = kWorkspaceAlignment / 2;
  constexpr size_t kScratchBytes = kWorkspaceAlignment;
  constexpr size_t kOriginalBytes = kWorkspaceAlignment + kMaxWorkspaceStreamSlots * kScratchBytes;
  constexpr size_t kDescriptorBytes = 2 * kWorkspaceAlignment + 1;
  constexpr size_t kDescriptorSlotStride = PaddedBytes(kDescriptorBytes);
  constexpr size_t kExtendedBytes =
      kOriginalBytes + kMaxWorkspaceStreamSlots * kDescriptorSlotStride;
  constexpr float kCachedTableValue = 7.5f;
  constexpr unsigned char kUninitializedByte = 0xcc;
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("gelu", kTableBytes, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("dbias", kScratchBytes);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kOriginalBytes> old_data{};
  workspace.commit(plan, old_data.data());
  workspace.get<float>(table)[0] = kCachedTableValue;
  workspace.mark_initialized(table, nullptr);
  const auto *old_table = workspace.get(table);
  const size_t table_bytes = workspace.info(table).size;

  auto extended = workspace.plan();
  extended.require("tma", kDescriptorBytes);
  EXPECT_EQ(extended.size(), kExtendedBytes);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kExtendedBytes> new_data;
  new_data.fill(kUninitializedByte);
  workspace.commit(extended, new_data.data());
  EXPECT_EQ(workspace.find("gelu"), table);
  EXPECT_EQ(workspace.num_regions(), 3);
  EXPECT_FALSE(workspace.initialized(table));
  std::memcpy(workspace.get(table), old_table, table_bytes);
  workspace.mark_initialized(table, nullptr);
  EXPECT_TRUE(workspace.initialized(table));
  EXPECT_EQ(workspace.get<float>(table)[0], kCachedTableValue);
  EXPECT_EQ(workspace.get(scratch)[0], kUninitializedByte);
  workspace.invalidate(table);
  EXPECT_FALSE(workspace.initialized(table));
}

TEST(WorkspaceApiTest, CommitPreservesInitializationEnqueuedAfterSnapshot) {
  constexpr size_t kExtendedBytes = (1 + kMaxWorkspaceStreamSlots) * kWorkspaceAlignment;
  WorkspaceWrapper workspace;
  auto first = workspace.plan();
  const auto table = first.require("gelu", kWorkspaceAlignment, kNVTEWorkspacePersistent);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kExtendedBytes> data{};
  workspace.commit(first, data.data());
  auto extended = workspace.plan();
  extended.require("dbias", kWorkspaceAlignment);
  workspace.mark_initialized(table, nullptr);
  workspace.commit(extended);
  EXPECT_TRUE(workspace.initialized(table));
}

TEST(WorkspaceApiTest, StreamSlotsFollowFirstUseNotHandleOrder) {
  constexpr size_t kScratchBytes = kWorkspaceAlignment + 1;
  constexpr size_t kScratchSlotStride = PaddedBytes(kScratchBytes);
  constexpr size_t kDescriptorBytes = kWorkspaceAlignment;
  constexpr size_t kDescriptorSlotStride = kWorkspaceAlignment;
  // Mutable copies are contiguous per region; the immutable table has a single copy.
  constexpr size_t kDescriptorsOffset = kMaxWorkspaceStreamSlots * kScratchSlotStride;
  constexpr size_t kTableOffset =
      kDescriptorsOffset + kMaxWorkspaceStreamSlots * kDescriptorSlotStride;
  constexpr size_t kWorkspaceBytes = kTableOffset + kWorkspaceAlignment;
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto scratch = plan.require("scratch", kScratchBytes);
  const auto descriptors = plan.require("tma", kDescriptorBytes, kNVTEWorkspaceStreamPersistent);
  const auto table = plan.require("gelu", kWorkspaceAlignment, kNVTEWorkspacePersistent);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kWorkspaceBytes> data{};
  workspace.commit(plan, data.data());
  EXPECT_EQ(workspace.info(descriptors).offset, kDescriptorsOffset);
  EXPECT_EQ(workspace.info(table).offset, kTableOffset);
  EXPECT_EQ(workspace.size(), kWorkspaceBytes);
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
    EXPECT_EQ(workspace.get(scratch, order[slot]),
              workspace.get(scratch) + slot * kScratchSlotStride);
    EXPECT_EQ(workspace.get(descriptors, order[slot]),
              workspace.get(descriptors) + slot * kDescriptorSlotStride);
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
  constexpr size_t kWorkspaceBytes = (1 + kMaxWorkspaceStreamSlots) * kWorkspaceAlignment;
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("gelu", kWorkspaceAlignment, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("dbias", kWorkspaceAlignment);
  workspace.commit(plan);
  EXPECT_THROW(workspace.get(table), std::runtime_error);
  EXPECT_THROW(workspace.mark_initialized(table, nullptr), std::runtime_error);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kWorkspaceBytes> data{};
  workspace.bind(data.data());
  EXPECT_THROW(workspace.mark_initialized(scratch, nullptr), std::runtime_error);
  workspace.mark_initialized(table, nullptr);
  nvte_workspace_invalidate_region(workspace.data(), table);
  EXPECT_FALSE(workspace.initialized(table));
  EXPECT_THROW(nvte_workspace_get_region_info(workspace.data(), 0), std::runtime_error);
  EXPECT_THROW(workspace.get(workspace.num_regions() + 1), std::runtime_error);
}

TEST(WorkspaceApiTest, RecreatedStreamDoesNotReuseRetiredSlot) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  plan.require("scratch", kWorkspaceAlignment);
  alignas(kWorkspaceAlignment)
      std::array<unsigned char, kWorkspaceAlignment * kMaxWorkspaceStreamSlots>
          data{};
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
  constexpr size_t kScratchReservationBytes = kMaxWorkspaceStreamSlots * kWorkspaceAlignment;
  WorkspaceWrapper workspace, other;
  auto first = workspace.plan();
  auto stale = workspace.plan();
  first.require("dbias", kWorkspaceAlignment);
  EXPECT_THROW(other.commit(first), std::runtime_error);
  workspace.commit(first);
  EXPECT_THROW(workspace.commit(first), std::runtime_error);
  EXPECT_THROW(workspace.commit(stale), std::runtime_error);
  auto before_binding = workspace.plan();
  alignas(kWorkspaceAlignment) std::array<unsigned char, kScratchReservationBytes> data{};
  workspace.bind(data.data());
  EXPECT_THROW(workspace.commit(before_binding), std::runtime_error);
}

TEST(WorkspaceApiTest, InvalidRequirementsLeavePlanUnchanged) {
  constexpr size_t kNonPowerOfTwoAlignment = 3;
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  plan.require("dbias", kWorkspaceAlignment);
  EXPECT_THROW(plan.require("", 1), std::runtime_error);
  EXPECT_THROW(plan.require("bad-key", 1, kNVTEWorkspaceScratch, kWorkspaceAlignment, nullptr, 1),
               std::runtime_error);
  EXPECT_THROW(plan.require("dbias", 1, kNVTEWorkspacePersistent), std::runtime_error);
  EXPECT_THROW(plan.require("dbias", 1, kNVTEWorkspaceScratch, kNonPowerOfTwoAlignment),
               std::runtime_error);
  EXPECT_THROW(plan.require("huge", std::numeric_limits<size_t>::max()), std::runtime_error);
  EXPECT_EQ(plan.size(), kMaxWorkspaceStreamSlots * kWorkspaceAlignment);
  workspace.commit(plan);
  EXPECT_EQ(nvte_workspace_num_regions(workspace.data()), 1);
}

TEST(WorkspaceApiTest, MoveOnlyWrappersPreservePlanAndBinding) {
  constexpr size_t kScratchReservationBytes = kMaxWorkspaceStreamSlots * kWorkspaceAlignment;
  static_assert(!std::is_copy_constructible<WorkspaceWrapper>::value);
  static_assert(!std::is_copy_constructible<WorkspacePlanWrapper>::value);
  WorkspaceWrapper original;
  auto plan = original.plan();
  const auto region = plan.require("dbias", kWorkspaceAlignment);
  WorkspaceWrapper moved(std::move(original));
  EXPECT_EQ(original.data(), nullptr);
  WorkspacePlanWrapper moved_plan(std::move(plan));
  EXPECT_EQ(plan.data(), nullptr);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kScratchReservationBytes> data{};
  moved.commit(moved_plan, data.data());
  WorkspaceWrapper assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.get(region), data.data());
  EXPECT_EQ(moved.data(), nullptr);
}

TEST(WorkspaceApiTest, PlainTensorWrapperOwnsCatalogAndByteRequirements) {
  constexpr size_t kTableBytes = 2 * kWorkspaceAlignment + 1;
  constexpr size_t kTablePaddedBytes = PaddedBytes(kTableBytes);
  TensorWrapper tensor;
  WorkspacePlanWrapper plan(tensor.data());
  const auto table = plan.require("activation", kTableBytes, kNVTEWorkspacePersistent);
  nvte_workspace_commit_plan(tensor.data(), plan.data());
  EXPECT_EQ(plan.size(), kTablePaddedBytes);
  EXPECT_EQ(tensor.dtype(), DType::kByte);
  ASSERT_EQ(tensor.shape().ndim, 1);
  EXPECT_EQ(tensor.shape().data[0], plan.size());

  alignas(kWorkspaceAlignment) std::array<unsigned char, kTablePaddedBytes> data{};
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
  constexpr size_t kScratchReservationBytes = kMaxWorkspaceStreamSlots * kWorkspaceAlignment;
  alignas(kWorkspaceAlignment) std::array<unsigned char, kScratchReservationBytes> data{};
  TensorWrapper tensor(data.data(), std::vector<size_t>{data.size()}, DType::kByte);
  WorkspacePlanWrapper plan(tensor.data());
  const auto region = plan.require("scratch", kWorkspaceAlignment);
  nvte_workspace_commit_plan(tensor.data(), plan.data());
  EXPECT_EQ(nvte_workspace_generation(tensor.data()), 1);
  EXPECT_EQ(nvte_workspace_get_region_data(tensor.data(), region), data.data());
}

TEST(WorkspaceApiTest, DestroyedTensorDoesNotLeakCatalogIntoReusedHandle) {
  {
    TensorWrapper tensor;
    WorkspacePlanWrapper plan(tensor.data());
    plan.require("activation", kWorkspaceAlignment, kNVTEWorkspacePersistent);
    nvte_workspace_commit_plan(tensor.data(), plan.data());
    EXPECT_EQ(nvte_workspace_num_regions(tensor.data()), 1);
  }
  TensorWrapper next;
  EXPECT_EQ(nvte_workspace_num_regions(next.data()), 0);
  EXPECT_EQ(nvte_workspace_size(next.data()), 0);
  EXPECT_EQ(nvte_workspace_generation(next.data()), 0);
}
