/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "util/workspace.h"

using namespace transformer_engine;

TEST(WorkspaceTest, EmptyAndOptionalBlocks) {
  WorkspaceLayout layout;
  const auto absent = layout.add<float>(0);
  layout.add_bytes(0, 512);
  EXPECT_EQ(layout.size(), 0);
  EXPECT_EQ(layout.alignment(), 128);
  EXPECT_EQ(absent.size, 0);
  const WorkspaceView view(nullptr, layout);
  EXPECT_EQ(view.get<float>(absent), nullptr);
}

TEST(WorkspaceTest, AlignedIndependentRegions) {
  WorkspaceLayout layout;
  const auto table = layout.add<float>(17);
  const auto flags = layout.add_bytes(1);
  const auto metadata = layout.add<size_t>(65);
  const auto activations = layout.add<float>(3);
  EXPECT_EQ(table.offset, 0);
  EXPECT_EQ(flags.offset, 128);
  EXPECT_EQ(metadata.offset, 256);
  EXPECT_EQ(activations.offset, 896);
  EXPECT_EQ(layout.size(), 1024);

  alignas(128) std::array<unsigned char, 1024> data{};
  const WorkspaceView view(data.data(), layout);
  for (const auto &block : {table, flags, metadata, activations}) {
    EXPECT_EQ(block.offset % 128, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.get<unsigned char>(block)) % 128, 0);
  }
  view.get<float>(table)[16] = 4.5f;
  view.get<unsigned char>(flags)[0] = 7;
  view.get<size_t>(metadata)[64] = 12345;
  view.get<float>(activations)[0] = 9.5f;
  EXPECT_EQ(view.get<const float>(table)[16], 4.5f);
  EXPECT_EQ(view.get<unsigned char>(flags)[0], 7);
  EXPECT_EQ(view.get<size_t>(metadata)[64], 12345);
  EXPECT_EQ(view.get<float>(activations)[0], 9.5f);
}

TEST(WorkspaceTest, StrongerAlignmentAndAlignedTypes) {
  struct alignas(256) AlignedElement {
    unsigned char data[256];
  };
  WorkspaceLayout layout;
  const auto small = layout.add_bytes(1, 16);
  const auto aligned = layout.add<AlignedElement>(1);
  const auto stronger = layout.add_bytes(1, 512);
  EXPECT_EQ(small.alignment, 128);
  EXPECT_EQ(aligned.offset, 256);
  EXPECT_EQ(aligned.alignment, 256);
  EXPECT_EQ(stronger.offset, 512);
  EXPECT_EQ(layout.alignment(), 512);
  EXPECT_EQ(layout.size(), 1024);

  alignas(512) std::array<unsigned char, 2048> data{};
  const WorkspaceView view(data.data(), layout);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(view.get<AlignedElement>(aligned)) % 256, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(view.get<unsigned char>(stronger)) % 512, 0);
}

TEST(WorkspaceTest, RejectsInvalidBlocksAndElementTypes) {
  WorkspaceLayout layout;
  const auto bytes = layout.add_bytes(7);
  alignas(128) std::array<unsigned char, 1024> data{};
  const WorkspaceView view(data.data(), layout);
  EXPECT_THROW(view.get<float>(bytes), std::runtime_error);
  // Extra allocation capacity does not make blocks outside the declared layout valid.
  WorkspaceLayout other;
  const auto outside = other.add_bytes(256);
  EXPECT_THROW(view.get<unsigned char>(outside), std::runtime_error);
  const WorkspaceBlock overflow{std::numeric_limits<size_t>::max(), 1, 128};
  EXPECT_THROW(view.get<unsigned char>(overflow), std::runtime_error);
  const WorkspaceBlock unaligned{1, 1, 128};
  EXPECT_THROW(view.get<unsigned char>(unaligned), std::runtime_error);
  const WorkspaceBlock bad_alignment{0, 1, 3};
  EXPECT_THROW(view.get<unsigned char>(bad_alignment), std::runtime_error);
}

TEST(WorkspaceTest, RejectsInvalidAlignmentWithoutChangingLayout) {
  WorkspaceLayout layout;
  layout.add_bytes(1);
  for (size_t alignment : {size_t{0}, size_t{3}, size_t{192}}) {
    EXPECT_THROW(layout.add_bytes(1, alignment), std::runtime_error);
    EXPECT_THROW(layout.add<float>(1, alignment), std::runtime_error);
    EXPECT_EQ(layout.size(), 128);
    EXPECT_EQ(layout.alignment(), 128);
  }
}

TEST(WorkspaceTest, RejectsMultiplicationAndPaddingOverflow) {
  WorkspaceLayout layout;
  const size_t max_size = std::numeric_limits<size_t>::max();
  EXPECT_THROW(layout.add<uint64_t>(max_size / sizeof(uint64_t) + 1), std::runtime_error);
  EXPECT_THROW(layout.add_bytes(max_size), std::runtime_error);
  EXPECT_EQ(layout.size(), 0);
  EXPECT_EQ(layout.add_bytes(1).offset, 0);
  EXPECT_EQ(layout.size(), 128);
}

TEST(WorkspaceTest, RejectsAdditionAndOffsetAlignmentOverflow) {
  WorkspaceLayout layout;
  const size_t max_aligned = std::numeric_limits<size_t>::max() - 127;
  layout.add_bytes(max_aligned);
  EXPECT_THROW(layout.add_bytes(128), std::runtime_error);
  EXPECT_THROW(layout.add_bytes(1, 256), std::runtime_error);
  EXPECT_EQ(layout.size(), max_aligned);
  EXPECT_EQ(layout.alignment(), 128);
}

TEST(WorkspaceTest, AppendingAndExplicitCacheMigration) {
  WorkspaceLayout original;
  const auto table = original.add<float>(16);
  const auto scratch = original.add_bytes(33);
  WorkspaceLayout extended = original;
  const auto extra = extended.add_bytes(512);
  EXPECT_EQ(extra.offset, original.size());
  EXPECT_EQ(extended.size(), 768);

  alignas(128) std::array<unsigned char, 256> old_data{};
  alignas(128) std::array<unsigned char, 1024> new_data;
  new_data.fill(0xcc);
  const WorkspaceView old_view(old_data.data(), original);
  for (size_t i = 0; i < 16; ++i) {
    old_view.get<float>(table)[i] = static_cast<float>(i) / 4;
  }
  const WorkspaceView new_view(new_data.data(), extended);
  EXPECT_NE(new_view.get<float>(table), old_view.get<float>(table));
  // The owner explicitly migrates a valid cache. Views do not initialize/copy any region.
  std::memcpy(new_view.get<float>(table), old_view.get<float>(table), table.size);
  for (size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(new_view.get<float>(table)[i], static_cast<float>(i) / 4);
  }
  EXPECT_EQ(new_view.get<unsigned char>(scratch)[0], 0xcc);
  EXPECT_EQ(new_view.get<unsigned char>(extra)[0], 0xcc);
  EXPECT_EQ(old_view.get<float>(table)[15], 15.f / 4);
}

TEST(WorkspaceTest, ExtendingWithinExistingCapacity) {
  WorkspaceLayout layout;
  const auto table = layout.add<float>(16);
  alignas(128) std::array<unsigned char, 512> data{};
  const WorkspaceView original(data.data(), layout);
  original.get<float>(table)[0] = 7.5f;

  const auto extra = layout.add_bytes(129);
  EXPECT_EQ(layout.size(), 384);
  EXPECT_THROW(original.get<unsigned char>(extra), std::runtime_error);
  // Rebuild the view for the larger layout; the allocation and cache stay in place.
  const WorkspaceView extended(data.data(), layout);
  EXPECT_EQ(original.get<float>(table), extended.get<float>(table));
  EXPECT_EQ(extended.get<float>(table)[0], 7.5f);
  EXPECT_EQ(extended.get<unsigned char>(extra), data.data() + 128);
}
