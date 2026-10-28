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

namespace {

// Independent expected packing: round the byte count up to a whole alignment unit.
// Test sizes are small; overflow cases are exercised directly against WorkspaceLayout.
constexpr size_t PaddedBytes(size_t bytes, size_t alignment = kWorkspaceAlignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

}  // namespace

TEST(WorkspaceTest, EmptyAndOptionalBlocks) {
  constexpr size_t kUnusedAlignment = 4 * kWorkspaceAlignment;
  WorkspaceLayout layout;
  const auto absent = layout.add<float>(0);
  layout.add_bytes(0, kUnusedAlignment);
  EXPECT_EQ(layout.size(), 0);
  EXPECT_EQ(layout.alignment(), kWorkspaceAlignment);
  EXPECT_EQ(absent.size, 0);
  const WorkspaceView view(nullptr, layout);
  EXPECT_EQ(view.get<float>(absent), nullptr);
}

TEST(WorkspaceTest, AlignedIndependentRegions) {
  // Leave padding after the table and overflow four alignment units with metadata.
  constexpr size_t kTableElements = kWorkspaceAlignment / (2 * sizeof(float)) + 1;
  constexpr size_t kFlagBytes = sizeof(unsigned char);
  constexpr size_t kMetadataElements = 4 * kWorkspaceAlignment / sizeof(size_t) + 1;
  constexpr size_t kActivationElements = 3;
  constexpr size_t kTableOffset = 0;
  constexpr size_t kFlagsOffset = PaddedBytes(kTableOffset + kTableElements * sizeof(float));
  constexpr size_t kMetadataOffset = PaddedBytes(kFlagsOffset + kFlagBytes);
  constexpr size_t kMetadataPaddedBytes = PaddedBytes(kMetadataElements * sizeof(size_t));
  constexpr size_t kActivationsOffset = kMetadataOffset + kMetadataPaddedBytes;
  constexpr size_t kWorkspaceBytes =
      PaddedBytes(kActivationsOffset + kActivationElements * sizeof(float));
  constexpr float kTableValue = 4.5f;
  constexpr unsigned char kFlagValue = 7;
  constexpr size_t kMetadataValue = 12345;
  constexpr float kActivationValue = 9.5f;

  WorkspaceLayout layout;
  const auto table = layout.add<float>(kTableElements);
  const auto flags = layout.add_bytes(kFlagBytes);
  const auto metadata = layout.add<size_t>(kMetadataElements);
  const auto activations = layout.add<float>(kActivationElements);
  EXPECT_EQ(table.offset, kTableOffset);
  EXPECT_EQ(flags.offset, kFlagsOffset);
  EXPECT_EQ(metadata.offset, kMetadataOffset);
  EXPECT_EQ(activations.offset, kActivationsOffset);
  EXPECT_EQ(layout.size(), kWorkspaceBytes);

  alignas(kWorkspaceAlignment) std::array<unsigned char, kWorkspaceBytes> data{};
  const WorkspaceView view(data.data(), layout);
  for (const auto &block : {table, flags, metadata, activations}) {
    EXPECT_EQ(block.offset % kWorkspaceAlignment, 0);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(view.get<unsigned char>(block)) % kWorkspaceAlignment, 0);
  }
  view.get<float>(table)[kTableElements - 1] = kTableValue;
  view.get<unsigned char>(flags)[0] = kFlagValue;
  view.get<size_t>(metadata)[kMetadataElements - 1] = kMetadataValue;
  view.get<float>(activations)[0] = kActivationValue;
  EXPECT_EQ(view.get<const float>(table)[kTableElements - 1], kTableValue);
  EXPECT_EQ(view.get<unsigned char>(flags)[0], kFlagValue);
  EXPECT_EQ(view.get<size_t>(metadata)[kMetadataElements - 1], kMetadataValue);
  EXPECT_EQ(view.get<float>(activations)[0], kActivationValue);
}

TEST(WorkspaceTest, StrongerAlignmentAndAlignedTypes) {
  constexpr size_t kSmallRequestedAlignment = kWorkspaceAlignment / 8;
  constexpr size_t kElementAlignment = 2 * kWorkspaceAlignment;
  constexpr size_t kStrongerAlignment = 2 * kElementAlignment;
  constexpr size_t kWorkspaceBytes = PaddedBytes(kStrongerAlignment + 1, kStrongerAlignment);
  struct alignas(kElementAlignment) AlignedElement {
    unsigned char data[kElementAlignment];
  };
  WorkspaceLayout layout;
  const auto small = layout.add_bytes(1, kSmallRequestedAlignment);
  const auto aligned = layout.add<AlignedElement>(1);
  const auto stronger = layout.add_bytes(1, kStrongerAlignment);
  EXPECT_EQ(small.alignment, kWorkspaceAlignment);
  EXPECT_EQ(aligned.offset, kElementAlignment);
  EXPECT_EQ(aligned.alignment, kElementAlignment);
  EXPECT_EQ(stronger.offset, kStrongerAlignment);
  EXPECT_EQ(layout.alignment(), kStrongerAlignment);
  // The final byte starts at kStrongerAlignment and requires another alignment unit.
  EXPECT_EQ(layout.size(), kWorkspaceBytes);

  alignas(kStrongerAlignment) std::array<unsigned char, kWorkspaceBytes> data{};
  const WorkspaceView view(data.data(), layout);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(view.get<AlignedElement>(aligned)) % kElementAlignment, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(view.get<unsigned char>(stronger)) % kStrongerAlignment, 0);
}

TEST(WorkspaceTest, RejectsInvalidBlocksAndElementTypes) {
  constexpr size_t kTruncatedFloatBytes = 2 * sizeof(float) - 1;
  constexpr size_t kOutsideBytes = 2 * kWorkspaceAlignment;
  constexpr size_t kNonPowerOfTwoAlignment = 3;
  WorkspaceLayout layout;
  const auto bytes = layout.add_bytes(kTruncatedFloatBytes);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kOutsideBytes> data{};
  const WorkspaceView view(data.data(), layout);
  EXPECT_THROW(view.get<float>(bytes), std::runtime_error);
  // Extra allocation capacity does not make blocks outside the declared layout valid.
  WorkspaceLayout other;
  const auto outside = other.add_bytes(kOutsideBytes);
  EXPECT_THROW(view.get<unsigned char>(outside), std::runtime_error);
  const WorkspaceBlock overflow{std::numeric_limits<size_t>::max(), 1, kWorkspaceAlignment};
  EXPECT_THROW(view.get<unsigned char>(overflow), std::runtime_error);
  const WorkspaceBlock unaligned{1, 1, kWorkspaceAlignment};
  EXPECT_THROW(view.get<unsigned char>(unaligned), std::runtime_error);
  const WorkspaceBlock bad_alignment{0, 1, kNonPowerOfTwoAlignment};
  EXPECT_THROW(view.get<unsigned char>(bad_alignment), std::runtime_error);
}

TEST(WorkspaceTest, RejectsInvalidAlignmentWithoutChangingLayout) {
  constexpr size_t kZeroAlignment = 0;
  constexpr size_t kNonPowerOfTwoBelowMinimum = 3;
  constexpr size_t kNonPowerOfTwoAboveMinimum = kWorkspaceAlignment + kWorkspaceAlignment / 2;
  WorkspaceLayout layout;
  layout.add_bytes(1);
  for (size_t alignment :
       {kZeroAlignment, kNonPowerOfTwoBelowMinimum, kNonPowerOfTwoAboveMinimum}) {
    EXPECT_THROW(layout.add_bytes(1, alignment), std::runtime_error);
    EXPECT_THROW(layout.add<float>(1, alignment), std::runtime_error);
    EXPECT_EQ(layout.size(), kWorkspaceAlignment);
    EXPECT_EQ(layout.alignment(), kWorkspaceAlignment);
  }
}

TEST(WorkspaceTest, RejectsMultiplicationAndPaddingOverflow) {
  WorkspaceLayout layout;
  const size_t max_size = std::numeric_limits<size_t>::max();
  EXPECT_THROW(layout.add<uint64_t>(max_size / sizeof(uint64_t) + 1), std::runtime_error);
  EXPECT_THROW(layout.add_bytes(max_size), std::runtime_error);
  EXPECT_EQ(layout.size(), 0);
  EXPECT_EQ(layout.add_bytes(1).offset, 0);
  EXPECT_EQ(layout.size(), kWorkspaceAlignment);
}

TEST(WorkspaceTest, RejectsAdditionAndOffsetAlignmentOverflow) {
  constexpr size_t kStrongerAlignment = 2 * kWorkspaceAlignment;
  // The largest size_t aligned to the minimum alignment has no room for another unit.
  const size_t max_aligned = std::numeric_limits<size_t>::max() - (kWorkspaceAlignment - 1);
  WorkspaceLayout layout;
  layout.add_bytes(max_aligned);
  EXPECT_THROW(layout.add_bytes(kWorkspaceAlignment), std::runtime_error);
  EXPECT_THROW(layout.add_bytes(1, kStrongerAlignment), std::runtime_error);
  EXPECT_EQ(layout.size(), max_aligned);
  EXPECT_EQ(layout.alignment(), kWorkspaceAlignment);
}

TEST(WorkspaceTest, AppendingAndExplicitCacheMigration) {
  constexpr size_t kTableElements = kWorkspaceAlignment / (2 * sizeof(float));
  constexpr size_t kScratchBytes = kWorkspaceAlignment / 4 + 1;
  constexpr size_t kExtraBytes = 4 * kWorkspaceAlignment;
  constexpr size_t kOriginalBytes =
      PaddedBytes(kTableElements * sizeof(float)) + PaddedBytes(kScratchBytes);
  constexpr size_t kExtendedBytes = kOriginalBytes + kExtraBytes;
  constexpr unsigned char kUninitializedByte = 0xcc;
  constexpr float kValueDivisor = 4.f;  // Exactly representable cache values.
  WorkspaceLayout original;
  const auto table = original.add<float>(kTableElements);
  const auto scratch = original.add_bytes(kScratchBytes);
  EXPECT_EQ(original.size(), kOriginalBytes);
  WorkspaceLayout extended = original;
  const auto extra = extended.add_bytes(kExtraBytes);
  EXPECT_EQ(extra.offset, kOriginalBytes);
  EXPECT_EQ(extended.size(), kExtendedBytes);

  alignas(kWorkspaceAlignment) std::array<unsigned char, kOriginalBytes> old_data{};
  alignas(kWorkspaceAlignment) std::array<unsigned char, kExtendedBytes> new_data;
  new_data.fill(kUninitializedByte);
  const WorkspaceView old_view(old_data.data(), original);
  for (size_t i = 0; i < kTableElements; ++i) {
    old_view.get<float>(table)[i] = static_cast<float>(i) / kValueDivisor;
  }
  const WorkspaceView new_view(new_data.data(), extended);
  EXPECT_NE(new_view.get<float>(table), old_view.get<float>(table));
  // The owner explicitly migrates a valid cache. Views do not initialize/copy any region.
  std::memcpy(new_view.get<float>(table), old_view.get<float>(table), table.size);
  for (size_t i = 0; i < kTableElements; ++i) {
    EXPECT_EQ(new_view.get<float>(table)[i], static_cast<float>(i) / kValueDivisor);
  }
  EXPECT_EQ(new_view.get<unsigned char>(scratch)[0], kUninitializedByte);
  EXPECT_EQ(new_view.get<unsigned char>(extra)[0], kUninitializedByte);
  EXPECT_EQ(old_view.get<float>(table)[kTableElements - 1],
            static_cast<float>(kTableElements - 1) / kValueDivisor);
}

TEST(WorkspaceTest, ExtendingWithinExistingCapacity) {
  constexpr size_t kTableElements = kWorkspaceAlignment / (2 * sizeof(float));
  constexpr size_t kExtraBytes = kWorkspaceAlignment + 1;  // Needs two alignment units.
  constexpr size_t kExtraOffset = PaddedBytes(kTableElements * sizeof(float));
  constexpr size_t kExtendedBytes = kExtraOffset + PaddedBytes(kExtraBytes);
  constexpr float kCachedValue = 7.5f;
  WorkspaceLayout layout;
  const auto table = layout.add<float>(kTableElements);
  alignas(kWorkspaceAlignment) std::array<unsigned char, kExtendedBytes> data{};
  const WorkspaceView original(data.data(), layout);
  original.get<float>(table)[0] = kCachedValue;

  const auto extra = layout.add_bytes(kExtraBytes);
  EXPECT_EQ(extra.offset, kExtraOffset);
  EXPECT_EQ(layout.size(), kExtendedBytes);
  EXPECT_THROW(original.get<unsigned char>(extra), std::runtime_error);
  // Rebuild the view for the larger layout; the allocation and cache stay in place.
  const WorkspaceView extended(data.data(), layout);
  EXPECT_EQ(original.get<float>(table), extended.get<float>(table));
  EXPECT_EQ(extended.get<float>(table)[0], kCachedValue);
  EXPECT_EQ(extended.get<unsigned char>(extra), data.data() + kExtraOffset);
}
