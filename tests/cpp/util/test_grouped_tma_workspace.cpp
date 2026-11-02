/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "cast/core/grouped_tma_workspace.h"

namespace {
using transformer_engine::dispatch::common::GroupedTmaWorkspaceLayout;
using transformer_engine::dispatch::common::kGroupedTmaWorkspaceAlignment;

TEST(GroupedTmaWorkspaceTest, EmptyLayoutHasNoArrays) {
  const GroupedTmaWorkspaceLayout layout(0);
  EXPECT_EQ(layout.size(), 0);
  const auto view = layout.view(nullptr);
  EXPECT_EQ(view.input, nullptr);
  EXPECT_EQ(view.act_input, nullptr);
  EXPECT_EQ(view.output_rowwise, nullptr);
  EXPECT_EQ(view.output_colwise, nullptr);
  EXPECT_EQ(view.rows, nullptr);
  EXPECT_EQ(view.cols, nullptr);
  EXPECT_EQ(view.offsets, nullptr);
}

TEST(GroupedTmaWorkspaceTest, DescriptorAndMetadataArraysHaveIndependentAlignedStorage) {
  constexpr size_t kGroups = 65;
  constexpr size_t kAlignment = kGroupedTmaWorkspaceAlignment;
  constexpr size_t kMapBytes = kGroups * sizeof(CUtensorMap);
  constexpr size_t kMapStride = ((kMapBytes + kAlignment - 1) / kAlignment) * kAlignment;
  constexpr size_t kMetadataBytes = kGroups * sizeof(size_t);
  constexpr size_t kMetadataStride = ((kMetadataBytes + kAlignment - 1) / kAlignment) * kAlignment;
  constexpr size_t kBytes = 4 * kMapStride + 3 * kMetadataStride;
  alignas(kAlignment) std::array<char, kBytes> buffer{};
  const GroupedTmaWorkspaceLayout layout(kGroups);
  EXPECT_EQ(layout.size(), kBytes);
  const auto view = layout.view(buffer.data());
  EXPECT_EQ(reinterpret_cast<char *>(view.input), buffer.data());
  EXPECT_EQ(reinterpret_cast<char *>(view.act_input), buffer.data() + kMapStride);
  EXPECT_EQ(reinterpret_cast<char *>(view.output_rowwise), buffer.data() + 2 * kMapStride);
  EXPECT_EQ(reinterpret_cast<char *>(view.output_colwise), buffer.data() + 3 * kMapStride);
  EXPECT_EQ(reinterpret_cast<char *>(view.rows), buffer.data() + 4 * kMapStride);
  EXPECT_EQ(reinterpret_cast<char *>(view.cols), buffer.data() + 4 * kMapStride + kMetadataStride);
  EXPECT_EQ(reinterpret_cast<char *>(view.offsets),
            buffer.data() + 4 * kMapStride + 2 * kMetadataStride);
}

TEST(GroupedTmaWorkspaceTest, RejectsNullAndMisalignedStorage) {
  const GroupedTmaWorkspaceLayout layout(1);
  alignas(kGroupedTmaWorkspaceAlignment) std::array<char, kGroupedTmaWorkspaceAlignment> buffer{};
  EXPECT_THROW(layout.view(nullptr), std::runtime_error);
  EXPECT_THROW(layout.view(buffer.data() + 1), std::runtime_error);
}

TEST(GroupedTmaWorkspaceTest, RejectsArrayAndTotalSizeOverflow) {
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  EXPECT_THROW((void)GroupedTmaWorkspaceLayout(kMax), std::runtime_error);
  EXPECT_THROW((void)GroupedTmaWorkspaceLayout(kMax / sizeof(CUtensorMap)), std::runtime_error);
}
}  // namespace
