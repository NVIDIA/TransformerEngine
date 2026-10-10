/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <transformer_engine/fused_router.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transformer_engine::DType;
using transformer_engine::TensorWrapper;

// Null input/output handles are allowed only for the metadata query. These
// calls also verify invalid workspaces fail before accessing other tensors.
void call_aux_loss(bool graph_safe, NVTETensor workspace, int num_rows) {
  if (graph_safe) {
    nvte_fused_moe_aux_loss_forward_graph_safe_v2(nullptr, nullptr, nullptr, 0, num_rows, 0, 0,
                                                  0.0f, nullptr, nullptr, workspace, nullptr);
  } else {
    nvte_fused_moe_aux_loss_forward_v2(nullptr, nullptr, 0, 0, num_rows, 0, 0, 0.0f, nullptr,
                                       nullptr, workspace, nullptr);
  }
}

class FusedMoeAuxLossWorkspaceTest : public ::testing::TestWithParam<bool> {};

void expect_workspace_error(bool graph_safe, NVTETensor workspace, int num_rows,
                            const char* message) {
  try {
    call_aux_loss(graph_safe, workspace, num_rows);
    FAIL() << "Invalid workspace was accepted";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
  }
}

TEST_P(FusedMoeAuxLossWorkspaceTest, QueryWithoutAccessingInputs) {
  size_t capacity = 0;
  for (int num_rows : {0, 3, 257, 4097}) {
    // A query replaces stale metadata as well as populating empty metadata.
    TensorWrapper workspace(nullptr, std::vector<size_t>{2, 3}, DType::kFloat16);
    ASSERT_NO_THROW(call_aux_loss(GetParam(), workspace.data(), num_rows));
    ASSERT_EQ(workspace.shape().ndim, 1);
    EXPECT_EQ(workspace.dtype(), DType::kFloat32);
    EXPECT_EQ(nvte_tensor_data(workspace.data()), nullptr);
    EXPECT_GT(workspace.shape().data[0], 0);
    if (capacity == 0) capacity = workspace.shape().data[0];
    EXPECT_EQ(workspace.shape().data[0], capacity);
  }
}

TEST_P(FusedMoeAuxLossWorkspaceTest, RejectInsufficientCapacity) {
  float storage = 0.0f;
  TensorWrapper workspace(&storage, std::vector<size_t>{1}, DType::kFloat32);
  expect_workspace_error(GetParam(), workspace.data(), 257, "workspace needs at least");
}

TEST_P(FusedMoeAuxLossWorkspaceTest, RejectWrongDtype) {
  float storage = 0.0f;
  TensorWrapper workspace(&storage, std::vector<size_t>{1}, DType::kFloat16);
  expect_workspace_error(GetParam(), workspace.data(), 1, "workspace must have FP32 dtype");
}

INSTANTIATE_TEST_SUITE_P(HostAndDeviceTokenCount, FusedMoeAuxLossWorkspaceTest, ::testing::Bool());

}  // namespace
