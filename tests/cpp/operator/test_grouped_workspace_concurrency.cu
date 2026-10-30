/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/workspace.h>

#include <array>
#include <memory>
#include <vector>

#include "../test_common.h"

namespace {
using namespace transformer_engine;
using test::Tensor;

// Above the former device-symbol limit; each stream uses different dimensions,
// pointers, and data so accidentally sharing descriptors corrupts the results.
constexpr size_t kGroups = 65;
constexpr size_t kTile = 128;
constexpr int kReplays = 20;

struct GroupData {
  std::vector<int64_t> rows, cols, offsets;
  size_t elements;
  Tensor input_data, rows_data, cols_data, offsets_data, quantized, scales, dequantized;
  GroupedTensorWrapper input, output, dequant_output;

  explicit GroupData(int index)
      : rows(kGroups),
        cols(kGroups),
        offsets(kGroups + 1),
        elements(0),
        input(kGroups, std::vector<size_t>{1, 1}),
        output(kGroups, std::vector<size_t>{1, 1}, NVTE_MXFP8_1D_SCALING),
        dequant_output(kGroups, std::vector<size_t>{1, 1}) {
    for (size_t i = 0; i < kGroups; ++i) {
      rows[i] = kTile * (1 + (i + index) % 2);
      cols[i] = kTile * (1 + (i + index) % 3);
      offsets[i + 1] = offsets[i] + rows[i] * cols[i];
    }
    elements = offsets.back();
    const std::vector<size_t> shape{1, elements}, metadata_shape{kGroups};
    input = GroupedTensorWrapper(kGroups, shape);
    output = GroupedTensorWrapper(kGroups, shape, NVTE_MXFP8_1D_SCALING);
    dequant_output = GroupedTensorWrapper(kGroups, shape);
    input_data = Tensor("input", shape, DType::kFloat32);
    rows_data = Tensor("rows", metadata_shape, DType::kInt64);
    cols_data = Tensor("cols", metadata_shape, DType::kInt64);
    offsets_data = Tensor("offsets", std::vector<size_t>{kGroups + 1}, DType::kInt64);
    quantized = Tensor("quantized", shape, DType::kFloat8E4M3);
    scales = Tensor("scales", std::vector<size_t>{elements / 32}, DType::kFloat8E8M0);
    dequantized = Tensor("dequantized", shape, DType::kFloat32);
    std::vector<float> values(elements);
    for (size_t i = 0; i < elements; ++i)
      values[i] = (static_cast<int>(i % 31) - 15) * 0.125f + index;
    NVTE_CHECK_CUDA(cudaMemcpy(input_data.rowwise_dptr(), values.data(), elements * sizeof(float),
                               cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(rows_data.rowwise_dptr(), rows.data(), kGroups * sizeof(int64_t),
                               cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(cols_data.rowwise_dptr(), cols.data(), kGroups * sizeof(int64_t),
                               cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(offsets_data.rowwise_dptr(), offsets.data(),
                               (kGroups + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
    for (auto *tensor : {&input, &output, &dequant_output}) {
      tensor->set_first_dims(rows_data.rowwise_dptr(), DType::kInt64, metadata_shape)
          .set_last_dims(cols_data.rowwise_dptr(), DType::kInt64, metadata_shape)
          .set_tensor_offsets(offsets_data.rowwise_dptr(), DType::kInt64,
                              std::vector<size_t>{kGroups + 1});
    }
    input.set_rowwise_data(input_data.rowwise_dptr(), DType::kFloat32,
                           std::vector<size_t>{elements});
    output
        .set_rowwise_data(quantized.rowwise_dptr(), DType::kFloat8E4M3,
                          std::vector<size_t>{elements})
        .set_rowwise_scale_inv(scales.rowwise_dptr(), DType::kFloat8E8M0,
                               std::vector<size_t>{elements / 32});
    dequant_output.set_rowwise_data(dequantized.rowwise_dptr(), DType::kFloat32,
                                    std::vector<size_t>{elements});
  }
};

std::vector<unsigned char> read_bytes(void *data, size_t bytes) {
  std::vector<unsigned char> result(bytes);
  NVTE_CHECK_CUDA(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost));
  return result;
}

// 0..4: activation, 5..9: backward, 10: dequantization. Every path used to
// reference shared mutable device symbols when the last dimension varied.
class GroupedWorkspaceConcurrency : public ::testing::TestWithParam<std::tuple<int, bool>> {};

TEST_P(GroupedWorkspaceConcurrency, IndependentStreamSlots) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) GTEST_SKIP();
  const int operation = std::get<0>(GetParam());
  const bool capture = std::get<1>(GetParam());
  using Forward = void (*)(NVTEGroupedTensor, NVTEGroupedTensor, NVTETensor, cudaStream_t);
  using Backward =
      void (*)(NVTEGroupedTensor, NVTEGroupedTensor, NVTEGroupedTensor, NVTETensor, cudaStream_t);
  const std::array<Forward, 5> forward = {nvte_group_gelu_v2, nvte_group_silu_v2,
                                          nvte_group_relu_v2, nvte_group_qgelu_v2,
                                          nvte_group_srelu_v2};
  const std::array<Backward, 5> backward = {nvte_group_dgelu_v2, nvte_group_dsilu_v2,
                                            nvte_group_drelu_v2, nvte_group_dqgelu_v2,
                                            nvte_group_dsrelu_v2};
  std::array<std::unique_ptr<GroupData>, kMaxWorkspaceStreamSlots> data;
  std::array<cudaStream_t, kMaxWorkspaceStreamSlots> streams{};
  std::array<cudaGraph_t, kMaxWorkspaceStreamSlots> graphs{};
  std::array<cudaGraphExec_t, kMaxWorkspaceStreamSlots> executions{};
  std::array<std::vector<unsigned char>, kMaxWorkspaceStreamSlots> expected, expected_scales;
  TensorWrapper workspace;
  auto launch = [&](int index, NVTETensor scratch, cudaStream_t stream) {
    auto &group = *data[index];
    if (operation < 5) {
      forward[operation](group.input.data(), group.output.data(), scratch, stream);
    } else if (operation < 10) {
      backward[operation - 5](group.input.data(), group.input.data(), group.output.data(), scratch,
                              stream);
    } else {
      nvte_group_dequantize_v2(group.output.data(), group.dequant_output.data(), scratch, stream);
    }
  };
  for (int i = 0; i < kMaxWorkspaceStreamSlots; ++i) {
    data[i] = std::make_unique<GroupData>(i);
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cudaSuccess);
    if (operation == 10) {
      TensorWrapper quant_workspace;
      nvte_group_quantize_v2(data[i]->input.data(), data[i]->output.data(), quant_workspace.data(),
                             nullptr, streams[i]);
      Tensor quant_storage("quant_workspace", quant_workspace.shape(), quant_workspace.dtype());
      quant_workspace.set_rowwise_data(quant_storage.rowwise_dptr(), quant_workspace.dtype(),
                                       quant_workspace.shape());
      nvte_group_quantize_v2(data[i]->input.data(), data[i]->output.data(), quant_workspace.data(),
                             nullptr, streams[i]);
      ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    }
    launch(i, workspace.data(), streams[i]);
  }
  Tensor storage("workspace", workspace.shape(), workspace.dtype());
  workspace.set_rowwise_data(storage.rowwise_dptr(), workspace.dtype(), workspace.shape());
  for (int i = 0; i < kMaxWorkspaceStreamSlots; ++i) {
    launch(i, workspace.data(), streams[i]);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    expected[i] =
        operation == 10
            ? read_bytes(data[i]->dequantized.rowwise_dptr(), data[i]->elements * sizeof(float))
            : read_bytes(data[i]->quantized.rowwise_dptr(), data[i]->elements);
    expected_scales[i] = read_bytes(data[i]->scales.rowwise_dptr(), data[i]->elements / 32);
    if (capture) {
      ASSERT_EQ(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeRelaxed), cudaSuccess);
      launch(i, workspace.data(), streams[i]);
      ASSERT_EQ(cudaStreamEndCapture(streams[i], &graphs[i]), cudaSuccess);
      ASSERT_EQ(cudaGraphInstantiate(&executions[i], graphs[i], nullptr, nullptr, 0), cudaSuccess);
    }
  }
  for (int i = 0; i < kMaxWorkspaceStreamSlots; ++i) {
    // Prevent a missing consumer launch from passing with the warmup's output.
    ASSERT_EQ(cudaMemsetAsync(operation == 10 ? data[i]->dequantized.rowwise_dptr()
                                              : data[i]->quantized.rowwise_dptr(),
                              0xa5, expected[i].size(), streams[i]),
              cudaSuccess);
  }
  for (int repeat = 0; repeat < kReplays; ++repeat) {
    for (int i = kMaxWorkspaceStreamSlots - 1; i >= 0; --i) {
      if (capture) {
        ASSERT_EQ(cudaGraphLaunch(executions[i], streams[i]), cudaSuccess);
      } else {
        launch(i, workspace.data(), streams[i]);
      }
    }
  }
  for (int i = 0; i < kMaxWorkspaceStreamSlots; ++i) {
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    EXPECT_EQ(read_bytes(operation == 10 ? data[i]->dequantized.rowwise_dptr()
                                         : data[i]->quantized.rowwise_dptr(),
                         expected[i].size()),
              expected[i]);
    EXPECT_EQ(read_bytes(data[i]->scales.rowwise_dptr(), expected_scales[i].size()),
              expected_scales[i]);
    if (capture) {
      ASSERT_EQ(cudaGraphExecDestroy(executions[i]), cudaSuccess);
      ASSERT_EQ(cudaGraphDestroy(graphs[i]), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, GroupedWorkspaceConcurrency,
                         ::testing::Combine(::testing::Range(0, 11), ::testing::Bool()));
}  // namespace
