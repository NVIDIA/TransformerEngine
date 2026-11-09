/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "../test_common.h"

namespace {
using namespace transformer_engine;
using test::Tensor;

std::vector<unsigned char> read_bytes(void *data, size_t bytes) {
  std::vector<unsigned char> result(bytes);
  NVTE_CHECK_CUDA(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost));
  return result;
}

// Above the former device-symbol limit; each stream uses different dimensions,
// pointers, and data so accidentally sharing descriptors corrupts the results.
constexpr size_t kGroups = 65;
constexpr size_t kTile = 128;
constexpr int kReplays = 20;
constexpr int kStreams = 4;

struct GroupData {
  std::vector<int64_t> rows, cols, offsets;
  size_t elements;
  Tensor input_data, rows_data, cols_data, offsets_data, quantized, scales, dequantized,
      requantized, requant_scales;
  GroupedTensorWrapper input, output, dequant_output, requant_output;

  explicit GroupData(int index)
      : rows(kGroups),
        cols(kGroups),
        offsets(kGroups + 1),
        elements(0),
        input(kGroups, std::vector<size_t>{1, 1}),
        output(kGroups, std::vector<size_t>{1, 1}, NVTE_MXFP8_1D_SCALING),
        dequant_output(kGroups, std::vector<size_t>{1, 1}),
        requant_output(kGroups, std::vector<size_t>{1, 1}, NVTE_MXFP8_1D_SCALING) {
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
    requant_output = GroupedTensorWrapper(kGroups, shape, NVTE_MXFP8_1D_SCALING);
    input_data = Tensor("input", shape, DType::kFloat32);
    rows_data = Tensor("rows", metadata_shape, DType::kInt64);
    cols_data = Tensor("cols", metadata_shape, DType::kInt64);
    offsets_data = Tensor("offsets", std::vector<size_t>{kGroups + 1}, DType::kInt64);
    quantized = Tensor("quantized", shape, DType::kFloat8E4M3);
    scales = Tensor("scales", std::vector<size_t>{elements / 32}, DType::kFloat8E8M0);
    dequantized = Tensor("dequantized", shape, DType::kFloat32);
    requantized = Tensor("requantized", shape, DType::kFloat8E4M3);
    requant_scales =
        Tensor("requant_scales", std::vector<size_t>{elements / 32}, DType::kFloat8E8M0);
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
    for (auto *tensor : {&input, &output, &dequant_output, &requant_output}) {
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
    requant_output
        .set_columnwise_data(requantized.rowwise_dptr(), DType::kFloat8E4M3,
                             std::vector<size_t>{elements})
        .set_columnwise_scale_inv(requant_scales.rowwise_dptr(), DType::kFloat8E8M0,
                                  std::vector<size_t>{elements / 32});
    dequant_output.set_rowwise_data(dequantized.rowwise_dptr(), DType::kFloat32,
                                    std::vector<size_t>{elements});
  }
};

// 0..4: activation, 5..9: backward, 10: dequantization, 11: requantization. Every path used to
// reference shared mutable device symbols when the last dimension varied.
class GroupedWorkspaceConcurrency : public ::testing::TestWithParam<std::tuple<int, bool>> {};

TEST_P(GroupedWorkspaceConcurrency, ConcurrentStreamsMatchSerialResults) {
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
  std::array<std::unique_ptr<GroupData>, kStreams> data;
  std::array<cudaStream_t, kStreams> streams{};
  std::array<cudaGraph_t, kStreams> graphs{};
  std::array<cudaGraphExec_t, kStreams> executions{};
  std::array<std::vector<unsigned char>, kStreams> expected, expected_scales;
  std::array<TensorWrapper, kStreams> workspaces;
  std::array<std::unique_ptr<Tensor>, kStreams> storage;
  auto launch = [&](int index, NVTETensor scratch, cudaStream_t stream) {
    auto &group = *data[index];
    if (operation < 5) {
      forward[operation](group.input.data(), group.output.data(), scratch, stream);
    } else if (operation < 10) {
      backward[operation - 5](group.input.data(), group.input.data(), group.output.data(), scratch,
                              stream);
    } else if (operation == 10) {
      nvte_group_dequantize_v2(group.output.data(), group.dequant_output.data(), scratch, stream);
    } else {
      nvte_group_requantize_v2(group.output.data(), group.requant_output.data(), nullptr, scratch,
                               nullptr, stream);
    }
  };
  for (int i = 0; i < kStreams; ++i) {
    data[i] = std::make_unique<GroupData>(i);
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking), cudaSuccess);
    if (operation >= 10) {
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
    auto &workspace = workspaces[i];
    launch(i, workspace.data(), streams[i]);
    storage[i] = std::make_unique<Tensor>("workspace", workspace.shape(), workspace.dtype());
    workspace.set_rowwise_data(storage[i]->rowwise_dptr(), workspace.dtype(), workspace.shape());
  }
  auto result_data = [&](int i) {
    return operation == 10   ? data[i]->dequantized.rowwise_dptr()
           : operation == 11 ? data[i]->requantized.rowwise_dptr()
                             : data[i]->quantized.rowwise_dptr();
  };
  auto result_scales = [&](int i) {
    return operation == 11 ? data[i]->requant_scales.rowwise_dptr()
                           : data[i]->scales.rowwise_dptr();
  };
  for (int i = 0; i < kStreams; ++i) {
    launch(i, workspaces[i].data(), streams[i]);
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    expected[i] =
        read_bytes(result_data(i), data[i]->elements * (operation == 10 ? sizeof(float) : 1));
    expected_scales[i] = read_bytes(result_scales(i), data[i]->elements / 32);
    if (capture) {
      ASSERT_EQ(cudaStreamBeginCapture(streams[i], cudaStreamCaptureModeRelaxed), cudaSuccess);
      launch(i, workspaces[i].data(), streams[i]);
      ASSERT_EQ(cudaStreamEndCapture(streams[i], &graphs[i]), cudaSuccess);
      ASSERT_EQ(cudaGraphInstantiate(&executions[i], graphs[i], nullptr, nullptr, 0), cudaSuccess);
    }
  }
  for (int i = 0; i < kStreams; ++i) {
    // Prevent a missing consumer launch from passing with the warmup's output.
    ASSERT_EQ(cudaMemsetAsync(result_data(i), 0xa5, expected[i].size(), streams[i]), cudaSuccess);
  }
  for (int repeat = 0; repeat < kReplays; ++repeat) {
    for (int i = kStreams - 1; i >= 0; --i) {
      if (capture) {
        ASSERT_EQ(cudaGraphLaunch(executions[i], streams[i]), cudaSuccess);
      } else {
        launch(i, workspaces[i].data(), streams[i]);
      }
    }
  }
  for (int i = 0; i < kStreams; ++i) {
    ASSERT_EQ(cudaStreamSynchronize(streams[i]), cudaSuccess);
    EXPECT_EQ(read_bytes(result_data(i), expected[i].size()), expected[i]);
    EXPECT_EQ(read_bytes(result_scales(i), expected_scales[i].size()), expected_scales[i]);
    if (capture) {
      ASSERT_EQ(cudaGraphExecDestroy(executions[i]), cudaSuccess);
      ASSERT_EQ(cudaGraphDestroy(graphs[i]), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamDestroy(streams[i]), cudaSuccess);
  }
}

std::string workspace_concurrency_test_name(
    const ::testing::TestParamInfo<GroupedWorkspaceConcurrency::ParamType> &info) {
  const std::array<const char *, 12> operations{"Gelu",   "Silu",   "Relu",       "Qgelu",
                                                "Srelu",  "Dgelu",  "Dsilu",      "Drelu",
                                                "Dqgelu", "Dsrelu", "Dequantize", "Requantize"};
  return std::string(operations[std::get<0>(info.param)]) +
         (std::get<1>(info.param) ? "Graph" : "Eager");
}

INSTANTIATE_TEST_SUITE_P(OperatorTest, GroupedWorkspaceConcurrency,
                         ::testing::Combine(::testing::Range(0, 12), ::testing::Bool()),
                         workspace_concurrency_test_name);

// Small buffers shared by query tests across formats. Device data is deliberately
// poisoned so a query that accidentally executes is observable.
struct QueryData {
  const std::vector<size_t> shape{256, 128};
  Tensor values{"query_values", shape, DType::kFloat32};
  Tensor result{"query_result", shape, DType::kByte};
  Tensor scales{"query_scales", std::vector<size_t>{4096}, DType::kFloat32};
  Tensor amax{"query_amax", std::vector<size_t>{256}, DType::kFloat32};
  Tensor offsets{"query_offsets", std::vector<size_t>{3}, DType::kInt64};
  Tensor rows{"query_rows", std::vector<size_t>{2}, DType::kInt64};
  GroupedTensorWrapper input{2, shape};
  GroupedTensorWrapper output;
  GroupedTensorWrapper dequantized{2, shape};

  explicit QueryData(NVTEScalingMode mode) : output(2, shape, mode) {
    input.set_rowwise_data(values.rowwise_dptr(), DType::kBFloat16, std::vector<size_t>{256 * 128});
    dequantized.set_rowwise_data(values.rowwise_dptr(), DType::kFloat32,
                                 std::vector<size_t>{256 * 128});
    output
        .set_rowwise_data(result.rowwise_dptr(), DType::kFloat8E4M3, std::vector<size_t>{256 * 128})
        .set_rowwise_scale_inv(scales.rowwise_dptr(), DType::kFloat32, std::vector<size_t>{4096})
        .set_scale(scales.rowwise_dptr(), DType::kFloat32, std::vector<size_t>{2});
    if (mode == NVTE_MXFP8_1D_SCALING) {
      output.set_rowwise_scale_inv(scales.rowwise_dptr(), DType::kFloat8E8M0,
                                   std::vector<size_t>{256, 4});
    }
    if (mode == NVTE_NVFP4_1D_SCALING) {
      output.set_row_scaled_nvfp4(true);
      output
          .set_rowwise_data(result.rowwise_dptr(), DType::kFloat4E2M1,
                            std::vector<size_t>{256 * 128})
          .set_rowwise_scale_inv(scales.rowwise_dptr(), DType::kFloat8E4M3,
                                 std::vector<size_t>{256, 8})
          .set_amax(amax.rowwise_dptr(), DType::kFloat32, std::vector<size_t>{256})
          .set_tensor_offsets(offsets.rowwise_dptr(), DType::kInt64, std::vector<size_t>{3})
          .set_first_dims(rows.rowwise_dptr(), DType::kInt64, std::vector<size_t>{2});
      const std::array<int64_t, 3> host_offsets{0, 128 * 128, 256 * 128};
      const std::array<int64_t, 2> host_rows{128, 128};
      NVTE_CHECK_CUDA(cudaMemcpy(offsets.rowwise_dptr(), host_offsets.data(), sizeof(host_offsets),
                                 cudaMemcpyHostToDevice));
      NVTE_CHECK_CUDA(cudaMemcpy(rows.rowwise_dptr(), host_rows.data(), sizeof(host_rows),
                                 cudaMemcpyHostToDevice));
    }
    NVTE_CHECK_CUDA(cudaMemset(values.rowwise_dptr(), 0x5a, 256 * 128 * sizeof(float)));
    NVTE_CHECK_CUDA(cudaMemset(result.rowwise_dptr(), 0x5a, 256 * 128));
    NVTE_CHECK_CUDA(cudaMemset(scales.rowwise_dptr(), 0x5a, 4096 * sizeof(float)));
    NVTE_CHECK_CUDA(cudaMemset(amax.rowwise_dptr(), 0x5a, 256 * sizeof(float)));
  }
};

class GroupedWorkspaceQuery : public ::testing::TestWithParam<NVTEScalingMode> {
 protected:
  void SetUp() override {
    const auto capability = test::getDeviceComputeCapability();
    if ((GetParam() == NVTE_NVFP4_1D_SCALING || GetParam() == NVTE_MXFP8_1D_SCALING) &&
        capability < test::blackwellComputeCapability) {
      GTEST_SKIP() << "Grouped MXFP8/NVFP4 requires Blackwell or newer.";
    }
    if ((GetParam() == NVTE_BLOCK_SCALING_1D || GetParam() == NVTE_BLOCK_SCALING_2D) &&
        capability < test::hopperComputeCapability) {
      GTEST_SKIP() << "Grouped FP8 block scaling requires Hopper or newer.";
    }
  }
};

TEST_P(GroupedWorkspaceQuery, RejectsUnsupportedActivation) {
  if (GetParam() == NVTE_DELAYED_TENSOR_SCALING || GetParam() == NVTE_MXFP8_1D_SCALING) {
    GTEST_SKIP() << "This format supports grouped activation.";
  }
  QueryData data(GetParam());
  Tensor storage("query_workspace", std::vector<size_t>{1}, DType::kByte);
  TensorWrapper workspace;
  const std::array<void (*)(NVTEGroupedTensor, NVTEGroupedTensor, NVTETensor, cudaStream_t), 5>
      forward{nvte_group_gelu_v2, nvte_group_silu_v2, nvte_group_relu_v2, nvte_group_qgelu_v2,
              nvte_group_srelu_v2};
  for (size_t i = 0; i < forward.size(); ++i) {
    SCOPED_TRACE(i);
    const auto operation = forward[i];
    EXPECT_THROW(operation(data.input.data(), data.output.data(), workspace.data(), 0),
                 std::runtime_error);
    TensorWrapper execution(storage.rowwise_dptr(), std::vector<size_t>{1}, DType::kByte);
    EXPECT_THROW(operation(data.input.data(), data.output.data(), execution.data(), 0),
                 std::runtime_error);
  }
}

TEST_P(GroupedWorkspaceQuery, RejectsUnsupportedInputType) {
  QueryData data(GetParam());
  data.input.set_rowwise_data(data.values.rowwise_dptr(), DType::kInt32,
                              std::vector<size_t>{256 * 128});
  QuantizationConfigWrapper config;
  config.set_force_pow_2_scales(true);
  TensorWrapper workspace;
  EXPECT_THROW(
      nvte_group_quantize_v2(data.input.data(), data.output.data(), workspace.data(), config, 0),
      std::runtime_error);
}

TEST_P(GroupedWorkspaceQuery, RejectsInvalidFormatConfiguration) {
  QueryData data(GetParam());
  QuantizationConfigWrapper config;
  if (GetParam() == NVTE_DELAYED_TENSOR_SCALING) {
    data.output.set_scale(data.scales.rowwise_dptr(), DType::kInt32, std::vector<size_t>{2});
  } else if (GetParam() == NVTE_NVFP4_1D_SCALING) {
    data.output.set_row_scaled_nvfp4(false);
  } else if (GetParam() == NVTE_MXFP8_1D_SCALING) {
    data.output.set_rowwise_scale_inv(nullptr, DType::kFloat8E8M0, std::vector<size_t>{256, 4});
  } else {
    if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
      GTEST_SKIP() << "Non-power-of-two block scales are supported before Blackwell.";
    }
    config.set_force_pow_2_scales(false);
  }
  TensorWrapper workspace;
  EXPECT_THROW(
      nvte_group_quantize_v2(data.input.data(), data.output.data(), workspace.data(), config, 0),
      std::runtime_error);
  Tensor storage("query_workspace", std::vector<size_t>{1}, DType::kByte);
  TensorWrapper execution(storage.rowwise_dptr(), std::vector<size_t>{1}, DType::kByte);
  EXPECT_THROW(
      nvte_group_quantize_v2(data.input.data(), data.output.data(), execution.data(), config, 0),
      std::runtime_error);
}

TEST_P(GroupedWorkspaceQuery, SupportedQueriesDoNotLaunchOrWrite) {
  QueryData data(GetParam());
  QuantizationConfigWrapper config;
  config.set_force_pow_2_scales(true);
  const auto expected = read_bytes(data.result.rowwise_dptr(), 256 * 128);
  const auto expected_scales = read_bytes(data.scales.rowwise_dptr(), 4096 * sizeof(float));
  const auto expected_amax = read_bytes(data.amax.rowwise_dptr(), 256 * sizeof(float));
  const auto expected_values = read_bytes(data.values.rowwise_dptr(), 256 * 128 * sizeof(float));
  TensorWrapper workspace;
  // Warm host-side CUDA state before capture; the second query must still be empty.
  nvte_group_quantize_v2(data.input.data(), data.output.data(), workspace.data(), config, 0);
  EXPECT_EQ(workspace.dtype(), DType::kByte);
  ASSERT_EQ(workspace.shape().ndim, 1);
  EXPECT_EQ(workspace.shape().data[0], 1);
  if (GetParam() == NVTE_BLOCK_SCALING_1D || GetParam() == NVTE_BLOCK_SCALING_2D ||
      GetParam() == NVTE_MXFP8_1D_SCALING) {
    nvte_group_dequantize_v2(data.output.data(), data.dequantized.data(), workspace.data(), 0);
  }
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  EXPECT_NO_THROW(nvte_group_quantize_v2(data.input.data(), data.output.data(), workspace.data(),
                                         config, stream));
  if (GetParam() == NVTE_BLOCK_SCALING_1D || GetParam() == NVTE_BLOCK_SCALING_2D ||
      GetParam() == NVTE_MXFP8_1D_SCALING) {
    EXPECT_NO_THROW(nvte_group_dequantize_v2(data.output.data(), data.dequantized.data(),
                                             workspace.data(), stream));
  }
  cudaGraph_t graph;
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  size_t nodes = 0;
  ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &nodes), cudaSuccess);
  EXPECT_EQ(nodes, 0);
  EXPECT_EQ(read_bytes(data.result.rowwise_dptr(), expected.size()), expected);
  EXPECT_EQ(read_bytes(data.scales.rowwise_dptr(), expected_scales.size()), expected_scales);
  EXPECT_EQ(read_bytes(data.amax.rowwise_dptr(), expected_amax.size()), expected_amax);
  EXPECT_EQ(read_bytes(data.values.rowwise_dptr(), expected_values.size()), expected_values);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_P(GroupedWorkspaceQuery, DequantizeRejectsInvalidLayoutAndType) {
  if (GetParam() != NVTE_BLOCK_SCALING_1D && GetParam() != NVTE_BLOCK_SCALING_2D &&
      GetParam() != NVTE_MXFP8_1D_SCALING) {
    GTEST_SKIP() << "This regression targets block-scaled and MXFP8 dequantization.";
  }
  QueryData data(GetParam());
  TensorWrapper workspace;
  data.output.set_with_gemm_swizzled_scales(true);
  EXPECT_THROW(
      nvte_group_dequantize_v2(data.output.data(), data.dequantized.data(), workspace.data(), 0),
      std::runtime_error);
  data.output.set_with_gemm_swizzled_scales(false);
  data.dequantized.set_rowwise_data(data.values.rowwise_dptr(), DType::kInt32,
                                    std::vector<size_t>{256 * 128});
  EXPECT_THROW(
      nvte_group_dequantize_v2(data.output.data(), data.dequantized.data(), workspace.data(), 0),
      std::runtime_error);
}

TEST(GroupedWorkspaceQueryConfig, Mxfp8RejectsMisalignedTensorMaps) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) GTEST_SKIP();
  QueryData data(NVTE_MXFP8_1D_SCALING);
  data.output.set_rowwise_data(static_cast<char *>(data.result.rowwise_dptr()) + 1,
                               DType::kFloat8E4M3, std::vector<size_t>{256 * 128});
  TensorWrapper workspace;
  EXPECT_THROW(
      nvte_group_quantize_v2(data.input.data(), data.output.data(), workspace.data(), nullptr, 0),
      std::runtime_error);
  EXPECT_THROW(
      nvte_group_dequantize_v2(data.output.data(), data.dequantized.data(), workspace.data(), 0),
      std::runtime_error);
}

TEST(GroupedWorkspaceQueryConfig, Mxfp8DbiasRejectsUnsupportedInputType) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) GTEST_SKIP();
  QueryData data(NVTE_MXFP8_1D_SCALING);
  data.input.set_rowwise_data(data.values.rowwise_dptr(), DType::kInt32,
                              std::vector<size_t>{256 * 128});
  GroupedTensorWrapper dbias(2, std::vector<size_t>{2, 128});
  dbias.set_rowwise_data(data.values.rowwise_dptr(), DType::kInt32, std::vector<size_t>{2, 128});
  TensorWrapper workspace;
  EXPECT_THROW(nvte_group_quantize_dbias(data.input.data(), data.output.data(), dbias.data(),
                                         workspace.data(), nullptr, 0),
               std::runtime_error);
}

TEST(GroupedWorkspaceQueryConfig, RejectsInvalidNvfp4Options) {
  if (test::getDeviceComputeCapability() < test::blackwellComputeCapability) {
    GTEST_SKIP() << "Grouped NVFP4 requires Blackwell or newer.";
  }
  QueryData data(NVTE_NVFP4_1D_SCALING);
  for (int option = 0; option < 3; ++option) {
    QuantizationConfigWrapper config;
    if (option == 0) config.set_stochastic_rounding(true);
    if (option == 1) config.set_nvfp4_2d_quantization(true);
    if (option == 2) config.set_nvfp4_4over6_mode(kNVTENVFP44Over6MinMAE);
    TensorWrapper workspace;
    EXPECT_THROW(
        nvte_group_quantize_v2(data.input.data(), data.output.data(), workspace.data(), config, 0),
        std::runtime_error);
  }
}

std::string workspace_query_test_name(const ::testing::TestParamInfo<NVTEScalingMode> &info) {
  switch (info.param) {
    case NVTE_DELAYED_TENSOR_SCALING:
      return "DelayedFp8";
    case NVTE_BLOCK_SCALING_1D:
      return "BlockFp8_1D";
    case NVTE_BLOCK_SCALING_2D:
      return "BlockFp8_2D";
    case NVTE_MXFP8_1D_SCALING:
      return "Mxfp8";
    case NVTE_NVFP4_1D_SCALING:
      return "RowScaledNvfp4";
    default:
      return "Unknown";
  }
}

INSTANTIATE_TEST_SUITE_P(Formats, GroupedWorkspaceQuery,
                         ::testing::Values(NVTE_DELAYED_TENSOR_SCALING, NVTE_BLOCK_SCALING_1D,
                                           NVTE_BLOCK_SCALING_2D, NVTE_MXFP8_1D_SCALING,
                                           NVTE_NVFP4_1D_SCALING),
                         workspace_query_test_name);
}  // namespace
