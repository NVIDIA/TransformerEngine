/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <transformer_engine/workspace.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "util/workspace.h"

using namespace transformer_engine;

namespace {

constexpr size_t kTableBytes = kWorkspaceAlignment;
constexpr size_t kTableElements = kTableBytes / sizeof(int);
constexpr int kConsumerMultiplier = 2;
constexpr int kInitialTableValue = 7;
constexpr unsigned int kGatePollSleepNanoseconds = 100;

__global__ void fill_table(int *table, int value) { table[threadIdx.x] = value; }

__global__ void consume_table(const int *table, int *output) {
  output[threadIdx.x] = table[threadIdx.x] * kConsumerMultiplier;
}

__global__ void hold_stream(int *gate, uint64_t max_cycles) {
  const auto start = clock64();
  while (atomicAdd(gate, 0) == 0 && clock64() - start < max_cycles)
    __nanosleep(kGatePollSleepNanoseconds);
}

__global__ void release_stream(int *gate) { atomicExch(gate, 1); }

struct Initializer {
  int calls = 0;
  int value = kInitialTableValue;
  static void enqueue(void *data, size_t bytes, cudaStream_t stream, void *context) {
    auto *self = static_cast<Initializer *>(context);
    ++self->calls;
    fill_table<<<1, bytes / sizeof(int), 0, stream>>>(static_cast<int *>(data), self->value);
  }
};

class WorkspaceGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
    for (auto &stream : streams_) {
      ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    }
  }
  void TearDown() override {
    for (auto stream : streams_) {
      if (stream != nullptr) {
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      }
    }
    if (buffer_ != nullptr) EXPECT_EQ(cudaFree(buffer_), cudaSuccess);
  }
  void allocate(WorkspaceWrapper &workspace, const WorkspacePlanWrapper &plan) {
    workspace.commit(plan);
    ASSERT_EQ(cudaMalloc(&buffer_, workspace.size()), cudaSuccess);
    workspace.bind(buffer_);
  }
  std::array<cudaStream_t, 2> streams_{};
  void *buffer_ = nullptr;
};

TEST_F(WorkspaceGpuTest, LazyTableAcrossGenerations) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", kTableBytes, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("scratch", kTableBytes);
  allocate(workspace, plan);
  Initializer initializer;
  auto stream = streams_[0];
  for (int iteration = 0; iteration < 3; ++iteration) {
    if (iteration == 1) workspace.bind(buffer_);
    if (iteration == 2) workspace.invalidate(table);
    ++initializer.value;
    workspace.initialize(table, stream, Initializer::enqueue, &initializer);
    workspace.initialize(table, stream, Initializer::enqueue, &initializer);
    consume_table<<<1, kTableElements, 0, stream>>>(workspace.get<int>(table),
                                                    workspace.get<int>(scratch));
    EXPECT_EQ(initializer.calls, iteration + 1);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::array<int, kTableElements> output;
    ASSERT_EQ(cudaMemcpy(output.data(), workspace.get<int>(scratch), sizeof(output),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (auto value : output) EXPECT_EQ(value, kConsumerMultiplier * initializer.value);
  }
}

TEST_F(WorkspaceGpuTest, CaptureRejectsPendingInitialization) {
  constexpr auto kGateTimeout = std::chrono::seconds(5);
  // cudaDevAttrClockRate is in kHz, so cycles = clock_khz * elapsed milliseconds.
  constexpr auto kGateTimeoutMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(kGateTimeout).count();
  int device;
  cudaDeviceProp properties;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  if (!properties.concurrentKernels) GTEST_SKIP();
  int clock_khz;
  ASSERT_EQ(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device), cudaSuccess);
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", kTableBytes, kNVTEWorkspacePersistent);
  allocate(workspace, plan);
  int *gate = nullptr;
  ASSERT_EQ(cudaMalloc(&gate, sizeof(int)), cudaSuccess);
  // Warm both kernels before holding a stream: CUDA lazy module loading may
  // otherwise wait for outstanding GPU work on their first launch.
  release_stream<<<1, 1, 0, streams_.back()>>>(gate);
  consume_table<<<1, 1, 0, streams_.back()>>>(gate, gate);
  ASSERT_EQ(cudaStreamSynchronize(streams_.back()), cudaSuccess);
  ASSERT_EQ(cudaMemset(gate, 0, sizeof(int)), cudaSuccess);
  auto release = [&](int *address) {
    release_stream<<<1, 1, 0, streams_.back()>>>(address);
    EXPECT_EQ(cudaStreamSynchronize(streams_.back()), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(streams_[1]), cudaSuccess);
    EXPECT_EQ(cudaFree(address), cudaSuccess);
  };
  std::unique_ptr<int, decltype(release)> gate_guard(gate, release);
  Initializer initializer;
  hold_stream<<<1, 1, 0, streams_[0]>>>(
      gate, static_cast<uint64_t>(clock_khz) * kGateTimeoutMilliseconds);
  // Initialization is queued behind the gate on the workspace's stream.
  fill_table<<<1, kTableElements, 0, streams_[0]>>>(workspace.get<int>(table), initializer.value);
  workspace.mark_initialized(table, streams_[0]);
  ASSERT_EQ(cudaStreamBeginCapture(streams_[0], cudaStreamCaptureModeRelaxed), cudaSuccess);
  EXPECT_THROW(workspace.initialize(table, streams_[0], Initializer::enqueue, &initializer),
               std::runtime_error);
  cudaGraph_t empty;
  EXPECT_EQ(cudaStreamEndCapture(streams_[0], &empty), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(empty), cudaSuccess);
  // Always release the gate before assertions/cleanup, including on regression.
  gate_guard.reset();
  EXPECT_EQ(initializer.calls, 0);
}

TEST_F(WorkspaceGpuTest, PreparedImmutableTableSupportsGraph) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", kTableBytes, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("scratch", kTableBytes);
  allocate(workspace, plan);
  Initializer initializer;
  auto stream = streams_[0];
  workspace.initialize(table, stream, Initializer::enqueue, &initializer);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), cudaSuccess);
  workspace.initialize(table, stream, Initializer::enqueue, &initializer);
  consume_table<<<1, kTableElements, 0, stream>>>(workspace.get<int>(table),
                                                  workspace.get<int>(scratch));
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
  ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
  constexpr int kGraphReplays = 5;
  for (int repeat = 0; repeat < kGraphReplays; ++repeat) {
    ASSERT_EQ(cudaGraphLaunch(executable, stream), cudaSuccess);
  }
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  std::array<int, kTableElements> output;
  ASSERT_EQ(cudaMemcpy(output.data(), workspace.get<int>(scratch), sizeof(output),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (auto value : output) EXPECT_EQ(value, kConsumerMultiplier * initializer.value);
  EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_EQ(initializer.calls, 1);
}

}  // namespace
