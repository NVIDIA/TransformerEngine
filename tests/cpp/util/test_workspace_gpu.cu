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
#include <memory>
#include <stdexcept>
#include <thread>

using namespace transformer_engine;

namespace {

__global__ void fill_table(int *table, int value) { table[threadIdx.x] = value; }

__global__ void consume_table(const int *table, int *output) {
  output[threadIdx.x] = table[threadIdx.x] * 2;
}

__global__ void hold_stream(int *gate, unsigned long long max_cycles) {
  const auto start = clock64();
  while (atomicAdd(gate, 0) == 0 && clock64() - start < max_cycles) __nanosleep(100);
}

__global__ void release_stream(int *gate) { atomicExch(gate, 1); }

struct Initializer {
  int calls = 0;
  int value = 7;
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
  std::array<cudaStream_t, kMaxWorkspaceStreamSlots + 1> streams_{};
  void *buffer_ = nullptr;
};

TEST_F(WorkspaceGpuTest, SharedLazyTableAcrossStreamsAndGenerations) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", 128, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("scratch", 128);
  allocate(workspace, plan);
  Initializer initializer;
  for (int iteration = 0; iteration < 3; ++iteration) {
    if (iteration == 1) workspace.bind(buffer_);
    if (iteration == 2) workspace.invalidate(table);
    ++initializer.value;
    for (int slot = 0; slot < kMaxWorkspaceStreamSlots; ++slot) {
      auto stream = streams_[kMaxWorkspaceStreamSlots - slot - 1];
      workspace.initialize(table, stream, Initializer::enqueue, &initializer);
      consume_table<<<1, 32, 0, stream>>>(workspace.get<int>(table, stream),
                                          workspace.get<int>(scratch, stream));
    }
    EXPECT_EQ(initializer.calls, iteration + 1);
    for (int slot = 0; slot < kMaxWorkspaceStreamSlots; ++slot) {
      auto stream = streams_[kMaxWorkspaceStreamSlots - slot - 1];
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      std::array<int, 32> output;
      ASSERT_EQ(cudaMemcpy(output.data(), workspace.get<int>(scratch, stream), sizeof(output),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (auto value : output) EXPECT_EQ(value, 2 * initializer.value);
    }
  }
}

TEST_F(WorkspaceGpuTest, TableReadinessDoesNotWaitForProducerConsumers) {
  int device;
  cudaDeviceProp properties;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  if (!properties.concurrentKernels) GTEST_SKIP();
  int clock_khz;
  ASSERT_EQ(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device), cudaSuccess);
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", 128, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("scratch", 128);
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
  cudaEvent_t producer_done, consumer_done;
  ASSERT_EQ(cudaEventCreateWithFlags(&producer_done, cudaEventDisableTiming), cudaSuccess);
  ASSERT_EQ(cudaEventCreateWithFlags(&consumer_done, cudaEventDisableTiming), cudaSuccess);
  Initializer initializer;
  workspace.initialize(table, streams_[0], Initializer::enqueue, &initializer);
  hold_stream<<<1, 1, 0, streams_[0]>>>(gate, static_cast<unsigned long long>(clock_khz) * 5000);
  ASSERT_EQ(cudaEventRecord(producer_done, streams_[0]), cudaSuccess);
  workspace.initialize(table, streams_[1], Initializer::enqueue, &initializer);
  consume_table<<<1, 32, 0, streams_[1]>>>(workspace.get<int>(table, streams_[1]),
                                           workspace.get<int>(scratch, streams_[1]));
  ASSERT_EQ(cudaEventRecord(consumer_done, streams_[1]), cudaSuccess);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  cudaError_t consumer_status;
  do {
    consumer_status = cudaEventQuery(consumer_done);
    if (consumer_status != cudaErrorNotReady) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  const auto producer_status = cudaEventQuery(producer_done);
  // A pre-capture wait is not retained by a graph replayed on another stream.
  // Re-enqueue a table producer behind the gate and reject capture until ready.
  workspace.invalidate(table);
  fill_table<<<1, 32, 0, streams_[0]>>>(workspace.get<int>(table), initializer.value);
  workspace.mark_initialized(table, streams_[0]);
  ASSERT_EQ(cudaStreamBeginCapture(streams_[0], cudaStreamCaptureModeRelaxed), cudaSuccess);
  EXPECT_THROW(workspace.initialize(table, streams_[0], Initializer::enqueue, &initializer),
               std::runtime_error);
  cudaGraph_t empty;
  EXPECT_EQ(cudaStreamEndCapture(streams_[0], &empty), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(empty), cudaSuccess);
  // Always release the gate before assertions/cleanup, including on regression.
  gate_guard.reset();
  EXPECT_EQ(consumer_status, cudaSuccess);
  EXPECT_EQ(producer_status, cudaErrorNotReady);
  EXPECT_EQ(initializer.calls, 1);
  EXPECT_EQ(cudaEventDestroy(producer_done), cudaSuccess);
  EXPECT_EQ(cudaEventDestroy(consumer_done), cudaSuccess);
}

TEST_F(WorkspaceGpuTest, PreparedImmutableTableSupportsIndependentGraphs) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", 128, kNVTEWorkspacePersistent);
  const auto scratch = plan.require("scratch", 128);
  allocate(workspace, plan);
  Initializer initializer;
  std::array<cudaGraph_t, kMaxWorkspaceStreamSlots> graphs{};
  std::array<cudaGraphExec_t, kMaxWorkspaceStreamSlots> executables{};
  for (int slot = 0; slot < kMaxWorkspaceStreamSlots; ++slot) {
    workspace.initialize(table, streams_[slot], Initializer::enqueue, &initializer);
    ASSERT_EQ(cudaStreamSynchronize(streams_[slot]), cudaSuccess);
    ASSERT_EQ(cudaStreamBeginCapture(streams_[slot], cudaStreamCaptureModeRelaxed), cudaSuccess);
    workspace.initialize(table, streams_[slot], Initializer::enqueue, &initializer);
    consume_table<<<1, 32, 0, streams_[slot]>>>(workspace.get<int>(table, streams_[slot]),
                                                workspace.get<int>(scratch, streams_[slot]));
    ASSERT_EQ(cudaStreamEndCapture(streams_[slot], &graphs[slot]), cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&executables[slot], graphs[slot], nullptr, nullptr, 0),
              cudaSuccess);
  }
  for (int repeat = 0; repeat < 5; ++repeat) {
    for (int slot = 0; slot < kMaxWorkspaceStreamSlots; ++slot) {
      ASSERT_EQ(cudaGraphLaunch(executables[slot], streams_[slot]), cudaSuccess);
    }
  }
  for (int slot = 0; slot < kMaxWorkspaceStreamSlots; ++slot) {
    ASSERT_EQ(cudaStreamSynchronize(streams_[slot]), cudaSuccess);
    std::array<int, 32> output;
    ASSERT_EQ(cudaMemcpy(output.data(), workspace.get<int>(scratch, streams_[slot]), sizeof(output),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (auto value : output) EXPECT_EQ(value, 2 * initializer.value);
    EXPECT_EQ(cudaGraphExecDestroy(executables[slot]), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graphs[slot]), cudaSuccess);
  }
  EXPECT_EQ(initializer.calls, 1);
}

}  // namespace
