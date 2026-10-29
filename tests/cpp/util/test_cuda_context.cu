/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <gtest/gtest.h>
#include <transformer_engine/cuda_context.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/normalization.h>
#include <transformer_engine/workspace.h>

#include <array>
#include <atomic>
#include <cmath>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include "util/rtc.h"

extern "C" cudnnHandle_t nvte_get_cudnn_handle();
using namespace transformer_engine;

namespace {
__global__ void fill(int *data, int value) { data[threadIdx.x] = value; }

struct TableInitializer {
  int calls = 0;
  int value = 7;
  static void enqueue(void *data, size_t bytes, cudaStream_t stream, void *user) {
    auto &self = *static_cast<TableInitializer *>(user);
    ++self.calls;
    fill<<<1, bytes / sizeof(int), 0, stream>>>(static_cast<int *>(data), self.value);
  }
};

class ContextResetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
  void reset(int device = 0) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    ASSERT_EQ(cudaDeviceReset(), cudaSuccess);
    const auto before = nvte_get_cuda_context_generation(device);
    ASSERT_NO_THROW(nvte_reset_cuda_state(device));
    EXPECT_EQ(nvte_get_cuda_context_generation(device), before + 1);
  }
};

TEST(CudaContextHostTest, GenerationDoesNotInitializeCuda) {
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  CUcontext before = nullptr, after = nullptr;
  ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);
  const auto initial = nvte_get_cuda_context_generation(0);
  nvte_reset_cuda_state(0);
  EXPECT_EQ(nvte_get_cuda_context_generation(0), initial + 1);
  nvte_reset_cuda_state(0);
  EXPECT_EQ(nvte_get_cuda_context_generation(0), initial + 2);
  ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
  EXPECT_EQ(before, after);
  EXPECT_THROW(nvte_reset_cuda_state(-1), std::runtime_error);
  EXPECT_THROW(nvte_get_cuda_context_generation(-1), std::runtime_error);
}

TEST_F(ContextResetTest, StreamsAndRtcRecoverRepeatedly) {
  auto &manager = rtc::KernelManager::instance();
  constexpr auto code = R"(__global__ void recovered(int *p) { *p = 37; })";
  manager.compile("context recovery test", "recovered", code, "recovered.cu");
  for (int cycle = 0; cycle < 4; ++cycle) {
    auto stream = nvte_get_compute_stream(0);
    auto event = nvte_get_compute_stream_event(0);
    int *data = nullptr;
    ASSERT_EQ(cudaMalloc(&data, sizeof(int)), cudaSuccess);
    manager.launch("context recovery test", 1, 1, 0, stream, data);
    ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);
    int value = 0;
    ASSERT_EQ(cudaMemcpy(&value, data, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(value, 37);
    reset();
    // The allocation died with the context; do not cudaFree the old pointer.
    EXPECT_TRUE(manager.is_compiled("context recovery test", 0));
  }
}

TEST_F(ContextResetTest, WorkspaceTablesAndOldPlansRecover) {
  WorkspaceWrapper workspace;
  auto plan = workspace.plan();
  const auto table = plan.require("activation", 128, kNVTEWorkspacePersistent);
  workspace.commit(plan);
  TableInitializer initializer;
  for (int cycle = 0; cycle < 3; ++cycle) {
    void *data = nullptr;
    ASSERT_EQ(cudaMalloc(&data, workspace.size()), cudaSuccess);
    workspace.bind(data);
    auto stream = nvte_get_compute_stream(0);
    workspace.initialize(table, stream, TableInitializer::enqueue, &initializer);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::array<int, 32> values;
    ASSERT_EQ(
        cudaMemcpy(values.data(), workspace.get(table), sizeof(values), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (auto value : values) EXPECT_EQ(value, initializer.value);
    EXPECT_EQ(initializer.calls, cycle + 1);
    {
      auto snapshot = workspace.plan();
      // Retire the event while a snapshot still owns it. Reset must make its
      // eventual destructor safe even though the active catalog no longer owns it.
      workspace.bind(data);
      workspace.mark_initialized(table, stream);
      auto current_snapshot = workspace.plan();
      const auto bound_generation = workspace.generation();
      reset();
      EXPECT_GT(workspace.generation(), bound_generation);
      EXPECT_FALSE(workspace.initialized(table));
      EXPECT_THROW(workspace.get(table), std::runtime_error);
      EXPECT_THROW(workspace.commit(snapshot), std::runtime_error);
      EXPECT_THROW(workspace.commit(current_snapshot), std::runtime_error);
    }
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    ++initializer.value;
  }
}

TEST_F(ContextResetTest, ResetPreservesOtherDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) GTEST_SKIP() << "Requires two CUDA devices";
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  auto stream = nvte_get_compute_stream(0);
  auto event = nvte_get_compute_stream_event(0);
  const auto other_generation = nvte_get_cuda_context_generation(1);
  reset(0);
  EXPECT_EQ(nvte_get_cuda_context_generation(1), other_generation);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  EXPECT_EQ(nvte_get_compute_stream(0), stream);
  EXPECT_EQ(nvte_get_compute_stream_event(0), event);
  EXPECT_EQ(cudaEventRecord(event, stream), cudaSuccess);
  EXPECT_EQ(cudaEventSynchronize(event), cudaSuccess);
  reset(1);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

TEST_F(ContextResetTest, OtherThreadsCudnnHandleRecovers) {
  std::promise<void> ready, resume;
  auto resumed = resume.get_future();
  std::atomic<int> errors{0};
  std::thread worker([&] {
    try {
      if (cudaSetDevice(0) != cudaSuccess) ++errors;
      auto handle = nvte_get_cudnn_handle();
      if (cudnnSetStream(handle, nvte_get_compute_stream(0)) != CUDNN_STATUS_SUCCESS) ++errors;
    } catch (...) {
      ++errors;
    }
    ready.set_value();
    resumed.wait();
    try {
      if (cudaSetDevice(0) != cudaSuccess) ++errors;
      auto handle = nvte_get_cudnn_handle();
      if (cudnnSetStream(handle, nvte_get_compute_stream(0)) != CUDNN_STATUS_SUCCESS) ++errors;
    } catch (...) {
      ++errors;
    }
  });
  ready.get_future().wait();
  reset();
  resume.set_value();
  worker.join();
  EXPECT_EQ(errors.load(), 0);
  reset();
}

TEST_F(ContextResetTest, CudnnNormalizationPlanRecoversNumerically) {
  nvte_enable_cudnn_norm_fwd(true);
  constexpr size_t hidden = 128, rows = 4;
  for (int cycle = 0; cycle < 3; ++cycle) {
    float *buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&buffer, (3 * rows * hidden + 2 * hidden + 2 * rows) * sizeof(float)),
              cudaSuccess);
    auto *x = buffer;
    auto *gamma = x + rows * hidden;
    auto *beta = gamma + hidden;
    auto *output = beta + hidden;
    auto *mu = output + rows * hidden;
    auto *rsigma = mu + rows;
    std::vector<float> input(rows * hidden);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % hidden);
    std::vector<float> weights(hidden, 1.0f);
    ASSERT_EQ(cudaMemcpy(x, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(gamma, weights.data(), hidden * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemset(beta, 0, hidden * sizeof(float)), cudaSuccess);
    TensorWrapper tx(x, std::vector<size_t>{rows, hidden}, DType::kFloat32);
    TensorWrapper tg(gamma, std::vector<size_t>{hidden}, DType::kFloat32);
    TensorWrapper tb(beta, std::vector<size_t>{hidden}, DType::kFloat32);
    TensorWrapper tz(output, std::vector<size_t>{rows, hidden}, DType::kFloat32);
    TensorWrapper tm(mu, std::vector<size_t>{rows}, DType::kFloat32);
    TensorWrapper tr(rsigma, std::vector<size_t>{rows}, DType::kFloat32);
    TensorWrapper workspace;
    const auto run = [&] {
      nvte_layernorm_fwd(tx.data(), tg.data(), tb.data(), 1e-5f, tz.data(), tm.data(), tr.data(),
                         workspace.data(), cuda::sm_count(0), false, nullptr);
    };
    run();
    void *scratch = nullptr;
    const auto shape = workspace.shape();
    size_t bytes = typeToSize(workspace.dtype());
    for (size_t i = 0; i < shape.ndim; ++i) bytes *= shape.data[i];
    ASSERT_EQ(cudaMalloc(&scratch, bytes), cudaSuccess);
    workspace.set_rowwise_data(scratch, workspace.dtype(), shape);
    run();
    std::vector<float> actual(input.size());
    ASSERT_EQ(
        cudaMemcpy(actual.data(), output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        cudaSuccess);
    const float mean = (hidden - 1) / 2.0f;
    const float variance = (hidden * hidden - 1) / 12.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], (input[i] - mean) / std::sqrt(variance + 1e-5f), 2e-5f);
    }
    reset();
  }
  nvte_enable_cudnn_norm_fwd(false);
}

TEST_F(ContextResetTest, CublasGemmRecoversNumerically) {
  constexpr size_t dim = 16, elements = dim * dim;
  constexpr size_t scratch_bytes = 32 * 1024 * 1024;
  for (int cycle = 0; cycle < 3; ++cycle) {
    float *data = nullptr;
    void *scratch = nullptr;
    ASSERT_EQ(cudaMalloc(&data, 3 * elements * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&scratch, scratch_bytes), cudaSuccess);
    std::vector<float> a(elements, 1.0f + cycle), b(elements, 2.0f), actual(elements);
    ASSERT_EQ(cudaMemcpy(data, a.data(), elements * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(data + elements, b.data(), elements * sizeof(float), cudaMemcpyHostToDevice),
        cudaSuccess);
    TensorWrapper ta(data, std::vector<size_t>{dim, dim}, DType::kFloat32);
    TensorWrapper tb(data + elements, std::vector<size_t>{dim, dim}, DType::kFloat32);
    TensorWrapper td(data + 2 * elements, std::vector<size_t>{dim, dim}, DType::kFloat32);
    TensorWrapper workspace(scratch, std::vector<size_t>{scratch_bytes}, DType::kByte);
    TensorWrapper absent;
    nvte_cublas_gemm(ta.data(), tb.data(), td.data(), absent.data(), absent.data(), true, false,
                     false, workspace.data(), false, true, 0, nullptr);
    ASSERT_EQ(cudaMemcpy(actual.data(), data + 2 * elements, elements * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (auto value : actual) EXPECT_EQ(value, 2.0f * dim * (cycle + 1));
    reset();
  }
}
}  // namespace
