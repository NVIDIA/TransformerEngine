/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <gtest/gtest.h>
#include <nvrtc.h>
#include <transformer_engine/cuda_context.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/multi_stream.h>
#include <transformer_engine/normalization.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "util/cuda_context.h"
#include "util/cuda_state_registration.h"
#include "util/rtc.h"

extern "C" cudnnHandle_t nvte_get_cudnn_handle();
using namespace transformer_engine;

namespace {
struct RegisteredOwner : cuda_context::FrontendResource {
  RegisteredOwner(std::vector<int> &events, int id, int phase = NVTE_CUDA_STATE_RELEASE_PLANS,
                  bool fail = false)
      : FrontendResource(-1, phase), events(events), id(id) {
    if (fail) throw std::runtime_error("owner construction failed");
  }
  void release_context() override { events.push_back(id); }
  void invalidate_context() noexcept override { events.push_back(id); }
  std::vector<int> &events;
  int id;
};

TEST(CudaContextHostTest, ConstructorRegistrationAndRollbackUnregister) {
  std::vector<int> events;
  events.reserve(8);
  RegisteredOwner first(events, 1);
  { RegisteredOwner removed(events, 2); }
  EXPECT_THROW(RegisteredOwner(events, 3, NVTE_CUDA_STATE_RELEASE_PLANS, true), std::runtime_error);
  RegisteredOwner last(events, 4);
  const NVTECudaStateCallbacks invalid{nullptr, nullptr, nullptr, nullptr, 0};
  EXPECT_THROW(nvte_register_cuda_state(-1, invalid, nullptr), std::runtime_error);
  nvte_reset_cuda_state();
  EXPECT_EQ(events, (std::vector<int>{4, 1}));
}

TEST(CudaContextHostTest, CleanupPhasesAndReverseRegistrationOrder) {
  std::vector<int> events;
  events.reserve(16);
  RegisteredOwner handle(events, 1, NVTE_CUDA_STATE_RELEASE_HANDLES);
  RegisteredOwner plan_first(events, 2, NVTE_CUDA_STATE_RELEASE_PLANS);
  RegisteredOwner module(events, 3, NVTE_CUDA_STATE_RELEASE_MODULES);
  RegisteredOwner plan_last(events, 4, NVTE_CUDA_STATE_RELEASE_PLANS);
  RegisteredOwner stream(events, 5, NVTE_CUDA_STATE_RELEASE_STREAMS);
  const std::vector<int> expected{4, 2, 1, 3, 5};
  nvte_reset_cuda_state();
  EXPECT_EQ(events, expected);
  events.clear();
  nvte_release_cuda_state();
  EXPECT_EQ(events, expected);
}

TEST(CudaContextHostTest, AffinityChangesPreserveOriginalRegistrationOrder) {
  std::vector<int> events;
  events.reserve(8);
  RegisteredOwner first(events, 1), last(events, 2);
  first.set_device(0);
  last.set_device(-1);
  EXPECT_THROW(first.set_device(-2), std::runtime_error);
  EXPECT_THROW(nvte_set_cuda_state_device(nullptr, 0), std::runtime_error);
  nvte_reset_cuda_state();
  EXPECT_EQ(events, (std::vector<int>{2, 1}));
}

TEST(CudaContextHostTest, SharedInitializationPublishesOnceAndCanBeInvalidated) {
  cuda_context::InitializationState state;
  std::atomic<int> initialized{0};
  std::atomic<size_t> waiting{0};
  std::promise<void> start;
  auto begin = start.get_future().share();
  std::array<int, 8> observed{};
  int value = 0;
  std::vector<std::thread> threads;
  for (size_t i = 0; i < observed.size(); ++i) {
    threads.emplace_back([&, i] {
      waiting.fetch_add(1);
      begin.wait();
      state.ensure_valid([&] {
        initialized.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        value = 37;
      });
      observed[i] = value;
    });
  }
  while (waiting.load() != observed.size()) std::this_thread::yield();
  start.set_value();
  for (auto &thread : threads) thread.join();
  EXPECT_EQ(initialized.load(), 1);
  for (auto result : observed) EXPECT_EQ(result, 37);
  state.invalidate();
  state.ensure_valid([&] {
    ++initialized;
    value = 41;
  });
  EXPECT_EQ(initialized.load(), 2);
  EXPECT_EQ(value, 41);
}

TEST(CudaContextHostTest, FailedInitializationCanBeRetried) {
  cuda_context::InitializationState state;
  int attempts = 0;
  EXPECT_THROW(state.ensure_valid([&] {
    ++attempts;
    throw std::runtime_error("initialization failure");
  }),
               std::runtime_error);
  state.ensure_valid([&] { ++attempts; });
  state.ensure_valid([&] { ++attempts; });
  EXPECT_EQ(attempts, 2);
}

std::string lifetime_test_ptx() {
  auto check = [](nvrtcResult status) {
    if (status != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(status));
  };
  int count;
  check(nvrtcGetNumSupportedArchs(&count));
  std::vector<int> architectures(count);
  check(nvrtcGetSupportedArchs(architectures.data()));
  const auto option = "--gpu-architecture=compute_" +
                      std::to_string(*std::min_element(architectures.begin(), architectures.end()));
  const char *options[] = {option.c_str()};
  constexpr auto source = R"(
    extern "C" __global__ void first(int *p) { *p = 11; }
    extern "C" __global__ void second(int *p) { *p = 22; }
  )";
  nvrtcProgram program;
  check(nvrtcCreateProgram(&program, source, "lifetime.cu", 0, nullptr, nullptr));
  std::string ptx;
  try {
    check(nvrtcCompileProgram(program, 1, options));
    size_t bytes;
    check(nvrtcGetPTXSize(program, &bytes));
    ptx.resize(bytes);
    check(nvrtcGetPTX(program, ptx.data()));
  } catch (...) {
    nvrtcDestroyProgram(&program);
    throw;
  }
  check(nvrtcDestroyProgram(&program));
  return ptx;
}

class ContextResetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) GTEST_SKIP();
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
  void reset() {
    int count;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      CUdevice device;
      unsigned flags;
      int active;
      ASSERT_EQ(cuDeviceGet(&device, ordinal), CUDA_SUCCESS);
      ASSERT_EQ(cuDevicePrimaryCtxGetState(device, &flags, &active), CUDA_SUCCESS);
      if (active) {
        ASSERT_EQ(cudaSetDevice(ordinal), cudaSuccess);
        ASSERT_EQ(cudaDeviceReset(), cudaSuccess);
      }
    }
    const auto before = nvte_get_cuda_state_version();
    ASSERT_NO_THROW(nvte_reset_cuda_state());
    EXPECT_EQ(nvte_get_cuda_state_version(), before + 1);
    // The external caller recreates a usable context after the host-only reset.
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

TEST(CudaContextHostTest, GenerationDoesNotInitializeCuda) {
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  CUcontext before = nullptr, after = nullptr;
  ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);
  const auto initial = nvte_get_cuda_state_version();
  nvte_reset_cuda_state();
  EXPECT_EQ(nvte_get_cuda_state_version(), initial + 1);
  nvte_reset_cuda_state();
  EXPECT_EQ(nvte_get_cuda_state_version(), initial + 2);
  ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
  EXPECT_EQ(before, after);
}

TEST(CudaContextHostTest, LiveReleaseDoesNotCreateUnusedContexts) {
  ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
  int count;
  ASSERT_EQ(cuDeviceGetCount(&count), CUDA_SUCCESS);
  std::vector<int> before(count);
  for (int device = 0; device < count; ++device) {
    unsigned flags;
    ASSERT_EQ(cuDevicePrimaryCtxGetState(device, &flags, &before[device]), CUDA_SUCCESS);
  }
  CUcontext previous, current;
  ASSERT_EQ(cuCtxGetCurrent(&previous), CUDA_SUCCESS);
  int calls = 0;
  auto callback = [](void *data) { ++*static_cast<int *>(data); };
  const NVTECudaStateCallbacks callbacks{nullptr, nullptr, callback, callback, 0};
  const auto registration = nvte_register_cuda_state(0, callbacks, &calls);
  const auto global = nvte_register_cuda_state(-1, callbacks, &calls);
  const auto version = nvte_get_cuda_state_version();
  EXPECT_NO_THROW(nvte_release_cuda_state());
  EXPECT_EQ(nvte_get_cuda_state_version(), version + 1);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(cuCtxGetCurrent(&current), CUDA_SUCCESS);
  EXPECT_EQ(previous, current);
  for (int device = 0; device < count; ++device) {
    unsigned flags;
    int active;
    EXPECT_EQ(cuDevicePrimaryCtxGetState(device, &flags, &active), CUDA_SUCCESS);
    EXPECT_EQ(active, before[device]);
  }
  nvte_unregister_cuda_state(registration);
  nvte_unregister_cuda_state(global);
}

TEST(CudaContextHostTest, PartialReleaseContinuesAndPublishesOneVersion) {
  int failed = 0, released = 0;
  auto release = [](void *data) { ++*static_cast<int *>(data); };
  auto fail = [](void *data) {
    ++*static_cast<int *>(data);
    throw std::runtime_error("release failure");
  };
  const NVTECudaStateCallbacks bad{nullptr, nullptr, fail, release, 0};
  const NVTECudaStateCallbacks good{nullptr, nullptr, release, release, 0};
  const auto second = nvte_register_cuda_state(-1, good, &released);
  const auto first = nvte_register_cuda_state(-1, bad, &failed);
  const auto version = nvte_get_cuda_state_version();
  EXPECT_THROW(nvte_release_cuda_state(), std::runtime_error);
  EXPECT_EQ(failed, 1);
  EXPECT_EQ(released, 1);
  EXPECT_EQ(nvte_get_cuda_state_version(), version + 1);
  nvte_unregister_cuda_state(first);
  nvte_unregister_cuda_state(second);
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

TEST_F(ContextResetTest, GlobalResetRecoversBothDevicesAndKeepsRtcCode) {
  int count;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) GTEST_SKIP() << "Requires two CUDA devices";
  auto &manager = rtc::KernelManager::instance();
  constexpr auto code = R"(__global__ void recovered(int *p, int value) { *p = value; })";
  for (int device = 0; device < 2; ++device) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    manager.compile("global recovery test", "recovered", code, "recovered.cu");
  }
  for (int cycle = 0; cycle < 3; ++cycle) {
    for (int device = 0; device < 2; ++device) {
      ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
      EXPECT_TRUE(manager.is_compiled("global recovery test", device));
      int *data;
      ASSERT_EQ(cudaMalloc(&data, sizeof(int)), cudaSuccess);
      auto stream = nvte_get_compute_stream(0);
      const int expected = 10 * cycle + device;
      manager.launch("global recovery test", 1, 1, 0, stream, data, expected);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      int value = -1;
      ASSERT_EQ(cudaMemcpy(&value, data, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(value, expected);
    }
    reset();
  }
}

TEST_F(ContextResetTest, GlobalReleaseRecreatesBothDevicesAndRestoresContext) {
  int count;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) GTEST_SKIP() << "Requires two CUDA devices";
  auto &manager = rtc::KernelManager::instance();
  constexpr auto code = R"(__global__ void recovered(int *p, int value) { *p = value; })";
  std::vector<int *> storage(2);
  for (int device = 0; device < 2; ++device) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&storage[device], sizeof(int)), cudaSuccess);
    manager.compile("global release test", "recovered", code, "recovered.cu");
  }
  for (int cycle = 0; cycle < 3; ++cycle) {
    for (int device = 0; device < 2; ++device) {
      ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
      auto stream = nvte_get_compute_stream(0);
      auto event = nvte_get_compute_stream_event(0);
      auto handle = nvte_get_cudnn_handle();
      ASSERT_EQ(cudnnSetStream(handle, stream), CUDNN_STATUS_SUCCESS);
      const int expected = cycle + device;
      manager.launch("global release test", 1, 1, 0, stream, storage[device], expected);
      ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
      ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);
      int value = -1;
      ASSERT_EQ(cudaMemcpy(&value, storage[device], sizeof(int), cudaMemcpyDeviceToHost),
                cudaSuccess);
      EXPECT_EQ(value, expected);
    }
    CUcontext before, after;
    ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);
    const auto version = nvte_get_cuda_state_version();
    nvte_release_cuda_state();
    EXPECT_EQ(nvte_get_cuda_state_version(), version + 1);
    ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
    EXPECT_EQ(before, after);
    int current;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 1);
  }
  for (int device = 0; device < 2; ++device) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    ASSERT_EQ(cudaFree(storage[device]), cudaSuccess);
  }
  reset();
}

TEST_F(ContextResetTest, GlobalReleaseErrorRestoresContextAndVisitsOtherDevices) {
  int count;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) GTEST_SKIP() << "Requires two CUDA devices";
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  struct Owner {
    int device;
    bool fail;
    int calls = 0;
  } first{0, true}, second{1, false};
  auto cleanup = [](void *data) {
    auto &owner = *static_cast<Owner *>(data);
    int current;
    EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, owner.device);
    ++owner.calls;
    if (owner.fail) throw std::runtime_error("affine release failure");
  };
  const NVTECudaStateCallbacks callbacks{nullptr, nullptr, cleanup, [](void *) {}, 0};
  auto b = nvte_register_cuda_state(1, callbacks, &second);
  auto a = nvte_register_cuda_state(0, callbacks, &first);
  CUcontext before, after;
  ASSERT_EQ(cuCtxGetCurrent(&before), CUDA_SUCCESS);
  const auto version = nvte_get_cuda_state_version();
  EXPECT_THROW(nvte_release_cuda_state(), std::runtime_error);
  EXPECT_EQ(first.calls, 1);
  EXPECT_EQ(second.calls, 1);
  EXPECT_EQ(nvte_get_cuda_state_version(), version + 1);
  ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
  EXPECT_EQ(before, after);
  nvte_unregister_cuda_state(a);
  nvte_unregister_cuda_state(b);
  reset();
}

TEST_F(ContextResetTest, MovedRtcKernelReinitializesWithItsCurrentParameters) {
  const auto ptx = lifetime_test_ptx();
  rtc::Kernel first("first", ptx), second("second", ptx);
  swap(first, second);
  int *data;
  ASSERT_EQ(cudaMalloc(&data, sizeof(int)), cudaSuccess);
  const auto execute = [&](rtc::Kernel &kernel, int expected) {
    kernel.launch(0, 1, 1, 0, nullptr, data);
    int value = 0;
    ASSERT_EQ(cudaMemcpy(&value, data, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(value, expected);
  };
  execute(first, 22);
  rtc::Kernel moved(std::move(first));
  nvte_release_cuda_state();
  execute(moved, 22);
  execute(second, 11);
  nvte_release_cuda_state();
  ASSERT_EQ(cudaFree(data), cudaSuccess);
}

TEST_F(ContextResetTest, FailedRtcInitializationDiscardsPartialStateOnReset) {
  rtc::Kernel kernel("missing_symbol", lifetime_test_ptx());
  const auto lookup_error = [&]() -> std::string {
    try {
      kernel.get_function(0);
    } catch (const std::runtime_error &error) {
      return error.what();
    }
    return {};
  };
  const auto first_error = lookup_error();
  ASSERT_FALSE(first_error.empty());
  ASSERT_NO_THROW(nvte_release_cuda_state());
  // Acquire another partial module before context loss. Its initializer never
  // becomes valid, but reset must still discard the dead module handle.
  EXPECT_EQ(lookup_error(), first_error);
  reset();
  EXPECT_EQ(lookup_error(), first_error);
  nvte_release_cuda_state();
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

TEST_F(ContextResetTest, FinishedThreadsReturnHandlesForReleaseAndReuse) {
  cudnnHandle_t first = nullptr, second = nullptr;
  std::thread([&]() {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    first = nvte_get_cudnn_handle();
  }).join();
  std::thread([&]() {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    second = nvte_get_cudnn_handle();
  }).join();
  ASSERT_NE(first, nullptr);
  // Deliberate handoff of the same live cached handle to a subsequent thread.
  EXPECT_EQ(first, second);
  nvte_release_cuda_state();
  std::thread([&]() {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto handle = nvte_get_cudnn_handle();
    EXPECT_EQ(cudnnSetStream(handle, nvte_get_compute_stream(0)), CUDNN_STATUS_SUCCESS);
  }).join();
  nvte_release_cuda_state();
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

struct CallbackOwner {
  NVTECudaStateRegistration child = nullptr;
  int calls = 0;
  static void invalidate(void *data) {
    auto &self = *static_cast<CallbackOwner *>(data);
    ++self.calls;
    nvte_unregister_cuda_state(self.child);
    self.child = nullptr;
  }
  static void reject(void *) { throw std::runtime_error("resource requires pre-loss cleanup"); }
};

TEST(CudaContextHostTest, CallbackPreflightAndNestedUnregister) {
  CallbackOwner parent, child;
  const NVTECudaStateCallbacks normal{nullptr, nullptr, CallbackOwner::invalidate,
                                      CallbackOwner::invalidate, 0};
  parent.child = nvte_register_cuda_state(0, normal, &child);
  auto registration = nvte_register_cuda_state(1, normal, &parent);
  const NVTECudaStateCallbacks rejecting{CallbackOwner::reject, nullptr, CallbackOwner::invalidate,
                                         CallbackOwner::invalidate, 0};
  auto rejected = nvte_register_cuda_state(-1, rejecting, &child);
  const auto initial = nvte_get_cuda_state_version();
  EXPECT_THROW(nvte_reset_cuda_state(), std::runtime_error);
  EXPECT_EQ(nvte_get_cuda_state_version(), initial);
  EXPECT_EQ(parent.calls, 0);
  EXPECT_EQ(child.calls, 0);
  nvte_unregister_cuda_state(rejected);
  nvte_reset_cuda_state();
  EXPECT_EQ(parent.calls, 1);
  EXPECT_EQ(child.calls, 0);
  EXPECT_EQ(nvte_get_cuda_state_version(), initial + 1);
  nvte_unregister_cuda_state(registration);
}

TEST_F(ContextResetTest, LiveReleaseRecreatesStreamsAndLibraryHandles) {
  auto &manager = rtc::KernelManager::instance();
  constexpr auto code = R"(__global__ void recovered(int *p, int value) { *p = value; })";
  manager.compile("live release test", "recovered", code, "recovered.cu");
  int *storage = nullptr;
  ASSERT_EQ(cudaMalloc(&storage, sizeof(int)), cudaSuccess);
  for (int cycle = 0; cycle < 4; ++cycle) {
    auto stream = nvte_get_compute_stream(0);
    auto event = nvte_get_compute_stream_event(0);
    auto handle = nvte_get_cudnn_handle();
    ASSERT_EQ(cudnnSetStream(handle, stream), CUDNN_STATUS_SUCCESS);
    manager.launch("live release test", 1, 1, 0, stream, storage, cycle);
    ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(event), cudaSuccess);
    int value = -1;
    ASSERT_EQ(cudaMemcpy(&value, storage, sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(value, cycle);
    const auto previous = nvte_get_cuda_state_version();
    nvte_release_cuda_state();
    EXPECT_EQ(nvte_get_cuda_state_version(), previous + 1);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  }
  ASSERT_EQ(cudaFree(storage), cudaSuccess);
}
}  // namespace
