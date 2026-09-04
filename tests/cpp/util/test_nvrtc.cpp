/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

#include "util/rtc.h"

using namespace transformer_engine;

TEST(UtilTest, NVRTC) {
  if (!rtc::is_enabled()) {
    GTEST_SKIP() << "NVRTC not enabled, skipping tests";
  }

  // GPU data buffer
  int* device_buffer;
  std::vector<int> host_buffer(2);
  cudaMalloc((void**)&device_buffer, 2 * sizeof(int));  // NOLINT(*)
  cudaMemset(device_buffer, 0, 2 * sizeof(int));

  // CUDA kernel implementations
  const char code1[] = R"code(
#include <cuda_runtime.h>
__global__ void my_kernel(int2 *data) {
  data->x = 123;
  data->y = -456;
}
)code";
  const char code2[] = R"code(
#include "utils.cuh"
__global__ void my_kernel(uint32_t *data) {
  data[0] = 789;
  data[1] = 12;
}
)code";
  const char code3[] = R"code(
#ifndef NVTE_GTEST_RTC_VALUE
#error "NVTE_GTEST_RTC_VALUE must be provided"
#endif
__global__ void my_kernel(int *data) {
  data[0] = NVTE_GTEST_RTC_VALUE;
  data[1] = 34;
}
)code";
  const char header4[] = R"code(
#define NVTE_GTEST_RTC_HEADER_VALUE 78
)code";
  const char code4[] = R"code(
#include "test_nvrtc_header.h"
__global__ void my_kernel(int *data) {
  data[0] = NVTE_GTEST_RTC_HEADER_VALUE;
  data[1] = 90;
}
)code";

  // Make sure kernels are not available
  auto& nvrtc_manager = rtc::KernelManager::instance();
  EXPECT_FALSE(nvrtc_manager.is_compiled("my gtest kernel1"));
  EXPECT_FALSE(nvrtc_manager.is_compiled("my gtest kernel2"));
  EXPECT_THROW(nvrtc_manager.launch("my gtest kernel1", 1, 1, 0, 0, device_buffer),
               std::runtime_error);
  EXPECT_THROW(nvrtc_manager.launch("my gtest kernel2", 1, 1, 0, 0, device_buffer),
               std::runtime_error);

  // Compile and run first kernel
  EXPECT_NO_THROW(
      nvrtc_manager.compile("my gtest kernel1", "my_kernel", code1, "test_nvrtc_kernel1.cu"));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel1"));
  EXPECT_FALSE(nvrtc_manager.is_compiled("my gtest kernel2"));
  EXPECT_NO_THROW(nvrtc_manager.launch("my gtest kernel1", 1, 1, 0, 0, device_buffer));
  EXPECT_EQ(cudaMemcpy(host_buffer.data(), device_buffer, 2 * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host_buffer[0], 123);
  EXPECT_EQ(host_buffer[1], -456);

  // Compile and run second kernel
  EXPECT_NO_THROW(
      nvrtc_manager.compile("my gtest kernel2", "my_kernel", code2, "test_nvrtc_kernel2.cu"));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel1"));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel2"));
  EXPECT_NO_THROW(nvrtc_manager.launch("my gtest kernel2", 1, 1, 0, 0, device_buffer));
  EXPECT_EQ(cudaMemcpy(host_buffer.data(), device_buffer, 2 * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host_buffer[0], 789);
  EXPECT_EQ(host_buffer[1], 12);

  // Compile and run kernel with extra compile options
  EXPECT_NO_THROW(nvrtc_manager.compile("my gtest kernel3", "my_kernel", code3,
                                        "test_nvrtc_kernel3.cu", {"-DNVTE_GTEST_RTC_VALUE=56"}));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel3"));
  EXPECT_NO_THROW(nvrtc_manager.launch("my gtest kernel3", 1, 1, 0, 0, device_buffer));
  EXPECT_EQ(cudaMemcpy(host_buffer.data(), device_buffer, 2 * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host_buffer[0], 56);
  EXPECT_EQ(host_buffer[1], 34);

  // Compile and run kernel with an extra in-memory header
  EXPECT_NO_THROW(nvrtc_manager.compile("my gtest kernel4", "my_kernel", code4,
                                        "test_nvrtc_kernel4.cu", {},
                                        {{header4, "test_nvrtc_header.h"}}));
  EXPECT_TRUE(nvrtc_manager.is_compiled("my gtest kernel4"));
  EXPECT_NO_THROW(nvrtc_manager.launch("my gtest kernel4", 1, 1, 0, 0, device_buffer));
  EXPECT_EQ(cudaMemcpy(host_buffer.data(), device_buffer, 2 * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host_buffer[0], 78);
  EXPECT_EQ(host_buffer[1], 90);

  EXPECT_EQ(cudaFree(device_buffer), cudaSuccess);
}

TEST(UtilTest, NVRTCConcurrentCompile) {
  if (!rtc::is_enabled()) {
    GTEST_SKIP() << "NVRTC not enabled, skipping tests";
  }

  constexpr int num_threads = 8;
  constexpr char kernel_label[] = "my concurrent gtest kernel";
  const char code[] = R"code(
__global__ void my_kernel(int *data) {
  data[0] = 314;
  data[1] = 159;
}
)code";

  int device_id = 0;
  ASSERT_EQ(cudaGetDevice(&device_id), cudaSuccess);

  auto& nvrtc_manager = rtc::KernelManager::instance();
  ASSERT_FALSE(nvrtc_manager.is_compiled(kernel_label));

  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::vector<std::exception_ptr> errors(num_threads);
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads.emplace_back([&, thread_id] {
      ready.fetch_add(1, std::memory_order_release);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        const cudaError_t status = cudaSetDevice(device_id);
        if (status != cudaSuccess) {
          throw std::runtime_error(cudaGetErrorString(status));
        }
        (void)nvrtc_manager.is_compiled(kernel_label);
        nvrtc_manager.compile(kernel_label, "my_kernel", code, "test_nvrtc_concurrent_kernel.cu");
      } catch (...) {
        errors[thread_id] = std::current_exception();
      }
    });
  }

  while (ready.load(std::memory_order_acquire) != num_threads) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (const auto& error : errors) {
    if (error != nullptr) {
      try {
        std::rethrow_exception(error);
      } catch (const std::exception& e) {
        ADD_FAILURE() << e.what();
      }
    }
  }

  ASSERT_TRUE(nvrtc_manager.is_compiled(kernel_label));
  int* device_buffer = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device_buffer), 2 * sizeof(int)), cudaSuccess);
  ASSERT_NO_THROW(nvrtc_manager.launch(kernel_label, 1, 1, 0, 0, device_buffer));
  std::vector<int> host_buffer(2);
  ASSERT_EQ(cudaMemcpy(host_buffer.data(), device_buffer, 2 * sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(host_buffer[0], 314);
  EXPECT_EQ(host_buffer[1], 159);
  EXPECT_EQ(cudaFree(device_buffer), cudaSuccess);
}
