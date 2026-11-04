/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <transformer_engine/multi_stream.h>

#include "cuda_runtime.h"

TEST(MultiStream, DeviceSwitch) {
  int num_devices = 0;
  ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);
  if (num_devices < 2) GTEST_SKIP() << "Requires two CUDA devices";
  int original_device = 0;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  for (int device : {0, 1, 0, 1}) {
    ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
    cudaStream_t caller;
    ASSERT_EQ(cudaStreamCreate(&caller), cudaSuccess);
    for (int idx = 0; idx < nvte_get_num_compute_streams(); ++idx) {
      auto stream = nvte_get_compute_stream(idx);
      auto event = nvte_get_compute_stream_event(idx);
      ASSERT_EQ(cudaEventRecord(event, caller), cudaSuccess);
      ASSERT_EQ(cudaStreamWaitEvent(stream, event), cudaSuccess);
      ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess);
      ASSERT_EQ(cudaStreamWaitEvent(caller, event), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(caller), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(caller), cudaSuccess);
  }
  ASSERT_EQ(cudaSetDevice(original_device), cudaSuccess);
}
