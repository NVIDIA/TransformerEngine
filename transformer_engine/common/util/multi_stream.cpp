/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
#define TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_

#include "multi_stream.h"

#include <transformer_engine/multi_stream.h>

#include <mutex>
#include <vector>

#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine::detail {

cudaStream_t get_compute_stream(int idx) {
  const size_t num_streams = nvte_get_num_compute_streams();
  NVTE_CHECK(0 <= idx && idx < num_streams, "Invalid compute stream (requested idx ", idx,
             ", but there are ", num_streams, " streams)");
  static std::vector<cudaStream_t> streams(num_streams);
  static std::once_flag stream_init_flag;
  auto init = [&]() {
    for (size_t i = 0; i < num_streams; i++) {
      NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, -1));
    }
  };
  std::call_once(stream_init_flag, init);
  return streams[idx];
}

cudaEvent_t get_compute_stream_event(int idx) {
  const size_t num_streams = nvte_get_num_compute_streams();
  NVTE_CHECK(0 <= idx && idx < num_streams, "Invalid compute stream (requested idx ", idx,
             ", but there are ", num_streams, " streams)");
  static std::vector<cudaEvent_t> events(num_streams);
  static std::once_flag event_init_flag;
  auto init = [&]() {
    for (size_t i = 0; i < num_streams; i++) {
      NVTE_CHECK_CUDA(cudaEventCreate(&events[i]));
    }
  };
  std::call_once(event_init_flag, init);
  return events[idx];
}

int get_num_compute_streams() {
  static constexpr int num_compute_streams = 4;
  return num_compute_streams;
}

}  // namespace transformer_engine::detail

int nvte_get_num_compute_streams() { return transformer_engine::detail::get_num_compute_streams(); }

cudaStream_t nvte_get_compute_stream(const int idx) {
  return transformer_engine::detail::get_compute_stream(idx);
}

cudaEvent_t nvte_get_compute_stream_event(const int idx) {
  return transformer_engine::detail::get_compute_stream_event(idx);
}

#endif  // TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
