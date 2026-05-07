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
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine::detail {

namespace {

// CUDA streams and events are device-bound: a stream / event created
// on device A cannot be recorded into / waited on from device B
// (CUDA returns ``cudaErrorInvalidResourceHandle``). The previous
// implementation used ``std::call_once`` to lazily create one
// process-global vector of streams + one of events, which works for
// the single-device case (PyTorch eager / single-host single-device
// JAX) but breaks for single-process *multi*-device JAX: the first
// worker thread to win the ``call_once`` would create streams /
// events on its own device, and subsequent calls from other devices
// would receive those same handles and fail at ``cudaEventRecord``.
//
// We now key the cache on the active CUDA device. Each device gets
// its own ``num_compute_streams`` streams and events, created lazily
// the first time a thread on that device asks for one.
template <typename CreateFn>
auto& per_device_pool(CreateFn&& create) {
  static std::mutex mu;
  using PoolT = decltype(std::vector{create()});
  static std::unordered_map<int, PoolT> pools;
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  std::lock_guard<std::mutex> lock(mu);
  auto it = pools.find(device);
  if (it == pools.end()) {
    const size_t num_streams = nvte_get_num_compute_streams();
    PoolT v;
    v.reserve(num_streams);
    for (size_t i = 0; i < num_streams; i++) {
      v.push_back(create());
    }
    it = pools.emplace(device, std::move(v)).first;
  }
  return it->second;
}

}  // namespace

cudaStream_t get_compute_stream(int idx) {
  const size_t num_streams = nvte_get_num_compute_streams();
  NVTE_CHECK(0 <= idx && idx < num_streams, "Invalid compute stream (requested idx ", idx,
             ", but there are ", num_streams, " streams)");
  auto& streams = per_device_pool([] {
    cudaStream_t s;
    NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, -1));
    return s;
  });
  return streams[idx];
}

cudaEvent_t get_compute_stream_event(int idx) {
  const size_t num_streams = nvte_get_num_compute_streams();
  NVTE_CHECK(0 <= idx && idx < num_streams, "Invalid compute stream (requested idx ", idx,
             ", but there are ", num_streams, " streams)");
  auto& events = per_device_pool([] {
    cudaEvent_t e;
    NVTE_CHECK_CUDA(cudaEventCreate(&e));
    return e;
  });
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
