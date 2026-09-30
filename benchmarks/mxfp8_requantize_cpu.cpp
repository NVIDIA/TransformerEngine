/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Host timing helper: Python/ctypes overhead is outside the timed C API call.
#include <transformer_engine/cast.h>

#include <cuda_runtime_api.h>
#include <time.h>

#include <chrono>
#include <cstdint>

namespace {
uint64_t thread_ns() {
  timespec timestamp{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &timestamp);
  return uint64_t(timestamp.tv_sec) * 1000000000 + timestamp.tv_nsec;
}

uint64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

extern "C" int measure_requantize_cpu(
    NVTEGroupedTensor input, NVTEGroupedTensor output, NVTEQuantizationConfig config,
    cudaStream_t stream, int iterations, int mode, uint64_t *wall_samples,
    uint64_t *cpu_samples) {
  // mode 0: eager launches, synchronize BEFORE each timing interval.
  // mode 1: graph-node capture, no GPU work runs during the timed intervals.
  // mode 2: timer-only calibration, with no TE or CUDA call inside the interval.
  // mode 3: eager batches of at most 64 queued launches. Drain before a batch,
  //         outside timing, to keep GPU queue backpressure out of the result.
  cudaGraph_t graph = nullptr;
  auto status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) return status;
  if (mode == 1) {
    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) return status;
  }
  for (int i = 0; i < iterations; ++i) {
    if (mode == 0 || (mode == 3 && i % 64 == 0)) {
      status = cudaStreamSynchronize(stream);
      if (status != cudaSuccess) return status;
    }
    const auto cpu_begin = thread_ns();
    const auto wall_begin = wall_ns();
    if (mode != 2) nvte_grouped_requantize(input, output, config, stream);
    const auto wall_end = wall_ns();
    const auto cpu_end = thread_ns();
    wall_samples[i] = wall_end - wall_begin;
    cpu_samples[i] = cpu_end - cpu_begin;
  }
  if (mode == 1) {
    status = cudaStreamEndCapture(stream, &graph);
    if (status != cudaSuccess) return status;
    status = cudaGraphDestroy(graph);
    if (status != cudaSuccess) return status;
  }
  return cudaStreamSynchronize(stream);
}
