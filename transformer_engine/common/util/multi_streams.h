/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
#define TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_

#include <transformer_engine/multi_streams.h>

#include <mutex>
#include <vector>

#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine::detail {

static std::once_flag init_flag;
static cudaStream_t compute_streams[num_streams];
static cudaEvent_t events[num_streams];

// Warning: only call once per device!
static void init_streams_and_events() {
  for (int i = 0; i < num_streams; i++) {
    NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&compute_streams[i], cudaStreamNonBlocking, -1));
    NVTE_CHECK_CUDA(cudaEventCreate(&events[i]));
  }
}

}  // namespace transformer_engine::detail

#endif  // TRANSFORMER_ENGINE_UTIL_MULTI_STREAM_H_
