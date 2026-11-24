/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cub/cub.cuh>

#include "../util/logging.h"
#include "ep_overlap_workspace.h"

namespace transformer_engine {
namespace ep {

namespace {

constexpr int kThreads = 1024;

__global__ void reset_workspace_kernel(WorkspaceReset reset) {
  const int tid = threadIdx.x;
  const int index = blockIdx.x * kThreads + tid;
  const int stride = gridDim.x * kThreads;
#pragma unroll
  for (int f = 0; f < WorkspaceReset::kMaxFills; ++f) {
    if (f < reset.num_fills) {
      for (int i = index; i < reset.fills[f].count; i += stride) {
        reset.fills[f].words[i] = reset.fills[f].value;
      }
    }
  }
  if (blockIdx.x != 0 || reset.counts == nullptr) {
    return;
  }
  // Inclusive prefix sum, kThreads experts per step.
  using BlockScan = cub::BlockScan<int32_t, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int32_t running = 0;
  for (int first = 0; first < reset.num_experts; first += kThreads) {
    const int e = first + tid;
    const int32_t count = e < reset.num_experts ? reset.counts[e] : 0;
    int32_t row_end = 0;
    int32_t step_total = 0;
    BlockScan(scan_storage).InclusiveSum(count, row_end, step_total);
    if (e < reset.num_experts) {
      reset.row_ends[e] = running + row_end;
    }
    running += step_total;
    __syncthreads();
  }
}

}  // namespace

void WorkspaceReset::fill(void* words, int count, int32_t value) {
  if (count == 0) {
    return;
  }
  NVTE_CHECK(num_fills < kMaxFills, "too many workspace fills");
  fills[num_fills++] = {static_cast<int32_t*>(words), count, value};
}

void reset_workspace(const WorkspaceReset& reset, cudaStream_t stream) {
  // One block per kThreads words of the largest fill.
  int most_words = 1;
  for (int f = 0; f < reset.num_fills; ++f) {
    most_words = std::max(most_words, reset.fills[f].count);
  }
  const int blocks = (most_words + kThreads - 1) / kThreads;
  reset_workspace_kernel<<<blocks, kThreads, 0, stream>>>(reset);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace ep
}  // namespace transformer_engine
