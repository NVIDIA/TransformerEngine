/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_OVERLAP_WORKSPACE_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_OVERLAP_WORKSPACE_H_

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace transformer_engine {
namespace ep {

// Bit patterns of the words a reset fills with: int32 0 and -1, float 0.0f and 1.0f.
constexpr int32_t kFillZero = 0;
constexpr int32_t kFillMinusOne = -1;
constexpr int32_t kFillOne = 0x3F800000;

// The words one launch writes before a call: constant fills and, optionally, the expert row ends
// as the inclusive prefix sums of the padded int32 counts.
struct WorkspaceReset {
  static constexpr int kMaxFills = 8;
  struct Fill {
    int32_t* words;
    int count;
    int32_t value;
  };
  Fill fills[kMaxFills];
  int num_fills = 0;
  const int32_t* counts = nullptr;  // [num_experts] padded counts; null: no row ends
  int32_t* row_ends = nullptr;      // [num_experts] row ends
  int num_experts = 0;

  // Fill count words from words with value (a kFill* pattern); a zero count adds nothing.
  void fill(void* words, int count, int32_t value);
};

// One launch performing every fill and the row-ends scan of reset.
void reset_workspace(const WorkspaceReset& reset, cudaStream_t stream);

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_OVERLAP_WORKSPACE_H_
