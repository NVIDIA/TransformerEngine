/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include <climits>
#include <vector>

#include "../common.h"
#include "../util/logging.h"
#include "async_loader.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

// =============================================================================
// Simple forward kernel — exact upstream structure (no async loader, no
// persistent grid, runtime score_function dispatch).  Faster for small topk
// due to lower scheduling overhead and separate load/compute/store phases.
// =============================================================================

template <typename DataType, typename BiasType, NVTERoutingMapFormat RoutingMapFormat,
          TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_topk_forward_simple_kernel(const DataType *logits, int num_tokens,
                                                 int num_experts, int topk, bool use_pre_softmax,
                                                 int num_groups, int group_topk,
                                                 float scaling_factor, int score_function,
                                                 const BiasType *expert_bias, DataType *probs,
                                                 uint8_t *routing_map,
                                                 CompType *intermediate_output) {
  constexpr bool kIsBitmap = (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  CompType *scores_buf = reinterpret_cast<CompType *>(shmem);
  CompType *topk_scores_buf = scores_buf + num_experts * num_token_per_block;
  CompType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = topk_scores_buf + topk * num_token_per_block;
    group_scores_buf = masked_scores_buf + num_experts * num_token_per_block;
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }
  const int bitmap_words_per_warp = (num_experts + 31) / 32;
  const int bitmap_row_bytes = (num_experts + 7) / 8;
  uint32_t *bitmap_words_buf = nullptr;
  if constexpr (kIsBitmap) {
    bitmap_words_buf = reinterpret_cast<uint32_t *>(topk_indices_buf + topk * num_token_per_block);
  }
  CompType *scores = scores_buf + warp_id * num_experts;
  CompType *topk_scores = topk_scores_buf + warp_id * topk;
  CompType *masked_scores = masked_scores_buf + warp_id * num_experts;
  CompType *group_scores = group_scores_buf + warp_id * num_groups;
  int *topk_indices = topk_indices_buf + warp_id * topk;
  uint32_t *local_bitmap_words =
      (bitmap_words_buf != nullptr) ? bitmap_words_buf + warp_id * bitmap_words_per_warp : nullptr;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) break;

    int pos_offset = token_offset_cur_warp * num_experts;
    // Clear the probs/routing_map (num_experts)
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      probs[pos_offset + i] = 0.0;
      if (score_function == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
      }
    }
    if constexpr (!kIsBitmap) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        routing_map[pos_offset + i] = 0;
      }
    } else {
      for (int i = lane_id; i < bitmap_words_per_warp; i += kThreadsPerWarp) {
        local_bitmap_words[i] = 0u;
      }
    }
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[i] = logits[pos_offset + i];
    }
    // If group_topk > 0, init the masked_scores to -inf
    if (group_topk > 0) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<CompType>::infinity();
      }
    }
    __threadfence_block();
    __syncwarp();

    // Preprocess: apply score function in-place on shmem
    if (use_pre_softmax && score_function == 1) {
      apply_softmax_on_float(scores, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 0) {
      apply_sigmoid_on_float(scores, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 2) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(scores, num_experts, lane_id);
    }

    __syncwarp();

    // Expert bias (sigmoid/sqrtsoftplus only)
    if (expert_bias && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[i] += static_cast<CompType>(expert_bias[i]);
      }
      __syncwarp();
    }

    // Topk selection
    if (group_topk > 0) {
      int group_size = num_experts / num_groups;
      for (int i = 0; i < num_groups; i++) {
        topk_and_mask<TopkFunc>(scores + i * group_size, group_size, topk / group_topk,
                                topk_indices, topk_scores, lane_id);
        __syncwarp();
        if (lane_id == 0) {
          CompType tmp = 0.0;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp = tmp + topk_scores[j];
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }
      topk_and_mask<TopkFunc>(group_scores, num_groups, group_topk, topk_indices, topk_scores,
                              lane_id);
      __syncwarp();
      for (int i = 0; i < group_topk; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      topk_and_mask<TopkFunc>(masked_scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    } else {
      topk_and_mask<TopkFunc>(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }
    __syncwarp();

    // Postprocess: revert bias, softmax, normalization
    if (expert_bias && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        topk_scores[i] = topk_scores[i] - static_cast<CompType>(expert_bias[topk_indices[i]]);
      }
      __syncwarp();
    }

    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_on_float(topk_scores, topk, lane_id);
      __syncwarp();
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
      }
      __syncwarp();
    }

    if (score_function == 0 || score_function == 2) {
      if (topk > 1) {
        CompType sum_scores =
            warp_reduce_on_shmem<CompType, ReduceFuncType::SUM>(topk_scores, topk, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    // Write outputs
    if constexpr (!kIsBitmap) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        routing_map[pos_offset + topk_indices[i]] = 1;
        probs[pos_offset + topk_indices[i]] = scaling_factor * topk_scores[i];
      }
    } else {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        int e = topk_indices[i];
        atomicOr(&local_bitmap_words[e / 32], 1u << (e % 32));
        probs[pos_offset + e] = scaling_factor * topk_scores[i];
      }
      __syncwarp();
      uint8_t *bitmap_row =
          routing_map + static_cast<size_t>(token_offset_cur_warp) * bitmap_row_bytes;
      const uint8_t *local_bitmap_bytes = reinterpret_cast<const uint8_t *>(local_bitmap_words);
      for (int i = lane_id; i < bitmap_row_bytes; i += kThreadsPerWarp) {
        bitmap_row[i] = local_bitmap_bytes[i];
      }
    }
    __threadfence_block();
    __syncwarp();
  }
}

// =============================================================================
// Optimized forward kernel — async loader, persistent grid, double buffering.
// Used for larger topk where radix selection and compute dominate.
// =============================================================================

template <typename DataType, typename BiasType, NVTERoutingMapFormat RoutingMapFormat,
          TopkFuncType TopkFunc = TopkFuncType::Naive, int ScoreFunc = 0>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, const BiasType *expert_bias,
    DataType *probs, uint8_t *routing_map, CompType *intermediate_output, int num_buffers) {
  constexpr bool kIsBitmap = (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ char shmem_raw[];

  // Shmem layout: logits_raw (async) | scores | topk_scratch (+ group_topk scratch)
  char *shmem_ptr = shmem_raw;
  DataType *logits_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> loader(logits_shmem_base, warp_id, num_experts, num_token_per_block,
                                  num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *scores_buf = reinterpret_cast<CompType *>(shmem_ptr);
  shmem_ptr += num_experts * num_token_per_block * sizeof(CompType);

  CompType *topk_scores_buf = reinterpret_cast<CompType *>(shmem_ptr);
  CompType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = topk_scores_buf + topk * num_token_per_block;
    group_scores_buf = masked_scores_buf + num_experts * num_token_per_block;
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }
  const int bitmap_words_per_warp = (num_experts + 31) / 32;
  const int bitmap_row_bytes = (num_experts + 7) / 8;
  uint32_t *bitmap_words_buf = nullptr;
  if constexpr (kIsBitmap) {
    bitmap_words_buf = reinterpret_cast<uint32_t *>(topk_indices_buf + topk * num_token_per_block);
  }
  // The address of buffers on the current warp
  CompType *scores = scores_buf + warp_id * num_experts;
  CompType *topk_scores = topk_scores_buf + warp_id * topk;
  CompType *masked_scores =
      (masked_scores_buf != nullptr) ? masked_scores_buf + warp_id * num_experts : nullptr;
  CompType *group_scores =
      (group_scores_buf != nullptr) ? group_scores_buf + warp_id * num_groups : nullptr;
  int *topk_indices = topk_indices_buf + warp_id * topk;
  uint32_t *local_bitmap_words =
      (bitmap_words_buf != nullptr) ? bitmap_words_buf + warp_id * bitmap_words_per_warp : nullptr;

  /***
     * Section: Main Loop — persistent grid with double-buffered async load
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  // Kick off first async load
  {
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      loader.load_current(logits + first_token * num_experts, num_experts, lane_id);
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) break;

    // Single-buffer: load current round here (no prefetch possible)
    if (num_buffers == 1 && round != first_round) {
      loader.load_current(logits + token_offset_cur_warp * num_experts, num_experts, lane_id);
    }

    // Wait for current round's async load to complete
    loader.wait();
    DataType *raw_logits = loader.current_buf();

    // Prefetch next round (only when double-buffered, overlaps with compute)
    if (num_buffers > 1) {
      int next_round = round + gridDim.x;
      if (next_round < total_round) {
        int next_token = next_round * num_token_per_block + warp_id;
        if (next_token < num_tokens) {
          loader.start_load(logits + next_token * num_experts, num_experts, lane_id);
        }
      }
    }

    /***
         * Section: Init buffer + Preprocess
         * - Clear the global output buffers (probs, routing_map)
         * - Convert raw logits (DataType) → apply score function → save intermediate → add bias
         *
         * Fused into a single loop per score function where possible:
         *   score_function == 0 (sigmoid):      convert, sigmoid, save, +bias → scores
         *   score_function == 1 (softmax):      convert → shmem, softmax (multi-pass), save
         *   score_function == 2 (sqrtsoftplus): convert, save logits, sqrtsoftplus, +bias → scores
         *
         * Expert bias is only used with sigmoid/sqrtsoftplus and is fused into
         * the same loop that computes the score.
         */
    int pos_offset = token_offset_cur_warp * num_experts;

    // Clear the probs/routing_map (num_experts)
    vec_fill_global(probs + pos_offset, static_cast<DataType>(0.0f), num_experts, lane_id);
    if constexpr (!kIsBitmap) {
      vec_fill_global(routing_map + pos_offset, static_cast<uint8_t>(0), num_experts, lane_id);
    } else {
      for (int i = lane_id; i < bitmap_words_per_warp; i += kThreadsPerWarp) {
        local_bitmap_words[i] = 0u;
      }
    }

    if constexpr (ScoreFunc == 1) {  // Softmax
      if (use_pre_softmax) {
        // Pre-softmax: apply softmax to all logits before topk, save for backward.
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores[i] = static_cast<CompType>(raw_logits[i]);
        }
        __syncwarp();
        apply_softmax_on_float(scores, num_experts, lane_id);
        __syncwarp();
        // Save the softmax output for backward
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + i] = scores[i];
        }
      } else {
        // Post-softmax: softmax applied after topk; init intermediate to -inf
        // (only the topk positions will be filled in the postprocess section).
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores[i] = static_cast<CompType>(raw_logits[i]);
          intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
        }
      }
    } else if constexpr (ScoreFunc == 0) {  // Sigmoid
      // Fused: convert → sigmoid → save sigmoid output for backward → add bias → scores
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(raw_logits[i]));
        intermediate_output[pos_offset + i] = val;  // Save sigmoid output for backward
        if (expert_bias) val += static_cast<CompType>(expert_bias[i]);
        scores[i] = val;
      }
    } else if constexpr (ScoreFunc == 2) {  // Sqrtsoftplus
      // Fused: convert → save original logit for backward → sqrtsoftplus → add bias → scores
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(raw_logits[i]);
        intermediate_output[pos_offset + i] = logit;  // Save original logits for backward
        float val = sqrtsoftplus_scalar(logit);
        if (expert_bias) val += static_cast<CompType>(expert_bias[i]);
        scores[i] = val;
      }
    }
    __syncwarp();

    // If group_topk > 0, init the masked_scores to -inf
    if (group_topk > 0) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<CompType>::infinity();
      }
      __syncwarp();
    }

    /***
         * Section: Topk
         * Get the topk indices
         * - group_topk
         * - naive topk
         * - topk with expert bias
         */
    // Topk on the scores
    // The bias being not empty happens at the sigmoid/sqrtsoftplus case
    if (group_topk > 0) {
      int group_size = num_experts / num_groups;
      // Top2
      for (int i = 0; i < num_groups; i++) {
        topk_and_mask<TopkFunc>(
            /*scores ptr = */ scores + i * group_size,
            /*data size = */ group_size,
            /*topk = */ topk / group_topk,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
        __syncwarp();
        // Compute the group score
        if (lane_id == 0) {
          CompType tmp = 0.0;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp = tmp + topk_scores[j];
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }

      // select the topk groups
      topk_and_mask<TopkFunc>(
          /*scores ptr = */ group_scores,
          /*data size = */ num_groups,
          /*topk = */ group_topk,
          /*topk indices ptr = */ topk_indices,
          /*topk scores ptr = */ topk_scores,
          /*lane id = */ lane_id);
      __syncwarp();
      // Copy the unmasked scores to the buffer
      for (int i = 0; i < group_topk; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      topk_and_mask<TopkFunc>(masked_scores, num_experts, topk, topk_indices, topk_scores, lane_id);

    } else {
      topk_and_mask<TopkFunc>(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }
    __syncwarp();

    /***
         * Section: Postprocess
         * Possible postprocess the scores after the topk operation
         * - Revert Expert bias
         * - Softmax
         * - Sigmoid/Sqrtsoftplus post-processing when topk > 1
         * - Write the result with scaling_factor
         */
    // Revert Expert bias from the topk scores
    if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
      if (expert_bias) {
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] - static_cast<CompType>(expert_bias[topk_indices[i]]);
        }
        __syncwarp();
      }
    }

    if constexpr (ScoreFunc == 1) {
      if (!use_pre_softmax) {
        // Apply softmax to the topk logits
        apply_softmax_on_float(topk_scores, topk, lane_id);
        __syncwarp();
        // Save the softmax output for backward
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
        }
        __syncwarp();
      }
    }

    // Sigmoid/Sqrtsoftplus post-processing when topk > 1
    if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
      if (topk > 1) {
        CompType sum_scores =
            warp_reduce_on_shmem<CompType, ReduceFuncType::SUM>(topk_scores, topk, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    // Write the probs/routing_map to the output tensor
    if constexpr (!kIsBitmap) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        routing_map[pos_offset + topk_indices[i]] = 1;
        probs[pos_offset + topk_indices[i]] = scaling_factor * topk_scores[i];
      }
    } else {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        int e = topk_indices[i];
        atomicOr(&local_bitmap_words[e / 32], 1u << (e % 32));
        probs[pos_offset + e] = scaling_factor * topk_scores[i];
      }
      __syncwarp();
      uint8_t *bitmap_row =
          routing_map + static_cast<size_t>(token_offset_cur_warp) * bitmap_row_bytes;
      const uint8_t *local_bitmap_bytes = reinterpret_cast<const uint8_t *>(local_bitmap_words);
      for (int i = lane_id; i < bitmap_row_bytes; i += kThreadsPerWarp) {
        bitmap_row[i] = local_bitmap_bytes[i];
      }
    }
    __syncwarp();

    loader.flip();
  }
}

template <typename DataType, typename BiasType, NVTERoutingMapFormat RoutingMapFormat>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, uint8_t *routing_map,
    CompType *intermediate_output, cudaStream_t stream) {
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
             " num_experts=", num_experts);
  NVTE_CHECK(static_cast<int64_t>(num_tokens) * num_experts <= INT_MAX,
             "num_tokens * num_experts exceeds INT_MAX (kernel uses int offsets), got ",
             static_cast<int64_t>(num_tokens) * num_experts);
  NVTE_CHECK(score_function >= 0 && score_function <= 2,
             "Unsupported score_function: ", score_function);
  if (group_topk > 0) {
    NVTE_CHECK(topk % group_topk == 0, "topk must be divisible by group_topk, got topk=", topk,
               " group_topk=", group_topk);
  }
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t scores_shmem = num_experts * num_token_per_block * sizeof(CompType);
  size_t scratch_shmem =
      topk * num_token_per_block * sizeof(CompType) + topk * num_token_per_block * sizeof(int);
  if (group_topk > 0) {
    scratch_shmem += num_groups * num_token_per_block * sizeof(CompType);
    scratch_shmem += num_experts * num_token_per_block * sizeof(CompType);
  }
  if constexpr (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    scratch_shmem += ((num_experts + 31) / 32) * num_token_per_block * sizeof(uint32_t);
  }
  size_t other_shmem = scores_shmem + scratch_shmem;
  size_t logits_single_buf =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(logits_single_buf, other_shmem);
  size_t logits_raw_shmem =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  size_t shared_memory_size = logits_raw_shmem + other_shmem;

  auto launch = [&](auto kernel) {
    check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, expert_bias, probs, routing_map, intermediate_output, num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  // Dispatch: use radix only when it is profitable and supported. Otherwise use the
  // naive path, which handles very large expert counts without the radix histogram limit.
  const bool use_radix = topk >= get_radix_topk_threshold() && num_experts <= kMaxExpertsRadixTopk;
  if (!use_radix) {
    // Simple path: no async loader, no persistent grid.
    // Uses the exact upstream kernel structure with runtime score_function dispatch.
    check_shared_memory_capacity_num_experts(other_shmem, num_experts);

    auto launch_simple = [&](auto kernel) {
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, other_shmem));
      kernel<<<total_blocks, kThreadsPerBlock, other_shmem, stream>>>(
          logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
          scaling_factor, score_function, expert_bias, probs, routing_map, intermediate_output);
      NVTE_CHECK_CUDA(cudaGetLastError());
    };

    launch_simple(fused_topk_forward_simple_kernel<DataType, BiasType, RoutingMapFormat,
                                                   TopkFuncType::Naive>);
  } else {
    // Optimized path: async loader + persistent grid + radix topk.
    switch (score_function) {
      case 0:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                             TopkFuncType::Radix, 0>);
        break;
      case 1:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                             TopkFuncType::Radix, 1>);
        break;
      case 2:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                             TopkFuncType::Radix, 2>);
        break;
      default:
        NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
    }
  }
}

// Build the expected routing_map shape for a given NVTERoutingMapFormat.
//   BYTEMAP   -> [num_tokens, num_experts]
//   BITMAP_U8 -> [num_tokens, ceil(num_experts/8)]
static std::vector<size_t> expected_routing_map_shape(int num_tokens, int num_experts,
                                                      NVTERoutingMapFormat format) {
  const size_t t = static_cast<size_t>(num_tokens);
  const size_t e = static_cast<size_t>(num_experts);
  if (format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    return {t, (e + 7) / 8};
  }
  return {t, e};
}

void fused_topk_with_score_function_forward(const Tensor logits, int num_tokens, int num_experts,
                                            int topk, bool use_pre_softmax, int num_groups,
                                            int group_topk, float scaling_factor,
                                            int score_function, const Tensor expert_bias,
                                            Tensor probs, Tensor routing_map,
                                            NVTERoutingMapFormat routing_map_format,
                                            Tensor intermediate_output, cudaStream_t stream) {
  NVTE_CHECK(num_tokens > 0 && num_experts > 0,
             "num_tokens and num_experts must be positive; got num_tokens=", num_tokens,
             ", num_experts=", num_experts);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  NVTE_CHECK(logits.data.shape == dense_shape, "logits shape must be [num_tokens, num_experts]=[",
             num_tokens, ", ", num_experts, "], got ", logits.data.shape);
  NVTE_CHECK(probs.data.shape == dense_shape, "probs shape must be [num_tokens, num_experts]=[",
             num_tokens, ", ", num_experts, "], got ", probs.data.shape);
  NVTE_CHECK(intermediate_output.data.shape == dense_shape,
             "intermediate_output shape must be [num_tokens, num_experts]=[", num_tokens, ", ",
             num_experts, "], got ", intermediate_output.data.shape);
  const auto routing_map_shape =
      expected_routing_map_shape(num_tokens, num_experts, routing_map_format);
  NVTE_CHECK(routing_map.data.shape == routing_map_shape, "routing_map shape mismatch for ",
             (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8 ? "BITMAP_U8" : "BYTEMAP"),
             "; expected ", routing_map_shape, ", got ", routing_map.data.shape);
  if (expert_bias.has_data()) {
    NVTE_CHECK(expert_bias.data.shape == std::vector<size_t>{static_cast<size_t>(num_experts)},
               "expert_bias shape must be [num_experts]=[", num_experts, "], got ",
               expert_bias.data.shape);
  }
#define ROUTER_FORWARD_DISPATCH(RoutingMapFormatVal)                                           \
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                             \
      logits.data.dtype, DataType,                                                             \
      if (expert_bias.has_data()) {                                                            \
        TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                       \
            expert_bias.data.dtype, BiasType,                                                  \
            fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType,         \
                                                                   RoutingMapFormatVal>(       \
                reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk, \
                use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,       \
                reinterpret_cast<BiasType *>(expert_bias.data.dptr),                           \
                reinterpret_cast<DataType *>(probs.data.dptr),                                 \
                reinterpret_cast<uint8_t *>(routing_map.data.dptr),                            \
                reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream););        \
      } else {                                                                                 \
        fused_topk_with_score_function_forward_kernel_launcher<DataType, DataType,             \
                                                               RoutingMapFormatVal>(           \
            reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,     \
            use_pre_softmax, num_groups, group_topk, scaling_factor, score_function, nullptr,  \
            reinterpret_cast<DataType *>(probs.data.dptr),                                     \
            reinterpret_cast<uint8_t *>(routing_map.data.dptr),                                \
            reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream);              \
      });
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    ROUTER_FORWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
  } else {
    ROUTER_FORWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BYTEMAP)
  }
#undef ROUTER_FORWARD_DISPATCH
}

// Backward: grad_probs + intermediate_output + routing_map → grad_logits.
//
// Double-buffered cp.async loads all 3 inputs in original types.  Two-pass
// fused approach (eliminates the comp_buf shmem buffer):
//   Pass 1 (reduction): accumulate warp-level sums needed by normalization/softmax bwd.
//   Pass 2 (element-wise): compute per-element gradient and write to global memory.
//
// Shmem layout (B = num_buffers, W = warps/block):
//   grad_raw:  B × E × W × sizeof(DataType) — async-loaded grad
//   act_buf:   B × E × W × sizeof(CompType) — async-loaded activations
//   mask_buf:  B × E × W × sizeof(bool)     — async-loaded routing mask

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat, int ScoreFunc>
__global__ void fused_topk_with_score_function_backward_kernel(
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    DataType *grad_logits, int num_buffers) {
  constexpr bool kIsBitmap = (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  extern __shared__ char shmem_bwd[];
  char *shmem_ptr = shmem_bwd;

  DataType *grad_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> grad_loader(grad_shmem_base, warp_id, num_experts, num_token_per_block,
                                       num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *act_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> act_loader(act_shmem_base, warp_id, num_experts, num_token_per_block,
                                      num_buffers);
  shmem_ptr += RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  uint8_t *mask_shmem_base = reinterpret_cast<uint8_t *>(shmem_ptr);
  RawAsyncLoader<uint8_t> mask_loader(mask_shmem_base, warp_id, num_experts, num_token_per_block,
                                      num_buffers);

  /***
     * Section: Main Loop — persistent grid with double-buffered async load
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  // Kick off first async load
  {
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      int pos = first_token * num_experts;
      grad_loader.load_current(grad_probs + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
      if constexpr (!kIsBitmap) {
        mask_loader.load_current(routing_map + pos, num_experts, lane_id);
      }
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    if (num_buffers == 1 && round != first_round) {
      grad_loader.load_current(grad_probs + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
      if constexpr (!kIsBitmap) {
        mask_loader.load_current(routing_map + pos, num_experts, lane_id);
      }
    }

    /***
         * Section: Wait for async load + prefetch next round
         */
    grad_loader.wait();
    act_loader.wait();
    if constexpr (!kIsBitmap) {
      mask_loader.wait();
    }

    DataType *raw_grad = grad_loader.current_buf();
    CompType *local_act = act_loader.current_buf();
    uint8_t *local_mask = mask_loader.current_buf();
    if constexpr (kIsBitmap) {
      const int bitmap_row_bytes = (num_experts + 7) / 8;
      const uint8_t *bitmap_row = routing_map + static_cast<size_t>(token_idx) * bitmap_row_bytes;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_mask[i] = (bitmap_row[i / 8] >> (i % 8)) & 1u;
      }
      __syncwarp();
    }

    // Prefetch next round only when double-buffered; single-buffer loads above.
    if (num_buffers > 1) {
      int next_round = round + gridDim.x;
      if (next_round < total_round) {
        int next_token = next_round * num_token_per_block + warp_id;
        if (next_token < num_tokens) {
          int next_pos = next_token * num_experts;
          grad_loader.start_load(grad_probs + next_pos, num_experts, lane_id);
          act_loader.start_load(intermediate_output + next_pos, num_experts, lane_id);
          if constexpr (!kIsBitmap) {
            mask_loader.start_load(routing_map + next_pos, num_experts, lane_id);
          }
        }
      }
    }

    /***
         * Section: Pass 1 — Reduction
         * Accumulate warp-level sums needed by the backward passes:
         *   sigmoid/sqrtsoftplus (topk>1): sum_act, sum_grad_act for normalization bwd
         *   softmax:                       sum_output_x_grad = Σ(grad * softmax_output)
         *
         * For sqrtsoftplus, intermediate_output stores original logits, so we
         * recompute sqrtsoftplus(x) on the fly to get the activation value.
         */
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    bool need_reduce = ((ScoreFunc == 0 || ScoreFunc == 2) && topk > 1) || (ScoreFunc == 1);
    if (need_reduce) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        CompType g = static_cast<CompType>(raw_grad[i]) * scaling_factor;
        CompType act = local_act[i];
        bool routed = local_mask[i];

        if constexpr (ScoreFunc == 0) {  // Sigmoid
          // act = sigmoid output; accumulate over routed experts only
          if (routed) {
            sum_act += act;
            sum_grad_act += g * act;
          }
        } else if constexpr (ScoreFunc == 2) {  // Sqrtsoftplus
          // act = original logit; recompute sqrtsoftplus to get activation
          if (routed) {
            CompType v = sqrtsoftplus_scalar(act);
            sum_act += v;
            sum_grad_act += g * v;
          }
        } else if constexpr (ScoreFunc == 1) {  // Softmax
          if (!use_pre_softmax) {
            // Post-softmax: act = softmax output (routed positions only)
            if (routed) sum_output_x_grad += g * act;
          } else {
            // Pre-softmax: act = softmax output (all experts)
            sum_output_x_grad += (routed ? g : 0.0f) * act;
          }
        }
      }
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        sum_act = warp_allreduce_sum(sum_act);
        sum_grad_act = warp_allreduce_sum(sum_grad_act);
      }
      if constexpr (ScoreFunc == 1) {
        sum_output_x_grad = warp_allreduce_sum(sum_output_x_grad);
      }
    }

    /***
         * Section: Pass 2 — Element-wise gradient
         * Compute per-element gradient using the warp-level sums from Pass 1.
         * Applies backward ops in reverse of forward order:
         *   1. Backward of scaling_factor (multiply grad by scaling_factor)
         *   2. Backward of normalization (sigmoid/sqrtsoftplus with topk > 1)
         *   3. Backward of post-softmax / topk mask
         *   4. Backward of pre-softmax
         *   5. Backward of activation (sigmoid / sqrtsoftplus)
         * Write the grad_logits to the global mem
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(raw_grad[i]) * scaling_factor;
      CompType act = local_act[i];
      bool routed = local_mask[i];

      // Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1 (normalization backward)
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        if (topk > 1) {
          g = normalize_bwd_scalar(g, routed, sum_act, sum_grad_act);
        }
      }

      // Softmax bwd if use_pre_softmax is false (routed subset only)
      if constexpr (ScoreFunc == 1) {
        if (!use_pre_softmax) {
          g = routed ? softmax_bwd_scalar(g, act, sum_output_x_grad) : 0.0f;
        }
      }

      // Backward of topk: mask the unselected position in the grad
      if (!routed) g = 0.0f;

      // Pre-softmax bwd (all experts participate)
      if constexpr (ScoreFunc == 1) {
        if (use_pre_softmax) {
          g = softmax_bwd_scalar(g, act, sum_output_x_grad);
        }
      }

      // Sigmoid bwd: dy/dx = y * (1 - y), where y = sigmoid output
      if constexpr (ScoreFunc == 0) {
        g = sigmoid_bwd_scalar(g, act);
        // Sqrtsoftplus bwd: dy/dx = sigmoid(x) / (2 * y), where x = original logit
      } else if constexpr (ScoreFunc == 2) {
        g = sqrtsoftplus_bwd_scalar(g, act, sqrtsoftplus_scalar(act));
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }

    grad_loader.flip();
    act_loader.flip();
    mask_loader.flip();
  }
}

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat>
void fused_topk_with_score_function_backward_kernel_launcher(
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(static_cast<int64_t>(num_tokens) * num_experts <= INT_MAX,
             "num_tokens * num_experts exceeds INT_MAX (kernel uses int offsets), got ",
             static_cast<int64_t>(num_tokens) * num_experts);
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  size_t single_buf_shmem =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, 1) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, 1) +
      RawAsyncLoader<uint8_t>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(single_buf_shmem, 0);
  size_t shmem_bytes =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers) +
      RawAsyncLoader<uint8_t>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  check_shared_memory_capacity_num_experts(shmem_bytes, num_experts);

  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    size_t grid_size = compute_persistent_grid(kernel, kThreadsPerBlock, shmem_bytes, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shmem_bytes, stream>>>(
        routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
        use_pre_softmax, scaling_factor, grad_logits, num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  switch (score_function) {
    case 0:
      launch(fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat, 0>);
      break;
    case 1:
      launch(fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat, 1>);
      break;
    case 2:
      launch(fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat, 2>);
      break;
    default:
      NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
  }
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             NVTERoutingMapFormat routing_map_format,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, cudaStream_t stream) {
  NVTE_CHECK(num_tokens > 0 && num_experts > 0,
             "num_tokens and num_experts must be positive; got num_tokens=", num_tokens,
             ", num_experts=", num_experts);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  NVTE_CHECK(intermediate_output.data.shape == dense_shape,
             "intermediate_output shape must be [num_tokens, num_experts]=[", num_tokens, ", ",
             num_experts, "], got ", intermediate_output.data.shape);
  NVTE_CHECK(grad_probs.data.shape == dense_shape,
             "grad_probs shape must be [num_tokens, num_experts]=[", num_tokens, ", ", num_experts,
             "], got ", grad_probs.data.shape);
  NVTE_CHECK(grad_logits.data.shape == dense_shape,
             "grad_logits shape must be [num_tokens, num_experts]=[", num_tokens, ", ", num_experts,
             "], got ", grad_logits.data.shape);
  const auto routing_map_shape =
      expected_routing_map_shape(num_tokens, num_experts, routing_map_format);
  NVTE_CHECK(routing_map.data.shape == routing_map_shape, "routing_map shape mismatch for ",
             (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8 ? "BITMAP_U8" : "BYTEMAP"),
             "; expected ", routing_map_shape, ", got ", routing_map.data.shape);
#define ROUTER_BACKWARD_DISPATCH(RoutingMapFormatVal)                                         \
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                            \
      grad_logits.data.dtype, DataType,                                                       \
      fused_topk_with_score_function_backward_kernel_launcher<DataType, RoutingMapFormatVal>( \
          reinterpret_cast<uint8_t *>(routing_map.data.dptr),                                 \
          reinterpret_cast<CompType *>(intermediate_output.data.dptr),                        \
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,  \
          use_pre_softmax, scaling_factor, score_function,                                    \
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    ROUTER_BACKWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
  } else {
    ROUTER_BACKWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BYTEMAP)
  }
#undef ROUTER_BACKWARD_DISPATCH
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_forward_v2(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTERoutingMapFormat routing_map_format, NVTETensor intermediate_output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward_v2);
  using namespace transformer_engine;
  fused_router::fused_topk_with_score_function_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), num_groups, group_topk, scaling_factor, score_function,
      *convertNVTETensorCheck(expert_bias), *convertNVTETensorCheck(probs),
      *convertNVTETensorCheck(routing_map), routing_map_format,
      *convertNVTETensorCheck(intermediate_output), stream);
}

void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward);
  nvte_fused_topk_with_score_function_forward_v2(
      logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
      scaling_factor, score_function, expert_bias, probs, routing_map,
      NVTE_ROUTING_MAP_FORMAT_BYTEMAP, intermediate_output, stream);
}

void nvte_fused_topk_with_score_function_backward_v2(const NVTETensor routing_map,
                                                     NVTERoutingMapFormat routing_map_format,
                                                     const NVTETensor intermediate_output,
                                                     const NVTETensor grad_probs, int num_tokens,
                                                     int num_experts, int topk, int use_pre_softmax,
                                                     float scaling_factor, int score_function,
                                                     NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward_v2);
  using namespace transformer_engine;
  fused_router::fused_topk_with_score_function_backward(
      *convertNVTETensorCheck(routing_map), routing_map_format,
      *convertNVTETensorCheck(intermediate_output), *convertNVTETensorCheck(grad_probs), num_tokens,
      num_experts, topk, static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *convertNVTETensorCheck(grad_logits), stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  nvte_fused_topk_with_score_function_backward_v2(
      routing_map, NVTE_ROUTING_MAP_FORMAT_BYTEMAP, intermediate_output, grad_probs, num_tokens,
      num_experts, topk, use_pre_softmax, scaling_factor, score_function, grad_logits, stream);
}
