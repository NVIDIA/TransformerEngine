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
#include "../utils.cuh"
#include "async_loader.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

// =============================================================================
// Simple aux_loss forward kernel — exact upstream structure (no async loader,
// no persistent grid, runtime score_function dispatch).
// =============================================================================

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat,
          TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_score_for_moe_aux_loss_forward_simple_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    float *scores, uint8_t *routing_map, CompType *intermediate_output) {
  constexpr bool kIsBitmap = (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_scores_for_aux_loss[];
  CompType *logits_buf = reinterpret_cast<CompType *>(shmem_scores_for_aux_loss);
  CompType *topk_logits_buf =
      reinterpret_cast<CompType *>(logits_buf + num_experts * num_token_per_block);
  int *topk_indices_buf = reinterpret_cast<int *>(topk_logits_buf + topk * num_token_per_block);
  const int bitmap_words_per_warp = (num_experts + 31) / 32;
  const int bitmap_row_bytes = (num_experts + 7) / 8;
  uint32_t *bitmap_words_buf = nullptr;
  if constexpr (kIsBitmap) {
    bitmap_words_buf = reinterpret_cast<uint32_t *>(topk_indices_buf + topk * num_token_per_block);
  }
  CompType *local_logits = logits_buf + warp_id * num_experts;
  CompType *topk_logits = topk_logits_buf + warp_id * topk;
  int *topk_indices = topk_indices_buf + warp_id * topk;
  uint32_t *local_bitmap_words =
      (bitmap_words_buf != nullptr) ? bitmap_words_buf + warp_id * bitmap_words_per_warp : nullptr;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) break;

    int pos_offset = token_offset_cur_warp * num_experts;
    // Clear the routing_map
    if constexpr (!kIsBitmap) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        routing_map[pos_offset + i] = 0;
      }
    } else {
      for (int i = lane_id; i < bitmap_words_per_warp; i += kThreadsPerWarp) {
        local_bitmap_words[i] = 0u;
      }
    }
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (score_function == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
      }
    }
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_logits[i] = static_cast<CompType>(logits[pos_offset + i]);
    }
    __threadfence_block();
    __syncwarp();

    // Preprocess: apply score function
    if (score_function == 1) {
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 0) {
      apply_sigmoid_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 2) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(local_logits, num_experts, lane_id);
    }

    __syncwarp();

    // Sigmoid/Sqrtsoftplus post-processing: normalize
    if (score_function == 0 || score_function == 2) {
      auto sum_logits =
          warp_reduce_on_shmem<CompType, ReduceFuncType::SUM>(local_logits, num_experts, lane_id);
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] /= (sum_logits + epsilon);
      }
      __syncwarp();
    }

    // Topk
    topk_and_mask<TopkFunc>(local_logits, num_experts, topk, topk_indices, topk_logits, lane_id);
    __syncwarp();

    // Write outputs
    if constexpr (!kIsBitmap) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        routing_map[pos_offset + topk_indices[i]] = 1;
      }
    } else {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        int e = topk_indices[i];
        atomicOr(&local_bitmap_words[e / 32], 1u << (e % 32));
      }
      __syncwarp();
      uint8_t *bitmap_row =
          routing_map + static_cast<size_t>(token_offset_cur_warp) * bitmap_row_bytes;
      const uint8_t *local_bitmap_bytes = reinterpret_cast<const uint8_t *>(local_bitmap_words);
      for (int i = lane_id; i < bitmap_row_bytes; i += kThreadsPerWarp) {
        bitmap_row[i] = local_bitmap_bytes[i];
      }
    }
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[pos_offset + i] = local_logits[i];
    }
    __threadfence_block();
    __syncwarp();
  }
}

// =============================================================================
// Optimized aux_loss forward kernel — async loader, persistent grid.
// =============================================================================

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat,
          TopkFuncType TopkFunc = TopkFuncType::Naive, int ScoreFunc = 0>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType *logits, int num_tokens,
                                                            int num_experts, int topk,
                                                            float *scores, uint8_t *routing_map,
                                                            CompType *intermediate_output,
                                                            int num_buffers) {
  constexpr bool kIsBitmap = (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8);
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ char shmem_raw_aux[];

  // Shmem layout: logits_raw (async) | logits_work | topk_scratch
  char *shmem_ptr = shmem_raw_aux;
  DataType *logits_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> loader(logits_shmem_base, warp_id, num_experts, num_token_per_block,
                                  num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *logits_work_buf = reinterpret_cast<CompType *>(shmem_ptr);
  shmem_ptr += num_experts * num_token_per_block * sizeof(CompType);

  CompType *topk_logits_buf = reinterpret_cast<CompType *>(shmem_ptr);
  int *topk_indices_buf = reinterpret_cast<int *>(topk_logits_buf + topk * num_token_per_block);
  const int bitmap_words_per_warp = (num_experts + 31) / 32;
  const int bitmap_row_bytes = (num_experts + 7) / 8;
  uint32_t *bitmap_words_buf = nullptr;
  if constexpr (kIsBitmap) {
    bitmap_words_buf = reinterpret_cast<uint32_t *>(topk_indices_buf + topk * num_token_per_block);
  }

  // The address of buffers on the current warp
  CompType *local_logits = logits_work_buf + warp_id * num_experts;
  CompType *topk_logits = topk_logits_buf + warp_id * topk;
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

    loader.wait();
    DataType *raw_logits = loader.current_buf();

    // Prefetch next round (only when double-buffered)
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
         * - Convert raw logits (DataType) → apply score function → save intermediate_output
         *
         * Fused into a single loop per score function where possible:
         *   score_function == 0 (sigmoid):      convert, sigmoid, save → shmem
         *   score_function == 1 (softmax):      convert → shmem, softmax (multi-pass), save
         *   score_function == 2 (sqrtsoftplus): convert, save logits, sqrtsoftplus → shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    if constexpr (kIsBitmap) {
      for (int i = lane_id; i < bitmap_words_per_warp; i += kThreadsPerWarp) {
        local_bitmap_words[i] = 0u;
      }
    }

    if constexpr (ScoreFunc == 1) {  // Softmax
      // Apply softmax to all logits, save softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] = static_cast<CompType>(raw_logits[i]);
      }
      __syncwarp();
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if constexpr (ScoreFunc == 0) {  // Sigmoid
      // Fused: convert → sigmoid → save sigmoid output for backward → shmem
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(raw_logits[i]));
        intermediate_output[pos_offset + i] = val;  // Save sigmoid output for backward
        local_logits[i] = val;
      }
    } else if constexpr (ScoreFunc == 2) {  // Sqrtsoftplus
      // Fused: convert → save original logit for backward → sqrtsoftplus → shmem
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(raw_logits[i]);
        intermediate_output[pos_offset + i] = logit;  // Save original logits for backward
        local_logits[i] = sqrtsoftplus_scalar(logit);
      }
    }
    __syncwarp();

    // Sigmoid/Sqrtsoftplus post-processing: normalize scores to sum to 1
    if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
      auto sum_logits =
          warp_reduce_on_shmem<CompType, ReduceFuncType::SUM>(local_logits, num_experts, lane_id);
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] /= (sum_logits + epsilon);
      }
      __syncwarp();
    }

    /***
         * Section: Topk
         * Get the topk indices
         */
    topk_and_mask<TopkFunc>(local_logits, num_experts, topk, topk_indices, topk_logits, lane_id);
    __syncwarp();

    // Write the routing_map to the output tensor
    if constexpr (!kIsBitmap) {
      vec_fill_global(routing_map + pos_offset, static_cast<uint8_t>(0), num_experts, lane_id);
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        routing_map[pos_offset + topk_indices[i]] = 1;
      }
    } else {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        int e = topk_indices[i];
        atomicOr(&local_bitmap_words[e / 32], 1u << (e % 32));
      }
      __syncwarp();
      uint8_t *bitmap_row =
          routing_map + static_cast<size_t>(token_offset_cur_warp) * bitmap_row_bytes;
      const uint8_t *local_bitmap_bytes = reinterpret_cast<const uint8_t *>(local_bitmap_words);
      for (int i = lane_id; i < bitmap_row_bytes; i += kThreadsPerWarp) {
        bitmap_row[i] = local_bitmap_bytes[i];
      }
    }
    // Write the scores to the output tensor
    vec_store_global(scores + pos_offset, local_logits, num_experts, lane_id);
    __syncwarp();

    loader.flip();
  }
}

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat>
void fused_score_for_moe_aux_loss_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    float *scores, uint8_t *routing_map, CompType *intermediate_output, cudaStream_t stream) {
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(topk > 0 && topk <= num_experts, "topk must be in [1, num_experts], got topk=", topk,
             " num_experts=", num_experts);
  NVTE_CHECK(static_cast<int64_t>(num_tokens) * num_experts <= INT_MAX,
             "num_tokens * num_experts exceeds INT_MAX (kernel uses int offsets), got ",
             static_cast<int64_t>(num_tokens) * num_experts);
  NVTE_CHECK(score_function >= 0 && score_function <= 2,
             "Unsupported score_function: ", score_function);
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  size_t scores_shmem = num_experts * num_token_per_block * sizeof(CompType);
  size_t scratch_shmem =
      topk * num_token_per_block * sizeof(CompType) + topk * num_token_per_block * sizeof(int);
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
        logits, num_tokens, num_experts, topk, scores, routing_map, intermediate_output,
        num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  // Dispatch: use radix only when it is profitable and supported. Otherwise use the
  // naive path, which handles very large expert counts without the radix histogram limit.
  const bool use_radix = topk >= get_radix_topk_threshold() && num_experts <= kMaxExpertsRadixTopk;
  if (!use_radix) {
    // Simple path: exact upstream structure — no async loader, no persistent grid.
    check_shared_memory_capacity_num_experts(other_shmem, num_experts);

    auto launch_simple = [&](auto kernel) {
      NVTE_CHECK_CUDA(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, other_shmem));
      kernel<<<total_blocks, kThreadsPerBlock, other_shmem, stream>>>(
          logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
          intermediate_output);
      NVTE_CHECK_CUDA(cudaGetLastError());
    };

    launch_simple(fused_score_for_moe_aux_loss_forward_simple_kernel<DataType, RoutingMapFormat,
                                                                     TopkFuncType::Naive>);
  } else {
    // Optimized path: async loader + persistent grid + radix topk.
    switch (score_function) {
      case 0:
        launch(fused_score_for_moe_aux_loss_forward_kernel<DataType, RoutingMapFormat,
                                                           TopkFuncType::Radix, 0>);
        break;
      case 1:
        launch(fused_score_for_moe_aux_loss_forward_kernel<DataType, RoutingMapFormat,
                                                           TopkFuncType::Radix, 1>);
        break;
      case 2:
        launch(fused_score_for_moe_aux_loss_forward_kernel<DataType, RoutingMapFormat,
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

void fused_score_for_moe_aux_loss_forward(const Tensor &logits, int num_tokens, int num_experts,
                                          int topk, int score_function, Tensor &scores,
                                          Tensor &routing_map,
                                          NVTERoutingMapFormat routing_map_format,
                                          Tensor &intermediate_output, cudaStream_t stream) {
  NVTE_CHECK(num_tokens > 0 && num_experts > 0,
             "num_tokens and num_experts must be positive; got num_tokens=", num_tokens,
             ", num_experts=", num_experts);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  NVTE_CHECK(logits.data.shape == dense_shape, "logits shape must be [num_tokens, num_experts]=[",
             num_tokens, ", ", num_experts, "], got ", logits.data.shape);
  NVTE_CHECK(scores.data.shape == dense_shape, "scores shape must be [num_tokens, num_experts]=[",
             num_tokens, ", ", num_experts, "], got ", scores.data.shape);
  NVTE_CHECK(intermediate_output.data.shape == dense_shape,
             "intermediate_output shape must be [num_tokens, num_experts]=[", num_tokens, ", ",
             num_experts, "], got ", intermediate_output.data.shape);
  const auto routing_map_shape =
      expected_routing_map_shape(num_tokens, num_experts, routing_map_format);
  NVTE_CHECK(routing_map.data.shape == routing_map_shape, "routing_map shape mismatch for ",
             (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8 ? "BITMAP_U8" : "BYTEMAP"),
             "; expected ", routing_map_shape, ", got ", routing_map.data.shape);
#define AUX_LOSS_FORWARD_DISPATCH(RoutingMapFormatVal)                                     \
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                         \
      logits.data.dtype, DataType,                                                         \
      fused_score_for_moe_aux_loss_forward_kernel_launcher<DataType, RoutingMapFormatVal>( \
          reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,   \
          score_function, reinterpret_cast<float *>(scores.data.dptr),                     \
          reinterpret_cast<uint8_t *>(routing_map.data.dptr),                              \
          reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream););
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    AUX_LOSS_FORWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
  } else {
    AUX_LOSS_FORWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BYTEMAP)
  }
#undef AUX_LOSS_FORWARD_DISPATCH
}

// Backward: grad_scores + intermediate_output → grad_logits.
// No routing_map — all experts participate (unlike topk backward).
// Double-buffered cp.async loads both inputs.  Two-pass fused approach.
//
// Shmem layout (B = num_buffers, W = warps/block):
//   grad_buf: B × E × W × sizeof(CompType)   — async-loaded grad
//   act_buf:  B × E × W × sizeof(CompType)   — async-loaded activations

template <typename DataType, int ScoreFunc>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const CompType *intermediate_output,
                                                             const float *grad_scores,
                                                             int num_tokens, int num_experts,
                                                             DataType *grad_logits,
                                                             int num_buffers) {
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  extern __shared__ char shmem_aux_bwd[];
  char *shmem_ptr = shmem_aux_bwd;

  CompType *grad_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> grad_loader(grad_shmem_base, warp_id, num_experts, num_token_per_block,
                                       num_buffers);
  shmem_ptr += RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *act_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> act_loader(act_shmem_base, warp_id, num_experts, num_token_per_block,
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
      grad_loader.load_current(grad_scores + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    if (num_buffers == 1 && round != first_round) {
      grad_loader.load_current(grad_scores + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
    }

    grad_loader.wait();
    act_loader.wait();

    CompType *raw_grad = grad_loader.current_buf();
    CompType *raw_act = act_loader.current_buf();

    // Prefetch next round only when double-buffered; single-buffer loads above.
    if (num_buffers > 1) {
      int next_round = round + gridDim.x;
      if (next_round < total_round) {
        int next_token = next_round * num_token_per_block + warp_id;
        if (next_token < num_tokens) {
          int next_pos = next_token * num_experts;
          grad_loader.start_load(grad_scores + next_pos, num_experts, lane_id);
          act_loader.start_load(intermediate_output + next_pos, num_experts, lane_id);
        }
      }
    }

    /***
         * Section: Pass 1 — Reduction
         * Accumulate warp-level sums needed by the backward passes:
         *   sigmoid/sqrtsoftplus: sum_act, sum_grad_act for normalization bwd
         *   softmax:              sum_output_x_grad = Σ(grad * softmax_output)
         *
         * For sqrtsoftplus, intermediate_output stores original logits, so we
         * recompute sqrtsoftplus(x) on the fly to get the activation value.
         */
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(raw_grad[i]);
      CompType act = raw_act[i];
      if constexpr (ScoreFunc == 0) {  // Sigmoid
        // act = sigmoid output; accumulate over all experts
        sum_act += act;
        sum_grad_act += g * act;
      } else if constexpr (ScoreFunc == 2) {  // Sqrtsoftplus
        // act = original logit; recompute sqrtsoftplus to get activation
        CompType v = sqrtsoftplus_scalar(act);
        sum_act += v;
        sum_grad_act += g * v;
      } else if constexpr (ScoreFunc == 1) {  // Softmax
        // act = softmax output
        sum_output_x_grad += g * act;
      }
    }
    if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
      sum_act = warp_allreduce_sum(sum_act);
      sum_grad_act = warp_allreduce_sum(sum_grad_act);
    }
    if constexpr (ScoreFunc == 1) {
      sum_output_x_grad = warp_allreduce_sum(sum_output_x_grad);
    }

    /***
         * Section: Pass 2 — Element-wise gradient
         * Compute per-element gradient using the warp-level sums from Pass 1.
         * Applies backward ops in reverse of forward order:
         *   sigmoid:      normalization bwd → sigmoid bwd
         *   sqrtsoftplus: normalization bwd → sqrtsoftplus bwd
         *   softmax:      softmax bwd
         * Write the grad_logits to the global mem
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(raw_grad[i]);
      CompType act = raw_act[i];

      if constexpr (ScoreFunc == 0) {  // Sigmoid bwd
        g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
        g = sigmoid_bwd_scalar(g, act);
      } else if constexpr (ScoreFunc == 2) {  // Sqrtsoftplus bwd
        g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
        g = sqrtsoftplus_bwd_scalar(g, act, sqrtsoftplus_scalar(act));
      } else if constexpr (ScoreFunc == 1) {  // Softmax bwd
        g = softmax_bwd_scalar(g, act, sum_output_x_grad);
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }

    grad_loader.flip();
    act_loader.flip();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const CompType *intermediate_output, const float *grad_scores, int num_tokens, int num_experts,
    int topk, int score_function, DataType *grad_logits, cudaStream_t stream) {
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(static_cast<int64_t>(num_tokens) * num_experts <= INT_MAX,
             "num_tokens * num_experts exceeds INT_MAX (kernel uses int offsets), got ",
             static_cast<int64_t>(num_tokens) * num_experts);
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  size_t single_buf_shmem =
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, 1) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(single_buf_shmem, 0);
  size_t shmem_bytes =
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  check_shared_memory_capacity_num_experts(shmem_bytes, num_experts);

  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    size_t grid_size = compute_persistent_grid(kernel, kThreadsPerBlock, shmem_bytes, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shmem_bytes, stream>>>(
        intermediate_output, grad_scores, num_tokens, num_experts, grad_logits, num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  switch (score_function) {
    case 0:
      launch(fused_score_for_moe_aux_loss_backward_kernel<DataType, 0>);
      break;
    case 1:
      launch(fused_score_for_moe_aux_loss_backward_kernel<DataType, 1>);
      break;
    case 2:
      launch(fused_score_for_moe_aux_loss_backward_kernel<DataType, 2>);
      break;
    default:
      NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
  }
}

void fused_score_for_moe_aux_loss_backward(const Tensor &intermediate_output,
                                           const Tensor &grad_scores, int num_tokens,
                                           int num_experts, int topk, int score_function,
                                           Tensor &grad_logits, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_backward_kernel_launcher<DataType>(
          reinterpret_cast<CompType *>(intermediate_output.data.dptr),
          reinterpret_cast<float *>(grad_scores.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_score_for_moe_aux_loss_forward_v2(const NVTETensor logits, int num_tokens,
                                                  int num_experts, int topk, int score_function,
                                                  NVTETensor scores, NVTETensor routing_map,
                                                  NVTERoutingMapFormat routing_map_format,
                                                  const NVTETensor intermediate_output,
                                                  cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward_v2);
  using namespace transformer_engine;
  fused_router::fused_score_for_moe_aux_loss_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk, score_function,
      *convertNVTETensorCheck(scores), *convertNVTETensorCheck(routing_map), routing_map_format,
      *convertNVTETensorCheck(intermediate_output), stream);
}

void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward);
  nvte_fused_score_for_moe_aux_loss_forward_v2(
      logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
      NVTE_ROUTING_MAP_FORMAT_BYTEMAP, intermediate_output, stream);
}

void nvte_fused_score_for_moe_aux_loss_backward(const NVTETensor intermediate_output,
                                                const NVTETensor grad_scores, int num_tokens,
                                                int num_experts, int topk, int score_function,
                                                NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_router::fused_score_for_moe_aux_loss_backward(
      *convertNVTETensorCheck(intermediate_output), *convertNVTETensorCheck(grad_scores),
      num_tokens, num_experts, topk, score_function, *convertNVTETensorCheck(grad_logits), stream);
}
