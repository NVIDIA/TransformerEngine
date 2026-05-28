/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

template <typename DataType, typename BiasType, NVTERoutingMapFormat RoutingMapFormat,
          TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, uint8_t *routing_map,
    CompType *intermediate_output) {
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
  // Per-warp bitmap accumulator (BITMAP_U8 only). uint32 packing is bit-for-bit
  // equivalent to uint8 LSB-first on little-endian devices (CUDA is always LE).
  const int bitmap_words_per_warp = (num_experts + 31) / 32;
  const int bitmap_row_bytes = (num_experts + 7) / 8;
  uint32_t *bitmap_words_buf = nullptr;
  if constexpr (kIsBitmap) {
    bitmap_words_buf = reinterpret_cast<uint32_t *>(topk_indices_buf + topk * num_token_per_block);
  }
  // The address of buffers on the current warp
  CompType *scores = scores_buf + warp_id * num_experts;
  CompType *topk_scores = topk_scores_buf + warp_id * topk;
  CompType *masked_scores = masked_scores_buf + warp_id * num_experts;
  CompType *group_scores = group_scores_buf + warp_id * num_groups;
  int *topk_indices = topk_indices_buf + warp_id * topk;
  uint32_t *local_bitmap_words =
      (bitmap_words_buf != nullptr) ? bitmap_words_buf + warp_id * bitmap_words_per_warp : nullptr;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the logits to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // BITMAP_U8 accumulates the row in shmem and writes it wholesale at the end
    // of the loop; only BYTEMAP needs a global clear here.
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
      for (int j = lane_id; j < bitmap_words_per_warp; j += kThreadsPerWarp) {
        local_bitmap_words[j] = 0u;
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

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Sqrtsoftplus
         * - Expert bias
         * This is in-place scores update
         */
    if (use_pre_softmax && score_function == 1) {  // score_function == 1 means softmax
      // Apply softmax to the logits before the topk
      apply_softmax_on_float(scores, num_experts, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 0) {  // score_function == 0 means sigmoid
      // Apply sigmoid to the logits
      apply_sigmoid_on_float(scores, num_experts, lane_id);
      __syncwarp();
      // Save the sigmoid output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 2) {  // score_function == 2 means sqrtsoftplus
      // First save the original logits for backward (needed for sqrtsoftplus gradient computation)
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];  // Save original logits
      }
      __syncwarp();
      // Apply sqrtsoftplus to the logits
      apply_sqrtsoftplus_on_float(scores, num_experts, lane_id);
    }

    __syncwarp();  //Confirm the scores is written to the output

    // Expert bias is only used at the sigmoid/sqrtsoftplus case
    if (expert_bias && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[i] += static_cast<CompType>(expert_bias[i]);
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
    if (expert_bias && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        topk_scores[i] = topk_scores[i] - static_cast<CompType>(expert_bias[topk_indices[i]]);
      }
      __syncwarp();
    }

    // score_function == 1 means softmax
    if (!use_pre_softmax && score_function == 1) {
      // Apply softmax to the topk logits
      apply_softmax_on_float(topk_scores, topk, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
      }
      __syncwarp();
    }

    // Sigmoid/Sqrtsoftplus post-processing when topk > 1
    if (score_function == 0 || score_function == 2) {
      if (topk > 1) {
        CompType sum_scores = warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id);
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
      // shmem atomicOr handles same-word collisions across the topk lanes.
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        int e = topk_indices[i];
        atomicOr(&local_bitmap_words[e / 32], 1u << (e % 32));
        probs[pos_offset + e] = scaling_factor * topk_scores[i];
      }
      __syncwarp();
      uint8_t *bitmap_row =
          routing_map + static_cast<size_t>(token_offset_cur_warp) * bitmap_row_bytes;
      const uint8_t *local_bitmap_bytes = reinterpret_cast<const uint8_t *>(local_bitmap_words);
      for (int j = lane_id; j < bitmap_row_bytes; j += kThreadsPerWarp) {
        bitmap_row[j] = local_bitmap_bytes[j];
      }
    }
    __threadfence_block();
    __syncwarp();
  }
}

template <typename DataType, typename BiasType, NVTERoutingMapFormat RoutingMapFormat>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, uint8_t *routing_map,
    CompType *intermediate_output, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType)  // scores
                              + topk * num_token_per_block * sizeof(CompType)       // topk_scores
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  if (group_topk > 0) {
    shared_memory_size += num_groups * num_token_per_block * sizeof(CompType);   // group_scores
    shared_memory_size += num_experts * num_token_per_block * sizeof(CompType);  // maksed_scores
  }
  if constexpr (RoutingMapFormat == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    size_t bitmap_words_per_warp = (num_experts + 31) / 32;
    shared_memory_size +=
        bitmap_words_per_warp * num_token_per_block * sizeof(uint32_t);  // bitmap accumulator
  }
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  // Radix selection is O(E), independent of K, but it needs 4 passes for 32-bit float;
  // switch at K=16 where naive O(K^2*E) starts to dominate
  if (topk < 16) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                      TopkFuncType::Naive>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                  TopkFuncType::Naive>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
            scaling_factor, score_function, expert_bias, probs, routing_map, intermediate_output);
  } else {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                      TopkFuncType::Radix>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    fused_topk_with_score_function_forward_kernel<DataType, BiasType, RoutingMapFormat,
                                                  TopkFuncType::Radix>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
            scaling_factor, score_function, expert_bias, probs, routing_map, intermediate_output);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
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
#define ROUTER_FORWARD_DISPATCH(RoutingMapFormatVal)                                              \
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                                \
      logits.data.dtype, DataType,                                                                \
      if (expert_bias.has_data()) {                                                               \
        TE_ROUTER_PROBS_TYPE_SWITCH_ALL(                                                          \
            expert_bias.data.dtype, BiasType,                                                     \
            fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType,            \
                                                                   RoutingMapFormatVal>(          \
                reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,    \
                use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,          \
                reinterpret_cast<BiasType *>(expert_bias.data.dptr),                              \
                reinterpret_cast<DataType *>(probs.data.dptr),                                    \
                reinterpret_cast<uint8_t *>(routing_map.data.dptr),                               \
                reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream););           \
      } else {                                                                                    \
        fused_topk_with_score_function_forward_kernel_launcher<DataType, DataType,                \
                                                               RoutingMapFormatVal>(              \
            reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,        \
            use_pre_softmax, num_groups, group_topk, scaling_factor, score_function, nullptr,     \
            reinterpret_cast<DataType *>(probs.data.dptr),                                        \
            reinterpret_cast<uint8_t *>(routing_map.data.dptr),                                   \
            reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream);                 \
      });
  if (routing_map_format == NVTE_ROUTING_MAP_FORMAT_BITMAP_U8) {
    ROUTER_FORWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BITMAP_U8)
  } else {
    ROUTER_FORWARD_DISPATCH(NVTE_ROUTING_MAP_FORMAT_BYTEMAP)
  }
#undef ROUTER_FORWARD_DISPATCH
}

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat>
__global__ void fused_topk_with_score_function_backward_kernel(
    // Inputs tensor
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    // Other parameters
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function,
    // Output tensor
    DataType *grad_logits) {
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
  extern __shared__ float shmem[];
  CompType *grad_probs_buf = reinterpret_cast<CompType *>(shmem);
  // To store the output of softmax/sigmoid from fwd, or original logits for sqrtsoftplus
  CompType *act_from_fwd_buf = grad_probs_buf + num_experts * num_token_per_block;
  CompType *comp_buf = act_from_fwd_buf + num_experts * num_token_per_block;
  // To store the routing_map from the fwd
  bool *routing_map_buf = reinterpret_cast<bool *>(comp_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  CompType *local_grad = grad_probs_buf + warp_id * num_experts;
  CompType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  CompType *local_comp_buf = comp_buf + warp_id * num_experts;
  bool *local_routing_map = routing_map_buf + warp_id * num_experts;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the dgrad/output_from_fwd to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the dgrad/output_from_fwd to shmem. The routing_map source layout
    // depends on the RoutingMapFormat template parameter (see NVTERoutingMapFormat).
    const int bitmap_row_bytes = (num_experts + 7) / 8;
    const uint8_t *bitmap_row =
        routing_map + static_cast<size_t>(token_offset_cur_warp) * bitmap_row_bytes;
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_probs[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      if constexpr (!kIsBitmap) {
        local_routing_map[i] = routing_map[pos_offset + i] != 0;
      } else {
        local_routing_map[i] = (bitmap_row[i / 8] >> (i % 8)) & 1u;
      }
    }
    __threadfence_block();
    __syncwarp();

    /***
         * Section: Backward of ops after the topk
         * - Backward of the used scaling_factor
         * - Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1
         * - Softmax bwd if use_pre_softmax is false
         */
    // Backward of the used scaling_factor
    // In-place update
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = local_grad[i] * scaling_factor;
      }
    }
    __syncwarp();

    // Sqrtsoftplus: First compute sqrtsoftplus output from original logits
    // (needed for both post-processing bwd and activation bwd, compute once here)
    // For sqrtsoftplus, intermediate_output stores original logits
    if (score_function == 2) {
      // Copy original logits to local_comp_buf and apply sqrtsoftplus in-place
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = local_act_from_fwd[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(local_comp_buf, num_experts, lane_id);
      __syncwarp();
    }

    // Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1 (normalization backward)
    if (topk > 1 && (score_function == 0 || score_function == 2)) {
      // Select the correct activation output buffer:
      // - Sigmoid: local_act_from_fwd already contains sigmoid output
      // - Sqrtsoftplus: local_comp_buf contains sqrtsoftplus output computed above
      CompType *act_output = (score_function == 0) ? local_act_from_fwd : local_comp_buf;

      CompType sum_fwd_input = masked_warp_reduce_on_shmem(
          /*data ptr = */ act_output,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // Compute sum of output * grad using registers
      CompType local_sum_Output_x_Grad = 0.0;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_sum_Output_x_Grad += local_grad[i] * act_output[i];
        }
      }
      CompType sum_Output_x_Grad = warp_reduce_sum_float(local_sum_Output_x_Grad);
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] =
              local_grad[i] / (sum_fwd_input + epsilon) -
              sum_Output_x_Grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
        } else {
          local_grad[i] = 0.0;
        }
      }
      __syncwarp();
    }

    // Softmax bwd if use_pre_softmax is false
    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, local_routing_map,
                                 num_experts, lane_id);
      __syncwarp();
    }

    /***
         * Section: Backward of topk
         * mask the unselected position in the grad
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (!local_routing_map[i]) {
        local_grad[i] = 0.0;
      }
    }
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid bwd
         * - Sqrtsoftplus bwd
         * - Write the grad_logits to the global mem
         */
    // Pre-softmax bwd
    if (score_function == 1 && use_pre_softmax) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, nullptr,
                                 num_experts, lane_id);
      __syncwarp();
    }
    // Sigmoid bwd
    if (score_function == 0) {
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
      __syncwarp();
    }
    // Sqrtsoftplus bwd
    // For sqrtsoftplus, local_comp_buf already contains sqrtsoftplus output computed earlier
    // Now compute gradient: dy/dx = sigmoid(x) / (2 * y)
    if (score_function == 2) {
      apply_sqrtsoftplus_bwd_on_float(local_grad, local_comp_buf, local_act_from_fwd, num_experts,
                                      lane_id);
      __syncwarp();
    }
    // Write the grad_logits to the global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = local_grad[i];
    }
    __syncwarp();
  }
}

template <typename DataType, NVTERoutingMapFormat RoutingMapFormat>
void fused_topk_with_score_function_backward_kernel_launcher(
    const uint8_t *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType)  // grad_probs
                              +
                              num_experts * num_token_per_block * sizeof(CompType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(CompType)  // comp_buf
                              + num_experts * num_token_per_block * sizeof(bool);     // routing_map
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(
      fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  fused_topk_with_score_function_backward_kernel<DataType, RoutingMapFormat>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function, grad_logits);
  NVTE_CHECK_CUDA(cudaGetLastError());
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

// Deprecated V1 entry point: forwards to the V2 above with the BYTEMAP layout.
// Kept for ABI compatibility with external C API consumers.
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

// Deprecated V1 entry point: forwards to the V2 above with the BYTEMAP layout.
// Kept for ABI compatibility with external C API consumers.
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
