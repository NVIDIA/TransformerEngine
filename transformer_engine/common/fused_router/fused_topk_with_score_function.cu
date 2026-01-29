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
#include "../utils.cuh"
#include "utils.h"

namespace transformer_engine {

template <typename DataType, typename BiasType>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, bool *routing_map,
    DataType *intermediate_output) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     *   So DataType address is assigned firstly to avoid the alignment issue
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  DataType *scores_buf = reinterpret_cast<DataType *>(shmem);
  DataType *topk_scores_buf =
      reinterpret_cast<DataType *>(scores_buf + num_experts * num_token_per_block);
  DataType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = reinterpret_cast<DataType *>(topk_scores_buf + topk * num_token_per_block);
    group_scores_buf =
        reinterpret_cast<DataType *>(masked_scores_buf + num_experts * num_token_per_block);
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }
  // The address of buffers on the current warp
  DataType *scores = scores_buf + warp_id * num_experts;
  DataType *topk_scores = topk_scores_buf + warp_id * topk;
  DataType *masked_scores = masked_scores_buf + warp_id * num_experts;
  DataType *group_scores = group_scores_buf + warp_id * num_groups;
  int *topk_indices = topk_indices_buf + warp_id * topk;

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
    // Clear the probs/routing_map (num_experts)
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      probs[pos_offset + i] = 0.0f;
      routing_map[pos_offset + i] = false;
      if (score_function == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<DataType>::infinity();
      }
    }
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[i] = logits[pos_offset + i];
    }
    // If group_topk > 0, init the masked_scores to -inf
    if (group_topk > 0) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<DataType>::infinity();
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
      // First save the original logits for backward (needed for sigmoid computation)
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
        scores[i] = static_cast<DataType>(static_cast<double>(scores[i]) +
                                          static_cast<double>(expert_bias[i]));
      }
    }
    __syncwarp();

    /***
         * Section: Topk
         * Get the topk indices
         * - group_topk
         * - naive topk
         * - topk with expert bias
         */
    // Topk on the scores
    // The bias is not empty only happens at the sigmod case
    if (group_topk > 0) {
      int group_size = num_experts / num_groups;
      // Top2
      for (int i = 0; i < num_groups; i++) {
        naive_topk_and_mask(
            /*scores ptr = */ scores + i * group_size,
            /*data size = */ group_size,
            /*topk = */ topk / group_topk,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
        __syncwarp();
        // Compute the group score
        if (lane_id == 0) {
          DataType tmp = 0.0f;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp = tmp + topk_scores[j];
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }

      // select the topk groups
      naive_topk_and_mask(
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
      naive_topk_and_mask(masked_scores, num_experts, topk, topk_indices, topk_scores, lane_id);

    } else {
      naive_topk_and_mask(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }
    __syncwarp();

    /***
         * Section: Postprocess
         * Possible postprocess the scores after the topk operation
         * - Revert Expert bias
         * - Softmax
         * - Sigmoid post-processing when topk > 1
         * - Write the result with scaling_factor
         */
    // Revert Expert bias from the topk scores
    if (expert_bias && score_function == 0) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        topk_scores[i] =
            static_cast<double>(topk_scores[i]) - static_cast<double>(expert_bias[topk_indices[i]]);
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
        double sum_scores = warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = static_cast<double>(topk_scores[i]) / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    // score_function == 2 means sqrtsoftplus
    if (score_function == 2) {
      if (topk > 1) {
        double sum_scores = warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = static_cast<double>(topk_scores[i]) / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    // Write the probs/routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] = scaling_factor * static_cast<double>(topk_scores[i]);
    }
    __threadfence_block();
    __syncwarp();
  }
}

template <typename DataType, typename BiasType>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, bool *routing_map, DataType *intermediate_output,
    cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // scores
                              + topk * num_token_per_block * sizeof(DataType)       // topk_scores
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  if (group_topk > 0) {
    shared_memory_size += num_groups * num_token_per_block * sizeof(DataType);   // group_scores
    shared_memory_size += num_experts * num_token_per_block * sizeof(DataType);  // maksed_scores
  }
  fused_topk_with_score_function_forward_kernel<DataType, BiasType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
          scaling_factor, score_function, expert_bias, probs, routing_map, intermediate_output);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_topk_with_score_function_forward(const Tensor logits, int num_tokens, int num_experts,
                                            int topk, bool use_pre_softmax, int num_groups,
                                            int group_topk, float scaling_factor,
                                            int score_function, const Tensor expert_bias,
                                            Tensor probs, Tensor routing_map,
                                            Tensor intermediate_output, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
          expert_bias.data.dtype, BiasType,
          fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType>(
              reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
              use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
              reinterpret_cast<BiasType *>(expert_bias.data.dptr),
              reinterpret_cast<DataType *>(probs.data.dptr),
              reinterpret_cast<bool *>(routing_map.data.dptr),
              reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream);););
}

template <typename DataType>
__global__ void fused_topk_with_score_function_backward_kernel(
    // Inputs tensor
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    // Other parameters
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function,
    // Output tensor
    DataType *grad_logits) {
  /***
     * Section: Global Variables/Addresses init
     * - Assume the sizeof(DataType) >= sizeof(int),
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  DataType *grad_probs_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_probs_buf + num_experts * num_token_per_block);
  DataType *comp_buf =
      reinterpret_cast<DataType *>(act_from_fwd_buf + num_experts * num_token_per_block);
  // To store the routing_map from the fwd
  bool *routing_map_buf = reinterpret_cast<bool *>(comp_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_probs_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  DataType *local_comp_buf = comp_buf + warp_id * num_experts;
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
    // Clear the logits_grad in global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = 0.0f;
    }
    // Load the dgrad/output_from_fwd to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_probs[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      local_routing_map[i] = routing_map[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    /***
         * Section: Backward of ops after the topk
         * - Backward of the used scaling_factor
         * - Sigmoid Post-processing bwd when topk > 1
         * - Softmax bwd if use_pre_softmax is false
         */
    // Backward of the used scaling_factor
    // In-place update
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = static_cast<double>(local_grad[i]) * scaling_factor;
      }
    }
    __syncwarp();

    // Sigmoid Post-processing bwd when topk > 1
    if (topk > 1 && score_function == 0) {
      double sum_fwd_input = masked_warp_reduce_on_shmem(
          /*data ptr = */ local_act_from_fwd,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // Put the result of output * grad to the comp_buf
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = (local_routing_map[i] ? static_cast<double>(local_grad[i]) *
                                                        static_cast<double>(local_act_from_fwd[i])
                                                  : 0.0f);
      }
      __syncwarp();
      double sum_Output_x_Grad = masked_warp_reduce_on_shmem(
          /*data ptr = */ local_comp_buf,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] =
              static_cast<double>(local_grad[i]) / (sum_fwd_input + epsilon) -
              sum_Output_x_Grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
        } else {
          local_grad[i] = 0.0f;
        }
      }
      __syncwarp();
    }

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

    // Sqrtsoftplus Post-processing bwd when topk > 1 (normalization backward)
    if (topk > 1 && score_function == 2) {
      // Now do the normalization backward (same as sigmoid)
      double sum_fwd_input = masked_warp_reduce_on_shmem(
          /*data ptr = */ local_comp_buf,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // Compute sum of output * grad using registers instead of shared memory
      double local_sum_Output_x_Grad = 0.0;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_sum_Output_x_Grad +=
              static_cast<double>(local_grad[i]) * static_cast<double>(local_comp_buf[i]);
        }
      }
      // Warp reduce the sum
      for (int s = 16; s > 0; s /= 2) {
        local_sum_Output_x_Grad += __shfl_xor_sync(0xffffffff, local_sum_Output_x_Grad, s);
      }
      double sum_Output_x_Grad = local_sum_Output_x_Grad;
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] =
              static_cast<double>(local_grad[i]) / (sum_fwd_input + epsilon) -
              sum_Output_x_Grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
        } else {
          local_grad[i] = 0.0f;
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
        local_grad[i] = 0.0f;
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

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const DataType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // grad_probs
                              +
                              num_experts * num_token_per_block * sizeof(DataType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(DataType)  // comp_buf
                              + num_experts * num_token_per_block * sizeof(bool);     // routing_map
  fused_topk_with_score_function_backward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function, grad_logits);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_topk_with_score_function_backward_kernel_launcher<DataType>(
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function,
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward);
  using namespace transformer_engine;
  fused_topk_with_score_function_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), num_groups, group_topk, scaling_factor, score_function,
      *convertNVTETensorCheck(expert_bias), *convertNVTETensorCheck(probs),
      *convertNVTETensorCheck(routing_map), *convertNVTETensorCheck(intermediate_output), stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  using namespace transformer_engine;
  fused_topk_with_score_function_backward(
      *convertNVTETensorCheck(routing_map), *convertNVTETensorCheck(intermediate_output),
      *convertNVTETensorCheck(grad_probs), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *convertNVTETensorCheck(grad_logits), stream);
}
