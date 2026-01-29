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

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType *logits, int num_tokens,
                                                            int num_experts, int topk,
                                                            int score_function, DataType *scores,
                                                            bool *routing_map,
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
  extern __shared__ float shmem_scores_for_aux_loss[];
  DataType *logits_buf = reinterpret_cast<DataType *>(shmem_scores_for_aux_loss);
  DataType *topk_logits_buf =
      reinterpret_cast<DataType *>(logits_buf + num_experts * num_token_per_block);
  int *topk_indices_buf = reinterpret_cast<int *>(topk_logits_buf + topk * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_logits = logits_buf + warp_id * num_experts;
  DataType *topk_logits = topk_logits_buf + warp_id * topk;
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
    // Clear the routing_map (num_experts)
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      routing_map[pos_offset + i] = false;
      if (score_function == 1) {
        intermediate_output[pos_offset + i] = -std::numeric_limits<DataType>::infinity();
      }
    }
    // Load the logits to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_logits[i] = logits[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    /***
         * Section: Preprocess
         * Possible preprocess the scores before the topk operation
         * - Pre-softmax
         * - Sigmoid
         * - Sqrtsoftplus
         * - Sigmoid/Sqrtsoftplus post-processing when topk > 1
         * This is in-place scores update
         */
    if (score_function == 1) {  // score_function == 1 means softmax
      // Apply softmax to the logits before the topk
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 0) {  // score_function == 0 means sigmoid
      // Apply sigmoid to the logits
      apply_sigmoid_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      // Save the sigmoid output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 2) {  // score_function == 2 means sqrtsoftplus
      // First save the original logits for backward (needed for gradient computation)
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];  // Save original logits
      }
      __syncwarp();
      // Apply sqrtsoftplus to the logits
      apply_sqrtsoftplus_on_float(local_logits, num_experts, lane_id);
    }

    __syncwarp();  //Confirm the scores is written to the output

    // Sigmoid/Sqrtsoftplus post-processing when topk > 1
    if (score_function == 0 || score_function == 2) {
      if (topk > 1) {
        auto sum_logits =
            warp_reduce_on_shmem(local_logits, num_experts, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          local_logits[i] = static_cast<DataType>(static_cast<double>(local_logits[i]) /
                                                  (static_cast<double>(sum_logits) + epsilon));
        }
      }
      __syncwarp();
    }

    /***
         * Section: Topk
         * Get the topk indices
         */
    naive_topk_and_mask(local_logits, num_experts, topk, topk_indices, topk_logits, lane_id);
    __syncwarp();

    // Write the routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
    // Write the scores to the output tensor
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      scores[pos_offset + i] = local_logits[i];
    }
    __threadfence_block();
    __syncwarp();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    DataType *scores, bool *routing_map, DataType *intermediate_output, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // logits
                              + topk * num_token_per_block * sizeof(DataType)       // topk_logits
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  fused_score_for_moe_aux_loss_forward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
          intermediate_output);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_score_for_moe_aux_loss_forward(const Tensor &logits, int num_tokens, int num_experts,
                                          int topk, int score_function, Tensor &scores,
                                          Tensor &routing_map, Tensor &intermediate_output,
                                          cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_forward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(scores.data.dptr),
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<DataType *>(intermediate_output.data.dptr), stream););
}

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const DataType *intermediate_output,
                                                             const DataType *grad_scores,
                                                             int num_tokens, int num_experts,
                                                             int topk, int score_function,
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
  DataType *grad_scores_buf = reinterpret_cast<DataType *>(shmem);
  // To store the output of softmax/sigmoid from the fwd
  DataType *act_from_fwd_buf =
      reinterpret_cast<DataType *>(grad_scores_buf + num_experts * num_token_per_block);
  DataType *comp_buf =
      reinterpret_cast<DataType *>(act_from_fwd_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  DataType *local_grad = grad_scores_buf + warp_id * num_experts;
  DataType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  DataType *local_comp_buf = comp_buf + warp_id * num_experts;

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
      local_grad[i] = grad_scores[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1
         * - Sigmoid bwd
         * - Sqrtsoftplus bwd
         * - Write the grad_logits to the global mem
         */
    // Sigmoid Post-processing bwd when topk > 1
    if (topk > 1 && score_function == 0) {
      auto sum_fwd_input =
          warp_reduce_on_shmem(local_act_from_fwd, num_experts, ReduceFuncType::SUM, lane_id);
      // Put the result of output * grad to the comp_buf
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = local_grad[i] * local_act_from_fwd[i];
      }
      __syncwarp();
      auto sum_Output_x_Grad =
          warp_reduce_on_shmem(local_comp_buf, num_experts, ReduceFuncType::SUM, lane_id);
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_grad[i] =
            static_cast<double>(local_grad[i]) / (static_cast<double>(sum_fwd_input) + epsilon) -
            static_cast<double>(sum_Output_x_Grad) /
                ((static_cast<double>(sum_fwd_input) + epsilon) *
                 (static_cast<double>(sum_fwd_input) + epsilon));
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
      auto sum_fwd_input =
          warp_reduce_on_shmem(local_comp_buf, num_experts, ReduceFuncType::SUM, lane_id);
      // Compute sum of output * grad using registers instead of shared memory
      double local_sum_Output_x_Grad = 0.0;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_sum_Output_x_Grad +=
            static_cast<double>(local_grad[i]) * static_cast<double>(local_comp_buf[i]);
      }
      // Warp reduce the sum
      for (int s = 16; s > 0; s /= 2) {
        local_sum_Output_x_Grad += __shfl_xor_sync(0xffffffff, local_sum_Output_x_Grad, s);
      }
      double sum_Output_x_Grad = local_sum_Output_x_Grad;
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_grad[i] =
            static_cast<double>(local_grad[i]) / (static_cast<double>(sum_fwd_input) + epsilon) -
            sum_Output_x_Grad / ((static_cast<double>(sum_fwd_input) + epsilon) *
                                 (static_cast<double>(sum_fwd_input) + epsilon));
      }
      __syncwarp();
    }

    // Pre-softmax bwd
    if (score_function == 1) {
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
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const DataType *intermediate_output, const DataType *grad_scores, int num_tokens,
    int num_experts, int topk, int score_function, DataType *grad_logits, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(DataType)  // grad_scores
                              +
                              num_experts * num_token_per_block * sizeof(DataType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(DataType);  // comp_buf
  fused_score_for_moe_aux_loss_backward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          intermediate_output, grad_scores, num_tokens, num_experts, topk, score_function,
          grad_logits);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_score_for_moe_aux_loss_backward(const Tensor &intermediate_output,
                                           const Tensor &grad_scores, int num_tokens,
                                           int num_experts, int topk, int score_function,
                                           Tensor &grad_logits, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_scores.data.dtype, DataType,
      fused_score_for_moe_aux_loss_backward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_scores.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace transformer_engine

void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, const NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_score_for_moe_aux_loss_forward(*convertNVTETensorCheck(logits), num_tokens, num_experts,
                                       topk, score_function, *convertNVTETensorCheck(scores),
                                       *convertNVTETensorCheck(routing_map),
                                       *convertNVTETensorCheck(intermediate_output), stream);
}

void nvte_fused_score_for_moe_aux_loss_backward(const NVTETensor intermediate_output,
                                                const NVTETensor grad_scores, int num_tokens,
                                                int num_experts, int topk, int score_function,
                                                NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_score_for_moe_aux_loss_backward(
      *convertNVTETensorCheck(intermediate_output), *convertNVTETensorCheck(grad_scores),
      num_tokens, num_experts, topk, score_function, *convertNVTETensorCheck(grad_logits), stream);
}
