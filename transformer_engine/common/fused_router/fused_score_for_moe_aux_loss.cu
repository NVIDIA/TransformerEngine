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
namespace fused_router {

template <typename DataType, TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(const DataType *logits, int num_tokens,
                                                            int num_experts, int topk,
                                                            int score_function, float *scores,
                                                            bool *routing_map,
                                                            CompType *intermediate_output) {
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_scores_for_aux_loss[];
  CompType *logits_buf = reinterpret_cast<CompType *>(shmem_scores_for_aux_loss);
  CompType *topk_logits_buf =
      reinterpret_cast<CompType *>(logits_buf + num_experts * num_token_per_block);
  int *topk_indices_buf = reinterpret_cast<int *>(topk_logits_buf + topk * num_token_per_block);
  // The address of buffers on the current warp
  CompType *local_logits = logits_buf + warp_id * num_experts;
  CompType *topk_logits = topk_logits_buf + warp_id * topk;
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
         * Section: Init buffer + Preprocess
         * - Load logits → apply score function → save intermediate_output
         *
         * Fused into a single loop per score function where possible:
         *   score_function == 0 (sigmoid):      load, sigmoid, save → shmem
         *   score_function == 1 (softmax):      load → shmem, softmax (multi-pass), save
         *   score_function == 2 (sqrtsoftplus): load, save logits, sqrtsoftplus → shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;

    if (score_function == 1) {  // Softmax
      // Apply softmax to all logits, save softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] = static_cast<CompType>(logits[pos_offset + i]);
      }
      __syncwarp();
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      // Save the softmax output for backward
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 0) {  // Sigmoid
      // Fused: load logit → sigmoid → save sigmoid output for backward → shmem
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(logits[pos_offset + i]));
        intermediate_output[pos_offset + i] = val;  // Save sigmoid output for backward
        local_logits[i] = val;
      }
    } else if (score_function == 2) {  // Sqrtsoftplus
      // Fused: load logit → save original logit for backward → sqrtsoftplus → shmem
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(logits[pos_offset + i]);
        intermediate_output[pos_offset + i] = logit;  // Save original logits for backward
        local_logits[i] = sqrtsoftplus_scalar(logit);
      }
    }
    __syncwarp();

    // Sigmoid/Sqrtsoftplus post-processing: normalize scores to sum to 1
    if (score_function == 0 || score_function == 2) {
      auto sum_logits =
          warp_reduce_on_shmem(local_logits, num_experts, ReduceFuncType::SUM, lane_id);
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
    float *scores, bool *routing_map, CompType *intermediate_output, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType)  // logits
                              + topk * num_token_per_block * sizeof(CompType)       // topk_logits
                              + topk * num_token_per_block * sizeof(int);           // topk_indices
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  // Radix selection is O(E), independent of K, but it needs 4 passes for 32-bit float;
  // switch at K=16 where naive O(K^2*E) starts to dominate
  if (topk < 16) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Naive>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Naive>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
            intermediate_output);
  } else {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Radix>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Radix>
        <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
            logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
            intermediate_output);
  }
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
          score_function, reinterpret_cast<float *>(scores.data.dptr),
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream););
}

// Backward: grad_scores + intermediate_output → grad_logits.
// No routing_map — all experts participate (unlike topk backward).
//
// Two-pass fused approach (eliminates the comp_buf shmem buffer):
//   Pass 1 (reduction): accumulate warp-level sums for normalization/softmax bwd.
//   Pass 2 (element-wise): compute per-element gradient and write to global memory.
//
// Shmem layout (W = warps/block):
//   grad_buf: E × W × sizeof(CompType)   — grad_scores loaded from global
//   act_buf:  E × W × sizeof(CompType)   — intermediate_output (sigmoid/softmax out, or logits)
template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const CompType *intermediate_output,
                                                             const float *grad_scores,
                                                             int num_tokens, int num_experts,
                                                             int topk, int score_function,
                                                             DataType *grad_logits) {
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  CompType *grad_buf = reinterpret_cast<CompType *>(shmem);
  CompType *act_buf = grad_buf + num_experts * num_token_per_block;
  // The address of buffers on the current warp
  CompType *local_grad = grad_buf + warp_id * num_experts;
  CompType *local_act = act_buf + warp_id * num_experts;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    /***
         * Section: Load inputs to shmem
         * - Load the grad_scores/intermediate_output to shmem
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_scores[pos + i];
      local_act[i] = intermediate_output[pos + i];
    }
    __syncwarp();

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
      CompType g = local_grad[i];
      CompType act = local_act[i];
      if (score_function == 0) {  // Sigmoid
        // act = sigmoid output; accumulate over all experts
        sum_act += act; sum_grad_act += g * act;
      } else if (score_function == 2) {  // Sqrtsoftplus
        // act = original logit; recompute sqrtsoftplus to get activation
        CompType v = sqrtsoftplus_scalar(act);
        sum_act += v; sum_grad_act += g * v;
      } else if (score_function == 1) {  // Softmax
        // act = softmax output
        sum_output_x_grad += g * act;
      }
    }
    if (score_function == 0 || score_function == 2) {
      sum_act = warp_allreduce_sum(sum_act);
      sum_grad_act = warp_allreduce_sum(sum_grad_act);
    }
    if (score_function == 1) {
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
      CompType g = local_grad[i];
      CompType act = local_act[i];

      if (score_function == 0) {  // Sigmoid bwd
        g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
        g = sigmoid_bwd_scalar(g, act);
      } else if (score_function == 2) {  // Sqrtsoftplus bwd
        g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
        g = sqrtsoftplus_bwd_scalar(g, act, sqrtsoftplus_scalar(act));
      } else if (score_function == 1) {  // Softmax bwd
        g = softmax_bwd_scalar(g, act, sum_output_x_grad);
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }
    __syncwarp();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const CompType *intermediate_output, const float *grad_scores, int num_tokens, int num_experts,
    int topk, int score_function, DataType *grad_logits, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  // Shmem: grad_buf + act_buf (no comp_buf — eliminated by fused scalar approach)
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType)  // grad_buf
                              + num_experts * num_token_per_block * sizeof(CompType);  // act_buf
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(fused_score_for_moe_aux_loss_backward_kernel<DataType>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       shared_memory_size));
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
      grad_logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_backward_kernel_launcher<DataType>(
          reinterpret_cast<CompType *>(intermediate_output.data.dptr),
          reinterpret_cast<float *>(grad_scores.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, const NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_router::fused_score_for_moe_aux_loss_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk, score_function,
      *convertNVTETensorCheck(scores), *convertNVTETensorCheck(routing_map),
      *convertNVTETensorCheck(intermediate_output), stream);
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
