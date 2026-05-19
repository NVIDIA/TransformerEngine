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
#include "async_loader.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

template <typename DataType, TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk,
    int score_function, float *scores, bool *routing_map, CompType *intermediate_output,
    int num_buffers) {
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

  // The address of buffers on the current warp
  CompType *local_logits = logits_work_buf + warp_id * num_experts;
  CompType *topk_logits = topk_logits_buf + warp_id * topk;
  int *topk_indices = topk_indices_buf + warp_id * topk;

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

    loader.wait();
    DataType *raw_logits = loader.current_buf();

    // Prefetch next round
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        loader.start_load(logits + next_token * num_experts, num_experts, lane_id);
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

    if (score_function == 1) {  // Softmax
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
    } else if (score_function == 0) {  // Sigmoid
      // Fused: convert → sigmoid → save sigmoid output for backward → shmem
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(raw_logits[i]));
        intermediate_output[pos_offset + i] = val;  // Save sigmoid output for backward
        local_logits[i] = val;
      }
    } else if (score_function == 2) {  // Sqrtsoftplus
      // Fused: convert → save original logit for backward → sqrtsoftplus → shmem
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(raw_logits[i]);
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
    vec_fill_global(routing_map + pos_offset, false, num_experts, lane_id);
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
    // Write the scores to the output tensor
    vec_store_global(scores + pos_offset, local_logits, num_experts, lane_id);
    __syncwarp();

    loader.flip();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    float *scores, bool *routing_map, CompType *intermediate_output, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  size_t scores_shmem = num_experts * num_token_per_block * sizeof(CompType);
  size_t scratch_shmem = topk * num_token_per_block * sizeof(CompType)
                         + topk * num_token_per_block * sizeof(int);
  size_t other_shmem = scores_shmem + scratch_shmem;
  size_t logits_single_buf =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(logits_single_buf, other_shmem);
  size_t logits_raw_shmem =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  size_t shared_memory_size = logits_raw_shmem + other_shmem;
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);

  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
        intermediate_output, num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  // Radix selection is O(E), independent of K, but it needs 4 passes for 32-bit float;
  // switch at K=16 where naive O(K^2*E) starts to dominate
  if (topk < 16) {
    launch(fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Naive>);
  } else {
    NVTE_CHECK(num_experts <= kMaxExpertsRadixTopk,
               "Radix topk requires num_experts <= ", kMaxExpertsRadixTopk,
               " (packed 8-bit histogram), got ", num_experts, ".");
    launch(fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Radix>);
  }
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
// Double-buffered cp.async loads both inputs.  Two-pass fused approach.
//
// Shmem layout (B = 2, W = warps/block):
//   grad_buf: B × E × W × sizeof(CompType)   — double-buffered async load
//   act_buf:  B × E × W × sizeof(CompType)   — double-buffered async load
constexpr int kAuxBwdNumBuffers = 2;

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

  extern __shared__ char shmem_aux_bwd[];
  char *shmem_ptr = shmem_aux_bwd;

  CompType *grad_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> grad_loader(grad_shmem_base, warp_id, num_experts,
                                       num_token_per_block, kAuxBwdNumBuffers);
  shmem_ptr += RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block,
                                                     kAuxBwdNumBuffers);

  CompType *act_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> act_loader(act_shmem_base, warp_id, num_experts,
                                      num_token_per_block, kAuxBwdNumBuffers);

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

    grad_loader.wait();
    act_loader.wait();

    CompType *raw_grad = grad_loader.current_buf();
    CompType *raw_act = act_loader.current_buf();

    // Prefetch next round
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        int next_pos = next_token * num_experts;
        grad_loader.start_load(grad_scores + next_pos, num_experts, lane_id);
        act_loader.start_load(intermediate_output + next_pos, num_experts, lane_id);
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
      CompType g = static_cast<CompType>(raw_grad[i]);
      CompType act = raw_act[i];

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

    grad_loader.flip();
    act_loader.flip();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const CompType *intermediate_output, const float *grad_scores, int num_tokens, int num_experts,
    int topk, int score_function, DataType *grad_logits, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  size_t shmem_bytes =
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, kAuxBwdNumBuffers) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, kAuxBwdNumBuffers);
  check_shared_memory_capacity_num_experts(shmem_bytes, num_experts);

  auto kernel = fused_score_for_moe_aux_loss_backward_kernel<DataType>;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
  size_t grid_size =
      compute_persistent_grid(kernel, kThreadsPerBlock, shmem_bytes, total_blocks);
  kernel<<<grid_size, kThreadsPerBlock, shmem_bytes, stream>>>(
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
