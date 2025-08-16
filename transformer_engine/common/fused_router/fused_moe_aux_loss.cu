/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"
#include "common/util/cuda_runtime.h"
#include "utils.h"

namespace transformer_engine {

// Using Double to hanld all the calculations
using CompType = double;

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_kernel(const DataType* probs,
                                                  const IndexType* tokens_per_expert,
                                                  int total_num_tokens, int num_experts,
                                                  int num_rows, int num_cols, int topk, float coeff,
                                                  DataType* aux_loss, float* Const_buf) {
#if __CUDA_ARCH__ >= 900
  // Using cooperative_groups to manage the cluster
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  int thread_id = cg::this_grid().thread_rank();
  int lane_id = thread_id % kThreadsPerWarp;
  int warp_id = thread_id / kThreadsPerWarp;
  int warp_num = blockDim.x * gridDim.x / kThreadsPerWarp;
  // Only 1 block in the cluster
  int block_id = cluster.block_rank();
  int block_num = cluster.dim_blocks().x;
  int cluster_id = blockIdx.x / block_num;
  if (cluster_id > 0) return;  // Only use the cluster 0

  extern __shared__ float shmem_aux_loss[];
  CompType* aggregated_probs_per_expert = reinterpret_cast<CompType*>(shmem_aux_loss);
  // Clear the shmem
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] = CompType(0);
  }
  __syncthreads();

  /**
     * Section: Reduce the probs to the aggregated_probs_per_expert
     * 1. reduce on the block
     * 2. reduce on the cluster
     */
  // Loop: for all positions in each row
  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    CompType tmp = CompType(0);
    // Loop: for all rows that this warp is responsible for
    for (int j = warp_id; j < num_rows; j += warp_num) {
      tmp += CompType(probs[j * num_cols + i]);
    }
    atomicAdd(&aggregated_probs_per_expert[i], tmp);
  }
  cluster.sync();
  // The block 0 will reduce the results of all blocks
  if (block_id == 0) {
    for (int i = 1; i < block_num; i++) {
      // Map the shared memory of the block i to the current block
      CompType* dst_smem = reinterpret_cast<CompType*>(cluster.map_shared_rank(shmem_aux_loss, i));
      for (int j = threadIdx.x; j < num_cols; j += blockDim.x) {
        atomicAdd(&aggregated_probs_per_expert[j], dst_smem[j]);
      }
    }
  }
  cluster.sync();

  /**
     * Section: aggregated_probs_per_expert * tokens_per_expert
     * In-place update on shmem
     */
  if (block_id == 0) {
    for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
      aggregated_probs_per_expert[i] *= CompType(tokens_per_expert[i]);
    }
    __syncthreads();

    if (warp_id == 0) {
      /**
             * Section: Reduce to get the sum of aggregated_probs_per_expert
             */
      CompType intermediate_result =
          warp_reduce_on_shmem(aggregated_probs_per_expert, num_cols, ReduceFuncType::SUM, lane_id);
      __syncwarp();

      if (lane_id == 0) {
        /**
                    * Section: Compute the aux_loss
                    */
        float C_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
        aux_loss[0] = static_cast<DataType>(static_cast<double>(intermediate_result) * C_coeff);
        Const_buf[0] = C_coeff;
      }
    }
  }
#else
  // Use Only 1 block/1024 threads to avoid the grid sync
  if (blockIdx.x > 0) return;
  int warp_num = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_aux_loss[];
  CompType* aggregated_probs_per_expert = reinterpret_cast<CompType*>(shmem_aux_loss);

  // Clear the shmem
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] = CompType(0);
  }
  __syncthreads();

  /**
     * Section: Reduce the probs to the aggregated_probs_per_expert
     */
  // Loop: for all positions in each row
  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    CompType tmp = CompType(0);
    // Loop: for all rows that this warp is responsible for
    for (int j = warp_id; j < num_rows; j += warp_num) {
      tmp += CompType(probs[j * num_cols + i]);
    }
    atomicAdd(&aggregated_probs_per_expert[i], tmp);
  }
  __syncthreads();

  /**
     * Section: aggregated_probs_per_expert * tokens_per_expert
     * In-place update on shmem
     */
  for (int i = threadIdx.x; i < num_cols; i += blockDim.x) {
    aggregated_probs_per_expert[i] *= CompType(tokens_per_expert[i]);
  }
  __syncthreads();

  if (warp_id == 0) {
    /**
         * Section: Reduce to get the sum of aggregated_probs_per_expert
         */
    CompType intermediate_result =
        warp_reduce_on_shmem(aggregated_probs_per_expert, num_cols, ReduceFuncType::SUM, lane_id);
    __syncwarp();

    if (lane_id == 0) {
      /**
             * Section: Compute the aux_loss
             */
      float C_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
      aux_loss[0] = static_cast<DataType>(static_cast<double>(intermediate_result) * C_coeff);
      Const_buf[0] = C_coeff;
    }
  }
#endif
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_forward_kernel_launcher(const DataType* probs,
                                                const IndexType* tokens_per_expert,
                                                int total_num_tokens, int num_experts, int num_rows,
                                                int num_cols, int topk, float coeff,
                                                DataType* aux_loss, float* Const_buf,
                                                cudaStream_t stream) {
  if (cuda::sm_arch(cuda::current_device()) >= 90) {
    cudaLaunchConfig_t config = {0};
    int cluster_size = 8;
    config.gridDim = cluster_size;
    config.blockDim = 1024;
    config.dynamicSmemBytes = sizeof(CompType) * num_cols;
    config.stream = stream;

    // Update the max cluster size based on the device
    NVTE_CHECK_CUDA(cudaOccupancyMaxPotentialClusterSize(
        &cluster_size,
        reinterpret_cast<void*>(fused_moe_aux_loss_forward_kernel<DataType, IndexType>), &config));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.numAttrs = 1;
    config.attrs = attribute;

    NVTE_CHECK_CUDA(cudaLaunchKernelEx(
        &config, fused_moe_aux_loss_forward_kernel<DataType, IndexType>, probs, tokens_per_expert,
        total_num_tokens, num_experts, num_rows, num_cols, topk, coeff, aux_loss, Const_buf));
  } else {
    size_t smem_size = sizeof(CompType) * num_cols;
    fused_moe_aux_loss_forward_kernel<DataType, IndexType>
        <<<1, 1024, smem_size, stream>>>(probs, tokens_per_expert, total_num_tokens, num_experts,
                                         num_rows, num_cols, topk, coeff, aux_loss, Const_buf);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void fused_moe_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert,
                                int total_num_tokens, int num_experts, int num_rows, int num_cols,
                                int topk, float coeff, Tensor& aux_loss, Tensor& Const_buf,
                                cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_forward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), total_num_tokens,
              num_experts, num_rows, num_cols, topk, coeff,
              reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Const_buf.data.dptr), stream);););
}

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_backward_kernel(const float* Const_buf,
                                                   const IndexType* tokens_per_expert, int num_rows,
                                                   int num_cols, DataType* grad_aux_loss,
                                                   DataType* grad_probs) {
  int global_warp_num = gridDim.x * blockDim.x / kThreadsPerWarp;
  int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  // Loop: for all positions in each row
  for (int i = lane_id; i < num_cols; i += kThreadsPerWarp) {
    float C_coeff = Const_buf[0];
    IndexType tokens_per_expert_i = tokens_per_expert[i];
    double grad_aux_loss_value = static_cast<double>(grad_aux_loss[0]);
    // Loop: for all rows
    for (int j = global_warp_id; j < num_rows; j += global_warp_num) {
      grad_probs[j * num_cols + i] = C_coeff * tokens_per_expert_i * grad_aux_loss_value;
    }
  }
}

template <typename DataType, typename IndexType>
void fused_moe_aux_loss_backward_kernel_launcher(const float* Const_buf,
                                                 const IndexType* tokens_per_expert, int num_rows,
                                                 int num_cols, DataType* grad_aux_loss,
                                                 DataType* grad_probs, cudaStream_t stream) {
  // Meta data for the kernel
  int block_size = 256;
  int grid_size = (num_rows + block_size - 1) / block_size;
  fused_moe_aux_loss_backward_kernel<DataType, IndexType><<<grid_size, block_size, 0, stream>>>(
      Const_buf, tokens_per_expert, num_rows, num_cols, grad_aux_loss, grad_probs);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_moe_aux_loss_backward(const Tensor& Const_buf, const Tensor& tokens_per_expert,
                                 int num_rows, int num_cols, Tensor& grad_aux_loss,
                                 Tensor& grad_probs, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_aux_loss.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_backward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<float*>(Const_buf.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), num_rows, num_cols,
              reinterpret_cast<DataType*>(grad_aux_loss.data.dptr),
              reinterpret_cast<DataType*>(grad_probs.data.dptr), stream);););
}

}  // namespace transformer_engine

void nvte_fused_moe_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                     int total_num_tokens, int num_experts, int num_rows,
                                     int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                     NVTETensor Const_buf, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_moe_aux_loss_forward(
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert), total_num_tokens,
      num_experts, num_rows, num_cols, topk, coeff, *convertNVTETensorCheck(aux_loss),
      *convertNVTETensorCheck(Const_buf), stream);
}

void nvte_fused_moe_aux_loss_backward(const NVTETensor Const_buf,
                                      const NVTETensor tokens_per_expert, int num_rows,
                                      int num_cols, NVTETensor grad_aux_loss, NVTETensor grad_probs,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_moe_aux_loss_backward(*convertNVTETensorCheck(Const_buf),
                              *convertNVTETensorCheck(tokens_per_expert), num_rows, num_cols,
                              *convertNVTETensorCheck(grad_aux_loss),
                              *convertNVTETensorCheck(grad_probs), stream);
}
