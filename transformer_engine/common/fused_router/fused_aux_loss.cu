#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"
#include "utils.h"

namespace transformer_engine {

template <typename DataType, typename IndexType>
__global__ void fused_aux_loss_forward_kernel(const DataType* probs,
                                              const IndexType* tokens_per_expert, int num_tokens,
                                              int num_experts, int topk, float coeff,
                                              DataType* aux_loss, float* Const_buf) {
  int warp_num = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ DataType aggregated_probs_per_expert[];
  // Clear the shmem
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    aggregated_probs_per_expert[i] = 0;
  }
  __syncthreads();

  /**
     * Section: Reduce the probs to the aggregated_probs_per_expert
     */
  // Loop: for all positions in each row
  for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
    DataType tmp = 0;
    // Loop: for all rows that this warp is responsible for
    for (int j = warp_id; j < num_tokens; j += warp_num) {
      tmp += probs[j * num_experts + i];
    }
    atomicAdd(&aggregated_probs_per_expert[i], tmp);
  }
  __syncthreads();

  /**
     * Section: aggregated_probs_per_expert * tokens_per_expert
     * In-place update on shmem
     */
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    aggregated_probs_per_expert[i] *= tokens_per_expert[i];
  }
  __syncthreads();

  if (warp_id == 0) {
    /**
         * Section: Reduce to get the sum of aggregated_probs_per_expert
         */
    DataType intermediate_result =
        warp_reduce_on_shmem(aggregated_probs_per_expert, num_experts, sum, lane_id);
    __syncwarp();

    if (lane_id == 0) {
      /**
             * Section: Compute the aux_loss
             */
      float C_coeff = (num_experts * coeff) / topk / num_tokens / num_tokens;
      aux_loss[0] = intermediate_result * C_coeff;
      Const_buf[0] = C_coeff;
    }
  }
}

template <typename DataType, typename IndexType>
void fused_aux_loss_forward_kernel_launcher(const DataType* probs,
                                            const IndexType* tokens_per_expert, int num_tokens,
                                            int num_experts, int topk, float coeff,
                                            DataType* aux_loss, float* Const_buf,
                                            cudaStream_t stream) {
  // Meta data for the kernel
  size_t shared_memory_size = sizeof(DataType) * num_experts * 2;
  // Use Only 1 block/1024 threads to avoid the grid sync
  int grid_size = 1;
  int block_size = 1024;
  fused_aux_loss_forward_kernel<DataType, IndexType>
      <<<grid_size, block_size, shared_memory_size, stream>>>(
          probs, tokens_per_expert, num_tokens, num_experts, topk, coeff, aux_loss, Const_buf);
}

void fused_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert, int num_tokens,
                            int num_experts, int topk, float coeff, Tensor& aux_loss,
                            Tensor& Const_buf, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_aux_loss_forward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), num_tokens, num_experts,
              topk, coeff, reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Const_buf.data.dptr), stream);););
}

template <typename DataType, typename IndexType>
__global__ void fused_aux_loss_backward_kernel(const float* Const_buf,
                                               const IndexType* tokens_per_expert, int num_tokens,
                                               int num_experts, DataType* grad_aux_loss,
                                               DataType* grad_probs) {
  int global_warp_num = gridDim.x * blockDim.x / kThreadsPerWarp;
  int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  // Loop: for all positions in each row
  for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
    DataType C_coeff = Const_buf[0];
    IndexType tokens_per_expert_i = tokens_per_expert[i];
    DataType grad_aux_loss_value = grad_aux_loss[0];
    // Loop: for all rows
    for (int j = global_warp_id; j < num_tokens; j += global_warp_num) {
      grad_probs[j * num_experts + i] = C_coeff * tokens_per_expert_i * grad_aux_loss_value;
    }
  }
}

template <typename DataType, typename IndexType>
void fused_aux_loss_backward_kernel_launcher(const float* Const_buf,
                                             const IndexType* tokens_per_expert, int num_tokens,
                                             int num_experts, DataType* grad_aux_loss,
                                             DataType* grad_probs, cudaStream_t stream) {
  // Meta data for the kernel
  int block_size = 256;
  int grid_size = (num_tokens + block_size - 1) / block_size;
  fused_aux_loss_backward_kernel<DataType, IndexType><<<grid_size, block_size, 0, stream>>>(
      Const_buf, tokens_per_expert, num_tokens, num_experts, grad_aux_loss, grad_probs);
}

void fused_aux_loss_backward(const Tensor& Const_buf, const Tensor& tokens_per_expert,
                             int num_tokens, int num_experts, Tensor& grad_aux_loss,
                             Tensor& grad_probs, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_aux_loss.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_aux_loss_backward_kernel_launcher<DataType, IndexType>(
              reinterpret_cast<float*>(Const_buf.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), num_tokens, num_experts,
              reinterpret_cast<DataType*>(grad_aux_loss.data.dptr),
              reinterpret_cast<DataType*>(grad_probs.data.dptr), stream);););
}

}  // namespace transformer_engine

void nvte_fused_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                 int num_tokens, int num_experts, int topk, float coeff,
                                 NVTETensor aux_loss, NVTETensor Const_buf, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_aux_loss_forward);
  using namespace transformer_engine;
  fused_aux_loss_forward(*convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert),
                         num_tokens, num_experts, topk, coeff, *convertNVTETensorCheck(aux_loss),
                         *convertNVTETensorCheck(Const_buf), stream);
}

void nvte_fused_aux_loss_backward(const NVTETensor Const_buf, const NVTETensor tokens_per_expert,
                                  int num_tokens, int num_experts, NVTETensor grad_aux_loss,
                                  NVTETensor grad_probs, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_aux_loss_backward);
  using namespace transformer_engine;
  fused_aux_loss_backward(*convertNVTETensorCheck(Const_buf),
                          *convertNVTETensorCheck(tokens_per_expert), num_tokens, num_experts,
                          *convertNVTETensorCheck(grad_aux_loss),
                          *convertNVTETensorCheck(grad_probs), stream);
}
