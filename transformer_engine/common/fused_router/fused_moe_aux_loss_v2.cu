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
#include "common/util/cuda_runtime.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

template <typename DataType, typename IndexType>
__global__ void fused_moe_aux_loss_forward_kernel_v2(const DataType* probs,
                                                     const IndexType* tokens_per_expert,
                                                     int total_num_tokens, int num_experts,
                                                     int num_rows, int num_cols, int topk,
                                                     float coeff, DataType* aux_loss,
                                                     float* Const_buf) {
  // -----------------------------------------------------------------------
  // 1) Compute the constant coefficient (identical for all threads)
  // -----------------------------------------------------------------------
  const float C_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    Const_buf[0] = C_coeff;
  }

  // -----------------------------------------------------------------------
  // 2) Each CTA computes a partial dot-product:
  //    Sigma_col ( Sigma_row probs[row, col] ) * tokens_per_expert[col]
  // -----------------------------------------------------------------------
  CompType thread_sum = CompType(0);

  // Grid-stride over rows so that every row is processed exactly once.
  // Each thread processes a subset of columns.
  for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
    CompType col_sum = CompType(0);

    // Accumulate probs over the rows assigned to this CTA (grid-stride).
    for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {
      col_sum += CompType(probs[row * num_cols + col]);
    }

    // Multiply by the token count for this expert.
    col_sum *= CompType(tokens_per_expert[col]);

    // Accumulate the per-column contribution into the thread-local sum.
    thread_sum += col_sum;
  }

  // -----------------------------------------------------------------------
  // 3) Block-level reduction of thread_sum using warp_reduce_on_shmem
  // -----------------------------------------------------------------------
  extern __shared__ float shmem[];
  CompType* shmem_block = reinterpret_cast<CompType*>(shmem);
  shmem_block[threadIdx.x] = thread_sum;
  __syncthreads();

  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane_id = threadIdx.x % kThreadsPerWarp;
  if (warp_id == 0) {
    CompType block_sum = warp_reduce_on_shmem(shmem_block, static_cast<int>(blockDim.x),
                                              ReduceFuncType::SUM, lane_id);
    if (lane_id == 0) {
      atomicAdd(&Const_buf[1], block_sum * Const_buf[0]);
    }
  }

  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    aux_loss[0] = static_cast<DataType>(Const_buf[1]);
  }
}

/* -------------------------------------------------------------------------
 *  Kernel launcher -- simplified (no cluster launch).
 * ------------------------------------------------------------------------- */
template <typename DataType, typename IndexType>
void fused_moe_aux_loss_forward_kernel_launcher_v2(const DataType* probs,
                                                   const IndexType* tokens_per_expert,
                                                   int total_num_tokens, int num_experts,
                                                   int num_rows, int num_cols, int topk,
                                                   float coeff, DataType* aux_loss,
                                                   float* Const_buf, cudaStream_t stream) {
  // Round up to a multiple of warp size for correct warp shuffles.
  const int block_size = ((std::min(1024, num_cols) + static_cast<int>(kThreadsPerWarp) - 1) /
                          static_cast<int>(kThreadsPerWarp)) *
                         static_cast<int>(kThreadsPerWarp);
  const int grid_size = cuda::sm_count() * 2;

  // One CompType per thread in shared memory.
  const size_t smem_size = block_size * sizeof(CompType);
  check_shared_memory_capacity_num_experts(smem_size, num_experts);

  // Zero the float accumulator (Const_buf[1]) before launch.
  NVTE_CHECK_CUDA(cudaMemsetAsync(Const_buf + 1, 0, sizeof(float), stream));

  fused_moe_aux_loss_forward_kernel_v2<DataType, IndexType>
      <<<grid_size, block_size, smem_size, stream>>>(probs, tokens_per_expert, total_num_tokens,
                                                     num_experts, num_rows, num_cols, topk, coeff,
                                                     aux_loss, Const_buf);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_moe_aux_loss_forward_v2(const Tensor& probs, const Tensor& tokens_per_expert,
                                   int total_num_tokens, int num_experts, int num_rows,
                                   int num_cols, int topk, float coeff, Tensor& aux_loss,
                                   Tensor& Const_buf, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_forward_kernel_launcher_v2<DataType, IndexType>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), total_num_tokens,
              num_experts, num_rows, num_cols, topk, coeff,
              reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Const_buf.data.dptr), stream);););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_moe_aux_loss_forward_v2(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                        int total_num_tokens, int num_experts, int num_rows,
                                        int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                        NVTETensor Const_buf, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward_v2);
  using namespace transformer_engine;
  fused_router::fused_moe_aux_loss_forward_v2(
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert), total_num_tokens,
      num_experts, num_rows, num_cols, topk, coeff, *convertNVTETensorCheck(aux_loss),
      *convertNVTETensorCheck(Const_buf), stream);
}
