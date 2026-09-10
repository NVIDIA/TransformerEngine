/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include <algorithm>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"
#include "common/util/cuda_runtime.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

// Fixed launch geometry for each input shape bounds scratch to at most 1 KiB.
// Each CTA contributes one FP32 partial sum.
constexpr int kAuxLossMaxReductionBlocks = 256;

int aux_loss_reduction_blocks(int num_rows) {
  NVTE_CHECK(num_rows >= 0, "num_rows must be nonnegative, got ", num_rows);
  return std::max(1, std::min(num_rows, kAuxLossMaxReductionBlocks));
}

void check_aux_loss_workspace(const Tensor& workspace, int num_rows) {
  const size_t num_partials = aux_loss_reduction_blocks(num_rows);
  NVTE_CHECK(workspace.data.dptr != nullptr, "Aux-loss workspace must be preallocated.");
  NVTE_CHECK(workspace.data.dtype == DType::kFloat32, "Aux-loss workspace must have FP32 dtype.");
  NVTE_CHECK(workspace.numel() >= num_partials, "Aux-loss workspace needs at least ", num_partials,
             " FP32 elements, got ", workspace.numel());
}

void check_aux_loss_const_buffer(const Tensor& const_buf) {
  NVTE_CHECK(const_buf.data.dtype == DType::kFloat32 && const_buf.numel() >= 2,
             "Aux-loss constant buffer must contain at least two FP32 elements.");
}

template <typename DataType, typename IndexType, bool kDeterministic>
__global__ void fused_moe_aux_loss_forward_kernel(const DataType* probs,
                                                  const IndexType* tokens_per_expert,
                                                  int total_num_tokens, int num_rows, int num_cols,
                                                  int topk, float coeff, float* Coeff_buf,
                                                  float* partial_sums) {
  // -----------------------------------------------------------------------
  // 1) Write the CPU-computed coefficient into a device buffer to re-use in BWD
  // -----------------------------------------------------------------------
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Coeff_buf[0] = coeff;
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
    CompType block_sum = warp_reduce_on_shmem<CompType, ReduceFuncType::SUM>(
        shmem_block, static_cast<int>(blockDim.x), lane_id);
    if (lane_id == 0) {
      if constexpr (kDeterministic) {
        // Every CTA writes its own slot, including empty row partitions.
        partial_sums[blockIdx.x] = static_cast<float>(block_sum * coeff);
      } else {
        // Retained only for callers of the original C API.
        atomicAdd(&Coeff_buf[1], static_cast<float>(block_sum * coeff));
      }
    }
  }
}

// Small kernel to convert the float accumulator to the output DataType.
template <typename DataType>
__global__ void convert_accum_to_output(const float* Coeff_buf, DataType* aux_loss) {
  aux_loss[0] = static_cast<DataType>(Coeff_buf[1]);
}

// One full warp reads fixed, strided slices and combines them with a fixed tree.
// This replaces the legacy cast kernel, so deterministic forward still uses two
// launches. The final FP32 sum remains in Coeff_buf[1] for compatibility.
template <typename DataType>
__global__ void reduce_aux_loss_partials(const float* partial_sums, int num_partials,
                                         float* Coeff_buf, DataType* aux_loss) {
  const int lane_id = threadIdx.x;
  float sum = 0.0f;
  for (int i = lane_id; i < num_partials; i += kThreadsPerWarp) {
    sum += partial_sums[i];
  }
  for (int offset = kThreadsPerWarp / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane_id == 0) {
    Coeff_buf[1] = sum;
    aux_loss[0] = static_cast<DataType>(sum);
  }
}

/* -------------------------------------------------------------------------
 *  Kernel launcher -- simplified (no cluster launch).
 * ------------------------------------------------------------------------- */
template <typename DataType, typename IndexType, bool kDeterministic = false>
void fused_moe_aux_loss_forward_kernel_launcher(
    const DataType* probs, const IndexType* tokens_per_expert, int total_num_tokens,
    int num_experts, int num_rows, int num_cols, int topk, float coeff, DataType* aux_loss,
    float* Coeff_buf, cudaStream_t stream, float* partial_sums = nullptr) {
  NVTE_CHECK(num_cols > 0, "num_cols must be positive, got ", num_cols);
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  // Sequence aux loss batches independent sequences along the expert dimension.
  NVTE_CHECK(num_cols % num_experts == 0, "Number of input columns (", num_cols,
             ") must be a multiple of number of experts (", num_experts, ").");

  // Round up to a multiple of warp size for correct warp shuffles.
  const int block_size = ((std::min(1024, num_cols) + static_cast<int>(kThreadsPerWarp) - 1) /
                          static_cast<int>(kThreadsPerWarp)) *
                         static_cast<int>(kThreadsPerWarp);
  const int grid_size = kDeterministic ? aux_loss_reduction_blocks(num_rows) : cuda::sm_count() * 2;

  // One CompType per thread in shared memory.
  const size_t smem_size = block_size * sizeof(CompType);
  check_shared_memory_capacity_num_experts(smem_size, num_cols);

  // Keep coefficient placement unchanged: scale each partial before summing.
  const float C_coeff = (num_experts * coeff) / topk / total_num_tokens / total_num_tokens;
  if constexpr (!kDeterministic) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(Coeff_buf + 1, 0, sizeof(float), stream));
  }
  fused_moe_aux_loss_forward_kernel<DataType, IndexType, kDeterministic>
      <<<grid_size, block_size, smem_size, stream>>>(probs, tokens_per_expert, total_num_tokens,
                                                     num_rows, num_cols, topk, C_coeff, Coeff_buf,
                                                     partial_sums);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if constexpr (kDeterministic) {
    reduce_aux_loss_partials<DataType>
        <<<1, kThreadsPerWarp, 0, stream>>>(partial_sums, grid_size, Coeff_buf, aux_loss);
  } else {
    convert_accum_to_output<DataType><<<1, 1, 0, stream>>>(Coeff_buf, aux_loss);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool kDeterministic = false>
void fused_moe_aux_loss_forward(const Tensor& probs, const Tensor& tokens_per_expert,
                                int total_num_tokens, int num_experts, int num_rows, int num_cols,
                                int topk, float coeff, Tensor& aux_loss, Tensor& Coeff_buf,
                                cudaStream_t stream, float* partial_sums = nullptr) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_forward_kernel_launcher<DataType, IndexType, kDeterministic>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr), total_num_tokens,
              num_experts, num_rows, num_cols, topk, coeff,
              reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Coeff_buf.data.dptr), stream, partial_sums);););
}

/* -------------------------------------------------------------------------
 *  CUDA-graph-safe variant: total_num_tokens lives in a 0-dim int64 device
 *  tensor whose value can change between graph replays. Each CTA's reduction
 *  lane computes C_coeff directly from the device value. The first CTA also
 *  writes C_coeff into Coeff_buf[0] for backward.
 * ------------------------------------------------------------------------- */
template <typename DataType, typename IndexType, bool kDeterministic>
__global__ void fused_moe_aux_loss_forward_kernel_graph_safe(
    const DataType* probs, const IndexType* tokens_per_expert, const int64_t* total_num_tokens_ptr,
    int num_experts, int num_rows, int num_cols, int topk, float coeff, float* Coeff_buf,
    float* partial_sums) {
  // Reduction body matches the scalar-input kernel above.
  CompType thread_sum = CompType(0);
  for (int col = threadIdx.x; col < num_cols; col += blockDim.x) {
    CompType col_sum = CompType(0);
    for (int row = blockIdx.x; row < num_rows; row += gridDim.x) {
      col_sum += CompType(probs[row * num_cols + col]);
    }
    col_sum *= CompType(tokens_per_expert[col]);
    thread_sum += col_sum;
  }

  extern __shared__ float shmem[];
  CompType* shmem_block = reinterpret_cast<CompType*>(shmem);
  shmem_block[threadIdx.x] = thread_sum;
  __syncthreads();

  const int warp_id = threadIdx.x / kThreadsPerWarp;
  const int lane_id = threadIdx.x % kThreadsPerWarp;
  if (warp_id == 0) {
    CompType block_sum = warp_reduce_on_shmem<CompType, ReduceFuncType::SUM>(
        shmem_block, static_cast<int>(blockDim.x), lane_id);
    if (lane_id == 0) {
      const float total_num_tokens = static_cast<float>(*total_num_tokens_ptr);
      const float C_coeff = (static_cast<float>(num_experts) * coeff) / static_cast<float>(topk) /
                            total_num_tokens / total_num_tokens;
      if (blockIdx.x == 0) {
        Coeff_buf[0] = C_coeff;
      }
      if constexpr (kDeterministic) {
        partial_sums[blockIdx.x] = static_cast<float>(block_sum * C_coeff);
      } else {
        atomicAdd(&Coeff_buf[1], static_cast<float>(block_sum * C_coeff));
      }
    }
  }
}

template <typename DataType, typename IndexType, bool kDeterministic = false>
void fused_moe_aux_loss_forward_kernel_launcher_graph_safe(
    const DataType* probs, const IndexType* tokens_per_expert, const int64_t* total_num_tokens_dev,
    int num_experts, int num_rows, int num_cols, int topk, float coeff, DataType* aux_loss,
    float* Coeff_buf, cudaStream_t stream, float* partial_sums = nullptr) {
  NVTE_CHECK(num_cols > 0, "num_cols must be positive, got ", num_cols);
  NVTE_CHECK(num_experts > 0, "num_experts must be positive, got ", num_experts);
  NVTE_CHECK(num_cols % num_experts == 0, "Number of input columns (", num_cols,
             ") must be a multiple of number of experts (", num_experts, ").");

  const int block_size = ((std::min(1024, num_cols) + static_cast<int>(kThreadsPerWarp) - 1) /
                          static_cast<int>(kThreadsPerWarp)) *
                         static_cast<int>(kThreadsPerWarp);
  const int grid_size = kDeterministic ? aux_loss_reduction_blocks(num_rows) : cuda::sm_count() * 2;
  const size_t smem_size = block_size * sizeof(CompType);
  check_shared_memory_capacity_num_experts(smem_size, num_cols);

  // The main kernel refreshes Coeff_buf[0] for backward on every graph replay.
  if constexpr (!kDeterministic) {
    NVTE_CHECK_CUDA(cudaMemsetAsync(Coeff_buf + 1, 0, sizeof(float), stream));
  }
  fused_moe_aux_loss_forward_kernel_graph_safe<DataType, IndexType, kDeterministic>
      <<<grid_size, block_size, smem_size, stream>>>(probs, tokens_per_expert, total_num_tokens_dev,
                                                     num_experts, num_rows, num_cols, topk, coeff,
                                                     Coeff_buf, partial_sums);
  NVTE_CHECK_CUDA(cudaGetLastError());

  if constexpr (kDeterministic) {
    reduce_aux_loss_partials<DataType>
        <<<1, kThreadsPerWarp, 0, stream>>>(partial_sums, grid_size, Coeff_buf, aux_loss);
  } else {
    convert_accum_to_output<DataType><<<1, 1, 0, stream>>>(Coeff_buf, aux_loss);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool kDeterministic = false>
void fused_moe_aux_loss_forward_graph_safe(const Tensor& probs, const Tensor& tokens_per_expert,
                                           const Tensor& total_num_tokens, int num_experts,
                                           int num_rows, int num_cols, int topk, float coeff,
                                           Tensor& aux_loss, Tensor& Coeff_buf, cudaStream_t stream,
                                           float* partial_sums = nullptr) {
  NVTE_CHECK(total_num_tokens.data.dtype == DType::kInt64,
             "total_num_tokens must be a 0-dim int64 tensor; got dtype ",
             static_cast<int>(total_num_tokens.data.dtype));
  NVTE_CHECK(total_num_tokens.numel() == 1,
             "total_num_tokens must contain exactly one element; got ", total_num_tokens.numel());
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      probs.data.dtype, DataType,
      TE_ROUTER_INDEX_TYPE_SWITCH_ALL(
          tokens_per_expert.data.dtype, IndexType,
          fused_moe_aux_loss_forward_kernel_launcher_graph_safe<DataType, IndexType,
                                                                kDeterministic>(
              reinterpret_cast<DataType*>(probs.data.dptr),
              reinterpret_cast<IndexType*>(tokens_per_expert.data.dptr),
              reinterpret_cast<const int64_t*>(total_num_tokens.data.dptr), num_experts, num_rows,
              num_cols, topk, coeff, reinterpret_cast<DataType*>(aux_loss.data.dptr),
              reinterpret_cast<float*>(Coeff_buf.data.dptr), stream, partial_sums);););
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
    CompType tokens_per_expert_i = static_cast<CompType>(tokens_per_expert[i]);
    CompType grad_aux_loss_value = static_cast<CompType>(grad_aux_loss[0]);
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

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_moe_aux_loss_forward(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                     int total_num_tokens, int num_experts, int num_rows,
                                     int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                     NVTETensor Coeff_buf, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_router::fused_moe_aux_loss_forward(
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert), total_num_tokens,
      num_experts, num_rows, num_cols, topk, coeff, *convertNVTETensorCheck(aux_loss),
      *convertNVTETensorCheck(Coeff_buf), stream);
}

void nvte_fused_moe_aux_loss_forward_graph_safe(const NVTETensor probs,
                                                const NVTETensor tokens_per_expert,
                                                const NVTETensor total_num_tokens, int num_experts,
                                                int num_rows, int num_cols, int topk, float coeff,
                                                NVTETensor aux_loss, NVTETensor Coeff_buf,
                                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward_graph_safe);
  using namespace transformer_engine;
  fused_router::fused_moe_aux_loss_forward_graph_safe(
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert),
      *convertNVTETensorCheck(total_num_tokens), num_experts, num_rows, num_cols, topk, coeff,
      *convertNVTETensorCheck(aux_loss), *convertNVTETensorCheck(Coeff_buf), stream);
}

size_t nvte_get_moe_aux_loss_workspace_size() {
  NVTE_API_CALL(nvte_get_moe_aux_loss_workspace_size);
  return transformer_engine::fused_router::kAuxLossMaxReductionBlocks * sizeof(float);
}

void nvte_fused_moe_aux_loss_forward_v2(const NVTETensor probs, const NVTETensor tokens_per_expert,
                                        int total_num_tokens, int num_experts, int num_rows,
                                        int num_cols, int topk, float coeff, NVTETensor aux_loss,
                                        NVTETensor Const_buf, NVTETensor workspace,
                                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward_v2);
  using namespace transformer_engine;
  auto& scratch = *convertNVTETensorCheck(workspace);
  fused_router::check_aux_loss_workspace(scratch, num_rows);
  auto& constants = *convertNVTETensorCheck(Const_buf);
  fused_router::check_aux_loss_const_buffer(constants);
  fused_router::fused_moe_aux_loss_forward<true>(
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert), total_num_tokens,
      num_experts, num_rows, num_cols, topk, coeff, *convertNVTETensorCheck(aux_loss), constants,
      stream, reinterpret_cast<float*>(scratch.data.dptr));
}

void nvte_fused_moe_aux_loss_forward_graph_safe_v2(
    const NVTETensor probs, const NVTETensor tokens_per_expert, const NVTETensor total_num_tokens,
    int num_experts, int num_rows, int num_cols, int topk, float coeff, NVTETensor aux_loss,
    NVTETensor Const_buf, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_forward_graph_safe_v2);
  using namespace transformer_engine;
  auto& scratch = *convertNVTETensorCheck(workspace);
  fused_router::check_aux_loss_workspace(scratch, num_rows);
  auto& constants = *convertNVTETensorCheck(Const_buf);
  fused_router::check_aux_loss_const_buffer(constants);
  fused_router::fused_moe_aux_loss_forward_graph_safe<true>(
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(tokens_per_expert),
      *convertNVTETensorCheck(total_num_tokens), num_experts, num_rows, num_cols, topk, coeff,
      *convertNVTETensorCheck(aux_loss), constants, stream,
      reinterpret_cast<float*>(scratch.data.dptr));
}

void nvte_fused_moe_aux_loss_backward(const NVTETensor Const_buf,
                                      const NVTETensor tokens_per_expert, int num_rows,
                                      int num_cols, NVTETensor grad_aux_loss, NVTETensor grad_probs,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_router::fused_moe_aux_loss_backward(*convertNVTETensorCheck(Const_buf),
                                            *convertNVTETensorCheck(tokens_per_expert), num_rows,
                                            num_cols, *convertNVTETensorCheck(grad_aux_loss),
                                            *convertNVTETensorCheck(grad_probs), stream);
}
