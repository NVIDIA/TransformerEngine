/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file quantize_fp8.cuh
 *  \brief CUDA kernels to quantize to FP8.
 */

#ifndef TRANSFORMER_ENGINE_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_QUANTIZE_FP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../common.h"
#include "../../transpose/cast_transpose.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../util/vectorized_pointwise.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace quantize_2D_kernel {

constexpr size_t FP8_CHUNK_DIM_Y = 128;
constexpr size_t FP8_CHUNK_DIM_X = 128;
constexpr size_t FP8_THREADS_PER_CHUNK = 128;
constexpr size_t FP8_BUFFERS_NUM = 2;
constexpr size_t FP8_PREFETCH_BUFFERS_NUM = 1;
static_assert(FP8_PREFETCH_BUFFERS_NUM < FP8_BUFFERS_NUM);

constexpr size_t FP8_BUFFER_DIM_Y = 16;
constexpr size_t FP8_BUFFER_DIM_X = FP8_CHUNK_DIM_X;  // 128
constexpr size_t FP8_SHMEM_DIM_Y = FP8_BUFFER_DIM_Y;  // 16
constexpr size_t FP8_SHMEM_DIM_X = FP8_BUFFER_DIM_X;  // 128

constexpr size_t FP8_BUFF_STAGES_NUM = FP8_BUFFER_DIM_Y;               //  16
constexpr size_t FP8_ITERATIONS = FP8_CHUNK_DIM_Y / FP8_BUFFER_DIM_Y;  //   8 = 128 / 16
static_assert(FP8_ITERATIONS >= FP8_PREFETCH_BUFFERS_NUM);

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &),
          typename IType, typename OType>
__global__ void __launch_bounds__(FP8_THREADS_PER_CHUNK)
    cast_fp8_2D_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                       const __grid_constant__ CUtensorMap tensor_map_act_input,
                       const __grid_constant__ CUtensorMap tensor_map_output,
                       float *const dbias_workspace, float *const amax_ptr,
                       float *const scale_inv_ptr, const float *const scale_ptr, const size_t rows,
                       const size_t cols) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const size_t block_offset_Y = blockIdx.y * FP8_CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * FP8_CHUNK_DIM_X;

  const size_t tid_Y = threadIdx.x / FP8_THREADS_PER_CHUNK;
  const size_t tid_X = threadIdx.x % FP8_THREADS_PER_CHUNK;

  const size_t thread_offset_Y = tid_Y;
  const size_t thread_offset_X = tid_X;

  const size_t dbias_offset_Y = blockIdx.y + tid_Y;
  const size_t my_column = blockIdx.x * FP8_CHUNK_DIM_X + thread_offset_X;
  const bool col_out_of_bounds = my_column >= cols;
  const size_t dbias_stride = cols;

  float partial_dbias = 0.f;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(TMA_SHMEM_ALIGNMENT)
      IType in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT)
      IType act_in_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT)
      OType out_sh[FP8_BUFFERS_NUM][FP8_SHMEM_DIM_Y][FP8_SHMEM_DIM_X];

  constexpr size_t shmem_buff_size = sizeof(in_sh) / FP8_BUFFERS_NUM;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[FP8_ITERATIONS];

  initialize_barriers<FP8_ITERATIONS, FP8_THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  const size_t chunk_offset_Y = block_offset_Y;
  const size_t chunk_offset_X = block_offset_X;

#pragma unroll
  for (int prefetch_buff = 0; prefetch_buff < FP8_PREFETCH_BUFFERS_NUM; ++prefetch_buff) {
    const size_t chunk_stage_offset_Y = chunk_offset_Y + prefetch_buff * FP8_BUFFER_DIM_Y;
    const size_t chunk_stage_offset_X = chunk_offset_X;
    if constexpr (IS_DACT) {
      copy_2d_to_sharedx2(&in_sh[prefetch_buff], &tensor_map_input, chunk_stage_offset_X,
                          chunk_stage_offset_Y, &act_in_sh[prefetch_buff], &tensor_map_act_input,
                          chunk_stage_offset_X, chunk_stage_offset_Y, shmem_buff_size,
                          &mbar[prefetch_buff], is_master_thread);
    } else {
      copy_2d_to_shared(&in_sh[prefetch_buff], &tensor_map_input, chunk_stage_offset_X,
                        chunk_stage_offset_Y, shmem_buff_size, &mbar[prefetch_buff],
                        is_master_thread);
    }
  }

#pragma unroll
  for (int iter = 0; iter < FP8_ITERATIONS; ++iter) {
    const size_t buff = iter % FP8_BUFFERS_NUM;
    const size_t next_iter = iter + FP8_PREFETCH_BUFFERS_NUM;
    const size_t row_base = block_offset_Y + iter * FP8_BUFFER_DIM_Y;
    if (next_iter < FP8_ITERATIONS) {
      const size_t next_buff = next_iter % FP8_BUFFERS_NUM;
      const size_t chunk_it_offset_y = chunk_offset_Y + next_iter * FP8_BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;
      if constexpr (IS_DACT) {
        copy_2d_to_sharedx2(&in_sh[next_buff], &tensor_map_input, chunk_it_offset_x,
                            chunk_it_offset_y, &act_in_sh[next_buff], &tensor_map_act_input,
                            chunk_it_offset_x, chunk_it_offset_y, shmem_buff_size, &mbar[next_iter],
                            is_master_thread);
      } else {
        copy_2d_to_shared(&in_sh[next_buff], &tensor_map_input, chunk_it_offset_x,
                          chunk_it_offset_y, shmem_buff_size, &mbar[next_iter], is_master_thread);
      }
    }

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

#pragma unroll
    for (int stage = 0; stage < FP8_BUFF_STAGES_NUM; ++stage) {
      const size_t stage_offset_Y = stage;
      const size_t shmem_offset_y = thread_offset_Y + stage_offset_Y;
      const size_t shmem_offset_x = thread_offset_X;
      const size_t row = row_base + shmem_offset_y;
      const bool row_out_of_bounds = row >= rows;
      const bool out_of_bounds = col_out_of_bounds || row_out_of_bounds;

      float elt = static_cast<float>(in_sh[buff][shmem_offset_y][shmem_offset_x]);
      if constexpr (IS_DACT) {
        float act_in_elt = static_cast<float>(act_in_sh[buff][shmem_offset_y][shmem_offset_x]);
        elt *= OP(act_in_elt, {});
      }
      if constexpr (IS_DBIAS) {
        if constexpr (IS_DACT) {
          if (!out_of_bounds) {
            partial_dbias += elt;
          }
        } else {
          // If no activation, elt is 0 so we can safely do this
          partial_dbias += elt;
        }
      }
      __builtin_assume(amax >= 0);
      if (IS_DACT) {
        if (!out_of_bounds) {
          amax = fmaxf(amax, fabsf(elt));
        }
      } else {
        // If no activation, elt is 0 so we can safely do this
        amax = fmaxf(amax, fabsf(elt));
      }
      out_sh[buff][shmem_offset_y][shmem_offset_x] = static_cast<OType>(elt * scale);
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t chunk_it_offset_y = chunk_offset_Y + iter * FP8_BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
          chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<FP8_PREFETCH_BUFFERS_NUM>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  parity ^= 1;

  if constexpr (IS_DBIAS) {
    const size_t dbias_offset_X = my_column;
    const size_t dbias_offset = dbias_offset_Y * dbias_stride + dbias_offset_X;
    if (!col_out_of_bounds) {
      dbias_workspace[dbias_offset] = partial_dbias;
    }
  }

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<FP8_THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  destroy_barriers<FP8_ITERATIONS>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace quantize_2D_kernel

namespace quantize_1D_kernel {
using namespace quantize_2D_kernel;

constexpr size_t CHUNKS_PER_BLOCK = 128;
constexpr size_t THREADS_PER_BLOCK = FP8_THREADS_PER_CHUNK;
constexpr size_t CHUNK_SIZE = THREADS_PER_BLOCK;
constexpr size_t ELEMS_PER_BLOCK = CHUNKS_PER_BLOCK * CHUNK_SIZE;
constexpr size_t CHUNKS_PER_ITERATION = 32;
constexpr size_t SHMEM_DIM = CHUNKS_PER_ITERATION * CHUNK_SIZE;
constexpr size_t ITERATIONS = CHUNKS_PER_BLOCK / CHUNKS_PER_ITERATION;
constexpr size_t SHMEM_BUFFERS = 2;
static_assert(CHUNKS_PER_BLOCK % CHUNKS_PER_ITERATION == 0);

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    cast_fp8_1D_kernel(const IType *input_ptr, OType *output_ptr, float *const amax_ptr,
                       float *const scale_inv_ptr, const float *const scale_ptr, const size_t N) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const size_t block_offset = blockIdx.x * ELEMS_PER_BLOCK;
  const IType *input = input_ptr + block_offset;
  OType *output = output_ptr + block_offset;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  // The destination shared memory buffer of a bulk tensor operation should be 128-byte aligned
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) IType in_sh[SHMEM_BUFFERS][SHMEM_DIM];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) OType out_sh[SHMEM_BUFFERS][SHMEM_DIM];

  constexpr size_t transaction_size_IN = sizeof(in_sh) / SHMEM_BUFFERS;
  constexpr size_t transaction_size_OUT = sizeof(out_sh) / SHMEM_BUFFERS;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_BLOCK>(mbar, is_master_thread);

  int parity = 0;

  copy_1d_to_shared(&(in_sh[0]), input, transaction_size_IN, &(mbar[0]), is_master_thread);

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const size_t buff = iter % SHMEM_BUFFERS;
    const size_t it_offset = iter * SHMEM_DIM;

    const size_t next_iter = iter + 1;
    const size_t next_buff = next_iter % SHMEM_BUFFERS;
    const size_t next_iter_offset = next_iter * SHMEM_DIM;

    if (next_iter < ITERATIONS) {
      copy_1d_to_shared(&(in_sh[next_buff]), input + next_iter_offset, transaction_size_IN,
                        &(mbar[next_iter]), is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

#pragma unroll
    for (int chunk = 0; chunk < CHUNKS_PER_ITERATION; ++chunk) {
      const size_t shmem_offset = chunk * CHUNK_SIZE + threadIdx.x;
      float elt = static_cast<float>(in_sh[buff][shmem_offset]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      __builtin_assume(amax >= 0);
      amax = fmaxf(amax, fabsf(elt));
      out_sh[buff][shmem_offset] = static_cast<OType>(elt * scale);
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      ptx::cp_async_bulk_tensor_1d_shared_to_global(
          reinterpret_cast<uint64_t *>(output + it_offset),
          reinterpret_cast<uint64_t *>(&out_sh[buff]), transaction_size_OUT);

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_BLOCK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  destroy_barriers<ITERATIONS>(mbar, is_master_thread);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace quantize_1D_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_1D(const Tensor &input, Tensor *output, cudaStream_t stream) {
  using namespace quantize_1D_kernel;
  const size_t N = product(input.data.shape);

  const bool isFullTile = (N % ELEMS_PER_BLOCK == 0);
  NVTE_CHECK(isFullTile, "Only full tiles are supported.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  const size_t chunks = DIVUP(N, CHUNK_SIZE);
  const size_t blocks = DIVUP(chunks, CHUNKS_PER_BLOCK);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  const float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(THREADS_PER_BLOCK);
  const dim3 grid(blocks);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          const IType *input_ptr = reinterpret_cast<const IType *>(input.data.dptr);
          OType *output_ptr = reinterpret_cast<OType *>(output->data.dptr);

          cast_fp8_1D_kernel<IS_ACT, ParamOP, OP, IType, OType><<<grid, block, 0, stream>>>(
              input_ptr, output_ptr, amax_ptr, scale_inv_ptr, scale_ptr, N););  // NOLINT(*)
  );                                                                            // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool IS_DBIAS, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void quantize_2D(const Tensor &input, const Tensor *act_input, Tensor *output, Tensor *dbias,
                 Tensor *workspace, cudaStream_t stream) {
  using namespace quantize_2D_kernel;
  checkCuDriverContext(stream);

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, FP8_CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, FP8_CHUNK_DIM_X);
  const size_t blocks_Y = chunks_Y;
  const size_t blocks_X = chunks_X;

  const size_t dbias_rows = blocks_Y;
  const size_t dbias_cols = cols;

  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale_inv.dptr != nullptr, "Scaling tensor must be allocated");

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.data.dtype, "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{cols}, "Wrong shape of DBias.");
    NVTE_CHECK(workspace != nullptr, "Workspace must be a tensor.");

    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {dbias_rows, dbias_cols};
      workspace->data.dtype = DType::kFloat32;
      return;
    }
  }
  float *const workspace_ptr = IS_DBIAS ? reinterpret_cast<float *>(workspace->data.dptr) : nullptr;
  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block(FP8_THREADS_PER_CHUNK);
  const dim3 grid(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->data.dtype, OType,

          alignas(64) CUtensorMap tensor_map_input{};
          alignas(64) CUtensorMap tensor_map_act_input{};
          alignas(64) CUtensorMap tensor_map_output{};

          create_2D_tensor_map(tensor_map_input, input.data, rows, cols, FP8_SHMEM_DIM_Y,
                               FP8_SHMEM_DIM_X, cols, 0, typeToNumBits(input.data.dtype));

          if constexpr (IS_DACT) {
            create_2D_tensor_map(tensor_map_act_input, act_input->data, rows, cols, FP8_SHMEM_DIM_Y,
                                 FP8_SHMEM_DIM_X, cols, 0, typeToNumBits(input.data.dtype));
          }

          create_2D_tensor_map(tensor_map_output, output->data, rows, cols, FP8_SHMEM_DIM_Y,
                               FP8_SHMEM_DIM_X, cols, 0, typeToNumBits(output->data.dtype));

          cast_fp8_2D_kernel<IS_DBIAS, IS_DACT, ParamOP, OP, IType, OType>
          <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_act_input, tensor_map_output,
                                       workspace_ptr, amax_ptr, scale_inv_ptr, scale_ptr, rows,
                                       cols);
          NVTE_CHECK_CUDA(cudaGetLastError());

          if constexpr (IS_DBIAS) {
            common::reduce_dbias<IType>(workspace_ptr, dbias, dbias_rows, dbias_cols, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

namespace detail {
using Empty = transformer_engine::Empty;
__device__ inline float identity(float value, const Empty &) { return value; }
}  // namespace detail

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
void CastVectorizedUnaryKernelLauncher(const Tensor &input, const Tensor *noop, Tensor *output,
                                       cudaStream_t stream) {
  constexpr float (*UnaryOP)(float, const ParamOP &) = (OP == nullptr) ? detail::identity : OP;
  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,
          if (!is_fp8_dtype(output->data.dtype) || is_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            VectorizedUnaryKernelLauncher<nvec, ParamOP, UnaryOP>(
                reinterpret_cast<const IType *>(input.data.dptr),
                reinterpret_cast<const fp32 *>(noop->data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
void CastVectorizedUnaryGradKernelLauncher(const Tensor &grad, const Tensor *input, Tensor *output,
                                           cudaStream_t stream) {
  constexpr float (*UnaryOP)(float, const ParamOP &) = (OP == nullptr) ? detail::identity : OP;
  const size_t N = product(input->data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input->data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->data.dtype, OType,
          if (!is_fp8_dtype(output->data.dtype) || is_tensor_scaling(output->scaling_mode)) {
            constexpr int nvec = 32 / sizeof(IType);
            VectorizedUnaryGradKernelLauncher<nvec, ParamOP, UnaryOP>(
                reinterpret_cast<const IType *>(grad.data.dptr),
                reinterpret_cast<const IType *>(input->data.dptr),
                reinterpret_cast<OType *>(output->data.dptr),
                reinterpret_cast<const fp32 *>(output->scale.dptr),
                reinterpret_cast<fp32 *>(output->amax.dptr),
                reinterpret_cast<fp32 *>(output->scale_inv.dptr), N, {}, stream);
          } else {
            NVTE_ERROR("Not implemented scaling mode: " + to_string(output->scaling_mode) + ".");
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void quantize(const Tensor &input, const Tensor *act_input, const Tensor *noop, Tensor *output,
              Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  using namespace quantize_1D_kernel;
  CheckNoopTensor(*noop, "cast_noop");
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias != nullptr);
    CheckOutputTensor(*dbias, "dbias");
  }
  if constexpr (IS_DACT) {
    NVTE_CHECK(act_input != nullptr);
    CheckInputTensor(*act_input, "activation_input");
    NVTE_CHECK(input.dtype() == act_input->dtype(), "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input->data.shape, "Shapes of both inputs must match.");
  }

  NVTE_CHECK(!is_fp8_dtype(input.dtype()), "Input must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  // Supported by the Arch >= 10.0
  if (is_supported_by_CC_100()) {
    if (!IS_DBIAS && !IS_DACT) {
      if (common::full_tile_1D_tensor(output, ELEMS_PER_BLOCK) && is_fp8_dtype(output->dtype()) &&
          is_aligned_tensor_data(input, TMA_GMEM_ALIGNMENT) &&
          is_aligned_tensor_data(*output, TMA_GMEM_ALIGNMENT)) {
        // Aligned AND FP8
        quantize_1D<IS_ACT, ParamOP, OP>(input, output, stream);
      } else {
        // Unaligned
        CastVectorizedUnaryKernelLauncher<ParamOP, OP>(input, noop, output, stream);
      }
    } else if (!IS_DBIAS && IS_DACT) {
      if (common::dimensions_supported_by_TMA(output) && is_fp8_dtype(output->dtype()) &&
          is_aligned_tensor_data(input, TMA_GMEM_ALIGNMENT) &&
          is_aligned_tensor_data(*output, TMA_GMEM_ALIGNMENT) &&
          is_aligned_tensor_data(*act_input, TMA_GMEM_ALIGNMENT)) {
        // Aligned AND FP8 (+dAct)
        quantize_2D<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias, workspace,
                                                    stream);
      } else {
        // Unaligned
        CastVectorizedUnaryGradKernelLauncher<ParamOP, OP>(input, act_input, output, stream);
      }
    } else {
      quantize_2D<IS_DBIAS, IS_DACT, ParamOP, OP>(input, act_input, output, dbias, workspace,
                                                  stream);
    }
  } else {
    if (IS_DBIAS) {
      // zhongboz: should we just ignore IS_ACT here?
      NVTE_ERROR("Not implemented scaling mode or fusion: " + to_string(output->scaling_mode) +
                 " or IS_DBIAS=true" + " on GPU with compute capability < 10.0.");
    }
    if (!IS_DACT) {
      CastVectorizedUnaryKernelLauncher<ParamOP, OP>(input, noop, output, stream);
    } else {
      CastVectorizedUnaryGradKernelLauncher<ParamOP, OP>(input, act_input, output, stream);
    }
  }
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_QUANTIZE_FP8_CUH_
