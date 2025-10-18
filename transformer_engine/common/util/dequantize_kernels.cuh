/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_kernels.cuh
 *  \brief CUDA kernels to cast from MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/activation.h"
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/transpose.h"

namespace transformer_engine {

namespace dequantization {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t BUFFERS_NUM = 2;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t BUFFER_DIM_Y = 16;           // only 32 is supported
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 16
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  //  8 = 128 / 16
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;                     //  128
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                       //    8 = 128 / 16
static_assert(ITERATIONS >= 1);

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxfp8_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                            const __grid_constant__ CUtensorMap tensor_map_output,
                            const e8m0_t *const scales_ptr, const size_t rows, const size_t cols,
                            const size_t scales_stride) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //    4 = 128 / 32

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //    4 = 128 / 32
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);  // 2 = 32 / 16

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  // const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) IType in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];
  __shared__ alignas(TMA_SHMEM_ALIGNMENT) OType out_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
  constexpr int iteration_zero = 0;
  constexpr int buffer_zero = 0;
  if (is_master_thread) {
    const int chunk_stage_offset_Y = chunk_offset_Y;
    const int chunk_stage_offset_X = chunk_offset_X;
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(&in_sh[buffer_zero]),
        reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
        chunk_stage_offset_Y, &mbar[iteration_zero]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[iteration_zero], transaction_size);

  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[iteration_zero]);
  }

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const int buff = iter % BUFFERS_NUM;
    const int next_iter = iter + 1;
    if (next_iter < ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_iter % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_iter * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
            chunk_it_offset_y, &mbar[next_iter]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_iter]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

    const int scale_offset_Y =
        USE_ROWWISE_SCALING ? (scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y + tid_rowwise_Y)
                            : (scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y);

    const int scale_offset_X =
        USE_ROWWISE_SCALING
            ? (scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE)
            : (scales_colwise_chunk_offset_X + tid_colwise_X);

    const int scale_idx = scale_offset_Y * scales_stride + scale_offset_X;
    const e8m0_t biased_exponent = scales_ptr[scale_idx];
    const float block_scale = ptx::exp2f(biased_exponent);

    if constexpr (USE_ROWWISE_SCALING) {
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;

      const int shmem_offset_y = thread_offset_Y;
      const int shmem_offset_x = thread_offset_X_rowwise;
      in.load_from(&in_sh[buff][shmem_offset_y][shmem_offset_x]);

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD; ++j) {
        out.data.elt[j] = static_cast<OType>(block_scale * static_cast<float>(in.data.elt[j]));
      }
      out.store_to(&out_sh[buff][shmem_offset_y][shmem_offset_x]);
    } else {
#pragma unroll
      for (int i = 0; i < BUFFER_DIM_Y; ++i) {
        const float elt = static_cast<float>(in_sh[buff][i][tid_colwise_X]);
        out_sh[buff][i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
      }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
          chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  parity ^= 1;

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

void fp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->data.dtype, OType,

          constexpr int nvec = 32 / sizeof(OType);
          detail::DequantizeParam p;
          p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
          VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
              reinterpret_cast<const IType *>(input.data.dptr), nullptr,
              reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, nullptr, N, p,
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

void mxfp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  bool use_rowwise_scaling = input.has_data();
  bool use_colwise_scaling = input.has_columnwise_data();
  checkCuDriverContext(stream);

  const auto &input_shape = input.data.shape;
  NVTE_CHECK(input_shape.size() >= 2, "Input must have at least 2 dimensions.");

  if (use_rowwise_scaling) {
    NVTE_CHECK(input.has_data(), "Cannot dequantize tensor without rowwise data.");
    NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  }

  if (use_colwise_scaling) {
    NVTE_CHECK(input.has_columnwise_data(), "Cannot dequantize tensor without columnwise data.");
    NVTE_CHECK(is_fp8_dtype(input.columnwise_data.dtype), "Input must have FP8 type.");
  }

  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  // TODO: Make more general
  const size_t scale_dim_X_rowwise = use_rowwise_scaling ? 32 : 1;
  const size_t scale_dim_Y_colwise = use_colwise_scaling ? 32 : 1;

  const size_t rows = input.flat_first_dim();
  const size_t cols = input.flat_last_dim();
  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);

  const size_t unpadded_scales_Y_rowwise = rows;
  const size_t unpadded_scales_X_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t unpadded_scales_Y_colwise = DIVUP(rows, scale_dim_Y_colwise);
  const size_t unpadded_scales_X_colwise = cols;

  const size_t scales_Y_rowwise =
      DIVUP(unpadded_scales_Y_rowwise, scale_tensor_alignment_Y_rowwise) *
      scale_tensor_alignment_Y_rowwise;
  const size_t scales_X_rowwise =
      DIVUP(unpadded_scales_X_rowwise, scale_tensor_alignment_X_rowwise) *
      scale_tensor_alignment_X_rowwise;
  const size_t scales_Y_colwise =
      DIVUP(unpadded_scales_Y_colwise, scale_tensor_alignment_Y_colwise) *
      scale_tensor_alignment_Y_colwise;
  const size_t scales_X_colwise =
      DIVUP(unpadded_scales_X_colwise, scale_tensor_alignment_X_colwise) *
      scale_tensor_alignment_X_colwise;

  const e8m0_t *const scales_ptr =
      use_rowwise_scaling ? reinterpret_cast<e8m0_t *>(input.scale_inv.dptr)
                          : reinterpret_cast<e8m0_t *>(input.columnwise_scale_inv.dptr);

  const size_t scales_stride = use_rowwise_scaling ? scales_X_rowwise : scales_X_colwise;

  const SimpleTensor &input_data = use_rowwise_scaling ? input.data : input.columnwise_data;

  const dim3 block(THREADS_PER_CHUNK);
  const dim3 grid(chunks_X, chunks_Y);

  TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
              input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
                  output->dtype(), OType,

                  alignas(64) CUtensorMap tensor_map_input{};
                  alignas(64) CUtensorMap tensor_map_output{};

                  create_2D_tensor_map(tensor_map_input, input_data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, cols, 0, typeToNumBits(input.dtype()));
                  create_2D_tensor_map(tensor_map_output, output->data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, cols, 0, typeToNumBits(output->dtype()));

                  dequantize_mxfp8_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X>
                  <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_output, scales_ptr,
                                               rows, cols, scales_stride););  // NOLINT(*)
          );                                                                  // NOLINT(*)
      );                                                                      // NOLINT(*)
  );                                                                          // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
}

#if CUDA_VERSION >= 12080
template <typename OType>
__global__ void __launch_bounds__(512)
    dequantize_fp4_kernel(const void *const input, OType *output, const fp8e4m3 *const scales,
                          const float *const tensor_amax, const size_t N, const size_t M,
                          const size_t scale_stride) {
  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t x = thread_idx % M;
  const size_t y = thread_idx / M;

  union fp4vec {
    uint64_t vec;
    fp4e2m1x4 small_vec[4];
  };
  using OVec = Vec<OType, 4>;
  const uint64_t *const input_vectorized = reinterpret_cast<const uint64_t *>(input);
  OVec *output_vec = reinterpret_cast<OVec *>(output);

  const size_t my_index = x + y * M;
  const size_t my_scale_index = x + y * scale_stride;
  const size_t my_output_index = (x + y * M) * 4;
  fp4vec value;
  value.vec = input_vectorized[my_index];
  fp8e4m3 scale = scales[my_scale_index];
  float amax = *tensor_amax;
  constexpr float factor_inv = 1.0 / (6.0 * 448.0);
  float final_scale = static_cast<float>(scale) * amax * factor_inv;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float4 current = static_cast<float4>(value.small_vec[i]);
    OVec out;
    out.data.elt[0] = static_cast<OType>(current.x * final_scale);
    out.data.elt[1] = static_cast<OType>(current.y * final_scale);
    out.data.elt[2] = static_cast<OType>(current.z * final_scale);
    out.data.elt[3] = static_cast<OType>(current.w * final_scale);
    output_vec[my_output_index + i] = out;
  }
}
#endif  // CUDA_VERSION

void fp4_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
#if CUDA_VERSION >= 12080
  CheckInputTensor(input, "input");
  CheckOutputTensor(*output, "output");
  NVTE_CHECK(input.data.dtype == DType::kFloat4E2M1, "Input must have FP4 type.");
  NVTE_CHECK(is_high_precision_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  constexpr int FP4_BLOCK_SIZE = 16;
  const size_t N = input.flat_first_dim();
  const size_t M = input.flat_last_dim();

  NVTE_CHECK(M % FP4_BLOCK_SIZE == 0, "Last dimension of FP4 tensors needs to be divisible by ",
             FP4_BLOCK_SIZE, ", but got ", input.data.shape, ".");

  const size_t Mread = M / FP4_BLOCK_SIZE;
  const size_t total = N * Mread;
  const size_t threads = 512;
  const size_t blocks = DIVUP(total, threads);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      output->data.dtype, OType,

      dequantize_fp4_kernel<<<blocks, threads, 0, stream>>>(
          input.data.dptr, reinterpret_cast<OType *>(output->data.dptr),
          reinterpret_cast<fp8e4m3 *>(input.scale_inv.dptr),
          reinterpret_cast<float *>(input.amax.dptr), N, Mread,
          input.scale_inv.shape.back()););  // NOLINT(*)
  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("CUDA 12.8 or higher is needed for FP4 calculation!");
#endif  // CUDA_VERSION >= 12080
}

}  // namespace dequantization

namespace detail {

void dequantize_helper(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  switch (input.scaling_mode) {
    case NVTE_DELAYED_TENSOR_SCALING: {
      dequantization::fp8_dequantize(input, output, stream);
      break;
    }
    case NVTE_MXFP8_1D_SCALING: {
      if (is_supported_by_CC_100()) {
        dequantization::mxfp8_dequantize(input, output, stream);
      } else {
        NVTE_ERROR("MXFP8 Dequantization is NOT supported by architectures < 10.0");
      }
      break;
    }
    case NVTE_NVFP4_1D_SCALING: {
      dequantization::fp4_dequantize(input, output, stream);
      break;
    }
    default:
      NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

}  // namespace detail

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_
