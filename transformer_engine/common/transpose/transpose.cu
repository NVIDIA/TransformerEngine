/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transpose.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>
#include "../utils.cuh"
#include "../common.h"

namespace transformer_engine {

template <int nvec_in, int nvec_out, typename IVec, typename OVec>
inline __device__ void transpose_regs(const IVec (&in)[nvec_out],
                                      OVec (&out_trans)[nvec_in]) {
    using T = typename OVec::type;
    using U = typename IVec::type;
    static_assert(std::is_same<T, U>::value, "Types of input and output to transpose must match!");
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
#pragma unroll
        for (unsigned int j = 0; j < nvec_in; ++j) {
            out_trans[j].data.elt[i] = in[i].data.elt[j];  // thread tile transpose
        }
    }
}


// STUFF TO TUNE
constexpr unsigned int n_warps_per_tile = 4;
constexpr int desired_load_size = 8;
constexpr int desired_store_size = 8;

constexpr unsigned int max_threads_per_block = 256;
static_assert(n_warps_per_tile * THREADS_PER_WARP <= max_threads_per_block);
constexpr unsigned int cast_transpose_num_threads = n_warps_per_tile * THREADS_PER_WARP;

template <int nvec_in, int nvec_out, typename CType, typename IType, typename OType>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
transpose_kernel(const IType * const input,
                      OType * const output_t,
                      const size_t row_length,
                      const size_t num_rows,
                      const size_t num_tiles) {
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
    warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = input + (tile_id_x * nvec_in +
                                               tile_id_y * row_length * nvec_out) *
                                             THREADS_PER_WARP;
  OType * const my_output_t_tile = output_t + (tile_id_y * nvec_out +
                                               tile_id_x * num_rows * nvec_in) *
                                             THREADS_PER_WARP;
  OVec * const my_scratch = reinterpret_cast<OVec*>(scratch) +
                            (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                            (THREADS_PER_WARP + 1);

  IVec in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                         THREADS_PER_WARP;
#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        in[current_in][j].load_from(my_input_tile,
                                    current_stride + my_place_in + stride * (nvec_out + j));
      }
    }
    OVec out_trans[nvec_in];  // NOLINT(*)
    transpose_regs<nvec_in, nvec_out, IVec, OVec>(in[current_in ^ 1], out_trans);
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space[i][j].data.vec = out_trans[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                  j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
               THREADS_PER_WARP;
    current_stride = i * output_stride +
                     warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }
}

template <int nvec_in, int nvec_out, typename CType, typename IType, typename OType>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
transpose_kernel_notaligned(const IType * const input,
                            OType * const output_t,
                            const size_t row_length,
                            const size_t num_rows,
                            const size_t num_tiles) {
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = (row_length + nvec_in * THREADS_PER_WARP - 1) /
                             (nvec_in * THREADS_PER_WARP);
  const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                         warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType * const my_input_tile = input + (tile_id_x * nvec_in +
                                               tile_id_y * row_length * nvec_out) *
                                             THREADS_PER_WARP;
  OType * const my_output_t_tile = output_t + (tile_id_y * nvec_out +
                                               tile_id_x * num_rows * nvec_in) *
                                             THREADS_PER_WARP;
  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length = row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_length_rest;
  const unsigned int tile_height = row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                      : row_height_rest;

  OVec * const my_scratch = reinterpret_cast<OVec*>(scratch) +
                            (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                            (THREADS_PER_WARP + 1);

  IVec in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                           warp_id_in_tile * n_iterations) %
                         THREADS_PER_WARP;
  {
    const bool valid_load = my_place < tile_length &&
                            warp_id_in_tile * n_iterations < tile_height;
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
      if (valid_load) {
        in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
      } else {
        in[0][i].clear();
      }
    }
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
      const bool valid_load = my_place_in < tile_length &&
                              warp_id_in_tile * n_iterations + i + 1 < tile_height;
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        if (valid_load) {
          in[current_in][j].load_from(my_input_tile,
                                      current_stride + my_place_in + stride * (nvec_out + j));
        } else {
          in[current_in][j].clear();
        }
      }
    }
    OVec out_trans[nvec_in];  // NOLINT(*)
    transpose_regs<nvec_in, nvec_out, IVec, OVec>(in[current_in ^ 1], out_trans);
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space[i][j].data.vec = out_trans[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                  j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
               THREADS_PER_WARP;
    current_stride = i * output_stride +
                     warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; warp_id_in_tile * n_iterations + j < tile_length; ++j) {
      const bool valid_store = my_place < tile_height;
      if (valid_store) {
        my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                                current_stride + my_place);
      }
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }
}

void transpose(const Tensor &input,
               Tensor *transposed_output,
               cudaStream_t stream) {
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(transposed_output->data.shape.size() == 2, "Output must have 2 dimensions.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  NVTE_CHECK(transposed_output->data.shape[0] == row_length, "Wrong dimension of output.");
  NVTE_CHECK(transposed_output->data.shape[1] == num_rows, "Wrong dimension of output.");

  NVTE_CHECK(input.data.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(transposed_output->data.dptr != nullptr, "Output is not allocated.");
  NVTE_CHECK(input.data.dtype == transposed_output->data.dtype,
             "Input and output type must match.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(input.data.dtype, Type,
    constexpr int type_size = sizeof(Type);
    constexpr int nvec_in = desired_load_size / type_size;
    constexpr int nvec_out = desired_store_size / type_size;

    NVTE_CHECK(row_length % nvec_in  == 0, "Unsupported shape.");
    NVTE_CHECK(num_rows   % nvec_out == 0, "Unsupported shape.");

    const size_t n_tiles = DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
                           DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
    const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
    const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

    const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                           num_rows % (nvec_out * THREADS_PER_WARP) == 0;

    if (full_tile) {
      cudaFuncSetAttribute(transpose_kernel<nvec_in, nvec_out, fp32, Type, Type>,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           100);
      transpose_kernel<nvec_in, nvec_out, fp32, Type, Type>
        <<<n_blocks,
           cast_transpose_num_threads,
           cast_transpose_num_threads / n_warps_per_tile *
             (THREADS_PER_WARP + 1) * sizeof(Vec<Type, nvec_out>),
           stream>>>(
            reinterpret_cast<const Type *>(input.data.dptr),
            reinterpret_cast<Type *>(transposed_output->data.dptr),
            row_length, num_rows, n_tiles);
    } else {
      cudaFuncSetAttribute(transpose_kernel_notaligned<nvec_in, nvec_out, fp32, Type, Type>,
                           cudaFuncAttributePreferredSharedMemoryCarveout,
                           100);
      transpose_kernel_notaligned<nvec_in, nvec_out, fp32, Type, Type>
        <<<n_blocks,
           cast_transpose_num_threads,
           cast_transpose_num_threads / n_warps_per_tile *
             (THREADS_PER_WARP + 1) * sizeof(Vec<Type, nvec_out>),
           stream>>>(
            reinterpret_cast<const Type *>(input.data.dptr),
            reinterpret_cast<Type *>(transposed_output->data.dptr),
            row_length, num_rows, n_tiles);
    }
  );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_transpose(const NVTETensor input,
                    NVTETensor transposed_output,
                    cudaStream_t stream) {
  using namespace transformer_engine;
  transpose(*reinterpret_cast<const Tensor*>(input),
            reinterpret_cast<Tensor*>(transposed_output),
            stream);
}
