/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <iostream>
#include <type_traits>

#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {

template <int nvec_in, int nvec_out, typename IVec, typename OVec, typename CVec, typename CType>
inline __device__ void transpose_regs_partial_dbias(const IVec (&in)[nvec_out],
                                                    OVec (&out_trans)[nvec_in],
                                                    CVec &out_dbias,  // NOLINT(*)
                                                    const CType scale_inv,
                                                    const int dbias_shfl_src_lane) {
  using T = typename OVec::type;
  using OVecC = Vec<T, nvec_in>;

  CVec step_dbias;
  step_dbias.clear();

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      const CType tmp = static_cast<CType>(in[i].data.elt[j]) * scale_inv;
      const T elt_o = in[i].data.elt[j];

      /* dbias: thread tile local accumulation */
      step_dbias.data.elt[j] += tmp;

      out_trans[j].data.elt[i] = elt_o;  // thread tile transpose
    }
  }

#pragma unroll
  for (unsigned int j = 0; j < nvec_in; ++j) {
    CType elt = step_dbias.data.elt[j];
    elt = __shfl_sync(0xffffffff, elt, dbias_shfl_src_lane);  // shuffle data in warp
    out_dbias.data.elt[j] += elt;
  }
}

// STUFF TO TUNE
constexpr unsigned int n_warps_per_tile = 4;
constexpr int desired_load_size = 8;
constexpr int desired_store_size = 8;

constexpr unsigned int max_threads_per_block = 256;
static_assert(n_warps_per_tile * THREADS_PER_WARP <= max_threads_per_block);
constexpr unsigned int cast_transpose_num_threads = n_warps_per_tile * THREADS_PER_WARP;

namespace {

template <typename IType, typename OType, typename CType>
struct TDBiasParam {
  using InputType = IType;
  using OutputType = OType;
  using ComputeType = CType;
  const IType *input;
  OType *output_t;
  const CType *scale_inv;
  CType *workspace;
};

}  // namespace

template <int nvec_in, int nvec_out, typename Param>
__global__ void __launch_bounds__(cast_transpose_num_threads)
    transpose_dbias_kernel(const Param param, const size_t row_length, const size_t num_rows,
                           const size_t num_tiles) {
  using IType = typename Param::InputType;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
  // const size_t num_tiles_y = num_rows / (nvec * THREADS_PER_WARP);
  const size_t tile_id =
      blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) + warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType *const my_input_tile =
      param.input + (tile_id_x * nvec_in + tile_id_y * row_length * nvec_out) * THREADS_PER_WARP;
  OType *const my_output_t_tile =
      param.output_t + (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP;
  CType *const my_partial_dbias_tile =
      param.workspace + (tile_id_x * (nvec_in * THREADS_PER_WARP) + tile_id_y * row_length);

  OVec *const my_scratch =
      reinterpret_cast<OVec *>(scratch) +
      (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) * (THREADS_PER_WARP + 1);

  CVec *const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];
  CVec partial_dbias;

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place =
      (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
  const CType scale_inv = param.scale_inv != nullptr ? *param.scale_inv : 1;

  partial_dbias.clear();

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
    transpose_regs_partial_dbias(
        in[current_in ^ 1], out_trans, partial_dbias, scale_inv,
        (my_id_in_warp + i + warp_id_in_tile * n_iterations) % THREADS_PER_WARP);

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
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  my_dbias_scratch[threadIdx.x] = partial_dbias;
  __syncthreads();
  // TODO(ptredak): check if the regular reduction is better
  if (warp_id_in_tile == 0) {
#pragma unroll
    for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
      CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
      for (unsigned int j = 0; j < nvec_in; ++j) {
        partial_dbias.data.elt[j] += tmp.data.elt[j];
      }
    }

    partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
  }
}

template <int nvec_in, int nvec_out, typename Param>
__global__ void __launch_bounds__(cast_transpose_num_threads)
    transpose_dbias_kernel_notaligned(const Param param, const size_t row_length,
                                      const size_t num_rows, const size_t num_tiles) {
  using IType = typename Param::InputType;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x =
      (row_length + nvec_in * THREADS_PER_WARP - 1) / (nvec_in * THREADS_PER_WARP);
  const size_t tile_id =
      blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) + warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType *const my_input_tile =
      param.input + (tile_id_x * nvec_in + tile_id_y * row_length * nvec_out) * THREADS_PER_WARP;
  OType *const my_output_t_tile =
      param.output_t + (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP;
  CType *const my_partial_dbias_tile =
      param.workspace + (tile_id_x * (nvec_in * THREADS_PER_WARP) + tile_id_y * row_length);

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length =
      row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP : row_length_rest;
  const unsigned int tile_height =
      row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP : row_height_rest;

  OVec *const my_scratch =
      reinterpret_cast<OVec *>(scratch) +
      (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) * (THREADS_PER_WARP + 1);

  CVec *const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];
  CVec partial_dbias;

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place =
      (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
  const CType scale_inv = param.scale_inv != nullptr ? *param.scale_inv : 1;

  partial_dbias.clear();

  {
    const bool valid_load = my_place < tile_length && warp_id_in_tile * n_iterations < tile_height;
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
      const bool valid_load =
          my_place_in < tile_length && warp_id_in_tile * n_iterations + i + 1 < tile_height;
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
    transpose_regs_partial_dbias(
        in[current_in ^ 1], out_trans, partial_dbias, scale_inv,
        (my_id_in_warp + i + warp_id_in_tile * n_iterations) % THREADS_PER_WARP);

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
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
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

  my_dbias_scratch[threadIdx.x] = partial_dbias;
  __syncthreads();
  // TODO(ptredak): check if the regular reduction is better
  if (warp_id_in_tile == 0) {
#pragma unroll
    for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
      CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
      for (unsigned int j = 0; j < nvec_in; ++j) {
        partial_dbias.data.elt[j] += tmp.data.elt[j];
      }
    }

    if (my_id_in_warp < tile_length) {
      partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
    }
  }
}

constexpr size_t reduce_dbias_num_threads = 256;

template <int nvec, typename ComputeType, typename OutputType>
__global__ void __launch_bounds__(reduce_dbias_num_threads)
    reduce_dbias_kernel(OutputType *const dbias_output, const ComputeType *const dbias_partial,
                        const int row_length, const int num_rows) {
  using ComputeVec = Vec<ComputeType, nvec>;
  using OutputVec = Vec<OutputType, nvec>;

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id * nvec >= row_length) return;

  const ComputeType *const thread_in_base = dbias_partial + thread_id * nvec;
  OutputType *const thread_out_base = dbias_output + thread_id * nvec;

  const int stride_in_vec = row_length / nvec;

  ComputeVec ldg_vec;
  ComputeVec acc_vec;
  acc_vec.clear();
  for (int i = 0; i < num_rows; ++i) {
    ldg_vec.load_from(thread_in_base, i * stride_in_vec);
#pragma unroll
    for (int e = 0; e < nvec; ++e) {
      acc_vec.data.elt[e] += ldg_vec.data.elt[e];
    }
  }

  OutputVec stg_vec;
#pragma unroll
  for (int e = 0; e < nvec; ++e) {
    stg_vec.data.elt[e] = OutputType(acc_vec.data.elt[e]);
  }
  stg_vec.store_to(thread_out_base, 0);
}

void populate_transpose_dbias_workspace_config(const Tensor &input, /*cast*/
                                               Tensor *workspace, const int nvec_out) {
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  const size_t tile_size_y = (nvec_out * THREADS_PER_WARP);
  NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");

  const size_t num_rows_partial_dbias = DIVUP(num_rows, tile_size_y);

  workspace->data.shape = {num_rows_partial_dbias, row_length};
  workspace->data.dtype = DType::kFloat32;
}

template <typename BiasType>
void reduce_dbias(const Tensor &workspace, Tensor *dbias, const size_t row_length,
                  const size_t num_rows, const int nvec_out, cudaStream_t stream) {
  constexpr int reduce_dbias_store_bytes = 8;  // stg.64
  constexpr int reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(BiasType);

  NVTE_CHECK(row_length % reduce_dbias_nvec == 0, "Unsupported shape.");

  const size_t reduce_dbias_row_length = row_length;
  const size_t reduce_dbias_num_rows =
      DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
  const size_t reduce_dbias_num_blocks =
      DIVUP(row_length, reduce_dbias_num_threads * reduce_dbias_nvec);

  reduce_dbias_kernel<reduce_dbias_nvec, fp32, BiasType>
      <<<reduce_dbias_num_blocks, reduce_dbias_num_threads, 0, stream>>>(
          reinterpret_cast<BiasType *>(dbias->data.dptr),
          reinterpret_cast<const fp32 *>(workspace.data.dptr), reduce_dbias_row_length,
          reduce_dbias_num_rows);
}

void fp8_transpose_dbias(const Tensor &input, Tensor *transposed_output, Tensor *dbias,
                         Tensor *workspace, cudaStream_t stream) {
  CheckInputTensor(input, "fp8_transpose_dbias_input");
  CheckOutputTensor(*transposed_output, "transposed_output");
  CheckOutputTensor(*dbias, "dbias");

  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(transposed_output->data.shape.size() == 2, "T output must have 2 dimensions.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  NVTE_CHECK(transposed_output->data.shape[0] == row_length, "Wrong dimension of T output.");
  NVTE_CHECK(transposed_output->data.shape[1] == num_rows, "Wrong dimension of T output.");

  NVTE_CHECK(transposed_output->data.dtype == input.data.dtype,
             "T output must have the same type as input.");
  NVTE_CHECK(dbias->data.shape == std::vector<size_t>{row_length}, "Wrong shape of DBias.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      dbias->data.dtype, BiasType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          input.data.dtype, Type, constexpr int type_size = sizeof(Type);
          constexpr int nvec_in = desired_load_size / type_size;
          constexpr int nvec_out = desired_store_size / type_size;

          if (workspace->data.dptr == nullptr) {
            populate_transpose_dbias_workspace_config(input, workspace, nvec_out);
            return;
          }

          NVTE_CHECK(row_length % nvec_in == 0, "Unsupported shape.");
          NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");
          const size_t n_tiles =
              DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
              DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
          const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
          const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

          const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                                 num_rows % (nvec_out * THREADS_PER_WARP) == 0;

          using ComputeType = fp32; constexpr size_t shared_size_transpose =
                                        cast_transpose_num_threads / n_warps_per_tile *
                                        (THREADS_PER_WARP + 1) * sizeof(Vec<Type, nvec_out>);
          constexpr size_t shared_size_dbias =
              cast_transpose_num_threads * sizeof(Vec<ComputeType, nvec_in>);
          static_assert(shared_size_transpose >= shared_size_dbias);
          using Param = TDBiasParam<Type, Type, ComputeType>; Param param;
          param.input = reinterpret_cast<const Type *>(input.data.dptr);
          param.output_t = reinterpret_cast<Type *>(transposed_output->data.dptr);
          param.scale_inv =
              reinterpret_cast<const ComputeType *>(transposed_output->scale_inv.dptr);
          param.workspace = reinterpret_cast<ComputeType *>(workspace->data.dptr);

          if (full_tile) {
            cudaFuncSetAttribute(transpose_dbias_kernel<nvec_in, nvec_out, Param>,
                                 cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            transpose_dbias_kernel<nvec_in, nvec_out, Param>
                <<<n_blocks, cast_transpose_num_threads, shared_size_transpose, stream>>>(
                    param, row_length, num_rows, n_tiles);
          } else {
            cudaFuncSetAttribute(transpose_dbias_kernel_notaligned<nvec_in, nvec_out, Param>,
                                 cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            transpose_dbias_kernel_notaligned<nvec_in, nvec_out, Param>
                <<<n_blocks, cast_transpose_num_threads, shared_size_transpose, stream>>>(
                    param, row_length, num_rows, n_tiles);
          }

          reduce_dbias<BiasType>(*workspace, dbias, row_length, num_rows, nvec_out,
                                 stream););  // NOLINT(*)
  );                                         // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_fp8_transpose_dbias(const NVTETensor input, NVTETensor transposed_output,
                              NVTETensor dbias, NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fp8_transpose_dbias);
  using namespace transformer_engine;
  fp8_transpose_dbias(
      *reinterpret_cast<const Tensor *>(input), reinterpret_cast<Tensor *>(transposed_output),
      reinterpret_cast<Tensor *>(dbias), reinterpret_cast<Tensor *>(workspace), stream);
}
