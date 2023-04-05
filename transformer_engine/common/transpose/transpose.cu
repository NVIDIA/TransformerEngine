/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transpose.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>
#include "../common.h"
#include "../utils.cuh"
#include "../util/string.h"
#include "../util/rtc.h"

#include "rtc_code_transpose.h"

namespace transformer_engine {

namespace {

// STUFF TO TUNE
constexpr int warps_per_tile = 4;
constexpr int vector_size = 8;

constexpr int block_size = THREADS_PER_WARP * warps_per_tile;

}  // namespace

template <typename Type>
__global__ void
__launch_bounds__(block_size)
transpose_general_kernel(const Type * const input,
                         Type * const output,
                         const int row_length,
                         const int num_rows) {
  // Vectorized load/store sizes
  constexpr int nvec_in = vector_size / sizeof(Type);
  constexpr int nvec_out = vector_size / sizeof(Type);
  using IVec = Vec<Type, nvec_in>;
  using OVec = Vec<Type, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = warps_per_tile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id_m = bid / num_tiles_n;
  const int tile_id_n = bid % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int num_iterations = THREADS_PER_WARP / warps_per_tile;

  // Load input and store to registers
  // Note: Each thread loads num_iterations subtiles and transposes in
  // registers.
  OVec local_output[nvec_in][num_iterations];
  #pragma unroll
  for (int iter = 0; iter < num_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
    #pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      local_input.clear();
      if (row < num_rows) {
        #pragma unroll
        for (int j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < row_length) {
            local_input.data.elt[j2] = input[row * row_length + col + j2];
          }
        }
      }
      #pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        local_output[j2][iter].data.elt[i2] = local_input.data.elt[j2];
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVec shared_output[THREADS_PER_WARP][THREADS_PER_WARP+1];
  #pragma unroll
  for (int j2 = 0; j2 < nvec_in; ++j2) {
    #pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      const int i1 = tidy + iter * bdimy;
      const int j1 = tidx;
      shared_output[j1][i1] = local_output[j2][iter];
    }
    __syncthreads();
    #pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      const int i1 = tidx;
      const int j1 = tidy + iter * bdimy;
      const int row = tile_row + i1 * nvec_out;
      const int col = tile_col + j1 * nvec_in + j2;
      if (col < row_length) {
        #pragma unroll
        for (int i2 = 0; i2 < nvec_out; ++i2) {
          if (row + i2 < num_rows) {
            output[col * num_rows + row + i2] = shared_output[j1][i1].data.elt[i2];
          }
        }
      }
    }
    __syncthreads();
  }
}

void transpose(const Tensor &input,
               Tensor &output,
               cudaStream_t stream) {
  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output.data.shape.size() == 2, "Output must have 2 dimensions.");
  const int row_length = input.data.shape[1];
  const int num_rows = input.data.shape[0];

  NVTE_CHECK(output.data.shape[0] == row_length, "Wrong dimension of output.");
  NVTE_CHECK(output.data.shape[1] == num_rows, "Wrong dimension of output.");

  NVTE_CHECK(input.data.dptr != nullptr, "Input is not allocated.");
  NVTE_CHECK(output.data.dptr != nullptr, "Output is not allocated.");
  NVTE_CHECK(input.data.dtype == output.data.dtype,
             "Input and output type must match.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(input.data.dtype, Type,
    // Check if data dims match tiling scheme
    constexpr int type_size = sizeof(Type);
    constexpr int tile_size = vector_size / type_size * THREADS_PER_WARP;
    const bool full_tile = row_length % tile_size == 0 && num_rows % tile_size == 0;

    if (full_tile && rtc::is_enabled() && input.data.dtype == DType::kFloat32) { /// TODO Support arbitrary types
      constexpr const char *type_name = TypeInfo<Type>::name;
      int load_size = 8;
      int store_size = 8;

      // Tuned runtime-compiled kernel
      auto& rtc_manager = rtc::KernelManager::instance();
      const std::string kernel_label = concat_strings("transpose"
                                                      ",type=",type_name,
                                                      ",load_size=",load_size,
                                                      ",store_size",store_size);
      if (!rtc_manager.is_compiled(kernel_label)) {
        std::string code = rtc_code_transpose;
        code = regex_replace(code, "__TYPE__", type_name);
        code = regex_replace(code, "__LOAD_SIZE__", load_size);
        code = regex_replace(code, "__STORE_SIZE__", store_size);
        rtc_manager.compile(kernel_label,
                            "transpose_optimized_kernel",
                            code,
                            "rtc/transpose.cu");
      }
      const int num_blocks = (row_length / tile_size) * (num_rows / tile_size);
      rtc_manager.launch(kernel_label, num_blocks, block_size, 0, stream,
                         input.data.dptr, output.data.dptr,
                         row_length, num_rows);
    } else {
      // General kernel
      const int num_blocks = DIVUP(row_length, tile_size) * DIVUP(num_rows, tile_size);
      transpose_general_kernel<Type><<<num_blocks,block_size,0,stream>>>(
        reinterpret_cast<const Type *>(input.data.dptr),
        reinterpret_cast<Type *>(output.data.dptr),
        row_length, num_rows);
    }
  );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_transpose(const NVTETensor input,
                    NVTETensor output,
                    cudaStream_t stream) {
  NVTE_API_CALL(nvte_transpose);
  using namespace transformer_engine;
  transpose(*reinterpret_cast<const Tensor*>(input),
            *reinterpret_cast<Tensor*>(output),
            stream);
}
