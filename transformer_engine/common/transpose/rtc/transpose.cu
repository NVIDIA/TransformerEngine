/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Parameters
using Type = __TYPE__;
constexpr int load_size = __LOAD_SIZE__;
constexpr int store_size = __STORE_SIZE__;
constexpr int warps_per_tile = __WARPS_PER_TILE__;
constexpr int block_size = __BLOCK_SIZE__;

}  // namespace

__global__ void
__launch_bounds__(block_size)
transpose_optimized_kernel(const Type * __restrict__ const input,
                           Type * __restrict__  const output,
                           const int row_length,
                           const int num_rows) {
  // Vectorized load/store sizes
  constexpr int nvec_in = load_size / sizeof(Type);
  constexpr int nvec_out = store_size / sizeof(Type);
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
  const int num_tiles_m = num_rows / tile_dim_m;
  const int tile_id_m = bid % num_tiles_m;
  const int tile_id_n = bid / num_tiles_m;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int num_iterations = THREADS_PER_WARP / warps_per_tile;

  // Load input to registers and transpose
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
      local_input.load_from(&input[row * row_length + col]);
      #pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        local_output[j2][iter].data.elt[i2] = local_input.data.elt[j2];
      }
    }
  }

  // Copy from registers to shared memory to global memory
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
      shared_output[j1][i1].store_to(&output[col * num_rows + row]);
    }
    __syncthreads();
  }
}
