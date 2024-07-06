/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Parameters
using CType = float;
using IType = __ITYPE__;
using OType = __OTYPE__;
constexpr size_t load_size = __LOAD_SIZE__;
constexpr size_t store_size = __STORE_SIZE__;
constexpr size_t warps_per_tile = __WARPS_PER_TILE__;
constexpr size_t block_size = __BLOCK_SIZE__;
constexpr bool aligned = __ALIGNED__;
constexpr size_t max_tensors_per_kernel = __MAX_TENSORS_PER_KERNEL__;

}  // namespace

namespace transformer_engine {
namespace multi_cast_transpose_impl {

// Duplicate definition from launching code
struct KernelArgs {
  void* input_list[max_tensors_per_kernel];
  void* output_c_list[max_tensors_per_kernel];
  void* output_t_list[max_tensors_per_kernel];
  void* scale_list[max_tensors_per_kernel];
  void* amax_list[max_tensors_per_kernel];
  size_t num_rows_list[max_tensors_per_kernel];
  size_t row_length_list[max_tensors_per_kernel];
  size_t block_range[max_tensors_per_kernel + 1];
  size_t num_tensors;
};

};  // namespace multi_cast_transpose_impl
};  // namespace transformer_engine

__global__ void __launch_bounds__(block_size) multi_cast_transpose_kernel(multi_cast_transpose_impl::KernelArgs args) {

  // Vectorized load/store sizes
  constexpr size_t nvec_in = load_size / sizeof(IType);
  constexpr size_t nvec_out = store_size / sizeof(OType);
  using IVec = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr size_t bdimx = THREADS_PER_WARP;
  constexpr size_t bdimy = warps_per_tile;
  const size_t tid = threadIdx.x;
  const size_t tidx = tid % bdimx;
  const size_t tidy = tid / bdimx;
  const size_t bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Find tensor corresponding to block
  size_t tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const IType* input = reinterpret_cast<const IType*>(args.input_list[tensor_id]);
  OType* output_c = reinterpret_cast<OType*>(args.output_c_list[tensor_id]);
  OType* output_t = reinterpret_cast<OType*>(args.output_t_list[tensor_id]);
  const CType* scale_ptr = reinterpret_cast<CType*>(args.scale_list[tensor_id]);
  CType* amax_ptr = reinterpret_cast<CType*>(args.amax_list[tensor_id]);
  const size_t num_rows = args.num_rows_list[tensor_id];
  const size_t row_length = args.row_length_list[tensor_id];

  // Position of tile within tensor
  const size_t num_tiles_m = num_rows / tile_dim_m;
  const size_t tile_id = bid - args.block_range[tensor_id];
  const size_t tile_id_m = tile_id % num_tiles_m;
  const size_t tile_id_n = tile_id / num_tiles_m;
  const size_t tile_row = tile_id_m * tile_dim_m;
  const size_t tile_col = tile_id_n * tile_dim_n;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr size_t num_iterations = THREADS_PER_WARP / warps_per_tile;

  // FP8 factors
  const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  CType amax = 0;

  // Load input to registers and transpose
  // Note: Each thread loads num_iterations subtiles, computes amax,
  // casts type, and transposes in registers.
  OVecT local_output_t[nvec_in][num_iterations];
#pragma unroll
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    const size_t i1 = tidy + iter * bdimy;
    const size_t j1 = tidx;
#pragma unroll
    for (size_t i2 = 0; i2 < nvec_out; ++i2) {
      const size_t row = tile_row + i1 * nvec_out + i2;
      const size_t col = tile_col + j1 * nvec_in;
      IVec local_input;
      OVecC local_output_c;
      if constexpr (aligned) {
        local_input.load_from(&input[row * row_length + col]);
      } else {
        local_input.clear();
        if (row < num_rows) {
#pragma unroll
          for (size_t j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              local_input.data.elt[j2] = input[row * row_length + col + j2];
            }
          }
        }
      }
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
        const CType in = static_cast<CType>(local_input.data.elt[j2]);
        const OType out = OType(in * scale);
        __builtin_assume(amax >= 0);
        amax = fmaxf(fabsf(in), amax);
        local_output_c.data.elt[j2] = out;
        local_output_t[j2][iter].data.elt[i2] = out;
      }
      if constexpr (aligned) {
        local_output_c.store_to(&output_c[row * row_length + col]);
      } else {
        if (row < num_rows) {
#pragma unroll
          for (size_t j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              output_c[row * row_length + col + j2] = local_output_c.data.elt[j2];
            }
          }
        }
      }
    }
  }

  // Copy from registers to shared memory to global memory
  __shared__ OVecT shared_output_t[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * bdimy;
      const size_t j1 = tidx;
      shared_output_t[j1][i1] = local_output_t[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidx;
      const size_t j1 = tidy + iter * bdimy;
      const size_t row = tile_row + i1 * nvec_out;
      const size_t col = tile_col + j1 * nvec_in + j2;
      if constexpr (aligned) {
        shared_output_t[j1][i1].store_to(&output_t[col * num_rows + row]);
      } else {
        if (col < row_length) {
#pragma unroll
          for (size_t i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < num_rows) {
              output_t[col * num_rows + row + i2] = shared_output_t[j1][i1].data.elt[i2];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Reduce amax over block
  if (amax_ptr != nullptr) {
    amax = reduce_max<warps_per_tile>(amax, tidy);
    if (threadIdx.x == 0) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }
}
