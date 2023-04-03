/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/// TODO Get from cstdint header
using uint32_t = size_t;
using uint64_t = size_t;

// Parameters
/// TODO Make configurable
using Type = float;
constexpr int load_size = 8;
constexpr int store_size = 8;
constexpr int warps_per_tile = 4;
constexpr int THREADS_PER_WARP = 32;

namespace {

constexpr int block_size = THREADS_PER_WARP * warps_per_tile;

/// TODO Use existing impl in utils.cuh
template<int BYTES>
struct BytesToType {};
template<>
struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8, "Unexpected type size");
};
template<typename Elt_type, uint32_t NUM_ELT>
struct Vec {
    enum { BYTES = NUM_ELT * sizeof(Elt_type) };

    using Vec_type = typename BytesToType<BYTES>::Type;
    using type = Elt_type;

    using Alias_type = union {
        Vec_type vec;
        Elt_type elt[NUM_ELT];
    };

    Alias_type data;

    // Pointer is cast to vector type
    inline __device__ void load_from(const void *base_ptr, size_t idx = 0) {
        this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
    }

    // Pointer is cast to vector type
    inline __device__ void store_to(void *base_ptr, size_t idx = 0) const {
        static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
    }
};

}  // namespace

__global__ void
__launch_bounds__(block_size)
transpose_optimized_kernel(const Type * const input,
                           Type * const output,
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
  const int num_tiles_n = row_length / tile_dim_n;
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
      local_input.load_from(&input[row * row_length + col]);
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
      shared_output[j1][i1].store_to(&output[col * num_rows + row]);
    }
    __syncthreads();
  }
}
