/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "util/math.h"
#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Parameters
using CType = float;
using IType = __ITYPE__;
using IType2 = __ITYPE2__;
using OType = __OTYPE__;
constexpr size_t LOAD_SIZE = __LOAD_SIZE__;
constexpr size_t STORE_SIZE = __STORE_SIZE__;
constexpr size_t WARPS_PER_TILE = __WARPS_PER_TILE__;
constexpr size_t BLOCK_SIZE = __BLOCK_SIZE__;
constexpr bool IS_DBIAS = __IS_DBIAS__;
constexpr bool IS_DACT = __IS_DACT__;
constexpr size_t DACT_TYPE = __DACTIVATION_TYPE__;

constexpr size_t NVEC_IN = LOAD_SIZE / sizeof(IType);
constexpr size_t NVEC_OUT = STORE_SIZE / sizeof(OType);
using CVec = Vec<CType, NVEC_IN>;
using IVec = Vec<IType, NVEC_IN>;
using IVec2 = Vec<IType2, NVEC_IN>;
using OVec = Vec<OType, NVEC_OUT>;
using Param = CTDBiasDActParam<IType, IType2, OType, CType>;

using OP = CType (*)(const CType, const Empty &);
constexpr OP Activation[] = {
    nullptr,                  // 0
    &dsigmoid<CType, CType>,  // 1
    &dgelu<CType, CType>,     // 2
    &dqgelu<CType, CType>,    // 3
    &dsilu<CType, CType>,     // 4
    &drelu<CType, CType>,     // 5
    &dsrelu<CType, CType>     // 6
};

}  // namespace

inline __device__ void cast_and_transpose_regs_optimized(const CVec (&in)[NVEC_OUT],
                                                         OVec (&out_trans)[NVEC_IN],
                                                         CVec &out_dbias,  // NOLINT(*)
                                                         typename OVec::type *output_cast_tile,
                                                         const size_t current_place,
                                                         const size_t stride, const CType scale,
                                                         CType &amax,  // NOLINT(*)
                                                         const int dbias_shfl_src_lane) {
  using OVecC = Vec<OType, NVEC_IN>;

  CVec step_dbias;
  if constexpr (IS_DBIAS) {
    step_dbias.clear();
  }

#pragma unroll
  for (unsigned int i = 0; i < NVEC_OUT; ++i) {
    OVecC out_cast;
#pragma unroll
    for (unsigned int j = 0; j < NVEC_IN; ++j) {
      const CType tmp = in[i].data.elt[j];
      if constexpr (IS_DBIAS) {
        step_dbias.data.elt[j] += tmp;  // dbias: thread tile local accumulation
      }
      out_cast.data.elt[j] = static_cast<OType>(tmp * scale);
      out_trans[j].data.elt[i] = static_cast<OType>(tmp * scale);  // thread tile transpose

      __builtin_assume(amax >= 0);
      amax = fmaxf(fabsf(tmp), amax);
    }
    out_cast.store_to(output_cast_tile, current_place + stride * i);
  }

  if constexpr (IS_DBIAS) {
#pragma unroll
    for (unsigned int j = 0; j < NVEC_IN; ++j) {
      CType elt = step_dbias.data.elt[j];
      elt = __shfl_sync(0xffffffff, elt, dbias_shfl_src_lane);  // shuffle data in a warp
      out_dbias.data.elt[j] += elt;
    }
  }
}

__global__ void __launch_bounds__(BLOCK_SIZE)
    cast_transpose_fusion_kernel_optimized(const Param param, const size_t row_length,
                                           const size_t num_rows, const size_t num_tiles) {
  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (NVEC_IN * THREADS_PER_WARP);
  const size_t tile_id =
      blockIdx.x * blockDim.x / (THREADS_PER_WARP * WARPS_PER_TILE) + warp_id / WARPS_PER_TILE;
  if (tile_id >= num_tiles) {
    return;
  }

  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const size_t tile_offset =
      (tile_id_x * NVEC_IN + tile_id_y * row_length * NVEC_OUT) * THREADS_PER_WARP;
  const size_t tile_offset_transp =
      (tile_id_y * NVEC_OUT + tile_id_x * num_rows * NVEC_IN) * THREADS_PER_WARP;

  const IType *const my_input_tile = param.input + tile_offset;
  const IType2 *const my_act_input_tile = param.act_input + tile_offset;
  OType *const my_output_c_tile = param.output_c + tile_offset;
  OType *const my_output_t_tile = param.output_t + tile_offset_transp;
  CType *const my_partial_dbias_tile =
      param.workspace + (tile_id_x * (NVEC_IN * THREADS_PER_WARP) + tile_id_y * row_length);

  OVec *const my_scratch =
      reinterpret_cast<OVec *>(scratch) +
      (my_id_in_warp + warp_id / WARPS_PER_TILE * THREADS_PER_WARP) * (THREADS_PER_WARP + 1);

  CVec *const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][NVEC_OUT];
  IVec2 act_in[2][NVEC_OUT];

  const unsigned int warp_id_in_tile = warp_id % WARPS_PER_TILE;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / WARPS_PER_TILE;
  OVec out_space[n_iterations][NVEC_IN];

  const size_t stride = row_length / NVEC_IN;
  const size_t output_stride = num_rows / NVEC_OUT;
  size_t current_stride = warp_id_in_tile * n_iterations * NVEC_OUT * stride;
  size_t current_row = (tile_id_y * THREADS_PER_WARP + warp_id_in_tile * n_iterations) * NVEC_OUT;
  unsigned int my_place =
      (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;

  CType amax = 0.0f;
  const CType scale = param.scale_ptr != nullptr ? *param.scale_ptr : 1;

  CVec partial_dbias;
  if constexpr (IS_DBIAS) {
    partial_dbias.clear();
  }

#pragma unroll
  for (unsigned int i = 0; i < NVEC_OUT; ++i) {
    in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
    if constexpr (IS_DACT) {
      act_in[0][i].load_from(my_act_input_tile, current_stride + my_place + stride * i);
    }
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < NVEC_OUT; ++j) {
        const size_t ld_offset = current_stride + my_place_in + stride * (NVEC_OUT + j);
        in[current_in][j].load_from(my_input_tile, ld_offset);
        if constexpr (IS_DACT) {
          act_in[current_in][j].load_from(my_act_input_tile, ld_offset);
        }
      }
    }
    CVec in_cast_fp32[NVEC_OUT];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < NVEC_OUT; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < NVEC_IN; ++k) {
        if constexpr (IS_DACT) {
          in_cast_fp32[j].data.elt[k] =
              static_cast<CType>(in[current_in ^ 1][j].data.elt[k]) *
              Activation[DACT_TYPE](act_in[current_in ^ 1][j].data.elt[k], {});
        } else {
          in_cast_fp32[j].data.elt[k] = static_cast<CType>(in[current_in ^ 1][j].data.elt[k]);
        }
      }
    }

    const int dbias_shfl_src_lane =
        (my_id_in_warp + i + warp_id_in_tile * n_iterations) % THREADS_PER_WARP;

    cast_and_transpose_regs_optimized(in_cast_fp32, out_space[i], partial_dbias, my_output_c_tile,
                                      current_place, stride, scale, amax, dbias_shfl_src_lane);

    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += NVEC_OUT * stride;
    current_row += NVEC_OUT;
  }

#pragma unroll
  for (unsigned int i = 0; i < NVEC_IN; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * NVEC_IN;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * NVEC_IN;
    }
    __syncthreads();
  }

  if constexpr (IS_DBIAS) {
    my_dbias_scratch[threadIdx.x] = partial_dbias;
    __syncthreads();
    if (warp_id_in_tile == 0) {
#pragma unroll
      for (unsigned int i = 1; i < WARPS_PER_TILE; ++i) {
        CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
        for (unsigned int j = 0; j < NVEC_IN; ++j) {
          partial_dbias.data.elt[j] += tmp.data.elt[j];
        }
      }
      partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
    }
  }

  // warp tile amax reduce
  const CType max_block = reduce_max<BLOCK_SIZE / THREADS_PER_WARP>(amax, warp_id);

  if (threadIdx.x == 0) {
    if (param.amax != nullptr) {
      atomicMaxFloat(param.amax, max_block);
    }
  }
}
