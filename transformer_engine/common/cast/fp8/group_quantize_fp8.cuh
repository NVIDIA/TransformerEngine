/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file group_quantize_fp8.cuh
 *  \brief CUDA kernels to quantize grouped tensors to FP8 tensor scaling.
 */

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_

#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <type_traits>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

using namespace dispatch::common;

constexpr size_t WARPS_PER_TILE = 8;
constexpr size_t THREADS_PER_TILE = THREADS_PER_WARP * WARPS_PER_TILE;
constexpr size_t ROWWISE_LOAD_SIZE_BYTES = 16;
constexpr size_t TRANSPOSE_LOAD_SIZE_BYTES = 8;
constexpr size_t ROWWISE_STORE_SIZE_BYTES = 16;
constexpr size_t TRANSPOSE_STORE_SIZE_BYTES = 8;
constexpr size_t ROWWISE_FLAT_LOAD_SIZE_BYTES = 32;
constexpr size_t ROWWISE_FLAT_THREADS = 512;
constexpr size_t VARYING_FIRST_ROWWISE_MAX_BLOCKS_PER_TENSOR = 1024;
constexpr size_t VARYING_FIRST_TRANSPOSE_MAX_ROW_TILES_PER_TENSOR = 4;
constexpr size_t TRANSPOSE_SHARED_PAD = 1;

template <typename IType, typename OType>
struct supports_fast_scaled_fp8_cvt_4 {
  static constexpr bool value =
      (std::is_same<IType, bf16>::value || std::is_same<IType, fp16>::value ||
       std::is_same<IType, fp32>::value) &&
      (std::is_same<OType, fp8e4m3>::value || std::is_same<OType, fp8e5m2>::value);
};

template <typename IType, typename OType>
__device__ __forceinline__ void fast_scaled_fp8_cvt_4(const IType *input, OType *output,
                                                      const float scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const ptx::floatx2 scale_2x{scale, scale};
  if constexpr (std::is_same<IType, bf16>::value) {
    const ptx::bf16x4 in{input[0], input[1], input[2], input[3]};
    if constexpr (std::is_same<OType, fp8e4m3>::value) {
      ptx::fp8e4m3x4 out;
      ptx::mul_cvt_4x(out, in, scale_2x);
      output[0] = out.x1;
      output[1] = out.x2;
      output[2] = out.x3;
      output[3] = out.x4;
    } else {
      ptx::fp8e5m2x4 out;
      ptx::mul_cvt_4x(out, in, scale_2x);
      output[0] = out.x1;
      output[1] = out.x2;
      output[2] = out.x3;
      output[3] = out.x4;
    }
  } else if constexpr (std::is_same<IType, fp16>::value) {
    const ptx::fp16x4 in{input[0], input[1], input[2], input[3]};
    if constexpr (std::is_same<OType, fp8e4m3>::value) {
      ptx::fp8e4m3x4 out;
      ptx::mul_cvt_4x(out, in, scale_2x);
      output[0] = out.x1;
      output[1] = out.x2;
      output[2] = out.x3;
      output[3] = out.x4;
    } else {
      ptx::fp8e5m2x4 out;
      ptx::mul_cvt_4x(out, in, scale_2x);
      output[0] = out.x1;
      output[1] = out.x2;
      output[2] = out.x3;
      output[3] = out.x4;
    }
  } else if constexpr (std::is_same<IType, fp32>::value) {
    const ptx::floatx4 in{input[0], input[1], input[2], input[3]};
    if constexpr (std::is_same<OType, fp8e4m3>::value) {
      ptx::fp8e4m3x4 out;
      ptx::mul_cvt_4x(out, in, scale_2x);
      output[0] = out.x1;
      output[1] = out.x2;
      output[2] = out.x3;
      output[3] = out.x4;
    } else {
      ptx::fp8e5m2x4 out;
      ptx::mul_cvt_4x(out, in, scale_2x);
      output[0] = out.x1;
      output[1] = out.x2;
      output[2] = out.x3;
      output[3] = out.x4;
    }
  }
#else
  (void)input;
  (void)output;
  (void)scale;
#endif
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, uint32_t NUM_ELTS>
__device__ __forceinline__ void scaled_fp8_cvt_vec(const Vec<IType, NUM_ELTS> &input,
                                                   Vec<OType, NUM_ELTS> &output,
                                                   const size_t valid_cols, const float scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool kUseFastCvt =
      !IS_ACT && (NUM_ELTS % 4 == 0) && supports_fast_scaled_fp8_cvt_4<IType, OType>::value;
#else
  constexpr bool kUseFastCvt = false;
#endif

  if constexpr (kUseFastCvt) {
    if (valid_cols == NUM_ELTS) {
#pragma unroll
      for (size_t base = 0; base < NUM_ELTS; base += 4) {
        fast_scaled_fp8_cvt_4(input.data.elt + base, output.data.elt + base, scale);
      }
      return;
    }
  }

#pragma unroll
  for (size_t j = 0; j < NUM_ELTS; ++j) {
    if (j < valid_cols) {
      float elt = static_cast<float>(input.data.elt[j]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      output.data.elt[j] = static_cast<OType>(elt * scale);
    }
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, uint32_t NUM_ELTS>
__device__ __forceinline__ void scaled_fp8_cvt_vec_full(const Vec<IType, NUM_ELTS> &input,
                                                        Vec<OType, NUM_ELTS> &output,
                                                        const float scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  constexpr bool kUseFastCvt =
      !IS_ACT && (NUM_ELTS % 4 == 0) && supports_fast_scaled_fp8_cvt_4<IType, OType>::value;
#else
  constexpr bool kUseFastCvt = false;
#endif

  if constexpr (kUseFastCvt) {
#pragma unroll
    for (size_t base = 0; base < NUM_ELTS; base += 4) {
      fast_scaled_fp8_cvt_4(input.data.elt + base, output.data.elt + base, scale);
    }
  } else {
#pragma unroll
    for (size_t j = 0; j < NUM_ELTS; ++j) {
      float elt = static_cast<float>(input.data.elt[j]);
      if constexpr (IS_ACT) {
        elt = OP(elt, {});
      }
      output.data.elt[j] = static_cast<OType>(elt * scale);
    }
  }
}

template <typename OType, uint32_t NUM_ELTS>
__device__ __forceinline__ void store_fp8_vec_streaming(OType *ptr,
                                                        const Vec<OType, NUM_ELTS> &vec) {
  constexpr size_t bytes = Vec<OType, NUM_ELTS>::BYTES;
  if constexpr (bytes == 4) {
    const uint32_t value = vec.data.vec;
    asm volatile("st.global.cs.u32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
  } else if constexpr (bytes == 8) {
    const uint64_t value = vec.data.vec;
    asm volatile("st.global.cs.u64 [%0], %1;" ::"l"(ptr), "l"(value) : "memory");
  } else if constexpr (bytes == 16) {
    const uint4 value = vec.data.vec;
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x),
                 "r"(value.y), "r"(value.z), "r"(value.w)
                 : "memory");
  } else {
    vec.store_to(ptr);
  }
}

// Copy a warp's contiguous SEG-byte column segment from shared to global using
// the widest store the destination alignment allows. The destination alignment
// is warp-uniform (all lanes of a warp target the same column, so the segment
// base offset is identical), so the width selection below never diverges. The
// shared source is contiguous and 8-byte aligned, so every width up to 8 bytes
// is safe to read. Striping element ``e = k*WARP + lane`` keeps each store
// instruction fully coalesced across the warp regardless of width.
template <typename T, size_t SEG>
__device__ __forceinline__ void copy_column_segment_strided(char *gl, const char *sh, size_t lane) {
  static_assert(SEG % (sizeof(T) * THREADS_PER_WARP) == 0,
                "segment must tile evenly into warp-wide element stripes");
  constexpr size_t STEPS = SEG / sizeof(T) / THREADS_PER_WARP;
  T *const g = reinterpret_cast<T *>(gl);
  const T *const s = reinterpret_cast<const T *>(sh);
#pragma unroll
  for (size_t k = 0; k < STEPS; ++k) {
    const size_t e = k * THREADS_PER_WARP + lane;
    g[e] = s[e];
  }
}

template <typename OVecT>
__device__ __forceinline__ void store_column_segment(OVecT *shared_segment, char *gl, size_t lane) {
  // The segment is the WARP contiguous OVecT slots produced for one column.
  constexpr size_t SEG = THREADS_PER_WARP * sizeof(OVecT);
  const char *const sh = reinterpret_cast<const char *>(shared_segment);
  const uint64_t a = reinterpret_cast<uint64_t>(gl);
  if ((a & 7) == 0) {
    copy_column_segment_strided<uint64_t, SEG>(gl, sh, lane);
  } else if ((a & 3) == 0) {
    copy_column_segment_strided<uint32_t, SEG>(gl, sh, lane);
  } else if ((a & 1) == 0) {
    copy_column_segment_strided<uint16_t, SEG>(gl, sh, lane);
  } else {
    copy_column_segment_strided<uint8_t, SEG>(gl, sh, lane);
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, ScalingType SCALING_TYPE, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_TILE)
    group_cast_fp8_kernel(const IType *__restrict__ input, OType *__restrict__ output_rowwise,
                          OType *__restrict__ output_colwise, const float *__restrict__ scale_ptr,
                          const float *__restrict__ noop, const size_t num_tensors,
                          const size_t first_logical_dim, const size_t last_logical_dim,
                          const int64_t *__restrict__ offsets_ptr,
                          const int64_t *__restrict__ first_dims_ptr,
                          const int64_t *__restrict__ last_dims_ptr) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr bool ROWWISE_OUTPUT =
      (SCALING_TYPE == ScalingType::ROWWISE) || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  constexpr bool COLWISE_OUTPUT =
      (SCALING_TYPE == ScalingType::COLWISE) || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  constexpr size_t LOAD_SIZE_BYTES =
      COLWISE_OUTPUT ? TRANSPOSE_LOAD_SIZE_BYTES : ROWWISE_LOAD_SIZE_BYTES;
  constexpr size_t STORE_SIZE_BYTES =
      COLWISE_OUTPUT ? TRANSPOSE_STORE_SIZE_BYTES : ROWWISE_STORE_SIZE_BYTES;
  constexpr size_t nvec_in = LOAD_SIZE_BYTES / sizeof(IType);
  constexpr size_t nvec_out = STORE_SIZE_BYTES / sizeof(OType);
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
  constexpr size_t num_iterations = THREADS_PER_WARP / WARPS_PER_TILE;

  using IVecT = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  size_t tile_col = blockIdx.x * tile_dim_n;
  size_t tensor_id = 0;
  size_t rows = 0;
  size_t tensor_base = 0;
  size_t tile_row = 0;
  size_t cols = last_logical_dim;
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
    if (tile_col >= last_logical_dim) {
      return;
    }
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    const size_t tiles_per_tensor = DIVUP(rows_per_tensor, tile_dim_m);
    if (tiles_per_tensor == 0) {
      return;
    }
    tensor_id = blockIdx.y / tiles_per_tensor;
    if (tensor_id >= num_tensors) {
      return;
    }
    rows = rows_per_tensor;
    tensor_base = tensor_id * rows * cols;
    tile_row = (blockIdx.y - tensor_id * tiles_per_tensor) * tile_dim_m;
  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_LAST_DIM) {
    rows = first_logical_dim;
    const size_t row_tiles_per_tensor = DIVUP(rows, tile_dim_m);
    if (row_tiles_per_tensor == 0) {
      return;
    }
    tensor_id = blockIdx.y / row_tiles_per_tensor;
    if (tensor_id >= num_tensors) {
      return;
    }
    cols = static_cast<size_t>(last_dims_ptr[tensor_id]);
    tensor_base = static_cast<size_t>(offsets_ptr[tensor_id]);
    tile_row = (blockIdx.y - tensor_id * row_tiles_per_tensor) * tile_dim_m;
    if (tile_col >= cols) {
      return;
    }
  } else {
    if (tile_col >= last_logical_dim) {
      return;
    }
    // The grid is sized to the (possibly over-allocated) logical first dim, so
    // for capacity-padded buffers up to half the Y blocks fall past the active
    // region. Bound the active tile count from offsets_ptr[num_tensors] (the
    // active element total, one cached load) and drop those blocks in O(1)
    // instead of paying the full O(num_tensors) scan below. This is graph-safe:
    // it reads device offsets rather than requiring a host-side active count.
    const size_t active_rows = static_cast<size_t>(offsets_ptr[num_tensors]) / last_logical_dim;
    const size_t max_active_tiles = DIVUP(active_rows, tile_dim_m) + num_tensors;
    if (blockIdx.y >= max_active_tiles) {
      return;
    }
    size_t local_tile_id = blockIdx.y;
    bool found_tensor = false;
    for (size_t i = 0; i < num_tensors; ++i) {
      const size_t tensor_rows = static_cast<size_t>(first_dims_ptr[i]);
      const size_t tensor_tiles = DIVUP(tensor_rows, tile_dim_m);
      if (local_tile_id < tensor_tiles) {
        tensor_id = i;
        rows = tensor_rows;
        tensor_base = static_cast<size_t>(offsets_ptr[i]);
        tile_row = local_tile_id * tile_dim_m;
        found_tensor = true;
        break;
      }
      local_tile_id -= tensor_tiles;
    }
    if (!found_tensor) {
      return;
    }
  }

  if (rows == 0 || cols == 0) {
    return;
  }
  if (tile_row >= rows) {
    return;
  }

  const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];
  const size_t tid = threadIdx.x;
  const size_t tidx = tid % THREADS_PER_WARP;
  const size_t tidy = tid / THREADS_PER_WARP;

  if constexpr (COLWISE_OUTPUT) {
    __shared__ OVecT
        shared_output_t[nvec_in][THREADS_PER_WARP][THREADS_PER_WARP + TRANSPOSE_SHARED_PAD];

    // Interior tiles (every fragment full) can skip *all* per-fragment bounds
    // checks. That instruction overhead -- not occupancy -- is what holds the
    // generic kernel to ~58% of HBM; the branchless path below is HBM-bound.
    // The condition is block-uniform (derived from tile_row/tile_col/rows/cols),
    // so there is no warp divergence: only edge tiles (partial trailing columns,
    // or the last row tile of a group) fall through to the bounds-checked path.
    //
    // Alignment is split between the two phases because they have independent
    // requirements. Phase 1 (load bf16 + convert + optional rowwise store +
    // transpose into shared) is governed by the *column* stride and is the bulk
    // of global traffic (a 2-byte read per element). Phase 2 (store the
    // transposed fp8 columns) is governed by the per-group *row* count, which
    // for varying-first groups is an arbitrary integer and is therefore rarely
    // a multiple of nvec_out. We still take the branchless interior path
    // whenever the loads are aligned -- the common case -- and only the final
    // store falls back to per-element stores when the row stride is unaligned,
    // so imbalanced group shapes keep the branchless phase-1 win.
    const bool full_tile = (tile_row + tile_dim_m <= rows) && (tile_col + tile_dim_n <= cols);
    const bool aligned_load = (cols % nvec_in == 0) && (tensor_base % nvec_in == 0);
    if (full_tile && aligned_load) {
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidy + iter * WARPS_PER_TILE;
        const size_t j1 = tidx;
        const size_t base_row = tile_row + i1 * nvec_out;
        const size_t base_col = tile_col + j1 * nvec_in;
        OVecT local_output_t[nvec_in];
#pragma unroll
        for (size_t i2 = 0; i2 < nvec_out; ++i2) {
          const size_t row = base_row + i2;
          IVecT local_input;
          OVecC local_output;
          local_input.load_from(input + tensor_base + row * cols + base_col);
          scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
          if constexpr (ROWWISE_OUTPUT) {
            OType *const output_ptr = output_rowwise + tensor_base + row * cols + base_col;
            if constexpr (SCALING_TYPE == ScalingType::BIDIMENSIONAL) {
              store_fp8_vec_streaming(output_ptr, local_output);
            } else {
              local_output.store_to(output_ptr);
            }
          }
#pragma unroll
          for (size_t j2 = 0; j2 < nvec_in; ++j2) {
            local_output_t[j2].data.elt[i2] = local_output.data.elt[j2];
          }
        }
#pragma unroll
        for (size_t j2 = 0; j2 < nvec_in; ++j2) {
          shared_output_t[j2][j1][i1] = local_output_t[j2];
        }
      }
      __syncthreads();
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
        for (size_t iter = 0; iter < num_iterations; ++iter) {
          // All 32 lanes of a warp share j1 (= tidy + iter*WARPS_PER_TILE) and
          // therefore the same column, so shared_output_t[j2][j1][0..WARP-1] is a
          // contiguous nvec_out*WARP-byte segment in shared and maps to a
          // contiguous segment of one output column. Store it with the widest
          // type the (warp-uniform) destination offset allows: row strides that
          // are multiples of nvec_out give a single vectorized store per lane;
          // arbitrary varying-first strides degrade gracefully to narrower
          // coalesced stores instead of faulting.
          const size_t j1 = tidy + iter * WARPS_PER_TILE;
          const size_t col = tile_col + j1 * nvec_in + j2;
          char *const gl =
              reinterpret_cast<char *>(output_colwise + tensor_base + col * rows + tile_row);
          store_column_segment(&shared_output_t[j2][j1][0], gl, tidx);
        }
      }
      return;
    }

#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * WARPS_PER_TILE;
      const size_t j1 = tidx;
      const size_t base_row = tile_row + i1 * nvec_out;
      const size_t base_col = tile_col + j1 * nvec_in;
      const size_t fragment_cols =
          base_col < cols ? ((cols - base_col) < nvec_in ? (cols - base_col) : nvec_in) : 0;
      if (base_row >= rows || fragment_cols == 0) {
        continue;
      }
      const bool full_fragment = (base_row + nvec_out <= rows) && (fragment_cols == nvec_in);
      OVecT local_output_t[nvec_in];
#pragma unroll
      for (size_t i2 = 0; i2 < nvec_out; ++i2) {
        const size_t row = base_row + i2;
        if (full_fragment || row < rows) {
          IVecT local_input;
          OVecC local_output;
          if (full_fragment) {
            const IType *const input_ptr = input + tensor_base + row * cols + base_col;
            if (reinterpret_cast<uint64_t>(input_ptr) % IVecT::BYTES == 0) {
              local_input.load_from(input_ptr);
            } else {
              local_input.load_from_elts(input_ptr);
            }
            scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
            if constexpr (ROWWISE_OUTPUT) {
              OType *const output_ptr = output_rowwise + tensor_base + row * cols + base_col;
              if constexpr (SCALING_TYPE == ScalingType::BIDIMENSIONAL) {
                if (reinterpret_cast<uint64_t>(output_ptr) % OVecC::BYTES == 0) {
                  store_fp8_vec_streaming(output_ptr, local_output);
                } else {
                  local_output.store_to_elts(output_ptr);
                }
              } else if (reinterpret_cast<uint64_t>(output_ptr) % OVecC::BYTES == 0) {
                local_output.store_to(output_ptr);
              } else {
                local_output.store_to_elts(output_ptr);
              }
            }
          } else {
            local_input.load_from_elts(input + tensor_base + row * cols + base_col, 0,
                                       fragment_cols);
            scaled_fp8_cvt_vec<IS_ACT, ParamOP, OP>(local_input, local_output, fragment_cols,
                                                    scale);
            if constexpr (ROWWISE_OUTPUT) {
              local_output.store_to_elts(output_rowwise + tensor_base + row * cols + base_col, 0,
                                         fragment_cols);
            }
          }
#pragma unroll
          for (size_t j2 = 0; j2 < nvec_in; ++j2) {
            if (j2 < fragment_cols) {
              local_output_t[j2].data.elt[i2] = local_output.data.elt[j2];
            }
          }
        }
      }
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
        if (j2 < fragment_cols) {
          shared_output_t[j2][j1][i1] = local_output_t[j2];
        }
      }
    }

    __syncthreads();
#pragma unroll
    for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidx;
        const size_t j1 = tidy + iter * WARPS_PER_TILE;
        const size_t row = tile_row + i1 * nvec_out;
        const size_t col = tile_col + j1 * nvec_in + j2;
        if (col < cols) {
          const size_t valid_rows =
              row < rows ? ((rows - row) < nvec_out ? (rows - row) : nvec_out) : 0;
          if (valid_rows > 0) {
            OType *const output_ptr = output_colwise + tensor_base + col * rows + row;
            const OVecT local_output_t = shared_output_t[j2][j1][i1];
            if (valid_rows == nvec_out &&
                reinterpret_cast<uint64_t>(output_ptr) % OVecT::BYTES == 0) {
              local_output_t.store_to(output_ptr);
            } else {
              local_output_t.store_to_elts(output_ptr, 0, valid_rows);
            }
          }
        }
      }
    }
  } else {
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * WARPS_PER_TILE;
      const size_t j1 = tidx;
#pragma unroll
      for (size_t i2 = 0; i2 < nvec_out; ++i2) {
        const size_t row = tile_row + i1 * nvec_out + i2;
        const size_t col = tile_col + j1 * nvec_in;
        if (row < rows) {
          IVecT local_input;
          OVecC local_output;
          if (col + nvec_in <= cols) {
            const IType *const input_ptr = input + tensor_base + row * cols + col;
            if (reinterpret_cast<uint64_t>(input_ptr) % IVecT::BYTES == 0) {
              local_input.load_from(input_ptr);
            } else {
              local_input.load_from_elts(input_ptr);
            }
            scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
            OType *const output_ptr = output_rowwise + tensor_base + row * cols + col;
            if (reinterpret_cast<uint64_t>(output_ptr) % OVecC::BYTES == 0) {
              local_output.store_to(output_ptr);
            } else {
              local_output.store_to_elts(output_ptr);
            }
          } else {
            const size_t valid_cols =
                col < cols ? ((cols - col) < nvec_in ? (cols - col) : nvec_in) : 0;
            if (valid_cols == 0) {
              continue;
            }
            local_input.load_from_elts(input + tensor_base + row * cols + col, 0, valid_cols);
            scaled_fp8_cvt_vec<IS_ACT, ParamOP, OP>(local_input, local_output, valid_cols, scale);
            local_output.store_to_elts(output_rowwise + tensor_base + row * cols + col, 0,
                                       valid_cols);
          }
        }
      }
    }
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType>
__global__ void __launch_bounds__(ROWWISE_FLAT_THREADS)
    group_cast_fp8_rowwise_flat_kernel(const IType *__restrict__ input,
                                       OType *__restrict__ output_rowwise,
                                       const float *__restrict__ scale_ptr,
                                       const float *__restrict__ noop, const size_t rows_per_tensor,
                                       const size_t cols, const size_t vecs_per_tensor) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr size_t nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
  using IVecT = Vec<IType, nvec>;
  using OVecT = Vec<OType, nvec>;

  const size_t tensor_id = blockIdx.y;
  const size_t tensor_base = tensor_id * rows_per_tensor * cols;
  const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];

  for (size_t vec_id = blockIdx.x * blockDim.x + threadIdx.x; vec_id < vecs_per_tensor;
       vec_id += gridDim.x * blockDim.x) {
    const size_t offset = tensor_base + vec_id * nvec;
    IVecT local_input;
    OVecT local_output;
    local_input.load_from(input + offset);
    scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
    local_output.store_to(output_rowwise + offset);
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, bool ALIGNED>
__global__ void __launch_bounds__(ROWWISE_FLAT_THREADS)
    group_cast_fp8_varying_first_rowwise_flat_kernel(const IType *__restrict__ input,
                                                     OType *__restrict__ output_rowwise,
                                                     const float *__restrict__ scale_ptr,
                                                     const float *__restrict__ noop,
                                                     const size_t num_tensors,
                                                     const size_t total_elements,
                                                     const int64_t *__restrict__ offsets_ptr) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr size_t nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
  using IVecT = Vec<IType, nvec>;
  using OVecT = Vec<OType, nvec>;

  const size_t tensor_id = blockIdx.y;
  if (tensor_id >= num_tensors) {
    return;
  }

  const size_t raw_tensor_base = static_cast<size_t>(offsets_ptr[tensor_id]);
  const size_t raw_tensor_end = static_cast<size_t>(offsets_ptr[tensor_id + 1]);
  const size_t tensor_base = raw_tensor_base < total_elements ? raw_tensor_base : total_elements;
  const size_t tensor_end = raw_tensor_end < total_elements ? raw_tensor_end : total_elements;
  if (tensor_end <= tensor_base) {
    return;
  }

  const size_t tensor_elements = tensor_end - tensor_base;
  const size_t tensor_vecs = DIVUP(tensor_elements, nvec);
  const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];

  for (size_t local_vec_id = blockIdx.x * blockDim.x + threadIdx.x; local_vec_id < tensor_vecs;
       local_vec_id += gridDim.x * blockDim.x) {
    const size_t offset = tensor_base + local_vec_id * nvec;
    const size_t remaining_elts = tensor_end - offset;
    IVecT local_input;
    OVecT local_output;
    if (remaining_elts >= nvec) {
      if constexpr (ALIGNED) {
        local_input.load_from(input + offset);
      } else {
        local_input.load_from_elts(input + offset);
      }
      scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
      if constexpr (ALIGNED) {
        local_output.store_to(output_rowwise + offset);
      } else {
        local_output.store_to_elts(output_rowwise + offset);
      }
    } else {
      const size_t valid_elts = remaining_elts < nvec ? remaining_elts : nvec;
#pragma unroll
      for (size_t i = 0; i < nvec; ++i) {
        if (i < valid_elts) {
          float elt = static_cast<float>(input[offset + i]);
          if constexpr (IS_ACT) {
            elt = OP(elt, {});
          }
          local_output.data.elt[i] = static_cast<OType>(elt * scale);
        }
      }
      local_output.store_to_elts(output_rowwise + offset, 0, valid_elts);
    }
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType>
__global__ void __launch_bounds__(ROWWISE_FLAT_THREADS)
    group_cast_fp8_varying_first_rowwise_aligned_flat_kernel(
        const IType *__restrict__ input, OType *__restrict__ output_rowwise,
        const float *__restrict__ scale_ptr, const float *__restrict__ noop,
        const size_t num_tensors, const size_t total_elements,
        const int64_t *__restrict__ offsets_ptr, const size_t target_blocks) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr size_t nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
  using IVecT = Vec<IType, nvec>;
  using OVecT = Vec<OType, nvec>;

  // Dynamic shared memory layout:
  // s_offsets: int64_t[num_tensors + 1]
  // s_block_offsets: int[num_tensors + 1]
  extern __shared__ char s_mem[];
  int64_t *s_offsets = reinterpret_cast<int64_t *>(s_mem);
  int *s_block_offsets = reinterpret_cast<int *>(s_mem + (num_tensors + 1) * sizeof(int64_t));

  const size_t tid = threadIdx.x;

  // 1. Copy offsets to shared memory
  for (size_t i = tid; i <= num_tensors; i += blockDim.x) {
    s_offsets[i] = offsets_ptr[i];
  }
  __syncthreads();

  // Derive the per-block chunk size from the *active* element count
  // (offsets_ptr[num_tensors]) rather than the over-allocated buffer, so that
  // over-allocation does not inflate the chunk and reduce the number of active
  // blocks. Every thread computes the same value from shared memory.
  const size_t active_elements = static_cast<size_t>(s_offsets[num_tensors]);
  size_t block_chunk_size = DIVUP(active_elements, target_blocks);
  block_chunk_size = DIVUP(block_chunk_size, nvec) * nvec;
  if (block_chunk_size < 8192) {
    block_chunk_size = 8192;
  }

  // 2. Compute block counts and prefix sum
  if (tid == 0) {
    int sum = 0;
    for (size_t i = 0; i < num_tensors; ++i) {
      s_block_offsets[i] = sum;
      size_t tensor_size = s_offsets[i + 1] - s_offsets[i];
      int blocks = (tensor_size + block_chunk_size - 1) / block_chunk_size;
      sum += blocks;
    }
    s_block_offsets[num_tensors] = sum;
  }
  __syncthreads();

  // Guard against blocks that are out of bounds of the active work
  if (blockIdx.x >= s_block_offsets[num_tensors]) {
    return;
  }

  // 3. Binary search to find which tensor this block belongs to
  size_t tensor_id = 0;
  {
    size_t low = 0;
    size_t hi = num_tensors;
    while (low < hi) {
      size_t mid = low + (hi - low) / 2;
      if (s_block_offsets[mid + 1] <= blockIdx.x) {
        low = mid + 1;
      } else {
        hi = mid;
      }
    }
    tensor_id = low;
  }

  // 4. Calculate this block's local work range within the tensor
  const size_t local_block_id = blockIdx.x - s_block_offsets[tensor_id];
  const size_t tensor_base = s_offsets[tensor_id];
  const size_t tensor_size = s_offsets[tensor_id + 1] - tensor_base;
  const size_t start_elt = local_block_id * block_chunk_size;
  const size_t end_elt =
      start_elt + block_chunk_size < tensor_size ? start_elt + block_chunk_size : tensor_size;

  if (start_elt >= end_elt) {
    return;
  }

  const IType *base_input = input + tensor_base + start_elt;
  OType *base_output = output_rowwise + tensor_base + start_elt;
  const size_t numel = end_elt - start_elt;
  const size_t tensor_vecs = numel / nvec;
  const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];

  for (size_t local_vec_id = tid; local_vec_id < tensor_vecs; local_vec_id += blockDim.x) {
    const size_t offset = local_vec_id * nvec;
    IVecT local_input;
    OVecT local_output;
    local_input.load_from(base_input + offset);
    scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
    local_output.store_to(base_output + offset);
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType>
__global__ void __launch_bounds__(ROWWISE_FLAT_THREADS) group_cast_fp8_variable_rowwise_flat_kernel(
    const IType *__restrict__ input, OType *__restrict__ output_rowwise,
    const float *__restrict__ scale_ptr, const float *__restrict__ noop, const size_t num_tensors,
    const size_t total_elements, const int64_t *__restrict__ offsets_ptr) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr size_t nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
  using IVecT = Vec<IType, nvec>;
  using OVecT = Vec<OType, nvec>;

  const size_t active_elements = static_cast<size_t>(offsets_ptr[num_tensors]);
  const size_t bounded_elements =
      active_elements < total_elements ? active_elements : total_elements;
  const size_t total_vecs = DIVUP(bounded_elements, nvec);
  for (size_t vec_id = blockIdx.x * blockDim.x + threadIdx.x; vec_id < total_vecs;
       vec_id += gridDim.x * blockDim.x) {
    const size_t offset = vec_id * nvec;
    const size_t tensor_id = transformer_engine::dispatch::common::find_tensor_from_offsets(
        offsets_ptr, num_tensors, offset);
    const size_t tensor_end = static_cast<size_t>(offsets_ptr[tensor_id + 1]);
    const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];
    IVecT local_input;
    OVecT local_output;
    if (offset + nvec <= tensor_end && offset + nvec <= bounded_elements &&
        reinterpret_cast<uint64_t>(input + offset) % IVecT::BYTES == 0 &&
        reinterpret_cast<uint64_t>(output_rowwise + offset) % OVecT::BYTES == 0) {
      local_input.load_from(input + offset);
      scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
      local_output.store_to(output_rowwise + offset);
    } else {
      const size_t valid_elts =
          offset < bounded_elements
              ? ((bounded_elements - offset) < nvec ? (bounded_elements - offset) : nvec)
              : 0;
      if (valid_elts == 0) {
        continue;
      }
#pragma unroll
      for (size_t i = 0; i < nvec; ++i) {
        if (i < valid_elts) {
          const size_t elt_offset = offset + i;
          const size_t elt_tensor_id =
              elt_offset < tensor_end
                  ? tensor_id
                  : transformer_engine::dispatch::common::find_tensor_from_offsets(
                        offsets_ptr, num_tensors, elt_offset);
          const float elt_scale = scale_ptr == nullptr ? 1.0f : scale_ptr[elt_tensor_id];
          float elt = static_cast<float>(input[elt_offset]);
          if constexpr (IS_ACT) {
            elt = OP(elt, {});
          }
          local_output.data.elt[i] = static_cast<OType>(elt * elt_scale);
        }
      }
      local_output.store_to_elts(output_rowwise + offset, 0, valid_elts);
    }
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, ScalingType SCALING_TYPE>
__global__ void __launch_bounds__(THREADS_PER_TILE)
    group_cast_fp8_same_shape_full_tile_kernel(const IType *__restrict__ input,
                                               OType *__restrict__ output_rowwise,
                                               OType *__restrict__ output_colwise,
                                               const float *__restrict__ scale_ptr,
                                               const float *__restrict__ noop,
                                               const size_t rows_per_tensor, const size_t cols) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr bool ROWWISE_OUTPUT =
      (SCALING_TYPE == ScalingType::ROWWISE) || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  constexpr bool COLWISE_OUTPUT =
      (SCALING_TYPE == ScalingType::COLWISE) || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  static_assert(COLWISE_OUTPUT, "The full-tile grouped FP8 kernel is for columnwise outputs.");

  constexpr size_t nvec_in = TRANSPOSE_LOAD_SIZE_BYTES / sizeof(IType);
  constexpr size_t nvec_out = TRANSPOSE_STORE_SIZE_BYTES / sizeof(OType);
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
  constexpr size_t num_iterations = THREADS_PER_WARP / WARPS_PER_TILE;

  using IVecT = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  const size_t tiles_per_tensor = rows_per_tensor / tile_dim_m;
  const size_t tensor_id = blockIdx.y / tiles_per_tensor;
  const size_t tensor_tile_id = blockIdx.y - tensor_id * tiles_per_tensor;
  const size_t tensor_base = tensor_id * rows_per_tensor * cols;
  const size_t tile_row = tensor_tile_id * tile_dim_m;
  const size_t tile_col = blockIdx.x * tile_dim_n;
  const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];

  const size_t tid = threadIdx.x;
  const size_t tidx = tid % THREADS_PER_WARP;
  const size_t tidy = tid / THREADS_PER_WARP;

  __shared__ OVecT
      shared_output_t[nvec_in][THREADS_PER_WARP][THREADS_PER_WARP + TRANSPOSE_SHARED_PAD];

#pragma unroll
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    const size_t i1 = tidy + iter * WARPS_PER_TILE;
    const size_t j1 = tidx;
    const size_t base_row = tile_row + i1 * nvec_out;
    const size_t base_col = tile_col + j1 * nvec_in;
    OVecT local_output_t[nvec_in];
#pragma unroll
    for (size_t i2 = 0; i2 < nvec_out; ++i2) {
      const size_t row = base_row + i2;
      IVecT local_input;
      OVecC local_output;
      const IType *const input_ptr = input + tensor_base + row * cols + base_col;
      local_input.load_from(input_ptr);
      scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
      if constexpr (ROWWISE_OUTPUT) {
        OType *const output_ptr = output_rowwise + tensor_base + row * cols + base_col;
        if constexpr (SCALING_TYPE == ScalingType::BIDIMENSIONAL) {
          store_fp8_vec_streaming(output_ptr, local_output);
        } else {
          local_output.store_to(output_ptr);
        }
      }
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
        local_output_t[j2].data.elt[i2] = local_output.data.elt[j2];
      }
    }
#pragma unroll
    for (size_t j2 = 0; j2 < nvec_in; ++j2) {
      shared_output_t[j2][j1][i1] = local_output_t[j2];
    }
  }

  __syncthreads();
#pragma unroll
  for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidx;
      const size_t j1 = tidy + iter * WARPS_PER_TILE;
      const size_t row = tile_row + i1 * nvec_out;
      const size_t col = tile_col + j1 * nvec_in + j2;
      OType *const output_ptr = output_colwise + tensor_base + col * rows_per_tensor + row;
      const OVecT local_output_t = shared_output_t[j2][j1][i1];
      local_output_t.store_to(output_ptr);
    }
  }
}

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, ScalingType SCALING_TYPE>
__global__ void __launch_bounds__(THREADS_PER_TILE) group_cast_fp8_varying_first_tile_kernel(
    const IType *__restrict__ input, OType *__restrict__ output_rowwise,
    OType *__restrict__ output_colwise, const float *__restrict__ scale_ptr,
    const float *__restrict__ noop, const size_t num_tensors, const size_t rows_upper_bound,
    const size_t cols, const size_t total_elements, const int64_t *__restrict__ offsets_ptr,
    const int64_t *__restrict__ first_dims_ptr) {
  if (noop != nullptr && noop[0] == 1.0f) {
    return;
  }

  constexpr bool ROWWISE_OUTPUT =
      (SCALING_TYPE == ScalingType::ROWWISE) || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  constexpr bool COLWISE_OUTPUT =
      (SCALING_TYPE == ScalingType::COLWISE) || (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
  static_assert(COLWISE_OUTPUT, "The varying-first tile kernel is for columnwise outputs.");

  constexpr size_t nvec_in = TRANSPOSE_LOAD_SIZE_BYTES / sizeof(IType);
  constexpr size_t nvec_out = TRANSPOSE_STORE_SIZE_BYTES / sizeof(OType);
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
  constexpr size_t num_iterations = THREADS_PER_WARP / WARPS_PER_TILE;

  using IVecT = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  const size_t tensor_id = blockIdx.z;
  if (tensor_id >= num_tensors || cols == 0) {
    return;
  }

  const size_t raw_tensor_base = static_cast<size_t>(offsets_ptr[tensor_id]);
  const size_t raw_tensor_end = static_cast<size_t>(offsets_ptr[tensor_id + 1]);
  const size_t tensor_base = raw_tensor_base < total_elements ? raw_tensor_base : total_elements;
  const size_t tensor_end = raw_tensor_end < total_elements ? raw_tensor_end : total_elements;
  if (tensor_end <= tensor_base) {
    return;
  }

  const size_t rows_from_offsets = (tensor_end - tensor_base) / cols;
  const size_t rows_from_dims = static_cast<size_t>(first_dims_ptr[tensor_id]);
  size_t rows = rows_from_dims < rows_from_offsets ? rows_from_dims : rows_from_offsets;
  rows = rows < rows_upper_bound ? rows : rows_upper_bound;
  if (rows == 0) {
    return;
  }

  const size_t tile_col = blockIdx.x * tile_dim_n;
  if (tile_col >= cols) {
    return;
  }
  const size_t row_tiles = DIVUP(rows, tile_dim_m);
  const float scale = scale_ptr == nullptr ? 1.0f : scale_ptr[tensor_id];
  const size_t tid = threadIdx.x;
  const size_t tidx = tid % THREADS_PER_WARP;
  const size_t tidy = tid / THREADS_PER_WARP;

  __shared__ OVecT
      shared_output_t[nvec_in][THREADS_PER_WARP][THREADS_PER_WARP + TRANSPOSE_SHARED_PAD];

  for (size_t tensor_tile_id = blockIdx.y; tensor_tile_id < row_tiles;
       tensor_tile_id += gridDim.y) {
    const size_t tile_row = tensor_tile_id * tile_dim_m;

#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * WARPS_PER_TILE;
      const size_t j1 = tidx;
      const size_t base_row = tile_row + i1 * nvec_out;
      const size_t base_col = tile_col + j1 * nvec_in;
      const size_t fragment_cols =
          base_col < cols ? ((cols - base_col) < nvec_in ? (cols - base_col) : nvec_in) : 0;
      if (base_row >= rows || fragment_cols == 0) {
        continue;
      }
      const bool full_fragment = (base_row + nvec_out <= rows) && (fragment_cols == nvec_in);
      OVecT local_output_t[nvec_in];
#pragma unroll
      for (size_t i2 = 0; i2 < nvec_out; ++i2) {
        const size_t row = base_row + i2;
        if (full_fragment || row < rows) {
          IVecT local_input;
          OVecC local_output;
          if (full_fragment) {
            const IType *const input_ptr = input + tensor_base + row * cols + base_col;
            if (reinterpret_cast<uint64_t>(input_ptr) % IVecT::BYTES == 0) {
              local_input.load_from(input_ptr);
            } else {
              local_input.load_from_elts(input_ptr);
            }
            scaled_fp8_cvt_vec_full<IS_ACT, ParamOP, OP>(local_input, local_output, scale);
            if constexpr (ROWWISE_OUTPUT) {
              OType *const output_ptr = output_rowwise + tensor_base + row * cols + base_col;
              if constexpr (SCALING_TYPE == ScalingType::BIDIMENSIONAL) {
                if (reinterpret_cast<uint64_t>(output_ptr) % OVecC::BYTES == 0) {
                  store_fp8_vec_streaming(output_ptr, local_output);
                } else {
                  local_output.store_to_elts(output_ptr);
                }
              } else if (reinterpret_cast<uint64_t>(output_ptr) % OVecC::BYTES == 0) {
                local_output.store_to(output_ptr);
              } else {
                local_output.store_to_elts(output_ptr);
              }
            }
          } else {
            local_input.load_from_elts(input + tensor_base + row * cols + base_col, 0,
                                       fragment_cols);
            scaled_fp8_cvt_vec<IS_ACT, ParamOP, OP>(local_input, local_output, fragment_cols,
                                                    scale);
            if constexpr (ROWWISE_OUTPUT) {
              local_output.store_to_elts(output_rowwise + tensor_base + row * cols + base_col, 0,
                                         fragment_cols);
            }
          }
#pragma unroll
          for (size_t j2 = 0; j2 < nvec_in; ++j2) {
            if (j2 < fragment_cols) {
              local_output_t[j2].data.elt[i2] = local_output.data.elt[j2];
            }
          }
        }
      }
#pragma unroll
      for (size_t j2 = 0; j2 < nvec_in; ++j2) {
        if (j2 < fragment_cols) {
          shared_output_t[j2][j1][i1] = local_output_t[j2];
        }
      }
    }

    __syncthreads();
#pragma unroll
    for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidx;
        const size_t j1 = tidy + iter * WARPS_PER_TILE;
        const size_t row = tile_row + i1 * nvec_out;
        const size_t col = tile_col + j1 * nvec_in + j2;
        if (col < cols) {
          const size_t valid_rows =
              row < rows ? ((rows - row) < nvec_out ? (rows - row) : nvec_out) : 0;
          if (valid_rows > 0) {
            OType *const output_ptr = output_colwise + tensor_base + col * rows + row;
            const OVecT local_output_t = shared_output_t[j2][j1][i1];
            if (valid_rows == nvec_out &&
                reinterpret_cast<uint64_t>(output_ptr) % OVecT::BYTES == 0) {
              local_output_t.store_to(output_ptr);
            } else {
              local_output_t.store_to_elts(output_ptr, 0, valid_rows);
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace group_quantize_kernel

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
void group_quantize(const GroupedTensor *input, const Tensor *noop, GroupedTensor *output,
                    const QuantizationConfig *quant_config, cudaStream_t stream) {
  using namespace group_quantize_kernel;
  (void)quant_config;

  CheckNoopTensor(*noop, "cast_noop");
  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");
  NVTE_CHECK(is_fp8_dtype(output->dtype()), "Output must have FP8 type.");
  NVTE_CHECK(output->scale.has_data(), "Grouped FP8 tensor-scaling output scale must be set.");
  NVTE_CHECK(output->scale.dtype == DType::kFloat32,
             "Grouped FP8 tensor-scaling scale must be FP32.");
  NVTE_CHECK(output->scale.numel() >= output->num_tensors,
             "Grouped FP8 tensor-scaling scale must have at least one entry per tensor.");
  const bool use_rowwise_output = output->has_data();
  const bool use_colwise_output = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise_output || use_colwise_output,
             "Either rowwise or columnwise output data need to be allocated.");
  if (use_rowwise_output) {
    NVTE_CHECK(output->scale_inv.has_data(), "Rowwise scale_inv must be allocated.");
  }
  if (use_colwise_output) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(), "Columnwise scale_inv must be allocated.");
  }

  ScalingType scaling_type = ScalingType::BIDIMENSIONAL;
  if (!use_colwise_output) {
    scaling_type = ScalingType::ROWWISE;
  } else if (!use_rowwise_output) {
    scaling_type = ScalingType::COLWISE;
  }

  ShapeRepresentation shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  if (output->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (output->all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else {
    NVTE_ERROR(
        "Grouped FP8 tensor-scaling quantization only supports same-shape, varying "
        "first-dimension, or varying last-dimension grouped tensors.");
  }

  NVTE_CHECK(input->logical_shape.data[1] == output->logical_shape.data[1],
             "Grouped FP8 tensor-scaling input and output must have the same last dimension.");
  NVTE_CHECK(output->logical_shape.data[0] <= input->logical_shape.data[0],
             "Grouped FP8 tensor-scaling output first dimension must not exceed input first "
             "dimension.");

  // For varying-dim grouped tensors, logical_shape may be larger than the active region
  // (sum of first_dims for varying-first, sum of last_dims for varying-last). The backing
  // buffer is sized to logical_shape, but the kernel must only touch the active rows/cols;
  // metadata (first_dims/last_dims/tensor_offsets) is consulted on device to skip the unused
  // tail. We size the grid from logical_shape so the launch covers every potential payload
  // row, and rely on per-block bounds checks to drop blocks past the active region.
  const size_t first_logical_dim = output->logical_shape.data[0];
  const size_t last_logical_dim = output->logical_shape.data[1];
  const size_t num_tensors = input->num_tensors;
  if (first_logical_dim == 0 || last_logical_dim == 0) {
    return;
  }
  const size_t store_size_bytes =
      use_colwise_output ? TRANSPOSE_STORE_SIZE_BYTES : ROWWISE_STORE_SIZE_BYTES;
  const size_t row_tile_size = THREADS_PER_WARP * store_size_bytes;
  size_t work_blocks_Y = DIVUP(first_logical_dim, row_tile_size);
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    NVTE_CHECK(first_logical_dim % num_tensors == 0,
               "First logical dimension must be divisible by num_tensors.");
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    work_blocks_Y = num_tensors * DIVUP(rows_per_tensor, row_tile_size);
  } else if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    // first_dims live on device; over-allocate the grid enough to cover one partial tile per group.
    work_blocks_Y += num_tensors;
  } else if (shape_rep == ShapeRepresentation::VARYING_LAST_DIM) {
    // last_dims live on device; map Y to per-group row tiles and over-allocate X by total width.
    work_blocks_Y = num_tensors * DIVUP(first_logical_dim, row_tile_size);
  }

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
  const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(output->last_dims.dptr);
  const float *noop_ptr = reinterpret_cast<const float *>(noop->data.dptr);

  const dim3 block(THREADS_PER_TILE);

  const float *const scale_ptr = reinterpret_cast<const float *>(output->scale.dptr);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input->dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          output->dtype(), OType,
          TRANSFORMER_ENGINE_SCALING_TYPE_SWITCH(
              scaling_type, SCALING_TYPE,
              {
                constexpr bool colwise_output = (SCALING_TYPE == ScalingType::COLWISE) ||
                                                (SCALING_TYPE == ScalingType::BIDIMENSIONAL);
                constexpr size_t load_size_bytes =
                    colwise_output ? TRANSPOSE_LOAD_SIZE_BYTES : ROWWISE_LOAD_SIZE_BYTES;
                constexpr size_t nvec_in = load_size_bytes / sizeof(IType);
                constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
                TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(shape_rep, SHAPE_REP, {
                  if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
                    NVTE_CHECK(offsets_ptr != nullptr && first_dims_ptr != nullptr,
                               "Varying first-dimension grouped FP8 quantization requires "
                               "first_dims and tensor_offsets.");
                  } else if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_LAST_DIM) {
                    NVTE_CHECK(offsets_ptr != nullptr && last_dims_ptr != nullptr,
                               "Varying last-dimension grouped FP8 quantization requires "
                               "last_dims and tensor_offsets.");
                  }
                  bool launched_fast_path = false;
                  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
                    const size_t rows_per_tensor = first_logical_dim / num_tensors;
                    if constexpr (SCALING_TYPE == ScalingType::ROWWISE) {
                      constexpr size_t flat_nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
                      using IVecT = Vec<IType, flat_nvec>;
                      using OVecT = Vec<OType, flat_nvec>;
                      const size_t elems_per_tensor = rows_per_tensor * last_logical_dim;
                      const bool flat_aligned =
                          reinterpret_cast<uintptr_t>(input->data.dptr) % IVecT::BYTES == 0 &&
                          reinterpret_cast<uintptr_t>(output->data.dptr) % OVecT::BYTES == 0;
                      if (elems_per_tensor % flat_nvec == 0 && flat_aligned) {
                        const size_t vecs_per_tensor = elems_per_tensor / flat_nvec;
                        const dim3 flat_block(ROWWISE_FLAT_THREADS);
                        const dim3 flat_grid(DIVUP(vecs_per_tensor, ROWWISE_FLAT_THREADS),
                                             num_tensors);
                        group_cast_fp8_rowwise_flat_kernel<IS_ACT, ParamOP, OP, IType, OType>
                            <<<flat_grid, flat_block, 0, stream>>>(
                                reinterpret_cast<const IType *>(input->data.dptr),
                                reinterpret_cast<OType *>(output->data.dptr), scale_ptr, noop_ptr,
                                rows_per_tensor, last_logical_dim, vecs_per_tensor);
                        launched_fast_path = true;
                      }
                    } else if constexpr (colwise_output) {
                      constexpr size_t store_size_bytes = TRANSPOSE_STORE_SIZE_BYTES;
                      constexpr size_t nvec_out = store_size_bytes / sizeof(OType);
                      constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
                      if (rows_per_tensor % tile_dim_m == 0 && last_logical_dim % tile_dim_n == 0) {
                        const dim3 full_grid(last_logical_dim / tile_dim_n,
                                             num_tensors * (rows_per_tensor / tile_dim_m));
                        group_cast_fp8_same_shape_full_tile_kernel<IS_ACT, ParamOP, OP, IType,
                                                                   OType, SCALING_TYPE>
                            <<<full_grid, block, 0, stream>>>(
                                reinterpret_cast<const IType *>(input->data.dptr),
                                use_rowwise_output ? reinterpret_cast<OType *>(output->data.dptr)
                                                   : nullptr,
                                reinterpret_cast<OType *>(output->columnwise_data.dptr), scale_ptr,
                                noop_ptr, rows_per_tensor, last_logical_dim);
                        launched_fast_path = true;
                      }
                    }
                  }
                  if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_LAST_DIM) {
                    if constexpr (SCALING_TYPE == ScalingType::ROWWISE) {
                      const size_t total_elements = first_logical_dim * last_logical_dim;
                      constexpr size_t flat_nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
                      const size_t total_vecs = DIVUP(total_elements, flat_nvec);
                      const dim3 flat_block(ROWWISE_FLAT_THREADS);
                      const dim3 flat_grid(DIVUP(total_vecs, ROWWISE_FLAT_THREADS));
                      group_cast_fp8_variable_rowwise_flat_kernel<IS_ACT, ParamOP, OP, IType, OType>
                          <<<flat_grid, flat_block, 0, stream>>>(
                              reinterpret_cast<const IType *>(input->data.dptr),
                              reinterpret_cast<OType *>(output->data.dptr), scale_ptr, noop_ptr,
                              num_tensors, total_elements, offsets_ptr);
                      launched_fast_path = true;
                    }
                  }
                  if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
                    if constexpr (SCALING_TYPE == ScalingType::ROWWISE) {
                      const size_t total_elements = first_logical_dim * last_logical_dim;
                      constexpr size_t flat_nvec = ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
                      using IVecT = Vec<IType, flat_nvec>;
                      using OVecT = Vec<OType, flat_nvec>;
                      const bool flat_aligned =
                          last_logical_dim % flat_nvec == 0 &&
                          reinterpret_cast<uintptr_t>(input->data.dptr) % IVecT::BYTES == 0 &&
                          reinterpret_cast<uintptr_t>(output->data.dptr) % OVecT::BYTES == 0;
                      const size_t rows_per_tensor_upper = DIVUP(first_logical_dim, num_tensors);
                      const size_t vecs_per_tensor_upper =
                          DIVUP(rows_per_tensor_upper * last_logical_dim, flat_nvec);
                      size_t blocks_per_tensor = DIVUP(vecs_per_tensor_upper, ROWWISE_FLAT_THREADS);
                      if (blocks_per_tensor == 0) {
                        blocks_per_tensor = 1;
                      }
                      if (blocks_per_tensor > VARYING_FIRST_ROWWISE_MAX_BLOCKS_PER_TENSOR) {
                        blocks_per_tensor = VARYING_FIRST_ROWWISE_MAX_BLOCKS_PER_TENSOR;
                      }
                      const dim3 flat_block(ROWWISE_FLAT_THREADS);
                      const dim3 flat_grid(blocks_per_tensor, num_tensors);
                      if (flat_aligned) {
                        const int num_sms = ::transformer_engine::cuda::sm_count();
                        // Target ~8 blocks per SM of *active* work. The actual
                        // per-block chunk size is derived inside the kernel from
                        // the device-side offsets (the active element count), so
                        // that an over-allocated backing buffer does not inflate
                        // the chunk and starve the grid of active blocks. We
                        // launch enough blocks to cover target_blocks plus up to
                        // one partial block per tensor (from per-tensor rounding).
                        const size_t target_blocks = 8 * num_sms;
                        const size_t grid_size = target_blocks + num_tensors;
                        const size_t shared_mem_bytes =
                            (num_tensors + 1) * sizeof(int64_t) + (num_tensors + 1) * sizeof(int);

                        group_cast_fp8_varying_first_rowwise_aligned_flat_kernel<IS_ACT, ParamOP,
                                                                                 OP, IType, OType>
                            <<<grid_size, flat_block, shared_mem_bytes, stream>>>(
                                reinterpret_cast<const IType *>(input->data.dptr),
                                reinterpret_cast<OType *>(output->data.dptr), scale_ptr, noop_ptr,
                                num_tensors, total_elements, offsets_ptr, target_blocks);
                      } else {
                        group_cast_fp8_varying_first_rowwise_flat_kernel<IS_ACT, ParamOP, OP, IType,
                                                                         OType, false>
                            <<<flat_grid, flat_block, 0, stream>>>(
                                reinterpret_cast<const IType *>(input->data.dptr),
                                reinterpret_cast<OType *>(output->data.dptr), scale_ptr, noop_ptr,
                                num_tensors, total_elements, offsets_ptr);
                      }
                      launched_fast_path = true;
                    }
                  }
                  // Varying-first columnwise intentionally falls through to the
                  // generic group_cast_fp8_kernel below. The dedicated
                  // group_cast_fp8_varying_first_tile_kernel used a grid-stride
                  // row loop that spilled to 118 registers/thread (2 blocks/SM,
                  // ~24% occupancy, ~1.1 TB/s). The generic kernel processes one
                  // tile per block and now carries a branchless interior-tile
                  // fast path, so it is both lighter on registers and HBM-bound
                  // on the interior of every group.
                  if (!launched_fast_path) {
                    const size_t work_blocks_X = DIVUP(last_logical_dim, tile_dim_n);
                    const dim3 grid(work_blocks_X, work_blocks_Y);
                    group_cast_fp8_kernel<IS_ACT, ParamOP, OP, IType, OType, SCALING_TYPE,
                                          SHAPE_REP><<<grid, block, 0, stream>>>(
                        reinterpret_cast<const IType *>(input->data.dptr),
                        use_rowwise_output ? reinterpret_cast<OType *>(output->data.dptr) : nullptr,
                        use_colwise_output ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                           : nullptr,
                        scale_ptr, noop_ptr, num_tensors, first_logical_dim, last_logical_dim,
                        offsets_ptr, first_dims_ptr, last_dims_ptr);
                  }
                  NVTE_CHECK_CUDA(cudaGetLastError());
                });
              }));  // NOLINT(*)
  );                // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
