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
  constexpr bool kUseFastCvt = !IS_ACT && (NUM_ELTS % 4 == 0) &&
                               supports_fast_scaled_fp8_cvt_4<IType, OType>::value;
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
  constexpr bool kUseFastCvt = !IS_ACT && (NUM_ELTS % 4 == 0) &&
                               supports_fast_scaled_fp8_cvt_4<IType, OType>::value;
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

template <bool IS_ACT, typename ParamOP, float (*OP)(float, const ParamOP &), typename IType,
          typename OType, ScalingType SCALING_TYPE, ShapeRepresentation SHAPE_REP>
__global__ void __launch_bounds__(THREADS_PER_TILE) group_cast_fp8_kernel(
    const IType *__restrict__ input, OType *__restrict__ output_rowwise,
    OType *__restrict__ output_colwise, const float *__restrict__ scale_ptr,
    const float *__restrict__ noop, const size_t num_tensors, const size_t first_logical_dim,
    const size_t last_logical_dim, const int64_t *__restrict__ offsets_ptr,
    const int64_t *__restrict__ first_dims_ptr) {
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

  const size_t tile_col = blockIdx.x * tile_dim_n;
  if (tile_col >= last_logical_dim) {
    return;
  }

  size_t tensor_id = 0;
  size_t rows = 0;
  size_t tensor_base = 0;
  size_t tile_row = 0;
  const size_t cols = last_logical_dim;
  if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
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
  } else {
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
    __shared__ OVecT shared_output_t[nvec_in][THREADS_PER_WARP]
                                         [THREADS_PER_WARP + TRANSPOSE_SHARED_PAD];

#pragma unroll
    for (size_t iter = 0; iter < num_iterations; ++iter) {
      const size_t i1 = tidy + iter * WARPS_PER_TILE;
      const size_t j1 = tidx;
      const size_t base_row = tile_row + i1 * nvec_out;
      const size_t base_col = tile_col + j1 * nvec_in;
      const size_t fragment_cols = base_col < cols
                                       ? ((cols - base_col) < nvec_in ? (cols - base_col)
                                                                      : nvec_in)
                                       : 0;
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
              if (reinterpret_cast<uint64_t>(output_ptr) % OVecC::BYTES == 0) {
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
__global__ void __launch_bounds__(ROWWISE_FLAT_THREADS) group_cast_fp8_rowwise_flat_kernel(
    const IType *__restrict__ input, OType *__restrict__ output_rowwise,
    const float *__restrict__ scale_ptr, const float *__restrict__ noop,
    const size_t rows_per_tensor, const size_t cols, const size_t vecs_per_tensor) {
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
          typename OType, ScalingType SCALING_TYPE>
__global__ void __launch_bounds__(THREADS_PER_TILE) group_cast_fp8_same_shape_full_tile_kernel(
    const IType *__restrict__ input, OType *__restrict__ output_rowwise,
    OType *__restrict__ output_colwise, const float *__restrict__ scale_ptr,
    const float *__restrict__ noop, const size_t rows_per_tensor, const size_t cols) {
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

  __shared__ OVecT shared_output_t[nvec_in][THREADS_PER_WARP]
                                       [THREADS_PER_WARP + TRANSPOSE_SHARED_PAD];

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
        local_output.store_to(output_ptr);
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
  NVTE_CHECK(input->all_same_last_dim() && output->all_same_last_dim(),
             "Grouped FP8 tensor-scaling quantization only supports a uniform last dimension.");

  const bool use_rowwise_output = output->has_data();
  const bool use_colwise_output = output->has_columnwise_data();
  NVTE_CHECK(use_rowwise_output || use_colwise_output,
             "Either rowwise or columnwise output data need to be allocated.");
  if (use_rowwise_output) {
    NVTE_CHECK(output->scale_inv.has_data(), "Rowwise scale_inv must be allocated.");
  }
  if (use_colwise_output) {
    NVTE_CHECK(output->columnwise_scale_inv.has_data(),
               "Columnwise scale_inv must be allocated.");
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
  } else {
    NVTE_ERROR("Grouped FP8 tensor-scaling quantization only supports same-shape or varying "
               "first-dimension grouped tensors.");
  }

  NVTE_CHECK(input->logical_shape.data[1] == output->logical_shape.data[1],
             "Grouped FP8 tensor-scaling input and output must have the same last dimension.");
  NVTE_CHECK(output->logical_shape.data[0] <= input->logical_shape.data[0],
             "Grouped FP8 tensor-scaling output first dimension must not exceed input first "
             "dimension.");

  // FP8 grouped outputs may carry an active logical shape while their backing buffers remain
  // overallocated. Size the launch from the output shape so metadata-backed groups do not launch
  // over unused tail rows. When host-side active offsets are unavailable, the output shape remains
  // the allocated fallback shape.
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
  } else {
    // first_dims live on device; over-allocate the grid enough to cover one partial tile per group.
    work_blocks_Y += num_tensors;
  }

  const int64_t *const offsets_ptr = reinterpret_cast<const int64_t *>(output->tensor_offsets.dptr);
  const int64_t *const first_dims_ptr = reinterpret_cast<const int64_t *>(output->first_dims.dptr);
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
                TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
                    shape_rep, SHAPE_REP,
                    {
                      if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
                        NVTE_CHECK(offsets_ptr != nullptr && first_dims_ptr != nullptr,
                                   "Varying first-dimension grouped FP8 quantization requires "
                                   "first_dims and tensor_offsets.");
                      }
                      bool launched_fast_path = false;
                      if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS) {
                        const size_t rows_per_tensor = first_logical_dim / num_tensors;
                        if constexpr (SCALING_TYPE == ScalingType::ROWWISE) {
                          constexpr size_t flat_nvec =
                              ROWWISE_FLAT_LOAD_SIZE_BYTES / sizeof(IType);
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
                                    reinterpret_cast<OType *>(output->data.dptr), scale_ptr,
                                    noop_ptr, rows_per_tensor, last_logical_dim, vecs_per_tensor);
                            launched_fast_path = true;
                          }
                        } else if constexpr (colwise_output) {
                          constexpr size_t store_size_bytes = TRANSPOSE_STORE_SIZE_BYTES;
                          constexpr size_t nvec_out = store_size_bytes / sizeof(OType);
                          constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
                          if (rows_per_tensor % tile_dim_m == 0 &&
                              last_logical_dim % tile_dim_n == 0) {
                            const dim3 full_grid(last_logical_dim / tile_dim_n,
                                                 num_tensors * (rows_per_tensor / tile_dim_m));
                            group_cast_fp8_same_shape_full_tile_kernel<
                                IS_ACT, ParamOP, OP, IType, OType, SCALING_TYPE>
                                <<<full_grid, block, 0, stream>>>(
                                    reinterpret_cast<const IType *>(input->data.dptr),
                                    use_rowwise_output
                                        ? reinterpret_cast<OType *>(output->data.dptr)
                                        : nullptr,
                                    reinterpret_cast<OType *>(output->columnwise_data.dptr),
                                    scale_ptr, noop_ptr, rows_per_tensor, last_logical_dim);
                            launched_fast_path = true;
                          }
                        }
                      }
                      if (!launched_fast_path) {
                        const size_t work_blocks_X = DIVUP(last_logical_dim, tile_dim_n);
                        const dim3 grid(work_blocks_X, work_blocks_Y);
                        group_cast_fp8_kernel<IS_ACT, ParamOP, OP, IType, OType, SCALING_TYPE,
                                              SHAPE_REP><<<grid, block, 0, stream>>>(
                            reinterpret_cast<const IType *>(input->data.dptr),
                            use_rowwise_output ? reinterpret_cast<OType *>(output->data.dptr)
                                               : nullptr,
                            use_colwise_output
                                ? reinterpret_cast<OType *>(output->columnwise_data.dptr)
                                : nullptr,
                            scale_ptr, noop_ptr, num_tensors, first_logical_dim, last_logical_dim,
                            offsets_ptr, first_dims_ptr);
                      }
                      NVTE_CHECK_CUDA(cudaGetLastError());
                    });
              }));  // NOLINT(*)
  );                     // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
