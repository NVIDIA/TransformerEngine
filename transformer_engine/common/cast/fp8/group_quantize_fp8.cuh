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

#include "../../common.h"
#include "../../util/math.h"
#include "../../utils.cuh"
#include "../core/common.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace group_quantize_kernel {

using namespace dispatch::common;

constexpr size_t WARPS_PER_TILE = 4;
constexpr size_t THREADS_PER_TILE = THREADS_PER_WARP * WARPS_PER_TILE;
constexpr size_t LOAD_SIZE_BYTES = 4;
constexpr size_t STORE_SIZE_BYTES = 4;

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
  constexpr size_t nvec_in = LOAD_SIZE_BYTES / sizeof(IType);
  constexpr size_t nvec_out = STORE_SIZE_BYTES / sizeof(OType);
  constexpr size_t tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
  constexpr size_t num_iterations = THREADS_PER_WARP / WARPS_PER_TILE;

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

  OVecT local_output_t[nvec_in][num_iterations];

#pragma unroll
  for (size_t iter = 0; iter < num_iterations; ++iter) {
    const size_t i1 = tidy + iter * WARPS_PER_TILE;
    const size_t j1 = tidx;
#pragma unroll
    for (size_t i2 = 0; i2 < nvec_out; ++i2) {
      const size_t row = tile_row + i1 * nvec_out + i2;
      const size_t col = tile_col + j1 * nvec_in;
      if (row < rows) {
#pragma unroll
        for (size_t j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < cols) {
            float elt = static_cast<float>(input[tensor_base + row * cols + col + j2]);
            if constexpr (IS_ACT) {
              elt = OP(elt, {});
            }
            const OType out = static_cast<OType>(elt * scale);
            if constexpr (ROWWISE_OUTPUT) {
              output_rowwise[tensor_base + row * cols + col + j2] = out;
            }
            if constexpr (COLWISE_OUTPUT) {
              local_output_t[j2][iter].data.elt[i2] = out;
            }
          }
        }
      }
    }
  }

  if constexpr (COLWISE_OUTPUT) {
    __shared__ OVecT shared_output_t[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
    for (size_t j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidy + iter * WARPS_PER_TILE;
        const size_t j1 = tidx;
        shared_output_t[j1][i1] = local_output_t[j2][iter];
      }
      __syncthreads();
#pragma unroll
      for (size_t iter = 0; iter < num_iterations; ++iter) {
        const size_t i1 = tidx;
        const size_t j1 = tidy + iter * WARPS_PER_TILE;
        const size_t row = tile_row + i1 * nvec_out;
        const size_t col = tile_col + j1 * nvec_in + j2;
        if (col < cols) {
#pragma unroll
          for (size_t i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < rows) {
              output_colwise[tensor_base + col * rows + row + i2] =
                  shared_output_t[j1][i1].data.elt[i2];
            }
          }
        }
      }
      __syncthreads();
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

  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  const size_t num_tensors = input->num_tensors;
  if (first_logical_dim == 0 || last_logical_dim == 0) {
    return;
  }
  constexpr size_t row_tile_size = THREADS_PER_WARP * STORE_SIZE_BYTES;
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
          constexpr size_t nvec_in = LOAD_SIZE_BYTES / sizeof(IType);
          constexpr size_t tile_dim_n = THREADS_PER_WARP * nvec_in;
          const size_t work_blocks_X = DIVUP(last_logical_dim, tile_dim_n);
          const dim3 grid(work_blocks_X, work_blocks_Y);
          TRANSFORMER_ENGINE_SCALING_TYPE_SWITCH(
              scaling_type, SCALING_TYPE,
              TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(
                  shape_rep, SHAPE_REP,
                  {
                    if constexpr (SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
                      NVTE_CHECK(offsets_ptr != nullptr && first_dims_ptr != nullptr,
                                 "Varying first-dimension grouped FP8 quantization requires "
                                 "first_dims and tensor_offsets.");
                    }
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
                    NVTE_CHECK_CUDA(cudaGetLastError());
                  })));  // NOLINT(*)
  );                     // NOLINT(*)
}

}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_FP8_CUH_
