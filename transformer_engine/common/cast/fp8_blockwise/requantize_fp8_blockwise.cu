/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file requantize_fp8_blockwise.cu
 *  \brief Grouped block128 FP8 requantization: rowwise wire tensor -> GEMM-ready.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include "../../common.h"
#include "../../recipe/recipe_common.cuh"
#include "../../util/logging.h"
#include "../../util/ptx.cuh"
#include "../../util/cuda_runtime.h"
#include "../../util/ptx_arch_spec.cuh"
#include "../../utils.cuh"
#include "../core/grouped_layout.cuh"

namespace transformer_engine {
namespace requantize {
namespace {
// Block scaling FP8 grouped requantization shares the common grouped requant entry point.
namespace ptx = transformer_engine::ptx;

constexpr int kTileDim = 128; // Fixed to 128 beacause we are using 1x128 quantization
constexpr int kThreadsPerBlock = 256; // Thread block size, 8 warps in total

template <ShapeRepresentation ShapeRep>
struct RequantizeTraits {
  static_assert(ShapeRep == ShapeRepresentation::SAME_BOTH_DIMS ||
                ShapeRep == ShapeRepresentation::VARYING_FIRST_DIM);
  static constexpr ShapeRepresentation SHAPE_REPRESENTATION = ShapeRep;
};

/*! \brief Validate per-tile group offsets once per kernel launch.
 *
 *  Runs in a single CTA (blockIdx == 0,0 / threadIdx == 0); other CTAs do a
 *  cheaper per-tile bounds check below. Traps on violation so release builds
 *  surface the failure rather than silently dropping tiles.
 */
__device__ void validate_group_offsets(const int64_t* tensor_offsets, size_t num_groups,
                                       size_t last_logical_dim, size_t first_logical_dim) {
  const int64_t last = tensor_offsets[num_groups];
  const bool power_of_two_last_dim = (last_logical_dim & (last_logical_dim - 1)) == 0;
  const size_t tile_elements = last_logical_dim * kTileDim;
  bool valid = tensor_offsets[0] == 0 && last >= 0 &&
               static_cast<uint64_t>(last) <= first_logical_dim * last_logical_dim;
  int64_t begin = 0;
  for (size_t group = 0; group < num_groups && valid; ++group) {
    const int64_t end = tensor_offsets[group + 1];
    valid = end >= begin && end <= last &&
            (power_of_two_last_dim ? (static_cast<uint64_t>(end) & (last_logical_dim - 1)) == 0
                                 : static_cast<uint64_t>(end) % last_logical_dim == 0) &&
            (begin == last ||
             (power_of_two_last_dim ? (static_cast<uint64_t>(begin) & (tile_elements - 1)) == 0
                                  : static_cast<uint64_t>(begin) % tile_elements == 0));
    begin = end;
  }
  if (!valid) {
    NVTE_DEVICE_ERROR("Block scaling grouped requant invalid offsets: require zero origin, "
                      "monotonic row boundaries within capacity, and 128-aligned group starts.");
    __trap();
  }
}

/*! \brief Requant one 128x128 tile from rowwise to columnwise block128 FP8.
 *
 *  Grid: (ceil(last_logical_dim/128), ceil(first_logical_dim/128)). Block: 256 threads.
 *  4 threads per column x 32 rows each, two column passes over the tile.
 *
 *  Input supports E4M3/E5M2; columnwise output uses E4M3.
 */
template <typename Traits, typename IType, typename OType, bool kReturnDequantized, bool kReturnColumnwise,
          bool kReturnRowwiseScale>
__global__ void __launch_bounds__(kThreadsPerBlock) block128_group_requantize_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const IType* __restrict__ rowwise_data, const float* __restrict__ rowwise_scale,
    float* __restrict__ gemm_rowwise_scale,
    OType* __restrict__ colwise_data, float* __restrict__ colwise_scale,
    bf16* __restrict__ dequantized, const size_t first_logical_dim, const size_t last_logical_dim,
    const size_t row_scale_cols, const bool compact_scale_layout,
    const size_t col_scale_rows,
    const int64_t* tensor_offsets, const int64_t* first_dims_ptr, const size_t num_groups,
    const float epsilon, const bool power_of_two) {
  const bool leading_thread = threadIdx.x == 0;
  constexpr ShapeRepresentation shape_rep = Traits::SHAPE_REPRESENTATION;
  const size_t tile_x = blockIdx.x;  // hidden block index
  const size_t tile_y = blockIdx.y;  // row block index
  const size_t global_col_base = tile_x * kTileDim;
  size_t tensor_base = 0;
  size_t group_rows = first_logical_dim;
  size_t local_row_base = 0;
  if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::SAME_BOTH_DIMS) {
    group_rows = first_logical_dim / num_groups;
    tensor_base = blockIdx.z * group_rows * last_logical_dim;
    local_row_base = tile_y * kTileDim;
  } else if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::VARYING_FIRST_DIM) {
    const size_t global_element_offset = tile_y * kTileDim * last_logical_dim;
    __shared__ size_t varying_first_metadata[3];
    if (leading_thread && blockIdx.x == 0 && blockIdx.y == 0) {
      validate_group_offsets(tensor_offsets, num_groups, last_logical_dim, first_logical_dim);
    }
    if (leading_thread) {
      const size_t active_elements = static_cast<size_t>(tensor_offsets[num_groups]);
      varying_first_metadata[0] = active_elements;
      if (global_element_offset < active_elements) {
        const size_t mapped_tensor = dispatch::common::find_tensor_from_offsets(
            tensor_offsets, num_groups, global_element_offset);
        varying_first_metadata[1] = dispatch::common::get_tensor_rows_num<shape_rep>(
            mapped_tensor, first_logical_dim, first_dims_ptr, num_groups);
        // The helper's assert is disabled in release builds.
        if (varying_first_metadata[1] % kTileDim != 0) __trap();
        varying_first_metadata[2] = static_cast<size_t>(tensor_offsets[mapped_tensor]);
      }
    }
    __syncthreads();
    if (global_element_offset >= varying_first_metadata[0]) return;
    group_rows = varying_first_metadata[1];
    tensor_base = varying_first_metadata[2];
    local_row_base = tile_y * kTileDim - tensor_base / last_logical_dim;
  }
  const size_t group_start = tensor_base / last_logical_dim;
  const size_t local_tile_y = local_row_base / kTileDim;
  const size_t global_row_base = group_start + local_row_base;
  const size_t group_end = group_start + group_rows;
  const size_t live_end = first_logical_dim < group_end ? first_logical_dim : group_end;

  // Load each row's scale once, shared by all 128 columns and the layout conversion.
  __shared__ float row_scales[kTileDim];
  if (threadIdx.x < kTileDim) {
    const size_t row = global_row_base + threadIdx.x;
    float value = 0.f;
    if (row < live_end) {
      const size_t index = compact_scale_layout
                               ? row * row_scale_cols + tile_x
                               : group_start * (last_logical_dim / kTileDim) +
                                     tile_x * group_rows + (row - group_start);
      value = rowwise_scale[index];
      if constexpr (kReturnRowwiseScale) {
        const size_t destination = group_start * (last_logical_dim / kTileDim) +
                                   tile_x * group_rows + (row - group_start);
        gemm_rowwise_scale[destination] = value;
      }
    }
    row_scales[threadIdx.x] = value;
  }

  // Scale-only specialization does not load or reconstruct activation data.
  if constexpr (!kReturnColumnwise && !kReturnDequantized) return;

  // Dynamic smem: rowwise FP8 tile [128][128], 128B aligned for TMA.
  // smem_T stages the transposed quantized tile: [128][128+4] for bank
  // conflict-free column reads followed by coalesced row writes.
  extern __shared__ char smem_raw[];
  char* smem_base = align_up(smem_raw, TMA_SHMEM_ALIGNMENT);
  IType(*smem)[kTileDim] = reinterpret_cast<IType(*)[kTileDim]>(smem_base);

  __shared__ uint64_t tma_mbar;
  const int tid = threadIdx.x;

  if (leading_thread) {
    ptx::mbarrier_init(&tma_mbar, 1);
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if (leading_thread) {
    constexpr uint32_t tx_bytes = kTileDim * kTileDim * (TypeInfo<IType>::size / 8);
    ptx::mbarrier_arrive_expect_tx(&tma_mbar, tx_bytes);
    ptx::cp_async_bulk_tensor_2d_global_to_shared_cta(
        reinterpret_cast<uint64_t*>(smem_base),
        reinterpret_cast<const uint64_t*>(&tensor_map_input),
        static_cast<uint32_t>(global_col_base), static_cast<uint32_t>(global_row_base), &tma_mbar);
  }
  ptx::mbarrier_wait_parity(&tma_mbar, 0);
  __syncthreads();
  if (leading_thread) ptx::mbarrier_invalid(&tma_mbar);

  // Static smem for the transposed output tile (bank-conflict-free pad).
  constexpr int kSMemTRowStride = kTileDim + 4;
  __shared__ OType smem_T[kTileDim][kSMemTRowStride];

  // Column passes bound the per-thread FP32 staging footprint.
  constexpr int kThreadsPerCol = 4;
  static_assert(kThreadsPerBlock % kThreadsPerCol == 0);
  constexpr int kColsPerPass = kThreadsPerBlock / kThreadsPerCol;
  static_assert(kTileDim % kColsPerPass == 0);
  constexpr int kColumnPasses = kTileDim / kColsPerPass;
  constexpr int kRowsPerThread = kTileDim / kThreadsPerCol;  // 32
  const int col_in_pass = tid / kThreadsPerCol;
  const int thr_row = tid % kThreadsPerCol;                  // 0..3
  const int row_start = thr_row;

#pragma unroll
  for (int pass = 0; pass < kColumnPasses; ++pass) {
    const int thr_col = pass * kColsPerPass + col_in_pass;
    const size_t col = global_col_base + thr_col;
    const bool col_valid = (col < last_logical_dim);

    // Dequantize this column's 32 rows and track the column amax.
    float reg_data[kRowsPerThread];
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      const int row_local = row_start + r * kThreadsPerCol;
      const size_t row = global_row_base + row_local;
      if (row < live_end && col_valid) {
        const float row_scale = row_scales[row_local];
        // Match direct grouped quantization of the reconstructed BF16 input.
        const float v = static_cast<float>(
            static_cast<bf16>(static_cast<float>(smem[row_local][thr_col]) * row_scale));
        reg_data[r] = v;
        if constexpr (kReturnColumnwise) {
          amax = fmaxf(amax, fabsf(v));
        }
        if constexpr (kReturnDequantized) {
          dequantized[tensor_base + (local_row_base + row_local) * last_logical_dim + col] =
              static_cast<bf16>(v);
        }
      } else {
        reg_data[r] = 0.f;
      }
    }
    if constexpr (kReturnColumnwise) {
      amax = subwarp_reduce_max_broadcast<kThreadsPerCol>(amax);

      // The direct grouped quantizer uses the reconstructed input type to select
      // overflow handling, not the FP8 carrier type.
      const float scale = compute_scale_from_types<float, OType>(amax, epsilon, power_of_two);
      if (thr_row == 0 && col_valid) {
        const size_t scale_stride = DIVUP_TO_MULTIPLE(last_logical_dim, 4);
        const size_t group_scale_base = (group_start / kTileDim) * scale_stride;
        colwise_scale[group_scale_base + local_tile_y * scale_stride + col] = __frcp_rn(scale);
      }

      // Requantize and stage into the transposed smem tile.
      if (col_valid) {
#pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
          smem_T[thr_col][row_start + r * kThreadsPerCol] = static_cast<OType>(reg_data[r] * scale);
        }
      }
    }
  }
  if constexpr (kReturnColumnwise) {
    __syncthreads();

    // Drain smem_T to global memory: vectorized coalesced writes.
    // Each thread writes one 16-element chunk of one output column.
    constexpr int kStorePerChunk = 16;
    constexpr int kRowChunksPerCol = kTileDim / kStorePerChunk;  // 8
    constexpr int kColsPerIter = kThreadsPerBlock / kRowChunksPerCol;    // 32
    constexpr int kColIters = kTileDim / kColsPerIter;           // 4
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int col_in_warp = lane / kRowChunksPerCol;  // 0..7
    const int row_chunk = lane % kRowChunksPerCol;    // 0..7
    const int out_row_off = row_chunk * kStorePerChunk;

#pragma unroll
    for (int it = 0; it < kColIters; ++it) {
      const int out_col_local = it * kColsPerIter + warp_id * 4 + col_in_warp;
      const size_t out_col = global_col_base + out_col_local;
      if (out_col < last_logical_dim) {
        const size_t out_row = global_row_base + out_row_off;
        if (out_row >= live_end) continue;
        OType* out_ptr = colwise_data + tensor_base + out_col * group_rows +
                          local_row_base + out_row_off;
        const size_t remaining = live_end - out_row;
        const size_t count = remaining < kStorePerChunk ? remaining : kStorePerChunk;
        Vec<OType, kStorePerChunk> v;
#pragma unroll
        for (int e = 0; e < kStorePerChunk; ++e) {
          v.data.elt[e] = smem_T[out_col_local][out_row_off + e];
        }
        v.store_to_elts(out_ptr, 0, count);
      }
    }
  }
}

template <typename Traits, typename IType, bool kReturnDequantized, bool kReturnColumnwise,
          bool kReturnRowwiseScale>
void launch_block128_group_requantize(const GroupedTensor* input, GroupedTensor* output,
                                     Tensor* dequantized, size_t first_logical_dim,
                                     size_t last_logical_dim, size_t row_scale_cols,
                                     bool compact_scale_layout, size_t col_scale_rows,
                                     float epsilon, bool power_of_two, cudaStream_t stream) {
  using OType = fp8e4m3;
  if (first_logical_dim == 0 || last_logical_dim == 0) return;

  NVTE_CHECK(last_logical_dim % kTileDim == 0, "EP block128 requant requires hidden divisible by 128");
  if constexpr (kReturnColumnwise) {
    NVTE_CHECK(output->columnwise_data.dptr != nullptr && output->columnwise_scale_inv.dptr != nullptr,
               "Columnwise requantization requires both data and scale output storage.");
  }
  if constexpr (kReturnRowwiseScale) {
    NVTE_CHECK(output->scale_inv.dptr != nullptr,
               "Rowwise scale conversion requires output storage.");
  }
  const size_t smem_bytes = kReturnColumnwise || kReturnDequantized
                               ? kTileDim * kTileDim * (TypeInfo<IType>::size / 8) +
                                     TMA_SHMEM_ALIGNMENT - 1
                               : 0;

  CUtensorMap tensor_map_input{};
  create_2D_tensor_map(tensor_map_input, input->data, first_logical_dim, last_logical_dim, kTileDim, kTileDim,
                       last_logical_dim, 0, TypeInfo<IType>::size);

  const unsigned int grid_x = static_cast<unsigned int>(last_logical_dim / kTileDim);
  dim3 grid(grid_x, (first_logical_dim + kTileDim - 1) / kTileDim);
  if constexpr (Traits::SHAPE_REPRESENTATION == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t group_rows = first_logical_dim / input->num_tensors;
    grid.y = (group_rows + kTileDim - 1) / kTileDim;
    grid.z = input->num_tensors;
  }

  auto* kernel = block128_group_requantize_kernel<Traits, IType, OType, kReturnDequantized,
                                                 kReturnColumnwise, kReturnRowwiseScale>;
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(smem_bytes)));
  kernel<<<grid, kThreadsPerBlock, smem_bytes, stream>>>(
      tensor_map_input, reinterpret_cast<const IType*>(input->data.dptr),
      reinterpret_cast<const float*>(input->scale_inv.dptr),
      kReturnRowwiseScale ? reinterpret_cast<float*>(output->scale_inv.dptr) : nullptr,
      kReturnColumnwise ? reinterpret_cast<OType*>(output->columnwise_data.dptr) : nullptr,
      kReturnColumnwise ? reinterpret_cast<float*>(output->columnwise_scale_inv.dptr) : nullptr,
      kReturnDequantized ? reinterpret_cast<bf16*>(dequantized->data.dptr) : nullptr,
      first_logical_dim, last_logical_dim, row_scale_cols, compact_scale_layout, col_scale_rows,
      reinterpret_cast<const int64_t*>(input->tensor_offsets.dptr),
      reinterpret_cast<const int64_t*>(input->first_dims.dptr),
      input->num_tensors, epsilon, power_of_two);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace

void group_requantize_block_scaling(const GroupedTensor& input, GroupedTensor* output,
                                   Tensor* dequantized, const QuantizationConfig* quant_config,
                                   cudaStream_t stream) {
  NVTE_CHECK(input.logical_shape.ndim == 2 && output->logical_shape.ndim == 2,
             "Block scaling requant requires 2D grouped shapes.");
  const size_t first_logical_dim = input.logical_shape.data[0];
  const size_t last_logical_dim = input.logical_shape.data[1];
  NVTE_CHECK(output->logical_shape.data[0] == first_logical_dim &&
                 output->logical_shape.data[1] == last_logical_dim &&
                 input.num_tensors == output->num_tensors,
             "Block scaling requant input and output grouped shapes must match.");
  const ShapeRepresentation shape_rep = input.all_same_shape()
      ? ShapeRepresentation::SAME_BOTH_DIMS
      : input.all_same_last_dim() ? ShapeRepresentation::VARYING_FIRST_DIM
      : input.all_same_first_dim() ? ShapeRepresentation::VARYING_LAST_DIM
                                  : ShapeRepresentation::VARYING_BOTH_DIMS;
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 shape_rep == ShapeRepresentation::VARYING_FIRST_DIM,
             "Block scaling requant supports only SAME_BOTH_DIMS and VARYING_FIRST_DIM.");
  NVTE_CHECK(input.all_same_first_dim() == output->all_same_first_dim() &&
                 input.all_same_last_dim() == output->all_same_last_dim(),
             "Input and output grouped shape representations must match.");
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM ||
      input.tensor_offsets.dptr != nullptr || input.tensor_offsets.numel() != 0) {
    NVTE_CHECK(input.tensor_offsets.shape == std::vector<size_t>{input.num_tensors + 1},
               "Block scaling requant requires num_tensors + 1 offsets.");
    NVTE_CHECK(input.tensor_offsets.dtype == DType::kInt64,
               "Block scaling grouped requant requires int64 tensor offsets.");
  }
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS && input.num_tensors != 0) {
    NVTE_CHECK(first_logical_dim % input.num_tensors == 0,
               "SAME_BOTH_DIMS requires an integral row count per tensor.");
    NVTE_CHECK((first_logical_dim / input.num_tensors) % kTileDim == 0,
               "SAME_BOTH_DIMS requires each tensor's row count divisible by 128.");
  }
  if (shape_rep == ShapeRepresentation::VARYING_FIRST_DIM) {
    NVTE_CHECK(input.first_dims.dtype == DType::kInt64 &&
                   input.first_dims.shape == std::vector<size_t>{input.num_tensors},
               "VARYING_FIRST_DIM requires num_tensors int64 first dimensions.");
  }
  NVTE_CHECK(input.data.numel() == first_logical_dim * last_logical_dim &&
                 output->data.dptr == input.data.dptr,
             "Block scaling requant output rowwise data must alias the input.");
  NVTE_CHECK(input.scaling_mode == NVTE_BLOCK_SCALING_1D,
             "Block scaling grouped requant requires 1D block scaling input.");
  NVTE_CHECK((input.data.dtype == DType::kFloat8E4M3 || input.data.dtype == DType::kFloat8E5M2) &&
                 input.scale_inv.dtype == DType::kFloat32,
             "Block scaling grouped requant requires 2D FP8 input and FP32 scales.");
  NVTE_CHECK(output->scaling_mode == NVTE_BLOCK_SCALING_1D,
             "Block scaling grouped requant requires 1D block scaling output.");
  NVTE_CHECK((output->columnwise_data.dptr == nullptr) ==
                 (output->columnwise_scale_inv.dptr == nullptr),
             "Block scaling columnwise data and scales must be supplied together.");
  if (dequantized != nullptr) {
    NVTE_CHECK(dequantized->data.dtype == DType::kBFloat16 &&
                   dequantized->data.shape ==
                       std::vector<size_t>({first_logical_dim, last_logical_dim}),
               "Block scaling dequantized output must be BF16 with the input shape.");
  }

  NVTE_CHECK(last_logical_dim > 0 && last_logical_dim % kTileDim == 0,
             "Block scaling grouped requant requires hidden divisible by 128.");
  const size_t blocks = last_logical_dim / kTileDim;
  if (input.with_gemm_swizzled_scales) {
    NVTE_CHECK(input.scale_inv.numel() == first_logical_dim * blocks,
               "Block scaling GEMM rowwise scale capacity must match logical shape.");
  } else {
    NVTE_CHECK(first_logical_dim == 0 || input.scale_inv.numel() % first_logical_dim == 0,
               "Block scaling compact scales must contain whole rows.");
  }
  NVTE_CHECK(output->scale_inv.numel() == first_logical_dim * blocks,
             "Block scaling output rowwise scale capacity must match logical shape.");
  NVTE_CHECK(input.with_gemm_swizzled_scales ==
                 (input.scale_inv.dptr == output->scale_inv.dptr) || first_logical_dim == 0,
             "Block scaling compact-to-GEMM scale conversion requires separate storage.");
  const bool compact_scale_layout = !input.with_gemm_swizzled_scales;
  const size_t row_scale_cols = compact_scale_layout && first_logical_dim != 0
                                   ? input.scale_inv.numel() / first_logical_dim : blocks;
  NVTE_CHECK(row_scale_cols >= blocks,
             "Block scaling compact scales must have at least one scale per column block.");
  NVTE_CHECK(output->scale_inv.dtype == DType::kFloat32,
             "Block scaling output rowwise scales must be FP32.");
  if (output->columnwise_data.dptr != nullptr) {
    const size_t scale_rows = (first_logical_dim + kTileDim - 1) / kTileDim;
    NVTE_CHECK(output->columnwise_data.dtype == DType::kFloat8E4M3 &&
                   output->columnwise_data.numel() == first_logical_dim * last_logical_dim &&
                   output->columnwise_scale_inv.dtype == DType::kFloat32 &&
                   output->columnwise_scale_inv.numel() >=
                       scale_rows * DIVUP_TO_MULTIPLE(last_logical_dim, 4),
               "Block scaling columnwise output must be E4M3 [hidden, rows] with sufficient "
               "FP32 scale storage for [ceil(rows/128), roundup(hidden, 4)].");
  }

  // Empty work needs valid metadata, but does not dereference backing storage.
  if (first_logical_dim == 0 || input.num_tensors == 0) {
    output->with_gemm_swizzled_scales = true;
    return;
  }

  NVTE_CHECK(input.data.dptr != nullptr && input.scale_inv.dptr != nullptr &&
                 output->scale_inv.dptr != nullptr,
             "Block scaling grouped requant requires rowwise data and scales.");
  NVTE_CHECK(shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                 input.tensor_offsets.dptr != nullptr,
             "Block scaling grouped requant requires tensor offsets storage.");
  NVTE_CHECK(dequantized == nullptr || dequantized->data.dptr != nullptr,
             "Block scaling dequantized output requires storage.");

  output->with_gemm_swizzled_scales = true;

  const size_t col_scale_rows = (first_logical_dim + kTileDim - 1) / kTileDim;
  const float epsilon = quant_config == nullptr ? 0.0f : quant_config->amax_epsilon;
  const bool power_of_two = quant_config == nullptr ? true : quant_config->force_pow_2_scales;

  TRANSFORMER_ENGINE_GROUP_TENSOR_SHAPE_REPRESENTATION_SWITCH(shape_rep, SHAPE_REP, {
    if constexpr (SHAPE_REP == ShapeRepresentation::SAME_BOTH_DIMS ||
                  SHAPE_REP == ShapeRepresentation::VARYING_FIRST_DIM) {
      using ActiveTraits = RequantizeTraits<SHAPE_REP>;
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          input.data.dtype, IType,
          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              dequantized != nullptr, RETURN_DEQUANTIZED,
              TRANSFORMER_ENGINE_SWITCH_CONDITION(
                  output->columnwise_data.dptr != nullptr, RETURN_COLUMNWISE,
                  TRANSFORMER_ENGINE_SWITCH_CONDITION(
                      compact_scale_layout, RETURN_ROWWISE_SCALE,
                      launch_block128_group_requantize<ActiveTraits, IType, RETURN_DEQUANTIZED,
                                                      RETURN_COLUMNWISE, RETURN_ROWWISE_SCALE>(
                          &input, output, dequantized, first_logical_dim, last_logical_dim,
                          row_scale_cols, compact_scale_layout, col_scale_rows,
                          epsilon, power_of_two, stream);););););  // NOLINT(*)
    } else {
      NVTE_ERROR("Block scaling requant supports only SAME_BOTH_DIMS and VARYING_FIRST_DIM.");
    }
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace requantize
}  // namespace transformer_engine
