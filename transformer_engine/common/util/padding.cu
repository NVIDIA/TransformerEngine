/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/padding.h>

#include <cfloat>
#include <iostream>
#include <vector>

#include "../common.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace {

// Parameters to tune
constexpr int n_warps_per_tile = 4;
constexpr int threads_per_block = THREADS_PER_WARP * n_warps_per_tile;
constexpr int desired_load_size = 8;
constexpr int desired_store_size = 8;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB

struct MultiPaddingArgs {
  // (input) Data buffers for input tensors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for cast output tensors
  void* output_list[kMaxTensorsPerKernel];
  // Input matrix heights
  int num_rows_list[kMaxTensorsPerKernel];
  // Input matrix heights (padded)
  int padded_num_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths
  int row_length_list[kMaxTensorsPerKernel];
  // tensor
  int block_range[kMaxTensorsPerKernel + 1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

template <int nvec_in, int nvec_out, typename CType, typename IType, typename OType>
__global__ void __launch_bounds__(threads_per_block) multi_padding_kernel(MultiPaddingArgs args) {
  using IVec = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = n_warps_per_tile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / n_warps_per_tile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const IType* input = reinterpret_cast<const IType*>(args.input_list[tensor_id]);
  OType* output = reinterpret_cast<OType*>(args.output_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int padded_num_rows = args.padded_num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  OType local_zero = static_cast<OType>(0.f);
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      OVecC local_output;
      local_input.clear();
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < row_length) {
            local_input.data.elt[j2] = input[row * row_length + col + j2];
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        const CType x = CType(local_input.data.elt[j2]);
        const OType y = OType(x);
        local_output.data.elt[j2] = y;
      }
      if (row < num_rows) {
        for (int j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_output.data.elt[j2];
          }
        }
      } else if (row < padded_num_rows) {
        // padding
        for (int j2 = 0; j2 < nvec_in; ++j2) {
          if (col + j2 < row_length) {
            output[row * row_length + col + j2] = local_zero;
          }
        }
      }
    }
  }
}

}  // namespace

void multi_padding(const std::vector<Tensor*> input_list, std::vector<Tensor*> output_list,
                   const std::vector<int> padded_num_rows_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(output_list.size() == input_list.size(),
             "Number of input and output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType itype = input_list[0]->data.dtype;
  DType otype = output_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& output = *output_list[tensor_id];
    CheckInputTensor(input, "multi_padding_input_" + std::to_string(tensor_id));
    CheckInputTensor(output, "multi_padding_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == itype, "Input tensor types do not match.");
    NVTE_CHECK(output.data.dtype == otype, "Output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions.");
    NVTE_CHECK(output.data.shape[0] == padded_num_rows_list[tensor_id],
               "output tensor shape does not match padded input shape.");
  }

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const int tile_dim_m = THREADS_PER_WARP * desired_store_size / typeToSize(otype);
  const int tile_dim_n = THREADS_PER_WARP * desired_load_size / typeToSize(itype);

  // Add tensors to kernel argument struct
  MultiPaddingArgs kernel_args;
  kernel_args.num_tensors = 0;
  kernel_args.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args.num_tensors == kMaxTensorsPerKernel) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          itype, InputType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
              otype, OutputType, constexpr int nvec_in = desired_load_size / sizeof(InputType);
              constexpr int nvec_out = desired_store_size / sizeof(OutputType);
              const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
              multi_padding_kernel<nvec_in, nvec_out, fp32, InputType, OutputType>
              <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
      );                                                                    // NOLINT(*)
      kernel_args.num_tensors = 0;
    }

    // Calculate number of thread blocks needed for tensor
    const int num_rows = input_list[tensor_id]->data.shape[0];
    const int padded_num_rows = padded_num_rows_list[tensor_id];
    const int row_length = input_list[tensor_id]->data.shape[1];
    const int num_tiles_m = (padded_num_rows + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    kernel_args.output_list[pos] = output_list[tensor_id]->data.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.padded_num_rows_list[pos] = padded_num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args.num_tensors > 0) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
        itype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
            otype, OutputType, constexpr int nvec_in = desired_load_size / sizeof(InputType);
            constexpr int nvec_out = desired_store_size / sizeof(OutputType);
            const int n_blocks = kernel_args.block_range[kernel_args.num_tensors];
            multi_padding_kernel<nvec_in, nvec_out, fp32, InputType, OutputType>
            <<<n_blocks, threads_per_block, 0, stream>>>(kernel_args););  // NOLINT(*)
    );                                                                    // NOLINT(*)
  }
}

}  // namespace transformer_engine

void nvte_multi_padding(size_t num_tensors, const NVTETensor* input_list, NVTETensor* output_list,
                        const int* padded_num_rows_list, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_padding);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, output_list_;
  std::vector<int> padded_num_rows_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(reinterpret_cast<Tensor*>(const_cast<NVTETensor&>(input_list[i])));
    output_list_.push_back(reinterpret_cast<Tensor*>(output_list[i]));
    padded_num_rows_list_.push_back(padded_num_rows_list[i]);
  }
  multi_padding(input_list_, output_list_, padded_num_rows_list_, stream);
}
