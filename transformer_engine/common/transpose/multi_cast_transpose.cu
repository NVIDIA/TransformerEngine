/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transpose.h>

#include <string>
#include <vector>

#include "../common.h"
#include "../util/rtc.h"
#include "../util/string.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace {

// String with RTC kernel implementation
#include "string_code_transpose_rtc_multi_cast_transpose_cu.h"

// Parameters to tune
constexpr size_t load_size = 8;
constexpr size_t store_size = 8;
constexpr size_t warps_per_tile = 4;
constexpr size_t block_size = THREADS_PER_WARP * warps_per_tile;
constexpr size_t max_tensors_per_kernel = 64;  // Args must be <4 KB

}  // namespace

namespace multi_cast_transpose_impl {

// Kernel arguments
// Note: Multi-cast-transpose kernel can handle a variable number of
// tensors.
struct KernelArgs {
  // (input) Data buffers for input tensors
  void* input_list[max_tensors_per_kernel];
  // (output) Data buffers for cast output tensors
  void* output_c_list[max_tensors_per_kernel];
  // (output) Data buffers for transpose output tensors
  void* output_t_list[max_tensors_per_kernel];
  // (input) Scaling factor for output tensors
  void* scale_list[max_tensors_per_kernel];
  // (output) AMAX's of input tensors
  void* amax_list[max_tensors_per_kernel];
  // Input matrix heights
  size_t num_rows_list[max_tensors_per_kernel];
  // Input matrix widths
  size_t row_length_list[max_tensors_per_kernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  size_t block_range[max_tensors_per_kernel + 1];
  // Number of tensors being processed by kernel
  size_t num_tensors;
};

}  // namespace multi_cast_transpose_impl

void multi_cast_transpose(const std::vector<Tensor*> input_list,
                          std::vector<Tensor*> cast_output_list,
                          std::vector<Tensor*> transposed_output_list, cudaStream_t stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(cast_output_list.size() == input_list.size(),
             "Found ", input_list.size(), " input tensors and ",
             cast_output_list.size(), " cast output tensors");
  NVTE_CHECK(transposed_output_list.size() == input_list.size(),
             "Found ", input_list.size(), " input tensors and ",
             transposed_output_list.size(), " transposed output tensors");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType itype = input_list[0]->data.dtype;
  DType otype = cast_output_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& cast_output = *cast_output_list[tensor_id];
    const auto& transposed_output = *transposed_output_list[tensor_id];
    CheckInputTensor(input, "multi_cast_transpose_input_" + std::to_string(tensor_id));
    CheckInputTensor(cast_output, "multi_cast_output_" + std::to_string(tensor_id));
    CheckInputTensor(transposed_output, "multi_transpose_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == itype, "Input tensor types do not match.");
    NVTE_CHECK(cast_output.data.dtype == otype, "C output tensor types do not match.");
    NVTE_CHECK(transposed_output.data.dtype == otype, "T output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2, "Input tensor must have 2 dimensions.");
    NVTE_CHECK(cast_output.data.shape == input.data.shape,
               "C output tensor shape does not match input tensor.");
    NVTE_CHECK(transposed_output.data.shape.size() == 2,
               "T output tensor shape does not match input tensor.");
    NVTE_CHECK(transposed_output.data.shape[0] == input.data.shape[1],
               "T output tensor shape does not match input tensor.");
    NVTE_CHECK(transposed_output.data.shape[1] == input.data.shape[0],
               "T output tensor shape does not match input tensor.");
  }

  // Type names
  std::string itype_name, otype_name;
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      itype, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          otype, OutputType,
          itype_name = TypeInfo<InputType>::name;
          otype_name = TypeInfo<OutputType>::name;
      );  // NOLINT(*)
  );  // NOLINT(*)

  // Labels for NVRTC kernel cache
  const std::string kernel_label_aligned = concat_strings(
      "multi_cast_transpose"
      ",itype=", itype_name, ",otype=", otype_name, ",aligned=", true);
  const std::string kernel_label_unaligned = concat_strings(
      "multi_cast_transpose"
      ",itype=", itype_name, ",otype=", otype_name, ",aligned=", false);

  // Arguments for NVRTC kernels
  multi_cast_transpose_impl::KernelArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;

  // Helper function to compile and launch NVRTC kernels
  auto launch_kernel = [&] (bool aligned) {
    auto& label = aligned ? kernel_label_aligned : kernel_label_unaligned;
    auto& args = aligned ? kernel_args_aligned : kernel_args_unaligned;
    if (args.num_tensors == 0) {
      return;
    }
    auto &rtc_manager = rtc::KernelManager::instance();
    if (!rtc_manager.is_compiled(label)) {
      std::string code = string_code_transpose_rtc_multi_cast_transpose_cu;
      code = regex_replace(code, "__ITYPE__", itype_name);
      code = regex_replace(code, "__OTYPE__", otype_name);
      code = regex_replace(code, "__LOAD_SIZE__", load_size);
      code = regex_replace(code, "__STORE_SIZE__", store_size);
      code = regex_replace(code, "__WARPS_PER_TILE__", warps_per_tile);
      code = regex_replace(code, "__BLOCK_SIZE__", block_size);
      code = regex_replace(code, "__ALIGNED__", aligned);
      code = regex_replace(code, "__MAX_TENSORS_PER_KERNEL__", max_tensors_per_kernel);
      rtc_manager.compile(label, "multi_cast_transpose_kernel", code,
                          "transformer_engine/common/transpose/rtc/multi_cast_transpose.cu");
    }
    const size_t num_blocks = args.block_range[args.num_tensors];
    rtc_manager.launch(label, num_blocks, block_size, 0, stream, args);
  };

  // Helper function to add tensor to NVRTC kernel arguments
  auto add_tensor_to_kernel_args = [&] (size_t tensor_id, bool aligned, size_t num_tiles) {

    // Kernel arguments
    auto& args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Launch kernel if arguments are already full
    if (args.num_tensors == max_tensors_per_kernel) {
      launch_kernel(aligned);
      args.num_tensors = 0;
    }

    // Add tensor to arguments
    const size_t i = args.num_tensors;
    args.input_list[i] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    args.output_c_list[i] = cast_output_list[tensor_id]->data.dptr;
    args.output_t_list[i] = transposed_output_list[tensor_id]->data.dptr;
    args.scale_list[i] = cast_output_list[tensor_id]->scale.dptr;
    args.amax_list[i] = cast_output_list[tensor_id]->amax.dptr;
    args.num_rows_list[i] = input_list[tensor_id]->data.shape[0];
    args.row_length_list[i] = input_list[tensor_id]->data.shape[1];
    args.block_range[i + 1] = args.block_range[i] + num_tiles;
    args.num_tensors++;

  };

  // Helper function to check pointer alignment to 16B
  auto ptr_is_aligned = [](const void *ptr) -> bool {
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const size_t tile_dim_m = THREADS_PER_WARP * store_size / typeToSize(otype);
  const size_t tile_dim_n = THREADS_PER_WARP * load_size / typeToSize(itype);

  // Add tensors to kernel arguments
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {

    // Calculate number of thread blocks needed for tensor
    const size_t num_rows = input_list[tensor_id]->data.shape[0];
    const size_t row_length = input_list[tensor_id]->data.shape[1];
    const size_t num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
    const size_t num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const size_t num_tiles = num_tiles_m * num_tiles_n;

    // Choose whether to use aligned or unaligned kernel
    const bool aligned =
      ((num_tiles_m * tile_dim_m == num_rows)
       && (num_tiles_n * tile_dim_n == row_length)
       && ptr_is_aligned(input_list[tensor_id]->data.dptr)
       && ptr_is_aligned(cast_output_list[tensor_id]->data.dptr)
       && ptr_is_aligned(transposed_output_list[tensor_id]->data.dptr));

    // Add tensor to kernel arguments
    // Note: Launches kernel if arguments are already full
    add_tensor_to_kernel_args(tensor_id, aligned, num_tiles);

  }

  // Launch kernels if needed
  launch_kernel(true);
  launch_kernel(false);

}

}  // namespace transformer_engine

void nvte_multi_cast_transpose(size_t num_tensors, const NVTETensor* input_list,
                               NVTETensor* cast_output_list, NVTETensor* transposed_output_list,
                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_cast_transpose);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_, cast_output_list_, transposed_output_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(reinterpret_cast<Tensor*>(const_cast<NVTETensor&>(input_list[i])));
    cast_output_list_.push_back(reinterpret_cast<Tensor*>(cast_output_list[i]));
    transposed_output_list_.push_back(reinterpret_cast<Tensor*>(transposed_output_list[i]));
  }
  multi_cast_transpose(input_list_, cast_output_list_, transposed_output_list_, stream);
}
