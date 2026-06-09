import re

with open("transformer_engine/common/swizzle/swizzle.cu", "r") as f:
    content = f.read()

# 1. Insert kernels before swizzle_grouped_scaling_factors
kernels_code = """
template <int SF_TILE_DIM_M, int SF_TILE_DIM_K>
__global__ void __launch_bounds__(TB_DIM* TB_DIM)
    grouped_swizzle_scaling_variable_shape_kernel(
        const void* input,
        void* output,
        const int64_t* m_array,
        const int64_t* k_array,
        const int* block_offsets,
        const size_t* scale_offsets,
        int* global_counter,
        int num_tensors,
        bool rowwise) {

  __shared__ int linear_block_id;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
      linear_block_id = atomicAdd(global_counter, 1);
  }
  __syncthreads();

  int tensor_id = -1;
  int low = 0;
  int high = num_tensors - 1;
  while (low <= high) {
      int mid = low + (high - low) / 2;
      if (linear_block_id >= block_offsets[mid] && linear_block_id < block_offsets[mid + 1]) {
          tensor_id = mid;
          break;
      } else if (linear_block_id < block_offsets[mid]) {
          high = mid - 1;
      } else {
          low = mid + 1;
      }
  }

  if (tensor_id == -1) return;

  int local_block_id = linear_block_id - block_offsets[tensor_id];

  size_t M = rowwise ? m_array[tensor_id] : k_array[tensor_id];
  size_t K = rowwise ? k_array[tensor_id] : m_array[tensor_id];

  size_t padded_m = round_up_to_multiple(M, 128);
  size_t padded_k = round_up_to_multiple(DIVUP(K, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);

  int num_tiles_m = padded_m / SF_TILE_DIM_M;
  int num_tiles_k = padded_k / SF_TILE_DIM_K;

  int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
  if (vec_load_size == 3) vec_load_size = 1;
  int n_tiles_in_tb = TB_DIM * vec_load_size;

  int grid_dim_x = rowwise ? DIVUP(num_tiles_k, n_tiles_in_tb) : DIVUP(num_tiles_k, TB_DIM);
  int grid_dim_y = rowwise ? num_tiles_m : DIVUP(num_tiles_m, vec_load_size);

  int block_x = local_block_id % grid_dim_x;
  int block_y = local_block_id / grid_dim_x;

  const uint8_t* input_base = reinterpret_cast<const uint8_t*>(input) + scale_offsets[tensor_id];
  uint8_t* output_base = reinterpret_cast<uint8_t*>(output) + scale_offsets[tensor_id];

  int original_M = static_cast<int>(M);
  int original_K = static_cast<int>(DIVUP(K, static_cast<size_t>(MXFP8_BLOCK_SIZE)));

  if (rowwise) {
      if (vec_load_size == 4) {
          swizzle_row_scaling_kernel_impl<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>(
              input_base, output_base, padded_m, padded_k, original_M, original_K,
              block_x, block_y, grid_dim_x, grid_dim_y);
      } else if (vec_load_size == 2) {
          swizzle_row_scaling_kernel_impl<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>(
              input_base, output_base, padded_m, padded_k, original_M, original_K,
              block_x, block_y, grid_dim_x, grid_dim_y);
      } else {
          swizzle_row_scaling_kernel_impl<int, SF_TILE_DIM_M, SF_TILE_DIM_K>(
              input_base, output_base, padded_m, padded_k, original_M, original_K,
              block_x, block_y, grid_dim_x, grid_dim_y);
      }
  } else {
      if (vec_load_size == 4) {
          swizzle_col_scaling_kernel_impl<int4, SF_TILE_DIM_M, SF_TILE_DIM_K>(
              input_base, output_base, padded_m, padded_k, original_M, original_K,
              block_x, block_y, grid_dim_x, grid_dim_y);
      } else if (vec_load_size == 2) {
          swizzle_col_scaling_kernel_impl<int2, SF_TILE_DIM_M, SF_TILE_DIM_K>(
              input_base, output_base, padded_m, padded_k, original_M, original_K,
              block_x, block_y, grid_dim_x, grid_dim_y);
      } else {
          swizzle_col_scaling_kernel_impl<int, SF_TILE_DIM_M, SF_TILE_DIM_K>(
              input_base, output_base, padded_m, padded_k, original_M, original_K,
              block_x, block_y, grid_dim_x, grid_dim_y);
      }
  }
}

__global__ void compute_grouped_swizzle_setup(
    const int64_t* m_array,
    const int64_t* k_array,
    int* block_offsets,
    size_t* scale_offsets,
    int* total_blocks,
    int* global_counter,
    size_t num_tensors,
    bool rowwise,
    size_t scale_elem_size) {

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int current_block_offset = 0;
    size_t current_scale_offset = 0;

    for (size_t i = 0; i < num_tensors; ++i) {
      block_offsets[i] = current_block_offset;
      scale_offsets[i] = current_scale_offset;

      size_t m = rowwise ? m_array[i] : k_array[i];
      size_t k = rowwise ? k_array[i] : m_array[i];

      size_t padded_m = round_up_to_multiple(m, 128);
      size_t padded_k = round_up_to_multiple(DIVUP(k, static_cast<size_t>(MXFP8_BLOCK_SIZE)), 4);

      int num_tiles_m = padded_m / 128;
      int num_tiles_k = padded_k / 4;

      int vec_load_size = (rowwise ? ((num_tiles_k - 1) % 4 + 1) : ((num_tiles_m - 1) % 4 + 1));
      if (vec_load_size == 3) vec_load_size = 1;

      int blocks_m = num_tiles_m;
      int blocks_k = DIVUP(num_tiles_k, TB_DIM * vec_load_size);
      if (!rowwise) {
          blocks_m = DIVUP(num_tiles_m, vec_load_size);
          blocks_k = DIVUP(num_tiles_k, TB_DIM);
      }

      current_block_offset += blocks_m * blocks_k;
      current_scale_offset += padded_m * padded_k * scale_elem_size;
    }

    block_offsets[num_tensors] = current_block_offset;
    scale_offsets[num_tensors] = current_scale_offset;
    *total_blocks = current_block_offset;
    *global_counter = 0;
  }
}

namespace transformer_engine {
"""
content = content.replace(
    "namespace transformer_engine {\n\nvoid swizzle_grouped_scaling_factors",
    kernels_code + "\nvoid swizzle_grouped_scaling_factors",
)

# 2. Modify swizzle_grouped_scaling_factors
old_func = """void swizzle_grouped_scaling_factors(const GroupedTensor* input, GroupedTensor* output,
                                     cudaStream_t stream) {"""
new_func = """void swizzle_grouped_scaling_factors(const GroupedTensor* input, GroupedTensor* output,
                                     void* workspace, cudaStream_t stream) {"""
content = content.replace(old_func, new_func)

# 3. Add variable shape logic
old_logic = """  // Only support uniform shapes for graph-safe grouped swizzle
  NVTE_CHECK(input->all_same_shape(), "Grouped swizzle requires uniform tensor shapes.");
  NVTE_CHECK(input->all_same_last_dim() && input->all_same_first_dim(),
             "Grouped swizzle requires uniform tensor shapes.");"""
new_logic = """  const int64_t* m_array = reinterpret_cast<const int64_t*>(input->first_dims.data_ptr);
  const int64_t* k_array = reinterpret_cast<const int64_t*>(input->last_dims.data_ptr);
  const bool is_variable_shape = (m_array != nullptr && k_array != nullptr);

  if (!is_variable_shape) {
    // Fallback to uniform shape implementation
    NVTE_CHECK(input->all_same_shape(), "Grouped swizzle requires uniform tensor shapes.");
    NVTE_CHECK(input->all_same_last_dim() && input->all_same_first_dim(),
               "Grouped swizzle requires uniform tensor shapes.");"""
content = content.replace(old_logic, new_logic)

# Close the if block and add the else block for variable shape
old_launch_end = """  if (has_rowwise_scale_inv) {
    launch_grouped_swizzle(true);
  }
  if (has_columnwise_scale_inv) {
    launch_grouped_swizzle(false);
  }"""
new_launch_end = """  if (has_rowwise_scale_inv) {
    launch_grouped_swizzle(true);
  }
  if (has_columnwise_scale_inv) {
    launch_grouped_swizzle(false);
  }
  } else {
    // Variable shape implementation using Device-Side Block Scheduler
    size_t num_tensors = input->num_tensors;
    NVTE_CHECK(workspace != nullptr, "Workspace must be provided for variable shape grouped swizzle.");

    int* d_block_offsets = reinterpret_cast<int*>(workspace);
    size_t* d_scale_offsets = reinterpret_cast<size_t*>(d_block_offsets + num_tensors + 2);
    int* d_global_counter = reinterpret_cast<int*>(d_scale_offsets + num_tensors + 1);
    int* d_total_blocks = d_global_counter + 1;

    constexpr int SF_TILE_DIM_M = 128;
    constexpr int SF_TILE_DIM_K = 4;
    const dim3 block_size(TB_DIM, TB_DIM);
    const int max_slm_size = TB_DIM * 4 * SF_TILE_DIM_M * SF_TILE_DIM_K * sizeof(int8_t);

    auto launch_grouped_swizzle_variable = [&](bool rowwise) {
      const size_t scale_elem_size = rowwise ? typeToSize(input->scale_inv.dtype)
                                             : typeToSize(input->columnwise_scale_inv.dtype);

      compute_grouped_swizzle_setup<<<1, 1, 0, stream>>>(
          m_array, k_array, d_block_offsets, d_scale_offsets, d_total_blocks,
          d_global_counter, num_tensors, rowwise, scale_elem_size);

      NVTE_CHECK_CUDA(cudaFuncSetAttribute(
          grouped_swizzle_scaling_variable_shape_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, max_slm_size));

      int persistent_blocks = 108 * 8;
      dim3 num_blocks(persistent_blocks);

      const void* input_ptr = rowwise ? input->scale_inv.dptr : input->columnwise_scale_inv.dptr;
      void* output_ptr = rowwise ? output->scale_inv.dptr : output->columnwise_scale_inv.dptr;

      grouped_swizzle_scaling_variable_shape_kernel<SF_TILE_DIM_M, SF_TILE_DIM_K>
          <<<num_blocks, block_size, max_slm_size, stream>>>(
              input_ptr, output_ptr, m_array, k_array, d_block_offsets,
              d_scale_offsets, d_global_counter, num_tensors, rowwise);

      NVTE_CHECK_CUDA(cudaGetLastError());
    };

    if (has_rowwise_scale_inv) {
      launch_grouped_swizzle_variable(true);
    }
    if (has_columnwise_scale_inv) {
      launch_grouped_swizzle_variable(false);
    }
  }"""
content = content.replace(old_launch_end, new_launch_end)

# 4. Modify nvte_swizzle_grouped_scaling_factors wrapper
old_wrapper = """void nvte_swizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                          cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_grouped_scaling_factors);
  using namespace transformer_engine;
  swizzle_grouped_scaling_factors(convertNVTEGroupedTensorCheck(input),
                                  convertNVTEGroupedTensorCheck(output), stream);
}"""
new_wrapper = """void nvte_swizzle_grouped_scaling_factors(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                          void* workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_swizzle_grouped_scaling_factors);
  using namespace transformer_engine;
  swizzle_grouped_scaling_factors(convertNVTEGroupedTensorCheck(input),
                                  convertNVTEGroupedTensorCheck(output), workspace, stream);
}"""
content = content.replace(old_wrapper, new_wrapper)

with open("transformer_engine/common/swizzle/swizzle.cu", "w") as f:
    f.write(content)
