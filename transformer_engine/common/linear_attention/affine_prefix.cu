/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cassert>

#include "../common.h"
#include "transformer_engine/linear_attention.h"

namespace transformer_engine {
namespace {

// Reuse transition loads across value rows, preserving FP32 accumulation.
constexpr int kRows = 4;

// Each block composes one value-row tile of a fragment's recurrent state for one head.
__global__ void affine_prefix_kernel(const float *initial_states, const float *additive,
                                     const float *transitions, const int *descriptors,
                                     float *output, size_t num_initial_states, size_t num_records,
                                     size_t heads, size_t value_dim, size_t key_dim) {
  // Shared memory bounce buffers to alternate between state and next_state
  extern __shared__ float storage[];
  float *state = storage;
  float *next_state = storage + kRows * key_dim;
  const size_t value_tiles = (value_dim + kRows - 1) / kRows;
  // Each fragment/head pair owns one block per tile of value rows.
  const size_t fragment = blockIdx.x / (heads * value_tiles);
  const size_t head = (blockIdx.x / value_tiles) % heads;
  const size_t first_value_row = (blockIdx.x % value_tiles) * kRows;
  const int *descriptor = descriptors + fragment * 4;
  const int initial_state_index = descriptor[0], first_record = descriptor[1],
            record_count = descriptor[2], stride = descriptor[3];
  // Confirm the descriptor is valid before the block uses it.
  assert((initial_state_index == -1 ||
          (initial_state_index >= 0 &&
           static_cast<size_t>(initial_state_index) < num_initial_states)) &&
         record_count >= 0);
  // An implicit zero initial state makes the first matrix product unnecessary.
  const int first_step = initial_state_index == -1 && record_count > 0 ? 1 : 0;
  for (size_t col = threadIdx.x; col < key_dim; col += blockDim.x) {
    for (int row = 0; row < kRows; ++row) {
      const size_t value = first_value_row + row;
      float initial = 0.f;
      if (value < value_dim) {
        if (first_step) {
          assert(first_record >= 0 && static_cast<size_t>(first_record) < num_records);
          // H[first_record, head, value, col]: summary record, head, value row, key column.
          // Flat offset = first_record*(heads*value_dim*key_dim) + head*(value_dim*key_dim)
          //             + value*key_dim + col.
          initial = additive[((first_record * heads + head) * value_dim + value) * key_dim + col];
        } else if (initial_state_index >= 0) {
          initial =
              initial_states[((initial_state_index * heads + head) * value_dim + value) * key_dim +
                             col];
        }
      }
      state[row * key_dim + col] = initial;
    }
  }
  // Make the initialized state visible to all threads before it is read.
  __syncthreads();
  for (int step = first_step; step < record_count; ++step) {
    const int record = first_record + step * stride;
    assert(record >= 0 && static_cast<size_t>(record) < num_records);
    const float *transition_matrix = transitions + (record * heads + head) * key_dim * key_dim;
    // Each thread accumulates H + state @ T for its key columns in registers.
    for (size_t col = threadIdx.x; col < key_dim; col += blockDim.x) {
      float result[kRows];
      for (int row = 0; row < kRows; ++row) {
        const size_t value = first_value_row + row;
        result[row] = value < value_dim
                          ? additive[((record * heads + head) * value_dim + value) * key_dim + col]
                          : 0.f;
      }
      // Reuse each transition element across the tile's value rows.
      for (size_t inner = 0; inner < key_dim; ++inner) {
        const float transition = transition_matrix[inner * key_dim + col];
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          result[row] = fmaf(state[row * key_dim + inner], transition, result[row]);
        }
      }
      for (int row = 0; row < kRows; ++row) next_state[row * key_dim + col] = result[row];
    }
    // Complete all shared-memory reads and writes before swapping state buffers.
    __syncthreads();
    float *swap = state;
    state = next_state;
    next_state = swap;
  }
  for (size_t col = threadIdx.x; col < key_dim; col += blockDim.x) {
    for (int row = 0; row < kRows; ++row) {
      const size_t value = first_value_row + row;
      if (value < value_dim) {
        output[((fragment * heads + head) * value_dim + value) * key_dim + col] =
            state[row * key_dim + col];
      }
    }
  }
}

}  // namespace
}  // namespace transformer_engine

void nvte_linear_attention_affine_prefix(const NVTETensor initial_states, const NVTETensor additive,
                                         const NVTETensor transitions, const NVTETensor descriptors,
                                         NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_linear_attention_affine_prefix);
  using namespace transformer_engine;
  const auto &initial_data = convertNVTETensorCheck(initial_states)->data;
  const auto &additive_data = convertNVTETensorCheck(additive)->data;
  const auto &transition_data = convertNVTETensorCheck(transitions)->data;
  const auto &descriptor_data = convertNVTETensorCheck(descriptors)->data;
  auto &output_data = convertNVTETensorCheck(output)->data;
  NVTE_CHECK(initial_data.dtype == DType::kFloat32 && additive_data.dtype == DType::kFloat32 &&
                 transition_data.dtype == DType::kFloat32 && output_data.dtype == DType::kFloat32,
             "Affine states and summaries must be FP32.");
  NVTE_CHECK(descriptor_data.dtype == DType::kInt32, "Affine descriptors must be INT32.");
  NVTE_CHECK(initial_data.shape.size() == 4 && additive_data.shape.size() == 4 &&
                 transition_data.shape.size() == 4 && output_data.shape.size() == 4 &&
                 descriptor_data.shape.size() == 2,
             "Invalid affine tensor ranks.");
  const size_t heads = initial_data.shape[1], value_dim = initial_data.shape[2],
               key_dim = initial_data.shape[3];
  NVTE_CHECK(heads > 0 && value_dim > 0 && key_dim > 0, "State dimensions must be positive.");
  NVTE_CHECK((additive_data.shape ==
              std::vector<size_t>{transition_data.shape[0], heads, value_dim, key_dim}) &&
                 (transition_data.shape ==
                  std::vector<size_t>{additive_data.shape[0], heads, key_dim, key_dim}) &&
                 (output_data.shape ==
                  std::vector<size_t>{descriptor_data.shape[0], heads, value_dim, key_dim}) &&
                 descriptor_data.shape[1] == 4,
             "Inconsistent affine tensor shapes.");
  if (output_data.shape[0] == 0) return;
  const size_t value_tiles = output_data.shape[0] * heads * ((value_dim + kRows - 1) / kRows);
  // Extra threads have no key columns to process when K is at most 64.
  const int threads = key_dim <= 64 ? 64 : 128;
  affine_prefix_kernel<<<value_tiles, threads, 2 * kRows * key_dim * sizeof(float), stream>>>(
      static_cast<const float *>(initial_data.dptr), static_cast<const float *>(additive_data.dptr),
      static_cast<const float *>(transition_data.dptr),
      static_cast<const int *>(descriptor_data.dptr), static_cast<float *>(output_data.dptr),
      initial_data.shape[0], additive_data.shape[0], heads, value_dim, key_dim);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
