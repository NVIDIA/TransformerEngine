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

__global__ void affine_prefix_kernel(const float *seeds, const float *additive,
                                     const float *transitions, const int *descriptors,
                                     float *output, size_t num_seeds, size_t num_records,
                                     size_t heads, size_t value_dim, size_t key_dim) {
  extern __shared__ float storage[];
  float *state = storage;
  float *next = storage + kRows * key_dim;
  const size_t tiles = (value_dim + kRows - 1) / kRows;
  const size_t span = blockIdx.x / (heads * tiles);
  const size_t head = (blockIdx.x / tiles) % heads;
  const size_t first_value = (blockIdx.x % tiles) * kRows;
  const int *desc = descriptors + span * 4;
  const int seed = desc[0], first = desc[1], count = desc[2], stride = desc[3];
  assert((seed == -1 || (seed >= 0 && static_cast<size_t>(seed) < num_seeds)) && count >= 0);
  // An implicit zero seed makes the first matrix product unnecessary.
  const int begin = seed == -1 && count > 0 ? 1 : 0;
  for (size_t col = threadIdx.x; col < key_dim; col += blockDim.x) {
    for (int row = 0; row < kRows; ++row) {
      const size_t value = first_value + row;
      float initial = 0.f;
      if (value < value_dim) {
        if (begin) {
          assert(first >= 0 && static_cast<size_t>(first) < num_records);
          initial = additive[((first * heads + head) * value_dim + value) * key_dim + col];
        } else if (seed >= 0) {
          initial = seeds[((seed * heads + head) * value_dim + value) * key_dim + col];
        }
      }
      state[row * key_dim + col] = initial;
    }
  }
  __syncthreads();
  for (int step = begin; step < count; ++step) {
    const int record = first + step * stride;
    assert(record >= 0 && static_cast<size_t>(record) < num_records);
    const float *matrix = transitions + (record * heads + head) * key_dim * key_dim;
    for (size_t col = threadIdx.x; col < key_dim; col += blockDim.x) {
      float result[kRows];
      for (int row = 0; row < kRows; ++row) {
        const size_t value = first_value + row;
        result[row] = value < value_dim
                          ? additive[((record * heads + head) * value_dim + value) * key_dim + col]
                          : 0.f;
      }
      for (size_t inner = 0; inner < key_dim; ++inner) {
        const float transition = matrix[inner * key_dim + col];
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
          result[row] = fmaf(state[row * key_dim + inner], transition, result[row]);
        }
      }
      for (int row = 0; row < kRows; ++row) next[row * key_dim + col] = result[row];
    }
    __syncthreads();
    float *swap = state;
    state = next;
    next = swap;
  }
  for (size_t col = threadIdx.x; col < key_dim; col += blockDim.x) {
    for (int row = 0; row < kRows; ++row) {
      const size_t value = first_value + row;
      if (value < value_dim) {
        output[((span * heads + head) * value_dim + value) * key_dim + col] =
            state[row * key_dim + col];
      }
    }
  }
}

}  // namespace
}  // namespace transformer_engine

void nvte_linear_attention_affine_prefix(const NVTETensor seeds, const NVTETensor additive,
                                         const NVTETensor transitions, const NVTETensor descriptors,
                                         NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_linear_attention_affine_prefix);
  using namespace transformer_engine;
  const auto &s = convertNVTETensorCheck(seeds)->data;
  const auto &a = convertNVTETensorCheck(additive)->data;
  const auto &t = convertNVTETensorCheck(transitions)->data;
  const auto &d = convertNVTETensorCheck(descriptors)->data;
  auto &o = convertNVTETensorCheck(output)->data;
  NVTE_CHECK(s.dtype == DType::kFloat32 && a.dtype == DType::kFloat32 &&
                 t.dtype == DType::kFloat32 && o.dtype == DType::kFloat32,
             "Affine states and summaries must be FP32.");
  NVTE_CHECK(d.dtype == DType::kInt32, "Affine descriptors must be INT32.");
  NVTE_CHECK(s.shape.size() == 4 && a.shape.size() == 4 && t.shape.size() == 4 &&
                 o.shape.size() == 4 && d.shape.size() == 2,
             "Invalid affine tensor ranks.");
  const size_t heads = s.shape[1], value_dim = s.shape[2], key_dim = s.shape[3];
  NVTE_CHECK(heads > 0 && value_dim > 0 && key_dim > 0, "State dimensions must be positive.");
  NVTE_CHECK((a.shape == std::vector<size_t>{t.shape[0], heads, value_dim, key_dim}) &&
                 (t.shape == std::vector<size_t>{a.shape[0], heads, key_dim, key_dim}) &&
                 (o.shape == std::vector<size_t>{d.shape[0], heads, value_dim, key_dim}) &&
                 d.shape[1] == 4,
             "Inconsistent affine tensor shapes.");
  if (o.shape[0] == 0) return;
  const size_t tiles = o.shape[0] * heads * ((value_dim + kRows - 1) / kRows);
  affine_prefix_kernel<<<tiles, 128, 2 * kRows * key_dim * sizeof(float), stream>>>(
      static_cast<const float *>(s.dptr), static_cast<const float *>(a.dptr),
      static_cast<const float *>(t.dptr), static_cast<const int *>(d.dptr),
      static_cast<float *>(o.dptr), s.shape[0], a.shape[0], heads, value_dim, key_dim);
  NVTE_CHECK_CUDA(cudaGetLastError());
}
