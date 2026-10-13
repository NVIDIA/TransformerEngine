/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Builds the activation tables from the scalar util/math.h functions. This file must not be
// compiled with --use_fast_math: the tables have to hold exactly what util/math.h computes.

#include <transformer_engine/activation.h>

#include "../common.h"
#include "../util/activation_table.cuh"

namespace transformer_engine {
namespace activation_table {
namespace {

constexpr int kBlockSize = 256;

template <NVTEActivationTableKind kKind>
__global__ void build_forward_kernel(__nv_bfloat16 *table) {
  using T = Table<kKind>;
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= T::kRows) return;
  const uint32_t magnitude = T::kLowSentinel + row;
  __nv_bfloat16 positive, negative;
  if (magnitude == T::kLowSentinel) {
    positive = negative = __ushort_as_bfloat16(0xff80);  // -inf
  } else if (magnitude == T::kHighSentinel) {
    positive = __ushort_as_bfloat16(0x7f80);  // +inf
    negative = __ushort_as_bfloat16(0x8000);  // -0
  } else {
    positive = __float2bfloat16_rn(T::op(__uint_as_float(magnitude << 16)));
    negative = __float2bfloat16_rn(T::op(__uint_as_float((magnitude | 0x8000u) << 16)));
  }
  table[2 * row] = positive;
  table[2 * row + 1] = negative;
}

template <NVTEActivationTableKind kKind>
__global__ void build_derivative_kernel(float *table) {
  using T = Table<kKind>;
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T::kSignStride + T::kEntriesPerSign) return;
  if (i >= T::kEntriesPerSign && i < T::kSignStride) {
    table[i] = 0.0f;  // Unused gap between the two signs.
    return;
  }
  const bool negative = i >= T::kSignStride;
  const uint32_t magnitude = T::kLoBits + (negative ? i - T::kSignStride : i);
  table[i] = T::op(__uint_as_float(((negative ? 0x8000u : 0u) | magnitude) << 16));
}

template <NVTEActivationTableKind kKind>
void build(void *table, cudaStream_t stream) {
  using T = Table<kKind>;
  if constexpr (T::kForward) {
    build_forward_kernel<kKind><<<DIVUP<uint32_t>(T::kRows, kBlockSize), kBlockSize, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(table));
  } else {
    constexpr uint32_t entries = T::kSignStride + T::kEntriesPerSign;
    build_derivative_kernel<kKind><<<DIVUP<uint32_t>(entries, kBlockSize), kBlockSize, 0, stream>>>(
        static_cast<float *>(table));
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace activation_table
}  // namespace transformer_engine

size_t nvte_activation_table_size(NVTEActivationTableKind kind) {
  NVTE_API_CALL(nvte_activation_table_size);
  const size_t bytes = transformer_engine::activation_table::table_bytes(kind);
  NVTE_CHECK(bytes != 0, "Invalid NVTEActivationTableKind (got ", static_cast<int>(kind), ")");
  return bytes;
}

void nvte_build_activation_table(NVTEActivationTableKind kind, NVTETensor table,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_build_activation_table);
  using namespace transformer_engine;
  using namespace transformer_engine::activation_table;
  void *ptr = const_cast<void *>(checked_table_ptr(*convertNVTETensorCheck(table), kind));
  switch (kind) {
    case kNVTEActivationTableGelu:
      build<kNVTEActivationTableGelu>(ptr, stream);
      break;
    case kNVTEActivationTableSilu:
      build<kNVTEActivationTableSilu>(ptr, stream);
      break;
    case kNVTEActivationTableDgelu:
      build<kNVTEActivationTableDgelu>(ptr, stream);
      break;
    case kNVTEActivationTableDsilu:
      build<kNVTEActivationTableDsilu>(ptr, stream);
      break;
    default:
      NVTE_ERROR("Invalid NVTEActivationTableKind (got ", static_cast<int>(kind), ")");
  }
}
