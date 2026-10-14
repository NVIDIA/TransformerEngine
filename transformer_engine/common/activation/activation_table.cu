/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Builds the activation tables from the scalar util/math.h functions. This file must not be
// compiled with --use_fast_math: the tables have to hold exactly what util/math.h computes.

#include <memory>
#include <mutex>

#include "../common.h"
#include "../util/activation_table.cuh"
#include "../util/cuda_runtime.h"

namespace transformer_engine {
namespace activation_table {
namespace {

constexpr int kBlockSize = 256;

template <typename T>
__global__ void build_forward_kernel(__nv_bfloat16 *table) {
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

template <typename T>
__global__ void build_derivative_kernel(float *table) {
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

template <typename T>
void build(void *table, cudaStream_t stream) {
  // Zero the padding up to the 16-byte size, so the table's bytes are deterministic.
  NVTE_CHECK_CUDA(cudaMemsetAsync(table, 0, T::kBytes, stream));
  if constexpr (T::kForward) {
    build_forward_kernel<T><<<DIVUP<uint32_t>(T::kRows, kBlockSize), kBlockSize, 0, stream>>>(
        static_cast<__nv_bfloat16 *>(table));
  } else {
    constexpr uint32_t entries = T::kSignStride + T::kEntriesPerSign;
    build_derivative_kernel<T><<<DIVUP<uint32_t>(entries, kBlockSize), kBlockSize, 0, stream>>>(
        static_cast<float *>(table));
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void build(const Kind kind, void *table, cudaStream_t stream) {
  switch (kind) {
    case kGelu:
      build<Table<kGelu>>(table, stream);
      break;
    case kSilu:
      build<Table<kSilu>>(table, stream);
      break;
    case kDgelu:
      build<Table<kDgelu>>(table, stream);
      break;
    case kDsilu:
      if (use_exact_dsilu_table()) {
        build<ExactDsiluTable>(table, stream);
      } else {
        build<Table<kDsilu>>(table, stream);
      }
      break;
    default:
      NVTE_ERROR("Invalid activation table kind (got ", static_cast<int>(kind), ")");
  }
}

// Sets this thread's stream capture mode, and restores the previous one when destroyed.
class CaptureModeGuard {
 public:
  explicit CaptureModeGuard(cudaStreamCaptureMode mode) : mode_(mode) {
    NVTE_CHECK_CUDA(cudaThreadExchangeStreamCaptureMode(&mode_));
  }
  ~CaptureModeGuard() { cudaThreadExchangeStreamCaptureMode(&mode_); }
  CaptureModeGuard(const CaptureModeGuard &) = delete;
  CaptureModeGuard &operator=(const CaptureModeGuard &) = delete;

 private:
  cudaStreamCaptureMode mode_;
};

// Allocates and fills a table on the current device, synchronously.
void *allocate_and_build(const Kind kind) {
  // The caller's stream, or another thread's, may be capturing a CUDA graph. In relaxed mode
  // this thread may still allocate, and a private stream keeps the build out of the capture.
  // The table is complete when this returns, so captured kernels can read it on replay.
  CaptureModeGuard relaxed(cudaStreamCaptureModeRelaxed);
  void *table = nullptr;
  NVTE_CHECK_CUDA(cudaMalloc(&table, built_table_bytes(kind)));
  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  build(kind, table, stream);
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
  return table;
}

// One set of tables per device. Tables are never freed: freeing them from a static destructor
// could run after the CUDA runtime has shut down.
struct DeviceTables {
  std::once_flag built[kNumKinds];
  void *table[kNumKinds] = {};
};

}  // namespace

TableRef device_table(const Kind kind) {
  NVTE_CHECK(kind >= 0 && kind < kNumKinds, "Invalid activation table kind (got ",
             static_cast<int>(kind), ")");
  static const int num_devices = cuda::num_devices();
  static const std::unique_ptr<DeviceTables[]> tables(new DeviceTables[num_devices]);
  const int device = cuda::current_device();
  NVTE_CHECK(device >= 0 && device < num_devices, "Invalid device (got ", device, ")");
  DeviceTables &mine = tables[device];
  std::call_once(mine.built[kind], [&] { mine.table[kind] = allocate_and_build(kind); });
  const bool exact = kind == kDsilu && use_exact_dsilu_table();
  return {mine.table[kind], exact ? kExactDsiluTable : kDefaultTable};
}

}  // namespace activation_table
}  // namespace transformer_engine
