/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "../extensions.h"

namespace transformer_engine {
namespace jax {

namespace {

struct CachedTable {
  void *ptr = nullptr;
  std::unique_ptr<TensorWrapper> tensor;
  bool ready = false;
};

NVTETensor activation_table(const NVTEActivationTableKind kind, cudaStream_t stream) {
  static std::mutex mutex;
  // Never destroyed: the tables live until the process exits.
  static auto *tables = new std::map<std::pair<int, int>, CachedTable>();
  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));

  std::lock_guard<std::mutex> lock(mutex);
  CachedTable &table = (*tables)[{device, static_cast<int>(kind)}];
  if (!table.tensor) {
    // XLA gives custom calls no persistent device allocation, so the table is allocated
    // directly. All tables together take tens of KB per device.
    const size_t bytes = nvte_activation_table_size(kind);
    NVTE_CHECK_CUDA(cudaMalloc(&table.ptr, bytes));
    table.tensor =
        std::make_unique<TensorWrapper>(table.ptr, std::vector<size_t>{bytes}, DType::kByte);
  }
  if (!table.ready) {
    nvte_build_activation_table(kind, table.tensor->data(), stream);
    cudaStreamCaptureStatus capture;
    NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
    if (capture == cudaStreamCaptureStatusNone) {
      // Later calls may come on other streams, so the build has to be complete.
      NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
      table.ready = true;
    }
    // A build issued under graph capture only runs when the graph replays, so the next call
    // outside capture builds again. Building is idempotent.
  }
  return table.tensor->data();
}

}  // namespace

NVTETensor activation_table_for(const NVTE_Activation_Type act_type, const bool derivative,
                                const DType input_dtype, const JAXX_Scaling_Mode scaling_mode,
                                cudaStream_t stream) {
  if (input_dtype != DType::kBFloat16 || scaling_mode != JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
    return nullptr;
  }
  switch (act_type) {
    case NVTE_Activation_Type::GELU:
      return activation_table(derivative ? kNVTEActivationTableDgelu : kNVTEActivationTableGelu,
                              stream);
    case NVTE_Activation_Type::SILU:
      return activation_table(derivative ? kNVTEActivationTableDsilu : kNVTEActivationTableSilu,
                              stream);
    default:
      return nullptr;
  }
}

}  // namespace jax
}  // namespace transformer_engine
