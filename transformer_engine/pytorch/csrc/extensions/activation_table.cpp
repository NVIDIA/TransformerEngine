/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "../common.h"

namespace transformer_engine::pytorch {

namespace {

struct CachedTable {
  at::Tensor buffer;
  std::unique_ptr<TensorWrapper> tensor;
  bool ready = false;
};

}  // namespace

NVTETensor activation_table(const NVTEActivationTableKind kind, cudaStream_t stream) {
  static std::mutex mutex;
  // Never destroyed: the tables live until the process exits, and destroying tensors after
  // the CUDA context has been torn down is not safe.
  static auto *tables = new std::map<std::pair<int, int>, CachedTable>();
  const int device = c10::cuda::current_device();

  std::lock_guard<std::mutex> lock(mutex);
  CachedTable &table = (*tables)[{device, static_cast<int>(kind)}];
  if (!table.tensor) {
    const size_t bytes = nvte_activation_table_size(kind);
    table.buffer = at::empty({static_cast<int64_t>(bytes)},
                             at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device));
    table.tensor = std::make_unique<TensorWrapper>(table.buffer.data_ptr(),
                                                   std::vector<size_t>{bytes}, DType::kByte);
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
    // A build issued under graph capture only runs when the graph replays, so the next
    // eager call builds again. Building is idempotent.
  }
  return table.tensor->data();
}

}  // namespace transformer_engine::pytorch
