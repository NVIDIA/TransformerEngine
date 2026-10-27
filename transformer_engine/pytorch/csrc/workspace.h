/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_WORKSPACE_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_WORKSPACE_H_

#include <c10/cuda/CUDAGuard.h>
#include <transformer_engine/cast.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include "common.h"

namespace transformer_engine::pytorch {

// Framework owner for callers that keep a workspace across operations. Each
// instance belongs to one device/stream. Retired tensors remain alive for graphs
// that captured earlier bindings; keep the owner alive for those graphs too.
class WorkspaceOwner {
 public:
  void quantize_grouped(NVTEGroupedTensor input, NVTEGroupedTensor output,
                        NVTEQuantizationConfig config, int device, cudaStream_t stream) {
    use(device, stream, [&](NVTETensor workspace) {
      nvte_group_quantize_v2(input, output, workspace, config, stream);
    });
  }

  template <typename Operation>
  void use(int device, cudaStream_t stream, Operation &&operation) {
    // Called with the GIL released. Release this lock before reacquiring the GIL.
    std::lock_guard<std::mutex> lock(mutex_);
    const c10::cuda::CUDAGuard device_guard(device);
    NVTE_CHECK(!has_context_ || (device_ == device && stream_ == stream),
               "A PyTorch Workspace must be used on one CUDA device and stream.");
    TensorWrapper requirement;
    operation(requirement.data());
    const auto shape = requirement.shape();
    NVTE_CHECK(requirement.dtype() == DType::kByte && shape.ndim == 1,
               "Grouped workspace query must return a one-dimensional byte tensor.");
    const size_t bytes = shape.data[0];
    const size_t capacity = storage_.defined() ? storage_.numel() : 0;
    if (!has_context_ || bytes > reserved_bytes_) {
      cudaStreamCaptureStatus capture_status;
      NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture_status));
      NVTE_CHECK(capture_status == cudaStreamCaptureStatusNone,
                 "Prepare Workspace with the required shapes before CUDA graph capture.");
      if (bytes > capacity) {
        constexpr size_t max_bytes = std::numeric_limits<int64_t>::max();
        NVTE_CHECK(bytes <= max_bytes, "Workspace is too large for a PyTorch tensor.");
        const size_t allocation_bytes =
            std::max(bytes, capacity <= max_bytes / 2 ? capacity * 2 : bytes);
        auto storage = at::empty({static_cast<int64_t>(allocation_bytes)},
                                 at::TensorOptions().device(at::kCUDA, device).dtype(at::kByte));
        if (storage_.defined()) {
          retired_.push_back(storage_);
        }
        storage_ = std::move(storage);
        ++generation_;
      }
      workspace_ = std::move(requirement);
      reserved_bytes_ = bytes;
      workspace_.set_rowwise_data(storage_.data_ptr(), workspace_.dtype(), workspace_.shape());
    }
    device_ = device;
    stream_ = stream;
    has_context_ = true;
    operation(workspace_.data());
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reserved_bytes_;
  }
  size_t capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return storage_.defined() ? storage_.numel() : 0;
  }
  uint64_t generation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generation_;
  }

 private:
  mutable std::mutex mutex_;
  TensorWrapper workspace_;
  size_t reserved_bytes_ = 0;
  uint64_t generation_ = 0;
  at::Tensor storage_;
  std::vector<at::Tensor> retired_;
  bool has_context_ = false;
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
};

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_WORKSPACE_H_
