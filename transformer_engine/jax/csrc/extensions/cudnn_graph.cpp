/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// JAX-owned execution of generic graphs built with the cuDNN Python frontend.
// This translation unit deliberately has no TE common dependencies.
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_version.h>
#include <xla/ffi/api/ffi.h>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace transformer_engine {
namespace jax {
namespace {

namespace ffi = xla::ffi;

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void CheckCudnn(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) throw std::runtime_error(cudnnGetErrorString(status));
}

template <typename T>
T Attribute(ffi::Dictionary &attrs, const char *name) {
  auto value = attrs.get<T>(name);
  if (value.has_error()) {
    throw std::runtime_error(std::string("Missing cuDNN graph attribute: ") + name);
  }
  return value.value();
}

struct Handle {
  cudnnHandle_t value = nullptr;
  Handle() { CheckCudnn(cudnnCreate(&value)); }
  ~Handle() { cudnnDestroy(value); }
};

cudnnHandle_t GetHandle() {
  // Handles are thread-local and device-local; assign the stream on each call.
  thread_local std::unordered_map<int, std::unique_ptr<Handle>> handles;
  int device;
  CheckCuda(cudaGetDevice(&device));
  auto &handle = handles[device];
  if (!handle) handle = std::make_unique<Handle>();
  return handle->value;
}

using Graph = cudnn_frontend::graph::Graph;

std::shared_ptr<Graph> GetGraph(cudnnHandle_t handle, ffi::Dictionary &attrs) {
  const auto version = Attribute<int64_t>(attrs, "cudnn_frontend_version");
  if (version != CUDNN_FRONTEND_VERSION) {
    throw std::runtime_error(
        "cuDNN frontend Python/C++ version mismatch for graph deserialization.");
  }
  int device;
  CheckCuda(cudaGetDevice(&device));
  const auto key = std::to_string(device) + ":" +
                   std::to_string(Attribute<int64_t>(attrs, "graph_hash0")) + ":" +
                   std::to_string(Attribute<int64_t>(attrs, "graph_hash1")) + ":" +
                   std::to_string(version);
  static std::mutex mutex;
  static std::unordered_map<std::string, std::shared_ptr<Graph>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto &graph = cache[key];
  if (!graph) {
    const auto data = Attribute<std::string_view>(attrs, "serialized_graph");
    auto candidate = std::make_shared<Graph>();
    auto status = candidate->deserialize(handle, std::vector<uint8_t>(data.begin(), data.end()),
                                         /*enforce_precompiled=*/true);
    if (!status.is_good()) throw std::runtime_error(status.get_message());
    graph = std::move(candidate);
  }
  return graph;
}

struct Scalar {
  alignas(16) std::array<uint8_t, 16> value{};
};

ffi::Error InitializeGraph(cudaStream_t stream, ffi::RemainingArgs, ffi::RemainingRets,
                           ffi::Dictionary attrs) {
  try {
    auto handle = GetHandle();
    CheckCudnn(cudnnSetStream(handle, stream));
    GetGraph(handle, attrs);
    return ffi::Error::Success();
  } catch (const std::exception &error) {
    return ffi::Error::Internal(std::string("JAX cuDNN attention initialization: ") + error.what());
  }
}

ffi::Error ExecuteGraph(cudaStream_t stream, ffi::RemainingArgs args, ffi::RemainingRets rets,
                        ffi::Dictionary attrs) {
  try {
    auto handle = GetHandle();
    CheckCudnn(cudnnSetStream(handle, stream));
    auto graph = GetGraph(handle, attrs);
    auto input_uids = Attribute<ffi::Span<const int64_t>>(attrs, "input_uids");
    auto output_uids = Attribute<ffi::Span<const int64_t>>(attrs, "output_uids");
    auto input_indices = attrs.get<ffi::Span<const int64_t>>("input_indices");
    auto input_offsets = attrs.get<ffi::Span<const int64_t>>("input_offsets");
    auto output_indices = attrs.get<ffi::Span<const int64_t>>("output_indices");
    auto output_offsets = attrs.get<ffi::Span<const int64_t>>("output_offsets");
    auto output_fill_values = attrs.get<ffi::Span<const int64_t>>("output_fill_values");
    if (rets.size() == 0 ||
        (output_fill_values.has_value() && output_fill_values.value().size() != rets.size() - 1)) {
      throw std::runtime_error("Invalid cuDNN output initialization bindings.");
    }
    std::unordered_map<int64_t, void *> pointers;
    for (size_t i = 0; i < input_uids.size(); ++i) {
      const auto index = input_indices.has_value() ? input_indices.value()[i] : i;
      auto buf = args.get<ffi::AnyBuffer>(index);
      if (buf.has_error()) throw std::runtime_error("Invalid cuDNN input buffer binding.");
      auto ptr = static_cast<uint8_t *>(buf.value().untyped_data());
      pointers.emplace(input_uids[i],
                       ptr + (input_offsets.has_value() ? input_offsets.value()[i] : 0));
    }
    for (size_t i = 0; i < rets.size() - 1; ++i) {
      auto buf = rets.get<ffi::AnyBuffer>(i);
      if (buf.has_error()) throw std::runtime_error("Invalid cuDNN output buffer binding.");
      // Negative or absent fill values leave fully written outputs uninitialized.
      const int64_t fill = output_fill_values.has_value() ? output_fill_values.value()[i] : -1;
      if (fill >= 0 && buf.value()->size_bytes()) {
        CheckCuda(
            cudaMemsetAsync(buf.value()->untyped_data(), fill, buf.value()->size_bytes(), stream));
      }
    }
    for (size_t i = 0; i < output_uids.size(); ++i) {
      if (output_uids[i] < 0) continue;
      const auto index = output_indices.has_value() ? output_indices.value()[i] : i;
      auto buf = rets.get<ffi::AnyBuffer>(index);
      if (buf.has_error()) throw std::runtime_error("Invalid cuDNN output buffer binding.");
      auto ptr = static_cast<uint8_t *>(buf.value()->untyped_data());
      pointers.emplace(output_uids[i],
                       ptr + (output_offsets.has_value() ? output_offsets.value()[i] : 0));
    }
    auto scalar_uids = Attribute<ffi::Span<const int64_t>>(attrs, "scalar_uids");
    auto scalar_sizes = Attribute<ffi::Span<const int64_t>>(attrs, "scalar_sizes");
    auto scalar_values = Attribute<ffi::Span<const uint8_t>>(attrs, "scalar_values");
    if (scalar_sizes.size() != scalar_uids.size() ||
        scalar_values.size() != scalar_uids.size() * 16) {
      throw std::runtime_error("Invalid cuDNN pass-by-value scalar bindings.");
    }
    std::vector<Scalar> scalars(scalar_uids.size());
    for (size_t i = 0; i < scalars.size(); ++i) {
      if (scalar_sizes[i] < 0 || scalar_sizes[i] > 16) {
        throw std::runtime_error("Invalid cuDNN pass-by-value scalar size.");
      }
      std::copy_n(scalar_values.begin() + i * 16, 16, scalars[i].value.begin());
      pointers.emplace(scalar_uids[i], scalars[i].value.data());
    }
    auto workspace = rets.get<ffi::AnyBuffer>(rets.size() - 1);
    if (workspace.has_error()) throw std::runtime_error("Missing cuDNN graph workspace.");
    auto status = graph->execute(handle, pointers, workspace.value()->untyped_data());
    if (!status.is_good()) throw std::runtime_error(status.get_message());
    CheckCuda(cudaGetLastError());
    return ffi::Error::Success();
  } catch (const std::exception &error) {
    return ffi::Error::Internal(std::string("JAX cuDNN attention: ") + error.what());
  }
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudnnGraphHandler, ExecuteGraph,
                              xla::ffi::Ffi::Bind()
                                  .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attrs(),
                              {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(CudnnGraphInitializeHandler, InitializeGraph,
                              xla::ffi::Ffi::Bind<xla::ffi::ExecutionStage::kInitialize>()
                                  .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
                                  .RemainingArgs()
                                  .RemainingRets()
                                  .Attrs());

}  // namespace jax
}  // namespace transformer_engine
