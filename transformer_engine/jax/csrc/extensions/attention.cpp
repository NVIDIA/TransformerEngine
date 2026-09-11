/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cudnn_frontend.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../extensions.h"
#include "attention_cache_debug.h"

namespace transformer_engine {
namespace jax {
namespace {

struct CudnnGraphScalarStorage {
  alignas(16) std::array<uint8_t, 16> data{};
};

struct CudnnGraphCacheKey {
  int device_id = 0;
  int64_t hash0 = 0;
  int64_t hash1 = 0;
  int64_t frontend_version = 0;

  bool operator==(const CudnnGraphCacheKey &other) const {
    return device_id == other.device_id && hash0 == other.hash0 && hash1 == other.hash1 &&
           frontend_version == other.frontend_version;
  }
};

struct CudnnGraphCacheKeyHash {
  size_t operator()(const CudnnGraphCacheKey &key) const {
    size_t seed = std::hash<int>{}(key.device_id);
    auto combine = [&seed](int64_t value) {
      seed ^= std::hash<int64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    combine(key.hash0);
    combine(key.hash1);
    combine(key.frontend_version);
    return seed;
  }
};

using CudnnGraphPtr = std::shared_ptr<cudnn_frontend::graph::Graph>;

std::unordered_map<CudnnGraphCacheKey, CudnnGraphPtr, CudnnGraphCacheKeyHash> &
GetCudnnGraphCache() {
  static std::unordered_map<CudnnGraphCacheKey, CudnnGraphPtr, CudnnGraphCacheKeyHash> cache;
  return cache;
}

std::mutex &GetCudnnGraphCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

struct CudnnHandleCache {
  std::unordered_map<int, cudnnHandle_t> handles;

  cudnnHandle_t GetHandle() {
    int device_id = 0;
    NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
    auto it = handles.find(device_id);
    if (it == handles.end()) {
      cudnnHandle_t handle = nullptr;
      NVTE_CHECK_CUDNN(cudnnCreate(&handle));
      it = handles.emplace(device_id, handle).first;
    }
    return it->second;
  }

  ~CudnnHandleCache() {
    for (auto &[_, handle] : handles) {
      cudnnDestroy(handle);
    }
  }
};

cudnnHandle_t GetCudnnHandle() {
  static thread_local CudnnHandleCache cache;
  return cache.GetHandle();
}

CudnnGraphCacheKey GetCudnnGraphCacheKey(Dictionary &attrs) {
  const int64_t frontend_version = get_attr_value<int64_t>(attrs, "cudnn_frontend_version");
  NVTE_CHECK(frontend_version == CUDNN_FRONTEND_VERSION,
             "cuDNN frontend version mismatch for graph deserialization: graph was serialized "
             "with Python frontend version ",
             frontend_version, ", but Transformer Engine C++ was built with version ",
             CUDNN_FRONTEND_VERSION, ".");

  int device_id = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  return CudnnGraphCacheKey{
      device_id,
      get_attr_value<int64_t>(attrs, "graph_hash0"),
      get_attr_value<int64_t>(attrs, "graph_hash1"),
      frontend_version,
  };
}

CudnnGraphPtr GetCudnnGraph(cudaStream_t stream, Dictionary &attrs) {
  const auto key = GetCudnnGraphCacheKey(attrs);
  {
    std::lock_guard<std::mutex> lock(GetCudnnGraphCacheMutex());
    auto &cache = GetCudnnGraphCache();
    auto it = cache.find(key);
    if (it != cache.end()) {
      return it->second;
    }
  }

  const auto serialized_graph = get_attr_value<std::string_view>(attrs, "serialized_graph");
  std::vector<uint8_t> serialized_data(serialized_graph.begin(), serialized_graph.end());
  auto handle = GetCudnnHandle();
  NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

  auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
  auto status = graph->deserialize(handle, serialized_data);
  NVTE_CHECK(status.is_good(),
             "Failed to deserialize cuDNN frontend graph: ", status.get_message());

  std::lock_guard<std::mutex> lock(GetCudnnGraphCacheMutex());
  auto &cache = GetCudnnGraphCache();
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  cache.emplace(key, graph);
  return graph;
}

Error_Type ExecuteCudnnGraph(cudaStream_t stream, Dictionary &attrs,
                             const std::vector<void *> &input_ptrs,
                             const std::vector<void *> &output_ptrs, void *workspace) {
  auto graph = GetCudnnGraph(stream, attrs);
  auto input_uids = get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "input_uids");
  auto input_buffer_indices =
      get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "input_buffer_indices");
  auto input_byte_offsets =
      get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "input_byte_offsets");
  auto output_uids = get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "output_uids");
  auto output_buffer_indices =
      get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "output_buffer_indices");
  auto output_byte_offsets =
      get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "output_byte_offsets");
  auto scalar_uids = get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "scalar_uids");
  auto scalar_sizes = get_attr_value<xla::ffi::Span<const int64_t>>(attrs, "scalar_sizes");
  auto scalar_values = get_attr_value<xla::ffi::Span<const uint8_t>>(attrs, "scalar_values");

  NVTE_CHECK(input_uids.size() == input_buffer_indices.size() &&
                 input_uids.size() == input_byte_offsets.size(),
             "Mismatched cuDNN graph input binding metadata.");
  NVTE_CHECK(output_uids.size() == output_buffer_indices.size() &&
                 output_uids.size() == output_byte_offsets.size(),
             "Mismatched cuDNN graph output binding metadata.");
  NVTE_CHECK(scalar_uids.size() == scalar_sizes.size(),
             "Mismatched cuDNN graph scalar uid/value-size counts.");
  NVTE_CHECK(scalar_values.size() == scalar_uids.size() * 16,
             "Mismatched cuDNN graph packed scalar value size.");

  std::unordered_map<int64_t, void *> variant_pack;
  for (size_t i = 0; i < input_uids.size(); ++i) {
    NVTE_CHECK(input_buffer_indices[i] >= 0 &&
                   static_cast<size_t>(input_buffer_indices[i]) < input_ptrs.size(),
               "cuDNN graph input binding index is out of range.");
    NVTE_CHECK(input_byte_offsets[i] >= 0, "cuDNN graph input byte offset must be non-negative.");
    auto *ptr = static_cast<uint8_t *>(input_ptrs[input_buffer_indices[i]]) + input_byte_offsets[i];
    variant_pack.emplace(input_uids[i], ptr);
  }
  for (size_t i = 0; i < output_uids.size(); ++i) {
    NVTE_CHECK(output_buffer_indices[i] >= 0 &&
                   static_cast<size_t>(output_buffer_indices[i]) < output_ptrs.size(),
               "cuDNN graph output binding index is out of range.");
    NVTE_CHECK(output_byte_offsets[i] >= 0, "cuDNN graph output byte offset must be non-negative.");
    auto *ptr =
        static_cast<uint8_t *>(output_ptrs[output_buffer_indices[i]]) + output_byte_offsets[i];
    variant_pack.emplace(output_uids[i], ptr);
  }

  std::vector<CudnnGraphScalarStorage> scalar_storage(scalar_uids.size());
  for (size_t i = 0; i < scalar_uids.size(); ++i) {
    NVTE_CHECK(scalar_sizes[i] >= 0 && scalar_sizes[i] <= 16,
               "cuDNN graph pass-by-value scalars must be at most 16 bytes.");
    std::copy_n(scalar_values.begin() + i * 16, 16, scalar_storage[i].data.begin());
    variant_pack.emplace(scalar_uids[i], scalar_storage[i].data.data());
  }

  auto handle = GetCudnnHandle();
  NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));
  int device_id = 0;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  attention_cache_debug::Record(
      get_attr_value<std::string_view>(attrs, "attention_backend"),
      get_attr_value<std::string_view>(attrs, "attention_direction"), "execute", device_id);
  auto status = graph->execute(handle, variant_pack, workspace);
  NVTE_CHECK(status.is_good(), "cuDNN frontend graph execution failed: ", status.get_message());
  return ffi_with_cuda_error_check();
}

void AppendRemainingBuffers(Variadic_Buffer_Type args, std::vector<void *> *ptrs) {
  ptrs->reserve(ptrs->size() + args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    auto maybe_buf = args.get<Buffer_Type>(i);
    NVTE_CHECK(!maybe_buf.has_error(), "Failed to decode variadic cuDNN graph input buffer.");
    ptrs->push_back(maybe_buf.value().untyped_data());
  }
}

size_t BufferBytes(const Buffer_Type &buffer) { return buffer.size_bytes(); }

void MemsetResultAsync(cudaStream_t stream, Result_Type result, int value) {
  NVTE_CHECK_CUDA(cudaMemsetAsync(result->untyped_data(), value, BufferBytes(*result), stream));
}

class FusedAttnOffsetManager {
 public:
  static FusedAttnOffsetManager &Instance() {
    static thread_local FusedAttnOffsetManager manager;
    return manager;
  }

  uint64_t GetAndUpdate(uint64_t increment) {
    uint64_t current = offset_;
    offset_ += increment;
    return current;
  }

 private:
  uint64_t offset_ = 0;
};

void PopulateRngStateAsync(cudaStream_t stream, const Buffer_Type &seed, Result_Type rng_state,
                           uint64_t increment) {
  NVTE_CHECK(BufferBytes(seed) >= sizeof(uint64_t), "Fused-attention seed buffer is too small.");
  NVTE_CHECK(BufferBytes(*rng_state) >= 2 * sizeof(uint64_t),
             "Fused-attention RNG-state buffer is too small.");
  const uint64_t offset = FusedAttnOffsetManager::Instance().GetAndUpdate(increment);
  PopulateFusedAttnRngState(rng_state->untyped_data(), seed.untyped_data(), offset, stream);
}

}  // namespace

Error_Type FusedAttnForwardFFI(cudaStream_t stream, Buffer_Type q_buf, Buffer_Type k_buf,
                               Buffer_Type v_buf, Buffer_Type bias_buf,
                               Buffer_Type softmax_offset_buf, Buffer_Type seed_buf,
                               Buffer_Type q_seqlens_buf, Buffer_Type kv_seqlens_buf,
                               Buffer_Type q_seq_offsets_buf, Buffer_Type k_seq_offsets_buf,
                               Variadic_Buffer_Type remaining_args, Result_Type output_buf,
                               Result_Type stats_buf, Result_Type max_buf,
                               Result_Type rng_state_buf, Result_Type workspace_buf,
                               Dictionary attrs) {
  const bool is_ragged = get_attr_value<bool>(attrs, "is_ragged");
  const uint64_t rng_increment =
      static_cast<uint64_t>(get_attr_value<int64_t>(attrs, "rng_offset_increment"));
  PopulateRngStateAsync(stream, seed_buf, rng_state_buf, rng_increment);
  if (is_ragged) {
    MemsetResultAsync(stream, output_buf, 0);
    MemsetResultAsync(stream, stats_buf, 0xF0);
    if (BufferBytes(*max_buf) != 0) {
      MemsetResultAsync(stream, max_buf, 0xF0);
    }
  }

  std::vector<void *> input_ptrs = {
      q_buf.untyped_data(),
      k_buf.untyped_data(),
      v_buf.untyped_data(),
      bias_buf.untyped_data(),
      softmax_offset_buf.untyped_data(),
      seed_buf.untyped_data(),
      q_seqlens_buf.untyped_data(),
      kv_seqlens_buf.untyped_data(),
      q_seq_offsets_buf.untyped_data(),
      k_seq_offsets_buf.untyped_data(),
  };
  AppendRemainingBuffers(remaining_args, &input_ptrs);
  std::vector<void *> output_ptrs = {output_buf->untyped_data(), stats_buf->untyped_data(),
                                     max_buf->untyped_data(), rng_state_buf->untyped_data()};
  return ExecuteCudnnGraph(stream, attrs, input_ptrs, output_ptrs, workspace_buf->untyped_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedAttnForwardHandler, FusedAttnForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .RemainingArgs()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Attrs(),
                              FFI_CudaGraph_Traits);

Error_Type FusedAttnBackwardFFI(cudaStream_t stream, Buffer_Type q_buf, Buffer_Type k_buf,
                                Buffer_Type v_buf, Buffer_Type bias_buf,
                                Buffer_Type softmax_offset_buf, Buffer_Type stats_buf,
                                Buffer_Type rng_state_buf, Buffer_Type output_buf,
                                Buffer_Type doutput_buf, Buffer_Type q_seqlens_buf,
                                Buffer_Type kv_seqlens_buf, Buffer_Type q_seq_offsets_buf,
                                Buffer_Type k_seq_offsets_buf, Variadic_Buffer_Type remaining_args,
                                Result_Type dq_buf, Result_Type dk_buf, Result_Type dv_buf,
                                Result_Type dbias_buf, Result_Type dsoftmax_offset_buf,
                                Result_Type workspace_buf, Dictionary attrs) {
  if (get_attr_value<bool>(attrs, "is_ragged")) {
    MemsetResultAsync(stream, dq_buf, 0);
    MemsetResultAsync(stream, dk_buf, 0);
    MemsetResultAsync(stream, dv_buf, 0);
  }
  std::vector<void *> input_ptrs = {
      q_buf.untyped_data(),
      k_buf.untyped_data(),
      v_buf.untyped_data(),
      bias_buf.untyped_data(),
      softmax_offset_buf.untyped_data(),
      stats_buf.untyped_data(),
      rng_state_buf.untyped_data(),
      output_buf.untyped_data(),
      doutput_buf.untyped_data(),
      q_seqlens_buf.untyped_data(),
      kv_seqlens_buf.untyped_data(),
      q_seq_offsets_buf.untyped_data(),
      k_seq_offsets_buf.untyped_data(),
  };
  AppendRemainingBuffers(remaining_args, &input_ptrs);
  std::vector<void *> output_ptrs = {
      dq_buf->untyped_data(),
      dk_buf->untyped_data(),
      dv_buf->untyped_data(),
      dbias_buf->untyped_data(),
      dsoftmax_offset_buf->untyped_data(),
  };
  return ExecuteCudnnGraph(stream, attrs, input_ptrs, output_ptrs, workspace_buf->untyped_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedAttnBackwardHandler, FusedAttnBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .RemainingArgs()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Attrs(),
                              FFI_CudaGraph_Traits);

Error_Type FusedAttnScoreModForwardFFI(cudaStream_t stream, Buffer_Type q_buf, Buffer_Type k_buf,
                                       Buffer_Type v_buf, Variadic_Buffer_Type score_mod_args,
                                       Result_Type output_buf, Result_Type stats_buf,
                                       Result_Type workspace_buf, Dictionary attrs) {
  std::vector<void *> input_ptrs = {q_buf.untyped_data(), k_buf.untyped_data(),
                                    v_buf.untyped_data()};
  AppendRemainingBuffers(score_mod_args, &input_ptrs);
  std::vector<void *> output_ptrs = {output_buf->untyped_data(), stats_buf->untyped_data()};
  return ExecuteCudnnGraph(stream, attrs, input_ptrs, output_ptrs, workspace_buf->untyped_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedAttnScoreModForwardHandler, FusedAttnScoreModForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .RemainingArgs()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Attrs(),
                              FFI_CudaGraph_Traits);

Error_Type FusedAttnScoreModBackwardFFI(cudaStream_t stream, Buffer_Type q_buf, Buffer_Type k_buf,
                                        Buffer_Type v_buf, Buffer_Type output_buf,
                                        Buffer_Type doutput_buf, Buffer_Type stats_buf,
                                        Variadic_Buffer_Type score_mod_args, Result_Type dq_buf,
                                        Result_Type dk_buf, Result_Type dv_buf,
                                        Result_Type workspace_buf, Dictionary attrs) {
  std::vector<void *> input_ptrs = {q_buf.untyped_data(),       k_buf.untyped_data(),
                                    v_buf.untyped_data(),       output_buf.untyped_data(),
                                    doutput_buf.untyped_data(), stats_buf.untyped_data()};
  AppendRemainingBuffers(score_mod_args, &input_ptrs);
  std::vector<void *> output_ptrs = {dq_buf->untyped_data(), dk_buf->untyped_data(),
                                     dv_buf->untyped_data()};
  return ExecuteCudnnGraph(stream, attrs, input_ptrs, output_ptrs, workspace_buf->untyped_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedAttnScoreModBackwardHandler, FusedAttnScoreModBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .Arg<Buffer_Type>()
                                  .RemainingArgs()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Ret<Buffer_Type>()
                                  .Attrs(),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
