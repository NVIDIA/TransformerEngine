/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_GEMM_CONFIG_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_GEMM_CONFIG_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "../tvm_ffi_bridge.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {
namespace ep {

// Signal-aware grouped GEMMs (CuTeDSL via TVM-FFI), keyed by configuration. The first
// get_kernel of a configuration compiles: call it at warmup, outside graph capture.

// Byte offsets inside a GEMM workspace: total size, then the words the entry points reset.
struct WorkspaceLayout {
  size_t total_bytes;
  size_t counter;  // dispatch GEMM, weight gradient: claim counter
  size_t signal;   // dispatch GEMM: tile flags; combine GEMM: column-chunk flags
  size_t state;    // dispatch GEMM: scheduler state
};

// Compiled functions and workspace layouts depend on the architecture of the current device.
inline std::string arch_key() { return "_sm" + std::to_string(cuda::sm_arch()); }

// Cache keyed by a configuration string, read under a shared lock. Configurations carry sizes that
// do not fit the 32-bit id of tvm_ffi_bridge::TVMFFIConfigCache.
template <typename Value>
class KeyedCache {
 public:
  // Process lifetime, as TVMFFIConfigCache: nothing is destroyed at static teardown.
  static KeyedCache &create() { return *new KeyedCache(); }

  // The value of key, from make() on a miss. A missing value is cached only with cache_missing.
  template <typename Make>
  std::optional<Value> get(const std::string &key, Make &&make, bool cache_missing = false) {
    {
      std::shared_lock<std::shared_mutex> lock(mutex_);
      auto it = map_.find(key);
      if (it != map_.end()) {
        return it->second;
      }
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
      return it->second;
    }
    std::optional<Value> value = make();
    if (value.has_value() || cache_missing) {
      map_.emplace(key, value);
    }
    return value;
  }

 private:
  KeyedCache() = default;

  std::shared_mutex mutex_;
  std::unordered_map<std::string, std::optional<Value>> map_;
};

// Cache key: the prefix, every argument of the configuration, and the device architecture.
template <typename Args>
std::string config_key(const char *prefix, const Args &args) {
  std::string key = prefix;
  std::apply([&](const auto &...value) { ((key += "_" + std::to_string(value)), ...); }, args);
  return key + arch_key();
}

template <typename Config>
std::optional<tvm::ffi::Function> get_cached_kernel(const Config &config) {
  tvm_ffi_bridge::TVMFFICentral &central = tvm_ffi_bridge::TVMFFICentral::getInstance();
  if (!central.get_cutedsl_backend_enabled()) {
    return central.load_tvm_ffi_function(config);
  }
  static auto &cache = KeyedCache<tvm::ffi::Function>::create();
  return cache.get(
      config.to_key(), [&] { return central.load_tvm_ffi_function(config); },
      /*cache_missing=*/true);
}

// Calls the global function `name` with the configuration's arguments, prefixed by `prefix`.
template <typename Config, typename... Prefix>
std::optional<tvm::ffi::Any> call_entry(const char *name, const Config &config,
                                        const Prefix &...prefix) {
  auto fn = tvm::ffi::Function::GetGlobal(name);
  if (!fn.has_value()) {
    return std::nullopt;
  }
  return std::apply([&](const auto &...args) { return (*fn)(prefix..., args...); }, config.args());
}

template <typename Config>
bool retrieve_from_python(const Config &config, const std::string &fn_name) {
  auto result = call_entry(Config::kEntrypointName, config, tvm::ffi::String(fn_name));
  return result.has_value() && result->template try_cast<bool>().value_or(false);
}

// The layout entry returns [total, counter, signal, state], [total, signal, entries] or
// [total, counter].
template <typename Config>
std::optional<WorkspaceLayout> query_layout_from_python(const Config &config) {
  auto result = call_entry(Config::kLayoutName, config);
  if (!result.has_value()) {
    return std::nullopt;
  }
  auto offsets = result->template try_cast<tvm::ffi::Array<int64_t>>();
  if (!offsets.has_value()) {
    return std::nullopt;
  }
  const auto at = [&](size_t i) { return static_cast<size_t>((*offsets)[i]); };
  if (offsets->size() == 4) {
    return WorkspaceLayout{at(0), at(1), at(2), at(3)};
  }
  if (offsets->size() == 3) {
    return WorkspaceLayout{at(0), 0, at(1), 0};
  }
  if (offsets->size() == 2) {
    return WorkspaceLayout{at(0), at(1), 0, 0};
  }
  return std::nullopt;
}

// query_layout_from_python, kept per configuration once found.
template <typename Config>
std::optional<WorkspaceLayout> query_layout(const Config &config) {
  static auto &cache = KeyedCache<WorkspaceLayout>::create();
  return cache.get(config.to_key(), [&] { return query_layout_from_python(config); });
}

// GEMM behind the dispatch: waits on the dispatch tile flags.
struct EpDispatchGemmConfig {
  static constexpr const char *kEntrypointName = "get_ep_dispatch_gemm_function";
  static constexpr const char *kLayoutName = "get_ep_dispatch_gemm_workspace_layout";

  size_t rows;  // receive capacity
  size_t k;     // reduction size
  size_t n;     // GEMM output columns
  size_t num_experts;
  int num_clusters;  // persistent clusters beside its NCCL EP call's SMs
  bool backward;  // the second GEMM's data gradient and SwiGLU backward instead of the first GEMM
  bool mxfp8;     // MXFP8 operands instead of BF16
  size_t rows_alignment;  // rows per expert slab are a multiple of it
  // Forward: the dispatch writes the expert offsets and the row total, the word before the row
  // ends and the row ends, after the caller set them to -1.
  bool device_layout = false;

  // In the parameter order of the get_ep_* function of CuTeDSL/ep_gemm/dispatch_gemm.py.
  auto args() const {
    return std::make_tuple(static_cast<int64_t>(rows), static_cast<int64_t>(k),
                           static_cast<int64_t>(n), static_cast<int64_t>(num_experts), num_clusters,
                           backward, mxfp8, static_cast<int64_t>(rows_alignment), device_layout);
  }

  std::string to_key() const { return config_key("ep_dispatch_gemm", args()); }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    return retrieve_from_python(*this, fn_name);
  }

  std::optional<tvm::ffi::Function> get_kernel() const { return get_cached_kernel(*this); }

  std::optional<WorkspaceLayout> workspace_layout() const { return query_layout(*this); }
};

// GEMM in front of the combine: publishes finished column chunks of its output.
struct EpCombineGemmConfig {
  static constexpr const char *kEntrypointName = "get_ep_combine_gemm_function";
  static constexpr const char *kLayoutName = "get_ep_combine_gemm_workspace_layout";

  size_t rows;  // receive capacity
  size_t k;     // reduction size
  size_t n;     // output columns
  size_t num_experts;
  int num_clusters;  // persistent clusters beside the combine's SMs
  size_t n_chunk;    // columns per readiness flag
  bool dependents_at_entry;
  bool backward;          // the data gradient of the first GEMM instead of the second GEMM
  bool mxfp8;             // MXFP8 operands instead of BF16
  size_t rows_alignment;  // rows per expert slab are a multiple of it
  // Clusters launched, 0: num_clusters. Those past num_clusters stop after the first chunk.
  int grid_clusters = 0;

  // In the parameter order of the get_ep_* function of CuTeDSL/ep_gemm/combine_gemm.py.
  auto args() const {
    return std::make_tuple(static_cast<int64_t>(rows), static_cast<int64_t>(k),
                           static_cast<int64_t>(n), static_cast<int64_t>(num_experts), num_clusters,
                           static_cast<int64_t>(n_chunk), dependents_at_entry, backward, mxfp8,
                           static_cast<int64_t>(rows_alignment), grid_clusters);
  }

  std::string to_key() const { return config_key("ep_combine_gemm", args()); }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    return retrieve_from_python(*this, fn_name);
  }

  std::optional<tvm::ffi::Function> get_kernel() const { return get_cached_kernel(*this); }

  // total_bytes and the byte offset of signal[]; the other words are unused.
  std::optional<WorkspaceLayout> workspace_layout() const { return query_layout(*this); }
};

// Weight gradient chained behind a backward GEMM: a persistent grid that claims output tiles
// from the workspace counter, so launches sharing one workspace split the tiles.
struct EpWgradConfig {
  static constexpr const char *kEntrypointName = "get_ep_wgrad_function";
  static constexpr const char *kLayoutName = "get_ep_wgrad_workspace_layout";
  static constexpr const char *kDescriptorsName = "get_ep_wgrad_reads_descriptors";

  size_t rows;  // receive capacity
  size_t m;     // rows of each expert's weight gradient
  size_t n;     // columns of each expert's weight gradient
  size_t num_experts;
  int num_clusters;            // persistent clusters
  bool fp32_out;               // FP32 instead of BF16 weight gradient
  bool accumulate;             // add to the weight gradient instead of writing it
  bool wait_primary_at_start;  // wait for the kernel before it ahead of any load
  size_t ready_slot_rows;      // wait for every dispatch flag of this many rows, 0: none
  // MXFP8 operands: the GEMM launches of some weight gradients (reads_descriptors) read the
  // per-expert descriptors that one launch with descriptors_only writes into the workspace first.
  bool mxfp8;
  bool descriptors_only;
  size_t rows_alignment;  // rows per expert slab are a multiple of it
  size_t routed_rows;     // rows a rank receives at capacity under uniform routing, unpadded

  // In the parameter order of the get_ep_* function of CuTeDSL/ep_gemm/wgrad_gemm.py.
  auto args() const {
    return std::make_tuple(static_cast<int64_t>(rows), static_cast<int64_t>(m),
                           static_cast<int64_t>(n), static_cast<int64_t>(num_experts), num_clusters,
                           fp32_out, accumulate, wait_primary_at_start,
                           static_cast<int64_t>(ready_slot_rows), mxfp8, descriptors_only,
                           static_cast<int64_t>(rows_alignment), static_cast<int64_t>(routed_rows));
  }

  std::string to_key() const { return config_key("ep_wgrad", args()); }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    return retrieve_from_python(*this, fn_name);
  }

  std::optional<tvm::ffi::Function> get_kernel() const { return get_cached_kernel(*this); }

  // total_bytes and the byte offset of the claim counter.
  std::optional<WorkspaceLayout> workspace_layout() const { return query_layout(*this); }

  // Whether the GEMM launches read the per-expert descriptors of a descriptors_only launch, kept
  // per configuration once found. Without an answer, the added and FP32 ones read them.
  bool reads_descriptors() const {
    static auto &cache = KeyedCache<bool>::create();
    return cache
        .get(to_key(),
             [&]() -> std::optional<bool> {
               auto result = call_entry(kDescriptorsName, *this);
               return result.has_value() ? result->template try_cast<bool>() : std::nullopt;
             })
        .value_or(mxfp8 && (accumulate || fp32_out));
  }
};

// Rowwise to columnwise MXFP8 requant of received rows for a weight gradient: one CTA per tile of
// the receive capacity.
struct EpRequantConfig {
  static constexpr const char *kEntrypointName = "get_ep_requant_function";

  size_t rows;  // receive capacity
  size_t cols;
  size_t num_experts;
  size_t ready_slot_rows;  // wait for the dispatch flags of a tile's rows, this many each; 0: none

  // In the parameter order of the get_ep_* function of CuTeDSL/ep_gemm/requantize.py.
  auto args() const {
    return std::make_tuple(static_cast<int64_t>(rows), static_cast<int64_t>(cols),
                           static_cast<int64_t>(num_experts),
                           static_cast<int64_t>(ready_slot_rows));
  }

  std::string to_key() const { return config_key("ep_requant", args()); }

  bool retrieve_func_from_python(const std::string &fn_name) const {
    return retrieve_from_python(*this, fn_name);
  }

  std::optional<tvm::ffi::Function> get_kernel() const { return get_cached_kernel(*this); }
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_GEMM_CONFIG_H_
