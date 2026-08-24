/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Fused-attention graph cache.
//
// The fused-attention backend calls cuDNN frontend at four sites, (f16/fp8, fwd/bwd). They each
// create a different graph with different computational operations and input/output tensors, but
// they share the same mechanism for caching, support queries, error messaging, and plan building.
// This file implements that shared mechanism and is called by all four (f16/fp8, fwd/bwd) sites.
//
// Cache types:
// - CacheEntry: a cuDNN graph, the tensors it binds, and a once_flag guarding its plan build.
// - GraphCache: a process-wide map from a normalized FusedAttnConfig to a CacheEntry.
//
// Internals (namespace detail):
// - query_support(graph, handle): takes a constructed graph through a series of cuDNN frontend calls:
//   validate, build_operation_graph, create_execution_plans and check_support; returns true if cuDNN
//   supports it, otherwise throws cuDNN's message.
// - cache_graph(cache, key, handle, build): returns a cached entry if hit; otherwise, builds the graph
//   anew, checks it with query_support() and if supported, inserts it. The work behind get_graph() below.
//
// Entry points by (f16/fp8, fwd/bwd) implementations:
// - get_graph<backend, pass, kCreateGraphFn>(cfg, handle): normalizes cfg into a cache key and owns
//   the cache for its one <backend, pass, creator> triple; kCreateGraphFn is a create_graph_f16/fp8_fwd/bwd
//   from a .cu file, the only piece a build site supplies.
// - support_verdict<...>(cfg, handle): wraps get_graph() in a try, returning the empty string when
//   cuDNN accepts the graph and its rejection reason when it does not.
// - build_plans(entry): runs the kernel compilation that cache_graph() deferred, once per entry.

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_

#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "../common.h"
#include "../cudnn_utils.h"
#include "config_and_params.h"
#include "graph_cache_debug.h"

namespace transformer_engine {
namespace fused_attn {

template <typename GraphAndTensors>
struct CacheEntry {
  explicit CacheEntry(GraphAndTensors graph_and_tensors)
      : graph_and_tensors(std::move(graph_and_tensors)) {}

  GraphAndTensors graph_and_tensors;
  std::once_flag build_plans_once;
};

template <typename GraphAndTensors>
struct GraphCache {
  std::mutex mutex;
  std::map<FusedAttnConfig, std::shared_ptr<CacheEntry<GraphAndTensors>>> entries;
};

namespace detail {

// Query if a constructed graph can be supported by cuDNN; throw cuDNN's message on refusal.
inline void query_support(Backend backend, Pass pass, cudnn_frontend::graph::Graph &graph,
                          cudnnHandle_t handle) {
  using graph_cache_debug::BuildStage;

  auto run = [&](BuildStage stage, const char *call_name, auto &&call) {
    const cudnn_frontend::error_t error =
        graph_cache_debug::record_time(backend, pass, stage, [&] { return call(); });
    if (error.is_good()) return;
    throw std::runtime_error(error.err_msg.empty() ? std::string(call_name) + " failed."
                                                   : error.err_msg);
  };

  run(BuildStage::Validate, "validate", [&] { return graph.validate(); });
  run(BuildStage::BuildOpGraph, "build_operation_graph",
      [&] { return graph.build_operation_graph(handle); });
  run(BuildStage::CreatePlans, "create_execution_plans",
      [&] { return graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}); });
  run(BuildStage::CheckSupport, "check_support", [&] { return graph.check_support(); });
}

// Look up `key` in `cache` and if missed, build the graph from fresh
//   hit  -> record HIT, return the cached entry
//   miss -> record MISS, build(), query_support(), insert if supported and throw on refusal
//
// The cache lookup and insert are guarded by mutex, but not the graph builds. Multiple threads
// may build for the same key concurrently, but only the first build will be inserted.
template <typename GraphAndTensors, typename BuildFn>
std::shared_ptr<CacheEntry<GraphAndTensors>> cache_graph(GraphCache<GraphAndTensors> &cache,
                                                         const FusedAttnConfig &key,
                                                         Backend backend, Pass pass,
                                                         cudnnHandle_t handle, BuildFn &&build) {
  using graph_cache_debug::LookupResult;

  auto find = [&]() -> std::shared_ptr<CacheEntry<GraphAndTensors>> {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.entries.find(key);
    return it != cache.entries.end() ? it->second : nullptr;
  };

  if (std::shared_ptr<CacheEntry<GraphAndTensors>> cached = find()) {
    graph_cache_debug::record_hit_miss(backend, pass, LookupResult::Hit, key);
    return cached;
  }

  graph_cache_debug::record_hit_miss(backend, pass, LookupResult::Miss, key);

  auto entry = std::make_shared<CacheEntry<GraphAndTensors>>(build());
  graph_cache_debug::record_create_graph(backend, pass);

  query_support(backend, pass, *std::get<0>(entry->graph_and_tensors), handle);
  graph_cache_debug::record_cache_graph(backend, pass);

  std::lock_guard<std::mutex> lock(cache.mutex);
  return cache.entries.insert({key, std::move(entry)}).first->second;
}

}  // namespace detail

// Create a cache for each (backend, pass) pair, and either get cached entry or build anew.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
auto get_graph(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  static GraphCache<decltype(kCreateGraphFn(cfg))> cache;
  cfg.check_derived();
  return detail::cache_graph(cache, cfg.make_cache_key(kPass), kBackend, kPass, handle,
                             [&] { return kCreateGraphFn(cfg); });
}

// Check if cuDNN supports a given config.
// Return an empty string if it does, or a diagnostic string if not.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
std::string support_verdict(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  auto label = [] {
    return std::string("support_verdict<") + backend_name(kBackend) + ", " + pass_name(kPass) + ">";
  };
  try {
    get_graph<kBackend, kPass, kCreateGraphFn>(cfg, handle);
    return "";
  } catch (const std::exception &e) {
    const char *reason = e.what();
    if (reason != nullptr && reason[0] != '\0') return reason;
    return label() + ": rejected without a reason.";
  } catch (...) {
    return label() + ": unknown failure.";
  }
}

// Compile kernels for the graph, once per cache entry. Most expensive cuDNN frontend call.
template <typename GraphAndTensors>
void build_plans(Backend backend, Pass pass, CacheEntry<GraphAndTensors> &entry) {
  std::call_once(entry.build_plans_once, [&] {
    cudnn_frontend::graph::Graph &graph = *std::get<0>(entry.graph_and_tensors);
    graph_cache_debug::record_time(backend, pass, graph_cache_debug::BuildStage::BuildPlans,
                                   [&] { NVTE_CHECK_CUDNN_FE(graph.build_plans()); });
    graph_cache_debug::record_build_plans(backend, pass);
  });
}

}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_
