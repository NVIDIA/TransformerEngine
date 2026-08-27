/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Fused-attention graph cache.
//
// The four fused-attention implementation sites, (f16/fp8 + fwd/bwd), each create a different graph.
// They differ in the operations in the graph and the input/output tensors that bind to the graph,
// but the mechanism used for their graph caching, support queries, error messaging, and plan building
// is the same. They all call these three functions in this file: get_graph(), support_verdict(), and
// build_plans().

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

// An entry in graph cache; contains a cuDNN graph, its input/output tensors, and
// a once_flag that guards its plan build
template <typename GraphAndTensors>
struct CacheEntry {
  explicit CacheEntry(GraphAndTensors graph_and_tensors)
      : graph_and_tensors(std::move(graph_and_tensors)) {}

  GraphAndTensors graph_and_tensors;
  std::once_flag build_plans_once;
};

// The graph cache; a process-wide map that maps a normalized FusedAttnConfig to a CacheEntry
template <typename GraphAndTensors>
struct GraphCache {
  std::mutex mutex;
  std::map<FusedAttnConfig, std::shared_ptr<CacheEntry<GraphAndTensors>>> entries;
};

namespace detail {

// Query if a cuDNN graph can be supported or not; if so, safely return; if not, throw with
// cuDNN frontend's original error message; times for the four stages are also recorded.
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

// Look up the key in the cache and if
//   hit  -> record HIT, return the cached entry
//   miss -> record MISS, run build() to get a new graph, run query_support() on the new graph,
//           if supported, insert it to the cache; if not, throw cuDNN frontend's original
//           error message
//
// The cache lookup and insert are guarded by mutex, not the graph builds. Multiple threads
// may build for the same key concurrently, but only the first successful build will be inserted.
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

// Check if cuDNN supports a given config; if yes, return an empty string; if not, return a diagnostic
// string with the reason that get_graph() throws.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
std::string support_verdict(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  const char *fallback = nullptr;
  try {
    get_graph<kBackend, kPass, kCreateGraphFn>(cfg, handle);
    return "";
  } catch (const std::exception &e) {
    if (e.what()[0] != '\0') return e.what();
    fallback = "rejected without a reason.";
  } catch (...) {
    fallback = "unknown failure.";
  }
  return std::string("support_verdict<") + backend_name(kBackend) + ", " + pass_name(kPass) +
         ">: " + fallback;
}

// Compile the kernels for the graph before execution; once per cache entry; most expensive cuDNN
// frontend call in the pre-execution, preparation process.
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
