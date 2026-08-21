/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// The fused-attention graph cache: what a cache entry is, and how one is built, cached and found
// again. The four build sites for cuDNN graphs -- f16/fp8 crossed with fwd/bwd -- differ only in
// what the graph computes and which tensors it binds; caching, lookup, locking, the support query
// and the plan build are the same for all four and live here.
//
// The pieces below elide the `backend, pass` pair most of them also take: it never steers the
// logic, only attributing debug counters and stage timings to a build site.
//
// - CacheEntry: a graph, the tensors it binds as inputs and outputs, and a once_flag guarding its
//   plan build.
// - GraphCache: process-wide map from a normalized FusedAttnConfig to a CacheEntry.
// - get_graph<backend, pass, kCreateGraphFn>(cfg, handle): the execution path's way in. Keys `cfg`
//   and owns the cache for its one triple; kCreateGraphFn is a create_graph_f16/fp8_fwd/bwd from a
//   .cu file, the only piece a build site supplies.
// - support_verdict<...>(cfg, handle): the backend selector's way in. get_graph() in a try,
//   returning the empty string when cuDNN accepts the graph and its complaint when it does not.
// - cache_graph(cache, key, handle, build): a hit, or a build under frontend_build_mutex() and an
//   insert. The work behind both of the above.
// - query_support(graph, handle): takes a constructed graph through validate,
//   build_operation_graph, create_execution_plans and check_support; throws cuDNN's message on
//   refusal.
// - build_plans(entry): the kernel compilation cache_graph() deferred, once per entry, no handle.

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

inline std::mutex &frontend_build_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Takes a constructed graph through the frontend calls that decide whether cuDNN can run it.
// `backend` and `pass` only name the build site the stage timers attribute the calls to.
//
// Reports by throwing, carrying cuDNN's message alone: that message is what support_verdict()
// returns as the reason a backend was refused.
inline void query_support(Backend backend, Pass pass, cudnn_frontend::graph::Graph &graph,
                          cudnnHandle_t handle) {
  auto run = [&](graph_cache_debug::BuildStage stage, const char *call_name, auto &&call) {
    const cudnn_frontend::error_t error =
        graph_cache_debug::record_time(backend, pass, stage, [&] { return call(); });
    if (error.is_good()) return;
    throw std::runtime_error(error.err_msg.empty() ? std::string(call_name) + " failed."
                                                   : error.err_msg);
  };

  run(graph_cache_debug::BuildStage::Validate, "validate", [&] { return graph.validate(); });
  run(graph_cache_debug::BuildStage::BuildOpGraph, "build_operation_graph",
      [&] { return graph.build_operation_graph(handle); });
  run(graph_cache_debug::BuildStage::CreatePlans, "create_execution_plans",
      [&] { return graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}); });
  run(graph_cache_debug::BuildStage::CheckSupport, "check_support",
      [&] { return graph.check_support(); });
}

// Cache for the entry `key`; build it first if absent. Record the lookup result and return the entry.
//   hit  -> record HIT, return the entry
//   miss -> take frontend_build_mutex(), look again (a thread that raced us has finished by now),
//           record MISS, build(), query_support(), insert
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

  std::lock_guard<std::mutex> build_lock(frontend_build_mutex());
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

// Each backend's graph cache per forward/backward pass, called by support_verdict() and the
// execution path. The cache is this instantiation's static local, so the callers naming one
// <backend, pass, creator> triple share one cache and each triple gets its own.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
auto get_graph(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  static GraphCache<decltype(kCreateGraphFn(cfg))> cache;
  cfg.check_derived();
  return cache_graph(cache, cfg.make_cache_key(kPass), kBackend, kPass, handle,
                     [&] { return kCreateGraphFn(cfg); });
}

// Check whether cuDNN can support a given config, per forward/backward pass.
// Returns an empty string if can; otherwise, a diagnostic string for the reason.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
std::string support_verdict(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  auto label = [] {
    return std::string("support_verdict<") + graph_cache_debug::backend_name(kBackend) + ", " +
           graph_cache_debug::pass_name(kPass) + ">";
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

// Previous calls only create the graph, caches it if verified to be supported. This function
// compiles the kernels via graph.build_plans(). It is the most expensive frontend call, and
// done only once per cache entry.
template <typename GraphAndTensors>
void build_plans(Backend backend, Pass pass, CacheEntry<GraphAndTensors> &entry) {
  std::call_once(entry.build_plans_once, [&] {
    std::lock_guard<std::mutex> build_lock(frontend_build_mutex());
    cudnn_frontend::graph::Graph &graph = *std::get<0>(entry.graph_and_tensors);
    graph_cache_debug::record_time(backend, pass, graph_cache_debug::BuildStage::BuildPlans,
                                   [&] { NVTE_CHECK_CUDNN_FE(graph.build_plans()); });
    graph_cache_debug::record_build_plans(backend, pass);
  });
}

}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_
