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

// A graph in the cache, plus the tensor attributes needed to bind runtime pointers to it.
//
// Entries stop at check_support(), which is all it takes to decide support. graph.build_plans() --
// the kernel compilation, and the most expensive frontend call -- is left to the execution path,
// since a support query never runs the graph. build_plans_once guards that completion, which must
// happen exactly once per entry: the entry is shared across threads and graph.build_plans() mutates
// it in place. A build that throws leaves the flag unset, so a later call retries.
template <typename GraphAndTensors>
struct CacheEntry {
  explicit CacheEntry(GraphAndTensors graph_and_tensors)
      : graph_and_tensors(std::move(graph_and_tensors)) {}

  GraphAndTensors graph_and_tensors;
  std::once_flag build_plans_once;
};

// One build site's cache, process-wide rather than per-thread so a graph is reused across threads
// instead of rebuilt by each: cuDNN >= 9.0 allows concurrent execution of a shared plan, and the
// frontend's execute() builds its variant pack in a local rather than in the graph. A graph and its
// plans are compiled artifacts bound to the device they were finalized against, with nothing in
// them belonging to the building thread, which is why the key stamps device_id and nothing
// thread-shaped (see make_cache_key).
//
// The mutex is declared first so that it is destroyed last, leaving the map destroyed while its
// guard is still valid.
//
// The map is unbounded: a probe-only entry holds no compiled kernels, none hold a workspace, and a
// model reuses a handful of configurations. A workload that does sweep shapes holds every graph for
// the life of the process, which `miss` climbing without settling is the way to see.
template <typename GraphAndTensors>
struct GraphCache {
  std::mutex mutex;  // guards everything below
  std::map<FusedAttnConfig, std::shared_ptr<CacheEntry<GraphAndTensors>>> entries;
};

// Every cuDNN frontend call except graph.execute() runs holding this. The frontend serializes none
// of them for us, so two threads building unrelated keys is a data race, not the harmless duplicate
// work the map's view suggests. Not a theoretical exposure: a PyTorch step runs the forward and the
// backward's support probe on the main thread and the backward itself on the autograd thread.
//
// One lock for the process rather than one per cache, since what is unsafe is the frontend rather
// than any single graph. Kept separate from GraphCache::mutex, which guards only the map, so a hit
// never waits behind somebody else's kernel compilation.
//
// Lock ordering, which a later edit has to preserve: always taken before GraphCache::mutex, never
// after, and never held on entry to build_plans() -- holding it while waiting on that once_flag
// would deadlock against the thread holding the flag.
inline std::mutex &frontend_build_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Takes a constructed graph through the frontend calls that decide whether cuDNN can run it.
// `backend` and `pass` only name the build site the stage timers attribute the calls to.
//
// Reports by throwing, carrying cuDNN's message alone: that message is what support_verdict()
// returns as the reason a backend was refused, so a bool would discard the one thing a probe exists
// to produce, and NVTE_ERROR would dress a plain refusal as an internal failure.
inline void query_support(Backend backend, Pass pass, cudnn_frontend::graph::Graph &graph,
                          cudnnHandle_t handle) {
  auto run = [&](graph_cache_debug::BuildStage stage, const char *call_name, auto &&call) {
    const cudnn_frontend::error_t error =
        graph_cache_debug::record_time(backend, pass, stage, [&] { return call(); });
    if (error.is_good()) return;
    // cuDNN normally explains itself; fall back to the call's name so that a refusal can never
    // arrive as an empty string, which support_verdict() would read as an endorsement.
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

// The cached entry for `key`, building and inserting it via `build` if absent:
//
//   hit  -> record HIT, return the entry
//   miss -> take frontend_build_mutex(), look again (a thread that raced us has finished by now),
//           record MISS, build(), query_support(), insert
//
// `build` only constructs a graph; this is what puts it through query_support(), so the cache holds
// exactly the graphs cuDNN agreed to run, and a hit skips those calls. A refusal throws and stores
// nothing, so the next query for a refused key is refused again -- fine for a settled run, since
// the frameworks re-enter the selector only when the configuration changes.
//
// `key` must be make_cache_key(pass)'s output for the same `pass`, not a raw execution config: two
// configs differing only in a field no graph reads (attn_scale, say) have to reach the same entry.
//
// The second look keeps a lost race cheap -- the loser would otherwise hold the one build lock to
// produce a graph it drops on the next line -- and keeps exactly one HIT or MISS per call, so miss
// still equals create_graph.
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

  // Recorded after the lock is dropped, so writing a trace line cannot hold up threads querying
  // other keys. The counters stay exact, but two lookups that raced can be recorded in the opposite
  // order, so a level-2 trace is the set of lookups, not their sequence.
  if (std::shared_ptr<CacheEntry<GraphAndTensors>> cached = find()) {
    graph_cache_debug::record_hit_miss(backend, pass, LookupResult::Hit, key);
    return cached;
  }

  std::lock_guard<std::mutex> build_lock(frontend_build_mutex());
  if (std::shared_ptr<CacheEntry<GraphAndTensors>> cached = find()) {
    graph_cache_debug::record_hit_miss(backend, pass, LookupResult::Hit, key);
    return cached;
  }
  // The one trace line written under the build lock. A build dwarfs an fprintf, and recording the
  // miss before the lock would count a raced key as both a miss and a hit.
  graph_cache_debug::record_hit_miss(backend, pass, LookupResult::Miss, key);

  auto entry = std::make_shared<CacheEntry<GraphAndTensors>>(build());
  graph_cache_debug::record_create_graph(backend, pass);
  // Every site's tensor tuple leads with its graph, the one element this needs; a tuple ordered
  // otherwise fails to compile rather than quietly validating the wrong object.
  //
  // The two counters bracket this call deliberately: a graph cuDNN refuses throws here, having
  // recorded its CREATE_GRAPH and never reaching CACHE_GRAPH, so the gap between those two columns
  // is cuDNN's refusals alone.
  query_support(backend, pass, *std::get<0>(entry->graph_and_tensors), handle);
  graph_cache_debug::record_cache_graph(backend, pass);
  // The insert always takes: a thread racing this key would have had to hold the build lock to do
  // it, and the look above already ruled that out.
  std::lock_guard<std::mutex> lock(cache.mutex);
  return cache.entries.insert({key, std::move(entry)}).first->second;
}

// A backend's graph cache for one pass, and the only route to it. The cache is this instantiation's
// static local, so the callers naming one <backend, pass, creator> triple share one cache and each
// triple gets its own.
//
// `kCreateGraphFn` is a template parameter rather than a `CreateFn &&` argument on purpose: as a
// parameter it makes the creator part of the instantiation. Passed as an argument, each distinct
// lambda type would instantiate its own static cache, and the two call sites for a pass would
// quietly stop sharing entries.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
auto get_graph(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  static GraphCache<decltype(kCreateGraphFn(cfg))> cache;
  // Asserted once here for both the key and the graph, which read the same derived fields.
  cfg.check_derived();
  return cache_graph(cache, cfg.make_cache_key(kPass), kBackend, kPass, handle,
                     [&] { return kCreateGraphFn(cfg); });
}

// Whether cuDNN can run the graph this config asks for, in one direction: the empty string if it
// can, otherwise cuDNN's own account of why not. This is the whole of what support_verdict_f16 and
// support_verdict_fp8 do; they exist only to reach their own translation unit's graph builders,
// which is also where a runtime direction becomes the compile-time one this needs.
//
// The question is answered by building the graph, so there is no separate list of rules to keep in
// step with the builder, and the graph lands in the cache the execution path reads.
//
// Refusals and failures read alike, because CUDNN_BACKEND_API_FAILED -- raised for any non-success
// cudnnStatus_t -- cannot separate CUDNN_STATUS_NOT_SUPPORTED from CUDNN_STATUS_ALLOC_FAILED.
// Either way this backend cannot serve this call and the caller wants the message.
//
// The direction is named by the caller rather than read off the config, which has both
// check_for_*_support set and so cannot say which graph is being probed.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
std::string support_verdict(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  // Support is signalled by returning the empty string, so a refusal that arrived without a message
  // of its own needs a label rather than reading as an endorsement.
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

// Runs graph.build_plans(), the plan build cache_graph() left undone, once per entry. Call only
// when the graph is about to be executed: a support query builds entries nothing ever runs, and
// kernel compilation is the most expensive frontend call. See CacheEntry.
//
// The once_flag settles which thread runs the build, not whether it may run alongside another one:
// graph.build_plans() is a frontend call like the rest, so it also needs frontend_build_mutex().
// The lock is taken inside the call_once rather than around it, so an entry whose plans are already
// built stays on the flag's atomic fast path and never touches the process-wide lock.
//
// Splitting the build in two means the thread that finishes it is often not the one that started it
// -- a sizing call caches the graph, an autograd thread is first to run it. What makes that safe:
// graph.build_plans() needs no handle (the overload accepting one ignores it, working from the
// operation graph descriptor and the device properties, which is where the >= 1.25.0 frontend the
// build requires is load-bearing); the handle that built that descriptor outlives the build because
// TE never destroys cuDNN handles (cudnnExecutionPlanManager leaves HandleManager's Destroy
// parameter null, so handles leak by design); the descriptor was finalized for one device, which is
// why the key carries device_id; and graph.execute() uses the running thread's own handle, so a
// handle is never used by two threads at once, which is what cuDNN asks in return for letting them
// share a plan.
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
