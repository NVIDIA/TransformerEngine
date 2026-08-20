/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// The fused-attention graph cache: what a cache entry is, how one is looked up
// or built, and the frontend calls that make a constructed graph usable.
//
// The four build sites (f16 and fp8, forward and backward) differ only in how
// they construct their graph and which tensors they hand back. Everything after
// that -- the cache each one keeps, the lookup, the locking, the support check,
// the once-per-entry plan build -- is shared, and lives here rather than in four
// copies: a site names its backend, pass and graph builder to get_graph().
//
// The five frontend calls a graph goes through, and which of our functions pays for
// each. The frontend's are written graph.*, since that is how they are invoked and
// since two of them share a name with ours:
//
//   on a miss, either caller:
//     graph.validate() -> graph.build_operation_graph()
//       -> graph.create_execution_plans(HeurMode_t::A) -> graph.check_support()
//                                                        all four via query_support()
//   the execution path only:
//     graph.build_plans()   build_plans(), once per entry, the kernel compilation
//     graph.execute()       every call, with its variant pack built in a local
//
// Not part of utils.h: this needs the cuDNN frontend, and utils.h is included by
// translation units (utils.cu) that otherwise do not.
// ============================================================================

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
// Entries are built only as far as check_support(), which is all it takes to decide whether a
// configuration is supported. graph.build_plans() -- the kernel compilation, and the most expensive
// of the five frontend calls -- is left to the execution path, since a support query never runs the
// graph and many of the keys it builds are never run by anything.
//
// build_plans_once guards that completion, which has to happen exactly once per entry: the entry is
// shared across threads and graph.build_plans() mutates it in place. Keeping the flag in the entry
// keeps it with the graph it describes and leaves unrelated keys free to build concurrently. A
// build that throws leaves it unset, so a later call retries rather than executing a graph with no
// plans.
template <typename GraphAndTensors>
struct CacheEntry {
  explicit CacheEntry(GraphAndTensors graph_and_tensors)
      : graph_and_tensors(std::move(graph_and_tensors)) {}

  GraphAndTensors graph_and_tensors;
  std::once_flag build_plans_once;
};

// One build site's cache. Process-wide rather than per-thread so a graph is reused across threads
// instead of rebuilt by each: cuDNN >= 9.0 allows concurrent execution of a shared plan, and the
// frontend's execute() builds its variant pack in a local rather than in the graph, so it does not
// write to the shared object.
//
// What lets one cache serve every thread is an asymmetry between the two objects a call needs. A
// cuDNN handle is per-thread mutable session state: it carries the stream execute() launches on, so
// each thread holds its own. A graph and its plans are the opposite -- compiled artifacts, bound to
// the device they were finalized against, with nothing in them belonging to the building thread. So
// the key stamps device_id and nothing thread-shaped (see make_cache_key), and build_plans() below
// covers the seam where the thread that finishes a build is not the one that started it.
//
// The mutex is declared first so that it is destroyed last -- members go in reverse declaration
// order, so the map is destroyed while its guard is still valid. Declaring the two together settles
// that rather than leaving it to whoever writes the next cache.
//
// The map is unbounded. Only executed graphs hold anything substantial -- an entry that stopped at
// check_support() has no compiled kernels behind it, and none hold a workspace, which the caller
// allocates per call -- and a model reuses a handful of configurations, so any bound worth setting
// would sit far above what real work reaches. A workload that does sweep shapes, such as a suite
// enumerating them, holds every graph for the life of the process; `miss` climbing without settling
// is what that looks like, and is the case for bringing a bound back.
template <typename GraphAndTensors>
struct GraphCache {
  std::mutex mutex;  // guards everything below
  std::map<FusedAttnConfig, std::shared_ptr<CacheEntry<GraphAndTensors>>> entries;
};

// Takes a constructed graph through the frontend calls that decide whether cuDNN can run it:
// validate, build_operation_graph, create_execution_plans, check_support. Identical for both passes
// and both backends, so it is defined once here; `backend` and `pass` only name the build site the
// stage timers attribute the calls to.
//
// Reports by throwing, and the throw carries cuDNN's message alone. That message is what
// support_verdict() returns as the reason a backend was refused, so a bool would discard the one
// thing a support probe exists to produce -- and NVTE_ERROR would wrap it in the file, line and
// advice of an internal failure, which a backend refused for a plain reason is not. One kind of
// throw for every failure; see support_verdict() for why that distinction is not drawn.
//
// graph.build_plans() and graph.execute() sit outside this function: they commit real resources,
// and the plan build belongs to whoever executes the graph, once. See CacheEntry.
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

// The cached entry for `key`, building and inserting it via `build` if absent. Throws if cuDNN
// refuses the graph, and remembers nothing when it does, so the next query for a refused key builds
// it again and is refused again. The frameworks only re-enter the selector when the attention
// configuration changes (in PyTorch, _attention_backends caches the choice), so a settled run pays
// for a refusal once; a suite that enumerates configurations pays each time it comes back around,
// which is the case a map of remembered refusals would serve.
//
// `build` only constructs a graph; this is what puts it through query_support(), so the entries in
// the cache are exactly the graphs cuDNN has agreed to run. Those calls belong to building an entry
// rather than reading one, which is why a hit skips them.
//
// `key` must be make_cache_key(pass)'s output, for the same `pass` given here, and not a raw
// execution config: two configs differing only in a field no graph reads (attn_scale, say) have to
// reach the same entry, and passing the raw config silently multiplies the cache by fields the
// graph never consumes.
//
// Only the map operations are locked, not `build`, so builds of unrelated keys proceed concurrently
// and two threads racing on one key may both build. That is wasted work rather than a correctness
// problem -- the loser drops its graph and takes the winner's entry, whose once_flag still governs
// the plan build -- and it reads in diagnostics as two MISS lines with the same key.
//
//   lock cache.mutex
//     entries[key]?  found -> copy the shared_ptr
//   unlock
//   record_hit_miss(HIT | MISS)
//
//   HIT   -> return the entry
//   MISS  -> build(), record_create_graph()    outside the lock, so builds of
//            query_support()                   unrelated keys proceed concurrently
//              ok    -> record_cache_graph(), then lock, insert, unlock; return the inserted
//                       entry, which on a lost race is the winner's
//              throw -> propagates; nothing is stored, so the key is built again if it comes back
template <typename GraphAndTensors, typename BuildFn>
std::shared_ptr<CacheEntry<GraphAndTensors>> cache_graph(GraphCache<GraphAndTensors> &cache,
                                                         const FusedAttnConfig &key,
                                                         Backend backend, Pass pass,
                                                         cudnnHandle_t handle, BuildFn &&build) {
  using graph_cache_debug::LookupResult;

  std::shared_ptr<CacheEntry<GraphAndTensors>> cached;
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.entries.find(key);
    if (it != cache.entries.end()) cached = it->second;
  }
  // Recorded after the lock is dropped, so writing a trace line cannot hold up threads querying
  // other keys. The counters stay exact, but two lookups that raced can be recorded in the opposite
  // order, so a level-2 trace is the set of lookups that happened, not their sequence.
  graph_cache_debug::record_hit_miss(
      backend, pass, cached != nullptr ? LookupResult::Hit : LookupResult::Miss, key);
  if (cached != nullptr) return cached;

  // No backend refuses a configuration from in here any more -- TE's own FP8 rules moved to
  // nvte_get_fused_attn_backend_v2, which answers them with a reason instead of throwing -- so
  // `build` returning is now the ordinary case and miss == create_graph in a run that behaves.
  // A throw that does get out still propagates with its message and leaves nothing behind, which
  // is what keeps miss - create_graph worth printing: it should read zero.
  auto entry = std::make_shared<CacheEntry<GraphAndTensors>>(build());
  graph_cache_debug::record_create_graph(backend, pass);
  // Every site's tensor tuple leads with its graph, the one element this needs. A tuple ordered
  // otherwise would fail to compile rather than quietly validate the wrong object.
  //
  // The two counters bracket this call deliberately: a graph cuDNN refuses throws here, having
  // already recorded its CREATE_GRAPH and never reaching CACHE_GRAPH, so the gap between those two
  // columns is cuDNN's refusals alone.
  query_support(backend, pass, *std::get<0>(entry->graph_and_tensors), handle);
  // Recorded on cuDNN's verdict rather than on the insert below, so the column counts the graphs
  // cuDNN agreed to run. That is the question worth a counter; how many entries a map ended up
  // holding is not, and tying it to the insert made a lost race -- which discards a supported graph
  // and takes the winner's -- read as a miscount rather than as the duplicate work it is.
  graph_cache_debug::record_cache_graph(backend, pass);
  std::lock_guard<std::mutex> lock(cache.mutex);
  // On a losing race the insert does nothing: the shared_ptr this thread built is dropped with
  // its graph, and what comes back is the winner's entry.
  return cache.entries.insert({key, std::move(entry)}).first->second;
}

// A backend's graph cache for one pass, and the only route to it. Both the execution path and the
// support probe come through here, so a probe leaves behind exactly the entry a later execution
// finds. That is what lets the probe's answer describe the graph that actually runs, rather than a
// separately built lookalike.
//
// The cache is this instantiation's static local, so the callers that name one <backend, pass,
// creator> triple share one cache, and each triple gets its own. Naming the triple is now what
// picks the cache, where before there was a per-backend function per pass to call.
//
// `kCreateGraphFn` is a template parameter rather than a `CreateFn &&` argument on purpose. As a
// parameter it makes the creator part of the instantiation, keeping the cache identified by the
// function that fills it. Passed as an argument, each distinct lambda type would instantiate its
// own copy of this function with its own static cache, and the two call sites for a pass would
// quietly stop sharing entries.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
auto get_graph(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  static GraphCache<decltype(kCreateGraphFn(cfg))> cache;
  // Asserted once here for both the key and the graph, which read the same derived fields.
  check_derived(cfg);
  return cache_graph(cache, cfg.make_cache_key(kPass), kBackend, kPass, handle,
                     [&] { return kCreateGraphFn(cfg); });
}

// Whether cuDNN can run the graph this config asks for, in one direction: the empty string if it
// can, otherwise cuDNN's own account of why not, which the backend selector reports to the caller.
// This is the whole of what support_verdict_f16 and support_verdict_fp8 do; they exist only to
// reach their own translation unit's graph builders, which is also where a runtime direction turns
// into the compile-time one this needs.
//
// Named for what it returns rather than the question it answers: support is the empty string, so
// an is_supported() spelling would read backwards wherever the result is tested.
//
// The question is answered by building the graph, which is where every rejection comes from --
// there is no separate list of rules to keep in step with the builder. The graph goes into the same
// cache the execution path reads, so the work is not thrown away and what was checked is what will
// run. It stops short of graph.build_plans(), the expensive step, which the first execution of the
// graph does instead; see CacheEntry.
//
// A refusal, by contrast, is not cached: nothing is stored for a key cuDNN rejected, so asking the
// same question again pays for the build again. See cache_graph.
//
// Refusals and failures on the way to a verdict read alike, because CUDNN_BACKEND_API_FAILED --
// raised for any non-success cudnnStatus_t -- cannot separate CUDNN_STATUS_NOT_SUPPORTED from
// CUDNN_STATUS_ALLOC_FAILED. Either way this backend cannot serve this call, and either way what
// the caller wants is the message.
//
// The direction is named by the caller rather than read off the config: a config arriving from a
// framework has both check_for_*_support set, so it cannot say which graph is being probed.
template <Backend kBackend, Pass kPass, auto kCreateGraphFn>
std::string support_verdict(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  // Built only where it is used, on the two paths where a refusal arrived without a message of its
  // own. Support is signalled by returning the empty string, so an empty refusal would otherwise
  // read as an endorsement.
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

// Runs graph.build_plans(), the plan build that cache_graph() left undone, once per entry. Named
// for the frontend call it wraps; the once-per-entry part is the whole reason it is a function
// rather than that call.
//
// Call only when the graph is about to be executed, which is why this is a separate step rather
// than the tail of the lookup: a support query builds entries nothing ever runs, and kernel
// compilation is the most expensive of the five frontend calls. See CacheEntry for why the flag
// lives inside the entry and what a throw here leaves behind.
//
// Splitting the build in two means the thread that finishes it is often not the thread that started
// it -- a sizing call caches the graph, and an autograd thread is the first to need it to run. Four
// facts make that safe, and only the first is visible here:
//   - graph.build_plans() takes no handle. The overload that accepts one ignores it (its body is
//     `(void)handle;`), working from the operation graph descriptor and the device properties
//     instead, which is how deviceless ahead-of-time compilation builds plans with no handle at
//     all. Unlike the plan sharing on GraphCache, this does lean on the >= 1.25.0 frontend the
//     build requires: it is where the handle-free overload arrived.
//   - The handle that built the operation graph outlives the build, held by the descriptor
//     graph.build_operation_graph(handle) finalized against it, and stays valid only because TE
//     never destroys cuDNN handles: cudnnExecutionPlanManager leaves HandleManager's Destroy
//     parameter at its nullptr default, so handles leak by design, one per thread per device.
//   - That descriptor was finalized for one device, which is why the cache key carries device_id
//     (see make_cache_key). Without it a thread could compile kernels from another device's
//     descriptor.
//   - graph.execute() is called with the running thread's own handle, so a handle is never used by
//     two threads at once, which is what cuDNN asks in return for letting them share a plan.
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
