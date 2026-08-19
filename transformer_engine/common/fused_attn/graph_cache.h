/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// The fused-attention graph cache: what a cache entry is, how one is looked up
// or built, and the frontend calls that make a constructed graph usable.
//
// Each of the four build sites (f16 and fp8, forward and backward) differs only
// in how it constructs its graph and which tensors it hands back. Everything
// after that -- the lookup, the locking, the once-per-entry plan build, the
// support check, and the remembering of what cuDNN refused -- is the same at all
// four, and lives here so it has one definition rather than four copies to keep
// in step.
//
// The five frontend calls a graph goes through, and which caller pays for each:
//
//   on a miss, either caller:
//     validate() -> build_operation_graph() -> create_execution_plans(HeurMode_t::A)
//       -> check_support()                            validate_and_check_support()
//   the execution path only:
//     build_plans()   ensure_plans_built(), once per entry, the kernel compilation
//     execute()       every call, with its variant pack built in a local
//
// This header is deliberately not part of utils.h: it needs the cuDNN frontend,
// and utils.h is included by translation units (utils.cu) that otherwise do not.
// ============================================================================

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_

#include <cstddef>
#include <cstdint>
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

// cuDNN's refusal to run a graph, as opposed to a failure to try. The distinction is what makes
// the negative cache in build_or_get_cached_graph() safe: a refusal is a verdict on the
// configuration and reproducible for a given key, so it can be remembered and replayed, whereas
// a failure that came from the machine's state at that moment (an allocation that did not fit, a
// CUDA error left behind by unrelated work) could well succeed on the next attempt and must not
// be turned into a permanent answer. Only a frontend call that returned one of the codes
// is_unsupported_verdict() names raises this; every other failure keeps its ordinary type and is
// re-attempted the next time the key comes around.
struct UnsupportedGraph : public std::runtime_error {
  explicit UnsupportedGraph(const std::string &reason) : std::runtime_error(reason) {}
};

// Whether a frontend error code is a verdict on the graph rather than a report of something
// that went wrong on the way to reaching one.
//
// cudnn-frontend distinguishes the two, and the negative cache is only sound for the first.
// Three codes are verdicts:
//
//   GRAPH_NOT_SUPPORTED is what the frontend's own support surface returns, from validate().
//     Nearly every rule it checks by hand -- the architecture gates, the head-dim limits, the
//     version-specific workarounds -- reports itself this way.
//   GRAPH_EXECUTION_PLAN_CREATION_FAILED is what check_support() returns when no engine config
//     cuDNN's heuristics offered can run the graph. This is the verdict for everything the
//     frontend does not rule on itself and defers to the backend, so leaving it out would
//     exclude most of what a support probe actually discovers.
//   UNSUPPORTED_GRAPH_FORMAT is a verdict by name and costs nothing to accept, though no
//     frontend release we build against returns it.
//
// All three are properties of the key and will be just as true the next time it is asked. Every
// other code -- CUDNN_BACKEND_API_FAILED, CUDA_API_FAILED, HEURISTIC_QUERY_FAILED, HANDLE_ERROR,
// INVALID_CUDA_DEVICE and the rest -- describes the process at that moment: an OOM under memory
// pressure, a sticky CUDA error left by unrelated work, a handle on the wrong device.
// CUDNN_BACKEND_API_FAILED is the one to be careful about, since the frontend raises it for any
// non-success cudnnStatus_t and so cannot tell CUDNN_STATUS_ALLOC_FAILED from
// CUDNN_STATUS_NOT_SUPPORTED; caching it would let a moment of memory pressure blacklist a
// configuration that is genuinely supported, for the life of the process, and the wider the
// cache's reach the worse that gets. So those are raised as ordinary errors, which leave nothing
// behind and are retried when the key next comes around. Either way cuDNN's own message reaches
// the caller; only whether it is remembered differs.
inline bool is_unsupported_verdict(cudnn_frontend::error_code_t code) {
  return code == cudnn_frontend::error_code_t::GRAPH_NOT_SUPPORTED ||
         code == cudnn_frontend::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED ||
         code == cudnn_frontend::error_code_t::UNSUPPORTED_GRAPH_FORMAT;
}

// The reason string an is_supported_* helper reports for `e`: its message, or `fallback` if it
// has none. Those helpers signal support by returning the empty string, so a refusal that
// arrives without an explanation would be read as an endorsement and the caller would go on to
// run a graph cuDNN has just declined. Nothing raised through NVTE_ERROR can be empty, since it
// prefixes file and line, but that is a property of our macros rather than of every exception
// that can reach a catch clause, and it is not what the contract should rest on.
inline std::string refusal_reason(const std::exception &e, const char *fallback) {
  const char *what = e.what();
  return (what != nullptr && what[0] != '\0') ? std::string(what) : std::string(fallback);
}

// A graph in the cache, plus the tensor attributes needed to bind runtime pointers to it.
//
// Entries are built only as far as check_support(), which is all it takes to decide whether
// a configuration is supported. build_plans() is the kernel-compilation step and the most
// expensive of the five frontend calls, so a support query stops short of it: the query never
// executes the graph, and many of the keys it builds are never executed by anything. The
// execution path finishes the build instead, the first time the graph is needed to run.
//
// plans_built guards that completion. It has to happen exactly once per entry, because the
// cached graph is shared across threads and build_plans() mutates it in place -- two threads
// reaching the same unfinished entry must not both build it. Keeping the flag inside the entry
// keeps it from drifting away from the graph it describes, and leaves unrelated keys free to
// build concurrently. A build that throws leaves the flag unset, so a later call retries
// rather than executing a graph with no plans.
template <typename GraphAndTensors>
struct CachedGraph {
  explicit CachedGraph(GraphAndTensors tensors) : tensors(std::move(tensors)) {}

  GraphAndTensors tensors;
  std::once_flag plans_built;
};

// One build site's cache. Process-wide rather than per-thread so that a graph is reused
// across threads instead of rebuilt by each: cuDNN >= 9.0 allows concurrent execution of a
// shared plan, and the frontend's execute() builds its variant pack in a local rather than in
// the graph, so it does not write to the shared object. No particular frontend version is
// relied on for that -- it has held for far longer than the >= 1.25.0 the build requirements
// ask for, which is there for unrelated features.
//
// What lets one cache serve every thread is an asymmetry between the two objects a call needs. A
// cuDNN handle is per-thread mutable session state: it carries the stream that execute() launches
// on, so each thread holds its own rather than racing to set that on a shared one. A graph and
// its plans are the opposite -- compiled artifacts, built for the properties of a device and
// bound to the device they were finalized against, with nothing in them belonging to the thread
// that did the building. So the cache can be keyed by device and shared by all threads, which is
// why make_cache_key() stamps device_id and nothing thread-shaped. ensure_plans_built() covers
// what that costs at the seam, where the thread that finishes a build is often not the thread
// that started it.
//
// Refusals are cached alongside the graphs, under the same keys and the same lock. A support
// query for an unsupported configuration is otherwise the most expensive thing this cache sees:
// it builds the whole graph, spends the four frontend calls, and throws the result away, and it
// does so again on every query, because a rejection left nothing behind to find. `unsupported`
// is what it leaves behind -- cuDNN's own account of the refusal, which is the entire useful
// output of a failed query, so nothing is lost by answering from it. Reasons are short strings
// and there is one per refused key, so this grows far slower than the graphs beside it.
// Holding the lock and the maps together is also what fixes their relative lifetimes. Members
// are destroyed in reverse declaration order, so the mutex is declared first to be destroyed
// last: the maps go while their guard is still valid, rather than the other way round. Declaring
// a cache and its lock as two separate objects leaves that ordering to whoever writes the next
// one; declaring them here settles it once.
//
// Both maps are bounded; see kCacheCapacity. `last_used` is what makes the bound an LRU rather
// than an arbitrary cull: it is stamped from `clock` on every insertion and every hit, so the
// entry with the smallest value is the one that has gone longest without being asked for. The
// clock is an ordinary member rather than an atomic because it is only ever touched under
// `mutex`, alongside the maps it orders.
template <typename GraphAndTensors>
struct GraphCache {
  struct Slot {
    std::shared_ptr<CachedGraph<GraphAndTensors>> entry;
    uint64_t last_used;
  };
  struct Refusal {
    std::string reason;
    uint64_t last_used;
  };

  std::mutex mutex;  // guards everything below
  uint64_t clock = 0;
  std::map<FusedAttnConfig, Slot> supported;
  std::map<FusedAttnConfig, Refusal> unsupported;
};

// The ceiling on entries in one of the maps of one build site's cache.
//
// Sized to be out of the way of real work rather than to be tight. A training step reuses a
// handful of configurations and an inference server with bucketed sequence lengths tens of them,
// so a hundred is already more shape diversity than a model exhibits. What the ceiling is for is
// the case where the key space is effectively unbounded -- a test suite sweeping shapes, or a
// serving workload that keys on something that never repeats -- where an unbounded cache is a
// slow leak of cuDNN graphs and their execution plans for the life of the process.
//
// Hard-coded rather than configurable, because nothing has yet needed a different number: the
// workloads that fit under it never notice the ceiling, and the ones that do not are better
// served by rebuilding a graph than by holding thousands. An environment variable can come back
// if a workload turns up that wants to trade the memory for the rebuilds.
constexpr size_t kCacheCapacity = 100;

// Make room in `entries` for one more, by dropping the least recently used until there is.
// Call under the cache's lock.
//
// Evicting a graph does not invalidate one that is in use. build_or_get_cached_graph() hands
// back a shared_ptr, so a thread that is executing an entry holds it alive regardless of what
// the map does; erasing here drops the cache's reference and nothing else. The scan is linear,
// but it runs only when the cache is full, and comparing a hundred integers is nothing beside
// the graph build it is making room for.
template <typename Map>
void evict_to_fit(Map &entries) {
  while (entries.size() >= kCacheCapacity) {
    auto oldest = entries.begin();
    for (auto it = entries.begin(); it != entries.end(); ++it) {
      if (it->second.last_used < oldest->second.last_used) oldest = it;
    }
    entries.erase(oldest);
  }
}

// Takes a constructed graph through the frontend calls that decide whether cuDNN can run it:
// validate, build_operation_graph, create_execution_plans, check_support. The sequence is
// identical for both passes and both backends, so it is defined once here; `backend` and `pass`
// only name the build site whose stage timers the calls are attributed to.
//
// Support is reported by throwing rather than by a return value. NVTE_CHECK_CUDNN_FE raises
// an exception carrying cuDNN's own explanation of the rejection, and that text is what the
// is_supported_* helpers return as the reason a backend was refused -- so a bool here would
// discard the one thing a support probe exists to produce. Callers that are about to execute
// the graph want the throw as well, since there is nothing useful to do with an unsupported
// graph but fail.
//
// A failure is raised as UnsupportedGraph only when the frontend's own error code says the graph
// was adjudicated and refused; see is_unsupported_verdict(). Anything else these calls can report
// is a failure to reach a verdict and is raised through NVTE_ERROR, so it is not remembered.
// Classifying on the code rather than on which call failed is what keeps that honest: all four of
// these calls can fail for environmental reasons too -- build_operation_graph() and
// create_execution_plans() both talk to the cuDNN backend -- so their position in the sequence
// says nothing about whether the failure was about the configuration.
//
// build_plans() and execute() sit outside this function entirely: they commit real resources, and
// build_plans() belongs to whoever executes the graph, once, the first time it is needed. See
// CachedGraph.
inline void validate_and_check_support(graph_cache_debug::Backend backend,
                                       graph_cache_debug::Pass pass,
                                       cudnn_frontend::graph::Graph &graph, cudnnHandle_t handle) {
  auto run = [&](graph_cache_debug::BuildStage stage, const char *call_name, auto &&call) {
    cudnn_frontend::error_t error;
    graph_cache_debug::timer(backend, pass, stage, [&] { error = call(); });
    if (error.is_good()) return;
    // cuDNN normally explains itself; fall back to the call's name so that a refusal can never
    // arrive as an empty string, which the is_supported_* helpers would read as an endorsement.
    const std::string reason =
        error.err_msg.empty() ? std::string(call_name) + " failed." : error.err_msg;
    if (is_unsupported_verdict(error.code)) throw UnsupportedGraph(reason);
    NVTE_ERROR("cuDNN Error in ", call_name, ": ", reason,
               " For more information, enable cuDNN error logging by setting CUDNN_LOGERR_DBG=1 "
               "and CUDNN_LOGDEST_DBG=stderr in the environment.");
  };

  run(graph_cache_debug::BuildStage::Validate, "validate", [&] { return graph.validate(); });
  run(graph_cache_debug::BuildStage::BuildOpGraph, "build_operation_graph",
      [&] { return graph.build_operation_graph(handle); });
  run(graph_cache_debug::BuildStage::CreatePlans, "create_execution_plans",
      [&] { return graph.create_execution_plans({cudnn_frontend::HeurMode_t::A}); });
  run(graph_cache_debug::BuildStage::CheckSupport, "check_support",
      [&] { return graph.check_support(); });
}

// The cached entry for `key`, building and inserting it via `build` if absent. Throws
// UnsupportedGraph if cuDNN refuses the graph -- this time or on an earlier call, the two being
// indistinguishable to the caller by design.
//
// `build` only constructs a graph; this is what puts it through validate_and_check_support(), so
// the entries in the cache are exactly the graphs cuDNN has agreed to run. Those four calls sit
// on the miss path because they are part of building an entry rather than reading one: repeating
// them on a hit would redo the operation graph and the plan search for a graph that has already
// been through both.
//
// `key` must be a normalized key -- FusedAttnConfig::make_cache_key()'s output -- and not a
// raw execution config. Two configs that differ only in a field no graph reads (attn_scale,
// say) have to reach the same entry, which is what normalization is for; passing the raw
// config instead silently multiplies the cache by fields the graph never consumes.
//
// Only the map operations are locked, not `build`. A graph build is the expensive part and
// holding the lock across it would serialize builds of unrelated keys, so two threads racing
// on the same key may both build. That is a wasted build, not a correctness problem: the
// loser drops its own graph and takes the winner's, so every caller of a given key gets one
// shared entry and the once-flag inside it still governs the plan build. Both threads record
// their own lookup, so the wasted build shows up in diagnostics as two MISS lines carrying the
// same key and a build_graph count above the number of distinct keys, rather than as anything
// missing. The same race on a refused key is equally harmless, both threads storing the same
// reason.
//
//   lock cache.mutex
//     supported[key]?    found -> last_used = ++clock, copy the shared_ptr
//     unsupported[key]?  found -> last_used = ++clock, copy the reason
//   unlock
//   record_cache_lookup(HIT | UNSUPPORTED | MISS)
//
//   HIT          -> return the entry
//   UNSUPPORTED  -> throw UnsupportedGraph(the remembered reason)
//   MISS         -> build()                          outside the lock, so builds of unrelated
//                   validate_and_check_support()     keys proceed concurrently
//                     ok      -> lock, evict_to_fit(supported), insert stamped ++clock, unlock,
//                                return the inserted entry, which on a lost race is the winner's
//                     verdict -> lock, evict_to_fit(unsupported), insert the reason, unlock,
//                                rethrow
//                     other   -> NVTE_ERROR: nothing remembered, retried when the key returns
template <typename GraphAndTensors, typename BuildFn>
std::shared_ptr<CachedGraph<GraphAndTensors>> build_or_get_cached_graph(
    GraphCache<GraphAndTensors> &cache, const FusedAttnConfig &key,
    graph_cache_debug::Backend backend, graph_cache_debug::Pass pass, cudnnHandle_t handle,
    BuildFn &&build) {
  using Entry = CachedGraph<GraphAndTensors>;
  using Slot = typename GraphCache<GraphAndTensors>::Slot;
  using Refusal = typename GraphCache<GraphAndTensors>::Refusal;

  std::shared_ptr<Entry> cached;
  bool refused = false;
  std::string reason;
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    auto it = cache.supported.find(key);
    if (it != cache.supported.end()) {
      it->second.last_used = ++cache.clock;
      cached = it->second.entry;
    } else {
      auto refusal = cache.unsupported.find(key);
      refused = (refusal != cache.unsupported.end());
      if (refused) {
        refusal->second.last_used = ++cache.clock;
        reason = refusal->second.reason;
      }
    }
  }
  using graph_cache_debug::LookupResult;
  LookupResult outcome = LookupResult::Miss;
  if (cached != nullptr) {
    outcome = LookupResult::Hit;
  } else if (refused) {
    outcome = LookupResult::Unsupported;
  }
  // Recorded after the lock is dropped, so that writing a trace line cannot hold up threads
  // querying other keys. The counters are exact, but two lookups that raced on the lock can be
  // recorded in the opposite order, so read a level-2 trace as the set of lookups that happened
  // rather than as the sequence they happened in.
  graph_cache_debug::record_cache_lookup(backend, pass, outcome, key);

  if (cached != nullptr) return cached;
  // Raised rather than returned so that a replayed refusal is the same event as a fresh one:
  // every caller already has to handle the build refusing, and none of them would have anything
  // else to do with a second, quieter way of saying so.
  if (refused) throw UnsupportedGraph(reason);

  std::shared_ptr<Entry> entry;
  try {
    entry = std::make_shared<Entry>(build());
    // Every site's tensor tuple leads with its graph, which is the one thing all four have in
    // common and the only element this needs. A tuple that stopped leading with it would fail to
    // compile here rather than quietly validate the wrong object.
    validate_and_check_support(backend, pass, *std::get<0>(entry->tensors), handle);
  } catch (const UnsupportedGraph &e) {
    {
      std::lock_guard<std::mutex> lock(cache.mutex);
      evict_to_fit(cache.unsupported);
      cache.unsupported.insert({key, Refusal{e.what(), ++cache.clock}});
    }
    graph_cache_debug::record_unsupported(backend, pass);
    throw;
  }
  graph_cache_debug::record_graph_built(backend, pass);
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    evict_to_fit(cache.supported);
    // On a losing race the insert does nothing: the temporary Slot is destroyed with the graph
    // this thread built, and what comes back is the winner's entry.
    auto inserted = cache.supported.insert({key, Slot{std::move(entry), ++cache.clock}});
    return inserted.first->second.entry;
  }
}

// Runs the plan build that build_or_get_cached_graph() left undone, once per entry.
//
// Call this only when the graph is about to be executed, which is why it is a separate step
// rather than the tail of the lookup: a support query builds entries that nothing ever runs, and
// kernel compilation is the most expensive of the five frontend calls, so a query that paid for
// it would be paying for nothing. See CachedGraph for why the flag lives inside the entry and
// what a throw here leaves behind.
//
// Splitting the build in two means the thread that finishes it is often not the thread that
// started it -- a sizing call on one thread caches the graph, and an autograd thread is the first
// to need it to run. Four facts make that safe, and only the first is visible here.
//
// build_plans() takes no handle. The overload that accepts one ignores it -- its body is
// `(void)handle;` -- and the build works from the operation graph descriptor and the device
// properties instead, which is how deviceless ahead-of-time compilation builds plans with no
// handle at all. Unlike the plan sharing described on GraphCache, this does lean on the >= 1.25.0
// frontend the build requires: it is where the handle-free overload arrived. Calling it means a
// plan build cannot reach for the handle of a thread that has since exited.
//
// The handle from the build does outlive the build, held by the operation graph descriptor that
// build_operation_graph(handle) finalized against it. It stays a valid object only because TE
// never destroys cuDNN handles: cudnnExecutionPlanManager leaves HandleManager's Destroy
// parameter at its nullptr default, so handles leak by design, one per thread per device.
//
// That descriptor was finalized for the device of the handle that built it, which is why the
// cache key carries device_id (see FusedAttnConfig::make_cache_key). Without it a thread could
// build plans, and compile kernels, from a descriptor belonging to another device.
//
// Execution stays clear of all of it: execute() is called with the running thread's own handle,
// so a handle is never used by two threads at once, which is what cuDNN asks in return for
// letting them share the plan.
template <typename GraphAndTensors>
void ensure_plans_built(graph_cache_debug::Backend backend, graph_cache_debug::Pass pass,
                        CachedGraph<GraphAndTensors> &entry) {
  std::call_once(entry.plans_built, [&] {
    cudnn_frontend::graph::Graph &graph = *std::get<0>(entry.tensors);
    graph_cache_debug::timer(backend, pass, graph_cache_debug::BuildStage::BuildPlans,
                             [&] { NVTE_CHECK_CUDNN_FE(graph.build_plans()); });
    graph_cache_debug::record_plans_built(backend, pass);
  });
}

}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_H_
