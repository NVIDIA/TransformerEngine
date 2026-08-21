/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Fused-attention graph cache diagnostics.
//
// Enable with NVTE_FUSED_ATTN_CACHE_DEBUG=<level>[:<ranks>]. The output format, how to read it and
// the rank suffix are documented for users in docs/envvars.rst; what follows is what maintaining
// this file needs.
//
//   level 1 (events) : one line per event that happens once per distinct cache key (CREATE_GRAPH,
//                      CACHE_GRAPH, BUILD_PLANS), plus the exit summary and its stage timings.
//   level 2 (trace)  : adds a line per lookup (HIT/MISS, with the normalized key) and per execution
//                      (EXECUTE). High volume, and it serializes threads on the stderr lock, which
//                      the stage timings are then measured under -- no timed region writes to
//                      stderr, so they stay sound, but they read a little high.
//
// Counters are kept per build site -- f16/fp8 crossed with fwd/bwd -- since one process can drive
// both backends, and every event name is also the counter column it increments. Where the events
// sit on the path nvte_fused_attn_fwd_v2 sketches: HIT/MISS on every get_graph() lookup,
// CREATE_GRAPH and CACHE_GRAPH inside it on a miss, BUILD_PLANS on an entry's first execution, and
// EXECUTE on every call.
//
// One level-1 training step, line prefixes and trailing columns elided:
//
//   tid=0   dev=0   | f16 fwd CREATE_GRAPH | hit=0, miss=1, create_graph=1, cache_graph=0, ...
//   tid=0   dev=0   | f16 fwd CACHE_GRAPH  | hit=0, miss=1, create_graph=1, cache_graph=1, ...
//   ===== summary begin =====
//   tid=0   dev=0   | f16 fwd | hit=5, miss=1, create_graph=1, cache_graph=1, ...
//   tid=1   dev=0   | f16 bwd | hit=4, build_plans=1, execute=1, ...
//   tid=all dev=all | f16 fwd | hit=5, miss=1, create_graph=1, cache_graph=1, ...
//   f16 fwd build_plans            | calls=1 | time=  262.104 ms/call
//   ===== summary end =====
//
// Those first two lines are one graph before and after cuDNN was asked to support it, which is why
// a CREATE_GRAPH with no CACHE_GRAPH after it is a refusal. Rows for a site a thread never reached
// are left out rather than zeroed, hence tid=1 having a backward row and no forward one: a PyTorch
// step runs the forward and the backward's probe on the main thread and the backward itself on the
// autograd thread, which finds the graph that probe left behind. That split is why the build
// identities hold on the totals rows and not on any single thread's.

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../util/cuda_runtime.h"
#include "config_and_params.h"

namespace transformer_engine {
namespace fused_attn {
namespace graph_cache_debug {

// ============================================================================
// Vocabulary: which build site an event came from, and which build stage or lookup outcome it
// reports. These four names and the recorders at the bottom are the whole interface. Backend and
// Pass are fused_attn's own, so a recorder and the key it prints share one notion of a site, and a
// mistake at a call site is a compile error rather than a mistyped "fwd".
// ============================================================================

inline constexpr const char *backend_name(Backend b) { return b == Backend::F16 ? "f16" : "fp8"; }
inline constexpr const char *pass_name(Pass p) { return p == Pass::Fwd ? "fwd" : "bwd"; }

enum class BuildStage { Validate, BuildOpGraph, CreatePlans, CheckSupport, BuildPlans, kCount };
enum class LookupResult { Miss, Hit };

namespace detail {

// Verbosity parsed once from NVTE_FUSED_ATTN_CACHE_DEBUG: 0=off, 1=events, 2=trace.
inline int debug_level() {
  static const int lvl = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    if (e == nullptr || e[0] == '\0' || e[0] == '0') return 0;
    const int v = std::atoi(e);  // stops at the optional ":<ranks>" suffix
    return v > 0 ? v : 1;        // any non-empty, non-"0" value enables at least level 1
  }();
  return lvl;
}

// Rank of this process as reported by the launcher, or -1 when there is none. First variable that
// is set wins.
inline int launcher_rank() {
  static const int rank = []() -> int {
    for (const char *var : {"RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"}) {
      const char *v = std::getenv(var);
      if (v != nullptr && v[0] != '\0') return std::atoi(v);
    }
    return -1;
  }();
  return rank;
}

// On at level >= 1, and only for the ranks the ":<ranks>" suffix selects. Rank 0 only by default.
inline bool enabled() {
  static const bool on = [] {
    if (debug_level() < 1) return false;
    const int rank = launcher_rank();
    if (rank < 0) return true;
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    const char *sep = (e != nullptr) ? std::strchr(e, ':') : nullptr;
    if (sep == nullptr) return rank == 0;
    const std::string list(sep + 1);
    if (list == "all") return true;
    for (size_t pos = 0; pos <= list.size();) {
      const size_t comma = list.find(',', pos);
      const std::string tok =
          list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!tok.empty() && std::atoi(tok.c_str()) == rank) return true;
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    return false;
  }();
  return on;
}

// The gate on the trace printouts. Tests enabled() rather than just the level.
inline bool enabled_with_trace() { return enabled() && debug_level() >= 2; }

// Names the emitting rank.
inline const std::string &rank_tag() {
  static const std::string *tag = [] {
    const int rank = launcher_rank();
    if (rank < 0) return new std::string();
    return new std::string("rank=" + std::to_string(rank) + " | ");
  }();
  return *tag;
}

// Short thread IDs (0, 1, 2, ...) in assignment order. tid=0 is whichever thread touched this cache first.
inline unsigned thread_seq_id() {
  static std::atomic<unsigned> next{0};
  static thread_local unsigned id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

// Registered at first use. On process exit, prints event counters and build timings. Only used by the summary handler.
inline void register_summary_once();

// Backend major, pass minor, so that the two passes of one backend are adjacent.
constexpr size_t kSiteCount = 4;
inline constexpr size_t site_index(Backend b, Pass p) {
  return (b == Backend::F16 ? 0u : 2u) + (p == Pass::Fwd ? 0u : 1u);
}

// Cache event counters, one block per build site. Each name is both the event tag on the line that
// records it and the column carrying its running total:
//   - create_graph: a graph constructed for a miss, counted before cuDNN is asked about it.
//   - cache_graph: one of those graphs cleared check_support(), so this is the graphs cuDNN agreed
//     to run. Counted on that verdict rather than on the insert that follows, so it says what cuDNN
//     accepted and not how many entries the map holds.
//   - build_plans: a cached graph finished with graph.build_plans(), the kernel compilation
//     cache_graph deferred, paid by that graph's first execution rather than by the probe.
//   - execute: an execution cuDNN accepted, counted once the enqueue returns. Not a completed
//     execution: the work is asynchronous, so a device-side fault is not reflected here.
//   - hit: a lookup answered from the cache. Need not lead to an execution -- it can be a backend
//     availability check, or a workspace-sizing call, which has no tensors to run with.
//   - miss: a lookup the cache did not answer; triggers a build.

struct EventCounters {
  std::atomic<uint64_t> create_graph{0};
  std::atomic<uint64_t> cache_graph{0};
  std::atomic<uint64_t> build_plans{0};
  std::atomic<uint64_t> execute{0};
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
};

inline EventCounters &counters(Backend b, Pass p) {
  static std::array<EventCounters, kSiteCount> table{};
  return table[site_index(b, p)];
}

// One counter block read out into plain values, so the summary can sum blocks for its per-backend and all-backends rows.
struct CounterSnapshot {
  uint64_t create_graph = 0;
  uint64_t cache_graph = 0;
  uint64_t build_plans = 0;
  uint64_t execute = 0;
  uint64_t hit = 0;
  uint64_t miss = 0;

  CounterSnapshot &operator+=(const CounterSnapshot &other) {
    create_graph += other.create_graph;
    cache_graph += other.cache_graph;
    build_plans += other.build_plans;
    execute += other.execute;
    hit += other.hit;
    miss += other.miss;
    return *this;
  }

  // Whether this block saw nothing at all.
  bool empty() const {
    return (create_graph | cache_graph | build_plans | execute | hit | miss) == 0;
  }
};

inline CounterSnapshot snapshot(const EventCounters &c) {
  CounterSnapshot s;
  s.create_graph = c.create_graph.load(std::memory_order_relaxed);
  s.cache_graph = c.cache_graph.load(std::memory_order_relaxed);
  s.build_plans = c.build_plans.load(std::memory_order_relaxed);
  s.execute = c.execute.load(std::memory_order_relaxed);
  s.hit = c.hit.load(std::memory_order_relaxed);
  s.miss = c.miss.load(std::memory_order_relaxed);
  return s;
}

// Per-thread counters, so the summary can break every column down by thread and backend.
struct ThreadCounters {
  unsigned tid = 0;
  std::atomic<int> device{-1};
  std::array<EventCounters, kSiteCount> sites;
};

// The registry and its mutex are heap-allocated and deliberately never freed. Static destructors
// and atexit handlers run as one sequence in reverse order of construction, and this registry is
// built lazily, so it can be constructed after the summary handler is registered -- and would then
// be destroyed before it runs, leaving the handler to lock a destroyed mutex and walk a destroyed
// vector. Leaking removes the ordering question, at a cost of one mutex and one vector.
inline std::mutex &thread_registry_mutex() {
  static std::mutex *m = new std::mutex();
  return *m;
}
inline std::vector<ThreadCounters *> &thread_registry() {
  static std::vector<ThreadCounters *> *v = new std::vector<ThreadCounters *>();
  return *v;
}

// This thread's block, leaked for a related but distinct reason: a worker thread can exit long
// before the process does, while the registry holds a pointer to its block for the exit summary.
inline ThreadCounters &thread_counters() {
  static thread_local ThreadCounters *tc = [] {
    auto *p = new ThreadCounters();
    p->tid = thread_seq_id();
    p->device.store(cuda::current_device(), std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(thread_registry_mutex());
      thread_registry().push_back(p);
    }
    return p;
  }();
  return *tc;
}

inline EventCounters &thread_counters(Backend b, Pass p) {
  return thread_counters().sites[site_index(b, p)];
}

// The one place diagnostics reach stderr.
inline void write_stderr(const std::string &text) {
  static std::atomic<bool> first_line{true};
  if (first_line.exchange(false, std::memory_order_relaxed)) {
    const std::string first = "\n" + text;
    std::fwrite(first.data(), 1, first.size(), stderr);
  } else {
    std::fwrite(text.data(), 1, text.size(), stderr);
  }
  std::fflush(stderr);
}

// Format one counter block -- one pass of one backend -- as one line.
inline std::string format_counter_line(const char *tid_field, const char *dev_field,
                                       const char *label, const CounterSnapshot &c) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "[FUSED-ATTN-CACHE] %s%-7s %-7s | %s | hit=%4" PRIu64 ", miss=%4" PRIu64
                ", create_graph=%4" PRIu64 ", cache_graph=%4" PRIu64 ", build_plans=%4" PRIu64
                ", execute=%4" PRIu64 "\n",
                rank_tag().c_str(), tid_field, dev_field, label, c.hit, c.miss, c.create_graph,
                c.cache_graph, c.build_plans, c.execute);
  return std::string(buf);
}

// One event line, from the thread the event happened on, carrying the running totals of the build site that raised it.
inline void print_counters(Backend b, Pass p, const char *event) {
  const int device = cuda::current_device();
  thread_counters().device.store(device, std::memory_order_relaxed);
  char label[32];
  char tid_field[16];
  char dev_field[16];
  std::snprintf(label, sizeof(label), "%s %s %-12s", backend_name(b), pass_name(p), event);
  std::snprintf(tid_field, sizeof(tid_field), "tid=%u", thread_seq_id());
  std::snprintf(dev_field, sizeof(dev_field), "dev=%d", device);
  write_stderr(format_counter_line(tid_field, dev_field, label, snapshot(counters(b, p))));
}

// The body every recorder shares: gate, register the exit summary, and add one to `column` in both the process-wide block and this thread's.
inline bool record_counter(Backend b, Pass p, std::atomic<uint64_t> EventCounters::*column) {
  if (!enabled()) return false;
  register_summary_once();
  (counters(b, p).*column).fetch_add(1, std::memory_order_relaxed);
  (thread_counters(b, p).*column).fetch_add(1, std::memory_order_relaxed);
  return true;
}

// The column a lookup lands in.
inline std::atomic<uint64_t> EventCounters::*lookup_column(LookupResult result) {
  switch (result) {
    case LookupResult::Hit:
      return &EventCounters::hit;
    case LookupResult::Miss:
      break;
  }
  return &EventCounters::miss;
}

inline const char *lookup_name(LookupResult result) {
  switch (result) {
    case LookupResult::Hit:
      return "HIT";
    case LookupResult::Miss:
      break;
  }
  return "MISS";
}

inline constexpr const char *kStageNames[] = {
    "validate", "build_operation_graph", "create_execution_plans", "check_support", "build_plans"};

// Totals for one (pass, stage) pair.
struct StageTiming {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> time_ns{0};
};

// Bucketed by build site, fp8 vs f16.
constexpr size_t kStageBuckets = kSiteCount * static_cast<size_t>(BuildStage::kCount);
inline StageTiming &stage_timing(Backend b, Pass p, BuildStage s) {
  static std::array<StageTiming, kStageBuckets> table{};
  const size_t idx =
      site_index(b, p) * static_cast<size_t>(BuildStage::kCount) + static_cast<size_t>(s);
  return table[idx];
}

// Times one stage: clock read in the constructor, accumulated in the destructor.
struct ScopedBuildTimer {
  BuildStage stage;
  bool on;
  Backend backend;
  Pass pass;
  std::chrono::steady_clock::time_point start;
  ScopedBuildTimer(Backend b, Pass p, BuildStage s) : stage(s), on(enabled()), backend(b), pass(p) {
    if (!on) return;
    register_summary_once();
    start = std::chrono::steady_clock::now();
  }
  ~ScopedBuildTimer() {
    if (!on) return;
    const uint64_t elapsed_ns =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
    StageTiming &t = stage_timing(backend, pass, stage);
    t.time_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    t.calls.fetch_add(1, std::memory_order_relaxed);
  }
};

inline constexpr Backend kSummaryBackends[] = {Backend::F16, Backend::FP8};

// Names one build site for a summary row.
inline std::string site_label(Backend b, Pass p) {
  return std::string(backend_name(b)) + " " + pass_name(p);
}

// How many backends the run used.
inline size_t active_backend_count() {
  size_t active = 0;
  for (const Backend b : kSummaryBackends) {
    if (!snapshot(counters(b, Pass::Fwd)).empty() || !snapshot(counters(b, Pass::Bwd)).empty()) {
      ++active;
    }
  }
  return active;
}

// Per-thread breakdown, sorted by tid, one row per build site that thread used.
inline void append_thread_rows(std::string &block) {
  std::lock_guard<std::mutex> lock(thread_registry_mutex());
  std::vector<ThreadCounters *> blocks = thread_registry();
  std::sort(blocks.begin(), blocks.end(),
            [](const ThreadCounters *a, const ThreadCounters *b) { return a->tid < b->tid; });
  for (const ThreadCounters *tc : blocks) {
    char tid_field[16];
    char dev_field[16];
    std::snprintf(tid_field, sizeof(tid_field), "tid=%u", tc->tid);
    std::snprintf(dev_field, sizeof(dev_field), "dev=%d",
                  tc->device.load(std::memory_order_relaxed));
    for (const Backend b : kSummaryBackends) {
      for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
        const CounterSnapshot c = snapshot(tc->sites[site_index(b, p)]);
        if (c.empty()) continue;
        block += format_counter_line(tid_field, dev_field, site_label(b, p).c_str(), c);
      }
    }
  }
}

// Totals, printed after the per-thread rows so they read as their sum: one row per build site, then one per pass across the backends when the run used more than one.
inline void append_total_rows(std::string &block) {
  CounterSnapshot all_fwd;
  CounterSnapshot all_bwd;
  for (const Backend b : kSummaryBackends) {
    for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
      const CounterSnapshot c = snapshot(counters(b, p));
      (p == Pass::Fwd ? all_fwd : all_bwd) += c;
      if (c.empty()) continue;
      block += format_counter_line("tid=all", "dev=all", site_label(b, p).c_str(), c);
    }
  }
  if (active_backend_count() <= 1) return;
  for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
    const CounterSnapshot &c = (p == Pass::Fwd ? all_fwd : all_bwd);
    if (c.empty()) continue;
    block +=
        format_counter_line("tid=all", "dev=all", (std::string("all ") + pass_name(p)).c_str(), c);
  }
}

// Mean time per call for each stage of each build site, skipping stages that no build reached.
inline void append_stage_rows(std::string &block) {
  for (const Backend b : kSummaryBackends) {
    for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
      for (int i = 0; i < static_cast<int>(BuildStage::kCount); ++i) {
        const StageTiming &t = stage_timing(b, p, static_cast<BuildStage>(i));
        const uint64_t n = t.calls.load(std::memory_order_relaxed);
        if (n == 0) continue;
        const double total_ms =
            static_cast<double>(t.time_ns.load(std::memory_order_relaxed)) / 1e6;
        char line[288];
        std::snprintf(
            line, sizeof(line),
            "[FUSED-ATTN-CACHE] %s%-3s %-3s %-22s | calls=%" PRIu64 " | time=%9.3f ms/call\n",
            rank_tag().c_str(), backend_name(b), pass_name(p), kStageNames[i], n, total_ms / n);
        block += line;
      }
    }
  }
}

inline void register_summary_once() {
  static const bool registered = [] {
    std::atexit([] {
      if (!enabled()) return;
      // Built in memory and emitted with one write, so that concurrently-exiting processes stay grouped.
      const std::string marker = "[FUSED-ATTN-CACHE] " + rank_tag() + "===== summary ";
      std::string block = marker + "begin =====\n";
      append_thread_rows(block);
      append_total_rows(block);
      append_stage_rows(block);
      block += marker + "end =====\n";
      write_stderr(block);
    });
    return true;
  }();
  (void)registered;
}

}  // namespace detail

// The recorders: everything a call site calls. Each takes the build site it is reporting for, adds
// one to that site's column, and prints a line when the level asks for it.

inline void record_create_graph(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::create_graph)) {
    detail::print_counters(b, p, "CREATE_GRAPH");
  }
}

inline void record_cache_graph(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::cache_graph)) {
    detail::print_counters(b, p, "CACHE_GRAPH");
  }
}

inline void record_build_plans(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::build_plans)) {
    detail::print_counters(b, p, "BUILD_PLANS");
  }
}

inline void record_execute(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::execute) &&
      detail::enabled_with_trace()) {
    detail::print_counters(b, p, "EXECUTE");
  }
}

inline void record_hit_miss(Backend b, Pass p, LookupResult result, const FusedAttnConfig &key) {
  if (!detail::record_counter(b, p, detail::lookup_column(result)) ||
      !detail::enabled_with_trace()) {
    return;
  }
  char prefix[128];
  std::snprintf(prefix, sizeof(prefix),
                "[FUSED-ATTN-CACHE] %stid=%-3u dev=%-3d | %-3s %-3s %-12s | ",
                detail::rank_tag().c_str(), detail::thread_seq_id(), key.device_id, backend_name(b),
                pass_name(p), detail::lookup_name(result));
  detail::write_stderr(prefix + key.to_string() + "\n");
}

template <typename Fn>
inline decltype(auto) record_time(Backend b, Pass p, BuildStage stage, Fn &&fn) {
  detail::ScopedBuildTimer scoped(b, p, stage);
  return fn();
}

}  // namespace graph_cache_debug
}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
