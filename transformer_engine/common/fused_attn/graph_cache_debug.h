/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// Fused-attention graph cache diagnostics.
//
// Enable at runtime with NVTE_FUSED_ATTN_CACHE_DEBUG. Two verbosity levels:
//   =1 : low volume. Cache event counters, a BUILD and a PLANS line per build, an
//        UNSUP line per configuration cuDNN refuses, and the end-of-run SUMMARY
//        (aggregate + per-thread) and stage timings. This is enough to diagnose
//        redundant rebuilds and profile build cost.
//   =2 : high volume (trace). Additionally emits a per-lookup HIT/MISS/NOSUP line
//        with the full shorthand cache key and a per-execution EXEC line. Use only
//        when you need to see *which* shapes are hitting/missing -- these fire on
//        every cache lookup and execution, so at suite scale they add I/O and
//        serialize threads on the stderr lock. No timed region writes to stderr, so
//        the stage timings stay sound, but they are measured under more contention
//        than at level 1 and read a little high.
//
// NOSUP is a hit on the negative cache: a key cuDNN has already refused, answered
// from the stored refusal instead of by building the graph again.
//
// An optional ":<ranks>" suffix picks which processes emit, defaulting to rank 0
// so that output does not scale with the world size: "1:all" for every rank,
// "2:0,3" for a specific set. See `rank_selected` for when overriding pays off.
// ============================================================================

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_

#include <sys/syscall.h>
#include <unistd.h>

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

#include "config_and_params.h"

namespace transformer_engine {
namespace fused_attn {
namespace graph_cache_debug {

// Rank of this process as reported by the launcher, or -1 when there is no
// launcher (a single-process run). First variable that is set wins.
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

// Verbosity level parsed once from NVTE_FUSED_ATTN_CACHE_DEBUG (0=off, 1=default,
// 2=trace). Single read at startup, cached. Negligible overhead when unset.
inline int debug_level() {
  static const int lvl = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    if (e == nullptr || e[0] == '\0' || e[0] == '0') return 0;
    const int v = std::atoi(e);  // stops at the optional ":<ranks>" suffix
    return v > 0 ? v : 1;        // any non-empty, non-"0" value enables at least level 1
  }();
  return lvl;
}

// Whether this process emits diagnostics. Every rank writes to the same stderr,
// so emitting from all of them multiplies the volume by the world size -- and
// under data/tensor parallelism the ranks are running identical shapes, so the
// copies say the same thing. Hence rank 0 only by default.
//
// Context parallelism is the case worth overriding for: the ranks run different
// subsets of the per-step regimes (under p2p, rank 0 never sees the lower-triangle
// config that the last rank does), so their build counts genuinely differ.
// Select with the ":<ranks>" suffix, e.g. "1:all" or "2:0,3".
inline bool rank_selected() {
  static const bool selected = [] {
    const int rank = launcher_rank();
    if (rank < 0) return true;  // sole process, nothing to filter
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
  return selected;
}

// Diagnostics are on at level >= 1, and only for the selected ranks. Unselected
// ranks skip the counters too, so they pay nothing beyond this check.
inline bool enabled() { return debug_level() >= 1 && rank_selected(); }

// Per-lookup / per-exec trace lines are gated behind level >= 2.
inline bool trace_enabled() { return debug_level() >= 2; }

// Identifies the emitting process. Distributed PyTorch runs one process per rank
// and they all share this stderr, so without this every line would be ambiguous
// (thread ids restart at 0 in each process). Rank comes from the launcher, if any.
inline const std::string &process_tag() {
  static const std::string *tag = [] {
    auto *s = new std::string("pid=" + std::to_string(static_cast<int64_t>(::getpid())));
    if (launcher_rank() >= 0) *s += " rank=" + std::to_string(launcher_rank());
    return s;
  }();
  return *tag;
}

// More readable, shorter thread IDs (0, 1, 2, ...). These are assignment order,
// not identity: tid=0 is whichever thread touched this cache first. The one-shot
// THREAD line below maps them to OS thread ids for correlating with nsys/gdb.
inline unsigned thread_seq_id() {
  static std::atomic<unsigned> next{0};
  static thread_local unsigned id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

// OS-level thread id, as reported by nsys/gdb/`top -H`. Via syscall rather than
// gettid() so this does not require glibc >= 2.30.
inline int64_t os_thread_id() { return static_cast<int64_t>(::syscall(SYS_gettid)); }

// Registered at first use. On process exit, prints overall event counters and
// graph build timings.
inline void register_summary_once();

// ============================================================================
// Cache event counters (forward/backward):
//   - BUILD: a graph built and cached in response to a cache miss. Built only as far as
//            check_support(), which is all a support probe needs.
//   - PLANS: a cached graph finished with build_plans(), the kernel compilation that the
//            BUILD above deferred. At most one per BUILD, on the first execution of that
//            graph, so BUILD minus PLANS is how many graphs were built for a support probe
//            and never used to run anything.
//   - EXEC: a graph execution call with valid runtime tensors
//   - HIT: a cache lookup that hit; may not trigger an EXEC, and may only be
//          a backend availability check or from the first shape-probing call of
//          nvte_fused_attn_fwd/bwd which has no runtime tensors
//   - MISS: a cache lookup that missed; triggers a graph build
// ============================================================================

struct EventCounters {
  std::atomic<uint64_t> built{0};
  std::atomic<uint64_t> plans{0};
  std::atomic<uint64_t> exec{0};
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
  std::atomic<uint64_t> unsup{0};
};

inline EventCounters &counters(bool is_fwd) {
  static EventCounters fwd;
  static EventCounters bwd;
  return is_fwd ? fwd : bwd;
}

// Per-thread counters, so the summary can break down build/exec/hit/miss by
// thread. In the single-process context-parallel case each device is driven by
// its own thread, so this reveals which thread built/executed what.
struct ThreadCounters {
  unsigned tid = 0;
  EventCounters fwd;
  EventCounters bwd;
};

// The registry and its mutex are heap-allocated and deliberately never freed.
// Function-local static destructors and atexit handlers run as a single sequence,
// in reverse order of construction/registration. This registry is built lazily, so
// it can be constructed *after* the summary handler is registered -- in which case
// it would be destroyed *before* that handler runs, leaving the handler to lock a
// destroyed mutex and walk a destroyed vector. Leaking removes the ordering
// question rather than reasoning about it, and the cost is bounded: one mutex and
// one vector for the process, reclaimed by the OS at exit anyway.
inline std::mutex &thread_registry_mutex() {
  static std::mutex *m = new std::mutex();
  return *m;
}
inline std::vector<ThreadCounters *> &thread_registry() {
  static std::vector<ThreadCounters *> *v = new std::vector<ThreadCounters *>();
  return *v;
}

// This thread's counter block, leaked for a related but distinct reason: a worker
// thread can exit long before the process does, while the registry keeps a pointer
// to its block for the end-of-run summary. Tying the block's lifetime to the
// thread would leave that pointer dangling. One small struct per thread.
inline ThreadCounters &thread_counters() {
  static thread_local ThreadCounters *tc = [] {
    auto *p = new ThreadCounters();
    p->tid = thread_seq_id();
    {
      std::lock_guard<std::mutex> lock(thread_registry_mutex());
      thread_registry().push_back(p);
    }
    // One line per thread, mapping the short id to something nsys/gdb can match.
    std::fprintf(stderr, "[FUSED-ATTN-CACHE] %s | THREAD      | tid=%-3u os_tid=%" PRId64 "\n",
                 process_tag().c_str(), p->tid, os_thread_id());
    std::fflush(stderr);
    return p;
  }();
  return *tc;
}

inline EventCounters &thread_counters(bool is_fwd) {
  ThreadCounters &tc = thread_counters();
  return is_fwd ? tc.fwd : tc.bwd;
}

// Format one counter block (aggregate or a single thread's) as one line.
// `tid_field` is the whole thread column, e.g. "tid=3"; the aggregate row passes
// "tid=all" so that it cannot be misread as thread 0's row.
//
// The columns are meant to be read against two identities. Every lookup lands in exactly one of
// miss and hit, and every miss ends in exactly one of built and unsup -- so `miss = built + unsup`
// and a shortfall in either means a build died of something other than a refusal. `built >= plans`
// always, the difference being graphs that a support query built and nothing has yet run.
inline std::string format_counter_line(const char *event, const char *tid_field,
                                       const EventCounters &f, const EventCounters &b) {
  char buf[768];
  std::snprintf(buf, sizeof(buf),
                "[FUSED-ATTN-CACHE] %s | %-11s | %-7s | fwd miss=%4" PRIu64 ", hit=%4" PRIu64
                ", built=%4" PRIu64 ", unsup=%4" PRIu64 ", plans=%4" PRIu64 ", exec=%4" PRIu64
                " | bwd miss=%4" PRIu64 ", hit=%4" PRIu64 ", built=%4" PRIu64 ", unsup=%4" PRIu64
                ", plans=%4" PRIu64 ", exec=%4" PRIu64 "\n",
                process_tag().c_str(), event, tid_field, f.miss.load(std::memory_order_relaxed),
                f.hit.load(std::memory_order_relaxed), f.built.load(std::memory_order_relaxed),
                f.unsup.load(std::memory_order_relaxed), f.plans.load(std::memory_order_relaxed),
                f.exec.load(std::memory_order_relaxed), b.miss.load(std::memory_order_relaxed),
                b.hit.load(std::memory_order_relaxed), b.built.load(std::memory_order_relaxed),
                b.unsup.load(std::memory_order_relaxed), b.plans.load(std::memory_order_relaxed),
                b.exec.load(std::memory_order_relaxed));
  return std::string(buf);
}

inline void print_counter_block(const char *event, const char *tid_field, const EventCounters &f,
                                const EventCounters &b) {
  const std::string line = format_counter_line(event, tid_field, f, b);
  std::fputs(line.c_str(), stderr);
  std::fflush(stderr);
}

inline void print_counters(const char *event) {
  char tid_field[16];
  std::snprintf(tid_field, sizeof(tid_field), "tid=%u", thread_seq_id());
  print_counter_block(event, tid_field, counters(/*is_fwd=*/true), counters(/*is_fwd=*/false));
}

// A graph built through check_support() and cached. Call after the build, from the miss
// path that performed it.
inline void record_build(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).built.fetch_add(1, std::memory_order_relaxed);
  thread_counters(is_fwd).built.fetch_add(1, std::memory_order_relaxed);
  print_counters(is_fwd ? "fwd BUILD" : "bwd BUILD");
}

// The build_plans() a BUILD deferred, now completed. Call from inside the std::call_once
// that runs it, after the call returns rather than before: build_plans() throws without
// setting the once_flag, leaving a later execution to retry it, so counting on the way out
// keeps this a count of graphs that reached a runnable state. Like BUILD this fires once
// per distinct cache key, so it stays on the level-1 path.
inline void record_plans_built(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).plans.fetch_add(1, std::memory_order_relaxed);
  thread_counters(is_fwd).plans.fetch_add(1, std::memory_order_relaxed);
  print_counters(is_fwd ? "fwd PLANS" : "bwd PLANS");
}

// A build that cuDNN refused, now remembered as a negative cache entry. Call from the miss path
// that attempted it, in place of record_build(): a refusal and a build are the two ways a miss
// can end, and counting both keeps `miss = built + unsup` true. Fires once per distinct refused
// key -- the second query for that key is a hit -- so it stays on the level-1 path.
inline void record_unsupported(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).unsup.fetch_add(1, std::memory_order_relaxed);
  thread_counters(is_fwd).unsup.fetch_add(1, std::memory_order_relaxed);
  print_counters(is_fwd ? "fwd UNSUP" : "bwd UNSUP");
}

inline void record_exec(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).exec.fetch_add(1, std::memory_order_relaxed);
  thread_counters(is_fwd).exec.fetch_add(1, std::memory_order_relaxed);
  // The per-exec line fires on every execution; keep it out of the level-1 path.
  if (!trace_enabled()) return;
  print_counters(is_fwd ? "fwd EXEC" : "bwd EXEC");
}

// What a lookup found. Unsupported is the negative-cache case: a key whose graph cuDNN has
// already refused, so the answer is a remembered refusal rather than a graph.
enum class LookupResult { Miss, Hit, Unsupported };

// `key` is the normalized cache key -- make_cache_key()'s output, the exact value the
// lookup was performed with -- not the execution config it was derived from. That is
// deliberate: HIT/MISS is decided by comparing keys, so a trace of anything else cannot
// explain its own outcome. Logging the pre-normalization config would show pairs of
// identical lines with opposite outcomes (normalization having collapsed a difference,
// e.g. bottom_right_diagonal or the THD token counts) and pairs of differing lines that
// both hit (the difference being in a field the key drops, e.g. attn_scale). Diffing two
// MISS lines here instead names exactly the fields responsible for the extra build.
//
// The cost is that fields normalization overwrites are no longer visible in their
// original form: attn_scale reads 1, ragged num_tokens read 0, and max_seqlen/batch_size
// read their bucketed values. Recover those from the caller if a line needs to be traced
// back to a specific test case.
inline void record_cache_lookup(const char *pass, LookupResult result, const FusedAttnConfig &key) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  // Unsupported counts as a hit: what the hit column measures is lookups that were answered
  // without building anything, and a remembered refusal is one of those. Which kind of answer
  // it was shows in the trace line, and the running total of refusals is the unsup column.
  const bool hit = (result != LookupResult::Miss);
  EventCounters &pc = counters(is_fwd);
  (hit ? pc.hit : pc.miss).fetch_add(1, std::memory_order_relaxed);
  EventCounters &tpc = thread_counters(is_fwd);
  (hit ? tpc.hit : tpc.miss).fetch_add(1, std::memory_order_relaxed);
  // The per-lookup config dump is the highest-volume line (one per cache lookup);
  // keep it out of the level-1 path and off the stderr lock unless tracing.
  if (!trace_enabled()) return;
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %s | %-3s %-5s | tid=%u dev=%d | train=%d det=%d cg=%d "
      "maxlogit=%d fwd=%d "
      "mask=%" PRId64 " bias=%" PRId64 " wl=%" PRId64 " wr=%" PRId64 " brd=%d softmax=%" PRId64
      " scale_mode=%" PRId64 " dropout=%g attn_scale=%g qkv_dt=%" PRId64 " o_dt=%" PRId64
      " do_dt=%" PRId64 " dqkv_dt=%" PRId64 " qkv_lay=%" PRId64 " o_fmt=%" PRId64 " do_fmt=%" PRId64
      " dqkv_lay=%" PRId64 " qkv_sif=%" PRId64 " do_sif=%" PRId64 " b=%" PRId64 " h=%" PRId64
      " hg=%" PRId64 " dqk=%" PRId64 " dv=%" PRId64 " sq=%" PRId64 " skv=%" PRId64 " tq=%" PRId64
      " tkv=%" PRId64 " bb=%" PRId64 " btq=%" PRId64 " btkv=%" PRId64 " npk=%" PRId64
      " npv=%" PRId64 " psk=%" PRId64 " psv=%" PRId64 " mppk=%" PRId64 " mppv=%" PRId64
      " bias_b=%" PRId64 " bias_h=%" PRId64 " bias_sq=%" PRId64 " bias_skv=%" PRId64 "\n",
      process_tag().c_str(), pass,
      result == LookupResult::Miss ? "MISS" : (result == LookupResult::Hit ? "HIT" : "NOSUP"),
      thread_seq_id(), key.device_id, static_cast<int>(key.is_training),
      static_cast<int>(key.deterministic), static_cast<int>(key.cuda_graph),
      static_cast<int>(key.return_max_logit), static_cast<int>(key.check_for_forward_support),
      static_cast<int64_t>(key.attn_mask_type), static_cast<int64_t>(key.bias_type),
      static_cast<int64_t>(key.window_size_left), static_cast<int64_t>(key.window_size_right),
      static_cast<int>(key.bottom_right_diagonal), static_cast<int64_t>(key.softmax_type),
      static_cast<int64_t>(key.scaling_mode), static_cast<double>(key.dropout),
      static_cast<double>(key.attn_scale), static_cast<int64_t>(key.qkv_dtype),
      static_cast<int64_t>(key.o_dtype), static_cast<int64_t>(key.do_dtype),
      static_cast<int64_t>(key.dqkv_dtype), static_cast<int64_t>(key.qkv_layout),
      static_cast<int64_t>(key.o_format), static_cast<int64_t>(key.do_format),
      static_cast<int64_t>(key.dqkv_layout), static_cast<int64_t>(key.qkv_scale_inv_format),
      static_cast<int64_t>(key.do_scale_inv_format), static_cast<int64_t>(key.batch_size),
      static_cast<int64_t>(key.num_attn_heads), static_cast<int64_t>(key.num_gqa_groups),
      static_cast<int64_t>(key.head_dim_qk), static_cast<int64_t>(key.head_dim_v),
      static_cast<int64_t>(key.max_seqlen_q), static_cast<int64_t>(key.max_seqlen_kv),
      static_cast<int64_t>(key.num_tokens_q), static_cast<int64_t>(key.num_tokens_kv),
      static_cast<int64_t>(key.bucketed_batch_size),
      static_cast<int64_t>(key.bucketed_num_tokens_q),
      static_cast<int64_t>(key.bucketed_num_tokens_kv), static_cast<int64_t>(key.num_pages_k),
      static_cast<int64_t>(key.num_pages_v), static_cast<int64_t>(key.page_size_k),
      static_cast<int64_t>(key.page_size_v), static_cast<int64_t>(key.max_pages_per_seq_k),
      static_cast<int64_t>(key.max_pages_per_seq_v), static_cast<int64_t>(key.bias_batch_size),
      static_cast<int64_t>(key.bias_num_heads), static_cast<int64_t>(key.bias_seqlen_q),
      static_cast<int64_t>(key.bias_seqlen_kv));
}

// ============================================================================
// Graph build timings.
//
// A cuDNN graph build is a fixed sequence of frontend calls, and which one
// dominates determines what to do about a slow build: time in `check_support`
// and `build_plans` is heuristic selection and kernel compilation, largely
// intrinsic to the shape, whereas time in `validate` or `build_operation_graph`
// is graph-construction cost on our side of the boundary. Timing the stages
// separately is what makes that distinction; one duration per build cannot.
//
// Each stage is wrapped where it is called -- graph_cache.h, which is where all
// five frontend calls live -- and accumulates into the table below, under the
// pass its caller was serving.
// The end-of-run summary reports each as a mean over its calls. Only sums are
// kept, so the mean is all that can be recovered -- and since a build happens
// once per distinct cache key, those calls span different shapes rather than
// repeating one. Read a stage mean as where build time goes in aggregate, not as
// the cost of any particular build.
// ============================================================================

// The frontend calls that make up a build, in the order they run. `kCount` must
// stay last: it sizes the table below. `kStageNames` is indexed by these values
// when the summary prints, so the two must be kept in the same order.
enum class BuildStage { Validate, BuildOpGraph, CreatePlans, CheckSupport, BuildPlans, kCount };
inline constexpr const char *kStageNames[] = {
    "validate", "build_operation_graph", "create_execution_plans", "check_support", "build_plans"};

// Totals for one (pass, stage) pair. Relaxed ordering is sufficient: these
// counters order nothing, and the only read happens once, after the threads that
// wrote them are done.
struct StageTiming {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> time_ns{0};
};

// Bucketed by pass, so the summary can report the cost of each stage separately
// for forward and backward. Unlike the thread registry above, this table needs no
// leak to outlive the exit handler that reads it: it holds nothing but atomics, so
// it is trivially destructible and no destructor is registered for it at all.
constexpr size_t kStageBuckets = 2 * static_cast<size_t>(BuildStage::kCount);
inline StageTiming &stage_timing(bool is_fwd, BuildStage s) {
  static std::array<StageTiming, kStageBuckets> table{};
  const size_t idx =
      (is_fwd ? 0u : 1u) * static_cast<size_t>(BuildStage::kCount) + static_cast<size_t>(s);
  return table[idx];
}

// Times one stage: clock read in the constructor, accumulated in the destructor.
// Recording on scope exit rather than at an explicit stop() keeps a failing stage
// measurable -- the frontend calls are wrapped in NVTE_CHECK_CUDNN_FE, which
// throws, and the destructor still runs during unwinding -- so a build that dies
// in `check_support` contributes its time to failure instead of vanishing from the
// summary. `on` is latched at construction rather than re-tested in the destructor,
// which is what keeps that symmetric: the destructor can never accumulate against a
// `start` the constructor left unset.
struct ScopedBuildTimer {
  BuildStage stage;
  bool on;
  bool is_fwd;
  std::chrono::steady_clock::time_point start;
  ScopedBuildTimer(bool is_fwd_, BuildStage s) : stage(s), on(enabled()), is_fwd(is_fwd_) {
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
    StageTiming &t = stage_timing(is_fwd, stage);
    t.time_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    t.calls.fetch_add(1, std::memory_order_relaxed);
  }
};

// Time `fn` as `stage` of the given pass ("fwd"/"bwd", matching the record_*
// helpers above). Preferred over declaring a ScopedBuildTimer at the call site:
// the measured region is exactly the call passed in, so surrounding work cannot
// drift into it as that code changes. With diagnostics off this costs the pass
// comparison and one cached-flag check; both are per build, not per lookup.
template <typename Fn>
inline void timer(const char *pass, BuildStage stage, Fn &&fn) {
  ScopedBuildTimer scoped(std::strcmp(pass, "fwd") == 0, stage);
  fn();
}

// ============================================================================
// Summary: on process exit, print cache event counters and graph build timings.
// ============================================================================
inline void register_summary_once() {
  static const bool registered = [] {
    std::atexit([] {
      if (!enabled()) return;
      // Build the whole summary in memory and emit it with a single write, so
      // that the blocks of concurrently-exiting processes (one per rank under
      // torchrun) stay grouped instead of interleaving line by line.
      std::string block;
      block += "[FUSED-ATTN-CACHE] " + process_tag() + " | ===== summary begin =====\n";
      // Per-thread breakdown (sorted by tid). Useful in the single-process
      // context-parallel case where each device runs on its own thread.
      {
        std::lock_guard<std::mutex> lock(thread_registry_mutex());
        std::vector<ThreadCounters *> blocks = thread_registry();
        std::sort(blocks.begin(), blocks.end(),
                  [](const ThreadCounters *a, const ThreadCounters *b) { return a->tid < b->tid; });
        for (const ThreadCounters *tc : blocks) {
          char tid_field[16];
          std::snprintf(tid_field, sizeof(tid_field), "tid=%u", tc->tid);
          block += format_counter_line("SUMMARY-TID", tid_field, tc->fwd, tc->bwd);
        }
      }
      // Totals last, so they read as the sum of the per-thread lines above.
      block += format_counter_line("SUMMARY", "tid=all", counters(/*is_fwd=*/true),
                                   counters(/*is_fwd=*/false));
      for (int p = 0; p < 2; ++p) {
        const bool is_fwd = (p == 0);
        const char *pass = is_fwd ? "fwd" : "bwd";
        for (int i = 0; i < static_cast<int>(BuildStage::kCount); ++i) {
          const BuildStage s = static_cast<BuildStage>(i);
          const StageTiming &t = stage_timing(is_fwd, s);
          const uint64_t n = t.calls.load(std::memory_order_relaxed);
          if (n == 0) continue;
          const double total_ms =
              static_cast<double>(t.time_ns.load(std::memory_order_relaxed)) / 1e6;
          char line[288];
          std::snprintf(line, sizeof(line),
                        "[FUSED-ATTN-CACHE] %s | %-3s %-22s | calls=%" PRIu64
                        " | time=%9.3f ms/call\n",
                        process_tag().c_str(), pass, kStageNames[i], n, total_ms / n);
          block += line;
        }
      }
      block += "[FUSED-ATTN-CACHE] " + process_tag() + " | ===== summary end =====\n";
      std::fwrite(block.data(), 1, block.size(), stderr);
      std::fflush(stderr);
    });
    return true;
  }();
  (void)registered;
}

}  // namespace graph_cache_debug
}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
