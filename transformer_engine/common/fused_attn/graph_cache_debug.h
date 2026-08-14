/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// Fused-attention graph cache diagnostics.
//
// Enable at runtime with NVTE_FUSED_ATTN_CACHE_DEBUG. Two verbosity levels:
//   =1 : low volume. Cache event counters, per-build BUILD lines, and the
//        end-of-run SUMMARY (aggregate + per-thread) and stage timings. This is
//        enough to diagnose redundant rebuilds and profile build cost.
//   =2 : high volume (trace). Additionally emits a per-lookup HIT/MISS line with
//        the full shorthand config and a per-execution EXEC line. Use only when
//        you need to see *which* shapes are hitting/missing -- these fire on
//        every cache lookup and execution, so at suite scale they add I/O and
//        serialize threads on the stderr lock (perturbing the build timings).
//
// An optional ":<ranks>" suffix picks which processes emit, defaulting to rank 0
// so that output does not scale with the world size: "1:all" for every rank,
// "2:0,3" for a specific set. See `rank_selected` for when overriding pays off.
// ============================================================================

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

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
    return v > 0 ? v : 1;  // any non-empty, non-"0" value enables at least level 1
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

// True while this thread is inside a backend-support probe (`is_supported_*`),
// which builds a graph speculatively just to answer "is this config supported?".
// Such a build may never be executed -- notably the context-parallel per-step
// probe, which checks regimes this rank never runs -- so it is counted apart
// from builds triggered by a real execution.
inline bool &tl_in_probe() {
  static thread_local bool v = false;
  return v;
}

// Marks the calling scope as a support probe. Saves/restores rather than
// clearing, so it stays correct if probes ever nest.
struct ScopedProbe {
  bool prev;
  ScopedProbe() : prev(tl_in_probe()) { tl_in_probe() = true; }
  ~ScopedProbe() { tl_in_probe() = prev; }
};

inline const char *src_tag() { return tl_in_probe() ? "probe" : "exec"; }

// Milliseconds since the first diagnostic event. Used to correlate build
// start/end across threads and detect whether same-shape builds on different
// devices overlap in wall-clock time.
inline double now_ms() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Per-thread build-start timestamp, set at the miss that triggers a build and
// read when the build completes, to report each build's duration.
inline double &tl_build_start_ms() {
  static thread_local double v = 0.0;
  return v;
}

// Registered at first use. On process exit, prints overall event counters and
// graph build timings.
inline void register_summary_once();

// ============================================================================
// Cache event counters (forward/backward):
//   - BUILD: a successful graph build; triggered by a cache miss
//   - EXEC: a graph execution call with valid runtime tensors
//   - HIT: a cache lookup that hit; may not trigger an EXEC, and may only be
//          a backend availability check or from the first shape-probing call of
//          nvte_fused_attn_fwd/bwd which has no runtime tensors
//   - MISS: a cache lookup that missed; triggers a graph build
// ============================================================================

struct EventCounters {
  std::atomic<uint64_t> built{0};
  // Subset of `built` that a support probe triggered. A large probe share means
  // time is going into graphs that may never run.
  std::atomic<uint64_t> built_probe{0};
  std::atomic<uint64_t> exec{0};
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
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
inline std::string format_counter_line(const char *event, const char *tid_field,
                                       const EventCounters &f, const EventCounters &b,
                                       const char *extra) {
  char buf[640];
  std::snprintf(buf, sizeof(buf),
                "[FUSED-ATTN-CACHE] %s | %-11s | %-7s | fwd miss=%4" PRIu64 ", hit=%4" PRIu64
                ", built=%4" PRIu64 " (for probe %4" PRIu64 "), exec=%4" PRIu64 " | bwd miss=%4" PRIu64
                ", hit=%4" PRIu64 ", built=%4" PRIu64 " (for probe %4" PRIu64 "), exec=%4" PRIu64 "%s\n",
                process_tag().c_str(), event, tid_field, f.miss.load(std::memory_order_relaxed),
                f.hit.load(std::memory_order_relaxed), f.built.load(std::memory_order_relaxed),
                f.built_probe.load(std::memory_order_relaxed),
                f.exec.load(std::memory_order_relaxed), b.miss.load(std::memory_order_relaxed),
                b.hit.load(std::memory_order_relaxed), b.built.load(std::memory_order_relaxed),
                b.built_probe.load(std::memory_order_relaxed),
                b.exec.load(std::memory_order_relaxed), extra);
  return std::string(buf);
}

inline void print_counter_block(const char *event, const char *tid_field, const EventCounters &f,
                                const EventCounters &b, const char *extra = "") {
  const std::string line = format_counter_line(event, tid_field, f, b, extra);
  std::fputs(line.c_str(), stderr);
  std::fflush(stderr);
}

inline void print_counters(const char *event, const char *extra = "") {
  char tid_field[16];
  std::snprintf(tid_field, sizeof(tid_field), "tid=%u", thread_seq_id());
  print_counter_block(event, tid_field, counters(/*is_fwd=*/true), counters(/*is_fwd=*/false),
                      extra);
}

inline void record_build(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).built.fetch_add(1, std::memory_order_relaxed);
  thread_counters(is_fwd).built.fetch_add(1, std::memory_order_relaxed);
  if (tl_in_probe()) {
    counters(is_fwd).built_probe.fetch_add(1, std::memory_order_relaxed);
    thread_counters(is_fwd).built_probe.fetch_add(1, std::memory_order_relaxed);
  }
  // Report build completion time and this build's wall-clock duration so we can
  // tell whether same-shape builds on different devices overlap.
  const double t_end = now_ms();
  char extra[80];
  std::snprintf(extra, sizeof(extra), " | src=%-5s t=%.1f dur=%.1f ms", src_tag(), t_end,
                t_end - tl_build_start_ms());
  print_counters(is_fwd ? "fwd BUILD" : "bwd BUILD", extra);
}

inline void record_exec(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).exec.fetch_add(1, std::memory_order_relaxed);
  thread_counters(is_fwd).exec.fetch_add(1, std::memory_order_relaxed);
  // The per-exec line fires on every execution; keep it out of the level-1 path.
  if (!trace_enabled()) return;
  char extra[32];
  std::snprintf(extra, sizeof(extra), " | t=%.1f", now_ms());
  print_counters(is_fwd ? "fwd EXEC" : "bwd EXEC", extra);
}

// `device_key` is the cache-scope discriminator from make_cache_key(), not a device ordinal:
// it is the packed (SM arch, SM count) when devices share plans, else the device id.
inline void record_cache_lookup(const char *pass, bool hit, const FusedAttnConfig &c,
                                int device_key) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  EventCounters &pc = counters(is_fwd);
  (hit ? pc.hit : pc.miss).fetch_add(1, std::memory_order_relaxed);
  EventCounters &tpc = thread_counters(is_fwd);
  (hit ? tpc.hit : tpc.miss).fetch_add(1, std::memory_order_relaxed);
  const double t = now_ms();
  // A miss triggers a build right after this call; stamp the build start so the
  // subsequent BUILD line can report duration. Do this even at level 1.
  if (!hit) tl_build_start_ms() = t;
  // The per-lookup config dump is the highest-volume line (one per cache probe);
  // keep it out of the level-1 path and off the stderr lock unless tracing.
  if (!trace_enabled()) return;
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %s | %-3s %-4s | tid=%u devkey=%d t=%.1f src=%-5s | train=%d det=%d cg=%d "
      "maxlogit=%d fwd=%d "
      "mask=%" PRId64 " bias=%" PRId64 " wl=%" PRId64 " wr=%" PRId64 " brd=%d softmax=%" PRId64
      " scale_mode=%" PRId64 " dropout=%g attn_scale=%g qkv_dt=%" PRId64 " o_dt=%" PRId64
      " do_dt=%" PRId64 " dqkv_dt=%" PRId64 " qkv_lay=%" PRId64 " o_fmt=%" PRId64 " do_fmt=%" PRId64
      " dqkv_lay=%" PRId64 " qkv_sif=%" PRId64 " do_sif=%" PRId64 " b=%" PRId64 " h=%" PRId64
      " hg=%" PRId64 " dqk=%" PRId64 " dv=%" PRId64 " sq=%" PRId64 " skv=%" PRId64 " tq=%" PRId64
      " tkv=%" PRId64 " bb=%" PRId64 " btq=%" PRId64 " btkv=%" PRId64 " npk=%" PRId64
      " npv=%" PRId64 " psk=%" PRId64 " psv=%" PRId64 " mppk=%" PRId64 " mppv=%" PRId64
      " bias_b=%" PRId64 " bias_h=%" PRId64 " bias_sq=%" PRId64 " bias_skv=%" PRId64 "\n",
      process_tag().c_str(), pass, hit ? "HIT" : "MISS", thread_seq_id(), device_key, t, src_tag(),
      static_cast<int>(c.is_training),
      static_cast<int>(c.deterministic), static_cast<int>(c.cuda_graph),
      static_cast<int>(c.return_max_logit), static_cast<int>(c.check_forward),
      static_cast<int64_t>(c.attn_mask_type), static_cast<int64_t>(c.bias_type),
      static_cast<int64_t>(c.window_size_left), static_cast<int64_t>(c.window_size_right),
      static_cast<int>(c.bottom_right_diagonal), static_cast<int64_t>(c.softmax_type),
      static_cast<int64_t>(c.scaling_mode), static_cast<double>(c.dropout),
      static_cast<double>(c.attn_scale), static_cast<int64_t>(c.qkv_dtype),
      static_cast<int64_t>(c.o_dtype), static_cast<int64_t>(c.do_dtype),
      static_cast<int64_t>(c.dqkv_dtype), static_cast<int64_t>(c.qkv_layout),
      static_cast<int64_t>(c.o_format), static_cast<int64_t>(c.do_format),
      static_cast<int64_t>(c.dqkv_layout), static_cast<int64_t>(c.qkv_scale_inv_format),
      static_cast<int64_t>(c.do_scale_inv_format), static_cast<int64_t>(c.batch_size),
      static_cast<int64_t>(c.num_attn_heads), static_cast<int64_t>(c.num_gqa_groups),
      static_cast<int64_t>(c.head_dim_qk), static_cast<int64_t>(c.head_dim_v),
      static_cast<int64_t>(c.max_seqlen_q), static_cast<int64_t>(c.max_seqlen_kv),
      static_cast<int64_t>(c.num_tokens_q), static_cast<int64_t>(c.num_tokens_kv),
      static_cast<int64_t>(c.bucketed_batch_size), static_cast<int64_t>(c.bucketed_num_tokens_q),
      static_cast<int64_t>(c.bucketed_num_tokens_kv), static_cast<int64_t>(c.num_pages_k),
      static_cast<int64_t>(c.num_pages_v), static_cast<int64_t>(c.page_size_k),
      static_cast<int64_t>(c.page_size_v), static_cast<int64_t>(c.max_pages_per_seq_k),
      static_cast<int64_t>(c.max_pages_per_seq_v), static_cast<int64_t>(c.bias_batch_size),
      static_cast<int64_t>(c.bias_num_heads), static_cast<int64_t>(c.bias_seqlen_q),
      static_cast<int64_t>(c.bias_seqlen_kv));
}

// ============================================================================
// Graph build timings for individual cuDNN-frontend calls in forward/backward:
// e.g. `validate`, `build_operation_graph`, `create_execution_plans`,
// `check_support`, `build_plans`
// ============================================================================

enum class BuildStage { Validate, BuildOpGraph, CreatePlans, CheckSupport, BuildPlans, kCount };
inline constexpr const char *kStageNames[] = {
    "validate", "build_operation_graph", "create_execution_plans", "check_support", "build_plans"};
struct StageTiming {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> time_ns{0};
};
// Bucketed by pass and by whether a support probe drove the build, so the
// summary can say how much of (notably) `build_plans` was speculative.
constexpr size_t kStageBuckets = 4 * static_cast<size_t>(BuildStage::kCount);
inline StageTiming &stage_timing(bool is_fwd, bool is_probe, BuildStage s) {
  static std::array<StageTiming, kStageBuckets> table{};
  const size_t idx = ((is_fwd ? 0u : 1u) * 2u + (is_probe ? 1u : 0u)) *
                         static_cast<size_t>(BuildStage::kCount) +
                     static_cast<size_t>(s);
  return table[idx];
}

struct ScopedBuildTimer {
  BuildStage stage;
  bool on;
  bool is_fwd;
  bool is_probe;
  std::chrono::steady_clock::time_point start;
  ScopedBuildTimer(bool is_fwd_, BuildStage s)
      : stage(s), on(enabled()), is_fwd(is_fwd_), is_probe(tl_in_probe()) {
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
    StageTiming &t = stage_timing(is_fwd, is_probe, stage);
    t.time_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
    t.calls.fetch_add(1, std::memory_order_relaxed);
  }
};

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
          block += format_counter_line("SUMMARY-TID", tid_field, tc->fwd, tc->bwd, "");
        }
      }
      // Totals last, so they read as the sum of the per-thread lines above.
      block += format_counter_line("SUMMARY", "tid=all", counters(/*is_fwd=*/true),
                                   counters(/*is_fwd=*/false), "");
      for (int p = 0; p < 2; ++p) {
        const bool is_fwd = (p == 0);
        const char *pass = is_fwd ? "fwd" : "bwd";
        for (int q = 0; q < 2; ++q) {
          const bool is_probe = (q == 1);
          for (int i = 0; i < static_cast<int>(BuildStage::kCount); ++i) {
            const BuildStage s = static_cast<BuildStage>(i);
            const StageTiming &t = stage_timing(is_fwd, is_probe, s);
            const uint64_t n = t.calls.load(std::memory_order_relaxed);
            if (n == 0) continue;
            const double total_ms =
                static_cast<double>(t.time_ns.load(std::memory_order_relaxed)) / 1e6;
            char line[288];
            std::snprintf(line, sizeof(line),
                          "[FUSED-ATTN-CACHE] %s | %-3s src=%-5s %-22s | calls=%" PRIu64
                          " | time=%9.3f ms/call\n",
                          process_tag().c_str(), pass, is_probe ? "probe" : "exec", kStageNames[i],
                          n, total_ms / n);
            block += line;
          }
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

// ============================================================================
// Single-flight graph cache coordination.
//
// The fused-attention graph caches are process-wide and shared across threads.
// The lock is intentionally released while a graph is compiled so that
// *different* graphs can build in parallel. The downside is that when several
// threads miss the *same* key at the same instant (e.g. the device-worker
// threads of a single-process context-parallel run stepping in lockstep), they
// all compile an identical graph and all but one discard the result at insert
// time -- wasted host-side `build_plans` work.
//
// A single-flight (a.k.a. "thundering herd") guard closes that gap: at most one
// thread compiles a given key while the others wait for it. Distinct keys still
// build concurrently, so the parallel-build win is kept.
//
// Usage in a get_graph path:
//   static SingleFlight<FusedAttnConfig> sf;
//   {
//     std::unique_lock<std::mutex> lock(sf.mutex);
//     sf.cv.wait(lock, [&] {
//       return cache.count(key) != 0 || sf.in_progress.count(key) == 0;
//     });
//     if (auto it = cache.find(key); it != cache.end()) { ...cache hit... }
//     sf.in_progress.insert(key);          // claim the build
//   }
//   ClaimGuard<FusedAttnConfig> guard{sf, key};   // auto-release + notify
//   ...build...
//   { std::lock_guard<std::mutex> lock(sf.mutex); cache.insert({key, graph}); }
// ============================================================================
namespace graph_cache {

// Opt-in with NVTE_FUSED_ATTN_CACHE_SINGLE_FLIGHT=1. When off, no thread claims a
// key, so the wait below falls through immediately and concurrent misses of one
// key each compile, with all but one result discarded at insert: wasted host work,
// but no thread ever blocks on another thread's compile.
inline bool single_flight_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_SINGLE_FLIGHT");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

template <typename KeyT>
struct SingleFlight {
  std::mutex mutex;
  std::condition_variable cv;
  std::set<KeyT> in_progress;
};

// RAII: on scope exit, drop this thread's build claim on `key` and wake any
// threads waiting on the same key. Clears the claim even if the build throws,
// so waiters never deadlock (they simply re-elect a builder).
template <typename KeyT>
struct ClaimGuard {
  SingleFlight<KeyT> &sf;
  const KeyT &key;
  ~ClaimGuard() {
    {
      std::lock_guard<std::mutex> lock(sf.mutex);
      sf.in_progress.erase(key);
    }
    sf.cv.notify_all();
  }
};

}  // namespace graph_cache
}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
