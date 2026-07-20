/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// [GRAPH-DEBUG] TEMPORARY DEBUG INSTRUMENTATION -- REMOVE AFTER VERIFICATION.
//
// Counts fused-attention cuDNN graph *builds* (cache misses that construct a new
// graph) vs. *executions* (real forward/backward runs, excluding workspace-sizing
// probes) to detect redundant graph construction. Also logs every graph-cache
// lookup (HIT/MISS + the key fields) to diagnose stale-cache reuse across tests.
//
// Enable at runtime with:  export NVTE_FUSED_ATTN_GRAPH_DEBUG=1
//   - A "BUILD" line is printed whenever a new graph is constructed.
//   - A "HIT"/"MISS" line with the key fields is printed on every cache lookup.
//   - A "thd ... path=legacy|direct" line is printed on every THD (ragged) lookup, showing
//     which impl path (bucketed batch vs. real batch) the graph was built for.
//   - A "SUMMARY" line with final build/exec totals is printed at process exit, followed by a
//     "THD-PATH" line with per-path lookup/build totals (low builds/lookups on the legacy path
//     means batch bucketing is collapsing distinct batch sizes onto shared graphs), and one
//     "STAGE <name>" line per FE build stage (validate ... build_plans) with total CPU/wall
//     time and call count -- on `main` these were static boolean checks (~0 cost).
//
// Separately, force every lookup to miss (never reuse a cached graph) with:
//   export NVTE_FUSED_ATTN_DISABLE_CACHE=1
// If a suite that fails with the cache enabled passes with it disabled, the bug
// is stale-cache reuse (an incomplete make_cache_key / operator<).
//
// To remove all of this instrumentation later:
//   1. Delete this file (graph_debug.h).
//   2. Remove every line tagged with the "[GRAPH-DEBUG]" marker in:
//        - fused_attn_fp8.cu
//        - fused_attn_f16_arbitrary_seqlen.cu
// ============================================================================

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_GRAPH_DEBUG_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_GRAPH_DEBUG_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

// [GRAPH-DEBUG] Backtrace printing needs glibc's <execinfo.h> and libstdc++'s <cxxabi.h>
// (demangling). Gate on availability so non-glibc toolchains still build; dump_backtrace()
// becomes a no-op there.
#if defined(__has_include)
#if __has_include(<execinfo.h>) && __has_include(<cxxabi.h>)
#define NVTE_FUSED_ATTN_GRAPH_DEBUG_HAVE_BACKTRACE 1
#include <cxxabi.h>
#include <execinfo.h>
#endif
#endif

#include "config_and_params.h"  // [GRAPH-DEBUG] for FusedAttnConfig field dump

namespace transformer_engine {
namespace fused_attn_graph_debug {

// Short, stable per-thread id (0, 1, 2, ...) assigned on first use. The graph caches are
// static thread_local, so a graph built on one thread is invisible to another; tagging every
// lookup with its thread id makes cross-thread rebuilds of an identical key visible.
inline unsigned thread_seq_id() {
  static std::atomic<unsigned> next{0};
  static thread_local unsigned id = next.fetch_add(1);
  return id;
}

inline std::atomic<uint64_t> &fwd_built() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t> &fwd_exec() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t> &bwd_built() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t> &bwd_exec() {
  static std::atomic<uint64_t> v{0};
  return v;
}

// THD (ragged) cache lookups split by which impl path the graph was built for:
//   legacy = batch quantized into a bucket (many batch sizes share one graph)
//   direct = cu_seqlens fed to cuDNN directly (real batch baked in, no batch sharing)
// "builds" counts the lookups that actually constructed a new graph. A low builds/lookups
// ratio on the legacy path is the visible sign that batch bucketing is collapsing distinct
// batch sizes onto shared graphs.
inline std::atomic<uint64_t> &thd_legacy_lookup() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t> &thd_legacy_build() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t> &thd_direct_lookup() {
  static std::atomic<uint64_t> v{0};
  return v;
}
inline std::atomic<uint64_t> &thd_direct_build() {
  static std::atomic<uint64_t> v{0};
  return v;
}

inline bool enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_GRAPH_DEBUG");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

inline void dump(const char *event) {
  std::fprintf(
      stderr,
      "[GRAPH-DEBUG] %-10s | tid=%u | fwd built=%llu exec=%llu | bwd built=%llu exec=%llu\n", event,
      thread_seq_id(), static_cast<unsigned long long>(fwd_built().load()),
      static_cast<unsigned long long>(fwd_exec().load()),
      static_cast<unsigned long long>(bwd_built().load()),
      static_cast<unsigned long long>(bwd_exec().load()));
  std::fflush(stderr);
}

inline void dump_thd_summary() {
  std::fprintf(stderr,
               "[GRAPH-DEBUG] THD-PATH   | legacy lookups=%llu builds=%llu | direct lookups=%llu "
               "builds=%llu\n",
               static_cast<unsigned long long>(thd_legacy_lookup().load()),
               static_cast<unsigned long long>(thd_legacy_build().load()),
               static_cast<unsigned long long>(thd_direct_lookup().load()),
               static_cast<unsigned long long>(thd_direct_build().load()));
  std::fflush(stderr);
}

// [GRAPH-DEBUG] Host-memory footprint of cached graphs, split fwd/bwd (index 0/1).
//   serialized bytes: size of fe::graph::Graph::serialize() output -- a proxy for the host
//     memory one built graph holds (its plan / engine config / tensor metadata). Summed over
//     builds; avg = sum / count gives the per-graph host cost.
//   cache entries: high-water number of live graphs in the shared std::map (one graph per key).
// Device memory (workspace) is separate and sized per execute(), not held by the cached graph.
inline int pass_index(const char *pass) { return (pass[0] == 'b') ? 1 : 0; }  // "bwd" -> 1

inline std::atomic<uint64_t> &serial_bytes(int i) {
  static std::array<std::atomic<uint64_t>, 2> v{};
  return v[i];
}
inline std::atomic<uint64_t> &serial_count(int i) {
  static std::array<std::atomic<uint64_t>, 2> v{};
  return v[i];
}
inline std::atomic<uint64_t> &cache_entries(int i) {
  static std::array<std::atomic<uint64_t>, 2> v{};
  return v[i];
}

inline void dump_memory_summary() {
  for (int i = 0; i < 2; ++i) {
    const char *pass = (i == 0) ? "fwd" : "bwd";
    uint64_t cnt = serial_count(i).load();
    uint64_t bytes = serial_bytes(i).load();
    uint64_t entries = cache_entries(i).load();
    double total_kb = static_cast<double>(bytes) / 1024.0;
    double avg_kb = cnt ? total_kb / static_cast<double>(cnt) : 0.0;
    std::fprintf(
        stderr,
        "[GRAPH-DEBUG] MEMORY %-3s | cache entries=%llu | serialized graphs=%llu total=%.1f KB (avg %.1f KB)\n",
        pass, static_cast<unsigned long long>(entries), static_cast<unsigned long long>(cnt),
        total_kb, avg_kb);
  }
  std::fflush(stderr);
}

// [GRAPH-DEBUG] Per-stage CPU/wall time for the FE build pipeline (validate ... build_plans).
// On `main` these were static boolean checks (~0 cost); this quantifies the added cost.
enum class BuildStage { Validate, BuildOpGraph, CreatePlans, CheckSupport, BuildPlans, kCount };

inline const char *stage_name(BuildStage s) {
  switch (s) {
    case BuildStage::Validate:
      return "validate";
    case BuildStage::BuildOpGraph:
      return "build_operation_graph";
    case BuildStage::CreatePlans:
      return "create_execution_plans";
    case BuildStage::CheckSupport:
      return "check_support";
    case BuildStage::BuildPlans:
      return "build_plans";
    default:
      return "?";
  }
}

inline std::atomic<uint64_t> &stage_calls(BuildStage s) {
  static std::array<std::atomic<uint64_t>, static_cast<size_t>(BuildStage::kCount)> v{};
  return v[static_cast<size_t>(s)];
}
inline std::atomic<uint64_t> &stage_cpu_ns(BuildStage s) {
  static std::array<std::atomic<uint64_t>, static_cast<size_t>(BuildStage::kCount)> v{};
  return v[static_cast<size_t>(s)];
}
inline std::atomic<uint64_t> &stage_wall_ns(BuildStage s) {
  static std::array<std::atomic<uint64_t>, static_cast<size_t>(BuildStage::kCount)> v{};
  return v[static_cast<size_t>(s)];
}

inline void dump_stage_summary() {
  for (int i = 0; i < static_cast<int>(BuildStage::kCount); ++i) {
    BuildStage s = static_cast<BuildStage>(i);
    uint64_t n = stage_calls(s).load();
    if (n == 0) continue;
    double cpu_ms = static_cast<double>(stage_cpu_ns(s).load()) / 1e6;
    double wall_ms = static_cast<double>(stage_wall_ns(s).load()) / 1e6;
    std::fprintf(stderr,
                 "[GRAPH-DEBUG] STAGE %-22s | calls=%llu | cpu=%.1f ms (avg %.3f ms) | wall=%.1f ms\n",
                 stage_name(s), static_cast<unsigned long long>(n), cpu_ms, cpu_ms / n, wall_ms);
  }
  std::fflush(stderr);
}

inline void register_summary_once() {
  static const bool registered = [] {
    std::atexit([] {
      if (enabled()) {
        dump("SUMMARY");
        dump_thd_summary();
        dump_stage_summary();
        dump_memory_summary();
      }
    });
    return true;
  }();
  (void)registered;
}

inline void note_fwd_build() {
  if (!enabled()) return;
  register_summary_once();
  fwd_built().fetch_add(1);
  dump("fwd BUILD");
}
inline void note_fwd_exec() {
  if (!enabled()) return;
  register_summary_once();
  fwd_exec().fetch_add(1);
}
inline void note_bwd_build() {
  if (!enabled()) return;
  register_summary_once();
  bwd_built().fetch_add(1);
  dump("bwd BUILD");
}
inline void note_bwd_exec() {
  if (!enabled()) return;
  register_summary_once();
  bwd_exec().fetch_add(1);
}

// [GRAPH-DEBUG] Record the serialized size (host-memory proxy) of one freshly built graph.
// Call only after a successful serialize() so the average reflects real graphs.
inline void note_graph_size(const char *pass, size_t serialized_bytes) {
  if (!enabled()) return;
  register_summary_once();
  int i = pass_index(pass);
  serial_bytes(i).fetch_add(serialized_bytes);
  serial_count(i).fetch_add(1);
}

// [GRAPH-DEBUG] Record the current shared-cache entry count (kept as a high-water mark).
inline void note_cache_size(const char *pass, size_t entries) {
  if (!enabled()) return;
  register_summary_once();
  int i = pass_index(pass);
  uint64_t prev = cache_entries(i).load();
  while (entries > prev && !cache_entries(i).compare_exchange_weak(prev, entries)) {
  }
}

// Returns true when the graph cache should be bypassed (every lookup treated as a
// miss so a fresh graph is built each call). Gated by NVTE_FUSED_ATTN_DISABLE_CACHE.
inline bool cache_disabled() {
  static const bool off = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_DISABLE_CACHE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return off;
}

// [GRAPH-DEBUG] Opt-in C++ backtrace printing next to each cache lookup. Kept separate from the
// main NVTE_FUSED_ATTN_GRAPH_DEBUG switch because a full stack per lookup is very verbose; enable
// with NVTE_FUSED_ATTN_GRAPH_DEBUG_BACKTRACE=1 (the main switch must also be on).
inline bool backtrace_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_GRAPH_DEBUG_BACKTRACE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

// [GRAPH-DEBUG] Frames to print per lookup (override with NVTE_FUSED_ATTN_GRAPH_DEBUG_BACKTRACE_DEPTH).
inline int backtrace_depth() {
  static const int depth = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_GRAPH_DEBUG_BACKTRACE_DEPTH");
    int d = (e != nullptr && e[0] != '\0') ? std::atoi(e) : 24;
    if (d < 1) d = 1;
    if (d > 128) d = 128;
    return d;
  }();
  return depth;
}

// [GRAPH-DEBUG] Print a symbolized (and, where possible, demangled) C++ backtrace, one frame per
// line, each tagged so the frames group visually under the HIT/MISS line they belong to. `skip`
// drops the top frames that are just this instrumentation (dump_backtrace + its caller). For
// readable function names the library must be built/linked with -rdynamic (or -g); otherwise
// non-exported frames show as "<path>(+0x<off>)".
inline void dump_backtrace(const char *tag, int skip = 2) {
  if (!backtrace_enabled()) return;
#if defined(NVTE_FUSED_ATTN_GRAPH_DEBUG_HAVE_BACKTRACE)
  const int max_frames = backtrace_depth() + skip;
  std::vector<void *> frames(static_cast<size_t>(max_frames));
  int n = ::backtrace(frames.data(), max_frames);
  if (n <= skip) return;
  char **symbols = ::backtrace_symbols(frames.data(), n);
  if (symbols == nullptr) return;
  for (int i = skip; i < n; ++i) {
    // glibc format: "<path>(<mangled>+0x<off>) [0x<addr>]"; demangle the "<mangled>" span.
    std::string line = symbols[i];
    char *open = std::strchr(symbols[i], '(');
    char *plus = open ? std::strchr(open, '+') : nullptr;
    if (open != nullptr && plus != nullptr && plus > open + 1) {
      std::string mangled(open + 1, plus);
      int status = 0;
      char *demangled = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
      if (status == 0 && demangled != nullptr) {
        line = std::string(symbols[i], open + 1) + demangled + plus;
        std::free(demangled);
      }
    }
    std::fprintf(stderr, "[GRAPH-DEBUG]   bt[%-4s] #%02d %s\n", tag, i - skip, line.c_str());
  }
  std::fflush(stderr);
  std::free(symbols);
#else
  (void)tag;
  (void)skip;
#endif
}

// Logs one graph-cache lookup with its outcome (HIT/MISS) and the *real* (pre-
// normalization) config fields. A std::map HIT means the two configs compare equal
// under operator<, so the field that actually distinguishes a wrongly-reused graph
// is one that make_cache_key() normalized away or that operator< omits -- pass the
// real cfg (not the normalized cache key) here so that difference is visible when
// diffing a wrong HIT against the earlier BUILD that created the reused graph.
inline void note_cache_lookup(const char *pass, bool hit, const FusedAttnConfig &c) {
  if (!enabled()) return;
  register_summary_once();
  std::fprintf(
      stderr,
      "[GRAPH-DEBUG] %-3s %-4s%s | tid=%u | train=%d det=%d cg=%d maxlogit=%d fwd=%d mask=%lld "
      "bias=%lld "
      "wl=%lld wr=%lld brd=%d softmax=%lld scale_mode=%lld dropout=%g attn_scale=%g "
      "qkv_dt=%lld o_dt=%lld do_dt=%lld dqkv_dt=%lld qkv_lay=%lld o_fmt=%lld do_fmt=%lld "
      "dqkv_lay=%lld qkv_sif=%lld do_sif=%lld b=%lld h=%lld hg=%lld dqk=%lld dv=%lld sq=%lld "
      "skv=%lld tq=%lld tkv=%lld bb=%lld btq=%lld btkv=%lld npk=%lld npv=%lld psk=%lld psv=%lld "
      "mppk=%lld mppv=%lld bias_b=%lld bias_h=%lld bias_sq=%lld bias_skv=%lld\n",
      pass, hit ? "HIT" : "MISS", (hit && cache_disabled()) ? " [cache-disabled->rebuild]" : "",
      thread_seq_id(), static_cast<int>(c.is_training), static_cast<int>(c.deterministic),
      static_cast<int>(c.cuda_graph), static_cast<int>(c.return_max_logit),
      static_cast<int>(c.is_forward), static_cast<long long>(c.attn_mask_type),
      static_cast<long long>(c.bias_type), static_cast<long long>(c.window_size_left),
      static_cast<long long>(c.window_size_right), static_cast<int>(c.bottom_right_diagonal),
      static_cast<long long>(c.softmax_type), static_cast<long long>(c.scaling_mode),
      static_cast<double>(c.dropout), static_cast<double>(c.attn_scale),
      static_cast<long long>(c.qkv_dtype), static_cast<long long>(c.o_dtype),
      static_cast<long long>(c.do_dtype), static_cast<long long>(c.dqkv_dtype),
      static_cast<long long>(c.qkv_layout), static_cast<long long>(c.o_format),
      static_cast<long long>(c.do_format), static_cast<long long>(c.dqkv_layout),
      static_cast<long long>(c.qkv_scale_inv_format), static_cast<long long>(c.do_scale_inv_format),
      static_cast<long long>(c.batch_size), static_cast<long long>(c.num_attn_heads),
      static_cast<long long>(c.num_gqa_groups), static_cast<long long>(c.head_dim_qk),
      static_cast<long long>(c.head_dim_v), static_cast<long long>(c.max_seqlen_q),
      static_cast<long long>(c.max_seqlen_kv), static_cast<long long>(c.num_tokens_q),
      static_cast<long long>(c.num_tokens_kv), static_cast<long long>(c.bucketed_batch_size),
      static_cast<long long>(c.bucketed_num_tokens_q),
      static_cast<long long>(c.bucketed_num_tokens_kv), static_cast<long long>(c.num_pages_k),
      static_cast<long long>(c.num_pages_v), static_cast<long long>(c.page_size_k),
      static_cast<long long>(c.page_size_v), static_cast<long long>(c.max_pages_per_seq_k),
      static_cast<long long>(c.max_pages_per_seq_v), static_cast<long long>(c.bias_batch_size),
      static_cast<long long>(c.bias_num_heads), static_cast<long long>(c.bias_seqlen_q),
      static_cast<long long>(c.bias_seqlen_kv));
  std::fflush(stderr);
  dump_backtrace(hit ? "HIT" : "MISS");  // [GRAPH-DEBUG] frames for this fwd/bwd lookup
}

// Records, for one THD (ragged) cache lookup, which impl path the graph was built for --
// "legacy" (batch quantized into a bucket) vs "direct" (real batch fed via cu_seqlens) -- and
// whether it hit the cache. `built` should reflect whether a new graph was actually constructed
// (i.e. a real miss, or a hit forced to rebuild by NVTE_FUSED_ATTN_DISABLE_CACHE). Comparing
// per-path lookups vs builds in the THD-PATH summary shows the batch-bucketing effect.
inline void note_thd_lookup(const char *pass, bool hit, bool built, bool legacy) {
  if (!enabled()) return;
  register_summary_once();
  if (legacy) {
    thd_legacy_lookup().fetch_add(1);
    if (built) thd_legacy_build().fetch_add(1);
  } else {
    thd_direct_lookup().fetch_add(1);
    if (built) thd_direct_build().fetch_add(1);
  }
  std::fprintf(stderr, "[GRAPH-DEBUG] thd %-3s %-4s | tid=%u | path=%s%s\n", pass,
               hit ? "HIT" : "MISS", thread_seq_id(), legacy ? "legacy" : "direct",
               (hit && built) ? " [cache-disabled->rebuild]" : "");
  std::fflush(stderr);
  dump_backtrace(hit ? "HIT" : "MISS");  // [GRAPH-DEBUG] frames for this THD (ragged) lookup
}

// [GRAPH-DEBUG] Thread-CPU clock (excludes time blocked on locks / GPU sync), in nanoseconds.
inline uint64_t cpu_now_ns() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// [GRAPH-DEBUG] RAII timer: records wall + thread-CPU time for one FE build stage. Zero cost
// (only an enabled() bool read) when NVTE_FUSED_ATTN_GRAPH_DEBUG is unset. The destructor records
// even on early return / thrown NVTE_CHECK, so it is safe to wrap the checked FE calls.
struct ScopedStageTimer {
  BuildStage stage;
  bool on;
  std::chrono::steady_clock::time_point w0;
  uint64_t c0{0};
  explicit ScopedStageTimer(BuildStage s) : stage(s), on(enabled()) {
    if (!on) return;
    register_summary_once();
    c0 = cpu_now_ns();
    w0 = std::chrono::steady_clock::now();
  }
  ~ScopedStageTimer() {
    if (!on) return;
    uint64_t cpu = cpu_now_ns() - c0;
    uint64_t wall = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0)
            .count());
    stage_cpu_ns(stage).fetch_add(cpu);
    stage_wall_ns(stage).fetch_add(wall);
    stage_calls(stage).fetch_add(1);
  }
};

}  // namespace fused_attn_graph_debug
}  // namespace transformer_engine

// [GRAPH-DEBUG] Wrap a single (possibly NVTE_CHECK_*-guarded) FE call to time it under `stage`.
#define GRAPH_DEBUG_TIME_STAGE(stage, expr)                                        \
  do {                                                                             \
    ::transformer_engine::fused_attn_graph_debug::ScopedStageTimer _gd_stage_timer( \
        ::transformer_engine::fused_attn_graph_debug::BuildStage::stage);          \
    expr;                                                                          \
  } while (0)

#endif  // TRANSFORMER_ENGINE_FUSED_ATTN_GRAPH_DEBUG_H_
