/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// Fused-attention graph cache diagnostics.
//
// Enable at runtime with NVTE_FUSED_ATTN_CACHE_DEBUG=1 to get the cache event
// counters and graph build timings, to help diagnose redundant graph rebuilds
// or stale-cache reuse, and to profile graph-build cost.
// ============================================================================

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "config_and_params.h"

namespace transformer_engine {
namespace fused_attn {
namespace graph_cache_debug {

// Enable diagnostics with NVTE_FUSED_ATTN_CACHE_DEBUG=1. Single read at startup, cached.
// Negligible overhead when unset.
inline bool enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

// More readable, shorter thread IDs (0, 1, 2, ...).
inline unsigned thread_seq_id() {
  static std::atomic<unsigned> next{0};
  static thread_local unsigned id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
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
  std::atomic<uint64_t> exec{0};
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
};

inline EventCounters &counters(bool is_fwd) {
  static EventCounters fwd;
  static EventCounters bwd;
  return is_fwd ? fwd : bwd;
}

inline void print_counters(const char *event) {
  const EventCounters &f = counters(/*is_fwd=*/true);
  const EventCounters &b = counters(/*is_fwd=*/false);
  std::fprintf(stderr,
               "[FUSED-ATTN-CACHE] %-10s | tid=%u | fwd built=%" PRIu64 " exec=%" PRIu64
               " hit=%" PRIu64 " miss=%" PRIu64 " | bwd built=%" PRIu64 " exec=%" PRIu64
               " hit=%" PRIu64 " miss=%" PRIu64 "\n",
               event, thread_seq_id(), f.built.load(std::memory_order_relaxed),
               f.exec.load(std::memory_order_relaxed), f.hit.load(std::memory_order_relaxed),
               f.miss.load(std::memory_order_relaxed), b.built.load(std::memory_order_relaxed),
               b.exec.load(std::memory_order_relaxed), b.hit.load(std::memory_order_relaxed),
               b.miss.load(std::memory_order_relaxed));
  std::fflush(stderr);
}

inline void record_build(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).built.fetch_add(1, std::memory_order_relaxed);
  print_counters(is_fwd ? "fwd BUILD" : "bwd BUILD");
}

inline void record_exec(const char *pass) {
  if (!enabled()) return;
  register_summary_once();
  const bool is_fwd = std::strcmp(pass, "fwd") == 0;
  counters(is_fwd).exec.fetch_add(1, std::memory_order_relaxed);
  print_counters(is_fwd ? "fwd EXEC" : "bwd EXEC");
}

inline void record_cache_lookup(const char *pass, bool hit, const FusedAttnConfig &c) {
  if (!enabled()) return;
  register_summary_once();
  EventCounters &pc = counters(std::strcmp(pass, "fwd") == 0);
  (hit ? pc.hit : pc.miss).fetch_add(1, std::memory_order_relaxed);
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %-3s %-4s | tid=%u | train=%d det=%d cg=%d maxlogit=%d fwd=%d "
      "mask=%" PRId64 " bias=%" PRId64 " wl=%" PRId64 " wr=%" PRId64 " brd=%d softmax=%" PRId64
      " scale_mode=%" PRId64 " dropout=%g attn_scale=%g qkv_dt=%" PRId64 " o_dt=%" PRId64
      " do_dt=%" PRId64 " dqkv_dt=%" PRId64 " qkv_lay=%" PRId64 " o_fmt=%" PRId64 " do_fmt=%" PRId64
      " dqkv_lay=%" PRId64 " qkv_sif=%" PRId64 " do_sif=%" PRId64 " b=%" PRId64 " h=%" PRId64
      " hg=%" PRId64 " dqk=%" PRId64 " dv=%" PRId64 " sq=%" PRId64 " skv=%" PRId64 " tq=%" PRId64
      " tkv=%" PRId64 " bb=%" PRId64 " btq=%" PRId64 " btkv=%" PRId64 " npk=%" PRId64 " npv=%" PRId64
      " psk=%" PRId64 " psv=%" PRId64 " mppk=%" PRId64 " mppv=%" PRId64 " bias_b=%" PRId64
      " bias_h=%" PRId64 " bias_sq=%" PRId64 " bias_skv=%" PRId64 "\n",
      pass, hit ? "HIT" : "MISS", thread_seq_id(), static_cast<int>(c.is_training),
      static_cast<int>(c.deterministic), static_cast<int>(c.cuda_graph),
      static_cast<int>(c.return_max_logit), static_cast<int>(c.is_forward),
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
  std::fflush(stderr);
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
constexpr size_t kStageBuckets = 2 * static_cast<size_t>(BuildStage::kCount);
inline StageTiming &stage_timing(bool is_fwd, BuildStage s) {
  static std::array<StageTiming, kStageBuckets> table{};
  const size_t idx =
      (is_fwd ? 0u : 1u) * static_cast<size_t>(BuildStage::kCount) + static_cast<size_t>(s);
  return table[idx];
}

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
      print_counters("SUMMARY");
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
          std::fprintf(stderr,
                       "[FUSED-ATTN-CACHE] %-3s %-22s | calls=%" PRIu64
                       " | time=%9.1f ms | avg=%9.3f ms/call\n",
                       pass, kStageNames[i], n, total_ms, total_ms / n);
        }
      }
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
