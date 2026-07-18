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
//     means batch bucketing is collapsing distinct batch sizes onto shared graphs).
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

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

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

inline void register_summary_once() {
  static const bool registered = [] {
    std::atexit([] {
      if (enabled()) {
        dump("SUMMARY");
        dump_thd_summary();
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

// Returns true when the graph cache should be bypassed (every lookup treated as a
// miss so a fresh graph is built each call). Gated by NVTE_FUSED_ATTN_DISABLE_CACHE.
inline bool cache_disabled() {
  static const bool off = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_DISABLE_CACHE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return off;
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
}

}  // namespace fused_attn_graph_debug
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_FUSED_ATTN_GRAPH_DEBUG_H_
