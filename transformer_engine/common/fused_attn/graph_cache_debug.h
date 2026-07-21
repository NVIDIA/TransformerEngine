/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// Fused-attention graph-cache diagnostics.
//
// Lightweight, opt-in instrumentation for the cuDNN fused-attention graph cache. All output is
// gated behind an env switch and costs ~one cached-bool branch per fwd/bwd launch when off, so it
// is safe to leave compiled in for production. Enable at runtime with:
//   export NVTE_FUSED_ATTN_CACHE_DEBUG=1
//
// What it reports (all lines prefixed "[FUSED-ATTN-CACHE]"):
//   - "BUILD" line whenever a new graph is constructed, plus a "SUMMARY" line at process exit with
//     total graph builds vs. executions (fwd/bwd). Rebuilds >> executions => redundant construction
//     (a make_cache_key / operator< that is missing a field, or a cache that is not being shared).
//   - "HIT"/"MISS" line per cache lookup with the full (pre-normalization) config key, to diagnose
//     stale-cache reuse: a HIT means two configs compared equal under operator<, so the field that
//     distinguishes a wrongly-reused graph is one make_cache_key() normalized away or operator<
//     omits -- diff a wrong HIT against the earlier BUILD to find it.
//   - "thd ... path=legacy|direct" per THD (ragged) lookup and a "THD-PATH" summary, showing which
//     impl path (bucketed batch vs. real cu_seqlens) the graph was built for. Low builds/lookups on
//     the legacy path means batch bucketing is collapsing distinct batch sizes onto shared graphs.
//   - Every line is tagged with a short per-thread id (tid=N) so cross-thread rebuilds of an
//     identical key are visible.
//
// Separately, force every lookup to miss (never reuse a cached graph) with:
//   export NVTE_FUSED_ATTN_DISABLE_CACHE=1
// If a suite that fails with the cache enabled passes with it disabled, the bug is stale-cache
// reuse (an incomplete make_cache_key / operator<).
//
// ----------------------------------------------------------------------------
// Reference numbers from earlier, heavier instrumentation (FE build-stage timing + cached-graph
// host-memory footprint), which was removed to keep this header lean. Collected by running
// tests/pytorch/attention/test_attention.py on GB200; each stage is invoked on the order of 2000
// times over the run.
//
//   FE build pipeline is dominated by build_plans() (cuDNN plan compilation / autotune):
//     stage                    avg/call     share of build cost
//     validate                 0.020 ms     ~0%   (was a static bool check previously)
//     build_operation_graph    1.828 ms     ~0.3%
//     create_execution_plans   2.163 ms     ~0.3%
//     check_support            0.021 ms     ~0%
//     build_plans            618.673 ms     >99%  (dominates total build time)
//   Note: avg/call is a full-suite mean; build_plans in particular scales with problem size and
//   varies widely from call to call, so treat ~600 ms as an order-of-magnitude figure, not a
//   constant.
//   => The "real check_support" availability probe is essentially free; the entire expense is
//      plan compilation, which only happens on a cache MISS. This is exactly what the graph cache +
//      make_cache_key() normalization exist to avoid, so cache correctness (not probe cost) is what
//      matters for performance.
//
//   Cached-graph host memory (serialized graph size; a proxy for the plan/engine/tensor metadata
//   each built graph holds -- device workspace is separate, sized per execute()):
//     pass   entries   graphs   avg/graph   total
//     fwd      670      1224     189.5 KB    ~232 MB
//     bwd      473       757     300.0 KB    ~227 MB
//   => ~190 KB (fwd) / ~300 KB (bwd) per distinct config; a long-lived process that sees many
//      distinct shapes can accumulate hundreds of MB of cached graph metadata. Worth remembering if
//      cache growth (rather than build time) ever becomes the concern.
// ----------------------------------------------------------------------------
// ============================================================================

#ifndef TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
#define TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "config_and_params.h"  // for FusedAttnConfig field dump

namespace transformer_engine {
namespace fused_attn {
namespace graph_cache_debug {

// Short, stable per-thread id (0, 1, 2, ...) assigned on first use. Tagging every lookup with its
// thread id makes cross-thread rebuilds of an identical key visible.
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
// "builds" counts the lookups that actually constructed a new graph. A low builds/lookups ratio on
// the legacy path is the visible sign that batch bucketing is collapsing distinct batch sizes onto
// shared graphs.
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
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

inline void dump(const char *event) {
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %-10s | tid=%u | fwd built=%llu exec=%llu | bwd built=%llu exec=%llu\n",
      event, thread_seq_id(), static_cast<unsigned long long>(fwd_built().load()),
      static_cast<unsigned long long>(fwd_exec().load()),
      static_cast<unsigned long long>(bwd_built().load()),
      static_cast<unsigned long long>(bwd_exec().load()));
  std::fflush(stderr);
}

inline void dump_thd_summary() {
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] THD-PATH   | legacy lookups=%llu builds=%llu | direct lookups=%llu builds=%llu\n",
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

// Returns true when the graph cache should be bypassed (every lookup treated as a miss so a fresh
// graph is built each call). Gated by NVTE_FUSED_ATTN_DISABLE_CACHE.
inline bool cache_disabled() {
  static const bool off = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_DISABLE_CACHE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return off;
}

// Logs one graph-cache lookup with its outcome (HIT/MISS) and the *real* (pre-normalization) config
// fields. A std::map HIT means the two configs compare equal under operator<, so the field that
// actually distinguishes a wrongly-reused graph is one that make_cache_key() normalized away or
// that operator< omits -- pass the real cfg (not the normalized cache key) here so that difference
// is visible when diffing a wrong HIT against the earlier BUILD that created the reused graph.
inline void note_cache_lookup(const char *pass, bool hit, const FusedAttnConfig &c) {
  if (!enabled()) return;
  register_summary_once();
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %-3s %-4s%s | tid=%u | train=%d det=%d cg=%d maxlogit=%d fwd=%d mask=%lld bias=%lld "
      "wl=%lld wr=%lld brd=%d softmax=%lld scale_mode=%lld dropout=%g attn_scale=%g "
      "qkv_dt=%lld o_dt=%lld do_dt=%lld dqkv_dt=%lld qkv_lay=%lld o_fmt=%lld do_fmt=%lld "
      "dqkv_lay=%lld qkv_sif=%lld do_sif=%lld b=%lld h=%lld hg=%lld dqk=%lld dv=%lld sq=%lld "
      "skv=%lld tq=%lld tkv=%lld bb=%lld btq=%lld btkv=%lld npk=%lld npv=%lld psk=%lld psv=%lld "
      "mppk=%lld mppv=%lld bias_b=%lld bias_h=%lld bias_sq=%lld bias_skv=%lld\n",
      pass, hit ? "HIT" : "MISS",
      (hit && cache_disabled()) ? " [cache-disabled->rebuild]" : "", thread_seq_id(),
      static_cast<int>(c.is_training),
      static_cast<int>(c.deterministic), static_cast<int>(c.cuda_graph),
      static_cast<int>(c.return_max_logit), static_cast<int>(c.is_forward),
      static_cast<long long>(c.attn_mask_type), static_cast<long long>(c.bias_type),
      static_cast<long long>(c.window_size_left), static_cast<long long>(c.window_size_right),
      static_cast<int>(c.bottom_right_diagonal), static_cast<long long>(c.softmax_type),
      static_cast<long long>(c.scaling_mode), static_cast<double>(c.dropout),
      static_cast<double>(c.attn_scale), static_cast<long long>(c.qkv_dtype),
      static_cast<long long>(c.o_dtype), static_cast<long long>(c.do_dtype),
      static_cast<long long>(c.dqkv_dtype), static_cast<long long>(c.qkv_layout),
      static_cast<long long>(c.o_format), static_cast<long long>(c.do_format),
      static_cast<long long>(c.dqkv_layout), static_cast<long long>(c.qkv_scale_inv_format),
      static_cast<long long>(c.do_scale_inv_format), static_cast<long long>(c.batch_size),
      static_cast<long long>(c.num_attn_heads), static_cast<long long>(c.num_gqa_groups),
      static_cast<long long>(c.head_dim_qk), static_cast<long long>(c.head_dim_v),
      static_cast<long long>(c.max_seqlen_q), static_cast<long long>(c.max_seqlen_kv),
      static_cast<long long>(c.num_tokens_q), static_cast<long long>(c.num_tokens_kv),
      static_cast<long long>(c.bucketed_batch_size), static_cast<long long>(c.bucketed_num_tokens_q),
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
  std::fprintf(stderr, "[FUSED-ATTN-CACHE] thd %-3s %-4s | tid=%u | path=%s%s\n", pass,
               hit ? "HIT" : "MISS", thread_seq_id(), legacy ? "legacy" : "direct",
               (hit && built) ? " [cache-disabled->rebuild]" : "");
  std::fflush(stderr);
}

}  // namespace graph_cache_debug
}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
