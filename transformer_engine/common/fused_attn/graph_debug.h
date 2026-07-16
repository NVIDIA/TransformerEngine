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
// probes) to detect redundant graph construction.
//
// Enable at runtime with:  export NVTE_FUSED_ATTN_GRAPH_DEBUG=1
// A running "BUILD" line is printed whenever a new graph is constructed, and a
// "SUMMARY" line with final totals is printed at process exit.
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

namespace transformer_engine {
namespace fused_attn_graph_debug {

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

inline bool enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_GRAPH_DEBUG");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return on;
}

inline void dump(const char *event) {
  std::fprintf(stderr,
               "[GRAPH-DEBUG] %-10s | fwd built=%llu exec=%llu | bwd built=%llu exec=%llu\n", event,
               static_cast<unsigned long long>(fwd_built().load()),
               static_cast<unsigned long long>(fwd_exec().load()),
               static_cast<unsigned long long>(bwd_built().load()),
               static_cast<unsigned long long>(bwd_exec().load()));
  std::fflush(stderr);
}

inline void register_summary_once() {
  static const bool registered = [] {
    std::atexit([] {
      if (enabled()) dump("SUMMARY");
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

}  // namespace fused_attn_graph_debug
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_FUSED_ATTN_GRAPH_DEBUG_H_
