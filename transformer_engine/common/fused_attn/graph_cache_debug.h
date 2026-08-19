/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// Fused-attention graph cache diagnostics.
//
// Enable with NVTE_FUSED_ATTN_CACHE_DEBUG=<level>[:<ranks>]. The output format, how
// to read it and the rank suffix are documented for users in docs/envvars.rst; what
// follows is what maintaining this file needs.
//
//   level 1 (events) : one line per event that happens once per distinct cache key
//                      (CREATE_GRAPH, BUILD_PLANS), plus the exit summary block and
//                      its stage timings. Low volume by construction.
//   level 2 (trace)  : adds a line per cache lookup (HIT/MISS, with the normalized
//                      key) and per execution (EXEC). High volume, and it serializes
//                      threads on the stderr lock, which the stage timings are then
//                      measured under -- no timed region writes to stderr, so they
//                      stay sound, but they read a little high.
//
// Counters are kept per build site -- f16/fp8 crossed with fwd/bwd -- since one
// process can drive both backends, and every event name is also the counter column
// it increments. What the columns mean, the identities they satisfy and the ratios
// worth reading are with the counter definitions below.
//
// One level-1 training step, line prefixes and trailing columns elided:
//
//   tid=0   dev=0   | f16 fwd CREATE_GRAPH | hit=0, miss=1, create_graph=1, ...
//   ===== summary begin =====
//   tid=0   dev=0   | f16 fwd | hit=5, miss=1, create_graph=1, ...
//   tid=1   dev=0   | f16 bwd | hit=4, build_plans=1, exec=1, ...
//   tid=all dev=all | f16 fwd | hit=5, miss=1, create_graph=1, ...
//   f16 fwd build_plans            | calls=1 | time=  262.104 ms/call
//   ===== summary end =====
//
// Rows for a site a thread never reached are left out rather than zeroed, which is
// why tid=1 has a backward row and no forward one: in a PyTorch step the forward and
// the backward's support probe run on the main thread, and the backward itself on the
// autograd thread, which finds the graph that probe left behind. That split is why
// the build identities hold on the totals rows and not on any single thread's.
// ============================================================================

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

// Verbosity level parsed once from NVTE_FUSED_ATTN_CACHE_DEBUG (0=off, 1=events,
// 2=trace). Single read at startup, cached; when unset every call site pays one
// cached-flag check and nothing else.
inline int debug_level() {
  static const int lvl = [] {
    const char *e = std::getenv("NVTE_FUSED_ATTN_CACHE_DEBUG");
    if (e == nullptr || e[0] == '\0' || e[0] == '0') return 0;
    const int v = std::atoi(e);  // stops at the optional ":<ranks>" suffix
    return v > 0 ? v : 1;        // any non-empty, non-"0" value enables at least level 1
  }();
  return lvl;
}

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

// Whether this process emits diagnostics. Every rank writes to the same stderr, and under
// data/tensor parallelism they run identical shapes, so emitting from all of them multiplies the
// volume by the world size to say the same thing. Hence rank 0 only by default, overridable with
// the ":<ranks>" suffix. Context parallelism is the case worth overriding for: the ranks run
// different subsets of the per-step regimes, so their build counts genuinely differ.
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

// Diagnostics are on at level >= 1, and only for the selected ranks. Unselected ranks skip the
// counters too, so they pay nothing beyond this check. Cached in its own flag rather than
// recomputed from the two above, so that the check every call site makes -- the per-lookup path
// included -- reads one initialized-once static instead of two. Both inputs are fixed for the
// life of the process.
inline bool enabled() {
  static const bool on = debug_level() >= 1 && rank_selected();
  return on;
}

// Per-lookup / per-exec trace lines are gated behind level >= 2.
inline bool trace_enabled() { return debug_level() >= 2; }

// Names the emitting rank, without which the ranks sharing one stderr would be indistinguishable.
// A run whose launcher exports no rank is left untagged rather than falling back to a pid, an
// OS-level identifier only being useful for correlating against a profiler. The tag carries its
// own trailing separator, so the untagged case prints no empty column.
inline const std::string &rank_tag() {
  static const std::string *tag = [] {
    const int rank = launcher_rank();
    if (rank < 0) return new std::string();
    return new std::string("rank=" + std::to_string(rank) + " | ");
  }();
  return *tag;
}

// Short thread IDs (0, 1, 2, ...) in assignment order, not identity: tid=0 is whichever thread
// touched this cache first, and the number means nothing outside this process. It attributes the
// per-thread summary rows and is not meant to be matched against anything external.
inline unsigned thread_seq_id() {
  static std::atomic<unsigned> next{0};
  static thread_local unsigned id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

// Registered at first use. On process exit, prints overall event counters and
// graph build timings.
inline void register_summary_once();

// ============================================================================
// The build site an event came from: f16 or fp8, forward or backward. Every recorder names both
// halves, since the counters are per site -- adding f16's builds into fp8's column would leave a
// run that drove both unable to say which paid for what. A pair of enums rather than the
// "fwd"/"bwd" strings this used to take also turns a mistake at a call site into a compile error.
//
// Backend::F16 is the arbitrary-seqlen backend; the max512 one keeps no graph cache. Pass is
// fused_attn::Pass, from config_and_params.h, so that a recorder and the key it prints share one
// notion of direction.
// ============================================================================

enum class Backend { F16, FP8 };

inline constexpr const char *backend_name(Backend b) { return b == Backend::F16 ? "f16" : "fp8"; }
inline constexpr const char *pass_name(Pass p) { return p == Pass::Fwd ? "fwd" : "bwd"; }

// Backend major, pass minor, so that the two passes of one backend are adjacent -- which is how
// the counter lines and the summary rows present them, one backend at a time.
constexpr size_t kSiteCount = 4;
inline constexpr size_t site_index(Backend b, Pass p) {
  return (b == Backend::F16 ? 0u : 2u) + (p == Pass::Fwd ? 0u : 1u);
}

// ============================================================================
// Cache event counters, one block per build site. Each name is both the event tag on the line
// that records it and the column carrying its running total:
//   - create_graph: a graph created and cached for a miss, only as far as check_support().
//   - build_plans: a cached graph finished with graph.build_plans(), the kernel compilation that
//     create_graph deferred. At most one per create_graph, paid by that graph's first execution
//     rather than by the probe that built it.
//   - exec: a graph execution call with valid runtime tensors.
//   - hit: a lookup answered from the cache. Need not lead to an exec -- it can be a backend
//     availability check, or the workspace-sizing call of nvte_fused_attn_fwd/bwd, which has no
//     runtime tensors to run with.
//   - miss: a lookup the cache did not answer; triggers a graph build.
//
// Identities, holding by construction, so a violation is a bug in the cache or in the counting
// rather than something the workload did:
//   - hit + miss = every lookup, one per entry into lookup_or_cache_graph, which makes it the
//     denominator for everything below.
//   - miss >= create_graph, the difference being builds that threw, whether cuDNN refused the graph
//     or could not reach a verdict. Nothing is cached for those, so this is the only place a
//     refusal shows up; its reason goes to the framework instead.
//   - create_graph >= build_plans, the gap being graphs a probe built that nothing has run.
//   - exec > 0 implies build_plans > 0, every site calling build_plans() ahead of the
//     workspace-sizing return, itself ahead of record_exec. Read backwards: a workspace-sizing
//     call pays build_plans and never exec.
//   - Both build identities belong to the totals rows, not to one thread's: the thread that builds
//     a graph need not compile its plans, and a PyTorch step splits exactly that way.
//   - Per-thread rows sum column by column to "tid=all dev=all", and the per-backend rows of one
//     pass to that pass's all-backends row.
//   - A lost build race disturbs none of the above -- the loser records its own miss and its own
//     create_graph, and the once_flag still permits one build_plans -- but it does break reading
//     create_graph as the number of graphs cached.
//   - Stage timing calls fall along validate >= build_operation_graph >= create_execution_plans >=
//     check_support, each drop being the builds that ended at the stage before, which localizes
//     where cuDNN refuses rather than only how long refusing took.
//   - The build_plans timing row can show more calls than the build_plans column, the difference
//     being plan builds that threw: the timer records while unwinding, the counter only on return.
//
// Signatures, workload-dependent, so read rather than asserted:
//   - After warmup only hit and exec should move; a late create_graph means something varies per
//     step that need not.
//   - Several hits per exec is normal, since selection, workspace sizing and execution all look
//     the same key up; what matters is that the ratio stays flat.
//   - exec / create_graph is the amortization figure, and a lower bound at that, a lost race adding
//     a build without a graph. Single digits after a long run means the cache is not earning its
//     keep.
//   - miss climbing while create_graph stays put is a configuration cuDNN keeps refusing, each query
//     paying a discarded build. It also says this site never runs fused, making it the pair to read
//     when attention is slower than expected and nothing raised an error.
//   - miss climbing without settling means the key space is not closing, and since the cache is
//     unbounded, every distinct key is held for the life of the process.
//   - A build count that looks doubled on a multi-device process usually is not: device_id is part
//     of the key, so the same shape on two devices is two entries. Read the dev column.
//   - Two MISS lines with the same key, create_graph above the number of distinct keys, is that lost
//     race: wasted work rather than a bug, worth chasing only if it repeats.
//   - A level-2 trace is the set of lookups, not their order, the line being written after the
//     cache lock is dropped.
// ============================================================================

struct EventCounters {
  std::atomic<uint64_t> create_graph{0};
  std::atomic<uint64_t> build_plans{0};
  std::atomic<uint64_t> exec{0};
  std::atomic<uint64_t> hit{0};
  std::atomic<uint64_t> miss{0};
};

inline EventCounters &counters(Backend b, Pass p) {
  static std::array<EventCounters, kSiteCount> table{};
  return table[site_index(b, p)];
}

// One counter block read out into plain values, so the summary can sum blocks for its per-backend
// and all-backends rows. The columns are not read as one indivisible operation, which nothing here
// wants: the summary runs at exit, after the writing threads are done, and an event line is a
// snapshot of a moving count by nature.
struct CounterSnapshot {
  uint64_t create_graph = 0;
  uint64_t build_plans = 0;
  uint64_t exec = 0;
  uint64_t hit = 0;
  uint64_t miss = 0;

  CounterSnapshot &operator+=(const CounterSnapshot &other) {
    create_graph += other.create_graph;
    build_plans += other.build_plans;
    exec += other.exec;
    hit += other.hit;
    miss += other.miss;
    return *this;
  }

  // Whether this block saw nothing at all, which is what lets the summary leave out the rows
  // for a backend the run never used rather than printing zeros for it.
  bool empty() const { return (create_graph | build_plans | exec | hit | miss) == 0; }
};

inline CounterSnapshot snapshot(const EventCounters &c) {
  CounterSnapshot s;
  s.create_graph = c.create_graph.load(std::memory_order_relaxed);
  s.build_plans = c.build_plans.load(std::memory_order_relaxed);
  s.exec = c.exec.load(std::memory_order_relaxed);
  s.hit = c.hit.load(std::memory_order_relaxed);
  s.miss = c.miss.load(std::memory_order_relaxed);
  return s;
}

// Per-thread counters, one block per build site, so the summary can break every column down by
// thread and backend: in the single-process context-parallel case each device is driven by its own
// thread, and under PyTorch this separates the main thread from the autograd one.
//
// `device` is the device this thread last drove, restamped on every event. Event lines print the
// live current device instead, which is exact; this exists for the per-thread summary rows, written
// at exit by whichever thread is exiting, which cannot ask the recorded thread what it was doing.
struct ThreadCounters {
  unsigned tid = 0;
  std::atomic<int> device{-1};
  std::array<EventCounters, kSiteCount> sites;
};

// The registry and its mutex are heap-allocated and deliberately never freed. Static destructors
// and atexit handlers run as one sequence in reverse order of construction, and this registry is
// built lazily, so it can be constructed *after* the summary handler is registered -- and would
// then be destroyed *before* it runs, leaving the handler to lock a destroyed mutex and walk a
// destroyed vector. Leaking removes the ordering question, at a cost of one mutex and one vector.
inline std::mutex &thread_registry_mutex() {
  static std::mutex *m = new std::mutex();
  return *m;
}
inline std::vector<ThreadCounters *> &thread_registry() {
  static std::vector<ThreadCounters *> *v = new std::vector<ThreadCounters *>();
  return *v;
}

// This thread's counter block, leaked for a related but distinct reason: a worker thread can exit
// long before the process does, while the registry holds a pointer to its block for the exit
// summary. Tying the block's lifetime to the thread would leave that pointer dangling.
inline ThreadCounters &thread_counters() {
  static thread_local ThreadCounters *tc = [] {
    auto *p = new ThreadCounters();
    p->tid = thread_seq_id();
    // Stamped here as well as on every event, so a thread that only ever hits the cache -- never
    // reaching print_counters() at level 1 -- still names a device rather than the -1 it began at.
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

// Format one counter block -- one pass of one backend -- as one line. One pass rather than both
// because a line carrying the forward and backward columns together ran past 300 characters and
// wrapped in most terminals; the two passes are adjacent rows instead.
//
// `tid_field` and `dev_field` are whole columns, e.g. "tid=3" and "dev=0". The totals rows pass
// "tid=all" and "dev=all", since those counters are summed across whatever the process drove and
// naming one thread or device would be a lie.
//
// `label` is the build site, "f16 fwd", plus the event name on an event line, and arrives padded to
// the width its own kind of line uses: 20 characters for an event line, 7 for a summary row.
// Deliberately not one width for both -- sharing it would put twelve blank columns on every summary
// row to align the scattered event lines against a block that is delimited and read on its own.
//
// The thread and device come first, so every line, level-2 trace lines included, shares one prefix
// to read down. What the columns mean and the identities they satisfy are with the definitions
// above.
inline std::string format_counter_line(const char *tid_field, const char *dev_field,
                                       const char *label, const CounterSnapshot &c) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "[FUSED-ATTN-CACHE] %s%-7s %-7s | %s | hit=%4" PRIu64 ", miss=%4" PRIu64
                ", create_graph=%4" PRIu64 ", build_plans=%4" PRIu64 ", exec=%4" PRIu64 "\n",
                rank_tag().c_str(), tid_field, dev_field, label, c.hit, c.miss, c.create_graph,
                c.build_plans, c.exec);
  return std::string(buf);
}

// One event line, from the thread the event happened on, carrying the running totals of the build
// site that raised it. The device is read live rather than remembered, so it is the device this
// event was actually issued against, and is recorded on the thread's block on the way past for
// the benefit of the exit summary.
inline void print_counters(Backend b, Pass p, const char *event) {
  const int device = cuda::current_device();
  thread_counters().device.store(device, std::memory_order_relaxed);
  char label[32];
  char tid_field[16];
  char dev_field[16];
  // The event name is padded to the longest of them, so that the counters of one event line fall
  // where the next one's do.
  std::snprintf(label, sizeof(label), "%s %s %-12s", backend_name(b), pass_name(p), event);
  std::snprintf(tid_field, sizeof(tid_field), "tid=%u", thread_seq_id());
  std::snprintf(dev_field, sizeof(dev_field), "dev=%d", device);
  const std::string line =
      format_counter_line(tid_field, dev_field, label, snapshot(counters(b, p)));
  std::fputs(line.c_str(), stderr);
  std::fflush(stderr);
}

// A graph created, taken through check_support() and cached. Call after that, from the miss path
// that did it -- after, because a graph cuDNN refuses throws instead of arriving here, which is
// what makes miss - create_graph the count of refused builds.
inline void record_graph_created(Backend b, Pass p) {
  if (!enabled()) return;
  register_summary_once();
  counters(b, p).create_graph.fetch_add(1, std::memory_order_relaxed);
  thread_counters(b, p).create_graph.fetch_add(1, std::memory_order_relaxed);
  print_counters(b, p, "CREATE_GRAPH");
}

// The graph.build_plans() a create_graph deferred, now completed. Call from inside the
// std::call_once that runs it, and after the call returns rather than before: it throws without
// setting the once_flag, leaving a later execution to retry, so counting on the way out keeps this
// a count of graphs that reached a runnable state.
inline void record_plans_built(Backend b, Pass p) {
  if (!enabled()) return;
  register_summary_once();
  counters(b, p).build_plans.fetch_add(1, std::memory_order_relaxed);
  thread_counters(b, p).build_plans.fetch_add(1, std::memory_order_relaxed);
  print_counters(b, p, "BUILD_PLANS");
}

inline void record_exec(Backend b, Pass p) {
  if (!enabled()) return;
  register_summary_once();
  counters(b, p).exec.fetch_add(1, std::memory_order_relaxed);
  thread_counters(b, p).exec.fetch_add(1, std::memory_order_relaxed);
  // The per-exec line fires on every execution; keep it out of the level-1 path.
  if (!trace_enabled()) return;
  print_counters(b, p, "EXEC");
}

// What a lookup found: an entry, or nothing.
enum class LookupResult { Miss, Hit };

// The column a lookup lands in. Written as a switch with no default so that adding an outcome
// fails to compile here rather than being silently counted as a miss.
inline std::atomic<uint64_t> &lookup_column(EventCounters &c, LookupResult result) {
  switch (result) {
    case LookupResult::Hit:
      return c.hit;
    case LookupResult::Miss:
      break;
  }
  return c.miss;
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

// `key` is the normalized cache key -- make_cache_key(pass)'s output, the exact value looked up --
// not the execution config it came from. HIT/MISS is decided by comparing keys, so a trace of
// anything else cannot explain its own outcome: the pre-normalization config would show identical
// lines with opposite outcomes, and differing lines that both hit. Diffing two MISS lines here
// names exactly the fields responsible for the extra build.
//
// The cost is that overwritten fields are no longer visible in their original form: attn_scale
// reads 1, ragged num_tokens read 0, max_seqlen and batch_size read their bucketed values.
inline void record_cache_lookup(Backend b, Pass p, LookupResult result,
                                const FusedAttnConfig &key) {
  if (!enabled()) return;
  register_summary_once();
  lookup_column(counters(b, p), result).fetch_add(1, std::memory_order_relaxed);
  lookup_column(thread_counters(b, p), result).fetch_add(1, std::memory_order_relaxed);
  // The per-lookup config dump is the highest-volume line (one per cache lookup);
  // keep it out of the level-1 path and off the stderr lock unless tracing.
  if (!trace_enabled()) return;
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %stid=%-3u dev=%-3d | %-3s %-3s %-12s | train=%d det=%d cg=%d "
      "maxlogit=%d fwd=%d "
      "mask=%" PRId64 " bias=%" PRId64 " wl=%" PRId64 " wr=%" PRId64 " brd=%d softmax=%" PRId64
      " scale_mode=%" PRId64 " dropout=%g attn_scale=%g qkv_dt=%" PRId64 " o_dt=%" PRId64
      " do_dt=%" PRId64 " dqkv_dt=%" PRId64 " qkv_lay=%" PRId64 " o_fmt=%" PRId64 " do_fmt=%" PRId64
      " dqkv_lay=%" PRId64 " qkv_sif=%" PRId64 " do_sif=%" PRId64 " b=%" PRId64 " h=%" PRId64
      " hg=%" PRId64 " dqk=%" PRId64 " dv=%" PRId64 " sq=%" PRId64 " skv=%" PRId64 " tq=%" PRId64
      " tkv=%" PRId64 " bb=%" PRId64 " btq=%" PRId64 " btkv=%" PRId64 " npk=%" PRId64
      " npv=%" PRId64 " psk=%" PRId64 " psv=%" PRId64 " mppk=%" PRId64 " mppv=%" PRId64
      " bias_b=%" PRId64 " bias_h=%" PRId64 " bias_sq=%" PRId64 " bias_skv=%" PRId64 "\n",
      rank_tag().c_str(), thread_seq_id(), key.device_id, backend_name(b), pass_name(p),
      lookup_name(result), static_cast<int>(key.is_training), static_cast<int>(key.deterministic),
      static_cast<int>(key.cuda_graph), static_cast<int>(key.return_max_logit),
      static_cast<int>(key.check_for_forward_support), static_cast<int64_t>(key.attn_mask_type),
      static_cast<int64_t>(key.bias_type), static_cast<int64_t>(key.window_size_left),
      static_cast<int64_t>(key.window_size_right), static_cast<int>(key.bottom_right_diagonal),
      static_cast<int64_t>(key.softmax_type), static_cast<int64_t>(key.scaling_mode),
      static_cast<double>(key.dropout), static_cast<double>(key.attn_scale),
      static_cast<int64_t>(key.qkv_dtype), static_cast<int64_t>(key.o_dtype),
      static_cast<int64_t>(key.do_dtype), static_cast<int64_t>(key.dqkv_dtype),
      static_cast<int64_t>(key.qkv_layout), static_cast<int64_t>(key.o_format),
      static_cast<int64_t>(key.do_format), static_cast<int64_t>(key.dqkv_layout),
      static_cast<int64_t>(key.qkv_scale_inv_format), static_cast<int64_t>(key.do_scale_inv_format),
      static_cast<int64_t>(key.batch_size), static_cast<int64_t>(key.num_attn_heads),
      static_cast<int64_t>(key.num_gqa_groups), static_cast<int64_t>(key.head_dim_qk),
      static_cast<int64_t>(key.head_dim_v), static_cast<int64_t>(key.max_seqlen_q),
      static_cast<int64_t>(key.max_seqlen_kv), static_cast<int64_t>(key.num_tokens_q),
      static_cast<int64_t>(key.num_tokens_kv), static_cast<int64_t>(key.bucketed_batch_size),
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
// Which stage dominates determines what to do about a slow build: time in
// `check_support` and `build_plans` is heuristic selection and kernel compilation,
// largely intrinsic to the shape, while time in `validate` or
// `build_operation_graph` is graph-construction cost on our side. One duration per
// build cannot make that distinction.
//
// Each stage is wrapped where it is called, in graph_cache.h, and accumulates into
// the table below under its build site. Only sums are kept, so the summary can
// report a mean and nothing else -- and since a build happens once per distinct
// cache key, those calls span different shapes rather than repeating one. Read a
// stage mean as where build time goes in aggregate, not as any one build's cost.
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

// Bucketed by build site, so the summary can report the cost of each stage separately for each
// backend and pass -- an fp8 build and an f16 build are different work, and averaging them
// together would describe neither. Unlike the thread registry above, this table needs no leak to
// outlive the exit handler that reads it: it holds nothing but atomics, so it is trivially
// destructible and no destructor is registered for it at all.
constexpr size_t kStageBuckets = kSiteCount * static_cast<size_t>(BuildStage::kCount);
inline StageTiming &stage_timing(Backend b, Pass p, BuildStage s) {
  static std::array<StageTiming, kStageBuckets> table{};
  const size_t idx =
      site_index(b, p) * static_cast<size_t>(BuildStage::kCount) + static_cast<size_t>(s);
  return table[idx];
}

// Times one stage: clock read in the constructor, accumulated in the destructor.
// Recording on scope exit rather than at an explicit stop() keeps a failing stage
// measurable, since `build_plans` throws through NVTE_CHECK_CUDNN_FE and the
// destructor still runs while unwinding, so a build that dies there contributes its
// time instead of vanishing. `on` is latched at construction rather than re-tested in
// the destructor, so the destructor can never accumulate against an unset `start`.
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

// Record how long `fn` takes as `stage` of the given build site. Unlike the record_* helpers above
// this wraps the work rather than reporting on work already done, which is the point: preferred
// over a ScopedBuildTimer at the call site because the measured region is exactly the call passed
// in, so surrounding work cannot drift into it as that code changes.
template <typename Fn>
inline void record_time(Backend b, Pass p, BuildStage stage, Fn &&fn) {
  ScopedBuildTimer scoped(b, p, stage);
  fn();
}

// ============================================================================
// Summary: on process exit, print cache event counters and graph build timings.
// ============================================================================
inline void register_summary_once() {
  static const bool registered = [] {
    std::atexit([] {
      if (!enabled()) return;
      // Built in memory and emitted with one write, so that concurrently-exiting
      // processes (one per rank under torchrun) stay grouped rather than interleaving.
      std::string block;
      block += "[FUSED-ATTN-CACHE] " + rank_tag() + "===== summary begin =====\n";
      constexpr Backend kBackends[] = {Backend::F16, Backend::FP8};
      // A backend the run never reached is left out rather than reported as a row of zeros.
      size_t active_backends = 0;
      for (const Backend b : kBackends) {
        if (!snapshot(counters(b, Pass::Fwd)).empty() ||
            !snapshot(counters(b, Pass::Bwd)).empty()) {
          ++active_backends;
        }
      }
      // Per-thread breakdown (sorted by tid), one row per build site that thread drove, with
      // unreached sites left out for the same reason an unused backend is.
      {
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
          for (const Backend b : kBackends) {
            for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
              const CounterSnapshot c = snapshot(tc->sites[site_index(b, p)]);
              if (c.empty()) continue;
              // No padding: a site name is exactly the width of the column on a summary row.
              char label[32];
              std::snprintf(label, sizeof(label), "%s %s", backend_name(b), pass_name(p));
              block += format_counter_line(tid_field, dev_field, label, c);
            }
          }
        }
      }
      // Totals last, so they read as the sum of the per-thread rows above: one row per build site,
      // then a row per pass across the backends only when the run used more than one, since with a
      // single backend those would repeat the rows above verbatim.
      CounterSnapshot all_fwd;
      CounterSnapshot all_bwd;
      for (const Backend b : kBackends) {
        for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
          const CounterSnapshot c = snapshot(counters(b, p));
          (p == Pass::Fwd ? all_fwd : all_bwd) += c;
          if (c.empty()) continue;
          char label[32];
          std::snprintf(label, sizeof(label), "%s %s", backend_name(b), pass_name(p));
          block += format_counter_line("tid=all", "dev=all", label, c);
        }
      }
      if (active_backends > 1) {
        for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
          const CounterSnapshot &c = (p == Pass::Fwd ? all_fwd : all_bwd);
          if (c.empty()) continue;
          char label[32];
          std::snprintf(label, sizeof(label), "all %s", pass_name(p));
          block += format_counter_line("tid=all", "dev=all", label, c);
        }
      }
      for (const Backend b : kBackends) {
        for (const Pass p : {Pass::Fwd, Pass::Bwd}) {
          for (int i = 0; i < static_cast<int>(BuildStage::kCount); ++i) {
            const BuildStage s = static_cast<BuildStage>(i);
            const StageTiming &t = stage_timing(b, p, s);
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
      block += "[FUSED-ATTN-CACHE] " + rank_tag() + "===== summary end =====\n";
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
