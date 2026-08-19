/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// ============================================================================
// Fused-attention graph cache diagnostics.
//
// Enable at runtime with NVTE_FUSED_ATTN_CACHE_DEBUG. Two verbosity levels:
//   =1 (events) : low volume. Cache event counters, a BUILD_GRAPH and a BUILD_PLANS line
//                 per build, an UNSUPPORTED line per configuration cuDNN refuses, and the
//                 end-of-run SUMMARY (per backend and per thread, plus a row across the
//                 backends when a run used more than one) with stage timings. Each of these
//                 fires once per distinct cache key, which is what keeps the volume low, and
//                 is enough to diagnose redundant rebuilds and profile build cost.
//   =2 (trace)  : high volume. Additionally emits a per-lookup HIT/MISS/UNSUPPORTED line
//                 with the full shorthand cache key and a per-execution EXEC line. Use only
//                 when you need to see *which* shapes are hitting/missing -- these fire on
//                 every cache lookup and execution, so at suite scale they add I/O and
//                 serialize threads on the stderr lock. No timed region writes to stderr, so
//                 the stage timings stay sound, but they are measured under more contention
//                 than at level 1 and read a little high.
//
// Every line names the build site behind it, "f16" or "fp8" followed by the pass, and the
// counters it carries belong to that backend alone -- the two keep separate columns, so a
// process that drives both can still say which of them built what. Every event name is also
// the counter column it increments, so a line and the totals beside it read with one
// vocabulary. UNSUPPORTED names both a level-1 event and a level-2 lookup outcome, which are
// the two halves of one story: the event records the refusal cuDNN just handed back, and the
// lookup line is a later query answered from that stored refusal instead of by building the
// graph again. Tell them apart by the line shape -- the event line carries counters, the
// lookup line carries the cache key.
//
// An optional ":<ranks>" suffix picks which processes emit, defaulting to rank 0
// so that output does not scale with the world size: "1:all" for every rank,
// "2:0,3" for a specific set. See `rank_selected` for when overriding pays off.
//
// Level 1 on one training step of a supported configuration. Every line begins with
// "[FUSED-ATTN-CACHE] rank=<n> | ", or with just "[FUSED-ATTN-CACHE] " when the launcher
// exports no rank (see `rank_tag`), elided below, and carries the running totals, of which
// only the pass being reported is shown (the counters are printed right-aligned in a fixed
// width, and are abbreviated here):
//
//   f16 fwd BUILD_GRAPH | tid=0   dev=0   | fwd hit_supported=0, miss=1, build_graph=1, ...
//   f16 bwd BUILD_GRAPH | tid=0   dev=0   | fwd ... | bwd hit_supported=0, miss=1, ...
//   f16 fwd BUILD_PLANS | tid=0   dev=0   | fwd hit_supported=1, miss=1, build_plans=1, ...
//   ===== summary begin =====
//   f16 SUMMARY-TID     | tid=0   dev=0   | fwd hit_supported=5, miss=1, build_graph=1, ...
//   f16 SUMMARY-TID     | tid=1   dev=0   | fwd ... | bwd hit_supported=4, build_plans=1, ...
//   f16 SUMMARY         | tid=all dev=all | fwd hit_supported=5, miss=1, build_graph=1, ...
//   f16 fwd check_support          | calls=1 | time=    0.031 ms/call
//   f16 fwd build_plans            | calls=1 | time=  262.104 ms/call
//   ===== summary end =====
//
// The two thread rows are what a PyTorch step really looks like: the forward, and the support
// probe for the backward, run on the main thread, while the backward itself runs on the
// autograd thread and finds the graph that probe left behind. Neither row satisfies
// `build_graph >= build_plans` by itself -- tid=1 compiled the plans of a graph tid=0 built --
// so read the identities off the totals rows rather than the per-thread ones.
//
// The device column matters as soon as one process drives more than one -- device_id is part of
// the cache key, so the same shape on two devices is two entries, and a build count that looks
// doubled is explained by reading which device each BUILD_GRAPH came from.
//
// A support query misses and builds, and every later lookup of that key is a hit_supported --
// including the workspace-sizing call that precedes each execution -- so the hit columns climb
// faster than exec. `build_graph=1, build_plans=1` says that graph went on to be executed;
// `build_graph` above `build_plans` counts graphs built for a query and never run. A refused
// configuration reads `miss=1, unsupported=1, build_graph=0` instead, and stays at one refusal
// however many times it is queried: the repeat queries land in hit_unsupported.
//
// Level 2 adds one line per lookup and per execution, with the key that decided it:
//
//   f16 fwd MISS        | tid=0 dev=0 | train=1 det=0 cg=0 ... b=2 h=16 sq=512 skv=512 ...
//   f16 fwd HIT         | tid=0 dev=0 | train=1 det=0 cg=0 ... b=2 h=16 sq=512 skv=512 ...
//
// where diffing two MISS lines names the fields that cost the extra build.
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
//
// Cached in its own flag rather than recomputed from the two above, so that this -- the check
// every call site makes, on the per-lookup path included -- reads one initialized-once static
// instead of two. Both inputs are fixed for the life of the process, so there is nothing to
// recompute; `rank_selected` is still only reached when the level says diagnostics are on.
inline bool enabled() {
  static const bool on = debug_level() >= 1 && rank_selected();
  return on;
}

// Per-lookup / per-exec trace lines are gated behind level >= 2.
inline bool trace_enabled() { return debug_level() >= 2; }

// Names the emitting rank. Distributed runs put one process per rank on the same stderr, so
// without this the ranks' lines would be indistinguishable. A run whose launcher exports no
// rank is left untagged rather than falling back to a pid: an OS-level identifier is only
// useful for correlating against a profiler or another process, which these logs are not for.
// The tag carries its own trailing separator, so the untagged case prints no empty column.
inline const std::string &rank_tag() {
  static const std::string *tag = [] {
    const int rank = launcher_rank();
    if (rank < 0) return new std::string();
    return new std::string("rank=" + std::to_string(rank) + " | ");
  }();
  return *tag;
}

// More readable, shorter thread IDs (0, 1, 2, ...). These are assignment order, not identity:
// tid=0 is whichever thread touched this cache first, and the number means nothing outside this
// process. It exists to attribute the per-thread SUMMARY rows, not to be matched against
// anything external.
inline unsigned thread_seq_id() {
  static std::atomic<unsigned> next{0};
  static thread_local unsigned id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

// Registered at first use. On process exit, prints overall event counters and
// graph build timings.
inline void register_summary_once();

// ============================================================================
// The build site an event came from: f16 or fp8, forward or backward. Every recorder names
// both halves, because the counters are kept per site rather than per pass. One process can
// drive both backends, and adding f16's builds into the same column as fp8's would leave such
// a run unable to say which of them paid for what.
//
// Backend::F16 is the arbitrary-seqlen f16 backend; the max512 one keeps no graph cache and so
// has nothing to report here. Naming the site with a pair of enums rather than with the
// "fwd"/"bwd" strings this used to take is also what turns a mistake at a call site into a
// compile error instead of an event silently counted against the wrong column.
// ============================================================================

enum class Backend { F16, FP8 };
enum class Pass { Fwd, Bwd };

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
//   - build_graph: a graph built and cached in response to a cache miss. Built only as far
//                  as check_support(), which is all a support probe needs.
//   - build_plans: a cached graph finished with build_plans(), the kernel compilation that
//                  build_graph deferred. At most one per build_graph, and paid by the first
//                  execution of that graph rather than by the probe that built it.
//   - unsupported: a configuration cuDNN refused, now remembered as a negative cache entry.
//                  The other way a miss can end. Counted once per refusal recorded, which is
//                  normally once per distinct refused key; later queries for it are
//                  hit_unsupported.
//   - exec: a graph execution call with valid runtime tensors
//   - hit_supported: a lookup answered from the graph map. May not lead to an exec: it can be
//                  a backend availability check, or the workspace-sizing call of
//                  nvte_fused_attn_fwd/bwd, which has no runtime tensors to run with.
//   - hit_unsupported: a lookup answered from the refusal map -- a key cuDNN has already
//                  refused, replayed instead of rebuilt. Both hit columns are named for the
//                  map that answered them, and `unsupported` above counts the refusals
//                  themselves rather than the queries that replay them.
//   - miss: a lookup neither map answered; triggers a graph build
//
// Identities. These hold by construction, so a violation is a bug in the cache or in the
// counting rather than something the workload did:
//   - hit_supported + hit_unsupported + miss = every lookup, one recorded per entry into
//     build_or_get_cached_graph, which makes it the denominator for everything below.
//   - miss = build_graph + unsupported. A shortfall in either means a build ended in
//     something cuDNN did not state as a verdict on the graph.
//   - build_graph >= build_plans, the gap being graphs a probe built that nothing has run.
//     Eviction grows both rather than closing it: a rebuilt key gets a fresh once_flag.
//   - exec > 0 implies build_plans > 0, every site calling ensure_plans_built ahead of the
//     workspace-sizing return, which is itself ahead of record_exec. The same ordering read
//     backwards: a workspace-sizing call pays build_plans and never exec.
//   - hit_unsupported > 0 implies unsupported > 0, a refusal being replayable only once some
//     earlier call has recorded it.
//   - The two build identities are properties of the totals rows, not of one SUMMARY-TID row:
//     the thread that builds a graph need not be the thread that compiles its plans, and a
//     PyTorch step splits exactly that way across the autograd thread.
//   - A backend's SUMMARY-TID rows sum column by column to its SUMMARY row, and the
//     per-backend rows to the all-backends one.
//   - A lost build race disturbs none of the above: the loser records its own miss and its own
//     build_graph, so both sides of miss = build_graph + unsupported move together, and the
//     entry's once_flag still permits only one build_plans. What a race does break is reading
//     build_graph as the number of graphs cached, two builds being able to stand behind one
//     entry; the same goes for unsupported and the number of keys cuDNN has refused.
//   - In the stage timing rows, calls only fall along the sequence validate >=
//     build_operation_graph >= create_execution_plans >= check_support, each drop being the
//     builds that ended at the stage before -- which localizes where cuDNN refuses, rather
//     than only how long refusing took.
//   - The build_plans timing row can show more calls than the build_plans column counts, the
//     difference being plan builds that threw: the timer records while unwinding, the counter
//     only after the call returns.
//
// Signatures. Workload-dependent, so these are read rather than asserted:
//   - After warmup only hit_supported and exec should move. A build_graph late in a run means
//     something varies per step that need not.
//   - Several hit_supported per exec is normal, since backend selection, workspace sizing and
//     execution all look the same key up; what matters is that the ratio stays flat.
//   - exec / build_graph is the amortization figure, how many executions each built graph
//     served, and a lower bound at that, since a race or an eviction adds a build without
//     adding a graph. Single digits after a long run means the cache is not earning its keep.
//   - hit_unsupported climbing while unsupported stays at one is the negative cache doing its
//     job. It also says this site never runs fused, which makes it the column to reach for
//     when attention is slower than expected and nothing raised an error.
//   - build_graph or unsupported past kCacheCapacity suggests that map has evicted. A hint
//     rather than an identity: a lost build race counts twice against one key.
//   - Two MISS lines carrying the same key, with build_graph above the number of distinct keys,
//     is that lost race. It is wasted work rather than a bug, and worth chasing only if it
//     repeats, which would mean threads are arriving on cold keys together every step.
//   - A level-2 trace is the set of lookups that happened, not the order they happened in: the
//     line is written after the cache lock is dropped, so two threads that raced for it can
//     print in the opposite order.
// ============================================================================

struct EventCounters {
  std::atomic<uint64_t> build_graph{0};
  std::atomic<uint64_t> build_plans{0};
  std::atomic<uint64_t> exec{0};
  std::atomic<uint64_t> hit_supported{0};
  std::atomic<uint64_t> hit_unsupported{0};
  std::atomic<uint64_t> miss{0};
  std::atomic<uint64_t> unsupported{0};
};

inline EventCounters &counters(Backend b, Pass p) {
  static std::array<EventCounters, kSiteCount> table{};
  return table[site_index(b, p)];
}

// One counter block read out into plain values. The summary sums blocks to get its per-backend
// and all-backends rows, atomics cannot be summed, and this is where the reading happens; it
// also keeps the loads out of the formatting. The columns are not read as one indivisible
// operation, which nothing here wants: the summary runs at exit, after the threads that wrote
// them are done, and an event line is a snapshot of a moving count by nature.
struct CounterSnapshot {
  uint64_t build_graph = 0;
  uint64_t build_plans = 0;
  uint64_t exec = 0;
  uint64_t hit_supported = 0;
  uint64_t hit_unsupported = 0;
  uint64_t miss = 0;
  uint64_t unsupported = 0;

  CounterSnapshot &operator+=(const CounterSnapshot &other) {
    build_graph += other.build_graph;
    build_plans += other.build_plans;
    exec += other.exec;
    hit_supported += other.hit_supported;
    hit_unsupported += other.hit_unsupported;
    miss += other.miss;
    unsupported += other.unsupported;
    return *this;
  }

  // Whether this block saw nothing at all, which is what lets the summary leave out the rows
  // for a backend the run never used rather than printing zeros for it.
  bool empty() const {
    return (build_graph | build_plans | exec | hit_supported | hit_unsupported | miss |
            unsupported) == 0;
  }
};

inline CounterSnapshot snapshot(const EventCounters &c) {
  CounterSnapshot s;
  s.build_graph = c.build_graph.load(std::memory_order_relaxed);
  s.build_plans = c.build_plans.load(std::memory_order_relaxed);
  s.exec = c.exec.load(std::memory_order_relaxed);
  s.hit_supported = c.hit_supported.load(std::memory_order_relaxed);
  s.hit_unsupported = c.hit_unsupported.load(std::memory_order_relaxed);
  s.miss = c.miss.load(std::memory_order_relaxed);
  s.unsupported = c.unsupported.load(std::memory_order_relaxed);
  return s;
}

// Per-thread counters, one block per build site, so the summary can break down every column by
// thread and backend. In the single-process context-parallel case each device is driven by its
// own thread, so this reveals which thread built and executed what; under PyTorch it also
// separates the main thread from the autograd thread that runs the backward.
//
// `device` is the device this thread last drove, restamped on every event. The event lines print
// the live current device, which is exact; this exists for the SUMMARY-TID rows, which are
// written at exit by whichever thread is exiting and so cannot ask the recorded thread what it
// was working on. A thread that stays on one device -- which is the arrangement everything here
// is built around, device_id being part of the cache key -- makes the two the same answer.
struct ThreadCounters {
  unsigned tid = 0;
  std::atomic<int> device{-1};
  std::array<EventCounters, kSiteCount> sites;
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
    // Stamped here as well as on every event, so that a thread which only ever hits the cache --
    // and so never reaches print_counters() at level 1 -- still names a device in the summary
    // rather than reporting the -1 it was constructed with.
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

// Format one pair of counter blocks -- the two passes of a single backend -- as one line.
// `label` is the event or summary tag, and names the backend whenever the line speaks for one.
// `tid_field` is the whole thread column, e.g. "tid=3"; the totals rows pass "tid=all" so that
// they cannot be misread as thread 0's row. `dev_field` is the device column and works the same
// way, "dev=all" on a totals row -- those counters are summed across whatever devices the
// process drove, so naming one of them would be a lie.
//
// What the columns mean, the identities they can be asserted against and the ratios worth
// reading are all with the counter definitions above.
inline std::string format_counter_line(const char *label, const char *tid_field,
                                       const char *dev_field, const CounterSnapshot &f,
                                       const CounterSnapshot &b) {
  char buf[768];
  std::snprintf(buf, sizeof(buf),
                "[FUSED-ATTN-CACHE] %s%-19s | %-7s %-7s | fwd hit_supported=%4" PRIu64
                ", hit_unsupported=%4" PRIu64 ", miss=%4" PRIu64 ", build_graph=%4" PRIu64
                ", unsupported=%4" PRIu64 ", build_plans=%4" PRIu64 ", exec=%4" PRIu64
                " | bwd hit_supported=%4" PRIu64 ", hit_unsupported=%4" PRIu64 ", miss=%4" PRIu64
                ", build_graph=%4" PRIu64 ", unsupported=%4" PRIu64 ", build_plans=%4" PRIu64
                ", exec=%4" PRIu64 "\n",
                rank_tag().c_str(), label, tid_field, dev_field, f.hit_supported, f.hit_unsupported,
                f.miss, f.build_graph, f.unsupported, f.build_plans, f.exec, b.hit_supported,
                b.hit_unsupported, b.miss, b.build_graph, b.unsupported, b.build_plans, b.exec);
  return std::string(buf);
}

inline void print_counter_block(const char *label, const char *tid_field, const char *dev_field,
                                const CounterSnapshot &f, const CounterSnapshot &b) {
  const std::string line = format_counter_line(label, tid_field, dev_field, f, b);
  std::fputs(line.c_str(), stderr);
  std::fflush(stderr);
}

// One event line, from the thread the event happened on, carrying the running totals of the
// backend that raised it. The device is read live rather than remembered, so it is the device
// this event was actually issued against, and is recorded on the thread's block on the way past
// for the benefit of the exit summary.
inline void print_counters(Backend b, Pass p, const char *event) {
  const int device = cuda::current_device();
  thread_counters().device.store(device, std::memory_order_relaxed);
  char label[32];
  char tid_field[16];
  char dev_field[16];
  std::snprintf(label, sizeof(label), "%s %s %s", backend_name(b), pass_name(p), event);
  std::snprintf(tid_field, sizeof(tid_field), "tid=%u", thread_seq_id());
  std::snprintf(dev_field, sizeof(dev_field), "dev=%d", device);
  print_counter_block(label, tid_field, dev_field, snapshot(counters(b, Pass::Fwd)),
                      snapshot(counters(b, Pass::Bwd)));
}

// A graph built through check_support() and cached. Call after the build, from the miss
// path that performed it.
inline void record_graph_built(Backend b, Pass p) {
  if (!enabled()) return;
  register_summary_once();
  counters(b, p).build_graph.fetch_add(1, std::memory_order_relaxed);
  thread_counters(b, p).build_graph.fetch_add(1, std::memory_order_relaxed);
  print_counters(b, p, "BUILD_GRAPH");
}

// The build_plans() a build_graph deferred, now completed. Call from inside the std::call_once
// that runs it, after the call returns rather than before: build_plans() throws without
// setting the once_flag, leaving a later execution to retry it, so counting on the way out
// keeps this a count of graphs that reached a runnable state. Like build_graph this fires once
// per distinct cache key, so it stays on the level-1 path.
inline void record_plans_built(Backend b, Pass p) {
  if (!enabled()) return;
  register_summary_once();
  counters(b, p).build_plans.fetch_add(1, std::memory_order_relaxed);
  thread_counters(b, p).build_plans.fetch_add(1, std::memory_order_relaxed);
  print_counters(b, p, "BUILD_PLANS");
}

// A build that cuDNN refused, now remembered as a negative cache entry. Call from the miss path
// that attempted it, in place of record_graph_built(): a refusal and a build are the two ways a
// miss can end, and counting both keeps `miss = build_graph + unsupported` true. Fires once per
// refused key -- later queries for it land in hit_unsupported -- so it stays on the level-1 path.
inline void record_unsupported(Backend b, Pass p) {
  if (!enabled()) return;
  register_summary_once();
  counters(b, p).unsupported.fetch_add(1, std::memory_order_relaxed);
  thread_counters(b, p).unsupported.fetch_add(1, std::memory_order_relaxed);
  print_counters(b, p, "UNSUPPORTED");
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

// What a lookup found. Unsupported is the negative-cache case: a key whose graph cuDNN has
// already refused, so the answer is a remembered refusal rather than a graph.
enum class LookupResult { Miss, Hit, Unsupported };

// The column a lookup lands in, which is the cache map that answered it. Written as a switch
// with no default so that adding an outcome fails to compile here rather than being silently
// counted as a miss.
inline std::atomic<uint64_t> &lookup_column(EventCounters &c, LookupResult result) {
  switch (result) {
    case LookupResult::Hit:
      return c.hit_supported;
    case LookupResult::Unsupported:
      return c.hit_unsupported;
    case LookupResult::Miss:
      break;
  }
  return c.miss;
}

inline const char *lookup_name(LookupResult result) {
  switch (result) {
    case LookupResult::Hit:
      return "HIT";
    case LookupResult::Unsupported:
      return "UNSUPPORTED";
    case LookupResult::Miss:
      break;
  }
  return "MISS";
}

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
inline void record_cache_lookup(Backend b, Pass p, LookupResult result,
                                const FusedAttnConfig &key) {
  if (!enabled()) return;
  register_summary_once();
  // A refusal replayed from the negative cache is counted apart from a graph hit, in
  // hit_unsupported rather than in hit_supported. Both were answered without building
  // anything, which is what the two hit columns have in common; which map answered is the
  // thing worth being able to read off a level-1 summary, since a run whose hits are mostly
  // replayed refusals is not reusing graphs at all.
  lookup_column(counters(b, p), result).fetch_add(1, std::memory_order_relaxed);
  lookup_column(thread_counters(b, p), result).fetch_add(1, std::memory_order_relaxed);
  // The per-lookup config dump is the highest-volume line (one per cache lookup);
  // keep it out of the level-1 path and off the stderr lock unless tracing.
  if (!trace_enabled()) return;
  std::fprintf(
      stderr,
      "[FUSED-ATTN-CACHE] %s%-3s %-3s %-11s | tid=%u dev=%d | train=%d det=%d cg=%d "
      "maxlogit=%d fwd=%d "
      "mask=%" PRId64 " bias=%" PRId64 " wl=%" PRId64 " wr=%" PRId64 " brd=%d softmax=%" PRId64
      " scale_mode=%" PRId64 " dropout=%g attn_scale=%g qkv_dt=%" PRId64 " o_dt=%" PRId64
      " do_dt=%" PRId64 " dqkv_dt=%" PRId64 " qkv_lay=%" PRId64 " o_fmt=%" PRId64 " do_fmt=%" PRId64
      " dqkv_lay=%" PRId64 " qkv_sif=%" PRId64 " do_sif=%" PRId64 " b=%" PRId64 " h=%" PRId64
      " hg=%" PRId64 " dqk=%" PRId64 " dv=%" PRId64 " sq=%" PRId64 " skv=%" PRId64 " tq=%" PRId64
      " tkv=%" PRId64 " bb=%" PRId64 " btq=%" PRId64 " btkv=%" PRId64 " npk=%" PRId64
      " npv=%" PRId64 " psk=%" PRId64 " psv=%" PRId64 " mppk=%" PRId64 " mppv=%" PRId64
      " bias_b=%" PRId64 " bias_h=%" PRId64 " bias_sq=%" PRId64 " bias_skv=%" PRId64 "\n",
      rank_tag().c_str(), backend_name(b), pass_name(p), lookup_name(result), thread_seq_id(),
      key.device_id, static_cast<int>(key.is_training), static_cast<int>(key.deterministic),
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
// measurable: `build_plans` throws through NVTE_CHECK_CUDNN_FE and the destructor
// still runs during unwinding, so a build that dies there contributes its time to
// the failure instead of vanishing from the summary. The four stages before it
// return their status instead of throwing, and are timed the same way for the same
// reason. `on` is latched at construction rather than re-tested in the destructor,
// which is what keeps that symmetric: the destructor can never accumulate against a
// `start` the constructor left unset.
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

// Time `fn` as `stage` of the given build site, named as the record_* helpers above name it.
// Preferred over declaring a ScopedBuildTimer at the call site: the measured region is exactly
// the call passed in, so surrounding work cannot drift into it as that code changes. With
// diagnostics off this costs one cached-flag check, and that is per build rather than per lookup.
template <typename Fn>
inline void timer(Backend b, Pass p, BuildStage stage, Fn &&fn) {
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
      // Build the whole summary in memory and emit it with a single write, so
      // that the blocks of concurrently-exiting processes (one per rank under
      // torchrun) stay grouped instead of interleaving line by line.
      std::string block;
      block += "[FUSED-ATTN-CACHE] " + rank_tag() + "===== summary begin =====\n";
      constexpr Backend kBackends[] = {Backend::F16, Backend::FP8};
      // A backend the run never reached is left out of the summary rather than reported as a
      // row of zeros, so the usual single-backend run reads as it did before this was split.
      size_t active_backends = 0;
      for (const Backend b : kBackends) {
        if (!snapshot(counters(b, Pass::Fwd)).empty() ||
            !snapshot(counters(b, Pass::Bwd)).empty()) {
          ++active_backends;
        }
      }
      // Per-thread breakdown (sorted by tid), one row per backend that thread drove. Useful in
      // the single-process context-parallel case where each device runs on its own thread.
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
            const CounterSnapshot fwd = snapshot(tc->sites[site_index(b, Pass::Fwd)]);
            const CounterSnapshot bwd = snapshot(tc->sites[site_index(b, Pass::Bwd)]);
            if (fwd.empty() && bwd.empty()) continue;
            char label[32];
            std::snprintf(label, sizeof(label), "%s SUMMARY-TID", backend_name(b));
            block += format_counter_line(label, tid_field, dev_field, fwd, bwd);
          }
        }
      }
      // Totals last, so they read as the sum of the per-thread rows above: one row per backend,
      // then a row across the backends only when the run used more than one. With a single
      // backend that row would repeat the one above it verbatim and say nothing extra.
      CounterSnapshot all_fwd;
      CounterSnapshot all_bwd;
      for (const Backend b : kBackends) {
        const CounterSnapshot fwd = snapshot(counters(b, Pass::Fwd));
        const CounterSnapshot bwd = snapshot(counters(b, Pass::Bwd));
        all_fwd += fwd;
        all_bwd += bwd;
        if (fwd.empty() && bwd.empty()) continue;
        char label[32];
        std::snprintf(label, sizeof(label), "%s SUMMARY", backend_name(b));
        block += format_counter_line(label, "tid=all", "dev=all", fwd, bwd);
      }
      if (active_backends > 1) {
        block += format_counter_line("SUMMARY", "tid=all", "dev=all", all_fwd, all_bwd);
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
