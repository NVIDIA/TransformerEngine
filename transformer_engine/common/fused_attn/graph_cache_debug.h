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
//                      (CREATE_GRAPH, CACHE_GRAPH, BUILD_PLANS), plus the exit summary
//                      block and its stage timings. Low volume by construction.
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
//   tid=0   dev=0   | f16 fwd CREATE_GRAPH | hit=0, miss=1, create_graph=1, cache_graph=0, ...
//   tid=0   dev=0   | f16 fwd CACHE_GRAPH  | hit=0, miss=1, create_graph=1, cache_graph=1, ...
//   ===== summary begin =====
//   tid=0   dev=0   | f16 fwd | hit=5, miss=1, create_graph=1, cache_graph=1, ...
//   tid=1   dev=0   | f16 bwd | hit=4, build_plans=1, execute=1, ...
//   tid=all dev=all | f16 fwd | hit=5, miss=1, create_graph=1, cache_graph=1, ...
//   f16 fwd build_plans            | calls=1 | time=  262.104 ms/call
//   ===== summary end =====
//
// Those first two lines are one graph, before and after cuDNN was asked to support it, which is why
// a CREATE_GRAPH with no CACHE_GRAPH following it is a refusal -- the one event pattern that says a
// build was paid for and thrown away.
//
// Rows for a site a thread never reached are left out rather than zeroed, which is
// why tid=1 has a backward row and no forward one: in a PyTorch step the forward and
// the backward's support probe run on the main thread, and the backward itself on the
// autograd thread, which finds the graph that probe left behind. That split is why
// the build identities hold on the totals rows and not on any single thread's.
//
// Reading this file: the interface is the four names under "vocabulary" and the six recorders at
// the bottom, and that is everything the rest of the library touches. In between, in namespace
// detail, is what they are built out of, in the order an event travels through it -- the gate, the
// counters, the line, the exit summary. A question about what the output means is answered by the
// counter definitions in the middle; a question about what to call is answered by the bottom.
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

// ============================================================================
// The vocabulary a call site needs: which build site an event came from, and which build stage or
// lookup outcome it is reporting. These four names and the recorders at the bottom of the file are
// the whole interface; everything between them is machinery, in namespace detail.
//
// Backend and Pass are fused_attn's own, from config_and_params.h, so that a recorder and the key
// it prints share one notion of a site; taking the pair rather than the "fwd"/"bwd" strings this
// used to also turns a mistake at a call site into a compile error.
//
// Every recorder names both halves, since the counters are per site -- adding f16's builds into
// fp8's column would leave a run that drove both unable to say which paid for what.
// ============================================================================

inline constexpr const char *backend_name(Backend b) { return b == Backend::F16 ? "f16" : "fp8"; }
inline constexpr const char *pass_name(Pass p) { return p == Pass::Fwd ? "fwd" : "bwd"; }

// The frontend calls that make up a build, in the order they run. `kCount` must stay last: it
// sizes the timing table, and detail::kStageNames is indexed by these values when the summary
// prints, so the two must be kept in the same order.
enum class BuildStage { Validate, BuildOpGraph, CreatePlans, CheckSupport, BuildPlans, kCount };

// What a lookup found: an entry, or nothing.
enum class LookupResult { Miss, Hit };

// ============================================================================
// Machinery: the gate, the counters, the formatting and the exit summary. Nothing outside this
// file names any of it.
//
// Reading order below is the order an event travels: whether to record at all, which site it
// belongs to, the counters it moves, the line it prints, and finally the summary that reports the
// lot at exit.
// ============================================================================
namespace detail {

// ============================================================================
// The gate: whether this process records anything, and how it names itself when it does. Every
// answer here is fixed for the life of the process and read out of an initialized-once static, so
// the check a disabled build pays at each call site is one load and one branch.
// ============================================================================

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

// Diagnostics are on at level >= 1, and only for the ranks the ":<ranks>" suffix selects. Every
// rank writes to the same stderr, and under data/tensor parallelism they run identical shapes, so
// emitting from all of them multiplies the volume by the world size to say the same thing. Hence
// rank 0 only by default. Context parallelism is the case worth overriding for: the ranks run
// different subsets of the per-step regimes, so their build counts genuinely differ.
//
// Both inputs are fixed for the life of the process, so the whole verdict is one initialized-once
// static -- which is all the check every call site makes reads, the per-lookup path included.
// Unselected ranks skip the counters too, so they pay nothing beyond it.
inline bool enabled() {
  static const bool on = [] {
    if (debug_level() < 1) return false;
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
  return on;
}

// The gate on the per-lookup and per-execution trace lines: everything enabled() asks for, level 2
// on top of it. Named for that conjunction, and testing it rather than just the level, so the
// answer holds wherever it is asked -- level 2 alone is true on a rank that emits nothing, which
// would make this read as "trace" on every rank in the job.
inline bool enabled_with_trace() { return enabled() && debug_level() >= 2; }

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
// Indexing the build site an event came from: f16 or fp8, forward or backward.
// ============================================================================

// Backend major, pass minor, so that the two passes of one backend are adjacent -- which is how
// the counter lines and the summary rows present them, one backend at a time.
constexpr size_t kSiteCount = 4;
inline constexpr size_t site_index(Backend b, Pass p) {
  return (b == Backend::F16 ? 0u : 2u) + (p == Pass::Fwd ? 0u : 1u);
}

// ============================================================================
// Cache event counters, one block per build site. Each name is both the event tag on the line
// that records it and the column carrying its running total:
//   - create_graph: a graph constructed for a miss, counted before cuDNN is asked to support it and
//     so regardless of what cuDNN goes on to say about it.
//   - cache_graph: one of those graphs cleared check_support(), so this is the graphs cuDNN agreed
//     to run. Counted on that verdict rather than on the insert that follows it, so it says what
//     cuDNN accepted and not how many entries the map holds; the two differ only when a build race
//     is lost and a supported graph is discarded for the winner's.
//   - build_plans: a cached graph finished with graph.build_plans(), the kernel compilation that
//     cache_graph deferred. At most one per cache_graph, paid by that graph's first execution
//     rather than by the probe that built it.
//   - execute: a graph execution cuDNN accepted, counted once the enqueue returns. Not a completed
//     execution: the work is asynchronous, so a device-side fault is not reflected here.
//   - hit: a lookup answered from the cache. Need not lead to an execution -- it can be a backend
//     availability check, or the workspace-sizing call of nvte_fused_attn_fwd/bwd, which has no
//     runtime tensors to run with.
//   - miss: a lookup the cache did not answer; triggers a graph build.
//
// Identities, holding by construction, so a violation is a bug in the cache or in the counting
// rather than something the workload did:
//   - hit + miss = every lookup, one per call to cache_graph(), which makes it the denominator for
//     everything below. (The function, not the column of the same name -- the column counts the
//     subset of those calls that ended in an entry.)
//   - miss >= create_graph >= cache_graph, where each drop is a build that threw.
//     create_graph - cache_graph is what cuDNN refused, or could not reach a verdict on; nothing is
//     cached for it, so this gap is where a refusal shows up, and the reason itself goes to the
//     framework. miss - create_graph would be a backend refusing a configuration from inside its
//     own build, and no backend does that any more -- TE's own FP8 rules over bias, ALiBi and the
//     recipes it does not serve moved to nvte_get_fused_attn_backend_v2, which answers them with a
//     reason rather than throwing. So this gap should read zero, and the column stays as the thing
//     that says so: a nonzero miss - create_graph means a build threw where none is expected to.
//   - cache_graph >= build_plans, the gap being graphs a probe built that nothing has run.
//   - execute > 0 implies build_plans > 0, every site calling build_plans() ahead of the
//     workspace-sizing return, itself ahead of record_execute. Read backwards: a workspace-sizing
//     call pays build_plans and never execute.
//   - Both build identities belong to the totals rows, not to one thread's: the thread that builds
//     a graph need not compile its plans, and a PyTorch step splits exactly that way.
//   - Per-thread rows sum column by column to "tid=all dev=all", and the per-backend rows of one
//     pass to that pass's all-backends row.
//   - A lost build race disturbs none of the above: the loser records its own miss, create_graph
//     and cache_graph, having built a graph cuDNN did agree to run, and the once_flag still permits
//     one build_plans on the winner's entry. What it costs is a build, which two MISS lines on one
//     key is the way to see.
//   - Stage timing calls fall along validate >= build_operation_graph >= create_execution_plans >=
//     check_support, each drop being the builds that ended at the stage before, which localizes
//     where cuDNN refuses rather than only how long refusing took.
//   - The build_plans timing row can show more calls than the build_plans column, the difference
//     being plan builds that threw: the timer records while unwinding, the counter only on return.
//
// Signatures, workload-dependent, so read rather than asserted. What a column stalling says about
// who rejected a configuration is the user-facing half of this and lives in docs/envvars.rst; what
// follows is what is worth knowing on top of it:
//   - After warmup only hit and execute should move; a late create_graph means something varies per
//     step that need not.
//   - Several hits per execution is normal, since selection, workspace sizing and execution all
//     look the same key up; what matters is that the ratio stays flat.
//   - execute / cache_graph is the amortization figure, and a lower bound at that, a lost race
//     counting a supported graph the cache did not keep. Single digits after a long run means the
//     cache is not earning its keep.
//   - miss climbing without settling means the key space is not closing, and since the cache is
//     unbounded, every distinct key is held for the life of the process.
//   - A build count that looks doubled on a multi-device process usually is not: device_id is part
//     of the key, so the same shape on two devices is two entries. Read the dev column.
//   - Two MISS lines with the same key, cache_graph above the number of distinct keys, is that lost
//     race: wasted work rather than a bug, worth chasing only if it repeats.
//   - A level-2 trace is the set of lookups, not their order, the line being written after the
//     cache lock is dropped.
// ============================================================================

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

// One counter block read out into plain values, so the summary can sum blocks for its per-backend
// and all-backends rows. The columns are not read as one indivisible operation, which nothing here
// wants: the summary runs at exit, after the writing threads are done, and an event line is a
// snapshot of a moving count by nature.
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

  // Whether this block saw nothing at all, which is what lets the summary leave out the rows
  // for a backend the run never used rather than printing zeros for it.
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

// ============================================================================
// The same counters again, per thread, and the registry the exit summary walks to find them.
// ============================================================================

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

// ============================================================================
// Turning a counter block into a line, and getting a line out. One formatter, shared by the event
// lines and the summary rows, so that the two cannot drift into presenting the same columns
// differently, and one writer, so that everything here reaches stderr the same way.
// ============================================================================

// The one place diagnostics reach stderr, and the reason it exists: the first line this process
// writes carries a leading newline. Diagnostics share stderr with whatever the framework is
// printing, and a test runner's progress output has no trailing newline of its own, so without
// this the first line continues someone else's -- which on a level-2 trace line, long enough to
// wrap already, leaves no way to find where it starts. Where the previous output did end cleanly
// the prefix reads as a blank line setting the diagnostics apart from it.
//
// One fwrite per line either way: a rank's summary block is assembled whole precisely so that
// concurrently exiting ranks do not interleave, and the extra allocation buys the same for the one
// line that gets the prefix.
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
                ", create_graph=%4" PRIu64 ", cache_graph=%4" PRIu64 ", build_plans=%4" PRIu64
                ", execute=%4" PRIu64 "\n",
                rank_tag().c_str(), tid_field, dev_field, label, c.hit, c.miss, c.create_graph,
                c.cache_graph, c.build_plans, c.execute);
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
  write_stderr(format_counter_line(tid_field, dev_field, label, snapshot(counters(b, p))));
}

// ============================================================================
// What the recorders at the bottom of the file are made of: moving one column, and naming a
// lookup's outcome.
// ============================================================================

// The body every recorder shares: gate, register the exit summary, and add one to `column`
// in both the process-wide block and this thread's. Returns whether diagnostics are on at all, so
// that a caller can skip building a line nobody will read.
//
// Both blocks or neither. A recorder that moved one and not the other would leave the per-thread
// rows failing to add up to the totals row, which the summary presents as an invariant, and the
// discrepancy would look like a threading bug in the cache rather than a miscount here.
inline bool record_counter(Backend b, Pass p, std::atomic<uint64_t> EventCounters::*column) {
  if (!enabled()) return false;
  register_summary_once();
  (counters(b, p).*column).fetch_add(1, std::memory_order_relaxed);
  (thread_counters(b, p).*column).fetch_add(1, std::memory_order_relaxed);
  return true;
}

// The column a lookup lands in, and the tag naming it. Both are written as a switch with no
// default so that adding an outcome fails to compile here rather than being silently counted as
// a miss.
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

// Indexed by BuildStage when the summary prints, so it must stay in that enum's order and carry
// one name per stage ahead of its kCount sentinel.
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

// ============================================================================
// Summary: on process exit, print cache event counters and graph build timings.
//
// Each section below appends its rows to the block the handler is assembling, in the order they
// are printed: per-thread rows, then totals, then stage timings. Split into named pieces rather
// than written inline because they are read one at a time -- a question about the output is a
// question about one of these -- and because the registration itself is already three constructs
// deep (an initialized-once static holding an atexit handler) before any row logic joins it.
// ============================================================================

// The two backends that keep a cache, in the order every part of the summary walks them.
inline constexpr Backend kSummaryBackends[] = {Backend::F16, Backend::FP8};

// Names one build site for a summary row. No padding: the site name is exactly the width of the
// column there, unlike the event lines, which pad it to keep their counters aligned.
inline std::string site_label(Backend b, Pass p) {
  return std::string(backend_name(b)) + " " + pass_name(p);
}

// How many backends the run actually drove. Decides whether the across-backend rows are worth
// printing: with one backend they would repeat that backend's own rows verbatim.
inline size_t active_backend_count() {
  size_t active = 0;
  for (const Backend b : kSummaryBackends) {
    if (!snapshot(counters(b, Pass::Fwd)).empty() || !snapshot(counters(b, Pass::Bwd)).empty()) {
      ++active;
    }
  }
  return active;
}

// Per-thread breakdown, sorted by tid, one row per build site that thread drove. Sites it never
// reached are left out, for the reason an unused backend is: a row of zeros says nothing.
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

// Totals, printed after the per-thread rows so they read as their sum: one row per build site,
// then one per pass across the backends when the run used more than one. Both come from the
// process-wide counters rather than by adding up the rows above, so the two agreeing is a check
// on the counting rather than an artifact of it.
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
    block += format_counter_line("tid=all", "dev=all", (std::string("all ") + pass_name(p)).c_str(),
                                 c);
  }
}

// Mean time per call for each stage of each build site, skipping stages nothing reached. A mean is
// all the sums kept can support; see the section above for why that is the right figure to read.
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
      // Built in memory and emitted with one write, so that concurrently-exiting
      // processes (one per rank under torchrun) stay grouped rather than interleaving.
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

// ============================================================================
// The recorders: everything a call site calls. Each takes the build site it is reporting for, adds
// one to that site's column, and prints a line when the level asks for it.
//
// Every one of them is called after the event it names, never before, so that a column counts what
// happened rather than what was attempted. That is what gives the gaps between columns their
// meaning: an event that can fail partway -- a build cuDNN refuses, an execution whose setup throws
// first -- leaves the earlier column moved and the later one not.
//
// record_time is the exception, and only because timing cannot be done after the fact: it brackets
// the call it measures, and accumulates while unwinding so that a stage which throws is still
// timed. Its timing rows can therefore outnumber the matching counter column.
//
// Which of them belongs where in the cache's flow is documented on each below and in graph_cache.h
// at the call sites.
// ============================================================================

// A graph constructed for a miss, whatever cuDNN goes on to make of it. Call from the miss path
// that built it, as soon as construction returns and before check_support() is asked. Before,
// because construction is where a backend would refuse a configuration on its own rules, and such
// a build never gets here -- which is what makes miss - create_graph builds that failed on this
// side of cuDNN. No backend does that now, so the gap is there to read as zero.
inline void record_create_graph(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::create_graph)) {
    detail::print_counters(b, p, "CREATE_GRAPH");
  }
}

// A created graph that cleared check_support(), so this counts the graphs cuDNN agreed to run. Call
// as soon as that verdict returns, ahead of the insert: a refused graph throws in between, leaving
// its CREATE_GRAPH unanswered, which is what makes create_graph - cache_graph cuDNN's refusals.
// Deliberately not the insert, so that a lost race reads as the extra build it is rather than as a
// count that disagrees with the size of the cache.
inline void record_cache_graph(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::cache_graph)) {
    detail::print_counters(b, p, "CACHE_GRAPH");
  }
}

// The graph.build_plans() a cache_graph deferred, now completed. Call from inside the
// std::call_once that runs it, and after the call returns rather than before: it throws without
// setting the once_flag, leaving a later execution to retry, so counting on the way out keeps this
// a count of graphs that reached a runnable state.
inline void record_build_plans(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::build_plans)) {
    detail::print_counters(b, p, "BUILD_PLANS");
  }
}

// An execution cuDNN accepted. Call after graph.execute() returns, as with the recorders above, so
// that a graph the surrounding setup never reached is not counted as having run -- the stream set
// and the cu_seqlens conversion kernels sit between the decision to execute and the execution, and
// either can throw.
//
// Accepted is as far as this can go. execute() enqueues on a stream and returns, so a fault the
// device raises later, surfacing at the next synchronization, still leaves the execution counted
// here. There is no synchronous completion point to hook without making the diagnostic change what
// it measures.
//
// Unlike the recorders above, this fires on every execution rather than once per distinct key, so
// its line is held back to level 2 while its column keeps counting.
inline void record_execute(Backend b, Pass p) {
  if (detail::record_counter(b, p, &detail::EventCounters::execute) &&
      detail::enabled_with_trace()) {
    detail::print_counters(b, p, "EXECUTE");
  }
}

// `key` is the normalized cache key -- make_cache_key(pass)'s output, the exact value looked up --
// not the execution config it came from. HIT/MISS is decided by comparing keys, so a trace of
// anything else cannot explain its own outcome: the pre-normalization config would show identical
// lines with opposite outcomes, and differing lines that both hit. Diffing two MISS lines here
// names exactly the fields responsible for the extra build.
//
// The cost is that overwritten fields are no longer visible in their original form: attn_scale
// reads 1, ragged num_tokens read 0, max_seqlen and batch_size read their bucketed values.
//
// This is the one line here not built from counters, so it does not go through
// format_counter_line: which fields it names is FusedAttnConfig::key_debug_string()'s to say,
// alongside the operator< that decides what a key compares on in the first place.
inline void record_hit_miss(Backend b, Pass p, LookupResult result, const FusedAttnConfig &key) {
  // The per-lookup config dump is the highest-volume line (one per cache lookup);
  // keep it out of the level-1 path and off the stderr lock unless tracing.
  if (!detail::record_counter(b, p, detail::lookup_column(result)) ||
      !detail::enabled_with_trace()) {
    return;
  }
  char prefix[128];
  std::snprintf(prefix, sizeof(prefix),
                "[FUSED-ATTN-CACHE] %stid=%-3u dev=%-3d | %-3s %-3s %-12s | ",
                detail::rank_tag().c_str(), detail::thread_seq_id(), key.device_id, backend_name(b),
                pass_name(p), detail::lookup_name(result));
  detail::write_stderr(prefix + key.key_debug_string() + "\n");
}

// Record how long `fn` takes as `stage` of the given build site. Unlike the recorders above this
// wraps the work rather than reporting on work already done, which is the point: the measured
// region is exactly the call passed in, so surrounding work cannot drift into it as that code
// changes. Stage timings feed the summary only; they print no line of their own.
template <typename Fn>
inline void record_time(Backend b, Pass p, BuildStage stage, Fn &&fn) {
  detail::ScopedBuildTimer scoped(b, p, stage);
  fn();
}

}  // namespace graph_cache_debug
}  // namespace fused_attn
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_FUSED_ATTN_GRAPH_CACHE_DEBUG_H_
